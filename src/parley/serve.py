"""Run Parley as a local, full-duplex live server: one process, one GPU, one browser page."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
from websockets.asyncio.server import serve as ws_serve

from parley.infer.gate import NoiseGate
from parley.infer.generate import load_generator
from parley.infer.live import LiveSession
from parley.infer.playout import Playout, PrefixDecoder
from parley.infer.talker import POSITIONS, SESAME, CsmSpeaker, Talker
from parley.infer.transport import (
    Workers,
    control,
    from_pcm16,
    serve_session,
    to_pcm16,
)
from parley.model.csm_depth import CSM_REPO, CSM_REVISION
from parley.model.prefix import PrefixBuilder as GemmaPrefixBuilder

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/capture-worklet.js": ("capture-worklet.js", "text/javascript; charset=utf-8"),
    "/playback-worklet.js": ("playback-worklet.js", "text/javascript; charset=utf-8"),
}
ASSETS = Path(__file__).resolve().parent / "infer" / "web_assets"
VOICE_FRAMES = 1500
WARM_UP_WORDS = "Hello there, how are you today?"


class GatedSession:
    def __init__(self, inner: LiveSession, gate: NoiseGate):
        self.inner = inner
        self.gate = gate

    def advance(self, pcm):
        raw = to_pcm16(pcm) if not isinstance(pcm, (bytes, bytearray)) else bytes(pcm)
        gated = self.gate(raw)
        gated_pcm = from_pcm16(gated)
        return self.inner.advance(gated_pcm)

    def render(self, codes):
        return self.inner.render(codes)

    @property
    def capture(self):
        return self.inner.capture


def load_voice(voice_wav: Path, voice_text: str, device: str):
    processor = AutoProcessor.from_pretrained(CSM_REPO, revision=CSM_REVISION)
    model = CsmForConditionalGeneration.from_pretrained(CSM_REPO, revision=CSM_REVISION, device_map={"": device})
    model.to(torch.bfloat16)
    model.codec_model.float()
    talker = Talker(model, books=32, positions=POSITIONS, sampling=SESAME)
    audio, _ = sf.read(str(voice_wav), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    segment = talker.segment(voice_text, audio)
    speaker = CsmSpeaker(processor, talker, segment)
    talker.warm_prefills()
    PrefixDecoder(model.codec_model).warm(32)
    return processor, talker, speaker, model


def make_playout(talker, speaker, model, seed: int, supersede: bool = False):
    decoder = PrefixDecoder(model.codec_model)
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    return Playout(talker, speaker, decoder, generator, supersede=supersede)


def run_server(
    checkpoint: Path,
    prompt: str,
    *,
    voice_wav: Path,
    voice_text: str,
    host: str = "127.0.0.1",
    port: int = 8766,
    session_seconds: float = 30.0,
    temperature: float = 0.8,
    stop_threshold: float = 0.5,
    gate_hold_frames: int = 12,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_generator(
        checkpoint,
        device=device,
        temperature=temperature,
        merge_adapters=True,
    )
    gen.stop_threshold = stop_threshold
    prefix = GemmaPrefixBuilder(gen.processor)(prompt)

    _processor, talker, speaker, csm_model = load_voice(voice_wav, voice_text, device)

    def make_voice(frames: int, seed: int):
        playout = make_playout(talker, speaker, csm_model, seed, supersede=False)
        return playout

    # Warm up the playout.
    warming = make_voice(0, 0)
    warming.tick(WARM_UP_WORDS, ended=True)
    while warming.speaking:
        warming.tick("", ended=True)

    import math

    session_frames = math.ceil(session_seconds / 0.08)

    def open_session(frames: int, seed: int):
        playout = make_voice(frames, seed)
        return GatedSession(
            LiveSession(gen, prefix, frames=frames, seed=seed, voice=playout),
            NoiseGate(gate_hold_frames),
        )

    def process_request(connection, request):
        if request.path == "/ws":
            return None
        if request.path in STATIC:
            name, mime = STATIC[request.path]
            body = (ASSETS / name).read_bytes()
            from websockets.datastructures import Headers
            from websockets.http11 import Response

            return Response(
                200,
                "OK",
                Headers([("Content-Type", mime), ("Cache-Control", "no-store")]),
                body,
            )
        return None

    async def main():
        with Workers.threads() as workers:
            async def handler(websocket):
                while True:
                    hello = json.loads(await websocket.recv())
                    if hello.get("type") == "shutdown":
                        await websocket.send(control("shutting-down"))
                        return
                    if hello.get("type") != "start":
                        await websocket.send(control("error", message="expected start"))
                        continue
                    frames = session_frames
                    seed = 0
                    loop = asyncio.get_event_loop()
                    session = await loop.run_in_executor(workers.model, open_session, frames, seed)
                    await websocket.send(control("ready", frames=frames))
                    records = await serve_session(websocket, session, workers, frames=frames)
                    await websocket.send(control("saved", frames=len(records)))

            async with ws_serve(handler, host, port, process_request=process_request):
                url = f"http://{host}:{port}/"
                print(f"[parley] open {url}", flush=True)
                try:
                    await asyncio.Event().wait()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass

    asyncio.run(main())


workers = None
