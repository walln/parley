"""A live session: 80 ms of microphone audio in, the model's frame out. It steps the same `Stream`
an offline run does, so a captured session replays exactly, and it decodes every frame in order
through one streaming Mimi decoder, silent frames too, since the codec carries state across them."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from parley.infer.generate import Generator
from parley.infer.playout import Playout, TalkerControl
from parley.rows import SAMPLES_PER_FRAME
from parley.rows.collate import SLICE_SAMPLES
from parley.rows.layout import USER_TOKENS_PER_FRAME


@dataclass(frozen=True)
class Played:
    """The talker's frame, rendered on the model's thread right after its step."""

    audio: np.ndarray
    speaking: bool
    ms: float


@dataclass(frozen=True)
class LiveCodes:
    frame: int
    token: int
    codes: np.ndarray  # int64 [C]
    step_ms: float
    played: Played | None = None


@dataclass(frozen=True)
class LiveFrame:
    frame: int
    token: int
    text: str
    speaking: bool
    audio: np.ndarray  # float32 [samples_per_frame] at the decoder's rate
    step_ms: float
    decode_ms: float


@dataclass(frozen=True)
class Capture:
    """What a session heard and said, enough to replay it offline."""

    seed: int
    user_pcm: np.ndarray  # float32 [frames * 1280] at 16 kHz
    tokens: np.ndarray  # int64 [frames]
    codes: np.ndarray  # int64 [frames, C]
    playing: np.ndarray  # bool [frames]: the talker played the frame before each step

    @property
    def frames(self) -> int:
        return int(self.tokens.shape[0])


def talker_control(gen: Generator) -> TalkerControl:
    """The playout's rules for the generator's layout. Holding the floor needs a stop threshold,
    since a SIL drawn while the talker plays would stop it."""
    held = gen.layout == "talker-floor"
    if held and gen.stop_threshold is None:
        raise ValueError("holding the floor needs a stop threshold")
    return TalkerControl(gen.token_ids.sil, gen.token_ids.eps, held=held)


def user_slices(pcm: np.ndarray, device: torch.device) -> torch.Tensor:
    """One frame of 16 kHz audio as the [1, 2, 640] slices a step hears."""
    if pcm.shape != (SAMPLES_PER_FRAME,):
        raise ValueError(f"a frame is {SAMPLES_PER_FRAME} samples, got {pcm.shape}")
    shaped = pcm.astype(np.float32).reshape(1, USER_TOKENS_PER_FRAME, SLICE_SAMPLES)
    return torch.from_numpy(shaped).to(device)


class LiveSession:
    """Opens a stream of `frames` frames. `advance` runs on one thread for the whole process,
    since compiled CUDA graphs belong to the thread that recorded them, so CSM-1B's playout of the
    words ticks there too. `render` may run beside it on another, decoding the model's own codes
    when there is no playout."""

    def __init__(
        self,
        gen: Generator,
        prefix: np.ndarray,
        *,
        frames: int,
        seed: int,
        voice: object,
    ) -> None:
        gen.set_sampling_seed(seed)
        self.gen, self.seed, self.voice = gen, seed, voice
        self.control = talker_control(gen)
        self.stream = gen.open([prefix], frames)
        self.rendered = 0
        self._pcm: list[np.ndarray] = []
        self._tokens: list[int] = []
        self._codes: list[np.ndarray] = []
        self._playing: list[bool] = []

    @torch.inference_mode()
    def advance(self, pcm: np.ndarray) -> LiveCodes:
        started = time.perf_counter()
        device = self.gen.backbone.device
        played_before = isinstance(self.voice, Playout) and self.control.playing(self.voice)
        self._playing.append(played_before)
        token, codes = self.stream.step(user_slices(pcm, device), playing=torch.tensor([played_before], device=device))
        frame_codes = codes[0].cpu().numpy()
        step_ms = 1000 * (time.perf_counter() - started)
        self._pcm.append(pcm.astype(np.float32))
        self._tokens.append(int(token[0]))
        self._codes.append(frame_codes)
        token_id = self._tokens[-1]
        voice, heard = self.voice, bool(np.any(pcm))
        played = self._play(voice, token_id, heard) if isinstance(voice, Playout) else None
        return LiveCodes(len(self._tokens) - 1, token_id, frame_codes, step_ms, played)

    def _play(self, playout: Playout, token: int, heard: bool) -> Played:
        started = time.perf_counter()
        if self.control.yields(token, saw_playing=self._playing[-1]):
            playout.stop()
        audio = playout.tick(self._text(token), ended=self.control.ends(token), heard=heard)
        return Played(audio, playout.speaking, 1000 * (time.perf_counter() - started))

    def _text(self, token: int) -> str:
        ids = self.gen.token_ids
        written = token not in (ids.sil, ids.eps)
        return self.gen.processor.tokenizer.decode([token]) if written else ""

    def render(self, codes: LiveCodes) -> LiveFrame:
        if codes.frame != self.rendered:
            raise ValueError(f"frames render in order: expected {self.rendered}, got {codes.frame}")
        started = time.perf_counter()
        match codes.played, self.voice:
            case Played() as played, _:
                audio, speaking, decode_ms = played.audio, played.speaking, played.ms
            case None, Playout():
                raise ValueError(f"frame {codes.frame} reached the codec without the talker's audio")
            case None, decoder:
                audio = decoder.decode(codes.codes)
                speaking = codes.token != self.gen.token_ids.sil
                decode_ms = 1000 * (time.perf_counter() - started)
        self.rendered += 1
        return LiveFrame(
            frame=codes.frame,
            token=codes.token,
            text=self._text(codes.token),
            speaking=speaking,
            audio=audio,
            step_ms=codes.step_ms,
            decode_ms=decode_ms,
        )

    @property
    def capture(self) -> Capture:
        codebooks = self.gen.codebooks
        return Capture(
            seed=self.seed,
            user_pcm=np.concatenate(self._pcm) if self._pcm else np.zeros(0, np.float32),
            tokens=np.asarray(self._tokens, dtype=np.int64),
            codes=np.stack(self._codes) if self._codes else np.zeros((0, codebooks), np.int64),
            playing=np.asarray(self._playing, dtype=bool),
        )


@torch.inference_mode()
def warm_up(gen: Generator, prefix: np.ndarray, *, frames: int, steps: int) -> None:
    """Steps a stream of `frames` frames over silence, so the graphed step compiles and records
    its graphs before anyone is waiting on it."""
    stream = gen.open([prefix], frames)
    device = gen.backbone.device
    silence = user_slices(np.zeros(SAMPLES_PER_FRAME, np.float32), device)
    idle = torch.tensor([False], device=device)
    for _ in range(steps):
        stream.step(silence, playing=idle)


@torch.inference_mode()
def replays_exactly(gen: Generator, prefix: np.ndarray, capture: Capture, frames: int) -> bool:
    """Whether the captured audio, stepped again with the session's seed through a stream of the
    same length, gives the same tokens and codes bit for bit."""
    gen.set_sampling_seed(capture.seed)
    stream = gen.open([prefix], frames)
    device = gen.backbone.device
    steps = [
        stream.step(user_slices(pcm, device), playing=torch.tensor([played], device=device))
        for pcm, played in zip(
            capture.user_pcm.reshape(capture.frames, SAMPLES_PER_FRAME),
            capture.playing,
            strict=True,
        )
    ]
    tokens = np.asarray([int(token[0]) for token, _ in steps], dtype=np.int64)
    codes = np.stack([c[0].cpu().numpy() for _, c in steps]) if steps else capture.codes
    return np.array_equal(tokens, capture.tokens) and np.array_equal(codes, capture.codes)


def percentiles(name: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{name}_median": float(np.median(values)),
        f"{name}_p95": float(np.percentile(values, 95)),
        f"{name}_max": float(np.max(values)),
    }


def frame_summary(records: list[dict]) -> dict[str, float]:
    """Each served frame's model step, codec decode and wait before its step, in milliseconds."""
    summary = {"frames": float(len(records))}
    for part in ("step_ms", "decode_ms", "queued_ms"):
        summary |= percentiles(part, [r[part] for r in records])
    return summary
