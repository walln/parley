"""The live wire protocol. A connection carries the attempt's token and opens with a JSON
{"type": "start"} for a session or {"type": "shutdown"}. In a session the client sends each 80 ms
frame of 16 kHz int16 audio as a binary message the moment it is captured, then a JSON
{"type": "end"}. The server answers every frame with its 24 kHz int16 audio as a binary message
followed by a JSON frame record, and after the last one a JSON {"type": "ended"}. Neither side
waits for the other between frames. This module imports no torch, so the bridge can use it."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from parley.infer.live import LiveSession

#: Where a main attempt tells the bridge how to reach it, in its attempt folder on the volume.
CONNECTION_FILE = "connection.json"
#: The label a session opens with: the live server's own test, a person at the page, a wav file.
SELF_TEST, CONVERSATION, SCRIPTED = "self-test", "conversation", "scripted"
IN_RATE, IN_FRAME = 16000, 1280
OUT_RATE, OUT_FRAME = 24000, 1920
IN_BYTES, OUT_BYTES = 2 * IN_FRAME, 2 * OUT_FRAME
#: Two seconds of frames in flight before a side gives up on the other keeping pace.
MAX_PENDING = 25
REPLY_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Connection:
    url: str
    token: str
    checkpoint: str
    frames: int
    gate_hold_frames: int

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: bytes) -> Connection:
        return cls(**json.loads(raw))


class Socket(Protocol):
    async def send(self, message: bytes | str) -> None: ...

    async def recv(self) -> bytes | str: ...

    def __aiter__(self) -> AsyncIterator[bytes | str]: ...


def to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()


def from_pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768


def control(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields})


async def gather_or_cancel(*coroutines) -> None:
    tasks = [asyncio.create_task(c) for c in coroutines]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True)
class Workers:
    model: Executor
    codec: Executor

    @classmethod
    @contextmanager
    def threads(cls) -> Iterator[Workers]:
        with ThreadPoolExecutor(1, "model") as model, ThreadPoolExecutor(1, "codec") as codec:
            yield cls(model, codec)


def user_frame(message: bytes | str) -> np.ndarray | None:
    """A binary message's audio, or None for the client's "end"."""
    if isinstance(message, str):
        if json.loads(message).get("type") != "end":
            raise ValueError(f"a client sends frames and then end, not {message}")
        return None
    if len(message) != IN_BYTES:
        raise ValueError(f"a user frame is {IN_BYTES} bytes, got {len(message)}")
    return from_pcm16(message)


@dataclass
class Serving:
    """One session's three stages, joined by queues: receiving frames, the model's step on its
    thread, and the codec's decode on its thread before each reply goes out."""

    socket: Socket
    session: LiveSession
    workers: Workers
    frames: int
    inputs: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(MAX_PENDING))
    made: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(MAX_PENDING))
    records: list[dict] = field(default_factory=list)

    async def receive(self) -> None:
        received = 0
        async for message in self.socket:
            pcm = user_frame(message)
            if pcm is None or received == self.frames:
                break
            if self.inputs.full():
                raise RuntimeError("the model fell two seconds behind the microphone")
            self.inputs.put_nowait((pcm, time.perf_counter()))
            received += 1
        await self.inputs.put(None)

    async def infer(self) -> None:
        loop = asyncio.get_running_loop()
        while (item := await self.inputs.get()) is not None:
            pcm, arrived = item
            queued_ms = 1000 * (time.perf_counter() - arrived)
            codes = await loop.run_in_executor(self.workers.model, self.session.advance, pcm)
            await self.made.put((codes, queued_ms))
        await self.made.put(None)

    async def emit(self) -> None:
        loop = asyncio.get_running_loop()
        while (item := await self.made.get()) is not None:
            codes, queued_ms = item
            frame = await loop.run_in_executor(self.workers.codec, self.session.render, codes)
            record = {
                "frame": frame.frame,
                "text": frame.text,
                "speaking": frame.speaking,
                "step_ms": round(frame.step_ms, 2),
                "decode_ms": round(frame.decode_ms, 2),
                "queued_ms": round(queued_ms, 2),
            }
            self.records.append(record)
            await self.socket.send(to_pcm16(frame.audio))
            await self.socket.send(control("frame", **record))
        await self.socket.send(control("ended"))


async def serve_session(socket: Socket, session: LiveSession, workers: Workers, *, frames: int) -> list[dict]:
    """Every frame the client sends, up to `frames`, through the model and then the codec in order,
    each reply sent once decoded. Returns the records of the frames sent back: all of them after
    "ended", or those before the client hung up."""
    serving = Serving(socket, session, workers, frames)
    try:
        await gather_or_cancel(serving.receive(), serving.infer(), serving.emit())
    except ConnectionClosed as closed:
        print(f"[live] the client hung up after {len(serving.records)} frames: {closed}", flush=True)
    return serving.records


@dataclass
class Pump:
    """Client side: when each frame was sent, to time its reply's round trip."""

    sent_at: list[float] = field(default_factory=list)
    replies: int = 0

    async def produce(self, socket: Socket, frames: AsyncIterator[bytes]) -> None:
        async for pcm in frames:
            if len(pcm) != IN_BYTES:
                raise ValueError(f"a user frame is {IN_BYTES} bytes, got {len(pcm)}")
            if len(self.sent_at) - self.replies >= MAX_PENDING:
                raise RuntimeError("the server fell two seconds behind the microphone")
            self.sent_at.append(time.perf_counter())
            await socket.send(pcm)
        await socket.send(control("end"))

    async def consume(self, socket: Socket, on_reply: Callable[[bytes, dict], None]) -> None:
        audio: bytes | None = None
        while True:
            async with asyncio.timeout(REPLY_TIMEOUT_S):
                message = await socket.recv()
            if isinstance(message, bytes):
                audio = message
                continue
            record = json.loads(message)
            if record["type"] == "ended":
                return
            if record["type"] != "frame" or audio is None or len(audio) != OUT_BYTES:
                raise ValueError(f"expected a frame's audio and then its record, got {record}")
            record["round_trip_ms"] = round(1000 * (time.perf_counter() - self.sent_at[record["frame"]]), 2)
            self.replies += 1
            on_reply(audio, record)
            audio = None


async def pump_frames(socket: Socket, frames: AsyncIterator[bytes], on_reply: Callable[[bytes, dict], None]) -> None:
    """Sends each frame as `frames` yields it, then "end", while handing each reply's audio and
    record to `on_reply` until the server has ended."""
    pump = Pump()
    await gather_or_cancel(pump.produce(socket, frames), pump.consume(socket, on_reply))
