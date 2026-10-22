"""The talker, live: CSM-1B speaks Gemma's reply chunk by chunk into a playout queue. It runs on the
frame clock, not the wall clock, so a session replays exactly. Each frame takes Gemma's new text,
renders up to RENDER_AHEAD of CSM-1B's frames, one once AHEAD_FRAMES are queued, and plays one
80 ms frame, or silence when none is ready. Rendering runs ahead, so a chunk waits for its sentence
to end until the queue is nearly empty (parley.infer.chunks cuts the text). Each chunk is spoken after
the voice clip and the last WINDOW_SECONDS of what was said. Chunks are joined without their own
leading and trailing silence, with a pause set by the punctuation between two chunks of a reply."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Protocol

import numpy as np
import torch
from transformers import MimiModel

from parley.codec.mimi import MIMI_SAMPLES_PER_FRAME, MIMI_SR
from parley.infer.chunks import Chunker, Piece
from parley.infer.talker import Segment, Talker

#: CSM-1B's 2,048 positions hold about 40 seconds of earlier audio beside the voice clip.
WINDOW_SECONDS = 40.0
SILENT_RMS = 10 ** (-45 / 20)
#: CSM-1B can fall silent mid-chunk for seconds; a chunk that has spoken ends after this many
#: silent frames in a row (about 1 s).
STALL_FRAMES = 12
RENDER_AHEAD = 2
AHEAD_FRAMES = 12
#: A chunk started with this few frames left to play renders its first frames in time.
RUN_DRY_FRAMES = 2
MAX_CHUNK_FRAMES = 375
#: The chunk so far is decoded padded to a multiple of this many frames, each length warmed.
DECODE_BUCKET = 8
#: The pause between two chunks of a reply, after a clause or a sentence; a chunk cut mid-clause
#: gets none.
CLAUSE_PAUSE_FRAMES = 2
SENTENCE_PAUSE_FRAMES = 4


class Speaker(Protocol):
    def embeds(self, text: str, context: Sequence[Segment]) -> torch.Tensor: ...

    def segment(self, text: str, wave: np.ndarray) -> Segment: ...


class StreamDecoder(Protocol):
    def decode(self, new: Sequence[torch.Tensor]) -> list[np.ndarray]: ...

    def reset(self) -> None: ...


@dataclass
class PrefixDecoder:
    """The new frames' 80 ms each, from decoding the chunk so far with Mimi, padded with silent
    codes to a multiple of DECODE_BUCKET. Mimi's decoder looks no frame ahead, so this matches
    decoding the whole chunk at once."""

    codec: MimiModel
    codes: list[torch.Tensor] = field(default_factory=list)

    @torch.inference_mode()
    def decode(self, new: Sequence[torch.Tensor]) -> list[np.ndarray]:
        if not new:
            return []
        self.codes += new
        audio = self._decoded(torch.stack(self.codes, dim=-1))
        frames = audio[(len(self.codes) - len(new)) * MIMI_SAMPLES_PER_FRAME :]
        return list(frames[: len(new) * MIMI_SAMPLES_PER_FRAME].reshape(len(new), -1))

    def reset(self) -> None:
        self.codes = []

    @torch.inference_mode()
    def warm(self, books: int) -> None:
        """Every padded length a chunk can reach, once."""
        device = next(self.codec.parameters()).device
        for frames in range(DECODE_BUCKET, MAX_CHUNK_FRAMES + DECODE_BUCKET, DECODE_BUCKET):
            self._decoded(torch.zeros((books, frames), dtype=torch.long, device=device))

    def _decoded(self, codes: torch.Tensor) -> np.ndarray:
        frames = codes.shape[1]
        padded = torch.nn.functional.pad(codes, (0, -frames % DECODE_BUCKET))
        audio = self.codec.decode(padded[None], return_dict=True).audio_values[0, 0]
        return audio[: frames * MIMI_SAMPLES_PER_FRAME].float().cpu().numpy()


def silent(frame: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(frame**2))) < SILENT_RMS


@dataclass
class Spoken:
    """What the talker has said so far, as context segments, with each one's length in seconds."""

    segments: list[Segment] = field(default_factory=list)

    def add(self, segment: Segment) -> None:
        self.segments.append(segment)

    def window(self) -> list[Segment]:
        kept, total = [], 0.0
        for segment in reversed(self.segments):
            total += segment.samples / MIMI_SR
            if total > WINDOW_SECONDS:
                break
            kept.append(segment)
        return kept[::-1]


@dataclass
class Chunk:
    text: str
    frames: Iterator[torch.Tensor]
    kept: list[np.ndarray] = field(default_factory=list)
    held: list[np.ndarray] = field(default_factory=list)
    silent_run: int = 0


@dataclass(frozen=True)
class TalkerControl:
    """How Gemma's control tokens drive the playout. In the talker layout STOP (carried by EPS)
    empties it and SIL ends a reply's writing. Holding the floor, a SIL chosen while the talker
    played empties it, and either control token ends the writing."""

    sil: int
    eps: int
    held: bool

    def yields(self, token: int, *, saw_playing: bool) -> bool:
        return token == self.sil and saw_playing if self.held else token == self.eps

    def ends(self, token: int) -> bool:
        return token == self.sil or (self.held and token == self.eps)

    def playing(self, playout: Playout) -> bool:
        return playout.speaking if self.held else playout.playing


@dataclass
class Playout:
    talker: Talker
    speaker: Speaker
    decoder: StreamDecoder
    generator: torch.Generator
    chunker: Chunker = field(default_factory=Chunker)
    waiting: deque[Piece] = field(default_factory=deque)
    ready: deque[np.ndarray] = field(default_factory=deque)
    history: Spoken = field(default_factory=Spoken)
    chunk: Chunk | None = None
    #: A reply Gemma starts after the user spoke over the one still playing replaces it.
    supersede: bool = False
    superseded: int = 0
    _writing: bool = False
    _heard: bool = False
    _last: str = ""

    @property
    def speaking(self) -> bool:
        return bool(self.playing or self.chunker.pending.strip())

    @property
    def playing(self) -> bool:
        """Whether there is audio rendered or to render, as a training row's speaking frames."""
        return bool(self.ready or self.waiting or self.chunk)

    def stop(self) -> None:
        """Yield to the user: the audio queued, the chunk in progress and every word not yet
        spoken are dropped."""
        self.ready.clear()
        self.waiting.clear()
        self.chunk = None
        self.chunker = Chunker()

    def tick(self, text: str, *, ended: bool, heard: bool = False) -> np.ndarray:
        """`heard`: whether the user's audio was heard in the frame that chose `text`."""
        if self.supersede:
            self._keep_one_reply(word=bool(text), heard=heard)
        hungry = self.chunk is None and not self.waiting and len(self.ready) <= RUN_DRY_FRAMES
        self.waiting.extend(self.chunker.feed(text, hungry=hungry))
        if ended:
            self.waiting.extend(self.chunker.end())
        self._render(RENDER_AHEAD if len(self.ready) < AHEAD_FRAMES else 1)
        if self.ready:
            return self.ready.popleft()
        if self.chunk is not None and self.chunk.held:
            frame = self.chunk.held.pop(0)
            self.chunk.kept.append(frame)
            return frame
        return np.zeros(MIMI_SAMPLES_PER_FRAME, dtype=np.float32)

    def _keep_one_reply(self, *, word: bool, heard: bool) -> None:
        if not self._writing:
            self._heard = self._heard or heard
        if word and not self._writing and self._heard and self.speaking:
            self.stop()
            self.superseded += 1
        self._writing = word
        self._heard = self._heard and not word

    def _render(self, budget: int) -> None:
        """Up to `budget` of CSM-1B's frames, decoded together. A chunk that ends is kept in the
        context, and the next one waiting starts."""
        while budget > 0 and (chunk := self._current()) is not None:
            codes = [frame[0] for frame in islice(chunk.frames, budget)]
            for frame in self.decoder.decode(codes):
                self._place(chunk, frame)
            if len(codes) < budget or (chunk.kept and chunk.silent_run >= STALL_FRAMES):
                self._finish(chunk)
            budget -= max(len(codes), 1)

    def _current(self) -> Chunk | None:
        if self.chunk is None and self.waiting:
            piece = self.waiting.popleft()
            if not piece.opens:
                self.ready.extend(_pause(self._last))
            embeds = self.speaker.embeds(piece.text, self.history.window())
            frames = self.talker.frames(embeds, MAX_CHUNK_FRAMES, self.generator)
            self.chunk = Chunk(piece.text, frames)
            self.decoder.reset()
        return self.chunk

    def _place(self, chunk: Chunk, frame: np.ndarray) -> None:
        chunk.silent_run = chunk.silent_run + 1 if silent(frame) else 0
        if not silent(frame):
            self._keep(chunk, [*chunk.held, frame])
            chunk.held = []
        elif chunk.kept:
            chunk.held.append(frame)

    def _finish(self, chunk: Chunk) -> None:
        """A chunk that spoke joins what was said; one CSM-1B rendered silent was never heard."""
        self._last = chunk.text
        if chunk.kept:
            self.history.add(self.speaker.segment(chunk.text, np.concatenate(chunk.kept)))
        self.chunk = None

    def _keep(self, chunk: Chunk, frames: list[np.ndarray]) -> None:
        chunk.kept += frames
        self.ready.extend(frames)


def _pause(after: str) -> list[np.ndarray]:
    pauses = dict.fromkeys(".!?", SENTENCE_PAUSE_FRAMES) | dict.fromkeys(",;:", CLAUSE_PAUSE_FRAMES)
    frames = pauses.get(after.rstrip()[-1:], 0)
    return [np.zeros(MIMI_SAMPLES_PER_FRAME, dtype=np.float32) for _ in range(frames)]
