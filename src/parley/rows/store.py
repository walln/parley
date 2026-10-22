"""Memory-mapped reader over a finalized shard. Row reads are slices; nothing is copied until
the collate asks for it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parley.rows.schema import (
    COLUMNS,
    FORMAT,
    META_FILE,
    SAMPLES_PER_FRAME,
    ShardMeta,
    column_path,
    offsets,
)


@dataclass(frozen=True)
class Row:
    index: int
    prompt: str
    source: str
    user_pcm: np.ndarray  # int16 [frames * 1280]
    codes: np.ndarray  # uint16 [frames, codebooks]
    frame_state: np.ndarray  # uint8 [frames]; SPEAKING / LISTENING
    text_counts: np.ndarray  # uint8 [frames]; text tokens starting in each frame
    text_tokens: np.ndarray  # int32 [sum(text_counts)]
    text_weights: np.ndarray  # float32 [frames]
    role: str = "assistant"  # "echo" rows transcribe the user channel: text loss only
    question_pcm: np.ndarray | None = None  # int16 [question_frames * 1280] for at / aa rows

    @property
    def frames(self) -> int:
        return int(self.frame_state.shape[0])


class RowStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} missing: not a finalized fdg-rows shard")
        self.meta = ShardMeta.model_validate(json.loads(meta_path.read_text()))
        if self.meta.format != FORMAT:
            raise ValueError(f"{self.directory}: format {self.meta.format!r}, expected {FORMAT!r}")
        self._frame_off = offsets([r.frames for r in self.meta.rows])
        self._pcm_off = offsets([r.question_frames + r.frames for r in self.meta.rows])
        self._text_off = offsets([r.text_tokens for r in self.meta.rows])
        self._maps: dict[str, np.memmap] | None = None

    def __len__(self) -> int:
        return len(self.meta.rows)

    def frames(self, index: int) -> int:
        return self.meta.rows[index].frames

    @property
    def frame_counts(self) -> np.ndarray:
        return np.diff(self._frame_off)

    def _mmaps(self) -> dict[str, np.memmap]:
        # Opened lazily so a store can cross a fork/pickle boundary before first read.
        if self._maps is None:
            totals = {
                "sample": int(self._pcm_off[-1]) * SAMPLES_PER_FRAME,  # question audio included
                "frame": int(self._frame_off[-1]),
                "text": int(self._text_off[-1]),
            }
            maps: dict[str, np.memmap] = {}
            for column, (dtype, tail, axis) in COLUMNS.items():
                count = totals[axis]
                path = column_path(self.directory, column)
                expected = count * int(np.prod(tail, dtype=np.int64)) * dtype.itemsize
                actual = path.stat().st_size
                if actual != expected:
                    raise ValueError(f"{path}: {actual} bytes, expected {expected}")
                maps[column] = np.memmap(path, dtype=dtype, mode="r", shape=(count, *tail))
            self._maps = maps
        return self._maps

    def __getitem__(self, index: int) -> Row:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        maps = self._mmaps()
        f0, f1 = int(self._frame_off[index]), int(self._frame_off[index + 1])
        p0, p1 = int(self._pcm_off[index]), int(self._pcm_off[index + 1])
        t0, t1 = int(self._text_off[index]), int(self._text_off[index + 1])
        meta = self.meta.rows[index]
        q = meta.question_frames * SAMPLES_PER_FRAME
        pcm = maps["user_pcm"][p0 * SAMPLES_PER_FRAME : p1 * SAMPLES_PER_FRAME]
        return Row(
            index=index,
            prompt=meta.prompt,
            source=meta.source,
            user_pcm=pcm[q:],
            codes=maps["codes"][f0:f1],
            frame_state=maps["frame_state"][f0:f1],
            text_counts=maps["text_counts"][f0:f1],
            text_tokens=maps["text_tokens"][t0:t1],
            text_weights=maps["text_weights"][f0:f1],
            role=meta.role,
            question_pcm=pcm[:q] if q else None,
        )

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_maps"] = None
        return state
