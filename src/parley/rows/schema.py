"""The fdg-rows-v1 shard: flat typed arrays for the whole shard, a row index of offsets, and one
meta.json. Rows store CONTENT (per-frame speak/listen state, a variable number of text tokens per
frame, codes, user PCM); the layout (parley.rows.layout) renders control tokens. Nothing per frame
is a Python object."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

FORMAT = "fdg-rows-v1"
SAMPLE_RATE = 16_000
FRAME_MS = 80
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 1280
META_FILE = "meta.json"

SPEAKING = 0
LISTENING = 1

#: Column -> (dtype, trailing shape, index axis). Axis "sample" = user PCM samples, "frame" =
#: Mimi frames, "text" = text tokens (ragged per frame via text_counts).
COLUMNS: dict[str, tuple[np.dtype, tuple[int, ...], str]] = {
    "user_pcm": (np.dtype(np.int16), (), "sample"),
    "codes": (np.dtype(np.uint16), (8,), "frame"),
    "frame_state": (np.dtype(np.uint8), (), "frame"),
    "text_counts": (np.dtype(np.uint8), (), "frame"),
    "text_weights": (np.dtype(np.float32), (), "frame"),
    "text_tokens": (np.dtype(np.int32), (), "text"),
}


class TokenIds(BaseModel):
    """Control-token ids the rows were written against; the model must agree at load."""

    model_config = ConfigDict(extra="forbid")

    channel_live: int
    user_audio: int
    eps: int
    sil: int
    assistant_slot: int


class RowMeta(BaseModel):
    """Per-row scalars; array offsets are derived cumulatively from `frames` / `text_tokens`."""

    model_config = ConfigDict(extra="forbid")

    frames: int = Field(gt=0)
    text_tokens: int = Field(ge=0)
    prompt: str
    source: str = ""
    role: str = "assistant"  # echo / native; tt / at / aa (native message tasks)
    #: Frames of question audio stored in the user_pcm column BEFORE the row's own
    #: frames; the native user turn's audio for `at` / `aa` rows. 0 for every other role.
    question_frames: int = Field(default=0, ge=0)


class ShardMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = FORMAT
    created: str
    codebooks: int = 8
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = FRAME_MS
    token_ids: TokenIds
    tokenizer: str
    has_frame_events: Literal[False] = False
    rows: list[RowMeta]
    sha256: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @property
    def frames(self) -> int:
        return sum(r.frames for r in self.rows)


def column_path(directory: Path, column: str) -> Path:
    return directory / f"{column}.bin"


def offsets(counts: list[int]) -> np.ndarray:
    """Exclusive prefix sum: item i spans [off[i], off[i + 1])."""
    out = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=out[1:])
    return out
