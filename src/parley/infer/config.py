"""The live server's config: which checkpoint answers and how, and how long the GPU waits for
conversations. The fdg CLI validates it, so this module imports no torch."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from parley.ids import CheckpointPath, Sha256
from parley.infer.gate import GATE_HOLD_FRAMES

#: The live function's container timeout leaves an hour past this for compiling and replaying.
MAX_SECONDS = 3 * 3600
FRAME_S = 0.08
#: Above temperature 0, text is sampled from the whole vocabulary, or the choice among listening,
#: still speaking and any word is sampled and the word is then the most likely one. Greedy
#: decoding alone mutes the model, since listening outweighs any single word.
WordChoice = Literal["sampled", "greedy"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HeldOutRow(Strict):
    rows: str
    meta_sha256: Sha256
    index: int = Field(ge=0)


class LiveConfig(Strict):
    checkpoint: CheckpointPath
    prompt: str = Field(min_length=1)
    #: Text sampling; the graphed frame step decodes audio greedily.
    temperature: float = Field(gt=0.0)
    #: The nth session of an attempt samples with seed + n.
    seed: int = Field(ge=0)
    session_seconds: float = Field(gt=0.0, le=600.0)
    warmup_frames: int = Field(ge=1)
    #: A held-out session whose user channel streams through the tunnel in real time before anyone
    #: else may connect.
    self_test: HeldOutRow
    #: The server shuts down after this long with no conversation, or at max_seconds.
    idle_seconds: int = Field(ge=60)
    max_seconds: int = Field(ge=60, le=MAX_SECONDS)
    #: The held-out row whose words and audio give CSM-1B its voice, to speak Gemma's words. Without
    #: one, the checkpoint's own codes are decoded.
    talker_voice: HeldOutRow | None = None
    word_choice: WordChoice = "sampled"
    #: The talker layout writes STOP once its probability reaches this, instead of drawing it;
    #: holding the floor, SIL while the talker plays, which is then never drawn.
    stop_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    #: A reply Gemma starts after the user spoke over the one still playing replaces it.
    supersede_replies: bool = False
    #: The bridge's noise gate sends the digital silence that ends every training turn once the
    #: microphone has been quiet for more than this many frames.
    gate_hold_frames: int = Field(default=GATE_HOLD_FRAMES, ge=0)

    @property
    def session_frames(self) -> int:
        return math.ceil(self.session_seconds / FRAME_S)


def load_live_config(path: str | Path) -> LiveConfig:
    with open(path) as handle:
        return LiveConfig.model_validate(yaml.safe_load(handle))
