"""The live bridge's noise gate, which every user frame passes before the model hears it: live
from Nick's microphone, offline for recorded voices, and in training for augmented rooms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parley.infer.transport import from_pcm16, to_pcm16
from parley.rows import SAMPLES_PER_FRAME

#: A stopgap: the model takes its turn only on exact digital silence, which ends every training
#: turn, and any room noise, even at -94 dBFS, keeps it waiting. So frames below GATE_RMS go to
#: zero once the hold's quiet frames have passed, and a pause longer than the hold reads as a turn
#: end. A model that answers in room tone would need no gate.
GATE_RMS = 0.004
GATE_HOLD_FRAMES = 12


@dataclass
class NoiseGate:
    hold_frames: int = GATE_HOLD_FRAMES
    quiet: int = 0

    def __call__(self, pcm: bytes) -> bytes:
        audio = from_pcm16(pcm)
        self.quiet = 0 if np.sqrt(np.mean(audio**2)) >= GATE_RMS else self.quiet + 1
        return pcm if self.quiet <= self.hold_frames else bytes(len(pcm))


def gated(audio: np.ndarray, hold_frames: int = GATE_HOLD_FRAMES) -> np.ndarray:
    """The clip frame by frame through a fresh noise gate, padded to whole frames."""
    frames = -(-audio.shape[0] // SAMPLES_PER_FRAME)
    padded = np.zeros(frames * SAMPLES_PER_FRAME, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    gate = NoiseGate(hold_frames)
    chunks = padded.reshape(frames, SAMPLES_PER_FRAME)
    return np.concatenate([from_pcm16(gate(to_pcm16(chunk))) for chunk in chunks])
