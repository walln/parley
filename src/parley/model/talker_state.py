"""The playing-or-idle input of the talker layout: one learned vector added to each frame's assistant
position when the talker played audio in the frame before. It starts at zero, so a checkpoint saved
without it behaves as before."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

FILE = "talker_state.pt"


class TalkerState(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.playing = nn.Parameter(torch.zeros(hidden))

    def forward(self, playing: torch.Tensor) -> torch.Tensor:
        """[..., H] from a [...] bool tensor."""
        return playing.unsqueeze(-1).to(self.playing.dtype) * self.playing


def save_talker_state(directory: Path, module: TalkerState) -> None:
    torch.save(module.state_dict(), directory / FILE)


def load_talker_state(directory: Path | None, hidden: int, device) -> TalkerState:
    module = TalkerState(hidden)
    if directory is not None and (directory / FILE).is_file():
        module.load_state_dict(torch.load(directory / FILE, map_location="cpu", weights_only=True))
    return module.to(device, dtype=torch.float32)
