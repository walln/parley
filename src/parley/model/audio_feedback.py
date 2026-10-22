"""Backbone audio feedback (Moshi-style): the previous frame's Mimi codes, embedded and summed
into the assistant-slot embedding, keep the intelligibility codebooks coherent under autoregression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AudioFeedbackConfig:
    num_codebooks: int
    codebook_size: int
    hidden_size: int
    init_std: float = 0.01


class AudioFeedbackEmbedding(nn.Module):
    """Sum of per-codebook embeddings of a frame's Mimi codes in backbone hidden space,
    RMS-normalized and scaled by a learned per-dimension gain: the unnormalized
    0.01-init sum stays ~0.1% of the token-embedding energy."""

    def __init__(self, config: AudioFeedbackConfig):
        super().__init__()
        self.config = config
        self.embeds = nn.ModuleList(
            [nn.Embedding(config.codebook_size, config.hidden_size) for _ in range(config.num_codebooks)]
        )
        for emb in self.embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)
        self.gain = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (frames, num_codebooks) long -> (frames, hidden_size)."""
        if codes.ndim != 2 or codes.shape[1] != self.config.num_codebooks:
            raise ValueError(f"expected codes (frames, {self.config.num_codebooks}), got {tuple(codes.shape)}")
        codes = codes.long().clamp(min=0, max=self.config.codebook_size - 1)
        total = self.embeds[0](codes[:, 0])
        for k in range(1, self.config.num_codebooks):
            total = total + self.embeds[k](codes[:, k])
        rms = total.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return total * rms.to(total.dtype) * self.gain


def save_audio_feedback(
    path: str | Path, module: AudioFeedbackEmbedding, *, meta: dict[str, Any] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": module.state_dict(), "config": module.config.__dict__, "meta": meta or {}},
        path,
    )


def load_audio_feedback(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[AudioFeedbackEmbedding, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = dict(payload["config"])
    normalize = config.pop("normalize", True)
    book_normalize = config.pop("book_normalize", False)
    if normalize is not True or book_normalize is not False:
        raise ValueError(
            f"{path} holds a feedback table with normalize={normalize} and "
            f"book_normalize={book_normalize}; only the shared RMS normalization is supported"
        )
    module = AudioFeedbackEmbedding(AudioFeedbackConfig(**config))
    module.load_state_dict(payload["state_dict"])
    module.to(map_location)
    return module, payload.get("meta", {})
