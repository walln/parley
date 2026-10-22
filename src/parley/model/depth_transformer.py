"""Depth transformer (RQ-style, Moshi-inspired): autoregressively predicts 8 residual Mimi
codebooks per 80 ms frame from the backbone hidden state."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

#: Rows of the frame-position embedding; later frames reuse the last row.
MAX_FRAME_POSITIONS = 4096


def _argmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def batched_codebook_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-frame cross entropy for every codebook in one call: (B, C, V) logits -> (C, B).

    Cross entropy is independent per row, so folding the codebook axis into the batch
    axis only changes kernel count: values are bit-identical to a per-codebook loop.
    """
    num_codebooks = logits.shape[1]
    per_frame = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return per_frame.view(-1, num_codebooks).t().contiguous()


class DepthDecodeCache:
    """Per-layer K/V cache for one frame's codebook decode; created and discarded per frame
    (the depth stack attends only within the frame's codebook sequence)."""

    __slots__ = ("keys", "values")

    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers

    def append(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one position's K/V and return the full cached sequence."""
        cached_k, cached_v = self.keys[layer_index], self.values[layer_index]
        keys = key if cached_k is None else torch.cat([cached_k, key], dim=2)
        values = value if cached_v is None else torch.cat([cached_v, value], dim=2)
        self.keys[layer_index] = keys
        self.values[layer_index] = values
        return keys, values

    @property
    def length(self) -> int:
        first = self.keys[0]
        return 0 if first is None else int(first.shape[2])


class DepthTransformer(nn.Module):
    """Small causal transformer over the codebook dimension, conditioned on the backbone hidden."""

    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        depth_layers: int = 4,
        depth_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        self.in_proj = nn.Linear(hidden_size, depth_hidden)
        self.frame_pos_emb = nn.Embedding(MAX_FRAME_POSITIONS, depth_hidden)

        self.code_embeds = nn.Embedding(codebook_size + 1, depth_hidden)  # +1 for <start>
        self.start_token = codebook_size

        layer = nn.TransformerEncoderLayer(
            d_model=depth_hidden,
            nhead=8,
            dim_feedforward=depth_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth_layers)

        self.heads = nn.ModuleList([nn.Linear(depth_hidden, codebook_size) for _ in range(num_codebooks)])

        self.register_buffer("causal_mask", self._build_causal_mask(num_codebooks + 1))

    @property
    def update_sentinel(self) -> nn.Parameter:
        return self.transformer.layers[0].norm1.weight

    def _apply_heads(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply every codebook head in one batched GEMM: (B, C, D) -> (B, C, V).

        Bit-exact against `self.heads[i](hidden[:, i])` in a loop. Heads stay a
        `ModuleList` of `nn.Linear`, so checkpoints, init, and the one-head-at-a-time
        decode path are unchanged.
        """
        weight = torch.stack([head.weight for head in self.heads])  # (C, V, D)
        bias = torch.stack([head.bias for head in self.heads])  # (C, V)
        stacked = torch.baddbmm(
            bias.unsqueeze(1),  # (C, 1, V) — broadcast over frames
            hidden.transpose(0, 1),  # (C, B, D)
            weight.transpose(1, 2),  # (C, D, V)
        )  # (C, B, V)
        return stacked.transpose(0, 1)

    def _cached_layer_step(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        cache: DepthDecodeCache,
        layer_index: int,
    ) -> torch.Tensor:
        """Run one `nn.TransformerEncoderLayer` over a single new position, using cached K/V.

        Mirrors the module's own `norm_first=True` computation exactly, reading the same
        parameters — no retraining. No attention mask needed: the cache holds only
        positions <= the new one, so causality is structural.
        """
        attn = layer.self_attn
        heads = attn.num_heads
        head_dim = attn.embed_dim // heads
        batch = x.shape[0]

        normed = layer.norm1(x)
        qkv = F.linear(normed, attn.in_proj_weight, attn.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, -1, heads, head_dim).transpose(1, 2)

        keys, values = cache.append(layer_index, split(key), split(value))
        attended = F.scaled_dot_product_attention(split(query), keys, values)
        attended = attended.transpose(1, 2).reshape(batch, -1, attn.embed_dim)
        x = x + layer.dropout1(attn.out_proj(attended))

        normed = layer.norm2(x)
        projected = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed))))
        return x + layer.dropout2(projected)

    def decode_codes(
        self,
        conditioning: torch.Tensor,
        *,
        choose: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode all codebooks for one frame from the already-projected conditioning; `choose`
        maps each book's logits to codes (argmax when None)."""
        if choose is None:
            choose = _argmax
        batch = conditioning.shape[0]
        device = conditioning.device
        bias = conditioning.unsqueeze(1)
        start = torch.full((batch, 1), self.start_token, device=device, dtype=torch.long)
        step_input = self.code_embeds(start) + bias

        cache = DepthDecodeCache(len(self.transformer.layers))
        generated = []
        for level in range(self.num_codebooks):
            hidden = step_input
            for index, layer in enumerate(self.transformer.layers):
                hidden = self._cached_layer_step(layer, hidden, cache, index)
            if self.transformer.norm is not None:
                hidden = self.transformer.norm(hidden)
            pred = choose(self.heads[level](hidden[:, -1]))
            generated.append(pred)
            if level + 1 < self.num_codebooks:
                step_input = self.code_embeds(pred[:, None]) + bias
        return torch.stack(generated, dim=1)

    def _build_causal_mask(self, length: int) -> torch.Tensor:
        # PyTorch boolean attention masks use True for BLOCKED positions.
        return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)

    def project_conditioning(self, backbone_hidden, frame_positions=None):
        if backbone_hidden.dim() == 3:
            backbone_hidden = backbone_hidden[:, -1]
        h = self.in_proj(backbone_hidden.to(self.in_proj.weight.dtype))
        if frame_positions is not None:
            positions = frame_positions.to(device=h.device, dtype=torch.long)
            positions = positions.clamp(min=0, max=MAX_FRAME_POSITIONS - 1)
            h = h + self.frame_pos_emb(positions)
        return h

    def forward(
        self,
        backbone_hidden: torch.Tensor,  # (B, hidden_size) or (B, T, hidden); last frame used
        target_codes: torch.Tensor | None = None,  # (B, num_codebooks) for teacher forcing
        frame_positions: torch.Tensor | None = None,  # (B,) absolute audio-frame positions
    ) -> dict[str, torch.Tensor]:
        """target_codes given → teacher-forced logits for CE training; otherwise a greedy
        KV-cached decode."""
        h = self.project_conditioning(backbone_hidden, frame_positions)
        if target_codes is None:
            return {"codes": self.decode_codes(h)}
        target_codes = target_codes.detach().clone().long()
        start = torch.full((h.size(0), 1), self.start_token, device=h.device, dtype=torch.long)
        embeds = self.code_embeds(torch.cat([start, target_codes], dim=1)) + h.unsqueeze(1)
        attn_mask = self.causal_mask[: embeds.shape[1], : embeds.shape[1]].to(h.device)
        out = self.transformer(embeds, mask=attn_mask)
        # out[:, i] predicts codebook i.
        return {"logits": self._apply_heads(out[:, : self.num_codebooks])}
