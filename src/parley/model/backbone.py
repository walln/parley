"""The one place that knows the HF/PEFT object layout of Gemma 4 Unified. Everything the training
step needs from the backbone is a method here, resolved and asserted once at construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from torch import nn
from transformers.cache_utils import Cache, StaticLayer
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedMultimodalEmbedder,
    Gemma4UnifiedTextModel,
)

from parley.model.vocab import mask_pad_vocab_, real_vocab_size

AUDIO_SLICE_SAMPLES = 640


@dataclass
class GemmaBackbone:
    model: nn.Module  # the trainable object (PeftModel or the bare model)
    embed_tokens: nn.Embedding
    lm_head: nn.Linear
    audio_embedder: Gemma4UnifiedMultimodalEmbedder
    decoder: Gemma4UnifiedTextModel
    final_norm: nn.Module
    audio_token_id: int
    softcap: float | None
    hidden_size: int

    @classmethod
    def wrap(cls, model: nn.Module) -> GemmaBackbone:
        base = model.get_base_model() if isinstance(model, PeftModel) else model
        if not isinstance(base, Gemma4UnifiedForConditionalGeneration):
            raise TypeError(
                f"expected Gemma4UnifiedForConditionalGeneration (optionally under PEFT), got {type(base).__name__}"
            )
        embedder = base.model.embed_audio
        if not isinstance(embedder, Gemma4UnifiedMultimodalEmbedder):
            raise TypeError("model has no audio embedder (audio_config is None)")
        if embedder.embedding_projection.in_features != AUDIO_SLICE_SAMPLES:
            raise ValueError(
                f"audio embedder consumes {embedder.embedding_projection.in_features}-sample "
                f"slices, expected {AUDIO_SLICE_SAMPLES}; this is not the raw-PCM projector"
            )
        decoder = base.get_decoder()
        if not isinstance(decoder, Gemma4UnifiedTextModel):
            raise TypeError(f"decoder is {type(decoder).__name__}, expected Gemma4UnifiedTextModel")
        audio_token_id = base.config.audio_token_id
        if audio_token_id is None:
            raise ValueError("config.audio_token_id is unset")
        text_cfg = base.config.text_config
        return cls(
            model=model,
            embed_tokens=base.get_input_embeddings(),
            lm_head=base.get_output_embeddings(),
            audio_embedder=embedder,
            decoder=decoder,
            final_norm=decoder.norm,
            audio_token_id=int(audio_token_id),
            softcap=text_cfg.final_logit_softcapping or None,
            hidden_size=int(text_cfg.hidden_size),
        )

    def merged(self) -> GemmaBackbone:
        """Folding rounds into the base dtype, and it unloads this backbone's PEFT model."""
        if not isinstance(self.model, PeftModel):
            raise TypeError("only a backbone under PEFT has adapters to merge")
        return GemmaBackbone.wrap(self.model.merge_and_unload())

    @property
    def device(self) -> torch.device:
        return stored_weight(self.embed_tokens).device

    @property
    def dtype(self) -> torch.dtype:
        return stored_weight(self.embed_tokens).dtype

    @property
    def head_weight(self) -> torch.Tensor:
        """The LM-head rows as the forward sees them (trainable-token deltas applied)."""
        return effective_weight(self.lm_head)

    def embed_audio(self, slices: torch.Tensor) -> torch.Tensor:
        """[N, 640] float PCM in [-1, 1] -> [N, H] in the backbone's dtype."""
        if slices.shape[-1] != AUDIO_SLICE_SAMPLES:
            raise ValueError(f"audio slices must be [..., {AUDIO_SLICE_SAMPLES}], got {tuple(slices.shape)}")
        return self.audio_embedder(slices.to(self.dtype))

    def final_hidden(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """The final-norm output [B, S, H], straight from the text decoder: the multimodal model
        around it would look for image, video and audio placeholders among the embeddings, which
        waits for the device. The explicit attention mask also keeps transformers from reading
        gapped position numbers as packed sequences."""
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

    def step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        past: object | None,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        """Incremental forward of only the new positions against a KV cache. Returns the
        final-norm hidden for those positions [B, new, H] and the updated cache. The cache is
        indexed by `cache_position`; rotary encoding reads `position_ids`, which can have gaps."""
        captured: dict[str, torch.Tensor] = {}

        def grab(_module, _inputs, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output

        handle = self.final_norm.register_forward_hook(grab)
        try:
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            handle.remove()
        return captured["h"], out.past_key_values

    def full_static_cache(self, max_positions: int) -> Cache:
        """Full length in the sliding layers too: Transformers' sliding static layer tracks its
        fill as a Python integer, which recompiles every frame."""
        return Cache(layers=[StaticLayer(max_cache_len=max_positions) for _ in self.decoder.layers])

    def cached_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cache: Cache,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """No attention mask: sessions batched together share their length, so the causal and
        sliding masks follow from `cache_position`."""
        return self.decoder(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
        ).last_hidden_state

    def head(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """LM head on the scored rows only: one GEMM, softcap, pad-vocab mask."""
        logits = self.lm_head(hidden_rows.to(self.dtype))
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return mask_pad_vocab_(logits, self.model)

    @property
    def real_vocab(self) -> int:
        size = real_vocab_size(self.model)
        return int(self.head_weight.shape[0]) if size is None else size

    def decode_view(self) -> DecodeView:
        """Snapshot for one decode. Exact: an embedding row depends only on its id, and the LM
        head is a bias-free GEMM on the same merged weight its forward builds."""
        table = next(m for m in self.embed_tokens.modules() if isinstance(m, nn.Embedding))
        linear = next(m for m in self.lm_head.modules() if isinstance(m, nn.Linear))
        if table.max_norm is not None or linear.bias is not None:
            raise ValueError("a decode view needs a plain embedding table and a bias-free LM head")
        if getattr(self.lm_head, "disable_adapters", False):
            raise ValueError("a decode view snapshots the adapted head; adapters are disabled")
        ids = torch.arange(table.num_embeddings, device=table.weight.device)
        return DecodeView(self, self.embed_tokens(ids), self.head_weight, self.dtype)


@dataclass(frozen=True)
class DecodeView:
    """The backbone's token rows and LM-head weight as its forward computes them, materialized
    once so decode steps skip PEFT's per-call merge of the trainable token rows into full
    tables. Valid only while the weights hold still: take a new view after any change."""

    backbone: GemmaBackbone
    rows: torch.Tensor
    head_weight: torch.Tensor
    dtype: torch.dtype

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.rows)

    def embed_audio(self, slices: torch.Tensor) -> torch.Tensor:
        return self.backbone.audio_embedder(slices.to(self.dtype))

    def head(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """`GemmaBackbone.head` on the snapshot weight."""
        logits = F.linear(hidden_rows.to(self.dtype), self.head_weight)
        if self.backbone.softcap:
            logits = torch.tanh(logits / self.backbone.softcap) * self.backbone.softcap
        return mask_pad_vocab_(logits, self.backbone.model)


def stored_weight(module: nn.Module) -> torch.Tensor:
    """An embedding's or head's own weight, without the trainable-token deltas, whose merge copies
    the whole table."""
    weight = getattr(module, "base_layer", module).weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__}.weight is a {type(weight).__name__}")
    return weight


def effective_weight(module: nn.Module) -> torch.Tensor:
    """Weight of an embedding/head, with PEFT trainable-token deltas merged in when present."""
    if isinstance(module, TrainableTokensLayer):
        return module.get_merged_weights(module.active_adapters)
    weight = module.weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__}.weight is a {type(weight).__name__}")
    return weight
