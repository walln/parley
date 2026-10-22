"""A depth head built on the decoder of Sesame's CSM-1B: the frame's conditioning is the first
position, each codebook has its own input table and output head, and the decoder can start from
CSM-1B's weights at a pinned revision."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.csm.configuration_csm import CsmConfig, CsmDepthDecoderConfig
from transformers.models.csm.modeling_csm import (
    CsmCodebooksHead,
    CsmDecoderLayer,
    CsmDepthDecoderModel,
)

from parley.model.depth_transformer import MAX_FRAME_POSITIONS

CSM_REPO = "sesame/csm-1b"
CSM_REVISION = "c92a71e1c419772e25be7dc14d952c2521a740ab"
CSM_SHARD_INDEX = "transformers.safetensors.index.json"
DECODER_PREFIX = "depth_decoder.model."
HEADS_KEY = "depth_decoder.codebooks_head.weight"
#: CSM ties the decoder's input tables to its backbone's audio embeddings, so its checkpoint may
#: store them under either name.
EMBED_KEYS = (
    f"{DECODER_PREFIX}embed_tokens.weight",
    "backbone_model.embed_tokens.embed_audio_tokens.weight",
)
#: The decoder's attention: every row is a whole causal prefix, so SDPA needs no mask, and building
#: one makes Transformers check whether it can skip it, which waits for the device.
CAUSAL_ATTENTION = "fdg_whole_prefix_causal"
ALL_ATTENTION_FUNCTIONS.register(CAUSAL_ATTENTION, sdpa_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(CAUSAL_ATTENTION, lambda **_kwargs: None)
NEW_KEYS = frozenset(
    {
        "in_proj.weight",
        "in_proj.bias",
        "frame_pos_emb.weight",
        "first_head.weight",
        "first_head.bias",
    }
)


def _argmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


class CsmDepthHead(nn.Module):
    """Positions are [conditioning, code 0, ..., code 6]. The first position's output predicts
    codebook 0 through a head of its own, and position k's predicts codebook k through CSM's."""

    def __init__(
        self,
        hidden_size: int,
        config: CsmDepthDecoderConfig,
        *,
        codebook_size: int,
        text_window: int = 0,
    ):
        super().__init__()
        if config.num_codebooks is None:
            raise ValueError("the CSM decoder config names no codebook count")
        if config.vocab_size < codebook_size:
            raise ValueError(f"CSM's vocabulary of {config.vocab_size} can't hold {codebook_size}")
        self.num_codebooks: int = config.num_codebooks
        self.vocab_size: int = config.vocab_size
        self.codebook_size = codebook_size
        std = config.initializer_range
        self.in_proj = nn.Linear(hidden_size, config.backbone_hidden_size)
        self.frame_pos_emb = nn.Embedding(MAX_FRAME_POSITIONS, config.backbone_hidden_size)
        self.decoder = CsmDepthDecoderModel(config)
        self.first_head = nn.Linear(config.hidden_size, codebook_size)
        self.heads = CsmCodebooksHead(config.hidden_size, self.num_codebooks, config.vocab_size)
        for weight in (self.in_proj.weight, self.frame_pos_emb.weight, self.first_head.weight):
            nn.init.normal_(weight, std=std)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.first_head.bias)
        nn.init.normal_(self.heads.weight, std=std)
        self.text_window = text_window
        self.text_proj = (
            nn.Linear(text_window * hidden_size, config.backbone_hidden_size, bias=False) if text_window else None
        )
        if self.text_proj is not None:
            nn.init.zeros_(self.text_proj.weight)

    @property
    def new_keys(self) -> frozenset[str]:
        return NEW_KEYS if self.text_proj is None else NEW_KEYS | {"text_proj.weight"}

    @property
    def update_sentinel(self) -> nn.Parameter:
        layer = self.decoder.layers[0]
        assert isinstance(layer, CsmDecoderLayer)
        return layer.input_layernorm.weight

    def project_conditioning(self, backbone_hidden, frame_positions=None, text_embeds=None):
        """`text_embeds` [N, text_window, H] are the chosen text tokens' embeddings, this frame's
        first, which a head with a text window needs and any other head refuses."""
        if backbone_hidden.dim() == 3:
            backbone_hidden = backbone_hidden[:, -1]
        h = self.in_proj(backbone_hidden.to(self.in_proj.weight.dtype))
        if frame_positions is not None:
            positions = frame_positions.to(device=h.device, dtype=torch.long)
            h = h + self.frame_pos_emb(positions.clamp(min=0, max=MAX_FRAME_POSITIONS - 1))
        if (self.text_proj is None) != (text_embeds is None):
            raise ValueError(f"this head reads a text window of {self.text_window} tokens")
        if self.text_proj is not None:
            h = h + self.text_proj(text_embeds.flatten(1).to(self.text_proj.weight.dtype))
        return h

    def _hidden(self, conditioning: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        placeholder = torch.zeros((codes.shape[0], 1), dtype=torch.long, device=codes.device)
        ids = torch.cat([placeholder, codes], dim=1)
        # Without a mask, transformers reads the decoder's one-dimensional positions as packed
        # sequences and fails whenever the prefix is exactly as long as the batch.
        out = self.decoder(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            backbone_last_hidden_state=conditioning,
            use_cache=False,
        )
        return out.last_hidden_state

    def _logits(self, hidden: torch.Tensor, codebook: int) -> torch.Tensor:
        if codebook == 0:
            return self.first_head(hidden)
        return (hidden @ self.heads.weight[codebook - 1])[..., : self.codebook_size]

    def decode_codes(
        self,
        conditioning: torch.Tensor,
        *,
        choose: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Each frame's codes, chosen one codebook after another after `prefix`, the [N, k] codes
        of its first k codebooks given rather than chosen."""
        choose = _argmax if choose is None else choose
        codes = (
            torch.zeros((conditioning.shape[0], 0), dtype=torch.long, device=conditioning.device)
            if prefix is None
            else prefix.long()
        )
        for codebook in range(codes.shape[1], self.num_codebooks):
            hidden = self._hidden(conditioning, codes)[:, -1]
            codes = torch.cat([codes, choose(self._logits(hidden, codebook))[:, None]], dim=1)
        return codes

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        target_codes: torch.Tensor | None = None,
        frame_positions: torch.Tensor | None = None,
        text_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditioning = self.project_conditioning(backbone_hidden, frame_positions, text_embeds)
        if target_codes is None:
            return {"codes": self.decode_codes(conditioning)}
        hidden = self._hidden(conditioning, target_codes.long()[:, :-1])
        rest = torch.einsum("bcd,cdv->bcv", hidden[:, 1:], self.heads.weight)
        first = self.first_head(hidden[:, 0])
        return {"logits": torch.cat([first[:, None], rest[..., : self.codebook_size]], dim=1)}


def decoder_state(tensors: Mapping[str, torch.Tensor], *, codebooks: int, vocab: int) -> dict[str, torch.Tensor]:
    embed_key = next((key for key in EMBED_KEYS if key in tensors), None)
    if embed_key is None or HEADS_KEY not in tensors:
        raise KeyError(f"CSM tensors lack the decoder's input tables or {HEADS_KEY}")
    state = {
        f"decoder.{key.removeprefix(DECODER_PREFIX)}": value
        for key, value in tensors.items()
        if key.startswith(DECODER_PREFIX) and key != EMBED_KEYS[0]
    }
    state["decoder.embed_tokens.weight"] = tensors[embed_key][: codebooks * vocab]
    state["heads.weight"] = tensors[HEADS_KEY][: codebooks - 1]
    return state


def load_pretrained_decoder(head: CsmDepthHead, tensors: Mapping[str, torch.Tensor]) -> None:
    state = decoder_state(tensors, codebooks=head.num_codebooks, vocab=head.vocab_size)
    result = head.load_state_dict(state, strict=False)
    if set(result.missing_keys) != head.new_keys or result.unexpected_keys:
        raise RuntimeError(
            f"CSM-1B's decoder doesn't fit this head: missing {sorted(result.missing_keys)}, "
            f"unexpected {sorted(result.unexpected_keys)}"
        )


def pinned_config(num_codebooks: int) -> CsmDepthDecoderConfig:
    config = CsmConfig.from_pretrained(CSM_REPO, revision=CSM_REVISION).depth_decoder_config
    assert isinstance(config, CsmDepthDecoderConfig)
    config.num_codebooks = num_codebooks
    config._attn_implementation = CAUSAL_ATTENTION
    return config


def _decoder_key(key: str) -> bool:
    return key.startswith(DECODER_PREFIX) or key in (HEADS_KEY, *EMBED_KEYS)


def pinned_tensors() -> dict[str, torch.Tensor]:
    index = json.loads(Path(hf_hub_download(CSM_REPO, CSM_SHARD_INDEX, revision=CSM_REVISION)).read_text())
    wanted = {key: shard for key, shard in index["weight_map"].items() if _decoder_key(key)}
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        path = hf_hub_download(CSM_REPO, shard, revision=CSM_REVISION)
        with safe_open(path, framework="pt") as file:
            tensors |= {key: file.get_tensor(key) for key, s in wanted.items() if s == shard}
    return tensors


@dataclass(frozen=True)
class CsmSource:
    config: Callable[[int], CsmDepthDecoderConfig]
    tensors: Callable[[], Mapping[str, torch.Tensor]]


PINNED_CSM = CsmSource(config=pinned_config, tensors=pinned_tensors)
