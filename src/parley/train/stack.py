"""Load the trainable stack: Gemma 4 Unified under LoRA, the depth head, the feedback embedding,
the duplex control tokens. Performance choices with a concluded experiment are constants here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, assert_never

import torch
import torch._dynamo as dynamo
import torch._functorch.config as functorch_config
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoProcessor

from parley.model.audio_feedback import (
    AudioFeedbackConfig,
    AudioFeedbackEmbedding,
    load_audio_feedback,
)
from parley.model.backbone import GemmaBackbone, stored_weight
from parley.model.csm_depth import PINNED_CSM, CsmDepthHead, CsmSource, load_pretrained_decoder
from parley.model.depth_head import DepthHead
from parley.model.depth_transformer import DepthTransformer
from parley.model.special_tokens import (
    ensure_duplex_special_tokens,
    load_duplex_special_token_state,
)
from parley.model.talker_state import TalkerState, load_talker_state
from parley.rows.layout import LAYOUT_L, Layout, has_own_voice
from parley.rows.schema import TokenIds
from parley.train.config import (
    BASE_MODEL,
    BASE_REVISION,
    CODEBOOKS,
    FEEDBACK_INIT_STD,
    LORA_ALPHA,
    LORA_RANK,
    CsmDepthCfg,
    DepthCfg,
    SharedDepthCfg,
)

# Performance choices whose experiments concluded; each changes only through a new experiment.
ATTN_IMPLEMENTATION = "fdg_training_attention"
PAD_VOCAB_TO_MULTIPLE = 128
COMPILE_DYNAMIC = True  # "regional-dyn": per-layer torch.compile, dynamic shapes
ADAPTER_COMPUTE = torch.bfloat16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CODEBOOK_SIZE = 2048


@dataclass
class TrainStack:
    backbone: GemmaBackbone
    processor: object
    #: A voice of the model's own: the depth head that decodes its codes and the feedback tables
    #: that read them back. The talker layouts have neither.
    depth: DepthHead | None
    feedback: AudioFeedbackEmbedding | None
    token_ids: TokenIds
    layout: Layout = LAYOUT_L
    talker_state: TalkerState | None = None

    @property
    def own_voice(self) -> AudioFeedbackEmbedding | None:
        """The feedback the layout adds to each assistant position, if any."""
        return self.feedback if has_own_voice(self.layout) else None

    @property
    def playing_input(self) -> TalkerState | None:
        """The playing-or-idle input, in a layout whose voice is a separate talker."""
        return None if has_own_voice(self.layout) else self.talker_state

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = [p for p in self.backbone.model.parameters() if p.requires_grad]
        params += self.voice_parameters()
        params += self.state_parameters()
        return params

    def voice_parameters(self) -> list[torch.nn.Parameter]:
        voice = [module for module in (self.depth, self.feedback) if module is not None]
        return [p for module in voice for p in module.parameters() if p.requires_grad]

    def state_parameters(self) -> list[torch.nn.Parameter]:
        playing = self.playing_input
        return [] if playing is None else [p for p in playing.parameters() if p.requires_grad]

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        """LoRA matrices and trainable token rows, optimised directly."""
        return [
            p
            for n, p in self.backbone.model.named_parameters()
            if p.requires_grad and ("lora_" in n or "trainable_tokens" in n)
        ]


def load_stack(
    depth_cfg: DepthCfg | None,
    *,
    layout: Layout,
    resume_dir: Path | None,
    device: str,
    purpose: str = "train",
    lora_dropout: float = 0.0,
) -> TrainStack:
    """One backbone per process, pinned to `device` (never device_map=auto under DDP).

    `purpose="train"` installs the training attention stack (flash attention for the sliding
    layers, static-shape compiled global attention, both causal masks assuming right-padded full
    sequences, per-layer compile). `purpose="infer"` loads plain SDPA with no compile: at
    decode a handful of new queries attend a long KV cache, which the training kernels' mask
    semantics do not cover (the p21 render smoke spoke garbage through them)."""
    if purpose not in ("train", "infer"):
        raise ValueError(f"purpose must be 'train' or 'infer', got {purpose!r}")
    training = purpose == "train"
    saved = resume_dir / "processor" if resume_dir else None
    processor = (
        AutoProcessor.from_pretrained(str(saved))
        if saved and saved.is_dir()
        else AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    report = ensure_duplex_special_tokens(processor, model, pad_vocab_to_multiple=PAD_VOCAB_TO_MULTIPLE)
    ids = report["token_ids"]
    token_ids = TokenIds(
        channel_live=int(ids["CHANNEL_LIVE"]),
        user_audio=int(ids["USER_AUDIO_SLOT"]),
        eps=int(ids["EPS"]),
        sil=int(ids["SIL"]),
        assistant_slot=int(ids["ASSISTANT_SLOT"]),
    )
    if resume_dir is not None:
        load_duplex_special_token_state(resume_dir / "duplex_special_tokens.pt", processor, model)

    control_rows = sorted(token_ids.model_dump().values())
    n_layers = int(model.config.text_config.num_hidden_layers)
    if resume_dir is not None:
        model = resume_adapters(model, resume_dir / "lora", lora_dropout=lora_dropout)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules=list(LORA_TARGETS),
                # Explicit, as every trained adapter records it; None would also match target
                # names outside the numbered decoder layers.
                layers_to_transform=list(range(n_layers)),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                trainable_token_indices=control_rows,
            ),
        )
    if training and not any("trainable_tokens" in name for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("PEFT exposes no trainable control-token deltas; the adapter was trained without them")
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_" in name or "trainable_tokens" in name):
            p.data = p.data.to(torch.float32)

    backbone = GemmaBackbone.wrap(model)
    if training:
        compute_adapters_in(model, ADAPTER_COMPUTE)
        overlay_trainable_tokens(model)
        _compile_decoder_layers(backbone)

    depth, feedback = (
        _own_voice(depth_cfg, backbone.hidden_size, device=device, resume_dir=resume_dir)
        if has_own_voice(layout)
        else (None, None)
    )
    if training and depth is not None:
        depth.forward = torch.compile(depth.forward, dynamic=COMPILE_DYNAMIC)  # ty: ignore[invalid-assignment] - an instance's own forward
    talker_state = load_talker_state(resume_dir, backbone.hidden_size, device)
    talker_state.requires_grad_(not has_own_voice(layout))

    model.train()
    return TrainStack(
        backbone=backbone,
        processor=processor,
        depth=depth,
        feedback=feedback,
        token_ids=token_ids,
        layout=layout,
        talker_state=talker_state,
    )


def _own_voice(
    cfg: DepthCfg | None, hidden_size: int, *, device: str, resume_dir: Path | None
) -> tuple[DepthHead, AudioFeedbackEmbedding]:
    if cfg is None:
        raise ValueError("a layout with a voice of its own needs a depth head")
    depth = build_depth(cfg, hidden_size, device=device, resume_dir=resume_dir).train()
    feedback = build_feedback(hidden_size, device=device, resume_dir=resume_dir).train()
    return depth, feedback


def resume_adapters(model: nn.Module, path: Path, *, lora_dropout: float) -> PeftModel:
    """The saved adapter, trainable, with the given dropout rather than the one saved with it."""
    config = LoraConfig.from_pretrained(str(path))
    config.lora_dropout = lora_dropout
    return PeftModel.from_pretrained(model, str(path), config=config, is_trainable=True)


def build_depth(
    cfg: DepthCfg,
    hidden_size: int,
    *,
    device: str,
    resume_dir: Path | None,
    csm: CsmSource = PINNED_CSM,
) -> DepthHead:
    depth: DepthHead
    match cfg:
        case SharedDepthCfg():
            depth = DepthTransformer(
                hidden_size=hidden_size,
                num_codebooks=CODEBOOKS,
                codebook_size=CODEBOOK_SIZE,
                depth_layers=cfg.layers,
                depth_hidden=cfg.hidden,
            )
        case CsmDepthCfg():
            depth = CsmDepthHead(
                hidden_size,
                csm.config(CODEBOOKS),
                codebook_size=CODEBOOK_SIZE,
                text_window=cfg.text_window,
            )
            if resume_dir is None and cfg.init == "sesame-csm-1b":
                load_pretrained_decoder(depth, csm.tensors())
        case unreachable:
            assert_never(unreachable)
    depth = depth.to(device, dtype=torch.float32)
    if resume_dir is not None:
        state = torch.load(resume_dir / "depth_transformer.pt", map_location=device, weights_only=True)
        depth.load_state_dict({k: v for k, v in state.items() if not _unused_depth_key(k)})
    return depth


def fetch_depth_weights(cfg: DepthCfg, *, csm: CsmSource = PINNED_CSM) -> None:
    """Ranks that build at once must find CSM's files already in the shared cache."""
    if isinstance(cfg, CsmDepthCfg):
        csm.config(CODEBOOKS)
        if cfg.init == "sesame-csm-1b":
            csm.tensors()


def _unused_depth_key(key: str) -> bool:
    """Checkpoints saved before the depth head lost its never-used cross-frame modules still
    carry their weights; everything else must match exactly."""
    return key in ("codebook_level_emb.weight", "prev_frame_start") or key.startswith("temporal_transformer.")


def build_feedback(hidden_size: int, *, device: str, resume_dir: Path | None) -> AudioFeedbackEmbedding:
    if resume_dir is not None:
        feedback, _ = load_audio_feedback(resume_dir / "audio_feedback.pt", map_location=device)
    else:
        feedback = AudioFeedbackEmbedding(
            AudioFeedbackConfig(
                num_codebooks=CODEBOOKS,
                codebook_size=CODEBOOK_SIZE,
                hidden_size=hidden_size,
                init_std=FEEDBACK_INIT_STD,
            )
        )
    return feedback.to(device, dtype=torch.float32)


def _compile_decoder_layers(backbone: GemmaBackbone) -> None:
    """Compile every decoder layer; attention graph-breaks out to its own kernels. Activation
    checkpointing can't wrap these layers: a compiled forward re-run inside
    torch.utils.checkpoint fails the recompute-metadata check."""
    decoder_layers = backbone.model.get_base_model().get_decoder().layers
    # One compiled-attention entry per (B, S_pad); past the limit dynamo falls back to eager
    # silently, which for that kernel is slower than sdpa.
    dynamo.config.cache_size_limit = 256
    dynamo.config.accumulated_cache_size_limit = 4096
    # The compiled global attention is a custom autograd.Function; without this every new
    # S_pad re-runs partitioning even when the inductor cache hits.
    functorch_config.autograd_cache_allow_custom_autograd_functions = True
    for layer in decoder_layers:
        layer.forward = torch.compile(layer.forward, dynamic=COMPILE_DYNAMIC, fullgraph=False)
    print(
        f"[stack] {len(decoder_layers)} decoder layers compiled (dynamic={COMPILE_DYNAMIC})",
        flush=True,
    )


def compute_adapters_in(model: nn.Module, dtype: torch.dtype) -> None:
    """PEFT multiplies each adapter in its weights' FP32, casting the projection's input up and
    adding an FP32 update at the projection's full width. The FP32 weights still take the gradients
    and updates."""
    forward = _adapter_forward_in(dtype)
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.forward = MethodType(forward, module)  # ty: ignore[invalid-assignment] - an instance's own forward


def _adapter_forward_in(dtype: torch.dtype):
    def forward(self: LoraLinear, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            return LoraLinear.forward(self, x, *args, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        for name in self.active_adapters:
            if name not in self.lora_A:
                continue
            a, b = self.lora_A[name].weight.to(dtype), self.lora_B[name].weight.to(dtype)
            inputs = self.lora_dropout[name](x.to(dtype))
            update = F.linear(F.linear(inputs, a), b) * self.scaling[name]
            result = result + update.to(result.dtype)
        return result

    return forward


def overlay_trainable_tokens(model: nn.Module) -> None:
    """PEFT's control-token layers copy their row indices from host memory on every call, which
    waits for the device, and merge the trained rows into a copy of the whole table, which then
    takes a gradient. The overlay looks the frozen table up and swaps the trained rows in, for
    the same values."""
    for module in model.modules():
        if isinstance(module, TrainableTokensLayer):
            weight = stored_weight(module)
            indices = {name: torch.tensor(rows, device=weight.device) for name, rows in module.token_indices.items()}
            padding = getattr(module.base_layer, "padding_idx", None)
            if padding is not None and any(padding in rows for rows in module.token_indices.values()):
                raise ValueError(f"the padding row {padding} can't be a trained control row")
            slots = {name: _slots(rows, weight.shape[0]) for name, rows in indices.items()}
            module.get_merged_weights = MethodType(_merged_weights_with(indices), module)  # ty: ignore[invalid-assignment] - an instance's own method
            module.forward = MethodType(_overlaid_forward(indices, slots), module)  # ty: ignore[invalid-assignment] - an instance's own forward


def _slots(rows: torch.Tensor, table_rows: int) -> torch.Tensor:
    """For every row of the table, its place among the trained rows, or their count if it isn't
    one: the index of a zero row that takes no gradient."""
    trained = rows.shape[0]
    slots = torch.full((table_rows,), trained, dtype=torch.long, device=rows.device)
    return slots.index_copy(0, rows, torch.arange(trained, device=rows.device))


def _overlaid_forward(indices: dict[str, torch.Tensor], slots: dict[str, torch.Tensor]):
    def forward(self: TrainableTokensLayer, x: torch.Tensor, *args: Any, **kwargs: Any):
        if self.disable_adapters or self.merged or not self.active_adapters:
            return TrainableTokensLayer.forward(self, x, *args, **kwargs)
        self._check_overlapping_tokens(self.active_adapters)
        base = self.base_layer
        if isinstance(base, nn.Embedding):
            result = F.embedding(x, base.weight, padding_idx=base.padding_idx)
            for name in self.active_adapters:
                slot = slots[name][x]
                trained = self.trainable_tokens_delta[name].to(result.dtype)
                table = torch.cat([trained, trained.new_zeros(1, trained.shape[1])])
                rows = F.embedding(slot, table, padding_idx=trained.shape[0])
                result = torch.where((slot < trained.shape[0])[..., None], rows, result)
            scale = self._get_embed_scale()
            return result if scale is None else result * scale.to(result.dtype)
        if isinstance(base, nn.Linear):
            result = F.linear(x, base.weight)
            for name in self.active_adapters:
                rows = self.trainable_tokens_delta[name].to(result.dtype)
                result = result.index_copy(-1, indices[name], F.linear(x, rows))
            return result
        raise TypeError(f"control-token rows can't overlay a {type(base).__name__}")

    return forward


def _merged_weights_with(indices: dict[str, torch.Tensor]):
    def get_merged_weights(self: TrainableTokensLayer, active_adapters) -> torch.Tensor:
        weight = stored_weight(self)
        for name in active_adapters:
            deltas = self.trainable_tokens_delta[name].to(weight)
            weight = weight.index_copy(0, indices[name], deltas)
        return weight

    return get_merged_weights
