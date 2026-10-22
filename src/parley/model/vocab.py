"""Real-vs-padded LM-head vocab bookkeeping: alignment pad columns must never reach a
softmax, an argmax or a top-k."""

from __future__ import annotations

from typing import Any

import torch

#: Recorded on every config that carries vocab facts, so a resume from a padded
#: checkpoint slices correctly even when the padding knob is off in that run.
REAL_VOCAB_SIZE_ATTR = "fdg_real_vocab_size"


def validate_pad_vocab_to_multiple(pad_to_multiple: Any) -> int:
    """0 disables. bf16 cuBLASLt needs an 8-element leading dim, so anything smaller
    (or non-power-of-two) buys nothing and would silently keep the align1 fallback:
    the LM-head GEMM measured 4-6% of B200 peak at vocab 262149."""
    pad = int(pad_to_multiple or 0)
    if pad == 0:
        return 0
    if pad < 8 or pad & (pad - 1):
        raise ValueError(f"model.pad_vocab_to_multiple must be 0 (off) or a power of two >= 8; got {pad_to_multiple!r}")
    return pad


def _vocab_configs(model: Any) -> list[Any]:
    config = getattr(model, "config", None)
    if config is None:
        return []
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)
    return configs


def set_real_vocab_size(model: Any, size: int) -> None:
    """Record the tokenizer-true vocab size (the padded table stays in ``vocab_size``)."""
    for config in _vocab_configs(model):
        setattr(config, REAL_VOCAB_SIZE_ATTR, int(size))


def real_vocab_size(model: Any) -> int | None:
    for config in _vocab_configs(model):
        size = getattr(config, REAL_VOCAB_SIZE_ATTR, None)
        if size is not None:
            return int(size)
    return None


def slice_real_vocab(logits: torch.Tensor, model: Any) -> torch.Tensor:
    """A view of ``logits`` without the alignment-pad columns; unchanged when unpadded.

    The head is frozen, so pad rows keep their resize-time init forever — entering a CE
    denominator or an argmax they would be pure noise.
    """
    size = real_vocab_size(model)
    if size is None:
        return logits
    width = int(logits.shape[-1])
    if width < size:
        raise ValueError(
            f"logits width {width} is narrower than the recorded real vocab size {size} "
            "— the head and the tokenizer have diverged"
        )
    if width == size:
        return logits
    return logits[..., :size]


def mask_pad_vocab_(logits: torch.Tensor, model: Any) -> torch.Tensor:
    """In-place ``-inf`` on the alignment-pad columns; the loss-path twin of
    ``slice_real_vocab``.

    Same CE and same gradient as the slice (exp(-inf) = 0 in the softmax denominator,
    zero gradient on the pad columns), but ``logits`` stays a contiguous [M, padded]
    tensor: the slice view made every downstream cat/CE copy the full [M, real] tensor
    and its backward zero-fill a [M, padded] one. Safe for autograd
    because a GEMM saves its inputs, not its output.
    """
    size = real_vocab_size(model)
    if size is None:
        return logits
    width = int(logits.shape[-1])
    if width < size:
        raise ValueError(
            f"logits width {width} is narrower than the recorded real vocab size {size} "
            "— the head and the tokenizer have diverged"
        )
    if width == size:
        return logits
    logits[..., size:] = float("-inf")
    return logits
