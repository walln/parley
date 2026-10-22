import torch

from parley.model.csm_depth import CsmDepthHead
from parley.model.depth_transformer import DepthTransformer

DepthHead = DepthTransformer | CsmDepthHead


def text_window(depth: DepthHead) -> int:
    return depth.text_window if isinstance(depth, CsmDepthHead) else 0


def _text(text_embeds: torch.Tensor | None) -> dict[str, torch.Tensor]:
    return {} if text_embeds is None else {"text_embeds": text_embeds.flatten(0, 1)}


def sequence_logits(
    depth: DepthHead,
    hidden: torch.Tensor,
    codes: torch.Tensor,
    frame_positions: torch.Tensor,
    text_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    b, frames, _ = hidden.shape
    logits = depth(
        hidden.flatten(0, 1),
        target_codes=codes.flatten(0, 1),
        frame_positions=frame_positions.flatten(),
        **_text(text_embeds),
    )["logits"]
    return logits.view(b, frames, *logits.shape[1:])
