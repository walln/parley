"""Pad a list of rows into the tensors one training step consumes. Built on the CPU (prefetch
child or main thread); the step moves the whole batch to the device in one shot. Every per-frame,
per-slot and per-slice array is as long as the padded sequence allows, so a batch's shapes follow
from its row count and padded length alone, and a captured step can replay on any batch of that
shape."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import torch

from parley.rows.layout import Layout, SequencePlan, plan_sequence, tokens_per_frame
from parley.rows.schema import TokenIds
from parley.rows.store import Row

SLICE_SAMPLES = 640  # one Gemma audio token = 40 ms at 16 kHz
PAD_ID = 0


@dataclass
class SessionBatch:
    input_ids: torch.Tensor  # [B, S] int64, padded with PAD_ID
    attention_mask: torch.Tensor  # [B, S] int64
    position_ids: torch.Tensor  # [B, S] int64, the position number the backbone sees
    user_pos: torch.Tensor  # [B, U] int64; padding cells hold S, a scratch position past the end
    user_pcm: torch.Tensor  # [B, U, 640] float32 in [-1, 1]
    anchor_pos: torch.Tensor  # [B, F] int64
    feedback_pos: torch.Tensor  # [B, F] int64: frame f's anchor for 1 <= f < frames, else S
    frame_mask: torch.Tensor  # [B, F] bool
    codes: torch.Tensor  # [B, F, C] int64
    playing: torch.Tensor  # [B, F] bool: the talker played the frame before (talker layout)
    frame_text_weights: torch.Tensor  # [B, F] float32 (data weights; 1.0 = plain)
    text_pos: torch.Tensor  # [B, T] int64
    text_mask: torch.Tensor  # [B, T] bool
    text_targets: torch.Tensor  # [B, T] int64
    text_kind: torch.Tensor  # [B, T] uint8
    text_frame: torch.Tensor  # [B, T] int64
    lengths: torch.Tensor  # [B] int64 real sequence lengths
    frames: torch.Tensor  # [B] int64
    row_indices: list[int]
    prompts: list[str]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> SessionBatch:
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> SessionBatch:
        return self._map(torch.Tensor.pin_memory)

    def _map(self, move) -> SessionBatch:
        moved = {}
        for f in fields(self):
            v = getattr(self, f.name)
            moved[f.name] = move(v) if torch.is_tensor(v) else v
        return SessionBatch(**moved)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])


def collate(
    rows: list[Row],
    prefixes: list[np.ndarray],
    *,
    ids: TokenIds,
    layout: Layout,
    pad_to_multiple: int = 1,
    user_audio: Callable[[Row], np.ndarray] | None = None,
) -> SessionBatch:
    if not rows:
        raise ValueError("collate needs at least one row")
    if len(prefixes) != len(rows):
        raise ValueError("one prefix per row")
    plans = [plan_sequence(r, prefix_len=int(p.shape[0]), ids=ids, layout=layout) for r, p in zip(rows, prefixes)]
    return collate_plans(
        plans,
        prefixes,
        rows,
        layout=layout,
        pad_to_multiple=pad_to_multiple,
        user_audio=user_audio,
    )


def collate_plans(
    plans: list[SequencePlan],
    prefixes: list[np.ndarray],
    rows: list[Row],
    *,
    layout: Layout,
    pad_to_multiple: int = 1,
    user_audio: Callable[[Row], np.ndarray] | None = None,
) -> SessionBatch:
    """The batch of sequences already planned, each with its row's audio, codes and weights.
    `user_audio` gives the audio the model hears for a row, float32 in [-1, 1]; the plans and
    their labels come from the row's own audio."""
    b = len(rows)
    seq = _round_up(max(p.length for p in plans), pad_to_multiple)
    per = tokens_per_frame(layout)
    frames = seq // per
    users = frames * (per - 1)
    texts = frames
    codebooks = int(rows[0].codes.shape[1])

    input_ids = np.full((b, seq), PAD_ID, dtype=np.int64)
    attention = np.zeros((b, seq), dtype=np.int64)
    position_ids = np.zeros((b, seq), dtype=np.int64)
    user_pos = np.full((b, users), seq, dtype=np.int64)
    user_pcm = np.zeros((b, users, SLICE_SAMPLES), dtype=np.float32)
    anchor_pos = np.zeros((b, frames), dtype=np.int64)
    feedback_pos = np.full((b, frames), seq, dtype=np.int64)
    frame_mask = np.zeros((b, frames), dtype=bool)
    codes = np.zeros((b, frames, codebooks), dtype=np.int64)
    frame_w = np.ones((b, frames), dtype=np.float32)
    text_pos = np.zeros((b, texts), dtype=np.int64)
    text_mask = np.zeros((b, texts), dtype=bool)
    text_targets = np.zeros((b, texts), dtype=np.int64)
    text_kind = np.zeros((b, texts), dtype=np.uint8)
    text_frame = np.zeros((b, texts), dtype=np.int64)

    for i, (row, prefix, plan) in enumerate(zip(rows, prefixes, plans)):
        n = plan.length
        input_ids[i, : plan.prefix_len] = prefix
        input_ids[i, plan.prefix_len : n] = plan.live_ids
        attention[i, :n] = 1
        position_ids[i, :n] = plan.position_ids
        position_ids[i, n:] = plan.position_ids[-1] + 1 + np.arange(seq - n)
        u = plan.user_pos.shape[0]
        user_pos[i, :u] = plan.user_pos
        if u:
            heard = _as_float(row.user_pcm) if user_audio is None else user_audio(row)
            user_pcm[i, :u] = _slices(heard, u)
        f = plan.frames
        anchor_pos[i, :f] = plan.anchor_pos
        feedback_pos[i, 1:f] = plan.anchor_pos[1:]
        frame_mask[i, :f] = True
        codes[i, :f] = row.codes[:f]
        frame_w[i, : row.frames] = row.text_weights
        t = plan.text_pos.shape[0]
        text_pos[i, :t] = plan.text_pos
        text_mask[i, :t] = True
        text_targets[i, :t] = plan.text_targets
        text_kind[i, :t] = plan.text_kind
        text_frame[i, :t] = plan.text_frame

    return SessionBatch(
        input_ids=torch.from_numpy(input_ids),
        attention_mask=torch.from_numpy(attention),
        position_ids=torch.from_numpy(position_ids),
        user_pos=torch.from_numpy(user_pos),
        user_pcm=torch.from_numpy(user_pcm),
        anchor_pos=torch.from_numpy(anchor_pos),
        feedback_pos=torch.from_numpy(feedback_pos),
        frame_mask=torch.from_numpy(frame_mask),
        codes=torch.from_numpy(codes),
        playing=torch.from_numpy(_playing(plans, frames)),
        frame_text_weights=torch.from_numpy(frame_w),
        text_pos=torch.from_numpy(text_pos),
        text_mask=torch.from_numpy(text_mask),
        text_targets=torch.from_numpy(text_targets),
        text_kind=torch.from_numpy(text_kind),
        text_frame=torch.from_numpy(text_frame),
        lengths=torch.tensor([p.length for p in plans], dtype=torch.int64),
        frames=torch.tensor([p.frames for p in plans], dtype=torch.int64),
        row_indices=[r.index for r in rows],
        prompts=[r.prompt for r in rows],
    )


def _playing(plans: list, frames: int) -> np.ndarray:
    """[B, F] bool: whether the talker played the frame before, in a layout with a talker."""
    playing = np.zeros((len(plans), frames), dtype=bool)
    for i, plan in enumerate(plans):
        if plan.playing is not None:
            playing[i, : plan.frames] = plan.playing[: plan.frames]
    return playing


def _as_float(pcm: np.ndarray) -> np.ndarray:
    # int16 straight from a shard; float32 input is taken as already scaled to [-1, 1].
    return pcm if pcm.dtype == np.float32 else pcm.astype(np.float32) / 32768.0


def _slices(pcm: np.ndarray, n_tokens: int) -> np.ndarray:
    """[n_tokens, 640]: each user token is one 40 ms slice, consumed in position order."""
    if pcm.shape[0] != n_tokens * SLICE_SAMPLES:
        raise ValueError(f"{pcm.shape[0]} audio samples for {n_tokens} user tokens")
    return pcm.reshape(n_tokens, SLICE_SAMPLES)


def _round_up(n: int, multiple: int) -> int:
    return n if multiple <= 1 else ((n + multiple - 1) // multiple) * multiple
