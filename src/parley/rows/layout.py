"""The frame layouts that turn a row into a backbone sequence: `prefix, <channel_live>`, then one
block of positions per 80 ms frame. Layout L gives each frame two user-audio positions and one
assistant position; the speech layout gives it the assistant position alone, for rows whose user
channel is silent. The talker layout keeps layout L's positions, writes each reply ahead of its
speech for a separate talker to say, and feeds back none of the model's own audio. The assistant
position's input is the previous frame's text/control token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np

from parley.rows.schema import LISTENING, SAMPLES_PER_FRAME, TokenIds
from parley.rows.store import Row

TEXT = 0
EPS = 1
SIL = 2
#: The SIL that ends a written-ahead reply.
END = 3
#: The EPS where a written-ahead reply yields to the user, its words not yet written dropped; in
#: the floor layout, the SIL owed on each frame the user talks over the talker.
STOP = 4
#: In the floor layout, the EPS that keeps the talker playing.
HOLD = 5

USER_TOKENS_PER_FRAME = 2
POSITIONS_PER_FRAME = USER_TOKENS_PER_FRAME + 1
#: Frames the text schedule leads the audio by (160 ms).
TEXT_LEAD_FRAMES = 2
#: Frames before its audio that a written-ahead reply starts (400 ms, the rows' lead-in).
WRITE_LEAD_FRAMES = 5
#: In the floor layout, the yield is owed from this many frames (160 ms) after the user starts
#: talking over the talker.
YIELD_ONSET_FRAMES = 2

#: The speech layout numbers its assistant positions as layout L does ("gapped", leaving the
#: user positions' numbers unused) or consecutively ("contiguous"). The talker layouts hand
#: speech to a separate talker; in "talker-floor" Gemma holds the floor frame by frame.
Layout = Literal["L", "speech-gapped", "speech-contiguous", "talker", "talker-floor"]
LAYOUT_L: Layout = "L"


def has_own_voice(layout: Layout) -> bool:
    return layout not in ("talker", "talker-floor")


@dataclass(frozen=True)
class SequencePlan:
    """Absolute indices into the full sequence (prefix included). All int64 arrays."""

    prefix_len: int
    live_ids: np.ndarray  # tokens after the prefix, starting with <channel_live>
    user_pos: np.ndarray  # [frames * user tokens per frame], time order
    anchor_pos: np.ndarray  # [frames]
    text_pos: np.ndarray  # [frames]; the token at text_pos[i] is predicted at -1
    text_targets: np.ndarray  # [frames]
    text_kind: np.ndarray  # [frames] uint8: TEXT / EPS / SIL
    text_frame: np.ndarray  # [frames] frame each slot is attributed to
    position_ids: np.ndarray  # [length] the position number the backbone sees at each index
    dropped_text: int = 0  # tokens still queued after the last frame
    #: [frames] bool in a talker layout: whether the talker played the frame before.
    playing: np.ndarray | None = None

    @property
    def length(self) -> int:
        return self.prefix_len + int(self.live_ids.shape[0])

    @property
    def frames(self) -> int:
        return int(self.anchor_pos.shape[0])


def tokens_per_frame(layout: Layout) -> int:
    match layout:
        case "L" | "talker" | "talker-floor":
            return POSITIONS_PER_FRAME
        case "speech-gapped" | "speech-contiguous":
            return 1
        case unreachable:
            assert_never(unreachable)


def frame_positions(layout: Layout, *, prefix_len: int, frames: np.ndarray) -> np.ndarray:
    """[len(frames), tokens_per_frame] position numbers of each listed frame's tokens."""
    start = prefix_len + 1 + POSITIONS_PER_FRAME * frames[:, None]
    match layout:
        case "L" | "talker" | "talker-floor":
            return start + np.arange(POSITIONS_PER_FRAME)
        case "speech-gapped":
            return start + USER_TOKENS_PER_FRAME
        case "speech-contiguous":
            return prefix_len + 1 + frames[:, None]
        case unreachable:
            assert_never(unreachable)


def text_schedule(row: Row, *, lead_frames: int, ids: TokenIds) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One text/control token per frame: a word's tokens are scheduled from `lead_frames`
    before its onset, one per frame, spilling forward; frames with no token carry EPS while
    the assistant's audio is speaking and SIL while it listens (frame_state of the same frame).
    Returns (tokens, kinds, onset frame per slot, dropped)."""
    frames = row.frames
    counts = row.text_counts.astype(np.int64)
    onsets = np.repeat(np.arange(frames), counts)
    due = np.maximum(onsets - lead_frames, 0)
    tokens = np.empty(frames, dtype=np.int64)
    kinds = np.empty(frames, dtype=np.uint8)
    slot_frame = np.empty(frames, dtype=np.int64)
    queue: list[tuple[int, int]] = []
    next_token = 0
    for t in range(frames):
        while next_token < onsets.shape[0] and due[next_token] <= t:
            queue.append((int(row.text_tokens[next_token]), int(onsets[next_token])))
            next_token += 1
        if queue:
            tokens[t], slot_frame[t] = queue.pop(0)
            kinds[t] = TEXT
        else:
            listening = row.frame_state[t] == LISTENING
            tokens[t] = ids.sil if listening else ids.eps
            kinds[t] = SIL if listening else EPS
            slot_frame[t] = t
    return tokens, kinds, slot_frame, len(queue)


def heard_frames(row: Row) -> np.ndarray:
    """[frames] bool: which frames carry any user audio."""
    samples = row.user_pcm[: row.frames * SAMPLES_PER_FRAME].reshape(row.frames, SAMPLES_PER_FRAME)
    return np.any(samples != 0, axis=1)


def yield_frames(row: Row) -> np.ndarray:
    """The frames where the assistant falls silent while the user is speaking: a reply cut off by
    an interruption, since every other turn waits for the one before to end."""
    speaking = row.frame_state != LISTENING
    ends = np.flatnonzero(~speaking[1:] & speaking[:-1]) + 1
    return ends[heard_frames(row)[ends - 1]]


def played_frames(row: Row) -> np.ndarray:
    """[frames] bool: whether the talker played the frame before, as a live frame knows it."""
    speaking = row.frame_state != LISTENING
    return np.concatenate([[False], speaking[:-1]])


def written_ahead_schedule(
    row: Row, *, lead_frames: int, ids: TokenIds
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One token per frame for a separate talker: each reply's tokens are written from
    `lead_frames` before its speech starts, but never before the user audio ahead of it has
    ended, then SIL ends the reply (kind END) and fills every frame until the next. A reply is a
    run of speaking frames. A reply cut off by the user yields with EPS (kind STOP) and its words
    not yet written are dropped. Returns (tokens, kinds, onset frame per slot, dropped)."""
    frames = row.frames
    onsets = np.repeat(np.arange(frames), row.text_counts.astype(np.int64))
    speaking = row.frame_state != LISTENING
    starts = np.flatnonzero(speaking & ~np.concatenate([[False], speaking[:-1]]))
    heard = heard_frames(row)
    last_heard = np.maximum.accumulate(np.where(heard, np.arange(frames), -1))
    before = np.where(starts > 0, last_heard[np.maximum(starts - 1, 0)], -1)
    reply_due = np.maximum(np.maximum(starts - lead_frames, before + 1), 0)
    reply = np.clip(np.searchsorted(starts, onsets, side="right") - 1, 0, None)
    due = reply_due[reply] if len(starts) else np.zeros_like(onsets)
    tokens = np.full(frames, ids.sil, dtype=np.int64)
    kinds = np.full(frames, SIL, dtype=np.uint8)
    slot_frame = np.arange(frames, dtype=np.int64)
    queue: list[tuple[int, int]] = []
    next_token, writing = 0, False
    yields = set(yield_frames(row).tolist())
    for t in range(frames):
        while next_token < onsets.shape[0] and due[next_token] <= t:
            queue.append((int(row.text_tokens[next_token]), int(onsets[next_token])))
            next_token += 1
        if t in yields:
            tokens[t], kinds[t] = ids.eps, STOP
            queue, writing = [], False
        elif queue:
            tokens[t], slot_frame[t] = queue.pop(0)
            kinds[t], writing = TEXT, True
        elif writing:
            kinds[t], writing = END, False
    return tokens, kinds, slot_frame, len(queue)


@dataclass(frozen=True)
class HeldSchedule:
    """The floor layout's frames. `inputs` is the token the next frame reads back: the target,
    except where a yield is owed and the talker plays on, as if Gemma had not taken it."""

    targets: np.ndarray
    inputs: np.ndarray
    kinds: np.ndarray
    slot_frame: np.ndarray
    playing: np.ndarray
    dropped: int


def _owed_yields(row: Row) -> np.ndarray:
    """[frames] bool: from YIELD_ONSET_FRAMES after the user starts talking over a reply to the
    frame its audio stops."""
    heard, owed = heard_frames(row), np.zeros(row.frames, dtype=bool)
    for stop in yield_frames(row).tolist():
        onset = stop - 1
        while onset > 0 and heard[onset - 1]:
            onset -= 1
        owed[min(onset + YIELD_ONSET_FRAMES, stop) : stop + 1] = True
    return owed


def held_schedule(row: Row, *, lead_frames: int, ids: TokenIds) -> HeldSchedule:
    """The written-ahead schedule with the floor held: while the talker plays, frames with no word
    carry EPS (kind HOLD, or END right after the last word), and SIL is owed (kind STOP) on every
    frame the user talks over it. The talker plays from the frame after a reply's first word to
    the frame after its audio ends."""
    frames = row.frames
    onsets = np.repeat(np.arange(frames), row.text_counts.astype(np.int64))
    speaking = row.frame_state != LISTENING
    edges = np.flatnonzero(np.diff(np.concatenate([[0], speaking.astype(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    last_heard = np.maximum.accumulate(np.where(heard_frames(row), np.arange(frames), -1))
    before = np.where(starts > 0, last_heard[np.maximum(starts - 1, 0)], -1)
    reply_due = np.maximum(np.maximum(starts - lead_frames, before + 1), 0)
    reply = np.clip(np.searchsorted(starts, onsets, side="right") - 1, 0, None)
    due = reply_due[reply] if len(starts) else np.zeros_like(onsets)
    owed, stops = _owed_yields(row), set(yield_frames(row).tolist())
    targets = np.full(frames, ids.sil, dtype=np.int64)
    inputs = targets.copy()
    kinds = np.full(frames, SIL, dtype=np.uint8)
    slot_frame = np.arange(frames, dtype=np.int64)
    playing = np.zeros(frames, dtype=bool)
    queue: list[tuple[int, int, int]] = []
    next_token, writing, window = 0, False, (frames, -1)
    for t in range(frames):
        while next_token < onsets.shape[0] and due[next_token] <= t:
            queue.append((int(row.text_tokens[next_token]), int(onsets[next_token]), next_token))
            next_token += 1
        playing[t] = window[0] <= t <= window[1]
        word = queue.pop(0) if queue and t not in stops else None
        if word is not None and (word[2] == 0 or reply[word[2]] != reply[word[2] - 1]):
            window = (t + 1, int(ends[reply[word[2]]]))
        if owed[t] and playing[t]:
            targets[t], kinds[t] = ids.sil, STOP
            missed = ids.sil if t in stops else ids.eps
            inputs[t] = word[0] if word is not None else missed
        elif word is not None:
            targets[t] = inputs[t] = word[0]
            kinds[t], slot_frame[t] = TEXT, word[1]
        else:
            targets[t] = inputs[t] = ids.eps if playing[t] else ids.sil
            kinds[t] = END if writing else (HOLD if playing[t] else SIL)
        if t in stops:
            queue, window = [], (frames, -1)
        writing = word is not None
    return HeldSchedule(targets, inputs, kinds, slot_frame, playing, len(queue))


def plan_sequence(row: Row, *, prefix_len: int, ids: TokenIds, layout: Layout) -> SequencePlan:
    """Each frame's positions end with the assistant position, whose input token is the PREVIOUS
    frame's text/control token (SIL start state; previous codes are added by build_embeds); its
    hidden predicts this frame's token (text_pos = a + 1 so the shared "predicted at text_pos - 1"
    convention reads it) and this frame's codes (anchor_pos = a)."""
    if row.role != "assistant" or row.question_pcm is not None:
        raise ValueError(
            f"row {row.index} is a {row.role!r} row; the frame layouts render assistant rows without question audio"
        )
    per = tokens_per_frame(layout)
    if per == 1 and np.any(row.user_pcm):
        raise ValueError(f"row {row.index} has user audio, which the speech layout has no room for")
    frames = row.frames
    playing: np.ndarray | None = None
    match layout:
        case "talker-floor":
            held = held_schedule(row, lead_frames=WRITE_LEAD_FRAMES, ids=ids)
            y, kinds, slot_frame, dropped = held.targets, held.kinds, held.slot_frame, held.dropped
            read, playing = held.inputs, held.playing
        case "talker":
            y, kinds, slot_frame, dropped = written_ahead_schedule(row, lead_frames=WRITE_LEAD_FRAMES, ids=ids)
            read, playing = y, played_frames(row)
        case "L" | "speech-gapped" | "speech-contiguous":
            y, kinds, slot_frame, dropped = text_schedule(row, lead_frames=TEXT_LEAD_FRAMES, ids=ids)
            read = y
        case unreachable:
            assert_never(unreachable)
    frame_tokens = np.full((frames, per), ids.user_audio, dtype=np.int64)
    frame_tokens[:, -1] = np.concatenate([[ids.sil], read])[:frames]
    index = prefix_len + 1 + per * np.arange(frames)[:, None] + np.arange(per)
    anchor_pos = index[:, -1]
    numbers = frame_positions(layout, prefix_len=prefix_len, frames=np.arange(frames))
    return SequencePlan(
        prefix_len=prefix_len,
        live_ids=np.concatenate([[ids.channel_live], frame_tokens.reshape(-1)]),
        user_pos=index[:, :-1].reshape(-1),
        anchor_pos=anchor_pos,
        text_pos=anchor_pos + 1,
        text_targets=y,
        text_kind=kinds,
        text_frame=slot_frame,
        position_ids=np.concatenate([np.arange(prefix_len + 1), numbers.reshape(-1)]),
        dropped_text=dropped,
        playing=playing,
    )
