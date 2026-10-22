"""Free-running generation, batched over sessions: per 80 ms frame one forward over the frame's
positions (in layout L its two user slices, then the assistant position), and the assistant
position picks the token and decodes the codes. A `Stream` advances sessions one frame at a time,
for offline runs and live serving alike. The graphed frame step runs each frame's backbone forward
and greedy depth decode as CUDA graphs over a session-long static cache."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, assert_never, get_args

import numpy as np
import torch
from transformers.cache_utils import Cache

from parley.codec.mimi import MimiCodec
from parley.infer.config import WordChoice
from parley.model.audio_feedback import AudioFeedbackEmbedding
from parley.model.backbone import GemmaBackbone
from parley.model.csm_depth import CsmDepthHead
from parley.model.depth_head import DepthHead, text_window
from parley.model.talker_state import TalkerState
from parley.rows.collate import SLICE_SAMPLES
from parley.rows.layout import (
    LAYOUT_L,
    USER_TOKENS_PER_FRAME,
    WRITE_LEAD_FRAMES,
    Layout,
    frame_positions,
    has_own_voice,
    tokens_per_frame,
)
from parley.rows.schema import SAMPLES_PER_FRAME, TokenIds
from parley.train.config import CODEBOOKS, load_depth_config, load_layout
from parley.train.stack import load_stack

FrameLap = Literal["backbone", "text", "depth"]
FRAME_LAPS: tuple[FrameLap, ...] = get_args(FrameLap)
Clock = Callable[[FrameLap, torch.Tensor], None]
InputObserver = Callable[[torch.Tensor, np.ndarray], None]
FrameStep = Literal["eager", "graphed"]
FRAME_STEPS: tuple[FrameStep, ...] = get_args(FrameStep)


def unclocked(_lap: FrameLap, _output: torch.Tensor) -> None:
    pass


def graphed(fn: Callable[..., torch.Tensor], device: torch.device) -> Callable[..., torch.Tensor]:
    """Each output is copied out of the graph's memory, which the next replay overwrites."""
    if device.type != "cuda":
        return fn
    compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=True, dynamic=False)

    def replay(*args, **kwargs) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        return compiled(*args, **kwargs).clone()

    return replay


#: How often, in frames, a call with `quiet_stop` checks whether every session is done.
QUIET_CHECK_FRAMES = 4


@dataclass(frozen=True)
class Session:
    prompt: str
    user_pcm: np.ndarray  # float32 in [-1, 1] at 16 kHz; padded up to whole frames here

    @property
    def frames(self) -> int:
        return -(-int(self.user_pcm.shape[0]) // SAMPLES_PER_FRAME)


class UserFeed(Protocol):
    def slices(self, frame: int, prev_token: torch.Tensor) -> torch.Tensor:
        """[B, 2, 640] user audio for `frame`, on the model's device, given each session's last
        chosen token."""
        ...

    def playing(self) -> torch.Tensor | None:
        """[B] bool: whether a talker played the frame before the one `slices` just gave; None
        when there is no talker."""
        ...


@dataclass(frozen=True)
class Heard:
    """What a frame hears: the user's [B, 2, 640] audio, and whether a talker played the frame
    before, [B] bool or None without a talker."""

    user_slices: torch.Tensor
    playing: torch.Tensor | None = None


@dataclass(frozen=True)
class FixedFeed:
    audio: torch.Tensor  # [B, frames * 2, 640]

    def slices(self, frame: int, prev_token: torch.Tensor) -> torch.Tensor:
        upf = USER_TOKENS_PER_FRAME
        return self.audio[:, upf * frame : upf * (frame + 1)]

    def playing(self) -> torch.Tensor | None:
        return None


#: Holding the floor, the stand-in talker takes this many frames to say each written token.
SPOKEN_FRAMES_PER_TOKEN = 4


@dataclass
class WrittenAheadFeed:
    """A fixed feed under a stand-in talker that plays as the talker layouts' rows do: from
    WRITE_LEAD_FRAMES after Gemma's first written word until it writes STOP, or holding the
    floor, from the frame after that word until it has said each token written, or until a SIL
    written while it plays."""

    fixed: FixedFeed
    ids: TokenIds
    held: bool = False
    _first: torch.Tensor | None = None
    _words: torch.Tensor | None = None
    _frame: int = 0
    _played: torch.Tensor | None = None

    def slices(self, frame: int, prev_token: torch.Tensor) -> torch.Tensor:
        token = prev_token.reshape(-1)
        first = torch.full_like(token, -1) if self._first is None else self._first
        words = torch.zeros_like(token) if self._words is None else self._words
        word = (token != self.ids.sil) & (token != self.ids.eps)
        first = torch.where((first < 0) & word, frame - 1, first)
        words = torch.where(first >= 0, words + word.long(), words)
        played = torch.zeros_like(word) if self._played is None else self._played
        if self.held:
            said = (first >= 0) & (frame > first + SPOKEN_FRAMES_PER_TOKEN * words)
            stop = ((token == self.ids.sil) & played) | said
        else:
            stop = token == self.ids.eps
        self._first = torch.where(stop, -1, first)
        self._words = torch.where(stop, 0, words)
        self._frame = frame
        return self.fixed.slices(frame, prev_token)

    def playing(self) -> torch.Tensor:
        if self._first is None:
            raise RuntimeError("playing is known only after the frame's slices")
        lead = 0 if self.held else WRITE_LEAD_FRAMES
        self._played = (self._first >= 0) & (self._frame > self._first + lead)
        return self._played


@dataclass
class Generation:
    frames: int
    tokens: np.ndarray  # int64 [F] the text/control token per frame (SIL = listening)
    codes: np.ndarray  # int64 [F, C]
    audio: np.ndarray  # float32 assistant channel, 16 kHz, F * 1280 samples
    text: str  # decoded text tokens in order

    def speaking(self, sil_id: int) -> np.ndarray:
        return self.tokens != sil_id


def choose_audio_code(
    logits: torch.Tensor, temperature: float, top_k: int, rng: torch.Generator | None
) -> torch.Tensor:
    """Argmax at temperature 0, else top-k (ties kept) temperature sampling on `rng`."""
    if not math.isfinite(temperature) or temperature < 0 or top_k < 1:
        raise ValueError("invalid audio sampling parameters")
    if temperature == 0:
        return logits.argmax(dim=-1)
    if rng is None:
        raise ValueError("audio sampling requires a session-local generator")
    scores = logits.float() / temperature
    threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
    scores = scores.masked_fill(scores < threshold, float("-inf"))
    return torch.multinomial(scores.softmax(-1), 1, generator=rng).squeeze(-1)


def control_twins(backbone: GemmaBackbone, ids: TokenIds) -> list[int]:
    """Vocab rows whose LM-head row equals a control token's row bit for bit. Control rows are
    initialised by copying a donor (EPS <- 'continue', SIL <- <eos>); until they train, the
    donor ties the control's logit exactly and argmax picks the lower id, so a listening model
    looks like it is saying '<eos>' every frame. Twins are masked out of the text choice."""
    real = backbone.head_weight[: backbone.real_vocab]
    twins: set[int] = set()
    with torch.no_grad():
        for control in (ids.eps, ids.sil):
            same = (real == real[control]).all(dim=1).nonzero().flatten().tolist()
            twins.update(int(i) for i in same if int(i) != control)
    return sorted(twins)


@dataclass
class Generator:
    backbone: GemmaBackbone
    #: A voice of the model's own; the talker layouts have neither.
    depth: DepthHead | None
    feedback: AudioFeedbackEmbedding | None
    mimi: MimiCodec
    processor: object
    token_ids: TokenIds
    temperature: float = 0.0  # 0 = greedy over the real vocab
    word_choice: WordChoice = "sampled"
    #: False reads the prompt with the adapters off, as base Gemma would; the live frames keep them.
    prompt_adapters: bool = True
    rng: torch.Generator | None = None
    audio_temperature: float = 0.0  # 0 = greedy on every codebook
    audio_top_k: int = 50
    audio_rng: torch.Generator | None = None
    layout: Layout = LAYOUT_L
    clock: Clock = unclocked
    frame_step: FrameStep = "eager"
    talker_state: TalkerState | None = None
    input_observer: InputObserver | None = None
    #: In the talker layout, STOP is written whenever its probability at `temperature` reaches
    #: this, instead of being drawn; the draw still happens, so the random stream is unchanged.
    #: Holding the floor, while the talker plays SIL is written once it reaches this, never drawn.
    stop_threshold: float | None = None

    @torch.inference_mode()
    def run(
        self,
        sessions: list[Session],
        prefixes: list[np.ndarray],
        *,
        forced_tokens: np.ndarray | None = None,
        quiet_stop: int | None = None,
    ) -> list[Generation]:
        """All sessions in one call share the prefix length (group by prompt upstream). With
        `quiet_stop`, the call ends once every session has written a word and then listened for
        that many frames in a row, and the frames left read as listening."""
        if not sessions:
            raise ValueError("no sessions")
        if tokens_per_frame(self.layout) == 1 and any(np.any(s.user_pcm) for s in sessions):
            raise ValueError("the speech layout has no user positions to hear the session's audio")
        b = len(sessions)
        frames = max(s.frames for s in sessions)
        forced = None
        if forced_tokens is not None:
            if forced_tokens.shape != (b, frames) or not np.issubdtype(forced_tokens.dtype, np.integer):
                raise ValueError("forced tokens must be an integer array [batch, frames]")
            if np.any((forced_tokens < 0) | (forced_tokens >= self.backbone.real_vocab)):
                raise ValueError("forced tokens are outside the real vocabulary")
            forced = torch.as_tensor(forced_tokens, device=self.backbone.device, dtype=torch.long)
        pcm = np.zeros((b, frames * SAMPLES_PER_FRAME), dtype=np.float32)
        for i, s in enumerate(sessions):
            pcm[i, : s.user_pcm.shape[0]] = s.user_pcm
        slices = torch.from_numpy(pcm.reshape(b, frames * USER_TOKENS_PER_FRAME, SLICE_SAMPLES))
        fixed = FixedFeed(slices.to(self.backbone.device))
        held = self.layout == "talker-floor"
        talker = WrittenAheadFeed(fixed, self.token_ids, held=held)
        feed = fixed if has_own_voice(self.layout) else talker
        tokens, codes = self._generate(prefixes, feed, frames, forced_tokens=forced, quiet_stop=quiet_stop)
        return [self._generation(tokens[i], codes[i], s.frames) for i, s in enumerate(sessions)]

    @torch.inference_mode()
    def run_fed(self, prefix: np.ndarray, feed: UserFeed, frames: int) -> Generation:
        """One session whose user audio `feed` chooses frame by frame, seeing the model's last
        token: a user who waits for the model before speaking again."""
        tokens, codes = self._generate([prefix], feed, frames)
        return self._generation(tokens[0], codes[0], frames)

    @torch.inference_mode()
    def open(self, prefixes: list[np.ndarray], frames: int) -> Stream:
        """Sessions prefilled with their prefixes, ready for at most `frames` steps."""
        if len({int(p.shape[0]) for p in prefixes}) != 1:
            raise ValueError("sessions in one batch must share the prefix length")
        b = len(prefixes)
        device = self.backbone.device
        prefix_len = int(prefixes[0].shape[0])
        prefix = torch.from_numpy(np.stack(prefixes)).to(device)
        live = torch.full((b, 1), self.token_ids.channel_live, device=device, dtype=torch.long)
        state = self._decode_state(b, device, prefix_len, frames)
        view = state.view
        embeds = view.embed_tokens(prefix)
        prefix_positions = np.arange(prefix_len)
        with nullcontext() if self.prompt_adapters else self.backbone.model.disable_adapter():
            self._observe_inputs(embeds, prefix_positions)
            state.forward(embeds, prefix_positions)
        live_embeds = view.embed_tokens(live)
        live_positions = np.array([prefix_len])
        self._observe_inputs(live_embeds, live_positions)
        state.forward(live_embeds, live_positions)
        start = torch.full((b,), self.token_ids.sil, device=device, dtype=torch.long)
        return Stream(self, state, frames, start)

    def _generate(
        self,
        prefixes: list[np.ndarray],
        feed: UserFeed,
        frames: int,
        *,
        forced_tokens: torch.Tensor | None = None,
        quiet_stop: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = self.open(prefixes, frames)
        device = self.backbone.device
        sil, eps = self.token_ids.sil, self.token_ids.eps
        tokens = torch.full((len(prefixes), frames), sil, device=device, dtype=torch.long)
        codes = torch.zeros((len(prefixes), frames, self.codebooks), device=device, dtype=torch.long)
        spoke = torch.zeros(len(prefixes), device=device, dtype=torch.bool)
        quiet = torch.zeros(len(prefixes), device=device, dtype=torch.long)
        for f in range(frames):
            forced = None if forced_tokens is None else forced_tokens[:, f]
            user = feed.slices(f, stream.prev_token)
            token, code = stream.step(user, playing=feed.playing(), forced_token=forced)
            tokens[:, f], codes[:, f] = token, code
            if quiet_stop is None:
                continue
            spoke |= (token != sil) & (token != eps)
            quiet = torch.where(token == sil, quiet + 1, 0)
            if f % QUIET_CHECK_FRAMES == 0 and bool((spoke & (quiet >= quiet_stop)).all()):
                break
        return tokens, codes

    def _generation(self, tokens: torch.Tensor, codes: torch.Tensor, frames: int) -> Generation:
        row_tokens = tokens[:frames].cpu().numpy()
        text_ids = [int(t) for t in row_tokens if t not in (self.token_ids.sil, self.token_ids.eps)]
        cds = codes[:frames].cpu().numpy()
        return Generation(
            frames=frames,
            tokens=row_tokens,
            codes=cds,
            audio=self.mimi.decode(torch.from_numpy(cds.T))[: frames * SAMPLES_PER_FRAME],
            text=self.processor.tokenizer.decode(text_ids, skip_special_tokens=True),
        )

    def _decode_state(self, batch: int, device: torch.device, prefix_len: int, frames: int) -> DecodeState:
        match self.frame_step:
            case "eager":
                return _DynamicDecode(self.backbone, batch, device, prefix_len)
            case "graphed":
                positions = prefix_len + 1 + frames * tokens_per_frame(self.layout)
                cache = self._static_cache(positions)
                return _StaticDecode(self.backbone, prefix_len, cache, self._backbone_frame)
            case unreachable:
                assert_never(unreachable)

    def _static_cache(self, positions: int) -> Cache:
        """One cache per length, zeroed for each decode: the graphs replay against its memory, and
        a new cache would record them again."""
        if (cache := self._caches.get(positions)) is not None:
            cache.reset()
            return cache
        return self._caches.setdefault(positions, self.backbone.full_static_cache(positions))

    def _observe_inputs(self, embeds: torch.Tensor, positions: np.ndarray) -> None:
        if self.input_observer is not None:
            self.input_observer(embeds, positions)

    def _reference_hidden(self, state, heard: Heard, prev_tok, prev_codes, frame, dtype):
        users = tokens_per_frame(self.layout) - 1
        user = state.view.embed_audio(heard.user_slices[:, :users]).to(dtype)
        assistant = state.view.embed_tokens(prev_tok)
        if prev_codes is not None and self.feedback is not None:
            assistant = assistant + self.feedback(prev_codes).to(dtype)
        if heard.playing is not None and self.talker_state is not None:
            assistant = assistant + self.talker_state(heard.playing).to(dtype)
        positions = frame_positions(self.layout, prefix_len=state.prefix_len, frames=np.array([frame]))[0]
        embeds = torch.cat([user, assistant.unsqueeze(1)], dim=1)
        self._observe_inputs(embeds, positions)
        return state.frame(embeds, positions)

    def reference_step(
        self,
        state: DecodeState,
        heard: Heard,
        prev_tok: torch.Tensor,
        prev_codes: torch.Tensor | None,
        frame: int,
        *,
        forced_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One 80 ms frame: the assistant input is the previous token's embedding plus the
        previous frame's code feedback (SIL start state, no feedback at frame 0), or in the talker
        layout the playing-or-idle input; the hidden at `a` chooses this frame's token and decodes
        its codes. A head with a text window also
        reads the state's embeddings of the tokens chosen so far, this frame's first. A layout
        without a voice of its own decodes no codes, and its frames carry zeros."""
        hidden = self._reference_hidden(state, heard, prev_tok, prev_codes, frame, state.view.rows.dtype)
        self.clock("backbone", hidden[:, -1])
        tok = (
            forced_token
            if forced_token is not None
            else self._choose_token(state.view.head(hidden[:, -1]), heard.playing)
        )
        self.clock("text", tok)
        if not has_own_voice(self.layout):
            return tok, tok.new_zeros((tok.shape[0], self.codebooks))
        state.text_history = self._text_history(state, tok, state.text_history)
        codes = self._frame_codes(hidden[:, -1], frame, state.text_history)
        self.clock("depth", codes)
        return tok, codes

    def _text_history(self, state: DecodeState, tok: torch.Tensor, history: torch.Tensor | None) -> torch.Tensor | None:
        window = text_window(self.head)
        if not window:
            return None
        chosen = state.view.embed_tokens(tok[:, None])
        if history is None:
            history = chosen.new_zeros((chosen.shape[0], window, chosen.shape[-1]))
        return torch.cat([chosen, history[:, :-1]], dim=1)

    @property
    def codebooks(self) -> int:
        """The codes each frame carries: the depth head's, or zeros in a talker layout's frames."""
        return CODEBOOKS if self.depth is None else self.depth.num_codebooks

    @property
    def head(self) -> DepthHead:
        if self.depth is None:
            raise ValueError("a talker layout has no depth head to decode codes")
        return self.depth

    def __post_init__(self) -> None:
        if self.audio_temperature and (self.audio_top_k < 1 or self.audio_temperature < 0):
            raise ValueError("audio sampling needs temperature >= 0 and top_k >= 1")
        if self.word_choice == "greedy" and self.temperature <= 0.0:
            raise ValueError("greedy words need a temperature to sample listening or speaking")
        if self.stop_threshold is not None and (has_own_voice(self.layout) or self.temperature <= 0.0):
            raise ValueError("a stop threshold needs the talker layout and a temperature")
        self._caches: dict[int, Cache] = {}
        if self.frame_step == "graphed":
            if self.audio_temperature:
                raise ValueError("the graphed frame step decodes audio greedily")
            device = self.backbone.device
            self._backbone_frame = graphed(self.backbone.cached_step, device)
            self._greedy_codes = graphed(self._decode_greedy, device)
        self._twins = control_twins(self.backbone, self.token_ids)
        device = self.backbone.device
        self._twin_columns = torch.tensor(self._twins, dtype=torch.long, device=device)
        ids = self.token_ids
        self._controls = torch.tensor([ids.sil, ids.eps], dtype=torch.long, device=device)
        self._duplex_columns = torch.tensor(sorted(set(ids.model_dump().values())), dtype=torch.long, device=device)
        if self._twins:
            print(f"[generate] masking control-token twins {self._twins}", flush=True)

    def set_sampling_seed(self, seed: int, *, audio_seed: int | None = None) -> None:
        """Separate text and audio streams: audio draws never shift the text sequence."""
        device = self.backbone.device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        audio = seed + 1 if audio_seed is None else audio_seed
        self.audio_rng = torch.Generator(device=device).manual_seed(audio)

    def _choose_token(self, logits: torch.Tensor, playing: torch.Tensor | None = None) -> torch.Tensor:
        real = logits[:, : self.backbone.real_vocab].float()
        if self._twins:
            real[:, self._twin_columns] = float("-inf")
        if self.temperature <= 0.0:
            return real.argmax(dim=-1)
        probs = torch.softmax(real / self.temperature, dim=-1)
        if self.layout == "talker-floor" and self.stop_threshold is not None:
            if playing is None:
                raise ValueError("holding the floor, each step must say whether the talker plays")
            return self._held_choice(real, probs, playing, self.stop_threshold)
        chosen = self._drawn(real, probs)
        if self.stop_threshold is None:
            return chosen
        eps = self.token_ids.eps
        return torch.where(probs[:, eps] >= self.stop_threshold, eps, chosen)

    def _held_choice(
        self, real: torch.Tensor, probs: torch.Tensor, playing: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """While the talker plays, SIL stops it: written once its probability reaches
        `threshold`, and otherwise left out of the draw."""
        sil = self.token_ids.sil
        owed = playing & (probs[:, sil] >= threshold)
        real[:, sil] = torch.where(playing, float("-inf"), real[:, sil])
        chosen = self._drawn(real, torch.softmax(real / self.temperature, dim=-1))
        return torch.where(owed, sil, chosen)

    def _drawn(self, real: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        match self.word_choice:
            case "sampled":
                return torch.multinomial(probs, 1, generator=self.rng).squeeze(1)
            case "greedy":
                return self._greedy_word(real, probs)
            case unreachable:
                assert_never(unreachable)

    def _greedy_word(self, real: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        controls = probs[:, self._controls]
        word = (1.0 - controls.sum(dim=-1, keepdim=True)).clamp_min(0.0)
        choice = torch.multinomial(torch.cat([controls, word], dim=-1), 1, generator=self.rng)
        best = real.index_fill(1, self._duplex_columns, float("-inf")).argmax(dim=-1)
        options = torch.cat([self._controls.expand(real.shape[0], -1), best[:, None]], dim=-1)
        return options.gather(1, choice).squeeze(1)

    def _frame_codes(
        self, anchor_hidden: torch.Tensor, frame: int, text_history: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Depth-decode the frame. Listening frames keep the head's own codes, and every frame
        draws audio, so the audio RNG stream does not depend on control."""
        positions = torch.full((anchor_hidden.shape[0],), frame, device=anchor_hidden.device)
        if self.frame_step == "graphed":
            return self._greedy_codes(anchor_hidden, positions, text_history)
        conditioning = self._conditioning(anchor_hidden, positions, text_history)
        return self.head.decode_codes(conditioning, choose=self._choose_audio)

    def _decode_greedy(
        self,
        anchor_hidden: torch.Tensor,
        positions: torch.Tensor,
        text_history: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.head.decode_codes(self._conditioning(anchor_hidden, positions, text_history))

    def _conditioning(
        self,
        anchor_hidden: torch.Tensor,
        positions: torch.Tensor,
        text_history: torch.Tensor | None,
    ) -> torch.Tensor:
        head = self.head
        hidden = anchor_hidden.to(head.in_proj.weight.dtype)
        if text_history is None:
            return head.project_conditioning(hidden, positions)
        if not isinstance(head, CsmDepthHead):
            raise ValueError("only the CSM head reads a text window")
        return head.project_conditioning(hidden, positions, text_embeds=text_history)

    def _choose_audio(self, logits: torch.Tensor) -> torch.Tensor:
        return choose_audio_code(logits, self.audio_temperature, self.audio_top_k, self.audio_rng)


@dataclass
class Stream:
    """Sessions advanced one 80 ms frame per `step`, each frame one forward over its positions
    (see `Generator.reference_step`)."""

    generator: Generator
    state: DecodeState
    frames: int
    prev_token: torch.Tensor
    prev_codes: torch.Tensor | None = None
    frame: int = 0

    def step(
        self,
        user_slices: torch.Tensor,
        *,
        playing: torch.Tensor | None = None,
        forced_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 2, 640] user audio for the next frame, and whether a talker played the frame
        before -> its tokens [B] and codes [B, C]."""
        if self.frame >= self.frames:
            raise RuntimeError(f"the stream was opened for {self.frames} frames")
        token, codes = self.generator.reference_step(
            self.state,
            Heard(user_slices, playing),
            self.prev_token,
            self.prev_codes,
            self.frame,
            forced_token=forced_token,
        )
        self.prev_token, self.prev_codes, self.frame = token, codes, self.frame + 1
        return token, codes


class _Decode:
    """The position bookkeeping of incremental forwards against one batch's KV cache."""

    def __init__(self, backbone: GemmaBackbone, prefix_len: int) -> None:
        self.backbone = backbone
        self.prefix_len = prefix_len
        self.length = 0
        self._view = None
        self.text_history: torch.Tensor | None = None

    @property
    def view(self):
        """The backbone's decode view, taken on first use: weights hold still within a decode."""
        if self._view is None:
            self._view = self.backbone.decode_view()
        return self._view

    def _slots(self, embeds: torch.Tensor, positions: np.ndarray) -> tuple[torch.Tensor, ...]:
        """The next n cache slots for `embeds` [B, n, H], and the position numbers `positions`
        [n] every session shares."""
        n, b = int(embeds.shape[1]), int(embeds.shape[0])
        if positions.shape != (n,):
            raise ValueError(f"{n} new positions need {n} position numbers, got {positions.shape}")
        slots = torch.arange(self.length, self.length + n, device=embeds.device)
        self.length += n
        return slots, torch.from_numpy(positions).to(embeds.device).expand(b, n)


class _DynamicDecode(_Decode):
    """A cache that grows by concatenation, beside a running attention mask."""

    def __init__(self, backbone: GemmaBackbone, batch: int, device: torch.device, prefix_len: int) -> None:
        super().__init__(backbone, prefix_len)
        self.mask = torch.zeros((batch, 0), device=device, dtype=torch.long)
        self.past = None

    def forward(self, embeds: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        ones = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        self.mask = torch.cat([self.mask, ones], dim=1)
        slots, position_ids = self._slots(embeds, positions)
        hidden, self.past = self.backbone.step(
            embeds,
            past=self.past,
            attention_mask=self.mask,
            cache_position=slots,
            position_ids=position_ids,
        )
        return hidden

    frame = forward


class _StaticDecode(_Decode):
    def __init__(
        self,
        backbone: GemmaBackbone,
        prefix_len: int,
        cache: Cache,
        frame_step: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__(backbone, prefix_len)
        self.cache = cache
        self._frame_step = frame_step

    def forward(self, embeds: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        return self._run(self.backbone.cached_step, embeds, positions)

    def frame(self, embeds: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        return self._run(self._frame_step, embeds, positions)

    def _run(self, step: Callable[..., torch.Tensor], embeds: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
        slots, position_ids = self._slots(embeds, positions)
        return step(embeds, cache=self.cache, cache_position=slots, position_ids=position_ids)


DecodeState = _DynamicDecode | _StaticDecode


def load_generator(
    checkpoint: Path,
    *,
    device: str,
    temperature: float = 0.0,
    audio_temperature: float = 0.0,
    audio_top_k: int = 50,
    merge_adapters: bool = False,
) -> Generator:
    stack = load_stack(
        load_depth_config(checkpoint),
        layout=load_layout(checkpoint),
        resume_dir=checkpoint,
        device=device,
        purpose="infer",
    )
    backbone = stack.backbone.merged() if merge_adapters else stack.backbone
    backbone.model.eval()
    for module in (stack.depth, stack.feedback):
        if module is not None:
            module.eval()
    codebooks = CODEBOOKS if stack.depth is None else stack.depth.num_codebooks
    return Generator(
        backbone=backbone,
        depth=stack.depth,
        feedback=stack.feedback,
        talker_state=stack.playing_input,
        mimi=MimiCodec(device=device, effective_codebooks=codebooks),
        processor=stack.processor,
        token_ids=stack.token_ids,
        temperature=temperature,
        audio_temperature=audio_temperature,
        audio_top_k=audio_top_k,
        layout=stack.layout,
    )
