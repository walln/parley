"""CSM-1B as the talker, frame by frame. A prefill reads the voice clip, any earlier speech and the
words to say into a static cache. Each 80 ms frame then samples codebook 0 from the backbone and
the frame's other codebooks from the depth decoder, and feeds the frame back to the backbone. On
CUDA both halves replay as graphs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import CsmForConditionalGeneration, ProcessorMixin
from transformers.cache_utils import Cache, StaticLayer

from parley.infer.generate import graphed
from parley.model.csm_depth import CAUSAL_ATTENTION


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_k: int


#: Sesame's own generator: every codebook at temperature 0.9 from its 50 likeliest codes.
SESAME = Sampling(temperature=0.9, top_k=50)
GREEDY = Sampling(temperature=1.0, top_k=1)
POSITIONS = 2048
PREFILL_BUCKET = 64


@dataclass(frozen=True)
class Segment:
    """Something said, as context: its words, its audio's length and its codes."""

    text: str
    samples: int
    codes: torch.Tensor  # [frames, codebooks]


def sampled(logits: torch.Tensor, noise: torch.Tensor, sampling: Sampling) -> torch.Tensor:
    """[N] codes from [N, V] logits, by Gumbel-max over the `top_k` likeliest with `noise`
    [N, top_k] uniform in (0, 1), so no random state lives inside a graph."""
    values, indices = logits.float().topk(sampling.top_k, dim=-1)
    scores = values / sampling.temperature - torch.log(-torch.log(noise))
    return indices.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)


class Talker:
    """Speaks with the first `books` codebooks of each frame, feeding back only those."""

    def __init__(
        self,
        model: CsmForConditionalGeneration,
        *,
        books: int,
        positions: int,
        sampling: Sampling,
    ):
        codebooks = model.config.num_codebooks
        if codebooks is None or not 1 <= books <= codebooks:
            raise ValueError(f"CSM-1B renders 1 to {codebooks} codebooks, not {books}")
        self.model, self.books, self.sampling = model, books, sampling
        self.backbone, self.depth = model.backbone_model, model.depth_decoder
        self.depth.config._attn_implementation = CAUSAL_ATTENTION
        self.device = model.lm_head.weight.device
        self.cache = Cache(layers=[StaticLayer(max_cache_len=positions) for _ in self.backbone.layers])
        self.positions = positions
        self._codes = graphed(self._frame_codes, self.device)
        self._step = graphed(self._backbone_step, self.device)

    def embedded(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """[1, S, H] embeddings of a processor's text and audio inputs, the audio encoded by
        CSM-1B's own codec."""
        return self.model._merge_input_ids_with_input_values(
            inputs["input_ids"], inputs.get("input_values"), inputs.get("input_values_cutoffs")
        )["inputs_embeds"]

    def segment(self, text: str, wave: np.ndarray) -> Segment:
        """`wave` as `embedded` would encode it in a prompt, once."""
        codec = self.model.codec_model
        audio = torch.from_numpy(wave).to(self.device, next(codec.parameters()).dtype)
        codes = codec.encode(audio[None, None], return_dict=True).audio_codes
        frames = int(codec.get_encoded_length(torch.tensor([len(wave)], device=self.device))[0])
        return Segment(text, len(wave), codes[0, :, :frames].T.contiguous())

    def embedded_segments(self, input_ids: torch.Tensor, segments: Sequence[Segment]) -> torch.Tensor:
        """What `embedded` makes of the same prompt, from each audio segment's codes, encoded once,
        in the order its placeholders come."""
        config = self.model.config
        embeds = self.model.embed_text_tokens(input_ids)
        audio = input_ids == config.audio_token_id
        codes = torch.cat([segment.codes for segment in segments])
        if int(audio.sum()) != codes.shape[0]:
            raise ValueError(f"{int(audio.sum())} audio positions for {codes.shape[0]} frames")
        embeds[audio] = self.backbone.embed_tokens(codes[None])[0].to(embeds.dtype)
        end_code = config.codebook_eos_token_id
        if not isinstance(end_code, int):
            raise ValueError(f"CSM-1B's end-of-audio code must be one integer, not {end_code!r}")
        eos = torch.full((1, 1, codes.shape[1]), end_code, device=self.device)
        ends = input_ids == config.audio_eos_token_id
        embeds[ends] = self.backbone.embed_tokens(eos)[0].to(embeds.dtype)
        return embeds

    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """The backbone's hidden state [1, H] at the end of `inputs_embeds` [1, S, H], from an empty
        cache, copied out so its strides don't depend on S: the graphs would recompile for each.
        The prompt is padded on the right to a multiple of PREFILL_BUCKET, since each new length
        costs eager kernels a first call. A static layer writes at its own count of what it holds,
        so the count goes back to the prompt's length, and each frame overwrites a padding slot
        before the causal mask lets it be read."""
        length = inputs_embeds.shape[1]
        padded = -(-length // PREFILL_BUCKET) * PREFILL_BUCKET
        if padded >= self.positions:
            raise ValueError(f"a prefill of {length} positions leaves no room in {self.positions}")
        self.cache.reset()
        cache_position = torch.arange(padded, device=self.device)
        hidden = (
            self.backbone(
                inputs_embeds=torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padded - length)),
                past_key_values=self.cache,
                cache_position=cache_position,
                position_ids=cache_position[None],
                use_cache=True,
            )
            .last_hidden_state[:, length - 1]
            .clone(memory_format=torch.contiguous_format)
        )
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(length)
        return hidden

    def warm_prefills(self) -> None:
        """Every padded prefill length once, so none pays its first call while someone listens."""
        size = self.backbone.config.hidden_size
        for length in range(PREFILL_BUCKET, self.positions, PREFILL_BUCKET):
            embeds = torch.zeros((1, length, size), device=self.device, dtype=self.model.dtype)
            self.prefill(embeds)

    def _frame_codes(self, hidden: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        codes = sampled(self.model.lm_head(hidden), noise[:1], self.sampling)[:, None]
        for book in range(1, self.books):
            ids = torch.cat([torch.zeros_like(codes[:, :1]), codes], dim=1)
            last = self.depth.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                backbone_last_hidden_state=hidden,
                use_cache=False,
            ).last_hidden_state[:, -1]
            logits = last @ self.depth.codebooks_head.weight[book - 1]
            codes = torch.cat([codes, sampled(logits, noise[book : book + 1], self.sampling)[:, None]], 1)
        return codes

    def _backbone_step(self, codes: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        table = self.backbone.embed_tokens
        offsets = table.audio_tokens_offsets[: self.books]
        embeds = table.embed_audio_tokens(codes + offsets).sum(dim=1, keepdim=True)
        return self.backbone(
            inputs_embeds=embeds,
            past_key_values=self.cache,
            cache_position=cache_position,
            position_ids=cache_position[None],
            use_cache=True,
        ).last_hidden_state[:, -1]

    def frames(
        self, inputs_embeds: torch.Tensor, max_frames: int, generator: torch.Generator
    ) -> Iterator[torch.Tensor]:
        """Each frame's [1, books] codes as it is sampled, until the first frame whose codes are
        all end of speech, which is left out. Nothing runs, the prefill included, until the first
        frame is asked for."""
        hidden = self.prefill(inputs_embeds)
        start = inputs_embeds.shape[1]
        eos = self.model.config.codebook_eos_token_id
        for frame in range(min(max_frames, self.positions - start)):
            noise = torch.rand((self.books, self.sampling.top_k), generator=generator, device=self.device).clamp_(
                1e-6, 1 - 1e-6
            )
            codes = self._codes(hidden, noise)
            if bool((codes == eos).all()):
                return
            yield codes
            position = torch.tensor([start + frame], device=self.device)
            hidden = self._step(codes, position)


@dataclass(frozen=True)
class CsmSpeaker:
    """CSM-1B's prompts in the voice of `voice`: the processor lays out the words and the audio
    placeholders, which the talker fills from codes encoded once rather than at every chunk."""

    processor: ProcessorMixin
    talker: Talker
    voice: Segment

    def embeds(self, text: str, context: Sequence[Segment]) -> torch.Tensor:
        said = [self.voice, *context]
        turns = [*map(_turn, said), {"role": "0", "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(turns, tokenize=True, return_dict=True)
        return self.talker.embedded_segments(inputs["input_ids"].to(self.talker.device), said)

    def segment(self, text: str, wave: np.ndarray) -> Segment:
        return self.talker.segment(text, wave)


def _turn(segment: Segment) -> dict:
    return {
        "role": "0",
        "content": [
            {"type": "text", "text": segment.text},
            {"type": "audio", "path": np.zeros(segment.samples, np.float32)},
        ],
    }
