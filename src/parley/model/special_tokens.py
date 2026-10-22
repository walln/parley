"""Special tokens for the full-duplex live channel protocol: <|channel>live (streaming duplex
mode), EPS (hold floor / continue speaking), SIL (listen quietly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from parley.model.vocab import set_real_vocab_size, validate_pad_vocab_to_multiple


@dataclass
class DuplexTokens:
    CHANNEL_LIVE: str = "<|channel>live"
    EPS: str = "<|eps|>"
    SIL: str = "<|sil|>"

    USER_AUDIO_SLOT: str = "<|user_audio|>"
    ASSISTANT_SLOT: str = "<|assistant|>"


DUPLEX_TOKENS = DuplexTokens()

DEFAULT_DUPLEX_TOKEN_INITIALIZERS = {
    "CHANNEL_LIVE": ("live", "stream", "channel"),
    "EPS": ("<unused0>", "continue", ","),
    "SIL": ("<unused1>", "silence", "<eos>"),
    "USER_AUDIO_SLOT": ("audio", "sound", "voice"),
    "ASSISTANT_SLOT": ("assistant", "model"),
}


def _tokenizer_single_id(tokenizer: Any, token: str) -> int | None:
    """Return a token id only when `token` is already represented by one id."""
    ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        return None
    token_id = int(ids[0])
    if token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return token_id


def _copy_embedding_row(embedding: Any, target_id: int, source_id: int) -> None:
    if embedding is None or target_id == source_id:
        return
    with torch.no_grad():
        embedding.weight[target_id].copy_(embedding.weight[source_id])


def initialize_duplex_special_token_embeddings(
    processor: Any,
    model: Any,
    token_ids: dict[str, int],
) -> dict[str, Any]:
    """
    Initialize newly-added duplex token rows from stable existing token rows.

    Fresh rows would stay random under LoRA (base embedding and LM-head are frozen);
    copying a single-token donor gives the text head a deterministic starting point.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    report: dict[str, Any] = {}

    for name, target_id in token_ids.items():
        candidates = tuple(DEFAULT_DUPLEX_TOKEN_INITIALIZERS.get(name, ()))
        source_token = None
        source_id = None
        for candidate in candidates:
            candidate_id = _tokenizer_single_id(tokenizer, candidate)
            if candidate_id is None or candidate_id == int(target_id):
                continue
            if candidate_id >= input_embeddings.weight.shape[0]:
                continue
            source_token = candidate
            source_id = candidate_id
            break
        if source_id is None:
            report[name] = {
                "initialized": False,
                "target_id": int(target_id),
                "source_token": None,
                "source_id": None,
            }
            continue

        # The caller decides whether the rows are newly created and safe to seed.
        _copy_embedding_row(input_embeddings, int(target_id), int(source_id))
        if output_embeddings is not None and int(source_id) < output_embeddings.weight.shape[0]:
            _copy_embedding_row(output_embeddings, int(target_id), int(source_id))
        report[name] = {
            "initialized": True,
            "target_id": int(target_id),
            "source_token": source_token,
            "source_id": int(source_id),
        }
    return report


def ensure_duplex_special_tokens(
    processor: Any,
    model: Any | None = None,
    *,
    pad_vocab_to_multiple: int = 0,
) -> dict[str, Any]:
    """
    Ensure live-channel control tokens are single tokenizer ids.

    If `model` is supplied, resize token embeddings whenever the tokenizer has ids
    beyond the current table — covers both freshly-added tokens and a checkpoint
    processor paired with a just-loaded base model. Returns ids + resize metadata.

    `pad_vocab_to_multiple` rounds the embedding table UP so the LM-head GEMM keeps a
    bf16-aligned leading dimension (262149 % 8 = 5 drops cuBLASLt onto a CUTLASS
    SM75 align1 kernel for both the head forward and its grad_input). Pad rows are
    trained-never-read; `parley.model.vocab.slice_real_vocab` keeps them out of every loss
    and sampler.
    """
    pad_vocab_to_multiple = validate_pad_vocab_to_multiple(pad_vocab_to_multiple)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokens = DUPLEX_TOKENS
    required = [
        tokens.CHANNEL_LIVE,
        tokens.EPS,
        tokens.SIL,
        tokens.USER_AUDIO_SLOT,
        tokens.ASSISTANT_SLOT,
    ]
    existing_additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged = list(dict.fromkeys(existing_additional + required))

    before_size = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": merged})
    after_size = len(tokenizer)
    model_embedding_size = None
    padded_vocab_size = None
    resized_model = False
    should_initialize = False
    if model is not None:
        input_embeddings = model.get_input_embeddings()
        model_embedding_size = int(input_embeddings.num_embeddings)
        if after_size != before_size or model_embedding_size < after_size:
            if pad_vocab_to_multiple:
                model.resize_token_embeddings(after_size, pad_to_multiple_of=pad_vocab_to_multiple)
            else:
                model.resize_token_embeddings(after_size)
            resized_model = True
            should_initialize = True
        padded_vocab_size = int(model.get_input_embeddings().num_embeddings)
        # Recorded unconditionally: resuming a padded checkpoint with the knob OFF must
        # still slice, or the pad columns silently enter the text CE.
        set_real_vocab_size(model, after_size)

    token_ids = {
        "CHANNEL_LIVE": int(tokenizer.convert_tokens_to_ids(tokens.CHANNEL_LIVE)),
        "EPS": int(tokenizer.convert_tokens_to_ids(tokens.EPS)),
        "SIL": int(tokenizer.convert_tokens_to_ids(tokens.SIL)),
        "USER_AUDIO_SLOT": int(tokenizer.convert_tokens_to_ids(tokens.USER_AUDIO_SLOT)),
        "ASSISTANT_SLOT": int(tokenizer.convert_tokens_to_ids(tokens.ASSISTANT_SLOT)),
    }
    for name, token_id in token_ids.items():
        if token_id < 0:
            raise ValueError(f"Failed to register duplex token {name}")
    for token in required:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Duplex token {token!r} is not a single token after registration: {ids}")

    embedding_initialization = {}
    if model is not None and should_initialize:
        embedding_initialization = initialize_duplex_special_token_embeddings(
            processor,
            model,
            token_ids,
        )

    return {
        "added": int(added),
        "before_size": int(before_size),
        "after_size": int(after_size),
        "model_embedding_size": model_embedding_size,
        "real_vocab_size": int(after_size),
        "padded_vocab_size": padded_vocab_size,
        "pad_vocab_to_multiple": int(pad_vocab_to_multiple),
        "resized_model": resized_model,
        "embedding_initialization": embedding_initialization,
        "token_ids": token_ids,
    }


def save_duplex_special_token_state(
    path: str | Path,
    processor: Any,
    model: Any,
    *,
    tokens: DuplexTokens = DUPLEX_TOKENS,
) -> dict[str, Any]:
    """Save the resized input/output embedding rows for duplex tokens."""
    tokenizer = getattr(processor, "tokenizer", processor)
    token_strings = {
        "CHANNEL_LIVE": tokens.CHANNEL_LIVE,
        "EPS": tokens.EPS,
        "SIL": tokens.SIL,
        "USER_AUDIO_SLOT": tokens.USER_AUDIO_SLOT,
        "ASSISTANT_SLOT": tokens.ASSISTANT_SLOT,
    }
    token_ids = {name: int(tokenizer.convert_tokens_to_ids(token)) for name, token in token_strings.items()}
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    state: dict[str, Any] = {
        "token_strings": token_strings,
        "token_ids": token_ids,
        "input_embeddings": {
            name: input_embeddings.weight[token_id].detach().cpu() for name, token_id in token_ids.items()
        },
        "output_embeddings": None,
    }
    if output_embeddings is not None:
        state["output_embeddings"] = {
            name: output_embeddings.weight[token_id].detach().cpu() for name, token_id in token_ids.items()
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def load_duplex_special_token_state(
    path: str | Path,
    processor: Any,
    model: Any,
) -> dict[str, Any]:
    """Restore saved duplex-token embedding rows into a resized model."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    with torch.no_grad():
        for name, token in state["token_strings"].items():
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            if token_id < 0:
                raise ValueError(f"Token {token!r} from {path} is not registered")
            input_embeddings.weight[token_id].copy_(
                state["input_embeddings"][name].to(
                    device=input_embeddings.weight.device,
                    dtype=input_embeddings.weight.dtype,
                )
            )
            output_state = state.get("output_embeddings")
            if output_embeddings is not None and output_state is not None:
                output_embeddings.weight[token_id].copy_(
                    output_state[name].to(
                        device=output_embeddings.weight.device,
                        dtype=output_embeddings.weight.dtype,
                    )
                )
    return state
