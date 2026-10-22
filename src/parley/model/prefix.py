"""Chat-template prefix for a live session: the user turn carries the prompt and the assistant
turn is opened and cut right before its content. Tokenised once per distinct prompt, on the CPU."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SENTINEL = "<fdg_assistant_prefix_sentinel_6b7f2c9d>"


class PrefixBuilder:
    def __init__(self, processor) -> None:
        self._processor = processor
        tok = processor.tokenizer
        # The sentinel tokenises differently after a space than at the start of a turn; both
        # spellings are searched, and the cut removes the space with it.
        self._sentinels = tuple(
            tuple(int(t) for t in tok(text, add_special_tokens=False)["input_ids"])
            for text in (SENTINEL, " " + SENTINEL)
        )
        self._build = lru_cache(maxsize=4096)(self._build_uncached)

    def __call__(self, prompt: str) -> np.ndarray:
        return self._build(prompt)

    def _build_uncached(self, prompt: str) -> np.ndarray:
        ids = self._processor.apply_chat_template(
            [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": SENTINEL}]},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="np",
            add_generation_prompt=False,
        )["input_ids"][0].astype(np.int64)
        ids = ids.tolist()
        for needle in self._sentinels:
            cut = _find(ids, needle)
            if cut is not None:
                return np.asarray(ids[:cut], dtype=np.int64)
        raise ValueError("sentinel not found in the rendered chat template")


def _find(haystack: list[int], needle: tuple[int, ...]) -> int | None:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if tuple(haystack[i : i + n]) == needle:
            return i
    return None
