"""Gemma's reply text, as it is written, cut into the chunks the talker speaks. The opening chunk
ends at the first sentence end, or at the first clause end once OPENING_WORDS words are written, or
after LONGEST_OPENING_WORDS words. Each later chunk takes every complete sentence written so far,
and whatever is left when the reply ends is the last. When the talker is about to fall silent and
no sentence is complete, the text up to its last clause end is cut, and nothing if it has none:
CSM-1B restarts its rhythm at every chunk, so a short pause is better than a cut mid-phrase. The
corpus spaces its punctuation ("Germany ."), so a mark joins the word before it, and a word counts
only once the next word has begun: the cuts don't depend on where the tokens split the text."""

from __future__ import annotations

import re
from dataclasses import dataclass

OPENING_WORDS = 8
LONGEST_OPENING_WORDS = 16
#: CSM-1B garbles the words around typographic quotes and apostrophes, quotes aren't spoken, and
#: it can fall silent for seconds at a parenthesis, so a parenthesis is read as a comma.
_PLAIN = str.maketrans({"’": "'", "‘": "'", "“": None, "”": None, '"': None, "(": ",", ")": ","})
_MARKS = ",;:.!?"
_LOOSE_MARK = re.compile(rf"\s+(?=[{_MARKS}])|,(?=\s*[{_MARKS}])")
_WHOLE_WORD = re.compile(rf"(?<!\S)\S+(?=\s+[^\s{_MARKS}])")
_CLAUSE_END = re.compile(rf"[{_MARKS}]$")
_SENTENCE_MARK = re.compile(r"[.!?]$")
_SENTENCE_END = re.compile(r"[.!?]\s+(?=\S)|\n+")


@dataclass(frozen=True)
class Piece:
    text: str
    #: The first chunk of a reply.
    opens: bool


def _opening_cut(text: str) -> int | None:
    words = list(_WHOLE_WORD.finditer(text))[:LONGEST_OPENING_WORDS]
    ends = (
        word.end()
        for n, word in enumerate(words, 1)
        if _SENTENCE_MARK.search(word.group()) or n >= OPENING_WORDS and _CLAUSE_END.search(word.group())
    )
    return next(ends, words[-1].end() if len(words) == LONGEST_OPENING_WORDS else None)


def _clause_cut(text: str) -> int | None:
    ends = [word.end() for word in _WHOLE_WORD.finditer(text) if _CLAUSE_END.search(word.group())]
    return ends[-1] if ends else None


@dataclass
class Chunker:
    pending: str = ""
    started: bool = False

    def feed(self, text: str, *, hungry: bool = False) -> list[Piece]:
        """`hungry`: the talker is about to fall silent, so a reply that has started is cut at its
        last clause end unless a sentence is complete."""
        self.pending = _LOOSE_MARK.sub("", self.pending + text.translate(_PLAIN))
        pieces = [Piece(chunk, False) for chunk in self._sentences()] if self.started else []
        early = _clause_cut if self.started else _opening_cut
        if (not self.started or hungry and not pieces) and (cut := early(self.pending)) is not None:
            pieces.append(Piece(self._take(cut), not self.started))
            self.started = True
            pieces += [Piece(chunk, False) for chunk in self._sentences()]
        return [piece for piece in pieces if piece.text]

    def _sentences(self) -> list[str]:
        ends = [end.end() for end in _SENTENCE_END.finditer(self.pending)]
        return [self._take(ends[-1])] if ends else []

    def end(self) -> list[Piece]:
        opens, rest = not self.started, self._take(len(self.pending))
        self.started = False
        return [Piece(rest, opens)] if rest else []

    def _take(self, cut: int) -> str:
        taken, self.pending = self.pending[:cut], self.pending[cut:]
        return " ".join(taken.split())


def chunked(text: str) -> list[str]:
    chunker = Chunker()
    return [piece.text for piece in chunker.feed(text) + chunker.end()]
