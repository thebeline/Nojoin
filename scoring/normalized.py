# SPDX-License-Identifier: AGPL-3.0-only
"""The two text lenses, applied identically to reference and hypothesis, and token provenance for lens (d).

(a) "frozen": wer.normalize_text, the scorer fixed before any result was seen: NFKC, casefold, apostrophe fold,
    punctuation and symbols to spaces. "uh" stays a word and "twenty" never equals "20".
(d) "normalized": Whisper's full EnglishTextNormalizer (numbers, British to American spelling, filler deletion,
    contractions, titles, bracketed text, diacritics), after the frozen lens's NFKC/apostrophe fold only, so its
    ASCII-apostrophe rules also see typographic apostrophes. It was made the primary lens after the frozen results
    were seen, so it is post-hoc.

provenance() maps each normalized token back to the timed hypothesis words it came from, so a normalized token can
be placed in time (the extra-words count needs that). Its output must equal the normalization of the whole text.
Source of the normalizer: whisper_normalizers/ (openai-whisper 20250625, MIT, byte-identical).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable

from .wer import normalize_text as frozen
from .whisper_normalizers.english import EnglishNumberNormalizer, EnglishTextNormalizer

NUM, FULL = EnglishNumberNormalizer(), EnglishTextNormalizer()
KEEP = ".%$¢€£"  # EnglishTextNormalizer's keep= set
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
FULL_FILLERS = ("hmm", "mm", "mhm", "mmm", "uh", "um")  # EnglishTextNormalizer.ignore_patterns
PERFECT = frozenset({"been", "gone", "done", "got"})  # its "'d been" etc. rewrites look across two words
NUMBERISH = frozenset(NUM.words) | {"a", "half", "st", "nd", "rd", "th", "s"}  # words its number rules can join
REPAIRS: Counter[str] = Counter()  # provenance segment merges, disclosed in the score


def fold(text: str) -> str:
    """The frozen lens's Unicode steps only: NFKC, casefold, typographic apostrophes to ASCII."""
    return unicodedata.normalize("NFKC", text).casefold().translate(APOSTROPHES)


def symbols_to_space(text: str, keep: str = "") -> str:
    """The frozen punctuation/symbol rule, sparing the characters in keep."""
    kept = (c if c == "'" or c in keep or not unicodedata.category(c).startswith(("P", "S")) else " " for c in text)
    return " ".join(re.sub(r"(?<!\w)'|'(?!\w)", " ", "".join(kept)).split())


def full(text: str) -> str:
    return " ".join(FULL(fold(text)).split())


LENSES: dict[str, Callable[[str], str]] = {"a": frozen, "d": full}
LABELS = {
    "a": "frozen: NFKC, casefold, punctuation to space (pre-registered)",
    "d": "normalized: Whisper EnglishTextNormalizer after the frozen fold (post-hoc, primary)",
}


def context_free(word: str, lens: str) -> bool:
    """True when the lens normalizes this word the same in any context, so a segment may end on either side."""
    if lens == "a":
        return True
    low = fold(word)
    pieces = (set(symbols_to_space(low, KEEP).split()) | set(re.split(r"[^\w]+", low))) - {""}
    if any(c.isdigit() or c in KEEP for c in low) or pieces & NUMBERISH:
        return False
    return not (any(c in "'()[]<>" for c in low) or pieces & (PERFECT | set(FULL_FILLERS)))


def _common(left: list[str], right: list[str]) -> int:
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return n


def _repair(
    words: list[str], segments: list[list[int]], norm: Callable[[str], str], lens: str
) -> tuple[list[list[int]], list[list[str]]]:
    """Merge a segment into its predecessor where the outputs first disagree with the whole text, until they agree."""
    outputs = [norm(" ".join(words[i] for i in segment)).split() for segment in segments]
    whole = norm(" ".join(words)).split()
    while True:
        flat = [t for out in outputs for t in out]
        if flat == whole:
            return segments, outputs
        if len(segments) == 1:
            raise ValueError(f"{lens}: one segment still differs from the whole text")
        at, seen, s = _common(flat, whole), 0, len(segments) - 1
        for k, out in enumerate(outputs):
            if seen + len(out) > at:
                s = k
                break
            seen += len(out)
        s = max(s, 1)
        segments[s - 1 : s + 1] = [segments[s - 1] + segments[s]]
        outputs[s - 1 : s + 1] = [norm(" ".join(words[i] for i in segments[s - 1])).split()]
        REPAIRS[lens] += 1


def _segments(words: list[str], lens: str) -> list[list[int]]:
    free = [context_free(w, lens) for w in words]
    segments: list[list[int]] = []
    for i, word in enumerate(words):
        glue = lens == "d" and word.lstrip().startswith(("'", "’", "‘"))  # the normalizer joins it leftwards
        if segments and ((not free[i] and not free[i - 1]) or glue):
            segments[-1].append(i)
        else:
            segments.append([i])
    return segments


def provenance(words: list[str], lens: str) -> tuple[list[str], list[tuple[int, int]]]:
    """The lens's tokens of ' '.join(words), each with its source word span (first, last), inclusive."""
    norm = LENSES[lens]
    segments, outputs = _repair(words, _segments(words, lens), norm, lens)
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for segment, final in zip(segments, outputs, strict=True):
        texts = [words[i] for i in segment]
        if len(segment) == 1:
            tokens += final
            spans += [(segment[0], segment[0])] * len(final)
            continue
        stable = [_common(norm(" ".join(texts[:k])).split(), final) for k in range(1, len(segment))] + [len(final)]
        for k in range(len(stable) - 2, -1, -1):
            stable[k] = min(stable[k], stable[k + 1])
        alone = [bool(norm(text).split()) for text in texts]  # a word that yields nothing alone never starts a span
        done = 0
        for j, token in enumerate(final):
            end = next(k for k, n in enumerate(stable, 1) if n > j)
            first = done if done < end else end - 1
            while first < end - 1 and not alone[first]:
                first += 1
            tokens.append(token)
            spans.append((segment[first], segment[end - 1]))
            done = end
    if tokens != norm(" ".join(words)).split():
        raise ValueError(f"{lens}: segment outputs differ from the whole text")
    return tokens, spans


def self_check() -> None:
    """Provenance on hand-built cases before any real number is computed."""
    tokens, spans = provenance(["uh", "I'd", "been", "there", "twenty", "uh", "five"], "d")
    assert tokens == ["i", "had", "been", "there", "25"], tokens
    assert spans == [(1, 1), (2, 2), (2, 2), (3, 3), (4, 6)], spans
    assert provenance(["Hello,", "world's", "end"], "a") == (["hello", "world's", "end"], [(0, 0), (1, 1), (2, 2)])
    REPAIRS.clear()
