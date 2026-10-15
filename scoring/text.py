# SPDX-License-Identifier: AGPL-3.0-only
"""Small text helpers shared by the scorers: 4-gram repetition and timed-word selection. No text is returned."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any


def repetition(texts: list[str], normalize: Callable[[str], str]) -> dict[str, Any]:
    """Count repeated four-grams without inventing grams across disjoint intervals."""
    counts: Counter[tuple[str, ...]] = Counter()
    word_count = 0
    for text in texts:
        words = normalize(text).split()
        word_count += len(words)
        counts.update(tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3)))
    total = sum(counts.values())
    excess = sum(max(count - 1, 0) for count in counts.values())
    return {
        "fourgrams": total,
        "excess_occurrences": excess,
        "excess_rate": excess / total if total else 0.0,
        "hypothesis_words": word_count,
    }


def identity(text: str) -> str:
    """For tokens that arrive already normalized."""
    return text


def final_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The product's timed words with positive duration, in start order."""
    words = [word for segment in result["segments"] for word in segment.get("words", []) if word["end"] > word["start"]]
    return sorted(words, key=lambda word: word["start"])


def within(rows: list[list[Any]], start: float, end: float) -> str:
    """Text of word-reference rows [start, end, word] fully inside [start, end] (the Whisper-proxy reference)."""
    return " ".join(row[2] for row in rows if start <= row[0] and row[1] <= end and row[1] > row[0])
