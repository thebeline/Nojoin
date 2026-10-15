# SPDX-License-Identifier: AGPL-3.0-only
"""WER against an edited official record, using a declared time window.

This is the "frozen" lens: the scorer fixed before any result was seen.
Normalisation: Unicode NFKC, casefold, typographic apostrophes to ASCII,
punctuation/symbols to spaces (except in-word apostrophes), collapse whitespace.
Digits stay digits; contractions are preserved, not expanded. Thus '21' versus
'twenty one', or 'don't' versus 'do not', remain differences. No text is returned.
JiWER uses RapidFuzz's compiled edit-distance implementation for long recordings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

from .coverage import Interval

NORMALIZER = "nfkc-casefold-apostrophes-punctuation-space-v1"
LABEL = "vs the edited official record (not verbatim)"


def normalize_text(text: str) -> str:
    """Apply the same conservative normalizer to reference and hypothesis."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "ʼ": "'"}))
    text = "".join(
        char
        if char == "'" or not unicodedata.category(char).startswith(("P", "S"))
        else " "
        for char in text
    )
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> dict:
    """Return edit counts and WER; an empty normalized reference is unavailable."""
    from jiwer import process_words

    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        return {"wer": None, "reason": "Official reference has no normalized words"}
    scores = process_words(reference, hypothesis)
    return {
        "wer": scores.wer,
        "reason": None,
        "reference_words": scores.hits + scores.substitutions + scores.deletions,
        "hypothesis_words": scores.hits + scores.substitutions + scores.insertions,
        "hits": scores.hits,
        "substitutions": scores.substitutions,
        "deletions": scores.deletions,
        "insertions": scores.insertions,
    }


def _timed_words(segment: dict) -> list[dict] | None:
    words = segment.get("words")
    if not words or not all(
        isinstance(word, dict)
        and word.get("start") is not None
        and word.get("end") is not None
        for word in words
    ):
        return None
    return words


def hypothesis_text(segments: list[dict], window: Interval) -> tuple[str, dict]:
    """Use complete timed words when available, else complete segments only.

    Boundary-straddling units are excluded rather than assigning their full text
    to a partial interval. Counts disclose this conservative alignment limitation.
    """
    selected: list[tuple[float, int, str]] = []
    counts = {
        "boundary_excluded_words": 0,
        "boundary_excluded_segments": 0,
        "word_timed_segments": 0,
        "segment_timed_segments": 0,
    }
    for segment in sorted(segments, key=lambda item: float(item["start"])):
        words = _timed_words(segment)
        units = words if words is not None else [segment]
        kind = "words" if words is not None else "segments"
        counts[
            "word_timed_segments" if words is not None else "segment_timed_segments"
        ] += 1
        for unit in units:
            start, end = float(unit["start"]), float(unit["end"])
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("Cannot score WER with non-finite timestamps")
            if end <= start or end <= window[0] or start >= window[1]:
                continue
            if start >= window[0] and end <= window[1]:
                text = str(unit.get("word", unit.get("text")) or "")
                selected.append((start, len(selected), text))
            else:
                counts["boundary_excluded_" + kind] += 1
    return " ".join(text for _, _, text in sorted(selected)), counts


def score_wer(segments: list[dict], path: Path, duration_s: float) -> dict:
    """Load a prepared, hashed official reference without returning its content."""
    reference = json.loads(path.read_text(encoding="utf-8"))
    text = reference["text"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != reference["sha256"]:
        raise ValueError(f"WER reference text hash mismatch in {path.name}")
    window = tuple(float(value) for value in reference["window"])
    if len(window) != 2 or not all(math.isfinite(value) for value in window):
        raise ValueError(f"Invalid WER window in {path.name}")
    if not 0 <= window[0] < window[1] <= duration_s + 0.1:
        raise ValueError(f"WER window outside recording duration in {path.name}")
    hypothesis, counts = hypothesis_text(segments, (window[0], window[1]))
    return {
        **word_error_rate(text, hypothesis),
        **counts,
        **{
            key: reference[key]
            for key in ("window_policy", "scope_limitation", "contribution_orders")
            if key in reference
        },
        "label": LABEL,
        "source": reference["source"],
        "reference_sha256": digest,
        "window": window,
        "normalizer": NORMALIZER,
        "boundary_policy": "fully contained words, or fully contained segments without word timing",
    }
