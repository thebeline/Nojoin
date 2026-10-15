# SPDX-License-Identifier: AGPL-3.0-only
"""Raw-token evidence from recorded recognize() calls, and the strict reference-speech-loss metric.

A recognize() call returns text plus parallel subword tokens and token start times. A word starts at its first
subtoken; its support runs to its last subtoken + 0.2 s, never past the call's end or the next word's start. The
strict metric counts reference words (from an independent word-timed transcript) that no support touches, after
widening every support by +/-0.1 s. It reads token times only: no text leaves this module.

The functions below are the frozen scorer's, unchanged apart from imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import coverage

Interval = tuple[float, float]


class ScoringError(ValueError):
    """Evidence is malformed or differs from its declared provenance."""


@dataclass(frozen=True)
class Word:
    start: float
    last: float
    call_end: float
    text: str


@dataclass(frozen=True)
class Evidence:
    words: list[Word]
    untimed_calls: int
    initial_calls: int
    recovery_calls: int


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScoringError(f"{label} must be a finite number")
    return float(value)


def integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ScoringError(f"{label} must be an integer >= {minimum}")
    return value


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _call_words(call: dict[str, Any], start: float, end: float) -> tuple[list[Word], bool]:
    text, tokens, timestamps = call.get("text"), call.get("tokens"), call.get("timestamps")
    if not isinstance(text, str) or (tokens is not None and not isinstance(tokens, list)):
        raise ScoringError("Recognition text/tokens have invalid types")
    if tokens is not None and any(not isinstance(token, str) for token in tokens):
        raise ScoringError("Raw tokens must be strings")
    if timestamps is None:
        return [], bool(text.strip())
    if not isinstance(timestamps, list) or tokens is None or len(tokens) != len(timestamps):
        raise ScoringError("Raw token/timestamp arrays differ in length or type")
    if not tokens:
        return [], bool(text.strip())
    words: list[Word] = []
    previous = -1.0
    for index, (token, stamp) in enumerate(zip(tokens, timestamps, strict=True)):
        stamp = finite(stamp, "Raw timestamp")
        if stamp < previous or not 0 <= stamp < end - start:
            raise ScoringError("Raw timestamp is unordered or outside its call")
        absolute = start + stamp
        if index == 0 or token.startswith(" "):
            words.append(Word(absolute, absolute, end, token))
        else:
            old = words[-1]
            words[-1] = Word(old.start, absolute, end, old.text + token)
        previous = stamp
    return words, False


def _call_bounds(call: object, index: int, sample_rate: int, bounds: tuple[int, int]) -> tuple[int, int, int]:
    if not isinstance(call, dict) or call.get("status") == "failed" or call.get("error") is not None:
        raise ScoringError("Failed or malformed recognition call")
    if call.get("status") not in {None, "complete"}:
        raise ScoringError("Unexpected recognition call status")
    if integer(call.get("index"), "Call index") != index:
        raise ScoringError("Call indices must be contiguous invocation order")
    if integer(call.get("sample_rate"), "Call sample rate", 1) != sample_rate:
        raise ScoringError("Ambiguous source clock: call sample rate differs")
    start = integer(call.get("start_sample"), "Call start")
    end = integer(call.get("end_sample"), "Call end", 1)
    parent = integer(call.get("initial_index"), "Parent index")
    if not bounds[0] <= start < end <= bounds[1]:
        raise ScoringError("Call bounds lie outside declared slice")
    return start, end, parent


def call_evidence(calls: object, sample_rate: int, bounds: tuple[int, int]) -> Evidence:
    """Validate call ancestry/clock and shift raw timestamps exactly once."""
    if not isinstance(calls, list) or not calls:
        raise ScoringError("Calls must be a nonempty array")
    parents: dict[int, tuple[int, int]] = {}
    words: list[Word] = []
    untimed = recovery = 0
    cursor = bounds[0]
    for index, call in enumerate(calls):
        start, end, parent = _call_bounds(call, index, sample_rate, bounds)
        if call.get("kind") == "initial":
            if parent != index or start != cursor:
                raise ScoringError("Initial calls must tile the slice in time order")
            parents[index] = (start, end)
            cursor = end
        elif call.get("kind") == "recovery":
            if parent not in parents or not parents[parent][0] <= start < end <= parents[parent][1]:
                raise ScoringError("Recovery call must lie within a preceding initial call")
            recovery += 1
        else:
            raise ScoringError("Unknown recognition call kind")
        current, missing = _call_words(call, start / sample_rate, end / sample_rate)
        words.extend(current)
        untimed += int(missing)
    if cursor != bounds[1]:
        raise ScoringError("Initial calls do not cover the declared slice")
    return Evidence(sorted(words, key=lambda word: word.start), untimed, len(parents), recovery)


def word_supports(words: list[Word]) -> list[Interval]:
    """Keep last-subtoken support; globally clip duplicates/retries at next start."""
    ordered = sorted(words, key=lambda word: word.start)
    supports: list[Interval] = []
    for index, word in enumerate(ordered):
        end = min(word.last + 0.2, word.call_end)
        if index + 1 < len(ordered):
            end = min(end, ordered[index + 1].start)
        if end > word.start:
            supports.append((word.start, end))
    return supports


def coverage_metrics(reference: list[Interval], evidence: Evidence, window: Interval) -> dict[str, Any]:
    """Strict loss: reference words with no positive overlap with any +/-0.1 s raw-token support."""
    disclosure = {"zero_duration_reference_words": sum(start == end for start, end in reference)}
    if evidence.untimed_calls:
        return unavailable("Nonblank recognition calls lack raw token timing") | disclosure
    if not any(end > start for start, end in reference):
        return unavailable("No positive-duration reference words in this slice") | disclosure
    supports = word_supports(evidence.words)
    tolerance = [(max(window[0], start - 0.1), min(window[1], end + 0.1)) for start, end in supports]
    result = coverage.word_coverage(reference, tolerance)
    runs = result.pop("runs_5s")
    return {
        "status": "available",
        "method": "raw-token-support-union-v1",
        **result,
        **disclosure,
        "support_count": len(supports),
        "overlap_tolerance_s": 0.1,
        "runs_5s_count": len(runs),
        "runs_5s_elapsed_s": sum(end - start for start, end in runs),
        "max_run_s": max((end - start for start, end in runs), default=0.0),
    }


def jc_supports(evidence: Evidence, duration: float) -> list[Interval]:
    """Supports widened by the same +/-0.1 s, clipped to the recording: one run's half of 'both policies have text'."""
    widened = [(max(0.0, s - 0.1), min(duration, e + 0.1)) for s, e in word_supports(evidence.words)]
    return coverage.union_intervals(widened)
