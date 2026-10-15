# SPDX-License-Identifier: AGPL-3.0-only
"""Interval coverage, including leading/trailing holes and overlapping inputs."""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable

Interval = tuple[float, float]


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping/touching intervals; reject non-finite timestamps."""
    ordered = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Interval timestamps must be finite")
        if end > start:
            ordered.append((start, end))
    merged: list[Interval] = []
    for start, end in sorted(ordered):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def clip_intervals(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    """Intersect intervals with one evaluation window and union the result."""
    return union_intervals(
        (max(start, window[0]), min(end, window[1]))
        for start, end in union_intervals(intervals)
    )


def interval_seconds(intervals: Iterable[Interval]) -> float:
    """Measure interval union, so duplicates cannot inflate coverage."""
    return sum(end - start for start, end in union_intervals(intervals))


def subtract_intervals(
    reference: Iterable[Interval], hypothesis: Iterable[Interval]
) -> list[Interval]:
    """Return reference minus hypothesis in linear time after sorting."""
    covered = union_intervals(hypothesis)
    holes: list[Interval] = []
    index = 0
    for start, end in union_intervals(reference):
        cursor = start
        while index < len(covered) and covered[index][1] <= cursor:
            index += 1
        current = index
        while current < len(covered) and covered[current][0] < end:
            left, right = covered[current]
            if left > cursor:
                holes.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
            current += 1
        if cursor < end:
            holes.append((cursor, end))
        index = current
    return holes


def _long_gaps(gaps: list[Interval], holes: list[Interval]) -> list[dict]:
    """Sum missing speech within disjoint transcript gaps with a linear sweep."""
    runs = []
    index = 0
    for start, end in gaps:
        seconds = 0.0
        while index < len(holes) and holes[index][0] < end:
            left, right = holes[index]
            seconds += max(0.0, min(right, end) - max(left, start))
            index += 1
        if seconds >= 5.0:
            runs.append({"start": start, "end": end, "vad_s": seconds})
    return runs


def vad_coverage(
    segments: list[dict], vad_intervals: list[Interval], duration_s: float
) -> dict:
    """Measure all missing VAD speech, including gaps at both recording edges.

    Long runs are transcript-free gaps containing at least five seconds of VAD
    speech, preserving the original p9 diagnostic across short VAD silences.
    """
    window = (0.0, duration_s)
    vad = clip_intervals(vad_intervals, window)
    speech = clip_intervals(
        ((s["start"], s["end"]) for s in segments if str(s.get("text") or "").strip()),
        window,
    )
    holes = subtract_intervals(vad, speech)
    total = interval_seconds(vad)
    missing = interval_seconds(holes)
    long = _long_gaps(subtract_intervals([window], speech), holes)
    return {
        "vad_s": total,
        "hole_s": missing,
        "hole_percent": 100 * missing / total if total else None,
        "reason": None if total else "No VAD speech in the evaluation window",
        "holes": holes,
        "runs_5s": long,
        "runs_5s_count": len(long),
        "runs_5s_s": sum(run["vad_s"] for run in long),
    }


def word_coverage(words: list[Interval], hypothesis: list[Interval]) -> dict:
    """Count words with no positive overlap, with no arbitrary look-back limit.

    Words contain timestamps only: this helper never returns transcript excerpts.
    Uncovered-word runs join across gaps strictly below two seconds.
    """
    covered = union_intervals(hypothesis)
    starts = [start for start, _ in covered]
    valid = sorted((start, end) for start, end in words if end > start)
    uncovered = []
    for start, end in valid:
        index = bisect.bisect_left(starts, end) - 1
        if index < 0 or covered[index][1] <= start:
            uncovered.append((start, end))
    runs: list[Interval] = []
    for start, end in uncovered:
        if runs and start - runs[-1][1] < 2.0:
            runs[-1] = (runs[-1][0], max(end, runs[-1][1]))
        else:
            runs.append((start, end))
    total = interval_seconds(valid)
    missing = interval_seconds(uncovered)
    return {
        "words": len(valid),
        "uncovered_words": len(uncovered),
        "reference_s": total,
        "uncovered_s": missing,
        "uncovered_percent": 100 * missing / total if total else None,
        "runs_5s": [run for run in runs if run[1] - run[0] >= 5.0],
    }
