# SPDX-License-Identifier: AGPL-3.0-only
"""Extra words: the post-hoc hallucination guard between a baseline run and a candidate run on one recording.

Definition (fixed before any corrected number was computed, and post-hoc relative to the original guard):
- jointly covered (JC) = time where both runs have text: the intersection of the two runs' raw-token support unions,
  each support widened by +/-0.1 s;
- a token sits at its first source word's start (a midpoint placement is reported beside it);
- VAD-silent = audio the VAD muted (complement of the nonzero 10 ms bins of the engine input);
- rate = |tokens the alignment marks inserted that sit inside JC  union  tokens that sit in VAD-silent audio|
  / reference tokens, in percentage points of reference words;
- a JC-restricted 4-gram repetition rate is kept beside it;
- the candidate passes when its rate is at most +1 pp above the baseline's and its JC repetition at most +1 pp.
Both lenses re-normalize the raw words and the reference and re-run the jiwer alignment.
Insertions are counted against the edited official record, so a verbatim filler the record omits counts too.
"""

from __future__ import annotations

import bisect
from typing import Any

from jiwer import process_words

from . import coverage as C
from . import wer
from .evidence import ScoringError, jc_supports
from .metrics import evidence_of, token_timed
from .normalized import LENSES, provenance
from .runs import Run, read, same_input
from .text import final_words, identity, repetition

LIMIT = 0.01


def intersect(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return C.subtract_intervals(left, C.subtract_intervals(left, right))


def locator(intervals: list[tuple[float, float]]) -> Any:
    """Return f(point) -> index of the disjoint, sorted half-open interval holding point, else None."""
    starts = [start for start, _ in intervals]

    def find(point: float) -> int | None:
        index = bisect.bisect_right(starts, point) - 1
        return index if index >= 0 and point < intervals[index][1] else None

    return find


def muted(run: Run) -> list[tuple[float, float]]:
    return C.subtract_intervals([(0.0, run.duration)], [tuple(x) for x in run.vad["intervals"]])


def side_of(run: Run, reference_text: str, window: tuple[float, float], lens: str) -> dict[str, Any]:
    """One run under one lens: provenance-tracked tokens, a fresh alignment, token placements and its JC support."""
    start, end = window
    words = [w for w in final_words(run.result) if start <= w["start"] and w["end"] <= end]
    text, _ = wer.hypothesis_text(run.result["segments"], window)
    tokens, spans = provenance([w["word"] for w in words], lens) if words else ([], [])
    if tokens != LENSES[lens](text).split():
        raise ValueError(f"{run.label} {lens}: timed word stream differs from the scored hypothesis text")
    reference = LENSES[lens](reference_text).split()
    chunks = process_words(" ".join(reference), " ".join(tokens)).alignments[0]
    return {
        "label": run.label,
        "tokens": tokens,
        "reference_words": len(reference),
        "inserted": {i for c in chunks if c.type == "insert" for i in range(c.hyp_start_idx, c.hyp_end_idx)},
        "at": {
            "start": [words[first]["start"] for first, _ in spans],
            "midpoint": [(words[first]["start"] + words[last]["end"]) / 2 for first, last in spans],
        },
        "support": jc_supports(evidence_of(run), run.duration),
    }


def measure(side: dict[str, Any], joint: list, silent: list, placement: str) -> dict[str, Any]:
    """Flagged-token rate and JC-restricted repetition for one run on one pair's JC region."""
    in_jc, in_muted = locator(joint), locator(silent)
    points = side["at"][placement]
    group = [in_jc(point) for point in points]
    in_silent = [in_muted(point) is not None for point in points]
    inserted = {i for i in side["inserted"] if group[i] is not None}
    flagged = inserted | {i for i, hit in enumerate(in_silent) if hit}
    texts: dict[Any, list[str]] = {}
    for i, key in enumerate(group):
        if key is not None:
            texts.setdefault(key, []).append(side["tokens"][i])
    repeated = repetition([" ".join(words) for words in texts.values()], identity)
    return {
        "inserted_in_jc": len(inserted),
        "vad_silent_words": sum(in_silent),
        "flagged_words": len(flagged),
        "rate": len(flagged) / side["reference_words"],
        "jc_hypothesis_words": sum(g is not None for g in group),
        "jc_repetition": repeated["excess_rate"],
    }


def pair(candidate: dict[str, Any], baseline: dict[str, Any], silent: list) -> dict[str, Any]:
    joint = intersect(candidate["support"], baseline["support"])
    row: dict[str, Any] = {
        "candidate": candidate["label"],
        "baseline": baseline["label"],
        "jc_s": C.interval_seconds(joint),
        "reference_words": candidate["reference_words"],
    }
    for placement in ("start", "midpoint"):
        mine, theirs = measure(candidate, joint, silent, placement), measure(baseline, joint, silent, placement)
        delta, rep = mine["rate"] - theirs["rate"], mine["jc_repetition"] - theirs["jc_repetition"]
        row[placement] = {
            "candidate": mine,
            "baseline": theirs,
            "rate_delta": delta,
            "jc_repetition_delta": rep,
            "pass": delta <= LIMIT and rep <= LIMIT,
        }
    row["pass"] = row["start"]["pass"]
    return row


def extra_words(baseline: Run, candidate: Run, reference_path: Any) -> dict[str, Any]:
    """Both lenses, candidate against baseline. VAD-silent audio is the baseline's (the runs must share an input)."""
    if not (token_timed(baseline) and token_timed(candidate)):
        return {"status": "unavailable", "reason": "Both runs need token timing (Parakeet)"}
    reference = read(reference_path)
    window = tuple(float(x) for x in reference["window"])
    silent = muted(baseline)
    out: dict[str, Any] = {
        "status": "available",
        "limit_pp": 100 * LIMIT,
        "same_input": same_input(baseline, candidate),
    }
    for lens in LENSES:
        try:
            sides = [side_of(run, reference["text"], window, lens) for run in (baseline, candidate)]
        except ScoringError as error:
            return {"status": "unavailable", "reason": f"Calls cannot be placed on the source clock: {error}"}
        out[lens] = pair(sides[1], sides[0], silent)
    return out
