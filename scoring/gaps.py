# SPDX-License-Identifier: AGPL-3.0-only
"""Long transcript gaps, the reference words inside them, and how many gaps end just after a window cut.

A gap is a pair of consecutive output-word starts more than `min_gap_s` apart. Its reference words are those whose
start lies strictly inside it. A gap "ends at a cut" when its end (the next word's start) falls 0 to `tolerance_s`
after an interior window boundary. Timestamps and counts only.
"""

from __future__ import annotations

import bisect
from typing import Any

from .runs import Run


def long_gaps(run: Run, rows: list[list[Any]], min_gap_s: float = 20.0, tolerance_s: float = 5.0) -> dict[str, Any]:
    starts = sorted(w["start"] for s in run.result["segments"] for w in s.get("words", []))
    references = sorted(row[0] for row in rows)
    cuts = [end for _, end in run.report.get("windows", [])[:-1]]
    gaps = [(a, b) for a, b in zip(starts, starts[1:], strict=False) if b - a > min_gap_s]
    inside = [bisect.bisect_left(references, b) - bisect.bisect_right(references, a) for a, b in gaps]
    at_cut = [any(0 <= b - c <= tolerance_s for c in cuts) for _, b in gaps]
    total = sum(inside)
    return {
        "min_gap_s": min_gap_s,
        "tolerance_s": tolerance_s,
        "gaps": len(gaps),
        "reference_words_inside": total,
        "reference_words": len(references),
        "share_of_reference": total / len(references) if references else None,
        "gaps_ending_at_a_cut": sum(at_cut),
        "their_reference_words": sum(n for n, hit in zip(inside, at_cut, strict=True) if hit),
    }
