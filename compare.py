#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare a candidate run with a baseline run of the same recording, in the units of the write-up.

  python compare.py BASELINE_DIR CANDIDATE_DIR [--words lords-words.json] [--wer-ref lords-wer.json] [--json out.json]

Prints a Markdown table: VAD speech lost, reference speech lost (strict; needs --words), WER under both lenses and
extra words (both need --wer-ref), ASR time, peak VRAM (GPU runs), windows and threads. Lower is better for every
loss, WER and extra-words row. The +1 pp limits are the ones the write-up uses for WER and extra words.
Numbers and counts only: no transcript text is printed or written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also under python -I
from scoring import extra_words, metrics, normalized, runs  # noqa: E402


def num(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def delta(a: float | None, b: float | None, scale: float = 1.0) -> str:
    return "" if a is None or b is None else f"{scale * (b - a):+.2f}"


def threads(facts: dict) -> str:
    sessions = facts.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        return "n/a"
    pairs = sorted({(s["intra_op_threads"], s["inter_op_threads"]) for s in sessions})
    providers = sorted({p for s in sessions for p in s["providers"][:1]})
    return ", ".join(f"{i}/{j}" for i, j in pairs) + " (" + ", ".join(providers) + ")"


def coverage_rows(base: dict, cand: dict) -> list[list[str]]:
    rows = []
    vb, vc = base["vad_speech_lost"], cand["vad_speech_lost"]
    if vb["status"] == vc["status"] == "available":
        rows.append(
            [
                "VAD speech lost, %",
                num(vb["hole_percent"]),
                num(vc["hole_percent"]),
                delta(vb["hole_percent"], vc["hole_percent"]),
            ]
        )
    if "reference_speech_lost" in base:
        rb, rc = base["reference_speech_lost"], cand["reference_speech_lost"]
        if rb["status"] == rc["status"] == "available":
            rows.append(
                [
                    "Reference speech lost (strict), % (runs >= 5 s)",
                    f"{num(rb['uncovered_percent'])} ({rb['runs_5s_count']})",
                    f"{num(rc['uncovered_percent'])} ({rc['runs_5s_count']})",
                    delta(rb["uncovered_percent"], rc["uncovered_percent"]),
                ]
            )
        else:
            rows.append(["Reference speech lost (strict), %", rb.get("reason", "ok"), rc.get("reason", "ok"), ""])
    return rows


def text_rows(base: dict, cand: dict, extra: dict | None) -> list[list[str]]:
    if "text" not in base:
        return []
    rows = []
    for lens, name in (("frozen", "frozen"), ("normalized", "normalized")):
        a, b = base["text"][lens]["wer"], cand["text"][lens]["wer"]
        verdict = "" if a is None or b is None else (" (within +1 pp)" if 100 * (b - a) <= 1 else " (over +1 pp)")
        rows.append([f"WER, {name}, %", pct(a), pct(b), delta(a, b, 100) + verdict])
    if extra and extra.get("status") == "available":
        for lens, name in (("a", "frozen"), ("d", "normalized")):
            row = extra[lens]["start"]
            verdict = "pass" if row["pass"] else "fail"
            rows.append(
                [
                    f"Extra words, {name}, % of reference words",
                    pct(row["baseline"]["rate"]),
                    pct(row["candidate"]["rate"]),
                    f"{100 * row['rate_delta']:+.2f} ({verdict}; "
                    f"JC 4-gram {100 * row['jc_repetition_delta']:+.2f}; midpoint "
                    f"{100 * extra[lens]['midpoint']['rate_delta']:+.2f})",
                ]
            )
    return rows


def facts_rows(base: dict, cand: dict) -> list[list[str]]:
    fb, fc = base["facts"], cand["facts"]
    ratio = f"x{fc['asr_wall_s'] / fb['asr_wall_s']:.2f}" if fb.get("asr_wall_s") and fc.get("asr_wall_s") else ""
    vram = []
    a, b = fb["peak_vram_mib"], fc["peak_vram_mib"]
    if a or b:
        vram = [["Peak VRAM, MiB (owned)", num(a, 0), num(b, 0), "" if None in (a, b) else f"{b - a:+,}"]]
    return [
        ["ASR time, s (model load excluded)", num(fb["asr_wall_s"], 1), num(fc["asr_wall_s"], 1), ratio],
        *vram,
        [
            "Windows (longest, s)",
            f"{fb['windows']} ({num(fb['window_max_s'], 1)})",
            f"{fc['windows']} ({num(fc['window_max_s'], 1)})",
            "",
        ],
        ["Recognize calls: initial / recovery / unplaced", calls(fb), calls(fc), ""],
        ["Words (frozen lens)", f"{fb['words']:,}", f"{fc['words']:,}", ""],
        ["Threads intra/inter (provider)", threads(fb), threads(fc), ""],
        [
            "Device, quantization",
            f"{'GPU' if fb['gpu_is_present'] else 'CPU'}, {fb['quantization']}",
            f"{'GPU' if fc['gpu_is_present'] else 'CPU'}, {fc['quantization']}",
            "",
        ],
    ]


def calls(facts: dict) -> str:
    c = facts.get("calls") or {}
    return f"{c.get('initial', 0)} / {c.get('recovery', 0)} / {c.get('unplaced', 0)}"


def warnings(base: dict, cand: dict, extra: dict | None, same: bool) -> list[str]:
    out = []
    if not same:
        out.append("The two runs transcribed different inputs (prep differs): coverage deltas mix prep and windowing.")
    for label, scores in (("baseline", base), ("candidate", cand)):
        facts = scores["facts"]
        if (facts.get("calls") or {}).get("unplaced"):
            out.append(f"{label}: some recognize() calls could not be placed; strict loss and extra words refuse.")
        if facts.get("windows_tile_input") is False:
            out.append(f"{label}: its windows do not tile the input; the strict scorer requires that they do.")
    if extra and extra.get("status") != "available":
        out.append(f"extra words: {extra.get('reason')}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--words", type=Path, help="word-timed reference from whisper_words.py (Lords)")
    parser.add_argument("--wer-ref", type=Path, help="reference JSON from build_wer_reference.py (Lords, London)")
    parser.add_argument("--json", type=Path, help="write every number behind the table here")
    args = parser.parse_args()
    normalized.self_check()
    base_run, cand_run = runs.load(args.baseline), runs.load(args.candidate)
    base = metrics.score(base_run, args.words, args.wer_ref)
    cand = metrics.score(cand_run, args.words, args.wer_ref)
    extra = extra_words.extra_words(base_run, cand_run, args.wer_ref) if args.wer_ref else None
    rows = coverage_rows(base, cand) + text_rows(base, cand, extra) + facts_rows(base, cand)
    print(f"| Metric | Baseline: {base_run.label} | Candidate: {cand_run.label} | Candidate - baseline |")
    print("|---|---:|---:|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    for line in warnings(base, cand, extra, runs.same_input(base_run, cand_run)):
        print(f"\nWARNING: {line}")
    if args.json:
        args.json.write_text(
            json.dumps({"baseline": base, "candidate": cand, "extra_words": extra}, indent=1, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
