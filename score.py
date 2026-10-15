#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Score one run directory written by run.sh. Prints a summary and writes RUN_DIR/score.json.

  python score.py RUN_DIR                                   # VAD speech lost, words, windows, time
  python score.py RUN_DIR --words lords-words.json          # + reference speech lost (strict)
  python score.py RUN_DIR --wer-ref lords-wer.json          # + WER, frozen and normalized lenses
  python score.py RUN_DIR --words W.json --proxy            # + WER against the word reference's own text

Numbers and counts only: no transcript text is printed or written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also under python -I
from scoring import metrics, normalized, runs  # noqa: E402


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def summary(scores: dict) -> list[str]:
    facts = scores["facts"]
    lines = [
        f"{scores['run']}: {facts['backend']}, {facts['words']} words, {facts['windows']} windows "
        f"(max {facts['window_max_s'] or 0:.1f} s), ASR {facts['asr_wall_s']:.1f} s, load {facts['load_wall_s']:.1f} s"
    ]
    vad = scores["vad_speech_lost"]
    lines.append(
        "  VAD speech lost: "
        + (
            f"{vad['hole_percent']:.2f} % ({vad['runs_5s_count']} runs >= 5 s)"
            if vad["status"] == "available"
            else vad["reason"]
        )
    )
    if "reference_speech_lost" in scores:
        ref = scores["reference_speech_lost"]
        lines.append(
            "  reference speech lost (strict): "
            + (
                f"{ref['uncovered_percent']:.2f} % ({ref['runs_5s_count']} runs >= 5 s)"
                if ref["status"] == "available"
                else ref["reason"]
            )
        )
    if "long_gaps" in scores:
        g = scores["long_gaps"]
        lines.append(
            f"  gaps > {g['min_gap_s']:.0f} s: {g['gaps']}, holding {g['reference_words_inside']:,} of "
            f"{g['reference_words']:,} reference words ({pct(g['share_of_reference'])} %); "
            f"{g['gaps_ending_at_a_cut']} end <= {g['tolerance_s']:.0f} s after a window cut, "
            f"holding {g['their_reference_words']:,}"
        )
    if "text" in scores:
        frozen, norm = scores["text"]["frozen"], scores["text"]["normalized"]
        lines.append(f"  WER frozen {pct(frozen['wer'])} % / normalized {pct(norm['wer'])} %")
    if "proxy_text" in scores and scores["proxy_text"].get("wer") is not None:
        lines.append(f"  WER against the word reference's text: {pct(scores['proxy_text']['wer'])} %")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--words", type=Path, help="word-timed reference from whisper_words.py (Lords)")
    parser.add_argument("--wer-ref", type=Path, help="reference JSON from build_wer_reference.py (Lords, London)")
    parser.add_argument("--proxy", action="store_true", help="also score WER against the --words reference's text")
    parser.add_argument("--json", type=Path, help="where to write the scores (default RUN_DIR/score.json)")
    args = parser.parse_args()
    normalized.self_check()
    scores = metrics.score(runs.load(args.run), args.words, args.wer_ref, args.proxy)
    out = args.json or args.run / "score.json"
    out.write_text(json.dumps(scores, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print("\n".join(summary(scores)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
