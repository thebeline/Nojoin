# SPDX-License-Identifier: AGPL-3.0-only
"""Per-run metrics: VAD speech lost, reference speech lost (strict), WER under both lenses, time and memory.

Every function returns numbers and counts only, never transcript text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jiwer import process_words

from . import coverage, wer
from .evidence import ScoringError, call_evidence, coverage_metrics, unavailable
from .gaps import long_gaps
from .normalized import full
from .runs import RATE, Run, read
from .text import final_words, identity, repetition, within


def token_timed(run: Run) -> bool:
    """Parakeet returns token times; Canary does not, so its window spans say nothing about speech coverage."""
    return run.report.get("backend") == "parakeet"


def vad_lost(run: Run) -> dict[str, Any]:
    """Share of the VAD's kept speech (nonzero 10 ms bins of the engine input) that no output segment covers."""
    if not token_timed(run):
        return unavailable("No usable token timing; window spans do not establish speech coverage")
    intervals = coverage.clip_intervals([tuple(x) for x in run.vad["intervals"]], (0.0, run.duration))
    result = coverage.vad_coverage(run.result["segments"], intervals, run.duration)
    result.pop("holes")
    result.pop("runs_5s")
    return {"status": "available", "method": "nonzero-10ms-v1 VAD speech minus segment spans", **result}


def evidence_of(run: Run) -> Any:
    if not run.calls:
        raise ScoringError("No calls.json: the run was not traced")
    return call_evidence(run.calls, RATE, (0, run.frames))


def reference_lost(run: Run, words_path: Path) -> dict[str, Any]:
    """Strict: reference word time that no raw-token support touches (+/-0.1 s), against a word-timed reference."""
    if not token_timed(run):
        return unavailable("No usable token timing")
    payload = read(words_path)
    rows = payload["words"]
    reference = [(left, right) for left, right, *_ in rows if 0 <= left < right <= run.duration]
    try:
        evidence = evidence_of(run)
    except ScoringError as error:
        return unavailable(f"Calls cannot be placed on the source clock: {error}")
    result = coverage_metrics(reference, evidence, (0.0, run.duration))
    clock = payload.get("duration")
    result["reference_clock_difference_s"] = None if clock is None else clock - run.duration
    result["recovery_calls"] = evidence.recovery_calls
    return result


def _counts(scores: Any, reference_words: int) -> dict[str, Any]:
    return {
        "wer": scores.wer,
        "reference_words": reference_words,
        "hypothesis_words": scores.hits + scores.substitutions + scores.insertions,
        "hits": scores.hits,
        "substitutions": scores.substitutions,
        "deletions": scores.deletions,
        "insertions": scores.insertions,
        "insertion_rate": scores.insertions / reference_words,
        "deletion_rate": scores.deletions / reference_words,
    }


def text_scores(run: Run, reference_path: Path) -> dict[str, Any]:
    """WER against an official reference: frozen lens (as pre-registered) and normalized lens (post-hoc)."""
    frozen = wer.score_wer(run.result["segments"], reference_path, run.duration)
    window = tuple(frozen["window"])
    text, _ = wer.hypothesis_text(run.result["segments"], window)
    frozen.update(
        insertion_rate=frozen["insertions"] / frozen["reference_words"],
        deletion_rate=frozen["deletions"] / frozen["reference_words"],
        repetition=repetition([text], wer.normalize_text),
    )
    reference = full(read(reference_path)["text"]).split()
    hypothesis = full(text).split()
    normalized = _counts(process_words(" ".join(reference), " ".join(hypothesis)), len(reference))
    normalized.update(
        repetition=repetition([" ".join(hypothesis)], identity),
        window=list(window),
        normalizer="whisper-english-v20250625",
    )
    return {"frozen": frozen, "normalized": normalized}


def proxy_scores(run: Run, words_path: Path) -> dict[str, Any]:
    """WER against the word reference's own text over the whole run, for runs with no official text (Canary slice)."""
    rows = read(words_path)["words"]
    if not rows or len(rows[0]) < 3:
        return unavailable("The word reference has no text column")
    text, excluded = wer.hypothesis_text(run.result["segments"], (0.0, run.duration))
    result = wer.word_error_rate(within(rows, 0.0, run.duration), text)
    result.update(
        excluded,
        repetition=repetition([text], wer.normalize_text),
        label="independent Whisper-text proxy; not acoustic ground truth",
        window=[0.0, run.duration],
    )
    return result


def run_facts(run: Run) -> dict[str, Any]:
    report = run.report
    windows = [b - a for a, b in report.get("windows", [])]
    words = final_words(run.result)
    return {
        "backend": report.get("backend"),
        "gpu_is_present": report.get("gpu_is_present"),
        "quantization": report.get("quantization"),
        "max_chunk_duration_s": report.get("max_chunk_duration_s"),
        "image": (run.image or {}).get("reference"),
        "image_id": (run.image or {}).get("id"),
        "input_sha256": report["audio"]["sha256"],
        "pcm_sha256": run.vad.get("pcm_sha256"),
        "duration_s": run.duration,
        "words": len(wer.normalize_text(run.result.get("text") or "").split()),
        "words_timed": len(words),
        "words_whitespace": len((run.result.get("text") or "").split()),
        "windows": len(windows),
        "window_max_s": max(windows, default=None),
        "window_mean_s": sum(windows) / len(windows) if windows else None,
        "calls": report.get("calls"),
        "windows_tile_input": report.get("windows_tile_input"),
        "windows_recomputed_match": report.get("windows_recomputed_match"),
        "asr_wall_s": report.get("asr_wall_s"),
        "load_wall_s": report.get("load_wall_s"),
        "transcribe_wall_s": report.get("transcribe_wall_s"),
        "max_rss_kib": report.get("max_rss_kib"),
        "peak_vram_mib": (run.vram or {}).get("peak_owned_mib"),
        "sessions": report.get("sessions"),
    }


def score(run: Run, words: Path | None, reference: Path | None, proxy: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"run": str(run.directory), "facts": run_facts(run), "vad_speech_lost": vad_lost(run)}
    if words:
        out["reference_speech_lost"] = reference_lost(run, words)
        if token_timed(run):
            out["long_gaps"] = long_gaps(run, read(words)["words"])
        if proxy:
            out["proxy_text"] = proxy_scores(run, words)
    if reference:
        out["text"] = text_scores(run, reference)
    return out
