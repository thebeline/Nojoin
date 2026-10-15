# SPDX-License-Identifier: AGPL-3.0-only
"""Run one recording through a Nojoin worker image's own prep and transcribe_audio, with nothing changed.

Runs inside the worker image; run.sh mounts this bundle at /bundle and starts it with the image's Python. Steps:
1. prep, exactly as the pipeline's VAD stage does it: preprocess_audio_for_vad (16 kHz mono), then
   mute_non_speech_segments (non-speech set to digital zero). The result is the file the engine transcribes,
   kept as /out/prepared.wav. --prepared skips this for a WAV that already is one;
2. --head-s: optionally keep only the first S seconds of it, sample-exactly (/out/input.wav);
3. vad-intervals.json: every 10 ms bin of the transcribed file holding a nonzero sample, i.e. the speech the
   VAD kept. The scorer needs no reference for "VAD speech lost";
4. transcribe_audio with the image's default settings and the chosen backend.

No product constant or setting is changed. A passive trace wraps the engine's model so that every recognize()
call is recorded (its span on the source clock, found by exact sample match, its text, raw subword tokens and
token times, and its wall time) and the library's own result is returned untouched. The model load inside
transcribe_audio is timed separately. Outputs in /out: prep.json, vad-intervals.json, result.json (the product's
output, verbatim), calls.json and report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from trace_calls import CallTrace, attach

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("drive")
RATE = 16000
BIN_FRAMES = 160  # 10 ms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object, indent: int | None = 1) -> None:
    path.write_text(json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def env_report() -> dict[str, Any]:
    import importlib.metadata as md

    import onnxruntime

    names = sorted(
        d.metadata["Name"] + "==" + d.version
        for d in md.distributions()
        if any(k in d.metadata["Name"].lower() for k in ("onnx", "torch", "silero", "soundfile", "numpy"))
    )
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ORT_LOG_SEVERITY_LEVEL",
    )
    return {
        "python": sys.version.split()[0],
        "onnxruntime": onnxruntime.__version__,
        "distributions": names,
        "available_providers": onnxruntime.get_available_providers(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: os.environ.get(k) for k in keys},
    }


def prep(source: Path, out: Path) -> Path:
    """The pipeline's VAD stage: 16 kHz mono, then non-speech muted. Returns /out/prepared.wav."""
    from backend.processing.audio_preprocessing import preprocess_audio_for_vad
    from backend.processing.vad import mute_non_speech_segments
    from backend.utils.config_manager import config_manager

    config = dict(config_manager.get_all())
    started = time.perf_counter()
    vad_in = preprocess_audio_for_vad(str(source))
    if not vad_in:
        raise RuntimeError("preprocess_audio_for_vad failed")
    vad_out = vad_in.replace("_vad.wav", "_vad_processed.wav")
    ok, speech = mute_non_speech_segments(vad_in, vad_out, config=config)
    if not ok:
        raise RuntimeError("mute_non_speech_segments failed")
    prepared = out / "prepared.wav"
    shutil.copyfile(vad_out, prepared)
    info = soundfile.info(str(prepared))
    record = {
        "source": source.name,
        "source_sha256": sha256(source),
        "unmuted_sha256": sha256(Path(vad_in)),
        "prepared_sha256": sha256(prepared),
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "subtype": info.subtype,
        "duration_s": info.duration,
        "speech_duration_s": speech,
        "prep_wall_s": time.perf_counter() - started,
    }
    for path in (vad_in, vad_out):
        Path(path).unlink(missing_ok=True)
    write_json(out / "prep.json", record)
    log.info("PREP %s", json.dumps(record))
    return prepared


def head(prepared: Path, seconds: float, out: Path) -> Path:
    """The first `seconds` of the prepared file, sample-exactly, as PCM_16."""
    data, rate = soundfile.read(str(prepared), dtype="int16", always_2d=False)
    frames = int(round(seconds * rate))
    path = out / "input.wav"
    soundfile.write(str(path), data[:frames], rate, subtype="PCM_16")
    log.info("HEAD %s frames=%d", path, min(frames, len(data)))
    return path


def _append_bins(active: np.ndarray, offset: int, runs: list[list[int]]) -> None:
    edges = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    for first, end in edges:
        first, end = int(first) + offset, int(end) + offset
        if runs and runs[-1][1] == first:
            runs[-1][1] = end
        else:
            runs.append([first, end])


def nonzero_intervals(path: Path, block_frames: int = 65537) -> dict[str, Any]:
    """Hash all decoded float32 PCM; classify only complete global 160-frame (10 ms) bins as kept or muted."""
    digest = hashlib.sha256()
    runs: list[list[int]] = []
    pending = np.empty(0, dtype=np.float32)
    consumed = bins = 0
    with soundfile.SoundFile(str(path)) as stream:
        if (stream.samplerate, stream.channels) != (RATE, 1):
            raise ValueError("The transcribed file is not 16 kHz mono")
        while True:
            chunk = stream.read(block_frames, dtype="float32", always_2d=False)
            if not len(chunk):
                break
            consumed += len(chunk)
            digest.update(chunk.astype("<f4", copy=False).tobytes())
            joined = np.concatenate((pending, chunk))
            complete = len(joined) // BIN_FRAMES * BIN_FRAMES
            active = np.any(joined[:complete].reshape(-1, BIN_FRAMES) != 0, axis=1)
            _append_bins(active, bins, runs)
            bins += len(active)
            pending = joined[complete:].copy()
    return {
        "method": "nonzero-10ms-v1",
        "sample_rate": RATE,
        "frames": consumed,
        "duration_s": consumed / RATE,
        "complete_bins": bins,
        "discarded_tail_frames": len(pending),
        "pcm_sha256": digest.hexdigest(),
        "pcm_encoding": "decoded-float32-little-endian-mono",
        "input_sha256": sha256(path),
        "intervals": [[first * BIN_FRAMES / RATE, end * BIN_FRAMES / RATE] for first, end in runs],
    }


def engine_facts(engine_module: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Settings the engine derives for this run, read through its own functions (each optional across versions)."""
    facts: dict[str, Any] = {}
    for name, call in (
        ("gpu_is_present", lambda: engine_module.gpu_is_present()),
        ("max_chunk_duration_s", lambda: engine_module.max_chunk_duration_s(config)),
        ("quantization", lambda: engine_module.onnx_asr_quantization(engine_module.gpu_is_present())),
        ("snap_radius_s", lambda: engine_module.CHUNK_SNAP_RADIUS_S),
    ):
        try:
            facts[name] = call()
        except (AttributeError, TypeError) as error:
            facts[name] = f"unavailable: {error}"
    return facts


def session_threads(engine: Any, config: dict[str, Any]) -> list[dict[str, Any]] | str:
    """Thread settings and providers of every onnxruntime session the run used."""
    try:
        from backend.processing.onnx_providers import iter_inference_sessions

        model = engine._model_cache[engine._onnx_id_for(config)]
        rows = []
        for session in iter_inference_sessions(model):
            options = session.get_session_options()
            rows.append(
                {
                    "providers": session.get_providers(),
                    "intra_op_threads": options.intra_op_num_threads,
                    "inter_op_threads": options.inter_op_num_threads,
                }
            )
        return rows
    except (ImportError, AttributeError, KeyError, TypeError) as error:
        return f"unavailable: {error}"


def recomputed_windows(engine_module: Any, audio_path: Path, window_s: Any) -> list[list[float]] | str:
    """The engine's own cut function re-run on the input, as a cross-check of the traced windows."""
    try:
        audio, rate = soundfile.read(str(audio_path), dtype="float32", always_2d=False)
        if len(audio) / rate <= float(window_s):
            return [[0.0, len(audio) / rate]]
        return [[a / rate, b / rate] for a, b in engine_module._chunk_boundaries(audio, rate, window_s)]
    except (AttributeError, TypeError, ValueError) as error:
        return f"unavailable: {error}"


def transcribe(audio_path: Path, backend: str, out: Path) -> dict[str, Any]:
    from backend.processing import transcribe as tr
    from backend.processing.engines import onnx_asr_engine as engine_module
    from backend.utils.config_manager import config_manager

    config = dict(config_manager.get_all())
    config["transcription_backend"] = backend
    engine = tr._get_engine(backend)
    trace = CallTrace(audio_path, RATE)
    traced = attach(engine, trace)
    report: dict[str, Any] = {
        "schema": "nojoin-asr-repro-run/v1",
        "backend": backend,
        "env": env_report(),
        "audio": {
            "path": str(audio_path),
            "sha256": sha256(audio_path),
            "frames": soundfile.info(str(audio_path)).frames,
            "sample_rate": RATE,
        },
        **engine_facts(engine_module, config),
        "traced": traced,
        "config_subset": {
            k: config.get(k)
            for k in (
                "transcription_backend",
                "parakeet_model",
                "canary_model",
                "transcription_language",
                "processing_device",
                "gpu_asr_window_s",
                "enable_vad",
            )
        },
    }
    log.info("REPORT-PRE %s", json.dumps(report))
    started = time.perf_counter()
    result = tr.transcribe_audio(str(audio_path), config=config)
    wall = time.perf_counter() - started
    report.update(
        trace.summary(),
        transcribe_wall_s=wall,
        load_wall_s=trace.load_s,
        trace_overhead_s=trace.overhead_s,
        asr_wall_s=wall - trace.load_s - trace.overhead_s,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    report["sessions"] = session_threads(engine, config)
    report["windows_recomputed"] = recomputed_windows(engine_module, audio_path, report.get("max_chunk_duration_s"))
    report["windows_recomputed_match"] = (
        report["windows_recomputed"] == report["windows"] if isinstance(report["windows_recomputed"], list) else None
    )
    write_json(out / "result.json", result, indent=None)
    if traced:
        write_json(out / "calls.json", trace.calls, indent=None)
    write_json(out / "report.json", report)
    words = len((result.get("text") or "").split())
    log.info(
        "DONE %s words=%d asr=%.1fs load=%.1fs windows=%d calls=%d",
        backend,
        words,
        report["asr_wall_s"],
        trace.load_s,
        len(report["windows"]),
        len(trace.calls),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path, help="recording inside the container")
    parser.add_argument("--out", required=True, type=Path, help="output directory inside the container")
    parser.add_argument("--backend", default="parakeet", choices=("parakeet", "canary"))
    parser.add_argument("--prepared", action="store_true", help="the audio already is a prepared (muted) 16 kHz WAV")
    parser.add_argument("--head-s", type=float, help="transcribe only the first S seconds of the prepared audio")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audio = args.audio if args.prepared else prep(args.audio, args.out)
    if args.head_s:
        audio = head(audio, args.head_s, args.out)
    write_json(args.out / "vad-intervals.json", nonzero_intervals(audio), indent=None)
    transcribe(audio, args.backend, args.out)


if __name__ == "__main__":
    main()
