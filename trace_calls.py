# SPDX-License-Identifier: AGPL-3.0-only
"""A passive trace of an onnx-asr engine's recognize() calls, placed on the source clock by exact sample match.

attach() wraps the engine instance's _get_model so the model it returns is proxied: with_timestamps() (and a bare
recognize()) hand back recognizers whose recognize() records the call and then returns the library's own result,
unchanged. Nothing else in the engine is touched, so the windowing under test is the image's own.

Placement needs no knowledge of how the engine cuts. Each call's audio (a WAV path, or a float32 array with
sample_rate) is read back as 16-bit samples, which the engine's PCM_16 writes preserve exactly, and matched against
the transcribed file: first at the end of the previous window (windows that tile the audio), then inside an earlier
window, the latest first (a re-decode of part of it, "recovery"), then anywhere later (a window that skips audio).
A call that matches nowhere, such as an all-zero chunk away from the cursor, is recorded as "unplaced", and the strict
scorer then refuses rather than guess.
"""

from __future__ import annotations

import math
import os
import time
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import soundfile


class CallTrace:
    def __init__(self, source: Path, rate: int) -> None:
        self.source, self.rate = soundfile.read(str(source), dtype="int16", always_2d=False)[0], rate
        self.calls: list[dict[str, Any]] = []
        self.cursor = 0
        self.load_s = self.overhead_s = 0.0

    def samples(self, audio: Any, kwargs: dict[str, Any]) -> np.ndarray | None:
        if isinstance(audio, (str, os.PathLike)):
            data, rate = soundfile.read(str(audio), dtype="int16", always_2d=False)
            return data if rate == self.rate and data.ndim == 1 else None
        if isinstance(audio, np.ndarray) and audio.ndim == 1 and kwargs.get("sample_rate", self.rate) == self.rate:
            if audio.dtype == np.int16:
                return audio
            scaled = audio.astype(np.float64) * 32768.0
            ints = np.rint(scaled)
            return np.clip(ints, -32768, 32767).astype(np.int16) if np.array_equal(ints, scaled) else None
        return None

    def find(self, data: np.ndarray, low: int, high: int) -> int | None:
        """First offset p in [low, high - len(data)] where the source equals data exactly."""
        n = len(data)
        nonzero = np.flatnonzero(data)
        if not len(nonzero) or high - low < n:
            return None
        anchor = int(nonzero[np.argmax(np.abs(data[nonzero].astype(np.int32)))])
        offsets = np.flatnonzero(self.source[low + anchor : high - n + anchor + 1] == data[anchor]) + low
        probes = np.unique(np.linspace(0, n - 1, 9).astype(np.int64))
        for probe in probes:
            offsets = offsets[self.source[offsets + probe] == data[probe]]
        for offset in offsets:
            if np.array_equal(self.source[offset : offset + n], data):
                return int(offset)
        return None

    def place(self, data: np.ndarray | None) -> tuple[str, int | None, int | None, int | None]:
        if data is None or not len(data):
            return "unplaced", None, None, None
        n, cursor = len(data), self.cursor
        if cursor + n <= len(self.source) and np.array_equal(self.source[cursor : cursor + n], data):
            self.cursor = cursor + n
            return "initial", cursor, cursor + n, None
        earlier = [c for c in reversed(self.calls) if c["kind"] == "initial"]  # the latest window first
        for parent in earlier:
            offset = self.find(data, parent["start_sample"], parent["end_sample"])
            if offset is not None:
                return "recovery", offset, offset + n, parent["index"]
        offset = self.find(data, cursor, len(self.source))
        if offset is not None:
            self.cursor = offset + n
            return "initial", offset, offset + n, None
        return "unplaced", None, None, None

    def record(self, audio: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        data = self.samples(audio, kwargs)
        kind, start, end, parent = self.place(data)
        index = len(self.calls)
        entry = {
            "index": index,
            "kind": kind,
            "initial_index": index if kind == "initial" else parent,
            "start_sample": start,
            "end_sample": end,
            "sample_rate": self.rate,
            "frames": None if data is None else len(data),
            "text": "",
            "tokens": None,
            "timestamps": None,
            "status": "in-progress",
        }
        self.calls.append(entry)
        return entry

    def recognize(self, inner: Any, audio: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        began = time.perf_counter()
        entry = self.record(audio, kwargs)
        called = time.perf_counter()
        try:
            result = inner.recognize(audio, *args, **kwargs)
        except BaseException as error:
            entry.update(status="failed", error=type(error).__name__)
            raise
        returned = time.perf_counter()
        entry.update(raw_result(result), wall_s=returned - called)
        entry.pop("status")
        self.overhead_s += (called - began) + (time.perf_counter() - returned)
        return result

    def summary(self) -> dict[str, Any]:
        kinds = [c["kind"] for c in self.calls]
        initial = [c for c in self.calls if c["kind"] == "initial"]
        return {
            "calls": {k: kinds.count(k) for k in ("initial", "recovery", "unplaced")},
            "windows": [[c["start_sample"] / self.rate, c["end_sample"] / self.rate] for c in initial],
            "windows_tile_input": bool(initial)
            and initial[0]["start_sample"] == 0
            and all(a["end_sample"] == b["start_sample"] for a, b in zip(initial, initial[1:], strict=False))
            and initial[-1]["end_sample"] == len(self.source),
        }


def raw_result(result: Any) -> dict[str, Any]:
    """Text, subword tokens and token start times of one recognize() result, or why they are missing."""
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        return {"text": "", "opaque_result": type(result).__name__}
    tokens = getattr(result, "tokens", None)
    stamps = getattr(result, "timestamps", None)
    tokens = None if tokens is None else [str(t) for t in tokens]
    if stamps is not None:
        stamps = [float(s) for s in stamps]
        finite = all(isinstance(s, Real) and math.isfinite(s) for s in stamps)
        if tokens is None or len(stamps) != len(tokens) or not finite:
            return {"text": text, "tokens": tokens, "timestamps": None, "malformed_timestamps": True}
    return {"text": text, "tokens": tokens, "timestamps": stamps}


class RecognizerProxy:
    def __init__(self, inner: Any, trace: CallTrace) -> None:
        self._inner, self._trace = inner, trace

    def recognize(self, audio: Any, *args: Any, **kwargs: Any) -> Any:
        return self._trace.recognize(self._inner, audio, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ModelProxy(RecognizerProxy):
    def with_timestamps(self, *args: Any, **kwargs: Any) -> RecognizerProxy:
        return RecognizerProxy(self._inner.with_timestamps(*args, **kwargs), self._trace)


def attach(engine: Any, trace: CallTrace) -> bool:
    """Proxy the model the engine loads; time the load. False (untraced run) if the engine has no _get_model."""
    original = getattr(engine, "_get_model", None)
    if original is None:
        return False

    def traced_get_model(config: dict[str, Any]) -> Any:
        started = time.perf_counter()
        model = original(config)
        trace.load_s += time.perf_counter() - started
        return ModelProxy(model, trace)

    engine._get_model = traced_get_model
    return True
