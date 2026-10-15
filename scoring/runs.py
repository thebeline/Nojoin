# SPDX-License-Identifier: AGPL-3.0-only
"""Load one run directory written by drive.py (and run.sh), and check its files agree with each other."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RATE = 16000


@dataclass(frozen=True)
class Run:
    directory: Path
    result: dict[str, Any]
    calls: list[dict[str, Any]] | None
    report: dict[str, Any]
    vad: dict[str, Any]
    vram: dict[str, Any] | None
    image: dict[str, Any] | None

    @property
    def frames(self) -> int:
        return int(self.report["audio"]["frames"])

    @property
    def duration(self) -> float:
        return self.frames / RATE

    @property
    def label(self) -> str:
        return self.directory.name


def same_input(a: Run, b: Run) -> bool:
    """The same decoded samples where both runs recorded pcm_sha256 (WAV headers can differ), else the same file."""
    pcm_a, pcm_b = a.vad.get("pcm_sha256"), b.vad.get("pcm_sha256")
    if pcm_a and pcm_b:
        return pcm_a == pcm_b and a.frames == b.frames
    return a.report["audio"]["sha256"] == b.report["audio"]["sha256"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path) -> Any:
    return read(path) if path.is_file() else None


def load(directory: Path) -> Run:
    report = read(directory / "report.json")
    vad = read(directory / "vad-intervals.json")
    if vad["frames"] != report["audio"]["frames"] or vad["input_sha256"] != report["audio"]["sha256"]:
        raise ValueError(f"{directory}: vad-intervals.json belongs to a different input than report.json")
    return Run(
        directory,
        read(directory / "result.json"),
        optional(directory / "calls.json"),
        report,
        vad,
        optional(directory / "vram.json"),
        optional(directory / "image.json"),
    )
