#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""An independent word-timed transcript for "reference speech lost": faster-whisper large-v3-turbo, not Nojoin.

Usage: python whisper_words.py AUDIO OUT.json [--device cpu|cuda]
Writes {"duration": s, "words": [[start, end, word], ...], "segments": [[start, end, text], ...], "made_with": {...}},
where made_with records the model, device, compute type, settings and library versions.

The settings are the ones our Lords reference used: model mobiuslabsgmbh/faster-whisper-large-v3-turbo, English,
word_timestamps=True, vad_filter=True, condition_on_previous_text=False, faster-whisper's other defaults (beam 5,
temperature fallback), int8 on CPU or float16 on CUDA, run on the original recording (the .m4a, which faster-whisper
decodes to 16 kHz itself). Ours (20,397 words for Lords) was made with this script on 2026-10-09; its log records
neither the device nor the versions, but the script was run for CPU int8 and the faster-whisper installed then was
1.2.1 (ctranslate2 4.8.2). Another device, version or model snapshot moves word times slightly, so your strict
numbers can differ from ours by a little; compare your own main and PR runs against your own reference.

The output is ASR text of the recording. Keep it local: for the Lords sitting it must not be published.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version

MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
SETTINGS = {"language": "en", "word_timestamps": True, "vad_filter": True, "condition_on_previous_text": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    from faster_whisper import WhisperModel

    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(MODEL, device=args.device, compute_type=compute_type)
    segments, info = model.transcribe(args.audio, **SETTINGS)
    made_with = {
        "model": MODEL,
        "device": args.device,
        "compute_type": compute_type,
        "settings": SETTINGS,
        "faster_whisper": version("faster-whisper"),
        "ctranslate2": version("ctranslate2"),
    }
    words, lines = [], []
    for segment in segments:
        lines.append([round(segment.start, 2), round(segment.end, 2), segment.text.strip()])
        for word in segment.words or []:
            words.append([round(word.start, 2), round(word.end, 2), word.word.strip()])
        if len(lines) % 200 == 0:
            print(f"{segment.end:.0f} s", flush=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"duration": info.duration, "words": words, "segments": lines, "made_with": made_with}, stream)
    print(f"{len(words)} words, {len(lines)} segments, duration {info.duration:.4f} s")


if __name__ == "__main__":
    main()
