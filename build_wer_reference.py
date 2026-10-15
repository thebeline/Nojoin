#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a WER reference JSON from the official record you downloaded. No text ships with this bundle.

  lords:  python build_wer_reference.py lords  --hansard-dir DIR --out lords-wer.json
          DIR holds debate-<ID>.json for the nine Hansard debates of the sitting (fetch.sh gets them).
  london: python build_wer_reference.py london --appendix-1 A1.docx --appendix-2 A2.docx --out london-wer.json

Lords (Hansard, House of Lords, 16 Sep 2026; Open Parliament Licence v3.0): member contributions of the nine debates
in sitting order, consecutive items by one member joined as one contribution, HTML tags and editorial [bracketed]
insertions removed. Contributions 1-103 are kept: every complete contribution inside the scored window. The window
is [40.95, 8801.45] s of the 2:30:00 fixture clock: from the speech onset of the first contribution to the onset of
contribution 104, which the fixture's end clips. Both are speech onsets measured on the audio, so they are approximate
(to within about a second). Hansard is edited, so it omits some procedural speech and disfluencies.

London (GLA, Mayor's Question Time, 17 Sep 2026): the verbatim transcript appendices of agenda items 3 (oral update to
the Mayor's report) and 4 (questions to the Mayor). Kept: paragraphs that open with a speaker label (an Assembly
Member "... AM:", or "(Mayor of London):") and the paragraphs that continue them; a printed question heading
("2026/1234 - ...") ends a speech; editorial [bracketed] text removed. The window is the whole recording, so speech
outside items 3 and 4 (opening business, procedural speech) counts as insertions.

The output is written deterministically; if the official text has not changed since we built ours, its sha256 equals
the pinned one below, and your WER numbers are comparable with ours.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

HANSARD_DEBATES = (  # hansard-api.parliament.uk debate ids, in sitting order
    "0720AC06-4EE1-43FF-8E1E-49E1645A5D24",
    "837FFD01-30FB-4045-81DF-C7CA55AAE9D8",
    "7BE42E5F-D56E-4E34-A456-32910E425F4E",
    "F287EEA9-AD27-400F-B27D-E06A143BC5D6",
    "2DAA5878-086E-4803-91AB-C1DA14715B28",
    "E4E8845A-95FB-4C4A-99AD-979BD490FE63",
    "E53C08F4-2C0B-455B-BF42-5021FE91E903",
    "ABD1D9B6-9C05-423D-A0F2-60B1E41B641E",
    "E200C38F-7290-48DD-9475-E986D53C3523",
)
LORDS_CONTRIBUTIONS = (1, 103)  # first and last complete contribution in the window (1-based)
LORDS_WINDOW = [40.95, 8801.45]  # anchored onsets of contribution 1 and of the clipped contribution 104
LONDON_WINDOW = [0.0, 10899.098413]  # the whole London fixture
PINNED = {  # sha256 of the reference text, and of the file this script writes, when we built them
    "lords": (
        "7aacf05deb88a9a163198d43b5b36805350a5cc10cae8003b5788b775529f4d0",
        "09936a1d75462ad33577ba18f0bb38742688e947944075e3bfdce15904dc5221",
    ),
    "london": (
        "38804ee77b9844c02b69dd417b9c6dc9dd67774a43e46c091cfeacb54b4b11cc",
        "c913f9cb0c0b35b77436a291a82c642f02651a209af747151882fd71c5cae958",
    ),
}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SPEAKER_LABEL = re.compile(r"^[^:]+?(?:\bAM(?: \([^)]*\))?|\(Mayor of London\)):\s*")
LONDON_SOURCE = "https://www.london.gov.uk/about-us/londonassembly/meetings/ieListDocuments.aspx?CId=183&MId=8008"
HANSARD_SOURCE = "https://hansard.parliament.uk/lords/2026-09-16"
LABEL = "vs the edited official record (not verbatim)"


def clean_text(text: str) -> str:
    """Collapse spacing and remove editorial bracket insertions from the spoken record."""
    return re.sub(r"\s+", " ", re.sub(r"\[[^]]*\]", " ", text)).strip()


def docx_paragraphs(path: Path) -> list[str]:
    """Read body text in document order, excluding headers, footers and deleted text."""
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        parts = [node.text or "" for node in paragraph.iter(f"{WORD_NS}t")]
        paragraphs.append(re.sub(r"\s+", " ", "".join(parts)).strip())
    return paragraphs


def london_text(paths: list[Path]) -> str:
    """Keep labelled speech and its continuation paragraphs in both appendices."""
    speech = []
    for path in paths:
        in_speech = False
        for paragraph in docx_paragraphs(path):
            label = SPEAKER_LABEL.match(paragraph)
            if label:
                in_speech = True
                paragraph = paragraph[label.end() :]
            elif re.match(r"^\d{4}/\d+\s*[-–]", paragraph):
                in_speech = False
            if in_speech:
                speech.append(clean_text(paragraph))
    text = " ".join(part for part in speech if part)
    if not text:
        raise ValueError("No labelled speech found in London transcript documents")
    return text


def hansard_contributions(paths: list[Path]) -> list[str]:
    """Member contributions in sitting order; consecutive items by one member are one contribution."""
    turns: list[str] = []
    for path in paths:
        debate = json.loads(path.read_text(encoding="utf-8"))
        previous_member = None
        for item in sorted(debate["Items"], key=lambda value: value["OrderInSection"]):
            if item["ItemType"] != "Contribution":
                continue
            member = item.get("MemberId")
            if not member:
                previous_member = None
                continue
            text = clean_text(html.unescape(re.sub(r"<[^>]+>", " ", item["Value"])))
            if previous_member == member:
                turns[-1] += " " + text
            else:
                turns.append(text)
            previous_member = member
    return turns


def lords_reference(hansard_dir: Path) -> dict:
    first, last = LORDS_CONTRIBUTIONS
    turns = hansard_contributions([hansard_dir / f"debate-{item}.json" for item in HANSARD_DEBATES])
    if len(turns) < last + 1:
        raise ValueError("Hansard contributions do not cover the fixed complete-turn window")
    policy = "Fixed anchor window; exclude final clipped contribution; include intervening interjections."
    scope = "Edited Hansard omits some procedural speech and disfluencies; anchored boundaries are approximate."
    return {
        "text": " ".join(turns[first - 1 : last]),
        "window": LORDS_WINDOW,
        "source": HANSARD_SOURCE,
        "contribution_orders": [first, last],
        "window_policy": policy,
        "scope_limitation": scope,
    }


def london_reference(appendix_1: Path, appendix_2: Path) -> dict:
    policy = "Full recording; no manually audited boundaries for agenda items 3 and 4."
    scope = "Reference covers agenda items 3 and 4 only. Other procedural speech counts as insertions."
    return {
        "text": london_text([appendix_1, appendix_2]),
        "window": LONDON_WINDOW,
        "source": LONDON_SOURCE,
        "window_policy": policy,
        "scope_limitation": scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", choices=("lords", "london"))
    parser.add_argument("--hansard-dir", type=Path, help="lords: directory of debate-<ID>.json files")
    parser.add_argument("--appendix-1", type=Path, help="london: Appendix 1, oral update to the Mayor's report")
    parser.add_argument("--appendix-2", type=Path, help="london: Appendix 2, transcript of questions to the Mayor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.fixture == "lords":
        if not args.hansard_dir:
            parser.error("lords needs --hansard-dir")
        payload = lords_reference(args.hansard_dir)
    else:
        if not (args.appendix_1 and args.appendix_2):
            parser.error("london needs --appendix-1 and --appendix-2")
        payload = london_reference(args.appendix_1, args.appendix_2)
    payload["label"] = LABEL
    payload["sha256"] = hashlib.sha256(payload["text"].encode("utf-8")).hexdigest()
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.out.write_text(body, encoding="utf-8")
    text_pin, file_pin = PINNED[args.fixture]
    file_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    same = payload["sha256"] == text_pin and file_sha == file_pin
    print(
        f"{args.out}: {len(payload['text'].split())} whitespace-separated words, text sha256 {payload['sha256']}, "
        f"{'identical to ours' if same else 'DIFFERS from ours (the official text changed; WER is not comparable)'}"
    )
    return 0 if same else 3


if __name__ == "__main__":
    sys.exit(main())
