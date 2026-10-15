#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Fetch the three Discussion #339 recordings and the official texts behind the WER references, then compare each
# file with the sha256 we recorded. Nothing here is redistributed: every file comes from its public source.
#
# Usage: fetch.sh DATA_DIR [seattle|london|lords|all]
# Needs curl, ffmpeg, sha256sum and a current yt-dlp (YouTube and parliamentlive.tv break old releases; set
# YTDLP=/path/to/yt-dlp to choose one). YouTube also wants a JavaScript runtime: node is used when it is on PATH.
# Existing files are kept. A "DIFFERENT" line means the source changed since we fetched it (or, for the cut Lords
# file, your ffmpeg muxes differently); prep.json and vad-intervals.json then tell you whether the audio still is.
set -euo pipefail
DATA="${1:?usage: fetch.sh DATA_DIR [seattle|london|lords|all]}"
WHICH="${2:-all}"
YTDLP="${YTDLP:-yt-dlp}"
mkdir -p "$DATA"
DATA="$(cd "$DATA" && pwd)"

check() {  # file expected-sha256
  local got
  got="$(sha256sum "$1" | cut -d' ' -f1)"
  if [ "$got" = "$2" ]; then echo "ok         $1"; else echo "DIFFERENT  $1 (ours $2, yours $got)"; fi
}

get() {  # url destination expected-sha256
  if [ ! -s "$2" ]; then
    mkdir -p "$(dirname "$2")"
    curl -fL --retry 3 -A "Mozilla/5.0" -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
  check "$2" "$3"
}

ytdlp() {
  if command -v node > /dev/null; then "$YTDLP" --js-runtimes "node:$(command -v node)" "$@"; else "$YTDLP" "$@"; fi
}

seattle() {  # Seattle City Council, 14 Apr 2026: public record (Seattle Municipal Archives)
  get https://video.seattle.gov/media/council/council_041426_2022629.mp4 \
    "$DATA/seattle-council-2026-04-14/council_041426_2022629.mp4" \
    0b61659fa7215ef1f1dee97c33f4772d0bbcd8ac88106a49c4ab9b8538bd1d70
}

london() {  # London Assembly, Mayor's Question Time, 17 Sep 2026: YouTube, CC BY (attribute the London Assembly)
  local dir="$DATA/london-mqt-2026-09-17" docs=https://www.london.gov.uk/about-us/londonassembly/meetings/documents
  mkdir -p "$dir"
  [ -s "$dir/london-mqt-2026-09-17.m4a" ] || ytdlp --no-playlist -f 140 -o "$dir/london-mqt-2026-09-17.m4a" \
    https://youtu.be/gJeQVOmCZjY
  check "$dir/london-mqt-2026-09-17.m4a" 72f1bdc982fb0de1e48bbee985bf8aadd9c3936f8ff8a4294bd922e43cea9d50
  get "$docs/s123813/Minutes%20-%20Appendix%201%20-%20Oral%20Update%20to%20the%20Mayors%20Report.docx?CT=2" \
    "$dir/appendix-1-mayors-report.docx" 619b72343161a22f5694260f6cedde3e6035afb019b45141e33951f8349a5d0f
  get "$docs/s123816/Minutes%20-%20Appendix%202%20-%20Transcript%20of%20Questions%20to%20the%20Mayor.docx?CT=2" \
    "$dir/appendix-2-questions.docx" 4c75f2dd677a3e2c002eb21b2267152736b8fae172a95a36224bdac4c4a398e2
}

lords() {  # House of Lords, 16 Sep 2026, 15:07:00-17:37:00 BST: parliamentlive.tv audio (see the terms in README)
  local dir="$DATA/lords-2026-09-16" event=1deca1f8-0b69-4626-8dbd-0d0554953860
  local api=https://hansard-api.parliament.uk/debates/debate
  mkdir -p "$dir/hansard"
  if [ ! -s "$dir/lords-2026-09-16.m4a" ]; then
    # The audio-only rendition (format 66: AAC 64 kb/s, 48 kHz stereo) of the whole sitting, then the 2:30:00 window
    # from 14:07:00Z cut by stream copy. The HLS playlist starts at 14:05:23Z with no discontinuity: t = 97 s.
    ytdlp -N 8 -f 66 -o "$dir/full.m4a" "https://parliamentlive.tv/Event/Index/$event"
    ffmpeg -v error -ss 97 -i "$dir/full.m4a" -t 9000 -map 0:a -c copy -map_metadata -1 -movflags +faststart \
      -y "$dir/lords-2026-09-16.m4a"
    rm -f "$dir/full.m4a"
  fi
  check "$dir/lords-2026-09-16.m4a" 0e739922f51d8759948a81c59ff8135f7cb316788f160aaf27417fb9d29dc1eb
  while read -r id sha; do
    get "$api/$id.json" "$dir/hansard/debate-$id.json" "$sha"
  done << 'EOF'
0720AC06-4EE1-43FF-8E1E-49E1645A5D24 4908ad882b821bbe308a19aa652628ca2150c48acb1f28534ba0330c55941b84
837FFD01-30FB-4045-81DF-C7CA55AAE9D8 623b76bbdc35104c4e395769673600a9f4bddd4178af4c80763fdd8a504dfbd4
7BE42E5F-D56E-4E34-A456-32910E425F4E 269f78e5591d1411e8374474b1f85ac48d783d7c6ead9168af9375db5af72283
F287EEA9-AD27-400F-B27D-E06A143BC5D6 839906c8a99b4820ea7ce0dc51df34ccc17c6b8509ff87ed6906f3eedafb0d4c
2DAA5878-086E-4803-91AB-C1DA14715B28 0c7b78c1c7a025c702b9a540a51c34636656529943718ce235552159c704d1ef
E4E8845A-95FB-4C4A-99AD-979BD490FE63 63886f9e6ba2d0592f3330400fe7235a33b04909ca6b32a479e672245852f5c3
E53C08F4-2C0B-455B-BF42-5021FE91E903 2ca2142c2feea29982f4b36631aba030eec17d732261f1c762d188f91715dd3c
ABD1D9B6-9C05-423D-A0F2-60B1E41B641E a9ae3f7e0581277554542c0ac3e212e0aa718a1c425bfc69354dcc41c7f18ed2
E200C38F-7290-48DD-9475-E986D53C3523 54261b2fb5ec619971dc386b409a10f094f92c6c8b5de4c7334b012d2e13ac06
EOF
}

case "$WHICH" in
  seattle | london | lords) "$WHICH" ;;
  all) seattle; london; lords ;;
  *) echo "unknown fixture: $WHICH" >&2; exit 2 ;;
esac
