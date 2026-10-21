# Nojoin ASR windowing: reproduction bundle

This bundle runs a Nojoin worker image CPU-only (or on a GPU, if you ask) through the product's own prep and
`transcribe_audio`, with no constant or setting changed, on any recording. It records the windows, every word with
its timestamps, the onnxruntime thread settings and the wall time, then scores the run and compares two runs in the
units of the write-up in Discussion #339:

- **VAD speech lost**, which needs no reference;
- **reference speech lost** (strict), against a word-timed reference you generate locally;
- **WER under two lenses**, against the official record, and **extra words**;
- **time** and, on a GPU, **peak VRAM**.

It ships code only: no audio, no transcripts, no official text. Every input comes from its public source.

## Status

Upstream 2.10.0 ([#449](https://github.com/Valtora/Nojoin/pull/449)) replaced the windowing this bundle was written
to measure. Every device now uses windows of at most 30 s, each closed at the quietest pause after its midpoint;
Parakeet loads fp32 on the CPU too (Canary stays int8); and a window that VAD muted to silence is skipped. On 2.10.0
and later:

- `report.json`'s `max_chunk_duration_s`, `quantization`, `snap_radius_s` and `windows_recomputed` read
  "unavailable": the engine dropped `max_chunk_duration_s()` and the snap radius, and `onnx_asr_quantization()` and
  `_chunk_boundaries()` changed signature.
- Strict loss and extra words read "unavailable" wherever a window is skipped, because the windows no longer tile
  the input (see Limits). All three #339 recordings skip at least one. VAD speech lost, WER and time still work.

The terms and results below describe the 2.7–2.9 windowing.

*Corrected 2026-10-21: added this Status section, and Limits now says the windows tile the input only up to 2.9.0.
The onnxruntime thread timings and thread-pool sizing now come from #429's controlled test, Seattle's speaker count is
marked as taken from the minutes, and the 2.7.0 re-run's time is 931.4 s with model load excluded, as in the sample
output (the 932.8 s included it). Sections 6 and 7 now say that the planned PR series was dropped when #449 shipped.*

## Terms

| Term | Meaning |
|---|---|
| Shipped | The released windowing: 240 s windows on CPU, 120 s on a GPU (the `gpu_asr_window_s` default), each cut snapped to the quietest point within 18 s of the ideal cut. |
| 20 s cap | The same windowing capped at 20 s, with each cut moved back to the latest muted silence, or else to the quietest point. |
| Backstop | Re-decodes VAD speech that came back with no words, up to 10 s per window. "Unlimited" has no per-window budget. |
| Cap + backstop | Both of the above. |
| 120 s, cuts moved back into silence | The cap's cut rule at the shipped 120 s GPU window. |
| 120 s silence cuts + unlimited backstop | 120 s windows with silence cuts, plus a backstop with no budget. |
| VAD speech lost | The share of the speech the VAD kept (the nonzero 10 ms bins of the muted file the engine transcribes) that no transcript segment covers. |
| Reference speech lost (strict) | The share of reference word time (an independent faster-whisper large-v3-turbo transcript) that no output word touches. An output word runs from its first subword token to its last token + 0.2 s, never past the next word or its window's end, widened by 0.1 s each side. "Runs ≥ 5 s" counts stretches of missed reference words, joined across gaps under 2 s, that last at least 5 s. |
| Frozen lens | Our pre-registered WER normalizer: NFKC, casefold, punctuation to space. "uh" counts as a word, and "twenty" never equals "20". |
| Normalized lens | Whisper's `EnglishTextNormalizer` (numbers, spelling, fillers, contractions), applied after the frozen lens's Unicode fold. It was made primary after the frozen results were seen, so it is post-hoc. |
| Extra words | Insertions where both runs have text, plus words placed in audio the VAD muted, as a share of reference words. The comparison reports the candidate's rate minus the baseline's, in points, with a limit of +1. |
| JC | "Jointly covered": time where both runs have text (both runs' output words, widened by 0.1 s, overlap). |

## Contents

| File | What it does | Runs on |
|---|---|---|
| `fetch.sh` | Downloads the three #339 recordings and the official texts; checks each sha256 against ours | host |
| `build_wer_reference.py` | Builds the Lords (Hansard) and London (GLA) WER references from the files you fetched | host |
| `whisper_words.py` | Builds the independent word-timed reference (faster-whisper large-v3-turbo) | host |
| `run.sh` | Runs one recording through one worker image; writes a run directory | host (Docker) |
| `drive.py`, `trace_calls.py` | Started by `run.sh` inside the image: prep, `transcribe_audio`, a passive trace of recognize() calls | container |
| `vram_monitor.py` | Started by `run.sh --gpu`: owned peak VRAM from the host | host |
| `score.py` | Scores one run directory | host |
| `compare.py` | Compares a candidate run with a baseline run of the same recording | host |
| `scoring/` | The scorers (coverage, strict loss, long gaps, both lenses, extra words) and the vendored Whisper normalizer | host |

## Prerequisites

- Docker, and the worker image(s) to test: a release (`ghcr.io/valtora/nojoin-worker:<tag>`), or one you build from a
  branch: `docker build -f docker/Dockerfile.worker -t nojoin-worker:<name> .` in a Nojoin checkout.
- The worker's model cache with the Parakeet model (and Canary, if you test it), passed as `--model-cache`. Your
  stack's `model_cache` volume works as is (`docker volume ls | grep model_cache`), or any directory laid out the same
  way (`huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx/...`). Runs are offline; add `--allow-network`
  once to download a missing model into a writable cache directory.
- Python 3.12 on the host for scoring: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
  (jiwer, rapidfuzz, regex, more-itertools, at the versions our numbers were scored with).
- For the word reference: `pip install -r requirements-whisper.txt` (faster-whisper) in any venv.
- For fetching: `curl`, `ffmpeg`, `sha256sum`, a current `yt-dlp`, and `node` (YouTube's JS challenge).
- About 300 MB of disk per run (each run keeps the prepared WAV).

## 1. Get the recordings

```sh
./fetch.sh ~/nojoin-339 all        # or: seattle | london | lords
```

| Fixture | Source | What `fetch.sh` gets |
|---|---|---|
| Seattle City Council, 14 Apr 2026 (2:14:18, 52 speakers per the minutes) | `https://video.seattle.gov/media/council/council_041426_2022629.mp4` | the MP4 as published |
| London Assembly, Mayor's Question Time, 17 Sep 2026 (3:01:39, 23 speakers) | `https://youtu.be/gJeQVOmCZjY` | `yt-dlp -f 140` (AAC m4a) |
| House of Lords, 16 Sep 2026, 15:07:00–17:37:00 BST (2:30:00, 53 speakers) | `https://parliamentlive.tv/Event/Index/1deca1f8-0b69-4626-8dbd-0d0554953860` | `yt-dlp -f 66` (audio only, AAC 64 kb/s), then `ffmpeg -ss 97 -t 9000 -c copy -map_metadata -1 -movflags +faststart`. The HLS playlist starts at 14:05:23Z with no discontinuity, so t = 97 s is 14:07:00Z |

It also fetches the London transcript appendices (agenda items 3 and 4) from `london.gov.uk` and the nine Hansard
debates of the Lords sitting from `hansard-api.parliament.uk`. Each file is checked against the sha256 we recorded.
A `DIFFERENT` line means the source changed, or, for the cut Lords file, that your ffmpeg may have muxed the same
audio differently. `prep.json` and `vad-intervals.json` (below) tell you whether the audio the engine sees matches
ours.

**Terms.** Seattle's council video is public record. London's YouTube recording is CC BY (attribute the London
Assembly). The parliamentlive.tv audio is not covered by the Open Parliament Licence: parliamentlive.tv recordings are
"for personal use only", and may be used "only to make a fair and accurate report of parliamentary proceedings". We
hold our copy only for computational analysis and publish aggregate scores only, never clips or ASR text. Please do
the same: fetch your own copy and keep it, and any transcript of it, on your machine. Hansard: Contains Parliamentary
information licensed under the Open Parliament Licence v3.0. The London transcript appendices are the GLA's, under its
terms. Like Hansard, they are fetched from the source and never shipped here.

## 2. Build the references

**WER references** (Lords and London; Seattle has no official text):

```sh
python build_wer_reference.py lords  --hansard-dir ~/nojoin-339/lords-2026-09-16/hansard --out refs/lords-wer.json
python build_wer_reference.py london \
  --appendix-1 ~/nojoin-339/london-mqt-2026-09-17/appendix-1-mayors-report.docx \
  --appendix-2 ~/nojoin-339/london-mqt-2026-09-17/appendix-2-questions.docx --out refs/london-wer.json
```

How each was trimmed (the script's docstring has the rules in full):
- **Lords, Hansard:** the member contributions of the nine debates in sitting order. Consecutive items by one member
  form one contribution. HTML and editorial `[brackets]` are removed. Contributions 1–103 are kept: every complete
  contribution in the scored window. The window is [40.95, 8801.45] s of the fixture clock, from the speech onset of
  contribution 1 to the onset of contribution 104, which the fixture's end clips. Hansard is edited, so it omits
  some procedural speech and disfluencies. The reference has 18,946 words under the frozen lens (18,920 normalized).
- **London, GLA:** Appendix 1 (oral update to the Mayor's report, agenda item 3) and Appendix 2 (transcript of
  questions to the Mayor, item 4). Kept: paragraphs that open with a speaker label (an Assembly Member's "… AM:", or
  "(Mayor of London):") and their continuation paragraphs. A printed question heading ("2026/1234 - …") ends a speech,
  and editorial `[brackets]` are removed. It is scored over the whole recording, so speech outside items 3 and 4
  (opening business, procedure) counts as insertions. 28,353 words under the frozen lens (28,264 normalized).

The script prints `identical to ours` when your text and file match the sha256 of the ones we scored with
(Lords text `7aacf05d…`, London `38804ee7…`). Otherwise it exits 3, and your WER is not comparable with ours, though
it is still comparable between your own runs.

**Word reference** (Lords, for reference speech lost):

```sh
python whisper_words.py ~/nojoin-339/lords-2026-09-16/lords-2026-09-16.m4a refs/lords-words.json            # CPU int8
python whisper_words.py ~/nojoin-339/lords-2026-09-16/lords-2026-09-16.m4a refs/lords-words.json --device cuda
```

Ours used faster-whisper large-v3-turbo (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`), English,
`word_timestamps=True`, `vad_filter=True`, `condition_on_previous_text=False`, other settings at their defaults
(beam 5), on the original m4a. It has 20,397 words for Lords. Another device, version or model snapshot moves word
times a little, so your strict numbers can differ slightly from ours. What matters is that main and the PR are scored
against the same reference. The file records its model, device, settings and library versions under `made_with`.
This file is ASR text of the Lords recording: keep it local.

## 3. Run an image

```sh
./run.sh --image ghcr.io/valtora/nojoin-worker:2.8.0 --audio ~/nojoin-339/lords-2026-09-16/lords-2026-09-16.m4a \
  --out runs/2.8.0-lords --cpus 6 --model-cache nojoin_model_cache
```

- **CPU (default).** Every GPU is hidden (`CUDA_VISIBLE_DEVICES=`), as on a CPU-only host. Add `--cpuset 2,4,6,8,10,12`
  to pin to specific host CPUs. Ours used 6 cores, one hardware thread per physical core, with `--cpus 6`.
- **GPU.** `--gpu cdi` adds `--device nvidia.com/gpu=all`, which is how our GPU runs were started (CDI, from the NVIDIA
  Container Toolkit). `--gpu docker` adds `--gpus all`, the classic NVIDIA runtime path. GPU runs also write
  `vram.json`: the peak of the GPU memory `nvidia-smi` attributes to the container's own processes, sampled from the
  host every 0.2 s. Check `report.json` → `gpu_is_present` and `sessions` (CUDA provider), because an image whose CUDA
  needs a newer driver than yours falls back to the CPU.
- **Shared machine.** `--lock /path/to/lockfile` holds `flock` on that file for the whole run, so runs that use the
  same lock never overlap. `--timeout S` kills a stuck run.
- **Other options.** `--backend canary` tests Canary. `--head-s 1200` transcribes only the first 20 minutes of the
  prepared audio. `--prepared` takes an earlier run's `prepared.wav` as is. `--env K=V` passes an environment variable
  to the container, for experiments (Nojoin itself sets no onnxruntime thread counts). `./run.sh --help` lists
  everything.
- **Progress.** `tail -f runs/2.8.0-lords/run.log` shows the product's own log ("window 12/40 done ...").

A full CPU run is about 10–20 minutes per recording on 6 cores, depending on the image: onnxruntime's default
thread pools make Parakeet take 2.35× as long as 6 threads, and Canary 3.6×
([#429](https://github.com/Valtora/Nojoin/issues/429)). A GPU run takes a few minutes, most of it the CPU prep (about
100 s for Lords).

### What a run writes

| File | Content |
|---|---|
| `prepared.wav` | The product's prep output, which the engine transcribes: `preprocess_audio_for_vad` (16 kHz mono) then `mute_non_speech_segments` (non-speech set to digital zero), as the pipeline's VAD stage does |
| `prep.json` | Hashes of the source and prepared files, frames, the VAD's speech duration, prep time |
| `vad-intervals.json` | The nonzero 10 ms bins of the transcribed file (the speech the VAD kept), and `pcm_sha256` of its decoded samples |
| `result.json` | `transcribe_audio`'s return value, verbatim: text, segments, words with timestamps |
| `calls.json` | Every recognize() call: its span on the source clock (`start_sample`, `end_sample`), `kind` (`initial` window, `recovery` re-decode inside a window, or `unplaced`), text, raw subword tokens, token times, wall time |
| `report.json` | Image environment (onnxruntime and package versions, providers, CPU affinity), the engine's derived settings (`max_chunk_duration_s`, quantization, snap radius), default config, windows, onnxruntime session threads and providers, `asr_wall_s` (transcription excluding model load), `load_wall_s`, peak RSS |
| `image.json`, `run.log`, `vram.json` | The image's id and digests, the full log, and GPU memory (GPU runs) |

The trace is passive. It wraps the model that the engine instance loads, records each recognize() call and returns
the library's own result unchanged. Each call is placed by exact sample match against the transcribed file, so no
knowledge of how the engine cuts is needed (the engine's PCM_16 chunk files round-trip sample-exactly). As a
cross-check, `windows_recomputed_match` re-runs the engine's own `_chunk_boundaries` on the input, where the image has
it.

**Is my input the same as yours?** For Lords, `prep.json` → `prepared_sha256` was
`14a45017db35d2f5955f959cdbbedb805cfb9fe19ed25c2e163ec07e6eb7d4ac` in both the 2.7.0 and 2.8.0 images. The decoded
samples (`vad-intervals.json` → `pcm_sha256`) were:

| Fixture | `pcm_sha256` of the prepared input | Checked in |
|---|---|---|
| Lords | `b141445d79d120a09ad4175f76b9dab165922b0e435aa94545cad8c4855942ba` | the 2.7.0 and 2.8.0 images |
| Seattle | `0c178db6664cad5a93e548b57bb1d193c9d946ebaf7eaf27d034f1429a6612a0` | our benchmark prep (the same product functions) |
| London | `70979df9458e445c34736a0e67086424cb90beeac7c55b433d40f266fafb3dd0` | our benchmark prep (the same product functions) |

## 4. Score a run

```sh
python score.py runs/2.8.0-lords --words refs/lords-words.json --wer-ref refs/lords-wer.json
python score.py runs/2.8.0-london --wer-ref refs/london-wer.json
python score.py runs/2.8.0-seattle
```

This prints a summary and writes `score.json` beside the run, with every number behind it and no text. With
`--words`, it also counts long gaps. A gap is a stretch of more than 20 s between consecutive output words. The count
covers the reference words inside the gaps and how many gaps end within 5 s after a window cut. For Canary, or a
`--head-s` slice with no official text, `--proxy` adds WER against the word reference's own text.

For 2.7.0 on Lords (CPU, 6 pinned cores), ours printed:

```text
runs/2.7.0-lords: parakeet, 12214 words, 40 windows (max 246.7 s), ASR 931.4 s, load 1.1 s
  VAD speech lost: 31.09 % (100 runs >= 5 s)
  reference speech lost (strict): 39.03 % (129 runs >= 5 s)
  gaps > 20 s: 49, holding 6,418 of 20,397 reference words (31.47 %); 13 end <= 5 s after a window cut, holding 3,216
  WER frozen 52.14 % / normalized 50.14 %
```

If your prepared input matches ours, expect the same windows, words and VAD speech lost. WER also matches when
`build_wer_reference.py` reported `identical to ours`. The strict and gap lines depend on your word reference, and the
times depend on your machine.

## 5. Compare two runs

```sh
python compare.py runs/main-lords runs/pr-lords --words refs/lords-words.json --wer-ref refs/lords-wer.json
```

The output is a Markdown table. This one is ours: GPU Lords, the shipped 120 s windowing against cap + backstop (one
run each). The rows after Windows (recognize calls, words, threads, device) are left out here:

| Metric | Baseline | Candidate | Candidate - baseline |
|---|---:|---:|---|
| VAD speech lost, % | 14.24 | 7.77 | -6.48 |
| Reference speech lost (strict), % (runs >= 5 s) | 6.43 (34) | 0.66 (3) | -5.78 |
| WER, frozen, % | 29.29 | 27.83 | -1.47 (within +1 pp) |
| WER, normalized, % | 26.03 | 22.73 | -3.30 (within +1 pp) |
| Extra words, frozen, % of reference words | 18.25 | 20.20 | +1.96 (fail; JC 4-gram -0.02; midpoint +1.62) |
| Extra words, normalized, % of reference words | 17.72 | 18.65 | +0.93 (pass; JC 4-gram -0.03; midpoint +0.80) |
| ASR time, s (model load excluded) | 36.5 | 42.5 | x1.16 |
| Peak VRAM, MiB (owned) | 4,844 | 3,596 | -1,248 |
| Windows (longest, s) | 84 (131.6) | 544 (20.0) |  |

How to read it:
- **Lower is better** on every loss, WER and extra-words row.
- **WER** gets a +1 pp limit, the same one the write-up uses.
- **Extra words** passes when the candidate's rate is at most +1 pp above the baseline's and its 4-gram repetition
  inside JC is at most +1 pp above too. Each word is placed at its start. The midpoint placement is a sensitivity
  check.
- **Insertions** are counted against the edited record, so a spoken filler the record leaves out counts as extra.
  That is why the frozen lens, which keeps fillers, is harsher.
- **Warnings** appear below the table: different inputs (prep differs between the images), unplaced calls, or windows
  that do not tile the input.

## 6. Your main against a PR branch, on the three #339 recordings

Run both images on one machine with the same `--cpus`/`--cpuset` (and the same `--gpu`), then compare:

```sh
D=~/nojoin-339; CACHE=nojoin_model_cache; LOCK=/tmp/nojoin-bench.lock
declare -A AUDIO=([lords]=$D/lords-2026-09-16/lords-2026-09-16.m4a
                  [seattle]=$D/seattle-council-2026-04-14/council_041426_2022629.mp4
                  [london]=$D/london-mqt-2026-09-17/london-mqt-2026-09-17.m4a)
for side in main=nojoin-worker:main pr=nojoin-worker:pr; do
  for fx in lords seattle london; do
    ./run.sh --image "${side#*=}" --audio "${AUDIO[$fx]}" --out "runs/${side%%=*}-$fx" \
      --cpus 6 --model-cache "$CACHE" --lock "$LOCK"
  done
done
python compare.py runs/main-lords   runs/pr-lords   --words refs/lords-words.json --wer-ref refs/lords-wer.json
python compare.py runs/main-london  runs/pr-london  --wer-ref refs/london-wer.json
python compare.py runs/main-seattle runs/pr-seattle
```

Add `--gpu cdi` (or `--gpu docker`) to both sides for the GPU column. Only the shipped rows can be reproduced from a
release. The other policies exist only in our experimental builds: we dropped the planned PR series once #449 shipped
the ASR-window work in 2.10.0.

## 7. What we got

**The shipped CPU default, from released images** (Lords, 6 pinned cores, no setting changed):

| | 2.7.0 | 2.8.0 |
|---|---:|---:|
| Windows | 40 | 40 (the same cuts) |
| Words (frozen lens) | 12,214 | 12,866 |
| VAD speech lost, % | 31.09 | 31.92 |
| Reference speech lost (strict), % | 39.03 ² | 36.31 ¹ |
| Gaps > 20 s: count, reference words inside | 49, 6,418 of 20,397 (31.47 %) | 43, 5,645 (27.68 %) |
| … of which end within 5 s after a cut | 13, holding 3,216 | 11, holding 2,452 |
| WER frozen / normalized, % | 52.14 / 50.14 | 50.28 / 48.11 |
| Transcription, s | 890.5 | 1,113.5 |
| onnxruntime, session threads | 1.20.2, 0 / 0 | 1.31.0, 0 / 0 |

¹ 2.8.0's text equals our 6-thread benchmark run word for word, and 36.31% is that run's score. 2.7.0 transcribes
the same samples with the same 40 cuts, but its words differ (sequence similarity 0.906): thread-count numerics.

² From this bundle's own run of 2.7.0 (below). The earlier 2.7.0 run did not record its calls, which strict loss needs.

**Speech lost, %, as in the write-up.** Lords is reference speech lost and Seattle and London are VAD speech lost.
CPU is int8 on 6 pinned cores. GPU is fp32 on one RTX 4070 Ti 12 GB, with an image built from upstream `0426af50`.

| Policy | CPU Lords | CPU Seattle | CPU London | GPU Lords | GPU Seattle | GPU London |
|---|---:|---:|---:|---:|---:|---:|
| Shipped: 240 s CPU, 120 s GPU | 36.31 | 15.07 | 21.48 | 6.43 | 9.76 | 5.74 |
| 120 s, cuts moved back into silence | | | | 8.02 | 8.97 | 6.05 |
| 120 s silence cuts + unlimited backstop | 4.18 | 10.94 | 11.33 | 0.56 | 7.13 | 4.57 |
| 20 s cap | 5.79 | 8.39 | 8.08 | 0.88 | 5.27 | 3.97 |
| 20 s cap + backstop | 2.74 | 7.23 | 6.79 | 0.66 | 5.19 | 3.54 |

Our CPU benchmark runs set onnxruntime to 6 intra-op threads and 1 inter-op thread. The product sets neither, so a
release's CPU text can differ slightly from ours, and its time differs a lot.

**How this bundle was checked.**
- `score.py` and `compare.py`, run on our benchmark runs' result and call files, reproduce every cell above but two.
  They also reproduce the write-up's GPU WER and extra-words figures:
  - Lords: WER 29.29 / 26.03 shipped against 27.83 / 22.73 with cap + backstop, extra words +0.93 normalized;
  - London: WER 26.28 / 16.83 against 28.32 / 17.65, extra words +0.51;
  - the 20 s cap: +0.81 / +0.55;
  - on CPU, cap + backstop: +3.26 / +1.15.
- The two exceptions are the CPU 20 s cap on Seattle and London (8.39, 8.08). Those runs were scored after re-mapping
  their recorded calls to upstream's last-word rule. No image produces that output itself, and none will now that
  the PR series is dropped (section 6).
- `run.sh` on 2.7.0 (Lords, CPU, 6 pinned cores, `--lock`) reproduces the earlier 2.7.0 run in the first table. The
  prepared input has the same sha256 (`14a45017…`), the 40 windows are the same, and all 794 segments match in
  text, start and end. Every score in that column follows: 12,214 words, VAD speech lost 31.09 % (100 runs ≥ 5 s),
  the same 49 gaps, WER 52.14 / 50.14. All 40 calls were placed on the input, the windows tile it, and they equal the
  windows recomputed from the engine's own rule. Transcription took 931.4 s (model load excluded), against 890.5 s
  the first time.

## Limits

- **Strict loss needs every call placed and the windows tiling the input.** That holds for every release up to
  2.9.0 and for a capped window. If a branch's windows skip audio (2.10.0 skips silent ones), or a call's audio is
  not an exact slice of the input, strict loss and extra words report "unavailable" rather than guess. VAD speech
  lost and WER still work.
- **The trace wraps the engine instance's `_get_model`.** If a branch renames it, the run is untraced
  (`report.json` → `traced: false`) and only VAD speech lost and WER are scored.
- **Timing is one run.** Repeat a run, or alternate main and PR, before reading small time differences. Keep the
  machine otherwise idle (`--lock`). Pin CPUs on both sides.
- **onnxruntime's defaults** create one pinned thread per physical core of the host, whatever the container's CPU
  limit, and log `pthread_setaffinity_np failed` inside a pinned container
  ([#429](https://github.com/Valtora/Nojoin/issues/429)). That is the product's behaviour, not the bundle's.
- **VRAM** comes from the host's `nvidia-smi` and `docker top`, so it needs a local Docker daemon that shares the
  host's PID view.
- **Memory.** `drive.py` keeps a 16-bit copy of the input to place calls: about 2 bytes per sample (290 MB for Lords).
- **Extra words** uses the baseline's VAD-muted audio for both runs. If the two inputs differ, `compare.py` warns.

## Licence

The code in this bundle is licensed under the GNU Affero General Public License v3.0 (`LICENSE`), the same licence as
Nojoin. Parts of it are adapted from Nojoin benchmark tooling under that licence. `scoring/whisper_normalizers/` is
vendored, byte-identical, from openai-whisper 20250625 and keeps its MIT licence (`scoring/whisper_normalizers/LICENSE`;
provenance in `PROVENANCE.txt` there).
