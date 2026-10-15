#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run one recording through a Nojoin worker image's own prep and transcribe_audio (drive.py), with nothing changed.
#
# Usage: run.sh --image IMAGE --audio FILE --out DIR --cpus N --model-cache DIR_OR_VOLUME [options]
#   --image IMAGE        worker image, e.g. ghcr.io/valtora/nojoin-worker:2.8.0 or one you built from a branch
#   --audio FILE         the recording (any format the product accepts), mounted read-only
#   --out DIR            output directory (created); see README "What a run writes"
#   --cpus N             docker --cpus limit, e.g. 6
#   --model-cache SRC    host directory or docker volume holding the worker's model cache (huggingface/...), as the
#                        compose file's model_cache volume does; mounted read-only and used offline
# Options:
#   --cpuset LIST        pin to these host CPUs (docker --cpuset-cpus), e.g. 2,4,6,8,10,12
#   --gpu cdi|docker     run on the GPU: "cdi" adds --device nvidia.com/gpu=all (CDI), "docker" adds --gpus all.
#                        Without it every GPU is hidden (CUDA_VISIBLE_DEVICES=) and the run is CPU-only.
#                        GPU runs also sample owned peak VRAM from the host with nvidia-smi (vram.json).
#   --backend NAME       parakeet (default) or canary
#   --head-s S           transcribe only the first S seconds of the prepared audio (e.g. 1200 for a 20-min slice)
#   --prepared           FILE already is a prepared (VAD-muted 16 kHz) WAV, e.g. an earlier run's prepared.wav
#   --lock FILE          hold flock(FILE) for the run, so runs sharing a machine never overlap
#   --timeout S          kill the run after S seconds
#   --allow-network      give the container a network and a writable cache, to download missing models once
#   --env K=V            pass an extra environment variable (repeatable), e.g. for a thread-count experiment
#   --name NAME          container name (default nojoin-asr-repro-<pid>)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

IMAGE="" AUDIO="" OUT="" CPUS="" CPUSET="" GPU="" CACHE="" BACKEND=parakeet HEAD_S="" PREPARED="" LOCK=""
TIMEOUT="" NETWORK=none NAME=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --audio) AUDIO="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --cpuset) CPUSET="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --model-cache) CACHE="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --head-s) HEAD_S="$2"; shift 2 ;;
    --prepared) PREPARED=1; shift ;;
    --lock) LOCK="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --allow-network) NETWORK=bridge; shift ;;
    --env) EXTRA+=(-e "$2"); shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h | --help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
if [ -z "$IMAGE" ] || [ -z "$AUDIO" ] || [ -z "$OUT" ] || [ -z "$CPUS" ] || [ -z "$CACHE" ]; then usage; fi
[ -f "$AUDIO" ] || { echo "no such audio file: $AUDIO" >&2; exit 2; }
case "$GPU" in "" | cdi | docker) ;; *) echo "--gpu takes cdi or docker" >&2; exit 2 ;; esac
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
AUDIO_DIR="$(cd "$(dirname "$AUDIO")" && pwd)"
AUDIO_NAME="$(basename "$AUDIO")"
NAME="${NAME:-nojoin-asr-repro-$$}"
if [ -d "$CACHE" ]; then CACHE="$(cd "$CACHE" && pwd)"; fi  # else a docker volume name
CACHE_MODE=ro
OFFLINE=(-e HF_HUB_OFFLINE=1)
if [ "$NETWORK" != none ]; then CACHE_MODE=rw; OFFLINE=(); fi

DOCKER=(docker run --rm --init --name "$NAME" --network "$NETWORK" --cpus "$CPUS"
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1
  -e XDG_CACHE_HOME=/cache -e HF_HOME=/cache/huggingface ${OFFLINE[@]+"${OFFLINE[@]}"}
  -v "$CACHE:/cache:$CACHE_MODE" -v "$AUDIO_DIR:/in:ro" -v "$HERE:/bundle:ro" -v "$OUT:/out" -w /app)
if [ -n "$CPUSET" ]; then DOCKER+=(--cpuset-cpus "$CPUSET"); fi
case "$GPU" in
  "") DOCKER+=(-e CUDA_VISIBLE_DEVICES=) ;;
  cdi) DOCKER+=(--device nvidia.com/gpu=all) ;;
  docker) DOCKER+=(--gpus all) ;;
esac
DOCKER+=(${EXTRA[@]+"${EXTRA[@]}"} --entrypoint python "$IMAGE" /bundle/drive.py
  --audio "/in/$AUDIO_NAME" --out /out --backend "$BACKEND")
if [ -n "$PREPARED" ]; then DOCKER+=(--prepared); fi
if [ -n "$HEAD_S" ]; then DOCKER+=(--head-s "$HEAD_S"); fi

docker image inspect --format \
  '{"reference":"'"$IMAGE"'","id":{{json .Id}},"repo_digests":{{json .RepoDigests}},"created":{{json .Created}},"labels":{{json .Config.Labels}}}' \
  "$IMAGE" > "$OUT/image.json"
LOG="$OUT/run.log"
echo "# $(date -Is) ${DOCKER[*]}" > "$LOG"
echo "run log: $LOG"
if [ -n "$LOCK" ]; then
  exec 9>>"$LOCK"
  echo "# $(date -Is) waiting for $LOCK" >> "$LOG"
  flock 9
  echo "# $(date -Is) lock acquired" >> "$LOG"
fi
MONITOR=""
if [ -n "$GPU" ]; then
  python3 "$HERE/vram_monitor.py" "$NAME" "$OUT/vram.json" >> "$LOG" 2>&1 &
  MONITOR=$!
fi
RC=0
if [ -n "$TIMEOUT" ]; then
  timeout --signal=TERM "$TIMEOUT" "${DOCKER[@]}" >> "$LOG" 2>&1 || RC=$?
  if [ "$RC" -eq 124 ]; then
    echo "# TIMEOUT after ${TIMEOUT}s; killing $NAME" >> "$LOG"
    docker kill "$NAME" >> "$LOG" 2>&1 || true
  fi
else
  "${DOCKER[@]}" >> "$LOG" 2>&1 || RC=$?
fi
if [ -n "$MONITOR" ]; then wait "$MONITOR" || true; fi
echo "# $(date -Is) exit=$RC" >> "$LOG"
grep -E "DONE|Error|Traceback|exit=" "$LOG" | tail -n 3
exit "$RC"
