#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Peak GPU memory owned by one container's processes, sampled from the host until the container exits.

Usage: vram_monitor.py CONTAINER OUT.json [--interval S]
run.sh starts it for --gpu runs. Every interval it lists the container's host PIDs (docker top) and sums
nvidia-smi --query-compute-apps=pid,used_gpu_memory over them: the memory the run itself holds, not the card's
total, so other processes on the card do not count. Writes {"peak_owned_mib", "samples", ...}. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time


def container_pids(name: str) -> set[int] | None:
    found = subprocess.run(["docker", "top", name, "-eo", "pid"], capture_output=True, text=True, check=False)
    if found.returncode:
        return None
    return {int(x) for x in found.stdout.split()[1:] if x.isdigit()}


def compute_apps() -> list[tuple[int, str]]:
    query = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    rows = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    return [(int(pid), mib.strip()) for pid, mib in (row.split(",") for row in rows if row.strip())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("container")
    parser.add_argument("out")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--start-timeout", type=float, default=3600.0, help="give up if the container never starts")
    args = parser.parse_args()
    peak, samples, unreadable, seen, began = 0, 0, 0, False, time.monotonic()
    while True:
        pids = container_pids(args.container)
        if pids is None:
            if seen or time.monotonic() - began > args.start_timeout:
                break
            time.sleep(args.interval)
            continue
        seen = True
        owned = [mib for pid, mib in compute_apps() if pid in pids]
        unreadable += sum(not mib.isdecimal() for mib in owned)
        peak = max(peak, sum(int(mib) for mib in owned if mib.isdecimal()))
        samples += 1
        time.sleep(args.interval)
    record = {
        "peak_owned_mib": peak if samples else None,
        "samples": samples,
        "interval_s": args.interval,
        "unreadable_samples": unreadable,
        "method": "max over samples of the sum of nvidia-smi used_gpu_memory for the container's host PIDs",
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1)
    print(f"vram_monitor: peak {record['peak_owned_mib']} MiB over {samples} samples")


if __name__ == "__main__":
    main()
