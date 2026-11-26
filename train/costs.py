"""Running cost from a run's per-round CSV log.

Round 1 is logged but flagged is_warmup = True: its time includes Ray starting the client actors.
The reported running cost is the MEDIAN round_time_s over rounds 2..T (and the mean bytes_per_house
over the same rounds). A run with T = 1 has no non-warm-up round and therefore no running cost.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path


def running_cost(csv_path: str | Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["is_warmup"] != "True"]
    if not rows:
        raise ValueError(f"{csv_path}: no rounds after the warm-up round")
    return {
        "median_round_time_s": statistics.median(float(r["round_time_s"]) for r in rows),
        "bytes_per_house": statistics.fmean(float(r["bytes_per_house"]) for r in rows),
        "rounds": [int(r["round"]) for r in rows],
    }
