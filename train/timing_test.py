"""Timing test: 50 rounds of level 3 (SecAgg+ + noise_share_mod), n = 20, eps = 1, into --out.

    python -m train.timing_test --out DIR --T 120 [--lr 0.1 --C 1.0 --clipping-range 16.0]

Writes DIR/timing.json with the median round_time_s over rounds 2..50 (round 1 is the Ray
warm-up), the mean bytes_per_house, the machine info, and

    projected_total_s = median_round_time_s * T * 27

the projected wall time of the 27 level-3 runs (3 eps x 3 n x 3 neighbourhoods) at T rounds each.
--lr, --C and --clipping-range follow train.run's rule: accepted only while config.yaml has null.
The data can be the fake stub: the cost does not depend on the readings. The run is always fresh
(any earlier timing run in DIR is overwritten).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train.costs import running_cost
from train.env import machine_info
from train.run import run, run_files
from train.settings import check_clipping_range, resolve

LEVEL, EPS, N, K, ROUNDS, RUNS = 3, 1, 20, 0, 50, 27


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.timing_test", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=int, required=True, help="rounds per run for the projection")
    p.add_argument("--rounds", type=int, default=ROUNDS, help=argparse.SUPPRESS)  # tests only
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--C", type=float, default=None)
    p.add_argument("--clipping-range", type=float, default=None)
    a = p.parse_args(argv)
    out = Path(a.out)
    try:
        s = resolve(LEVEL, EPS, N, K, lr=a.lr, C=a.C, clipping_range=a.clipping_range,
                    out_dir=out, rounds=a.rounds)
        check_clipping_range(s)
    except (ValueError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for f in run_files(s):
        f.unlink(missing_ok=True)
    run(s)
    cost = running_cost(s.logs_dir / f"{s.run_name}.csv")
    result = {
        "level": LEVEL, "eps": EPS, "n": N, "nb": K, "rounds": s.T, "clipping_range": s.clipping_range,
        "median_round_time_s": cost["median_round_time_s"],
        "median_over_rounds": [cost["rounds"][0], cost["rounds"][-1]],
        "bytes_per_house": cost["bytes_per_house"],
        "projection": {"T": a.T, "runs": RUNS,
                       "projected_total_s": cost["median_round_time_s"] * a.T * RUNS,
                       "formula": "median_round_time_s * T * 27"},
        "env": machine_info(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
