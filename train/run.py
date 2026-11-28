"""Entry point: python -m train.run --level 2 --eps 1 --n 5 --k 0 [--out DIR] [--lr 0.1 --T 10 --C 1.0 --clipping-range 8.0]

Launches a Flower simulation (flwr.simulation.run_simulation, Ray backend) with n clients.
--lr, --T, --C and --clipping-range are accepted only while the matching calibration value in
config.yaml is null.

--out DIR puts recordings/ and logs/ under DIR. Runs are resumable: a run whose recordings and
logs all exist in DIR, pass load_recording and logged the same clipping_range is skipped. A partial run (some of its files exist)
is deleted and rerun from scratch.

Level 3 with noise: before anything starts, the run refuses a clipping_range whose expected number
of cuts over the run, n*T*P*2*Phi(-clipping_range/(sigma/sqrt(n))), is above 0.01, and names the
smallest clipping_range that would pass (train.settings.check_clipping_range).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys

from flwr.simulation import run_simulation

from attacks.recording import load_recording
from train.app import backend_config, make_client_app, make_server_app
from train.recorder import recording_path, rounds_to_record
from train.settings import RunSettings, check_clipping_range, resolve


def norms_path(s: RunSettings):
    """Level 1's per-house update-norm log (logs/<run>_norms.csv); None at levels 2-3."""
    return s.logs_dir / f"{s.run_name}_norms.csv" if s.level == 1 else None


def run_files(s: RunSettings) -> list:
    """Every file a complete run writes: the recordings, the CSV log, the test JSON and, at level 1,
    the per-house update-norm log."""
    extra = [norms_path(s)] if s.level == 1 else []
    return ([recording_path(s, r) for r in rounds_to_record(s.T)]
            + [s.logs_dir / f"{s.run_name}.csv", s.logs_dir / f"{s.run_name}_test.json"] + extra)


def is_complete(s: RunSettings) -> bool:
    """True if every file of the run exists, every recording passes load_recording and matches the
    run, the CSV has rounds 1..T (at level 1 the norms log has n rows for each of them too), and the
    clipping_range logged in the test JSON equals the run's (None at levels 1-2; a JSON without the key counts as None, so an older level-3 run is redone)."""
    files = run_files(s)
    if not all(p.exists() for p in files):
        return False
    try:
        for r in rounds_to_record(s.T):
            rec = load_recording(recording_path(s, r))
            if (rec["level"], rec["eps"], rec["n"], rec["nb"], rec["round"]) != (s.level, s.eps, s.n, s.k, r):
                return False
        with open(s.logs_dir / f"{s.run_name}.csv", newline="") as f:
            rounds = [int(row["round"]) for row in csv.DictReader(f)]
        with open(s.logs_dir / f"{s.run_name}_test.json") as f:
            logged_cr = json.load(f).get("clipping_range")
        if s.level == 1:
            with open(norms_path(s), newline="") as f:
                per_round: dict[int, int] = {}
                for row in csv.DictReader(f):
                    float(row["update_norm"])
                    per_round[int(row["round"])] = per_round.get(int(row["round"]), 0) + 1
            if per_round != {r: s.n for r in range(1, s.T + 1)}:
                return False
    except (ValueError, KeyError, OSError, RuntimeError, pickle.UnpicklingError, EOFError):
        return False
    return rounds == list(range(1, s.T + 1)) and logged_cr == s.clipping_range


def run(s: RunSettings) -> None:
    check_clipping_range(s)   # before anything starts: refuses a clipping_range too close to the noise
    print(f"[{s.run_name}] houses={list(s.house_ids)} T={s.T} lr={s.lr} C={s.C} sigma={s.sigma} "
          f"run_seed={s.run_seed}", flush=True)
    run_simulation(
        server_app=make_server_app(s),
        client_app=make_client_app(s),
        num_supernodes=s.n,
        backend_config=backend_config(s),
    )
    if not is_complete(s):
        raise RuntimeError(f"[{s.run_name}] the simulation ended but the run's files are incomplete")


def run_resumable(s: RunSettings) -> bool:
    """Skip a complete run (returns False); otherwise delete any partial files and run (returns True).
    The pre-run clipping_range check comes first, so a refused run deletes nothing."""
    check_clipping_range(s)
    if is_complete(s):
        print(f"[{s.run_name}] complete in {s.logs_dir.parent}, skipping", flush=True)
        return False
    for p in run_files(s):
        if p.exists():
            print(f"[{s.run_name}] partial run: deleting {p}", flush=True)
            p.unlink()
    for p in s.recordings_dir.glob(f"{s.run_name}_r*.pt") if s.recordings_dir.exists() else []:
        p.unlink()
    run(s)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--eps", required=True, help="'none' or a number")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--out", default=None, help="output directory; recordings/ and logs/ go under it")
    p.add_argument("--lr", type=float, default=None, help="only while calibration.lr is null")
    p.add_argument("--T", type=int, default=None, help="only while calibration.T[n] is null")
    p.add_argument("--C", type=float, default=None, help="only while calibration.C is null (levels 2-3)")
    p.add_argument("--clipping-range", type=float, default=None,
                   help="only while calibration.clipping_range is null (level 3)")
    a = p.parse_args(argv)
    try:
        s = resolve(a.level, a.eps, a.n, a.k, lr=a.lr, T=a.T, C=a.C, clipping_range=a.clipping_range,
                    out_dir=a.out)
        check_clipping_range(s)
    except (ValueError, NotImplementedError) as e:   # includes ClippingRangeTooSmall
        print(f"error: {e}", file=sys.stderr)
        return 2
    run_resumable(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
