"""Calibration: pick lr, T[n], C, sigma[eps] and clipping_range from level-1 runs (README, "Calibration").

    python -m train.calibrate --out DIR

Every choice uses val MSE on the val split (the `val_mse` column of each run's CSV) or the level-1
per-house update norms (logs/<run>_norms.csv). The runs also write their usual test JSON, as every
train.run does, but calibration never reads it.

1. lr: level 1, n = calibration_procedure.lr.n, nb = 0, lr.rounds rounds per lr in lr.grid. The
   final val MSE is the mean over the last lr.final_rounds rounds (NaN/inf = diverged = +inf). Pick
   the lowest. If the pick is the lowest (highest) lr tried, run the next lr.extend_below
   (lr.extend_above) value and pick again, at most lr.max_extensions times in all.
2. T[n]: one level-1 run per n at that lr, T.max_rounds rounds. With m[t] = the best-so-far val MSE,
   T[n] = the first t with m[t] - m[t+window] < min_improvement * m[t]; max_rounds if none (hit_cap).
3. C: the median per-house update L2 norm over rounds 1..T[n] of the T runs, pooled over n.
4. sigma[eps] = common.sigma_for(eps, C, delta) for every eps of levels 2-3.
5. clipping_range = min_clipping_range(min n, T[min n], P, sigma[min eps]) + C, rounded up to 2
   significant figures; refused unless the pre-run check passes for every (n, eps) of level 3.

Each run goes in its own directory (DIR/lr_sweep/lr<lr>/, DIR/T_runs/n<n>/) because train.run's file
names do not carry lr or T. Runs are resumable through train.run.run_resumable: complete runs are
skipped, partial ones redone. DIR/calibration.json is rewritten at the end from the logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import common
from train.accountant import total_eps
from train.env import machine_info
from train.run import norms_path, run_resumable
from train.settings import (MAX_EXPECTED_CUTS, expected_clipping_cuts, min_clipping_range, parse_eps,
                            resolve)

RULES = {
    "lr": "level 1, n={n}, nb={nb}, {rounds} rounds per lr; final val MSE = mean of the last {final_rounds} "
          "rounds (NaN/inf = +inf); lowest wins; if the pick is the lowest (highest) lr tried, the next of "
          "{below} ({above}) is run and the pick redone, at most {max_ext} extensions",
    "T": "level 1, nb={nb}, one run per n at the chosen lr, {max_rounds} rounds; m[t] = best-so-far val MSE; "
         "T[n] = first t with m[t] - m[t+{window}] < {thr} * m[t]; {max_rounds} if none (hit_cap)",
    "C": "median per-house update L2 norm over rounds 1..T[n] of the T runs, all houses, pooled over n",
    "sigma": "common.sigma_for(eps, C, delta)",
    "clipping_range": "min_clipping_range(n={n}, T[{n}], P, sigma[{eps}]) + C, rounded up to 2 significant "
                      "figures; must pass the pre-run check for every level-3 (n, eps) with its T[n]",
}


# --------------------------------------------------------------------------- pure rules

def final_val_mse(curve: list[float], final_rounds: int) -> float:
    """Mean of the last final_rounds values; +inf if any is NaN/inf (the run diverged) or the curve is short."""
    tail = curve[-final_rounds:]
    if len(tail) < final_rounds or not all(math.isfinite(v) for v in tail):
        return math.inf
    return float(sum(tail) / len(tail))


def pick_lr(grid: list[float], extend_below: list[float], extend_above: list[float], max_extensions: int,
            final_of) -> dict:
    """Run final_of(lr) over the grid, extend at the edges (see module docstring) and pick the lowest.
    final_of(lr) -> final val MSE. Ties go to the smaller lr."""
    finals = {float(lr): float(final_of(float(lr))) for lr in grid}
    below, above = [float(x) for x in extend_below], [float(x) for x in extend_above]
    extensions = []

    def best():
        return min(sorted(finals), key=lambda lr: finals[lr])

    while len(extensions) < max_extensions:
        b = best()
        if b == min(finals) and below:
            nxt = below.pop(0)
        elif b == max(finals) and above:
            nxt = above.pop(0)
        else:
            break
        finals[nxt] = float(final_of(nxt))
        extensions.append(nxt)
    b = best()
    return {"lr": b, "tried": sorted(finals), "final_val_mse": {f"{k:g}": finals[k] for k in sorted(finals)},
            "extensions": extensions, "at_grid_edge": b in (min(finals), max(finals))}


def best_so_far(curve: list[float]) -> list[float]:
    """m[t] = min(curve[:t+1]), ignoring NaN (inf until a finite value appears)."""
    out, m = [], math.inf
    for v in curve:
        if math.isfinite(v) and v < m:
            m = v
        out.append(m)
    return out


def stopping_round(curve: list[float], window: int, min_improvement: float) -> dict:
    """T = the first round t (1-based) with m[t] - m[t+window] < min_improvement * m[t], m the
    best-so-far curve. If no t <= len(curve) - window qualifies, T = len(curve) and hit_cap = True."""
    m = best_so_far(curve)
    R = len(curve)
    for t in range(1, R - window + 1):
        a, b = m[t - 1], m[t - 1 + window]
        if math.isfinite(a) and a - b < min_improvement * a:
            return {"T": t, "hit_cap": False}
    return {"T": R, "hit_cap": True}


def median_norm(norms: list[float]) -> float:
    if not norms:
        raise ValueError("no update norms to take the median of")
    return float(np.median(np.asarray(norms, dtype=np.float64)))


def round_up_sig(x: float, digits: int = 2) -> float:
    """Round x > 0 up to `digits` significant figures (12.18 -> 13, 0.1234 -> 0.13, 13 -> 13)."""
    if x <= 0:
        raise ValueError(f"need x > 0, got {x}")
    e = math.floor(math.log10(x)) - digits + 1
    m = math.ceil(round(x / 10 ** e, 9))
    return round(m * 10 ** e, max(0, -e))


def choose_clipping_range(C: float, sigma: dict[float, float], T: dict[int, int], ns: list[int],
                          level3_eps: list[float], P: int) -> dict:
    """The clipping_range rule plus the all-(n, eps) check. Raises ValueError if any check fails."""
    n0, e0 = min(ns), min(level3_eps)
    base = min_clipping_range(n0, T[n0], P, sigma[e0])
    cr = round_up_sig(base + C, 2)
    checks = []
    for n in ns:
        for eps in level3_eps:
            cuts = expected_clipping_cuts(n, T[n], P, cr, sigma[eps])
            checks.append({"n": n, "eps": eps, "T": T[n], "expected_cuts": cuts, "ok": cuts <= MAX_EXPECTED_CUTS})
    bad = [c for c in checks if not c["ok"]]
    if bad:
        raise ValueError(f"clipping_range {cr} fails the pre-run check for {bad}")
    return {"clipping_range": cr, "from": {"n": n0, "eps": e0, "T": T[n0], "P": P, "sigma": sigma[e0],
                                           "min_clipping_range": base, "plus_C": C, "sum": base + C},
            "checks": checks}


# --------------------------------------------------------------------------- runs

def _read_val(s) -> list[float]:
    with open(s.logs_dir / f"{s.run_name}.csv", newline="") as f:
        return [float(r["val_mse"]) for r in csv.DictReader(f)]


def _read_norms(s, upto: int) -> list[float]:
    with open(norms_path(s), newline="") as f:
        return [float(r["update_norm"]) for r in csv.DictReader(f) if int(r["round"]) <= upto]


def _level1(n: int, nb: int, lr: float, T: int, out: Path, config_path):
    s = resolve(1, "none", n, nb, lr=lr, T=T, out_dir=out, config_path=config_path)
    run_resumable(s)
    return s


def refuse_if_calibrated(cfg: dict) -> None:
    cal, sig = cfg["calibration"], (cfg["dp"].get("sigma") or {})
    set_ = [k for k in ("lr", "C", "clipping_range") if cal.get(k) is not None]
    set_ += [f"T[{n}]" for n, v in (cal.get("T") or {}).items() if v is not None]
    set_ += [f"dp.sigma[{e}]" for e, v in sig.items() if v is not None]
    if set_:
        raise ValueError(f"config.yaml already has calibration values set ({', '.join(set_)}); "
                         "calibrate.py runs only while they are all null.")


def calibrate(out: str | Path, config_path: str | Path = "config.yaml") -> dict:
    out = Path(out)
    cfg = common.load_config(config_path)
    refuse_if_calibrated(cfg)
    proc, exp = cfg["calibration_procedure"], cfg["experiment"]
    if int(proc["level"]) != 1:
        raise ValueError("calibration_procedure.level must be 1")
    nb, pl, pt = int(proc["nb"]), proc["lr"], proc["T"]
    ns = [int(n) for n in exp["n"]]
    delta = float(cfg["dp"]["delta"])

    # 1. lr sweep (+ edge extensions)
    curves = {}

    def final_of(lr):
        s = _level1(int(pl["n"]), nb, lr, int(pl["rounds"]), out / "lr_sweep" / f"lr{lr:g}", config_path)
        curves[f"{lr:g}"] = _read_val(s)
        return final_val_mse(curves[f"{lr:g}"], int(pl["final_rounds"]))

    lr_pick = pick_lr(pl["grid"], pl["extend_below"], pl["extend_above"], int(pl["max_extensions"]), final_of)
    lr = lr_pick["lr"]
    print(f"[calibrate] lr = {lr:g} (tried {lr_pick['tried']}, at_grid_edge={lr_pick['at_grid_edge']})", flush=True)

    # 2. T[n]
    T, T_info, norms = {}, {}, []
    for n in ns:
        s = _level1(n, nb, lr, int(pt["max_rounds"]), out / "T_runs" / f"n{n}", config_path)
        curve = _read_val(s)
        stop = stopping_round(curve, int(pt["window"]), float(pt["min_improvement"]))
        T[n] = stop["T"]
        nrm = _read_norms(s, T[n])
        norms += nrm
        T_info[str(n)] = {**stop, "val_mse": curve, "best_so_far": best_so_far(curve), "num_norms": len(nrm)}
        print(f"[calibrate] T[{n}] = {T[n]} (hit_cap={stop['hit_cap']})", flush=True)

    # 3. C, 4. sigma
    C = median_norm(norms)
    q = np.quantile(np.asarray(norms), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
    eps_all = sorted({float(e) for e in exp["eps"]["level2"] + exp["eps"]["level3"] if e is not None})
    sigma = {e: common.sigma_for(e, C, delta) for e in eps_all}

    # 5. clipping_range
    l3_eps = sorted(float(parse_eps(e)) for e in exp["eps"]["level3"] if parse_eps(e) is not None)
    clip = choose_clipping_range(C, sigma, T, ns, l3_eps, common.num_params(common.Model()))
    cr = clip["clipping_range"]
    print(f"[calibrate] C = {C:.6g}, sigma = {sigma}, clipping_range = {cr:g}", flush=True)

    privacy = [{"n": n, "T": T[n], "eps_per_round": e, "sigma": sigma[e], "eps_total": total_eps(C, sigma[e], T[n], delta)}
               for n in ns for e in eps_all]
    result = {
        "values": {"lr": lr, "T": {str(n): T[n] for n in ns}, "C": C, "clipping_range": cr,
                   "sigma": {f"{e:g}": sigma[e] for e in eps_all}},
        "rules": {
            "lr": RULES["lr"].format(n=pl["n"], nb=nb, rounds=pl["rounds"], final_rounds=pl["final_rounds"],
                                     below=pl["extend_below"], above=pl["extend_above"], max_ext=pl["max_extensions"]),
            "T": RULES["T"].format(nb=nb, max_rounds=pt["max_rounds"], window=pt["window"], thr=pt["min_improvement"]),
            "C": RULES["C"], "sigma": RULES["sigma"],
            "clipping_range": RULES["clipping_range"].format(n=min(ns), eps=f"{min(l3_eps):g}"),
            "data": "val split only (val_mse) and level-1 update norms; the test split is never read",
        },
        "lr": {**lr_pick, "val_mse": curves},
        "T": T_info,
        "C": {"median": C, "num_norms": len(norms),
              "quantiles": dict(zip(["min", "q10", "q25", "median", "q75", "q90", "max"], q))},
        "sigma": {"delta": delta},
        "clipping_range": clip,
        "privacy_totals": privacy,
        "procedure": proc,
        "env": machine_info(),
    }
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "calibration.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out / "calibration.json")
    print(f"[calibrate] wrote {out / 'calibration.json'}", flush=True)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.calibrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory (runs and calibration.json)")
    p.add_argument("--config", default="config.yaml", help=argparse.SUPPRESS)  # tests only
    a = p.parse_args(argv)
    try:
        calibrate(a.out, a.config)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
