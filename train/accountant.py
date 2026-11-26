"""Privacy accountant for levels 2 and 3 (zCDP, Gaussian mechanism).

Model: Gaussian mechanism with L2 sensitivity C and noise std sigma, every house takes part in
every round (no subsampling), T rounds.

    rho per round = C^2 / (2 sigma^2)
    total rho     = T * rho per round                         (zCDP composes additively)
    eps_total     = rho + 2 sqrt(rho ln(1/delta))             (zCDP -> (eps, delta)-DP)

The same formula holds at levels 2 and 3:
- level 2: each house's own update is clipped to C and carries noise std sigma;
- level 3: each house adds noise std sigma/sqrt(n), so the SecAgg+ sum carries total std sigma,
  and one house changes the sum by at most C (its clipped update).
eps = None (level 3, SecAgg+ alone) means sigma = 0: no DP guarantee, eps_total = inf.

Caveat: sigma comes from the classic per-round Gaussian bound
sigma = C sqrt(2 ln(1.25/delta)) / eps, which is only proven for eps < 1. So a per-round eps of
8 is a *label* that sets the noise, not a guarantee. The zCDP total eps_total is the reported
guarantee.

    python -m train.accountant --out DIR [--C 1.0] [--T 10]
writes DIR/privacy_table.csv with one row per (level, per-round eps, T).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import common


def rho_per_round(C: float, sigma: float) -> float:
    """zCDP rho of one Gaussian-mechanism round: C^2 / (2 sigma^2). inf when sigma = 0."""
    if not sigma:
        return math.inf
    return C * C / (2.0 * sigma * sigma)


def total_eps(C: float, sigma: float, T: int, delta: float) -> float:
    """(eps, delta)-DP of T rounds: rho + 2 sqrt(rho ln(1/delta)), rho = T C^2 / (2 sigma^2)."""
    rho = T * rho_per_round(C, sigma)
    if math.isinf(rho):
        return math.inf
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def table(C: float, delta: float, T_values: list[int], eps_by_level: dict[int, list],
          sigma_by_eps: dict | None = None) -> list[dict]:
    """Rows {level, eps_per_round, sigma, T, eps_total}. sigma is sigma_by_eps[eps] if given (and not
    None), else common.sigma_for(eps, C, delta). eps None -> sigma 0, eps_total inf."""
    sigma_by_eps = sigma_by_eps or {}
    rows = []
    for level, eps_list in sorted(eps_by_level.items()):
        for eps in eps_list:
            s = sigma_by_eps.get(eps)
            sigma = float(s) if s is not None else common.sigma_for(eps, C, delta)
            for T in T_values:
                rows.append({"level": level, "eps_per_round": "none" if eps is None else f"{eps:g}",
                             "sigma": sigma, "T": int(T), "eps_total": total_eps(C, sigma, int(T), delta)})
    return rows


def write_table(path: str | Path, rows: list[dict]) -> Path:
    """Write the accountant table as CSV: level, eps_per_round (a label), sigma, T, eps_total."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "eps_per_round", "sigma", "T", "eps_total"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "sigma": f"{r['sigma']:.6g}", "eps_total": f"{r['eps_total']:.6g}"})
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.accountant", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory for privacy_table.csv")
    p.add_argument("--C", type=float, default=None, help="only while calibration.C is null")
    p.add_argument("--T", type=int, nargs="*", default=None, help="only while calibration.T is null")
    a = p.parse_args(argv)
    cfg = common.load_config()
    cal, dp, exp = cfg["calibration"], cfg["dp"], cfg["experiment"]
    from train.settings import _calibrated  # same null/override rule as train.run
    C = float(_calibrated("C", cal["C"], a.C))
    cfg_T = sorted({t for t in (cal["T"] or {}).values() if t is not None})
    T_values = _calibrated("T", cfg_T or None, a.T)
    rows = table(C, float(dp["delta"]), list(T_values),
                 {2: exp["eps"]["level2"], 3: exp["eps"]["level3"]}, dp.get("sigma"))
    print(f"wrote {write_table(Path(a.out) / 'privacy_table.csv', rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
