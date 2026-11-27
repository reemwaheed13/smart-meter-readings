"""Forecast cost: test error in kWh per level, eps and n, from the runs' logs.

    python -m eval.forecast_table --logs logs --out results

Reads every logs/L{level}_eps{eps}_n{n}_nb{k}_test.json (written by train.run after round T) and
writes results/forecast.csv and results/forecast.md: mean test MAE and RMSE over the
neighbourhoods k, with the standard error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

from eval.metrics import mean_se

NAME = re.compile(r"L(?P<level>\d)_eps(?P<eps>[\d.]+|none)_n(?P<n>\d+)_nb(?P<nb>\d+)_test\.json$")


def collect(logs: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(logs, "*_test.json"))):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            j = json.load(f)
        eps = m.group("eps")
        rows.append(dict(level=int(m.group("level")), eps=0.0 if eps in ("0", "none") else float(eps),
                         n=int(m.group("n")), nb=int(m.group("nb")),
                         mae=j["test_mae_kwh"], rmse=j["test_rmse_kwh"]))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (level, eps, n), g in df.groupby(["level", "eps", "n"]):
        mae, mae_se = mean_se(g["mae"])
        rmse, rmse_se = mean_se(g["rmse"])
        rows.append(dict(level=level, eps=eps, n=n, runs=len(g), mae_kwh=mae, mae_se=mae_se,
                         rmse_kwh=rmse, rmse_se=rmse_se))
    return pd.DataFrame(rows)


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["| Level | ε | Houses | Test MAE (kWh) | Test RMSE (kWh) |", "|---|---|---|---|---|"]
    def pm(m, se):
        return f"{m:.4f}" if pd.isna(se) else f"{m:.4f} ± {se:.4f}"

    for r in s.itertuples():
        eps = "none" if r.eps == 0 else f"{r.eps:g}"
        lines.append(f"| {int(r.level)} | {eps} | {int(r.n)} | {pm(r.mae_kwh, r.mae_se)} | {pm(r.rmse_kwh, r.rmse_se)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    df = collect(args.logs)
    if df.empty:
        raise SystemExit(f"No *_test.json files in {args.logs}")
    s = summarise(df)
    os.makedirs(args.out, exist_ok=True)
    s.to_csv(os.path.join(args.out, "forecast.csv"), index=False)
    with open(os.path.join(args.out, "forecast.md"), "w") as f:
        f.write(to_markdown(s))
    print(to_markdown(s))


if __name__ == "__main__":
    main()
