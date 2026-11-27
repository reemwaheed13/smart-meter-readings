"""
exposure.py - per-house exposure at level 3: WHICH houses leak most inside a sum, and why.

    python -m eval.exposure --results results

For every level-3 setting, it checks whether attack success (how much better than blind
guessing the attack rebuilt that house, in sMAPE points) goes up with:
    share_of_use  - the house's share of the neighbourhood's total use
    unusual_kwh   - how far the house's real curves are from the attacker pool's average day
using Spearman correlation (rho near +1 = strong link, near 0 = no link).

Writes results/exposure.csv and results/fig_exposure.png.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

BLUE, ORANGE = "#257de0", "#f47230"
INK, QUIET, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def analyse(per_house):
    g3 = per_house[per_house["level"] == 3].copy()
    g3["success"] = g3["smape_blind"] - g3["smape_attack"]
    rows = []
    for key, g in g3.groupby(["attack", "eps", "n", "round"]):
        row = dict(zip(["attack", "eps", "n", "round"], key), houses=len(g))
        for col in ("share_of_use", "unusual_kwh"):
            rho, p = spearmanr(g[col], g["success"]) if len(g) > 2 else (float("nan"),) * 2
            row[f"rho_{col}"], row[f"p_{col}"] = rho, p
        rows.append(row)
    return g3, pd.DataFrame(rows)


def plot(g3, path, rnd=None):
    """One row per epsilon (mixing them would blur everything), two measures side by side."""
    rnd = g3["round"].min() if rnd is None else rnd
    g = g3[g3["round"] == rnd]
    eps_values = sorted(g["eps"].unique())
    fig, axes = plt.subplots(len(eps_values), 2, figsize=(9, 2.9 * len(eps_values)),
                             sharey="row", squeeze=False)
    for row, eps in enumerate(eps_values):
        ge = g[g["eps"] == eps]
        for col_i, (col, label) in enumerate([("share_of_use", "Share of neighbourhood use"),
                                               ("unusual_kwh", "Distance from average day (kWh)")]):
            ax = axes[row][col_i]
            for attack, color, marker in [("inverter", BLUE, "o"), ("tsinverse", ORANGE, "s")]:
                a = ge[ge["attack"] == attack]
                if len(a):
                    ax.scatter(a[col], a["success"], s=22, color=color, marker=marker, alpha=0.8,
                               edgecolor="white", linewidth=0.8, label=attack)
            ax.axhline(0, color=QUIET, linewidth=1, linestyle="--")
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == len(eps_values) - 1:
                ax.set_xlabel(label, color=INK)
        name = "no noise" if eps == 0 else f"ε = {eps:g}"
        axes[row][0].set_ylabel(f"{name}\nbetter than blind\n(sMAPE points)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Level 3: which houses leak most inside the sum (round {rnd}); dashed = blind guess",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    per_house = pd.read_csv(os.path.join(args.results, "per_house.csv"))
    g3, table = analyse(per_house)
    if g3.empty:
        print("No level-3 rows yet.")
        return
    table.to_csv(os.path.join(args.results, "exposure.csv"), index=False)
    plot(g3, os.path.join(args.results, "fig_exposure.png"))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
