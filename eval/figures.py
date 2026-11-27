"""
figures.py - the main results figure: attack error vs number of houses,
one line per protection level and epsilon, one panel per attack.

    python -m eval.figures --results results [--round 1]

Lower sMAPE = attack rebuilt the curves better. The grey dashed line is blind guessing:
a line at or above it means the attack learned nothing useful.
Writes results/fig_attack_vs_houses_r{round}.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed order: same setting always gets the same colour and marker
SERIES = [
    ((1, 0.0), "Level 1, no protection", "#257de0", "o"),
    ((2, 1.0), "Level 2, local DP ε=1", "#f47230", "s"),
    ((2, 8.0), "Level 2, local DP ε=8", "#00b894", "^"),
    ((3, 0.0), "Level 3, SecAgg+ no noise", "#f2a600", "D"),
    ((3, 1.0), "Level 3, SecAgg+ ε=1", "#ee7fac", "v"),
    ((3, 8.0), "Level 3, SecAgg+ ε=8", "#008a17", "P"),
]
INK, QUIET, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def plot(summary, path, rnd):
    s = summary[summary["round"] == rnd]
    attacks = [a for a in ("inverter", "tsinverse") if a in set(s["attack"])]
    titles = {"inverter": "Learned inverter", "tsinverse": "TS-Inverse"}
    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 3.8), sharey=True, squeeze=False)
    for ax, attack in zip(axes[0], attacks):
        a = s[s["attack"] == attack]
        blind = a.groupby("n")["smape_blind"].mean()
        ax.plot(blind.index, blind.values, color=QUIET, linestyle="--", linewidth=1.5, label="Blind guess")
        for (level, eps), label, color, marker in SERIES:
            g = a[(a["level"] == level) & (a["eps"] == eps)].sort_values("n")
            if g.empty:
                continue
            ax.errorbar(g["n"], g["smape_attack"], yerr=2 * g["smape_se"].fillna(0), color=color,
                        marker=marker, markersize=6, linewidth=2, capsize=3,
                        markeredgecolor="white", markeredgewidth=0.8, label=label)
        ax.set_title(titles[attack], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Houses in the neighbourhood", color=INK)
        ax.set_xticks(sorted(a["n"].unique()))
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("Attack sMAPE (%), lower = more leakage", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"How well each attack rebuilds a house's curve (round {rnd}); bars = ±2 SE",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--round", type=int, default=None, help="default: every attacked round")
    args = ap.parse_args()
    summary = pd.read_csv(os.path.join(args.results, "summary.csv"))
    rounds = [args.round] if args.round else sorted(summary["round"].unique())
    for r in rounds:
        path = os.path.join(args.results, f"fig_attack_vs_houses_r{r}.png")
        plot(summary, path, r)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
