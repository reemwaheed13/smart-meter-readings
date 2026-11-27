"""
score_all.py - score every attack output against its recording, and write the result tables.

    python -m eval.score_all --recordings recordings --attacks attack_outputs --out results

INPUTS
    recordings/L{level}_eps{eps}_n{n}_nb{k}_r{round}.pt   (README, "Recording format"; read with
        attacks.recording.load_recording, which checks every key and shape)
    attack_outputs/<same name>__{attack}.pt               (README, "Attack output format")
        attack = "tsinverse" or "inverter"; torch.save({"rebuilt": tensor}), scaled 0-1:
            levels 1-2:  [n, 4, 96]   (house i's 4 rebuilt windows, any order)
            level 3   :  [4n, 96]     (all rebuilt windows from the aggregate, any order)

OUTPUTS (in --out)
    per_house.csv   one row per house per recording per attack (all scores)
    per_day.csv     one row per day: home/away labels (for balanced accuracy)
    summary.csv     one row per (attack, level, eps, n, round): means, improvement over
                    blind guess, standard error, and whether protection holds
    summary.md      the same, as a table you can paste into the report
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

import data  # noqa: E402
from attacks.recording import load_recording  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.baseline import blind_guess  # noqa: E402


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_attack_output(path):
    """{"rebuilt": tensor} (weights_only, so a file from the Drive cannot run code)."""
    out = torch.load(path, map_location="cpu", weights_only=True)
    return out["rebuilt"] if isinstance(out, dict) else out


def score_one(rec, rebuilt, attack, blind_mode="weekday"):
    """Score one attack on one recording. Returns (house_rows, day_rows)."""
    level, n = int(rec["level"]), int(rec["n"])
    eps = float(rec.get("eps", 0) or 0)
    true = data.unscale(_np(rec["true_windows"]).astype(float))          # [n, 4, 96] kWh
    n_win = true.shape[1]
    blind = data.unscale(blind_guess(rec["dates"], blind_mode).astype(float))  # [4, 96]
    rebuilt = data.unscale(_np(rebuilt).astype(float))

    # pair rebuilt windows with true ones (the attacker doesn't know the order)
    if level in (1, 2):
        rebuilt = rebuilt.reshape(n, n_win, -1)
        matched = np.stack([M.match_windows(rebuilt[h], true[h]) for h in range(n)])
    else:  # level 3: windows from the sum could belong to any house
        flat = M.match_windows(rebuilt.reshape(n * n_win, -1), true.reshape(n * n_win, -1))
        matched = flat.reshape(n, n_win, -1)

    total_use = true.sum()
    house_rows, day_rows = [], []
    for h in range(n):
        s_att, s_bl, rise_a, rise_b, peak_a, peak_b, ok_a, ok_b = ([] for _ in range(8))
        for w in range(n_win):
            t, a, b = true[h, w], matched[h, w], blind[w]
            s_att.append(M.smape(a, t))
            s_bl.append(M.smape(b, t))
            ra, pa = M.routine_errors(a, t)
            rb, pb = M.routine_errors(b, t)
            rise_a.append(ra); rise_b.append(rb); peak_a.append(pa); peak_b.append(pb)
            lt, la, lb = M.away_labels(t), M.away_labels(a), M.away_labels(b)
            for d in range(2):
                ok_a.append(la[d] == lt[d]); ok_b.append(lb[d] == lt[d])
                day_rows.append(dict(attack=attack, level=level, eps=eps, n=n, nb=int(rec["nb"]),
                                     round=int(rec["round"]), house=rec["house_ids"][h],
                                     true_away=lt[d], attack_away=la[d], blind_away=lb[d]))
        house_rows.append(dict(
            attack=attack, level=level, eps=eps, n=n, nb=int(rec["nb"]), round=int(rec["round"]),
            house=rec["house_ids"][h],
            smape_attack=np.mean(s_att), smape_blind=np.mean(s_bl),
            rise_err_attack=np.mean(rise_a), rise_err_blind=np.mean(rise_b),
            peak_err_attack=np.mean(peak_a), peak_err_blind=np.mean(peak_b),
            away_acc_attack=np.mean(ok_a), away_acc_blind=np.mean(ok_b),
            # for the per-house exposure analysis
            share_of_use=true[h].sum() / total_use if total_use > 0 else np.nan,
            unusual_kwh=float(np.sqrt(((true[h] - blind) ** 2).mean())),
        ))
    return house_rows, day_rows


# (column prefix, higher is better?)
SCORES = [("smape", False), ("rise_err", False), ("peak_err", False), ("away_acc", True)]


def summarise(per_house, per_day):
    keys = ["attack", "level", "eps", "n", "round"]
    rows = []
    for key, g in per_house.groupby(keys):
        row = dict(zip(keys, key), houses=len(g))
        for name, higher in SCORES:
            t = M.paired_test(g[f"{name}_attack"], g[f"{name}_blind"], higher_is_better=higher)
            row[f"{name}_attack"] = g[f"{name}_attack"].mean()
            row[f"{name}_blind"] = g[f"{name}_blind"].mean()
            row[f"{name}_improvement"] = t["improvement"]
            row[f"{name}_se"] = t["se"]
            row[f"{name}_holds"] = t["holds"]
        d = per_day.merge(pd.DataFrame([dict(zip(keys, key))]), on=keys)
        row["away_balanced_acc_attack"] = M.balanced_accuracy(d["attack_away"], d["true_away"])
        row["away_balanced_acc_blind"] = M.balanced_accuracy(d["blind_away"], d["true_away"])
        row["protection_holds"] = all(row[f"{name}_holds"] for name, _ in SCORES)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def to_markdown(summary):
    cols = ["attack", "level", "eps", "n", "round", "smape_attack", "smape_blind",
            "rise_err_attack", "peak_err_attack", "away_acc_attack", "protection_holds"]
    head = ["Attack", "Level", "ε", "Houses", "Round", "sMAPE attack (%)", "sMAPE blind (%)",
            "Rise error (h)", "Peak error (h)", "Home/away acc.", "Protection holds"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary[cols].iterrows():
        eps = "none" if r["eps"] == 0 else f"{r['eps']:g}"
        cells = [r["attack"], str(r["level"]), eps, str(r["n"]), str(r["round"]),
                 f"{r['smape_attack']:.1f}", f"{r['smape_blind']:.1f}",
                 f"{r['rise_err_attack']:.2f}", f"{r['peak_err_attack']:.2f}",
                 f"{r['away_acc_attack']:.2f}", "yes" if r["protection_holds"] else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--attacks", default="attack_outputs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--blind", default="weekday", choices=["weekday", "same_date"])
    args = ap.parse_args()

    outs = sorted(glob.glob(os.path.join(args.attacks, "*__*.pt")))
    if not outs:
        sys.exit(f"No attack outputs found in {args.attacks}")
    all_h, all_d = [], []
    for path in outs:
        stem, attack = os.path.basename(path)[:-3].split("__")
        rec_path = os.path.join(args.recordings, stem + ".pt")
        if not os.path.exists(rec_path):
            print(f"skip {path}: no recording {rec_path}")
            continue
        rec = load_recording(rec_path)
        h, d = score_one(rec, load_attack_output(path), attack, args.blind)
        all_h += h
        all_d += d
        print(f"scored {stem} [{attack}]")

    os.makedirs(args.out, exist_ok=True)
    per_house, per_day = pd.DataFrame(all_h), pd.DataFrame(all_d)
    per_house.to_csv(os.path.join(args.out, "per_house.csv"), index=False)
    per_day.to_csv(os.path.join(args.out, "per_day.csv"), index=False)
    summary = summarise(per_house, per_day)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(to_markdown(summary))
    print(f"wrote {len(per_house)} house rows and {len(summary)} summary rows to {args.out}/")


if __name__ == "__main__":
    main()
