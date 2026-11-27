# SPDX-License-Identifier: CC-BY-SA-4.0
#
# Reproduces the London Smartmeter FCN setting of TS-Inverse:
#   https://github.com/Capsar/ts-inverse  (Capsar, CC BY-SA 4.0)
#   Meijer et al., "TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series
#   Forecasting Models", IEEE SaTML 2025, arXiv:2503.20952.
# It follows their experiment protocol (seeds, attacked model state, attack configuration from their
# notebooks) and uses the port in attacks/ts_inverse.py. Licensed under CC BY-SA 4.0
# (https://creativecommons.org/licenses/by-sa/4.0/); see LICENSE-THIRD-PARTY.
"""Batch-1 reproduction gate: python -m attacks.reproduce_ts_inverse --out DIR [--house ID]

Run in Colab once the real data.py (London Smartmeter) is in. It refuses the FAKE stub unless --allow-fake.

What their London FCN run attacks (read from their code, src/ts_inverse/workers/*.py):
  * the model: FCN_Predictor (48 -> 64 -> sigmoid -> 64 -> sigmoid -> 48), built right after
    set_seed(seed) with PyTorch's default init. Nothing draws from torch's global RNG in between (the
    data loaders use their own seeded generator), so the weights are those of torch.manual_seed(seed)
    followed by common.Model(), which builds the same three Linear layers in the same order.
  * the state: UNTRAINED. warmup_number_of_batches = 0, number_of_batches = 1, update_model = False:
    the attacked gradient is the first training batch's, at the freshly initialised weights.
  * the data: one training window (batch_size 1, a shuffled draw from the series' train split, stride
    24 h, 1 day in -> 1 day out, min-max scaled), the gradient of the mean MSE: common.local_update.
  * seeds 10, 43, 28, 80, 71 (Table II).
  * the attack that fills Table II's TS-Inverse rows: their 06_final_results_wandb notebook takes them
    from experiment "ts-inverse_batch1_without_target_reconst_12-6-2024", i.e. the config in
    05_ts_inverse_reconstruction_attack.ipynb: L1 gradient loss, Adam lr 0.01, 500 * 1 * 10 = 5000
    steps, ReduceLROnPlateau (0.1, patience steps // 10), sign of the input gradient, clamp to [0, 1]
    every 2nd step, U(0, 1) init, NO inversion model (num_learn_epochs 0) and all regulariser weights 0.

Metric: sMAPE on the PAPER SCALE (0-1), REPRODUCTION ONLY: mean of 2|a-b| / (|a|+|b|) on the 0-1 scaled
values, 0/0 counted as 0, separately for the input half and the target half, as TS-Inverse reports it
(paper_smape below). This is not the grid's score: eval/metrics.py's sMAPE is in % on kWh and the grid
is scored only by eval.score_all. Matching (4-window row) and the blind guess (--blind same_date) come
from eval/metrics.py and eval/baseline.py; MAE / RMSE are in kWh (data.unscale).

Rows reported (sMAPE mean +- std over the 5 seeds, std with ddof 1, next to Table II's 6.3e-06 / 2.5e-06):
  table2_config      1 window, their Table II configuration (above)
  ts_inverse_b1      1 window, our configured TS-Inverse (inversion model, lambda_P = lambda_T = 0.5,
                     bounds 1 / 0.1; config.yaml attacks.ts_inverse)
  ours_level1_b4     OUR SETTING: 4 windows (one FedSGD step, level 1), our configured TS-Inverse,
                     same seeds and untrained state

Window choice: a torch.Generator seeded with the seed picks the date(s) from the house's train dates
(their exact draw depends on their dataset length and loader, which our data cannot replicate).
The inversion model's aux samples come from the attacker pool (attacks/aux_gradients.py); theirs came
from the attacked series' own val/test splits.

Resumable: each (row, seed) is saved under DIR/parts and reused if its settings are unchanged.
Writes DIR/reproduction.json and prints the table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

import common
import data
from attacks import aux_gradients, ts_inverse
from eval.baseline import blind_guess
from eval.metrics import match_windows

ROWS = {
    "table2_config": {"windows": 1, "inversion_model": False,
                      "overrides": {"periodicity_weight": 0.0, "trend_weight": 0.0,
                                    "bounds_weight_inputs": 0.0, "bounds_weight_targets": 0.0},
                      "label": "their Table II config, 1 window (no inversion model, no regularisers)"},
    "ts_inverse_b1": {"windows": 1, "inversion_model": True, "overrides": {},
                      "label": "our configured TS-Inverse, 1 window"},
    "ours_level1_b4": {"windows": 4, "inversion_model": True, "overrides": {},
                       "label": "OUR SETTING: level 1, 4 windows (one FedSGD step)"},
}
METRICS = ["smape_input", "smape_target", "mae_kwh", "rmse_kwh"]
SMAPE_SCALE = "paper scale (0-1), reproduction only"
BLIND = "same_date"
LR = 1.0   # local_update returns the gradient; lr is not applied, so its value does not matter here


def _seed(seed: int, row: str, tag: str) -> int:
    return zlib.crc32(f"reproduce|{seed}|{row}|{tag}".encode())


def paper_smape(a, b) -> float:
    """sMAPE on the PAPER SCALE (0-1), REPRODUCTION ONLY: mean(2|a-b| / (|a|+|b|)) on 0-1 scaled values,
    0/0 counted as 0, a fraction in [0, 2] (Makridakis 1993; the definition TS-Inverse reports in
    Table II). Written here, not ported. The grid uses eval.metrics.smape (in %, on kWh) via eval.score_all."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    num, den = 2 * np.abs(a - b), np.abs(a) + np.abs(b)
    return float(np.divide(num, den, out=np.zeros_like(num), where=den > 0).mean())


def window_metrics(true, guess) -> dict:
    """true, guess: [K, 96] on the 0-1 scale, already paired row by row."""
    L = common.INPUT_LEN
    true, guess = np.asarray(true, float), np.asarray(guess, float)
    err = data.unscale(true) - data.unscale(guess)
    return {"smape_input": paper_smape(true[:, :L], guess[:, :L]),
            "smape_target": paper_smape(true[:, L:], guess[:, L:]),
            "mae_kwh": float(np.abs(err).mean()), "rmse_kwh": float(np.sqrt((err ** 2).mean()))}


def attacked_state(seed: int) -> torch.nn.Module:
    """Their attacked model: default PyTorch init right after torch.manual_seed(seed), untrained."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return common.Model()


def pick_dates(house: str, seed: int, windows: int) -> list[str]:
    dates = data.valid_dates([house], "train")
    g = torch.Generator().manual_seed(seed)
    return [dates[i] for i in sorted(torch.randperm(len(dates), generator=g)[:windows].tolist())]


def run_one(row: str, seed: int, house: str, settings: dict, use_inv: bool) -> dict:
    W = ROWS[row]["windows"]
    model = attacked_state(seed)
    dates = pick_dates(house, seed, W)
    true = data.get_windows([house], dates, "train")[0]              # [W, 96]
    target = common.local_update(model, true, LR)
    q, inv_log = None, None
    if use_inv:
        grads, wins = aux_gradients.aux_samples(model.state_dict(), 1, None, 1, LR, int(settings["aux_samples"]),
                                                _seed(seed, row, "aux"), windows_per_house=W)
        inv, inv_log = ts_inverse.train_inversion_model(grads, wins, settings, _seed(seed, row, "inversion_model"))
        q = ts_inverse.predict_quantiles(inv, target)
    res = ts_inverse.invert(model, target, 1, None, LR, 1, W, settings, _seed(seed, row, "init"), q)
    t = true.numpy()
    recon = match_windows(res["recon"][0].numpy(), t)               # the attacker does not know the order
    blind = match_windows(blind_guess(dates, BLIND), t)
    return {"row": row, "seed": seed, "house": house, "dates": dates, "steps": res["steps"],
            "grad_loss": res["grad_loss"], "trace": res["trace"], "inversion_model": inv_log,
            "attack": window_metrics(t, recon), "blind": window_metrics(t, blind), "recon": recon.tolist(),
            "true": true.tolist()}


def _mean_std(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m attacks.reproduce_ts_inverse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--house", default=None, help="house ID (default: the first house of neighbourhood n=5, k=0)")
    p.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    p.add_argument("--allow-fake", action="store_true", help=argparse.SUPPRESS)        # tests only
    p.add_argument("--seeds", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--aux-samples", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if getattr(data, "FAKE", False) and not a.allow_fake:
        print("error: data.py is the FAKE stub; the reproduction needs the real London data.py", file=sys.stderr)
        return 2
    cfg = common.load_config()["attacks"]["reproduce"]
    seeds = a.seeds or list(cfg["seeds"])
    house = a.house or data.get_neighbourhood(5, 0)[0]
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)

    test_over = {}
    if a.steps is not None:
        test_over["steps"] = a.steps
    if a.aux_samples is not None:
        test_over["aux_samples"] = a.aux_samples
    if a.epochs is not None:
        test_over["inversion_model"] = {"epochs": a.epochs}

    results = {}
    for row in a.rows:
        spec = ROWS[row]
        settings = ts_inverse.load_settings({**spec["overrides"], **test_over})
        use_inv = spec["inversion_model"] and bool(settings["inversion_model"]["enabled"])
        fp = {"version": ts_inverse.VERSION, "row": row, "house": house, "inversion_model": use_inv,
              "settings": settings}
        parts = []
        for seed in seeds:
            pp = out / "parts" / f"{row}_seed{seed}.json"
            part = None
            if pp.exists():
                try:
                    part = json.loads(pp.read_text())
                    part = part if part.get("fingerprint") == json.loads(json.dumps(fp)) else None
                except ValueError:
                    part = None
            if part is None:
                print(f"--- {row}, seed {seed}", flush=True)
                part = {**run_one(row, seed, house, settings, use_inv), "fingerprint": fp}
                tmp = pp.with_suffix(".tmp")
                tmp.write_text(json.dumps(part))
                tmp.replace(pp)
            parts.append(part)
        results[row] = {"label": spec["label"], "windows": spec["windows"], "seeds": seeds,
                        "steps": parts[0]["steps"],
                        "attack": {k: _mean_std([pt["attack"][k] for pt in parts]) for k in METRICS},
                        "blind": {k: _mean_std([pt["blind"][k] for pt in parts]) for k in METRICS}}
    summary = {"house": house, "paper_table2_fcn_london_b1": cfg["paper_fcn_london_b1"],
               "smape_scale": SMAPE_SCALE, "blind": BLIND, "rows": results,
               "data_is_fake": bool(getattr(data, "FAKE", False))}
    (out / "reproduction.json").write_text(json.dumps(summary, indent=1))

    ref = cfg["paper_fcn_london_b1"]
    print(f"\nTS-Inverse reproduction, house {house}, seeds {seeds}  (sMAPE: {SMAPE_SCALE}; mean +- std)")
    print(f"{'row':16s} {'input sMAPE':>24s} {'target sMAPE':>24s} {'MAE kWh':>10s} {'blind MAE':>10s}")
    print(f"{'paper Table II':16s} {ref['input_smape']:>24.2g} {ref['target_smape']:>24.2g}")
    for row, r in results.items():
        si, st = r["attack"]["smape_input"], r["attack"]["smape_target"]
        print(f"{row:16s} {si['mean']:>12.3g} +- {si['std']:<9.2g} {st['mean']:>12.3g} +- {st['std']:<9.2g} "
              f"{r['attack']['mae_kwh']['mean']:>10.4f} {r['blind']['mae_kwh']['mean']:>10.4f}   {r['label']}")
    print(f"\nwritten: {out / 'reproduction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
