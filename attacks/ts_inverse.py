# SPDX-License-Identifier: CC-BY-SA-4.0
#
# This file contains code ported and adapted from TS-Inverse:
#   https://github.com/Capsar/ts-inverse  (Capsar, CC BY-SA 4.0)
#   Meijer et al., "TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series
#   Forecasting Models", IEEE SaTML 2025, arXiv:2503.20952.
# Ported: the attack loop (L1 gradient matching, Adam, sign of the input gradients, clamping every
# 2nd step, ReduceLROnPlateau), the periodicity / trend / quantile-bounds regularisers, the quantile
# inversion model (residual blocks) and its pinball-loss training. Changes are listed in README.md
# ("TS-Inverse: what is ported and what changed"). This file is licensed under CC BY-SA 4.0
# (https://creativecommons.org/licenses/by-sa/4.0/); see LICENSE-THIRD-PARTY.
"""TS-Inverse against one recording: python -m attacks.ts_inverse --recording PATH [PATH ...] --out DIR

Levels 1-2: the server sees one update per house (level 2: the gradient-space row g_hat converted back
from LocalDpMod). Each house's update is inverted into its 4 windows by optimising 4 fake windows so
that their update matches it:

    level 1:  local_update(fake)            vs  server_view[h]
    level 2:  clip(local_update(fake), C)   vs  server_view[h]

Level 3: server_view is the MEAN of the n protected updates (view_is_sum False; the sum is
server_view * n). 4n fake windows, 4 per fake house, are optimised so that

    mean_h clip(local_update(fake_h), C)    vs  server_view

which keeps the gradient term on the paper's single-gradient scale. The update is computed with
common.local_update(..., create_graph=True) at the recorded weights, so the matched quantity is
exactly the training gradient.

Loss (their code): sum_params |fake - observed|  +  lambda_P * periodicity(48)  +  lambda_T * trend
+ lambda_Q^obs * bounds(inputs) + lambda_Q^tar * bounds(targets), with the bounds from the quantile
inversion model, trained per recording on attacker-pool gradients at the recorded weights
(attacks/aux_gradients.py). All settings are in config.yaml, attacks.ts_inverse.

Every random draw (fake-window init, restarts, aux samples, inversion-model init and shuffling) is
seeded from the recording's run_seed (common.run_seed(n, nb)) and round, so reruns give identical
output. Resumable: a recording whose metrics.json exists with the same settings is skipped, and
finished houses / the trained inversion model are reused after an interruption.

Outputs, per recording, in DIR/<recording stem>/: metrics.json, recon.pt (the reconstructions),
inversion_model.pt, and the per-house parts.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

import torch
import torch.nn as nn

import common
from attacks import aux_gradients, scoring
from attacks.recording import load_recording

VERSION = "ts_inverse-1"
L_IN, L_OUT = common.INPUT_LEN, common.OUTPUT_LEN


# --------------------------------------------------------------------------- settings and seeds

def load_settings(overrides: dict | None = None) -> dict:
    """config.yaml attacks.ts_inverse, with overrides merged in (nested dicts merge key by key)."""
    s = copy.deepcopy(common.load_config()["attacks"]["ts_inverse"])
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(s.get(k), dict):
            s[k].update(v)
        else:
            s[k] = v
    return s


def num_steps(settings: dict, windows: int) -> int:
    """Their num_attack_steps = base_num_attack_steps * batch_size * _attack_step_multiplier. An
    explicit settings['steps'] (tests) wins."""
    if settings.get("steps") is not None:
        return int(settings["steps"])
    return int(settings["base_steps"]) * windows * int(settings["step_multiplier"])


def attack_seed(run_seed: int, rnd: int, house: int, restart: int, tag: str) -> int:
    """Seed for one random draw, from (run_seed, round, house index, restart, what it is for).
    house -1 means the whole recording (aux samples, inversion model)."""
    return zlib.crc32(f"ts_inverse|{run_seed}|{rnd}|{house}|{restart}|{tag}".encode())


# --------------------------------------------------------------------------- regularisers (ported)

def periodicity(seq: torch.Tensor, period: int) -> torch.Tensor:
    """Their periodicity_regularization, 'l1_mean': mean |S_t - S_{t+period}|. seq [K, 96]."""
    return (seq[:, :-period] - seq[:, period:]).abs().mean()


def trend_l1(seq: torch.Tensor) -> torch.Tensor:
    """Their trend_consistency_regularization, 'l1_mean': mean |S - trend(S)|, the least-squares line
    per sequence. As in their get_sequence_trend, the trend is computed without gradient."""
    with torch.no_grad():
        t = torch.arange(seq.shape[1], dtype=seq.dtype).view(1, -1)
        tc = t - t.mean()
        mean = seq.mean(dim=1, keepdim=True)
        beta = (tc * (seq - mean)).sum(dim=1, keepdim=True) / (tc ** 2).sum()
        trend = beta * tc + mean
    return (seq - trend).abs().mean()


def bounds(seq: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Quantile-bounds loss (paper Eq. 11; their out_of_bound_loss). seq [K, L]; q [L, Q] quantile
    predictions in increasing quantile order, the same for every sequence. For each pair
    (q_j, q_{Q-1-j}), j < Q/2: mean relu(S - upper) + mean relu(lower - S); the sum is halved, as in
    their code. Their code pairs the quantiles the other way round (upper = q_j), which adds only the
    band width, a constant, when the predicted quantiles are ordered; this is the paper's form."""
    loss = seq.new_zeros(())
    Q = q.shape[-1]
    for j in range(Q // 2):
        lower, upper = q[:, j].unsqueeze(0), q[:, Q - 1 - j].unsqueeze(0)
        loss = loss + torch.relu(seq - upper).mean() + torch.relu(lower - seq).mean()
    return loss / 2


# --------------------------------------------------------------------------- inversion model (ported)

class ResidualBlock(nn.Module):
    """Their ResidualBlock: FC -> BN -> ReLU -> dropout -> FC -> BN, plus a projected skip, ReLU."""

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.adapt = nn.Sequential()
        if in_features != out_features:
            self.adapt = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features))

    def forward(self, x):
        out = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return torch.relu(out + self.adapt(x))


class InversionModel(nn.Module):
    """Their ImprovedGradToInputNN_Quantile: gradient [P] -> quantiles of one window's input [48, Q]
    and target [48, Q], one branch each (inversion_batch_size 1: the same quantiles serve every
    window of the batch)."""

    def __init__(self, P: int, hidden: list[int], dropout: float, quantiles: list[float]):
        super().__init__()
        self.quantiles = list(quantiles)
        Q = len(self.quantiles)

        def branch():
            sizes = [P] + list(hidden)
            return nn.Sequential(*[ResidualBlock(sizes[i], sizes[i + 1], dropout) for i in range(len(hidden))])

        self.blocks_input, self.blocks_target = branch(), branch()
        self.fc_input = nn.Linear(hidden[-1], L_IN * Q)
        self.fc_target = nn.Linear(hidden[-1], L_OUT * Q)

    def forward(self, g: torch.Tensor):
        Q = len(self.quantiles)
        qi = self.fc_input(self.blocks_input(g)).view(g.shape[0], L_IN, Q)
        qt = self.fc_target(self.blocks_target(g)).view(g.shape[0], L_OUT, Q)
        return qi, qt


def pinball(pred: torch.Tensor, windows: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """Their pinball_loss over a batch: pred [S, 96, Q] (one prediction per sample, shared by its
    windows), windows [S, B, 96]. Mean over all of max((tau-1)e, tau*e), e = window - pred."""
    tau = torch.tensor(quantiles, dtype=pred.dtype)
    e = windows.unsqueeze(-1) - pred.unsqueeze(1)
    return torch.maximum((tau - 1) * e, tau * e).mean()


def train_inversion_model(grads: torch.Tensor, windows: torch.Tensor, settings: dict, seed: int) -> tuple[InversionModel, dict]:
    """Train the inversion model on aux samples (grads [S, P], windows [S, B, 96]): AdamW, pinball loss
    on inputs and targets together, ReduceLROnPlateau on the held-out loss, epochs + 1 passes as in
    their loop. Returns the model (eval mode) and a short training log."""
    cfg = settings["inversion_model"]
    S = grads.shape[0]
    n_val = max(1, int(round(S * float(cfg["val_fraction"])))) if S > 2 else 0
    tr, va = slice(0, S - n_val), slice(S - n_val, S)
    epochs, bs = int(cfg["epochs"]), int(cfg["batch_size"])
    with torch.random.fork_rng(devices=[]):          # init and dropout draw from torch's global RNG
        torch.manual_seed(seed)
        model = InversionModel(grads.shape[1], cfg["hidden"], float(cfg["dropout"]), cfg["quantiles"])
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(cfg["lr_decay_factor"]),
            patience=max(1, epochs // int(cfg["lr_patience_divisor"])))
        shuffle = torch.Generator().manual_seed(seed + 1)
        g_tr, w_tr = grads[tr], windows[tr]
        log = []
        for epoch in range(epochs + 1):
            model.train()
            order = torch.randperm(g_tr.shape[0], generator=shuffle)
            losses = []
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                if len(idx) < 2:                     # BatchNorm needs 2+ samples in training
                    continue
                qi, qt = model(g_tr[idx])
                loss = pinball(torch.cat([qi, qt], 1), w_tr[idx], model.quantiles)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            val = _eval_pinball(model, grads[va], windows[va]) if n_val else (sum(losses) / max(1, len(losses)))
            sched.step(val)
            log.append({"epoch": epoch, "train": sum(losses) / max(1, len(losses)), "val": val})
    model.eval()
    return model, {"epochs": epochs, "samples": S, "val_samples": n_val, "final_val_loss": log[-1]["val"]}


@torch.no_grad()
def _eval_pinball(model: InversionModel, grads: torch.Tensor, windows: torch.Tensor) -> float:
    model.eval()
    qi, qt = model(grads)
    return float(pinball(torch.cat([qi, qt], 1), windows, model.quantiles))


@torch.no_grad()
def predict_quantiles(model: InversionModel, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantile predictions for one observed view [P]: (inputs [48, Q], targets [48, Q])."""
    model.eval()
    qi, qt = model(view.unsqueeze(0))
    return qi[0], qt[0]


# --------------------------------------------------------------------------- the attack (ported loop)

def fake_view(model: nn.Module, windows: torch.Tensor, level: int, C: float | None, lr: float) -> torch.Tensor:
    """The view the fake windows [H, W, 96] would produce: level 1-2 (H = 1) the house's update,
    clipped at level 2; level 3 the mean over the H fake houses of clip(local_update, C)."""
    views = torch.stack([aux_gradients.house_view(model, windows[h], level, C, lr, create_graph=True)
                         for h in range(windows.shape[0])])
    return views.mean(0) if level == 3 else views[0]


def grad_match_loss(model, windows, target, level, C, lr) -> torch.Tensor:
    """Their 'l1' gradient loss: sum over parameters of |fake - observed| (= the flat L1 distance)."""
    return (fake_view(model, windows, level, C, lr) - target).abs().sum()


def regulariser(xin: torch.Tensor, xtar: torch.Tensor, settings: dict, q: tuple | None) -> torch.Tensor:
    seq = torch.cat([xin, xtar], 1)
    r = xin.new_zeros(())
    if settings["trend_weight"]:
        r = r + float(settings["trend_weight"]) * trend_l1(seq)
    if settings["periodicity_weight"]:
        r = r + float(settings["periodicity_weight"]) * periodicity(seq, L_OUT)
    if q is not None:
        if settings["bounds_weight_inputs"]:
            r = r + float(settings["bounds_weight_inputs"]) * bounds(xin, q[0])
        if settings["bounds_weight_targets"]:
            r = r + float(settings["bounds_weight_targets"]) * bounds(xtar, q[1])
    return r


def invert(model: nn.Module, target: torch.Tensor, level: int, C: float | None, lr: float, houses: int,
           windows_per_house: int, settings: dict, seed: int, quantiles: tuple | None = None) -> dict:
    """Optimise houses * windows_per_house fake windows against the observed view `target` [P].

    Their loop: init U(0, 1); Adam on (inputs, targets); total = L1 gradient distance + regularisers;
    the input gradient is replaced by its sign; after each step the windows are clamped to [0, 1] on
    every clamp_every-th step; ReduceLROnPlateau steps on the total. steps + 1 iterations, as in their
    range(0, num_attack_steps + 1). restarts > 1 keeps the restart with the lowest gradient loss.
    """
    K = houses * windows_per_house
    steps = num_steps(settings, K)
    log_every = max(1, steps // 200)
    best = None
    for restart in range(int(settings["restarts"])):
        g = torch.Generator().manual_seed(int(seed) + restart)
        xin = torch.rand(K, L_IN, generator=g).requires_grad_(True)
        xtar = torch.rand(K, L_OUT, generator=g).requires_grad_(True)
        opt = torch.optim.Adam([xin, xtar], lr=float(settings["lr"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(settings["lr_decay_factor"]),
            patience=max(1, steps // int(settings["lr_patience_divisor"])))
        trace = []
        for step in range(steps + 1):
            w = torch.cat([xin, xtar], 1).view(houses, windows_per_house, -1)
            gl = grad_match_loss(model, w, target, level, C, lr)
            total = gl + regulariser(xin, xtar, settings, quantiles)
            gin, gtar = torch.autograd.grad(total, [xin, xtar])
            if settings["sign_input_grads"]:
                gin = gin.sign()
            opt.zero_grad()
            xin.grad, xtar.grad = gin, gtar
            opt.step()
            if step % int(settings["clamp_every"]) == 0:
                with torch.no_grad():
                    xin.clamp_(0, 1)
                    xtar.clamp_(0, 1)
            sched.step(total.item())
            if step % log_every == 0 or step == steps:
                trace.append([step, gl.item(), total.item()])
        recon = torch.cat([xin, xtar], 1).detach().view(houses, windows_per_house, -1)
        final = float(grad_match_loss(model, recon, target, level, C, lr).detach())
        if best is None or final < best["grad_loss"]:
            best = {"recon": recon, "grad_loss": final, "trace": trace, "restart": restart, "steps": steps}
    return best


# --------------------------------------------------------------------------- recordings

def level3_views(rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean view, sum view) of a level-3 recording. Flower's SecAgg+ aggregate is the mean
    (view_is_sum False): the sum is server_view * n. A recording with view_is_sum True holds the sum."""
    sv, n = rec["server_view"], rec["n"]
    return (sv / n, sv) if rec["view_is_sum"] else (sv, sv * n)


def _model(rec: dict) -> nn.Module:
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    return m


def fingerprint(settings: dict, use_inversion: bool) -> dict:
    return {"version": VERSION, "inversion_model": bool(use_inversion), "settings": settings}


def _load_part(path: Path, fp: dict):
    if not path.exists():
        return None
    try:
        part = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    return part if json.loads(part.get("fingerprint", "null")) == fp else None


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def attack_recording(path: str | Path, out: str | Path, settings: dict, use_inversion: bool = True) -> dict:
    """Attack one recording and write DIR/<stem>/{metrics.json, recon.pt, ...}. Skips a recording whose
    metrics.json already holds the same settings. Returns the metrics."""
    path = Path(path)
    rec = load_recording(path)
    d = Path(out) / path.stem
    d.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(settings, use_inversion)
    fp_json = json.dumps(fp, sort_keys=True)
    mpath = d / "metrics.json"
    if mpath.exists():
        try:
            done = json.loads(mpath.read_text())
            if done.get("fingerprint") == json.loads(fp_json):
                print(f"[{path.stem}] complete in {d}, skipping", flush=True)
                return done
        except (ValueError, OSError):
            pass

    level, n, rnd, C, lr = rec["level"], rec["n"], rec["round"], rec["C"], rec["lr"]
    W = rec["true_windows"].shape[1]
    rs = common.run_seed(n, rec["nb"])
    model = _model(rec)
    C_match = None if level == 1 else C

    # Inversion model: one per recording (all houses share the recorded weights).
    inv, inv_log = None, None
    if use_inversion:
        ip = d / "inversion_model.pt"
        part = _load_part(ip, json.loads(fp_json))
        S = int(settings["aux_samples"])
        if part is None:
            print(f"[{path.stem}] aux samples ({S}) and inversion model", flush=True)
            grads, wins = aux_gradients.aux_samples(rec["global_model"], level, C, n, lr, S,
                                                    attack_seed(rs, rnd, -1, 0, "aux"), windows_per_house=W)
            inv, inv_log = train_inversion_model(grads, wins, settings, attack_seed(rs, rnd, -1, 0, "inversion_model"))
            _save({"fingerprint": fp_json, "state": inv.state_dict(), "log": json.dumps(inv_log)}, ip)
        else:
            cfg = settings["inversion_model"]
            inv = InversionModel(common.num_params(model), cfg["hidden"], float(cfg["dropout"]), cfg["quantiles"])
            inv.load_state_dict(part["state"])
            inv.eval()
            inv_log = json.loads(part["log"])

    if level in (1, 2):
        recon, attacks = torch.empty(n, W, rec["true_windows"].shape[2]), []
        for h in range(n):
            hp = d / f"house_{h}.pt"
            part = _load_part(hp, json.loads(fp_json))
            if part is None:
                target = rec["server_view"][h]
                q = predict_quantiles(inv, target) if inv is not None else None
                res = invert(model, target, level, C_match, lr, 1, W, settings, attack_seed(rs, rnd, h, 0, "init"), q)
                part = {"fingerprint": fp_json, "recon": res["recon"][0], "grad_loss": res["grad_loss"],
                        "trace": json.dumps(res["trace"]), "steps": res["steps"]}
                _save(part, hp)
                print(f"[{path.stem}] house {h + 1}/{n}: grad loss {res['grad_loss']:.4g}", flush=True)
            recon[h] = part["recon"]
            attacks.append({"house": h, "grad_loss": float(part["grad_loss"]), "steps": int(part["steps"]),
                            "trace": json.loads(part["trace"])})
        view_target = None
    else:
        mean_view, sum_view = level3_views(rec)
        hp = d / "level3.pt"
        part = _load_part(hp, json.loads(fp_json))
        if part is None:
            q = predict_quantiles(inv, mean_view) if inv is not None else None
            res = invert(model, mean_view, 3, C, lr, n, W, settings, attack_seed(rs, rnd, -1, 0, "init"), q)
            part = {"fingerprint": fp_json, "recon": res["recon"].reshape(n * W, -1), "grad_loss": res["grad_loss"],
                    "trace": json.dumps(res["trace"]), "steps": res["steps"]}
            _save(part, hp)
            print(f"[{path.stem}] level 3: grad loss {res['grad_loss']:.4g}", flush=True)
        recon = part["recon"]
        attacks = [{"house": None, "grad_loss": float(part["grad_loss"]), "steps": int(part["steps"]),
                    "trace": json.loads(part["trace"])}]
        view_target = (mean_view, sum_view)

    blind = aux_gradients.attacker_pool_mean(rec["dates"])
    scores = scoring.score(rec["true_windows"], recon, blind, rec["house_ids"], per_house_matching=level in (1, 2))
    metrics = {
        "recording": path.name, "level": level, "eps": rec["eps"], "n": n, "nb": rec["nb"], "round": rnd,
        "run_seed": rs, "house_ids": rec["house_ids"], "dates": rec["dates"],
        "fingerprint": json.loads(fp_json),
        "inversion_model": inv_log,
        "attack": attacks,
        "scores": scores,
    }
    if level == 3:
        mean_view, sum_view = view_target
        fake_sum = fake_view(model, recon.view(n, W, -1), 3, C, lr).detach() * n
        true_sum = fake_view(model, rec["true_windows"], 3, C, lr).detach() * n
        metrics["level3"] = {
            "view_is_sum": rec["view_is_sum"],
            "server_view_times_n": sum_view.tolist(),
            "sum_check": {   # L1 distances to server_view * n (true: noise + quantisation only)
                "reconstruction_l1": float((fake_sum - sum_view).abs().sum()),
                "true_windows_l1": float((true_sum - sum_view).abs().sum()),
                "server_view_times_n_l1_norm": float(sum_view.abs().sum()),
            },
        }
    if level == 1:
        ref = common.load_config()["attacks"]["reproduce"]["paper_fcn_london_b1"]
        metrics["paper_reference"] = {"table2_fcn_london_b1": ref,
                                      "note": "paper: 1 window per attack; here: 4 windows (one FedSGD step)"}
    _save({"recon": recon, "true_windows": rec["true_windows"], "matching": json.dumps(scores["matching"])},
          d / "recon.pt")
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=1))
    tmp.replace(mpath)
    m = scores["mean"]
    print(f"[{path.stem}] MAE {m['attack']['mae_kwh']:.4f} kWh (blind {m['blind']['mae_kwh']:.4f}), sMAPE in/tar "
          f"{m['attack']['smape_input']:.4g}/{m['attack']['smape_target']:.4g}", flush=True)
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m attacks.ts_inverse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", nargs="+", required=True, help="recording file(s)")
    p.add_argument("--out", required=True, help="output directory (one subdirectory per recording)")
    p.add_argument("--no-inversion-model", action="store_true",
                   help="TESTS ONLY: skip the inversion model and its bounds term (not a grid option)")
    p.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)          # tests only
    p.add_argument("--aux-samples", type=int, default=None, help=argparse.SUPPRESS)    # tests only
    p.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)         # tests only
    a = p.parse_args(argv)
    over = {}
    if a.steps is not None:
        over["steps"] = a.steps
    if a.aux_samples is not None:
        over["aux_samples"] = a.aux_samples
    if a.epochs is not None:
        over["inversion_model"] = {"epochs": a.epochs}
    settings = load_settings(over)
    use_inv = bool(settings["inversion_model"]["enabled"]) and not a.no_inversion_model
    for r in a.recording:
        attack_recording(r, a.out, settings, use_inv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
