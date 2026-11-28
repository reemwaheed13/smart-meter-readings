"""Learned inverter (the second attack, our own code): python -m attacks.learned_inverter
    --recording PATH [PATH ...] --out attack_outputs --aux-cache DIR

A network that maps what the server sees to the windows behind it, retrained for every recording at
that recording's weights on its aux set (attacks/aux_gradients.py: attacker-pool samples with the
level's clipping and noise, the same set TS-Inverse uses, shared through the aux cache).

  input   the update [P]: levels 1-2 one house's row server_view[h], level 3 server_view (the mean)
  output  K windows of 96 readings in 0-1, any order: K = 4 at levels 1-2, K = 4n at level 3

Each output slot k starts from a PRIOR, the attacker pool's same-date mean window for its date
(eval.baseline.blind_guess(dates, "same_date"); at level 3 the 4 date priors repeated for the n
houses), and the network predicts a correction on top:

    out_k = clamp(prior_k + (2 * sigmoid(z_k) - 1), 0, 1),     z = head(trunk(update))

The correction 2*sigmoid(z) - 1 lies in (-1, 1) and is exactly 0 at z = 0. The head starts at zero, so
an untrained inverter outputs exactly the blind guess, and anything it does better is learned from the
update: leakage.

Network: a residual MLP trunk (blocks of Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm
plus a projected skip, then ReLU; widths 768, 512, dropout 0.05), the same architecture as TS-Inverse's
inversion model, written here independently. Then one Linear head to K * 96.

Loss: MSE after Hungarian matching of the K outputs to the K true windows inside each sample (the
update does not say which window is which; the matching is computed without gradient, on squared error).
Training: AdamW 1e-3, batches of 32, 75 epochs, ReduceLROnPlateau on the held-out loss (factor 0.1,
patience epochs // 10), the last 10 % of the aux set held out to pick the best epoch. Epoch 0 (the
untrained inverter = the blind guess) is a candidate too. No other tuning. Settings: config.yaml,
attacks.learned_inverter.

Every random draw is seeded from (common.run_seed(n, nb), round, house index, purpose); the model is
trained once per recording, so house = -1 for all of them. The aux set's seed is
aux_gradients.aux_seed(run_seed, round), the same as TS-Inverse's.

Outputs, per recording, in DIR (README, "Attack output format"):
    DIR/<recording stem>__inverter.pt     {"rebuilt": tensor}, 0-1: [n, 4, 96] at levels 1-2, [4n, 96] at
                                          level 3. Scored by eval.score_all.
    DIR/<recording stem>__inverter.json   diagnostics only (train / held-out loss per epoch, best epoch,
                                          seeds, the aux set used). No scores.
Resumable: a recording whose outputs exist with the same settings (and aux key) is skipped; otherwise
it is redone.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import common
from attacks import aux_gradients
from attacks.recording import load_recording
from eval.baseline import blind_guess

VERSION = "learned_inverter-1"
ATTACK = "inverter"       # the attack name in the output file name (eval.score_all)
WINDOW = 2 * common.INPUT_LEN


# --------------------------------------------------------------------------- settings and seeds

def load_settings(overrides: dict | None = None) -> dict:
    s = copy.deepcopy(common.load_config()["attacks"]["learned_inverter"])
    s.update(overrides or {})
    return s


def inverter_seed(run_seed: int, rnd: int, house: int, purpose: str) -> int:
    """Seed for one random draw, from (run_seed, round, house index, purpose). house -1 = the recording."""
    return zlib.crc32(f"learned_inverter|{run_seed}|{rnd}|{house}|{purpose}".encode())


# --------------------------------------------------------------------------- network

class Block(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm, plus a (projected) skip, then ReLU."""

    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d_out, d_out), nn.BatchNorm1d(d_out))
        self.skip = nn.Identity() if d_in == d_out else nn.Sequential(nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class Inverter(nn.Module):
    """update [B, P], prior [B, K, 96] -> windows [B, K, 96] in 0-1 (= prior while the head is zero)."""

    def __init__(self, P: int, K: int, hidden: list[int], dropout: float):
        super().__init__()
        sizes = [P] + list(hidden)
        self.trunk = nn.Sequential(*[Block(sizes[i], sizes[i + 1], dropout) for i in range(len(hidden))])
        self.head = nn.Linear(hidden[-1], K * WINDOW)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.K = K

    def forward(self, update: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        z = self.head(self.trunk(update)).view(update.shape[0], self.K, WINDOW)
        return torch.clamp(prior + (2 * torch.sigmoid(z) - 1), 0.0, 1.0)   # brackets: exact prior at z = 0


def matched_mse(out: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """MSE after pairing, per sample, each output window with one true window (Hungarian on squared
    error, no gradient through the matching). out, true: [B, K, 96]."""
    with torch.no_grad():
        cost = ((out[:, :, None, :] - true[:, None, :, :]) ** 2).mean(-1).double().cpu().numpy()   # [B, K, K]
    perm = np.empty(cost.shape[:2], dtype=np.int64)
    for b in range(cost.shape[0]):
        rows, cols = linear_sum_assignment(cost[b])
        perm[b, rows] = cols                                  # out[b, k] <-> true[b, perm[b, k]]
    idx = torch.from_numpy(perm).unsqueeze(-1).expand(-1, -1, true.shape[-1])
    return ((out - torch.gather(true, 1, idx)) ** 2).mean()


# --------------------------------------------------------------------------- priors

def priors(dates: list[str], houses: int, mode: str) -> torch.Tensor:
    """[houses * len(dates), 96]: the blind guess for each date, repeated for each house (house-major,
    the same layout as the aux windows)."""
    return torch.from_numpy(blind_guess(dates, mode)).repeat(houses, 1)


def aux_priors(dates: list[list[str]], houses: int, mode: str) -> torch.Tensor:
    """[S, houses * W, 96] for the aux samples' dates."""
    uniq = sorted({d for ds in dates for d in ds})
    by_date = dict(zip(uniq, torch.from_numpy(blind_guess(uniq, mode))))
    return torch.stack([torch.stack([by_date[d] for d in ds]).repeat(houses, 1) for ds in dates])


# --------------------------------------------------------------------------- training

@torch.no_grad()
def _eval_loss(model: Inverter, g: torch.Tensor, p: torch.Tensor, w: torch.Tensor, bs: int) -> float:
    model.eval()
    tot = 0.0
    for i in range(0, len(g), bs):
        tot += float(matched_mse(model(g[i:i + bs], p[i:i + bs]), w[i:i + bs])) * len(g[i:i + bs])
    return tot / len(g)


def train(grads: torch.Tensor, prior: torch.Tensor, windows: torch.Tensor, settings: dict,
          init_seed: int, shuffle_seed: int) -> tuple[Inverter, dict]:
    """Train on aux samples: grads [S, P], prior [S, K, 96], windows [S, K, 96]. Returns the best model
    (eval mode) by held-out matched MSE, epoch 0 = untrained included, and the training log."""
    S, K = grads.shape[0], windows.shape[1]
    n_val = max(1, int(round(S * float(settings["val_fraction"])))) if S > 2 else 0
    tr, va = slice(0, S - n_val), slice(S - n_val, S)
    epochs, bs = int(settings["epochs"]), int(settings["batch_size"])
    with torch.random.fork_rng(devices=[]):          # init and dropout draw from torch's global RNG
        torch.manual_seed(init_seed)
        model = Inverter(grads.shape[1], K, settings["hidden"], float(settings["dropout"]))
        opt = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(settings["lr_decay_factor"]),
            patience=max(1, epochs // int(settings["lr_patience_divisor"])))
        shuffle = torch.Generator().manual_seed(shuffle_seed)
        g_tr, p_tr, w_tr = grads[tr], prior[tr], windows[tr]
        g_va, p_va, w_va = grads[va], prior[va], windows[va]

        def held_out():
            return _eval_loss(model, g_va, p_va, w_va, bs) if n_val else _eval_loss(model, g_tr, p_tr, w_tr, bs)

        initial = held_out()
        best = {"epoch": 0, "val": initial, "state": copy.deepcopy(model.state_dict())}
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(g_tr.shape[0], generator=shuffle)
            losses = []
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                if len(idx) < 2:                     # BatchNorm needs 2+ samples in training
                    continue
                loss = matched_mse(model(g_tr[idx], p_tr[idx]), w_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            val = held_out()
            sched.step(val)
            history.append({"epoch": epoch, "train": sum(losses) / max(1, len(losses)), "val": val,
                            "lr": opt.param_groups[0]["lr"]})
            if val < best["val"]:
                best = {"epoch": epoch, "val": val, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"samples": S, "val_samples": n_val, "epochs": epochs, "initial_val_loss": initial,
                   "best_epoch": best["epoch"], "best_val_loss": best["val"], "history": history}


# --------------------------------------------------------------------------- one recording

def output_paths(recording: str | Path, out: str | Path) -> tuple[Path, Path]:
    stem, out = Path(recording).stem, Path(out)
    return out / f"{stem}__{ATTACK}.pt", out / f"{stem}__{ATTACK}.json"


@torch.no_grad()
def rebuild(model: Inverter, rec: dict, mode: str) -> torch.Tensor:
    """Apply the trained inverter to the recording: [n, 4, 96] at levels 1-2, [4n, 96] at level 3."""
    model.eval()
    level, n, W = rec["level"], rec["n"], rec["true_windows"].shape[1]
    if level in (1, 2):
        p = priors(rec["dates"], 1, mode).unsqueeze(0).expand(n, -1, -1)
        return model(rec["server_view"].float(), p).reshape(n, W, WINDOW)
    p = priors(rec["dates"], n, mode).unsqueeze(0)
    return model(rec["server_view"].float().unsqueeze(0), p)[0]


def attack_recording(path: str | Path, out: str | Path, settings: dict, aux_cache: str | Path | None = None) -> dict:
    """Train the inverter on the recording's aux set and write DIR/<stem>__inverter.{pt,json}. Skips a
    recording whose outputs exist with the same settings. Returns the diagnostics."""
    path = Path(path)
    rec = load_recording(path)
    aux_cache = Path(out) / "_aux_cache" if aux_cache is None else Path(aux_cache)
    ppath, dpath = output_paths(path, out)
    level, n, rnd, W = rec["level"], rec["n"], rec["round"], rec["true_windows"].shape[1]
    rs = common.run_seed(n, rec["nb"])
    S, mode = int(settings["aux_samples"]), settings["blind"]
    H = n if level == 3 else 1
    aux_key = aux_gradients.cache_key(path, level, rec["C"], rec["sigma"], n, S, aux_gradients.aux_seed(rs, rnd),
                                      W, rec["lr"])
    fp = json.loads(json.dumps({"version": VERSION, "settings": settings, "aux": aux_key}, sort_keys=True))
    if dpath.exists() and ppath.exists():
        try:
            if json.loads(dpath.read_text()).get("fingerprint") == fp:
                print(f"[{path.stem}] complete in {out}, skipping", flush=True)
                return json.loads(dpath.read_text())
        except (ValueError, OSError):
            pass

    grads, windows, dates, aux_info = aux_gradients.cached_aux_samples(rec, path, S, aux_cache)
    print(f"[{path.stem}] aux set ({S}) {'loaded from' if aux_info['from_cache'] else 'saved to'} "
          f"{aux_info['path']}", flush=True)
    seeds = {"run_seed": rs, "round": rnd, "aux": aux_key["seed"],
             "model_init": inverter_seed(rs, rnd, -1, "model_init"),
             "shuffle": inverter_seed(rs, rnd, -1, "shuffle")}
    model, log = train(grads, aux_priors(dates, H, mode), windows, settings, seeds["model_init"], seeds["shuffle"])
    rebuilt = rebuild(model, rec, mode).contiguous()

    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = ppath.with_suffix(".pt.tmp")
    torch.save({"rebuilt": rebuilt}, tmp)
    tmp.replace(ppath)
    diagnostics = {"recording": path.name, "attack": ATTACK, "level": level, "eps": rec["eps"], "n": n,
                   "nb": rec["nb"], "round": rnd, "rebuilt_shape": list(rebuilt.shape), "seeds": seeds,
                   "aux": {"path": aux_info["path"], "from_cache": aux_info["from_cache"],
                           "noise_std": aux_gradients.noise_std(level, rec["sigma"], n)},
                   "training": log, "fingerprint": fp}
    tmp = dpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(diagnostics, indent=1))
    tmp.replace(dpath)
    print(f"[{path.stem}] wrote {ppath.name} {tuple(rebuilt.shape)}, best epoch {log['best_epoch']}, held-out "
          f"loss {log['best_val_loss']:.4g} (untrained {log['initial_val_loss']:.4g})", flush=True)
    return diagnostics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m attacks.learned_inverter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", nargs="+", required=True, help="recording file(s)")
    p.add_argument("--out", default="attack_outputs", help="attack output directory (eval.score_all --attacks)")
    p.add_argument("--aux-cache", default=None,
                   help="aux-set cache directory, shared with attacks.ts_inverse (default OUT/_aux_cache)")
    p.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)         # tests only
    p.add_argument("--aux-samples", type=int, default=None, help=argparse.SUPPRESS)    # tests only
    a = p.parse_args(argv)
    over = {k: v for k, v in (("epochs", a.epochs), ("aux_samples", a.aux_samples)) if v is not None}
    settings = load_settings(over)
    if settings["blind"] not in ("same_date", "weekday"):
        raise ValueError(f"attacks.learned_inverter.blind must be same_date or weekday, got {settings['blind']!r}")
    for r in a.recording:
        attack_recording(r, a.out, settings, a.aux_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
