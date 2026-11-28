"""Attacker-pool auxiliary samples for the attacks: (what the server would see, the windows behind it).

Shared by TS-Inverse's inversion model (attacks/ts_inverse.py) and the learned inverter
(attacks/learned_inverter.py). On the same recording both use ONE aux set (see cached_aux_samples).

The attacker holds the attacker pool (data.attacker_houses(), disjoint from every neighbourhood) and
knows the public protection settings of the recording (level, C, sigma, n). An aux sample mirrors the
server's view in a round, computed with the code the houses run (common.local_update, common.protect)
at the recorded weights. With g = local_update(model, 4 windows of one attacker house):

  level 1:  g                                     the raw mean gradient of the 4 windows
  level 2:  common.protect(g, 2, C, sigma, n)   = clip(g, C) + N(0, sigma^2) per coordinate
  level 3:  mean over n attacker houses of common.protect(g_h, 3, C, sigma, n)
                                                = mean_h [clip(g_h, C) + N(0, sigma^2 / n)]

Noise std per coordinate of a sample, which is the noise in the server's view (noise_std below):
  level 1: 0.
  level 2: sigma. Training runs add it with Flower's LocalDpMod in weight space: clip of the update
           -lr*g to lr*C, then N(0, (lr*C*sqrt(2 ln(1.25/delta))/eps)^2); the server's
           g_hat = (w - returned)/lr is therefore clip(g, C) + N(0, sigma_for(eps, C, delta)^2), the
           same distribution as common.protect(g, 2, ...). (Equal to the recording's sigma as long as
           config.yaml dp.sigma is null: LocalDpMod always uses the formula.)
  level 3: sigma / n. Each house adds a share N(0, sigma^2/n) (noise_share_mod -> common.protect);
           the n independent shares sum to N(0, sigma^2), and the server's view is the MEAN (Flower's
           SecAgg+ aggregate), so its noise is N(0, sigma^2 / n^2). SecAgg+'s stochastic rounding
           (step 2 * clipping_range / quantization_range, about 4e-6) is not modelled.
  sigma None or 0 (level 3 with eps = none) means no noise.

Each sample's windows are one (levels 1-2) or n (level 3) attacker houses on the same windows_per_house
random train dates (all houses in a round share the round's dates), drawn from the dates valid for all
of that sample's houses. Two seeded streams: the window draw (houses, dates) and the noise
(noise_seed(seed)), so changing sigma changes only the noise, not which windows are drawn.

Data access is through data.py only (attacker_houses, valid_dates and get_windows), always on the
train split. The blind guess lives in eval/baseline.py.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from collections import defaultdict
from pathlib import Path

import torch

import common
import data

SPLIT = "train"
MAX_DRAWS = 100   # attempts per sample to find houses sharing enough valid train dates
VERSION = "aux-2"  # bump when the samples change; part of the cache key


def _model(global_model: dict) -> torch.nn.Module:
    m = common.Model()
    m.load_state_dict(global_model)
    return m


def house_view(model: torch.nn.Module, windows: torch.Tensor, level: int, C: float | None,
               lr: float, *, create_graph: bool = False) -> torch.Tensor:
    """One house's CLEAN view [P]: its local_update, clipped to C at levels 2-3, no noise. TS-Inverse
    matches its fake windows with this (the noise is not differentiable and not known)."""
    g = common.local_update(model, windows, lr, create_graph=create_graph)
    return g if level == 1 else common.clip(g, C)


def noise_std(level: int, sigma: float | None, n: int) -> float:
    """Per-coordinate noise std of the server's view (and of an aux sample): see the module docstring."""
    if level == 1 or not sigma:
        return 0.0
    return float(sigma) if level == 2 else float(sigma) / n


def aux_seed(run_seed: int, rnd: int) -> int:
    """The recording's aux-set seed, from (run_seed, round). Shared by every attack on that recording."""
    return zlib.crc32(f"aux|{run_seed}|{rnd}|-1|windows".encode())


def noise_seed(seed: int) -> int:
    """The noise stream's seed, separate from the window draw's."""
    return zlib.crc32(f"aux|{seed}|noise".encode())


def aux_samples(global_model: dict, level: int, C: float | None, n: int, lr: float, num_samples: int,
                seed: int, *, sigma: float | None = None, windows_per_house: int | None = None,
                return_dates: bool = False):
    """num_samples attacker-pool samples at the recorded weights, with the level's noise.

    Returns (grads [S, P], windows [S, H*W, 96]) and, with return_dates, the dates [S][W] of each
    sample (the same for its H houses). H = n houses per sample at level 3 and H = 1 otherwise,
    W = windows_per_house (default experiment.windows_per_round = 4). Level 3: the sample's windows are
    house-major (house 0's W windows, then house 1's, ...), and its gradient is the mean over the H
    houses of common.protect(local_update, 3, C, sigma, n).
    """
    if level not in (1, 2, 3):
        raise ValueError(f"unknown level {level}")
    if level in (2, 3) and C is None:
        raise ValueError("levels 2-3 need C")
    W = windows_per_house or common.load_config()["experiment"]["windows_per_round"]
    H = n if level == 3 else 1
    pool = data.attacker_houses()
    if H > len(pool):
        raise ValueError(f"need {H} attacker houses per sample, the pool has {len(pool)}")

    g = torch.Generator().manual_seed(int(seed))
    draws = []                                   # per sample: (house indices [H], dates [W])
    for _ in range(num_samples):
        for _attempt in range(MAX_DRAWS):
            hs = torch.randperm(len(pool), generator=g)[:H].tolist()
            # Real data has gaps: only dates with a complete window for every drawn house qualify.
            dates = data.valid_dates([pool[h] for h in hs], SPLIT)
            if len(dates) >= W:
                break
        else:
            raise ValueError(f"no {H} attacker houses with {W} shared valid {SPLIT} dates in {MAX_DRAWS} draws")
        ds = [dates[i] for i in sorted(torch.randperm(len(dates), generator=g)[:W].tolist())]
        draws.append((hs, ds))

    # Fetch each house's windows once, for every date it is drawn with.
    need = defaultdict(set)
    for hs, ds in draws:
        for h in hs:
            need[h].update(ds)
    cache = {}
    for h, ds in need.items():
        ds = sorted(ds)
        win = data.get_windows([pool[h]], ds, SPLIT)[0]                 # [len(ds), 96]
        cache.update({(h, d): win[j] for j, d in enumerate(ds)})

    model = _model(global_model)
    gnoise = torch.Generator().manual_seed(noise_seed(seed))
    grads = torch.empty(num_samples, common.num_params(model))
    windows = torch.empty(num_samples, H * W, 2 * common.INPUT_LEN)
    for s, (hs, ds) in enumerate(draws):
        views = []
        for j, h in enumerate(hs):
            w = torch.stack([cache[(h, d)] for d in ds])
            windows[s, j * W:(j + 1) * W] = w
            g_h = common.local_update(model, w, lr)
            views.append(common.protect(g_h, level, C, sigma, n, generator=gnoise))
        grads[s] = torch.stack(views).mean(0)
    if return_dates:
        return grads, windows, [ds for _, ds in draws]
    return grads, windows


# --------------------------------------------------------------------------- one aux set per recording

def cache_key(recording: str, level: int, C: float | None, sigma: float | None, n: int, num_samples: int,
              seed: int, windows_per_house: int, lr: float) -> dict:
    """Everything the aux set depends on. The recording name stands for its weights and round."""
    return {"version": VERSION, "recording": Path(recording).name, "level": int(level),
            "C": None if C is None else float(C), "sigma": float(sigma or 0.0), "n": int(n),
            "num_samples": int(num_samples), "seed": int(seed), "windows_per_house": int(windows_per_house),
            "lr": float(lr)}


def cache_path(cache_dir: str | Path, key: dict) -> Path:
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{Path(key['recording']).stem}__aux_{h}.pt"


def cached_aux_samples(rec: dict, recording: str | Path, num_samples: int, cache_dir: str | Path | None):
    """The recording's aux set (grads, windows, dates), seeded with aux_seed(run_seed(n, nb), round) and
    noised as the recording's level and sigma. Loaded from cache_dir if a file with the same key exists,
    else computed and saved there (atomically). cache_dir None: computed, not saved.
    Returns (grads, windows, dates, info) with info = {"key", "path", "from_cache"}."""
    level, n, C, sigma, lr = rec["level"], rec["n"], rec["C"], rec["sigma"], rec["lr"]
    W = rec["true_windows"].shape[1]
    seed = aux_seed(common.run_seed(n, rec["nb"]), rec["round"])
    key = cache_key(recording, level, C, sigma, n, num_samples, seed, W, lr)
    path = cache_path(cache_dir, key) if cache_dir is not None else None
    if path is not None and path.exists():
        try:
            part = torch.load(path, map_location="cpu", weights_only=True)
            if json.loads(part["key"]) == key:
                return part["grads"], part["windows"], part["dates"], {"key": key, "path": str(path),
                                                                       "from_cache": True}
        except Exception:
            pass                                 # unreadable or another key: recompute
    grads, windows, dates = aux_samples(rec["global_model"], level, C, n, lr, num_samples, seed, sigma=sigma,
                                        windows_per_house=W, return_dates=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"key": json.dumps(key, sort_keys=True), "grads": grads, "windows": windows, "dates": dates}, tmp)
        tmp.replace(path)
    return grads, windows, dates, {"key": key, "path": None if path is None else str(path), "from_cache": False}

