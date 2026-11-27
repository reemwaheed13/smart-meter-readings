"""Attacker-pool data for the attacks: auxiliary gradients and the blind-guess baseline.

Shared by TS-Inverse's inversion model (attacks/ts_inverse.py) and the learned inverter.

The attacker holds the attacker pool (data.attacker_houses(), disjoint from every neighbourhood).
An aux sample mirrors one thing the server sees in a round, computed with the same code the houses run
(common.local_update) at the recorded weights:

  level 1:  g      = local_update(model, 4 windows of one attacker house)
  level 2:  clip(g, C)
  level 3:  mean over n attacker houses of clip(g_h, C)        (the [P] server_view is a mean too)

Each sample's windows are one or n attacker houses on the same windows_per_house random train dates
(all houses in a round share the round's dates). The noise is not added: a sample is the clean,
clipped view. Every draw comes from a torch.Generator seeded by the caller, so reruns are identical.
"""
from __future__ import annotations

from collections import defaultdict

import torch

import common
import data


def _model(global_model: dict) -> torch.nn.Module:
    m = common.Model()
    m.load_state_dict(global_model)
    return m


def house_view(model: torch.nn.Module, windows: torch.Tensor, level: int, C: float | None,
               lr: float, *, create_graph: bool = False) -> torch.Tensor:
    """One house's view [P]: its local_update, clipped to C at levels 2-3. This is the clean part of
    what the server sees for that house (no noise)."""
    g = common.local_update(model, windows, lr, create_graph=create_graph)
    return g if level == 1 else common.clip(g, C)


def aux_samples(global_model: dict, level: int, C: float | None, n: int, lr: float, num_samples: int,
                seed: int, *, windows_per_house: int | None = None, split: str = "train"
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """num_samples attacker-pool samples at the recorded weights.

    Returns (grads [S, P], windows [S, H*W, 96]) with H = n houses per sample at level 3 and H = 1
    otherwise, W = windows_per_house (default experiment.windows_per_round = 4). Level 3: the
    sample's windows are house-major (house 0's W windows, then house 1's, ...), and its gradient is
    the mean over the H houses of clip(local_update, C).
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
    dates = data.valid_dates(pool, split)
    if W > len(dates):
        raise ValueError(f"need {W} dates, the {split} split has {len(dates)}")

    g = torch.Generator().manual_seed(int(seed))
    draws = []                                   # per sample: (house indices [H], date indices [W])
    for _ in range(num_samples):
        hs = torch.randperm(len(pool), generator=g)[:H].tolist()
        ds = sorted(torch.randperm(len(dates), generator=g)[:W].tolist())
        draws.append((hs, ds))

    # Fetch each house's windows once, for every date it is drawn with.
    need = defaultdict(set)
    for hs, ds in draws:
        for h in hs:
            need[h].update(ds)
    cache = {}
    for h, ds in need.items():
        ds = sorted(ds)
        win = data.get_windows([pool[h]], [dates[d] for d in ds], split)[0]   # [len(ds), 96]
        cache.update({(h, d): win[j] for j, d in enumerate(ds)})

    model = _model(global_model)
    grads = torch.empty(num_samples, common.num_params(model))
    windows = torch.empty(num_samples, H * W, 2 * common.INPUT_LEN)
    for s, (hs, ds) in enumerate(draws):
        views = []
        for j, h in enumerate(hs):
            w = torch.stack([cache[(h, d)] for d in ds])
            windows[s, j * W:(j + 1) * W] = w
            views.append(house_view(model, w, level, C, lr))
        grads[s] = torch.stack(views).mean(0)
    return grads, windows


def attacker_pool_mean(dates: list[str], split: str = "train") -> torch.Tensor:
    """The blind guess: the attacker pool's mean window on each of the given dates, [len(dates), 96].
    The server knows the round's dates (it sends them in the fit config)."""
    return data.get_windows(data.attacker_houses(), list(dates), split).mean(0)
