"""Scoring reconstructed windows against the true ones (shared by the attacks).

Matching: the server's view does not say which reconstructed window is which true window (within a
house at levels 1-2, nor which house at level 3), so reconstructions are matched to true windows by
minimum total error (Hungarian, scipy.optimize.linear_sum_assignment) on the mean absolute error over
the whole 96-reading window. Matching uses the true windows: it is a scoring step, not part of the attack.

Metrics, per house (over its 4 true windows and their matched reconstructions):
  smape_input / smape_target: sMAPE on the 0-1 scaled values, the paper's metric, computed as in their
      code: mean of 2|a-b| / (|a|+|b|) with 0/0 counted as 0, separately for the input half (first 48)
      and the target half (last 48). Bounded in [0, 2]; it is 2 wherever the true value is 0 and the
      guess is not, so MAE/RMSE are the robust numbers.
  mae_kwh / rmse_kwh: over the whole window, after data.unscale.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import common
import data


def smape(a: torch.Tensor, b: torch.Tensor) -> float:
    """sMAPE = mean(2|a-b| / (|a|+|b|)) (Makridakis 1993), with 0/0 counted as 0: the same definition
    and convention TS-Inverse reports. Written independently here (not ported), so this module stays
    our own code."""
    num = 2 * (a - b).abs()
    den = a.abs() + b.abs()
    r = num / den
    r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
    return float(r.mean())


def match(true: torch.Tensor, recon: torch.Tensor) -> np.ndarray:
    """Hungarian matching. true, recon: [N, 96]. Returns perm with recon[perm[i]] matched to true[i],
    minimising the total mean-absolute error."""
    if true.shape != recon.shape:
        raise ValueError(f"shapes differ: {tuple(true.shape)} vs {tuple(recon.shape)}")
    cost = (true[:, None, :] - recon[None, :, :]).abs().mean(-1).double().numpy()
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=int)
    perm[rows] = cols
    return perm


def window_metrics(true: torch.Tensor, guess: torch.Tensor) -> dict:
    """Metrics of matched windows true[i] vs guess[i], both [K, 96] on the 0-1 scale."""
    L = common.INPUT_LEN
    t_kwh, g_kwh = data.unscale(true.double()), data.unscale(guess.double())
    return {
        "smape_input": smape(true[:, :L].double(), guess[:, :L].double()),
        "smape_target": smape(true[:, L:].double(), guess[:, L:].double()),
        "mae_kwh": float((t_kwh - g_kwh).abs().mean()),
        "rmse_kwh": float(((t_kwh - g_kwh) ** 2).mean().sqrt()),
    }


def score(true_windows: torch.Tensor, recon: torch.Tensor, blind: torch.Tensor, house_ids: list[str],
          per_house_matching: bool) -> dict:
    """Score a reconstruction and the blind guess.

    true_windows: [n, W, 96] in house order. recon: [n, W, 96] (levels 1-2, one reconstruction per
    house, per_house_matching=True) or [n*W, 96] (level 3: unordered, matched across all houses).
    blind: [W, 96], the attacker-pool mean on the round's W dates; every house gets these W guesses,
    matched the same way as the reconstruction.

    Returns per-house metrics (attack and blind), their means over houses, and the matching.
    """
    n, W, _ = true_windows.shape
    flat_true = true_windows.reshape(n * W, -1)
    blind_all = blind.unsqueeze(0).expand(n, W, -1)
    if per_house_matching:
        recon = recon.reshape(n, W, -1)
        perms = [match(true_windows[h], recon[h]) for h in range(n)]
        matched = torch.stack([recon[h][perms[h]] for h in range(n)])
        matching = [p.tolist() for p in perms]                    # per house: index within the house
        bperms = [match(true_windows[h], blind) for h in range(n)]
        bmatched = torch.stack([blind[p] for p in bperms])
    else:
        recon = recon.reshape(n * W, -1)
        perm = match(flat_true, recon)
        matched = recon[perm].reshape(n, W, -1)
        matching = perm.tolist()                                  # flat: true window i <- recon[perm[i]]
        bflat = blind_all.reshape(n * W, -1)
        bmatched = bflat[match(flat_true, bflat)].reshape(n, W, -1)
    houses = []
    for h in range(n):
        houses.append({"house_id": house_ids[h],
                       "attack": window_metrics(true_windows[h], matched[h]),
                       "blind": window_metrics(true_windows[h], bmatched[h])})
    keys = list(houses[0]["attack"])
    summary = {who: {k: float(np.mean([hh[who][k] for hh in houses])) for k in keys} for who in ("attack", "blind")}
    summary["beats_blind_mae_houses"] = int(sum(hh["attack"]["mae_kwh"] < hh["blind"]["mae_kwh"] for hh in houses))
    return {"per_house": houses, "mean": summary, "matching": matching}
