"""Write attack recordings in the handshake format (README, "Recording format")."""
from __future__ import annotations

from pathlib import Path

import torch

import data
from train.settings import RunSettings


def rounds_to_record(T: int) -> list[int]:
    """Rounds 1, T/2 and T, with T/2 rounded up ((T+1)//2) and duplicates removed.
    T=1 -> [1], T=2 -> [1, 2], T=3 -> [1, 2, 3], T=10 -> [1, 5, 10]."""
    return sorted({1, (T + 1) // 2, T})


def recording_path(s: RunSettings, server_round: int) -> Path:
    return s.recordings_dir / f"{s.run_name}_r{server_round}.pt"


def save_recording(s: RunSettings, last_round: dict, round_time_s: float, bytes_per_house: float) -> Path:
    r = last_round["round"]
    dates = last_round["dates"]
    rec = {
        "level": s.level,
        "eps": s.eps,
        "n": s.n,
        "nb": s.k,
        "round": r,
        "server_view": last_round["server_view"],
        # Only meaningful at level 3, where the server sees one aggregate [P] instead of [n, P].
        "view_is_sum": s.level == 3,
        "global_model": last_round["global_model"],
        # Ground truth for scoring the attacks. Attack code must never receive it as an input.
        "true_windows": data.get_windows(list(s.house_ids), dates, "train"),
        "house_ids": list(s.house_ids),
        "dates": list(dates),
        "lr": s.lr,
        "C": s.C,
        "sigma": s.sigma,
        "round_time_s": float(round_time_s),
        "bytes_per_house": float(bytes_per_house),
    }
    path = recording_path(s, r)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, path)
    return path
