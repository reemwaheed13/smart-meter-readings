"""FAKE data.py: random stand-in data, to be replaced by the real data.py (London Smartmeter).

It exists only so training and attack code can run before the real loader is ready. It has the
same interface as the real data.py (see "Handshakes" in README.md), but every reading is random.
Do not report any result computed from it.

Interface:
    get_neighbourhood(n, k)               -> list of n house IDs (deterministic in n, k)
    valid_dates(house_ids, split)         -> list of "YYYY-MM-DD" target dates
    get_windows(house_ids, dates, split)  -> float tensor [houses, dates, 96], scaled 0-1
    attacker_houses()                     -> list of attacker_pool_size house IDs
    unscale(x)                            -> x mapped from 0-1 back to kWh

A window for target date d is the 48 readings of day d-1 (input) followed by the 48 readings
of day d (output). Windows are split by target date, and windows whose input day falls in an
earlier split are dropped.
"""
from __future__ import annotations

import datetime as _dt
import zlib

import numpy as np
import torch

from common import load_config

FAKE = True

_cfg = load_config()["data"]
_PER_DAY = _cfg["readings_per_day"]
_N_FAKE_HOUSES = 2000
_FAKE_MIN_KWH, _FAKE_MAX_KWH = 0.0, 3.0

# Fake IDs that cannot be mistaken for real LCLids (MAC000002, ...).
_ALL_IDS = [f"FAKE{i:06d}" for i in range(_N_FAKE_HOUSES)]
_ATTACKER_IDS = _ALL_IDS[: _cfg["attacker_pool_size"]]
_FL_IDS = _ALL_IDS[_cfg["attacker_pool_size"]:]


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def get_neighbourhood(n: int, k: int) -> list[str]:
    """n house IDs for neighbourhood k. Depends only on (n, k); never overlaps the attacker pool."""
    rng = np.random.default_rng(_seed("neighbourhood", n, k))
    idx = rng.choice(len(_FL_IDS), size=n, replace=False)
    return [_FL_IDS[i] for i in idx]


def attacker_houses() -> list[str]:
    """The attacker's house IDs (attacker_pool_size of them)."""
    return list(_ATTACKER_IDS)


def _split_dates() -> dict[str, list[str]]:
    year = _cfg["year"]
    start = _dt.date(year, 1, 1)
    days = [start + _dt.timedelta(d) for d in range((_dt.date(year + 1, 1, 1) - start).days)]
    fr = _cfg["split"]
    n_tr = round(len(days) * fr["train"])
    n_va = round(len(days) * fr["val"])
    parts = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}
    out = {}
    for name, ds in parts.items():
        # The window for target date d also uses day d-1. Drop it if d-1 is outside this split
        # (this also drops 1 January, whose input day is in the previous year).
        if _cfg["drop_boundary_windows"]:
            ds = ds[1:]
        out[name] = [d.isoformat() for d in ds]
    return out


_DATES = _split_dates()


def valid_dates(house_ids: list[str], split: str) -> list[str]:
    """Target dates ("YYYY-MM-DD") in `split` with a complete window for every house given.
    The fake data has no gaps, so this is every date in the split."""
    if split not in _DATES:
        raise ValueError(f"split must be one of {list(_DATES)}")
    return list(_DATES[split])


def _day(house: str, date: _dt.date) -> np.ndarray:
    """One fake day of 48 readings in 0-1: a house-specific daily shape plus noise."""
    h = np.random.default_rng(_seed("house", house))
    base, morning, evening = h.uniform(0.05, 0.2), h.uniform(0.1, 0.4), h.uniform(0.2, 0.6)
    t = np.arange(_PER_DAY) / _PER_DAY * 24
    shape = (base + morning * np.exp(-((t - 7.5) ** 2) / 2)
             + evening * np.exp(-((t - 19) ** 2) / 4))
    r = np.random.default_rng(_seed("day", house, date.isoformat()))
    return np.clip(shape + r.normal(0, 0.05, _PER_DAY), 0.0, 1.0)


def get_windows(house_ids: list[str], dates: list[str], split: str) -> torch.Tensor:
    """Float tensor [houses, dates, 96] in 0-1, houses and dates in the order given."""
    allowed = set(valid_dates(house_ids, split))
    out = np.empty((len(house_ids), len(dates), 2 * _PER_DAY), dtype=np.float32)
    for j, ds in enumerate(dates):
        if ds not in allowed:
            raise ValueError(f"{ds} is not a valid {split} date")
        d = _dt.date.fromisoformat(ds)
        for i, h in enumerate(house_ids):
            out[i, j] = np.concatenate([_day(h, d - _dt.timedelta(1)), _day(h, d)])
    return torch.from_numpy(out)


def unscale(x):
    """Map 0-1 values back to kWh (fake min/max)."""
    return x * (_FAKE_MAX_KWH - _FAKE_MIN_KWH) + _FAKE_MIN_KWH
