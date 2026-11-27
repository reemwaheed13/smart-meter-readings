"""Blind guess: what you'd guess about a house WITHOUT attacking anything.

For a window with target date d (day d-1 + day d), the guess is the attacker pool's average day
for each of those two days:
    mode="weekday"   (default) average over the attacker pool's TRAIN days with the same weekday
    mode="same_date" average over the attacker pool on those exact dates (stronger: it also knows
                     the season and the weather). Report which one you used.

Uses only the data.py interface, so it works on the real data and on the fake stub.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache

import numpy as np

import data


@lru_cache(maxsize=1)
def _daily_means() -> dict[str, np.ndarray]:
    """{date: attacker pool's mean day [48], scaled 0-1} for every date any window covers."""
    sums, counts = {}, {}
    for h in data.attacker_houses():
        for split in ("train", "val", "test"):
            dates = data.valid_dates([h], split)
            if not dates:
                continue
            w = data.get_windows([h], dates, split)[0].numpy()        # [dates, 96]
            for ds, row in zip(dates, w):
                d = _dt.date.fromisoformat(ds)
                for day, half in ((d - _dt.timedelta(1), row[:48]), (d, row[48:])):
                    key = day.isoformat()
                    sums[key] = sums.get(key, 0) + half
                    counts[key] = counts.get(key, 0) + 1
    # day d-1 and day d of neighbouring windows are the same day: average each day's readings
    # over all the copies (every copy is identical, so this is just the per-day pool mean)
    return {k: sums[k] / counts[k] for k in sums}


@lru_cache(maxsize=1)
def _weekday_means() -> dict[int, np.ndarray]:
    train = set()
    for h in data.attacker_houses()[:50]:  # the train date range is the same for every house
        train |= set(data.valid_dates([h], "train"))
    train_days = {(_dt.date.fromisoformat(d) - _dt.timedelta(k)).isoformat() for d in train for k in (0, 1)}
    by_wd: dict[int, list] = {}
    for day, prof in _daily_means().items():
        if day in train_days:
            by_wd.setdefault(_dt.date.fromisoformat(day).weekday(), []).append(prof)
    return {wd: np.mean(v, axis=0) for wd, v in by_wd.items()}


def blind_guess(dates, mode: str = "weekday") -> np.ndarray:
    """Scaled 0-1 guesses [len(dates), 96], one per target date."""
    out = []
    for ds in dates:
        d = _dt.date.fromisoformat(str(ds))
        days = (d - _dt.timedelta(1), d)
        if mode == "same_date":
            prof = _daily_means()
            out.append(np.concatenate([prof[x.isoformat()] for x in days]))
        elif mode == "weekday":
            prof = _weekday_means()
            out.append(np.concatenate([prof[x.weekday()] for x in days]))
        else:
            raise ValueError("mode must be 'weekday' or 'same_date'")
    return np.stack(out).astype(np.float32)
