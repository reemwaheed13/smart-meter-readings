"""REAL data: London Smartmeter (Low Carbon London), half-hourly, 2013. Owner: Shifa.

data.py routes here by default (SMART_METER_DATA unset or "real"). Same interface as the
fake stub (README, "Handshakes"):

    get_neighbourhood(n, k)               -> list of n house IDs (deterministic in n, k)
    valid_dates(house_ids, split)         -> list of "YYYY-MM-DD" target dates
    get_windows(house_ids, dates, split)  -> float tensor [houses, dates, 96], scaled 0-1
    attacker_houses()                     -> list of attacker_pool_size house IDs
    unscale(x)                            -> x mapped from 0-1 back to kWh

ONE-TIME PREP (needs pandas)
    1. Download Kaggle "Smart meters in London" (jeanmidev). Put informations_households.csv and
       the halfhourly_dataset/ folder (block_*.csv) in data/raw/. The London Datastore single CSV
       (CC_LCL-FullData.csv) also works: put it in data/raw/.
    2. python data_real.py prepare     -> data/processed/clean_2013.npz
    3. python data_real.py summary     -> house counts, dates per split, scale, neighbourhoods
    The .npz is gitignored (data/*). Share it through the Drive; in Colab point
    SMART_METER_PROCESSED at it, e.g. /content/drive/MyDrive/smart-meter-fl/clean_2013.npz.

DEFINITIONS (settings from config.yaml "data")
    - Std tariff houses only. Houses missing > max_missing_fraction of the year are dropped.
      Gaps of <= max_fill_gap_readings are filled linearly; negative readings count as missing.
    - Window for target date d = the 48 readings of day d-1 (input) + the 48 of day d (output).
      A window with any remaining gap is not valid for that house.
    - The days of the year are split in order by the split fractions (rounded as in the stub).
      A window is kept only if day d-1 is in the same split as d (drops boundary windows and 1 Jan).
    - Attacker pool: attacker_pool_size houses chosen at random with a fixed seed. Neighbourhoods are
      drawn from the rest; for one n, k = 0, 1, 2 never share a house.
    - Scaling: one min-max (kWh) fitted on the attacker pool's train days, applied to every house.
"""
from __future__ import annotations

import datetime as _dt
import glob
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from common import load_config

FAKE = False
_REPO = Path(__file__).resolve().parent
_cfg = load_config()["data"]
_PER_DAY = _cfg["readings_per_day"]
RAW_DIR = _REPO / _cfg.get("raw_dir", "data/raw")
PROCESSED_FILE = Path(os.environ.get("SMART_METER_PROCESSED")
                      or _REPO / _cfg.get("processed_file", "data/processed/clean_2013.npz"))
POOL_SEED = 0          # fixed seed for choosing the attacker pool
SPLITS = ("train", "val", "test")


# =========================================================================== one-time prep
def _read_raw(raw_dir: Path, year: int, tariff: str):
    """Long DataFrame LCLid, ts, kwh for the chosen tariff and year."""
    import pandas as pd

    lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")

    def tidy(df):
        df = df[["LCLid", "ts", "kwh"]].copy()
        df["LCLid"] = df["LCLid"].str.strip()
        df["ts"] = pd.to_datetime(df["ts"].str.slice(0, 19), errors="coerce")
        df["kwh"] = pd.to_numeric(df["kwh"].str.strip(), errors="coerce")  # "Null" -> NaN
        df = df.dropna(subset=["ts"])
        df = df[(df["ts"] >= lo) & (df["ts"] < hi)]
        df["ts"] = df["ts"].dt.floor("30min")  # a few readings are seconds off the half hour
        return df

    blocks = sorted(glob.glob(str(raw_dir / "**" / "block_*.csv"), recursive=True))
    frames = []
    if blocks:  # Kaggle layout
        info = pd.read_csv(raw_dir / "informations_households.csv", dtype=str)
        keep = set(info.loc[info["stdorToU"].str.strip() == tariff, "LCLid"].str.strip())
        print(f"{len(keep)} houses on tariff {tariff}; reading {len(blocks)} block files")
        for i, path in enumerate(blocks, 1):
            df = pd.read_csv(path, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            energy = next(c for c in df.columns if c.lower().startswith("energy"))
            df = df.rename(columns={"tstp": "ts", energy: "kwh"})
            frames.append(tidy(df[df["LCLid"].str.strip().isin(keep)]))
            if i % 10 == 0:
                print(f"  {i}/{len(blocks)} blocks")
    else:  # London Datastore single CSV
        files = [f for f in sorted(glob.glob(str(raw_dir / "*.csv"))) if "information" not in f.lower()]
        if not files:
            raise FileNotFoundError(f"No raw data in {raw_dir}: see the docstring of data_real.py")
        for path in files:
            print(f"reading {path} in chunks")
            for ch in pd.read_csv(path, dtype=str, chunksize=2_000_000):
                ch.columns = [c.strip() for c in ch.columns]
                kwh = next(c for c in ch.columns if "kwh" in c.lower())
                ts = next(c for c in ch.columns if c.lower() in ("datetime", "tstp"))
                ch = ch[ch["stdorToU"].str.strip() == tariff].rename(columns={ts: "ts", kwh: "kwh"})
                frames.append(tidy(ch))
    return pd.concat(frames, ignore_index=True)


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap that have a real value on both sides."""
    x = x.copy()
    nan = np.isnan(x)
    if not nan.any():
        return x
    edges = np.diff(np.concatenate([[0], nan.astype(int), [0]]))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):  # run = [s, e)
        if e - s <= max_gap and s > 0 and e < len(x):
            x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def prepare(raw_dir: Path = RAW_DIR, out: Path = PROCESSED_FILE) -> None:
    import pandas as pd

    year = _cfg["year"]
    start = pd.Timestamp(f"{year}-01-01")
    n_slots = (pd.Timestamp(f"{year + 1}-01-01") - start) // pd.Timedelta("30min")
    df = _read_raw(Path(raw_dir), year, _cfg["tariff"])
    df = df.groupby(["LCLid", "ts"], as_index=False)["kwh"].mean()  # duplicate timestamps
    df["slot"] = ((df["ts"] - start) / pd.Timedelta("30min")).astype(int)

    ids, rows, dropped = [], [], 0
    for hid, g in df.groupby("LCLid"):
        x = np.full(n_slots, np.nan, dtype=np.float32)
        x[g["slot"].to_numpy()] = g["kwh"].to_numpy(dtype=np.float32)
        x[x < 0] = np.nan
        if np.isnan(x).mean() > _cfg["max_missing_fraction"]:
            dropped += 1
            continue
        ids.append(hid)
        rows.append(_fill_short_gaps(x, _cfg["max_fill_gap_readings"]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, house_ids=np.array(ids), readings=np.stack(rows), year=year)
    print(f"kept {len(ids)} houses, dropped {dropped} with > {_cfg['max_missing_fraction']:.0%} missing")
    print(f"saved {out}")
    _store.cache_clear()


# =========================================================================== loading
@lru_cache(maxsize=1)
def _store() -> dict:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{PROCESSED_FILE} not found. Run `python data_real.py prepare`, set SMART_METER_PROCESSED "
            "to the shared .npz, or set SMART_METER_DATA=fake to use the fake stub.")
    z = np.load(PROCESSED_FILE, allow_pickle=False)
    ids = [str(h) for h in z["house_ids"]]
    year = int(z["year"])
    n_days = z["readings"].shape[1] // _PER_DAY
    days = z["readings"].reshape(len(ids), n_days, _PER_DAY)       # [H, day, 48], NaN = missing
    first = _dt.date(year, 1, 1)
    dates = [(first + _dt.timedelta(d)).isoformat() for d in range(n_days)]

    fr = _cfg["split"]
    n_tr, n_va = round(n_days * fr["train"]), round(n_days * fr["val"])
    day_split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n_days - n_tr - n_va))
    target = np.arange(1, n_days)                                   # window w: days w and w+1
    in_split = day_split[target - 1] == day_split[target]
    complete = ~np.isnan(days).any(axis=2)
    win_ok = complete[:, target - 1] & complete[:, target]          # [H, window]

    order = np.random.default_rng(POOL_SEED).permutation(len(ids))
    size = _cfg["attacker_pool_size"]
    if len(ids) < size + 60:
        raise ValueError(f"Only {len(ids)} clean houses: need {size} for the attacker pool plus "
                         "at least 60 for the neighbourhoods. Use more block files.")
    attacker = sorted(ids[i] for i in order[:size])
    pool = sorted(ids[i] for i in order[size:])

    row = {h: i for i, h in enumerate(ids)}
    att_train = days[np.ix_([row[h] for h in attacker], np.where(day_split == "train")[0])]
    lo, hi = float(np.nanmin(att_train)), float(np.nanmax(att_train))
    return dict(ids=ids, days=days, dates=dates, day_split=day_split, target=target, in_split=in_split,
                win_ok=win_ok, row=row, date_idx={d: i for i, d in enumerate(dates)},
                attacker=attacker, pool=pool, lo=lo, hi=hi)


# =========================================================================== interface
def attacker_houses() -> list[str]:
    return list(_store()["attacker"])


def neighbourhood_pool() -> list[str]:
    return list(_store()["pool"])


def get_neighbourhood(n: int, k: int) -> list[str]:
    """n house IDs for neighbourhood k; depends only on (n, k). For one n, the k's are disjoint
    slices of one shuffle, so they never share a house. Never overlaps the attacker pool."""
    pool = _store()["pool"]
    if (k + 1) * n > len(pool):
        raise ValueError(f"neighbourhood pool has {len(pool)} houses; n={n}, k={k} needs {(k + 1) * n}")
    order = np.random.default_rng(10_000 + n).permutation(len(pool))
    return [pool[i] for i in order[k * n:(k + 1) * n]]


def valid_dates(house_ids: list[str], split: str) -> list[str]:
    """Target dates in `split` with a complete window for EVERY house given."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}")
    st = _store()
    t = st["target"]
    ok = st["in_split"] & (st["day_split"][t] == split)
    if house_ids:
        ok &= st["win_ok"][[st["row"][h] for h in house_ids]].all(axis=0)
    return [st["dates"][d] for d in t[ok]]


def get_windows(house_ids: list[str], dates: list[str], split: str) -> torch.Tensor:
    """Float tensor [houses, dates, 96] in 0-1, houses and dates in the order given."""
    st = _store()
    allowed = set(valid_dates(house_ids, split))
    bad = [d for d in dates if d not in allowed]
    if bad:
        raise ValueError(f"not valid {split} dates for these houses: {bad[:5]}")
    rows = [st["row"][h] for h in house_ids]
    t = np.array([st["date_idx"][d] for d in dates], dtype=int)
    sel = st["days"][rows]
    x = np.concatenate([sel[:, t - 1], sel[:, t]], axis=2)
    return torch.from_numpy(((x - st["lo"]) / (st["hi"] - st["lo"])).astype(np.float32))


def unscale(x):
    """Map 0-1 values back to kWh per half hour (numpy or torch)."""
    st = _store()
    return x * (st["hi"] - st["lo"]) + st["lo"]


def summary() -> None:
    st = _store()
    print(f"clean houses        : {len(st['ids'])}")
    print(f"attacker pool       : {len(st['attacker'])}")
    print(f"neighbourhood pool  : {len(st['pool'])}")
    for s in SPLITS:
        print(f"{s:5s} target dates   : {len(valid_dates([], s))}")
    print(f"scale (kWh/hh)      : {st['lo']:.3f} to {st['hi']:.3f}")
    for n in (5, 10, 20):
        for k in range(3):
            try:
                print(f"n={n:2d} k={k}: {len(valid_dates(get_neighbourhood(n, k), 'train'))} shared train dates")
            except ValueError as e:
                print(f"n={n:2d} k={k}: {e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "prepare":
        prepare()
        summary()
    elif cmd == "summary":
        summary()
    else:
        sys.exit("usage: python data_real.py [prepare|summary]")
