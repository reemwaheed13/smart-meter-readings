"""Load and validate an attack recording (README, "Recording format")."""
from __future__ import annotations

import re
from numbers import Real
from pathlib import Path

import torch

import common

KEYS = {
    "level", "eps", "n", "nb", "round", "server_view", "view_is_sum", "global_model",
    "true_windows", "house_ids", "dates", "lr", "C", "sigma", "round_time_s", "bytes_per_house",
}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fail(path, msg):
    raise ValueError(f"{path}: {msg}")


def _is_num(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)


def load_recording(path: str | Path) -> dict:
    """torch.load a recording (weights_only, so a file from the shared Drive cannot run code),
    check every key, type and shape, and return the dict."""
    path = Path(path)
    rec = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rec, dict):
        _fail(path, f"expected a dict, got {type(rec).__name__}")
    if set(rec) != KEYS:
        _fail(path, f"keys differ: missing {sorted(KEYS - set(rec))}, extra {sorted(set(rec) - KEYS)}")

    cfg = common.load_config()
    W = cfg["experiment"]["windows_per_round"]
    L = cfg["data"]["window_length"]
    template = common.Model().state_dict()
    P = sum(t.numel() for t in template.values())

    level, n = rec["level"], rec["n"]
    if level not in (1, 2, 3):
        _fail(path, f"level must be 1, 2 or 3, got {level!r}")
    for k in ("n", "nb", "round"):
        if not isinstance(rec[k], int) or isinstance(rec[k], bool):
            _fail(path, f"{k} must be an int, got {rec[k]!r}")
    if n < 1 or rec["nb"] < 0 or rec["round"] < 1:
        _fail(path, f"bad n/nb/round: {n}, {rec['nb']}, {rec['round']}")
    if rec["eps"] is not None and not (_is_num(rec["eps"]) and rec["eps"] > 0):
        _fail(path, f"eps must be None or > 0, got {rec['eps']!r}")
    if level == 1 and rec["eps"] is not None:
        _fail(path, "level 1 must have eps None")

    vis = rec["view_is_sum"]
    if not isinstance(vis, bool):
        _fail(path, f"view_is_sum must be a bool, got {vis!r}")
    if level in (1, 2) and vis:
        _fail(path, "view_is_sum must be False at levels 1-2")

    sv = rec["server_view"]
    want = (P,) if vis else (n, P)
    if not isinstance(sv, torch.Tensor) or tuple(sv.shape) != want or not sv.is_floating_point():
        _fail(path, f"server_view must be a float tensor {list(want)}, got "
                    f"{getattr(sv, 'shape', type(sv).__name__)}")

    gm = rec["global_model"]
    if not isinstance(gm, dict) or list(gm) != list(template):
        _fail(path, f"global_model keys must be {list(template)}")
    for k, t in template.items():
        if not isinstance(gm[k], torch.Tensor) or gm[k].shape != t.shape:
            _fail(path, f"global_model[{k}] must have shape {list(t.shape)}")

    tw = rec["true_windows"]
    if not isinstance(tw, torch.Tensor) or tuple(tw.shape) != (n, W, L) or not tw.is_floating_point():
        _fail(path, f"true_windows must be a float tensor [{n}, {W}, {L}], got "
                    f"{getattr(tw, 'shape', type(tw).__name__)}")

    hs = rec["house_ids"]
    if not isinstance(hs, list) or len(hs) != n or not all(isinstance(h, str) for h in hs) or len(set(hs)) != n:
        _fail(path, f"house_ids must be a list of {n} distinct strings")
    ds = rec["dates"]
    if not isinstance(ds, list) or len(ds) != W or not all(isinstance(d, str) and _DATE.match(d) for d in ds):
        _fail(path, f"dates must be a list of {W} 'YYYY-MM-DD' strings, got {ds!r}")

    if not (_is_num(rec["lr"]) and rec["lr"] > 0):
        _fail(path, f"lr must be > 0, got {rec['lr']!r}")
    for k in ("C", "sigma"):
        if rec[k] is not None and not (_is_num(rec[k]) and rec[k] >= 0):
            _fail(path, f"{k} must be None or >= 0, got {rec[k]!r}")
    if level in (2, 3) and rec["C"] is None:
        _fail(path, "levels 2-3 need C")
    if not (_is_num(rec["round_time_s"]) and rec["round_time_s"] >= 0):
        _fail(path, f"round_time_s must be >= 0, got {rec['round_time_s']!r}")
    if not (_is_num(rec["bytes_per_house"]) and rec["bytes_per_house"] > 0):
        _fail(path, f"bytes_per_house must be > 0, got {rec['bytes_per_house']!r}")
    return rec
