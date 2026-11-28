"""Freeze calibration.json into config.yaml.

    python -m train.freeze_calibration --calibration DIR/calibration.json [--config config.yaml]

Writes calibration.lr, calibration.T[n], calibration.C, calibration.clipping_range and dp.sigma[eps]
from calibration.json's "values". It refuses if any of them is already non-null in config.yaml.
From then on train.run takes them from config.yaml and refuses --lr / --T / --C / --clipping-range.

config.yaml is edited line by line (only the `key: null` lines change), so its comments and layout
are kept. The result is re-read and checked before it replaces the file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

_INDENT = 2


def _find(lines: list[str], path: list[str]) -> int:
    """Index of the line holding the key at `path` (YAML with 2-space indents)."""
    lo, hi = 0, len(lines)
    idx = -1
    for depth, key in enumerate(path):
        pat = re.compile(rf"^{' ' * (_INDENT * depth)}{re.escape(str(key))}\s*:(\s|$)")
        idx = next((i for i in range(lo, hi) if pat.match(lines[i])), -1)
        if idx < 0:
            raise KeyError(f"config.yaml has no key {'.'.join(map(str, path))}")
        # The block of this key ends at the next non-blank, non-comment line indented <= this one.
        lo, end = idx + 1, idx + 1
        while end < hi:
            s = lines[end]
            if s.strip() and not s.lstrip().startswith("#") and len(s) - len(s.lstrip()) <= _INDENT * depth:
                break
            end += 1
        hi = end
    return idx


def _fmt(v) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"calibration values must be numbers, got {v!r}")
    return str(int(v)) if isinstance(v, int) else repr(float(v))


def _set(lines: list[str], path: list[str], value) -> None:
    i = _find(lines, path)
    m = re.match(r"^(\s*[^:#]+:\s*)null(\s*(#.*)?)$", lines[i].rstrip("\n"))
    if not m:
        raise ValueError(f"{'.'.join(map(str, path))} is not null in config.yaml: {lines[i].strip()!r}")
    v, tail = _fmt(value), m.group(2)
    if m.group(3):  # keep an inline comment in its column where the padding allows (at least one space)
        pad = len(tail) - len(tail.lstrip())
        tail = " " * max(1, pad - (len(v) - len("null"))) + tail.lstrip()
    lines[i] = f"{m.group(1)}{v}{tail}\n"


def targets(values: dict) -> list[tuple[list[str], object]]:
    """(config path, value) pairs from calibration.json's "values"."""
    out = [(["calibration", "lr"], float(values["lr"]))]
    out += [(["calibration", "T", str(n)], int(t)) for n, t in values["T"].items()]
    out += [(["calibration", "C"], float(values["C"])),
            (["calibration", "clipping_range"], float(values["clipping_range"]))]
    out += [(["dp", "sigma", f"{float(e):g}"], float(s)) for e, s in values["sigma"].items()]
    return out


def _get(cfg: dict, path: list[str]):
    v = cfg
    for k in path:
        v = v[int(k) if isinstance(v, dict) and k.isdigit() and int(k) in v else k]
    return v


def freeze(calibration_json: str | Path, config_path: str | Path = "config.yaml") -> list:
    config_path = Path(config_path)
    values = json.loads(Path(calibration_json).read_text())["values"]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tg = targets(values)

    want_T = {str(n) for n in cfg["experiment"]["n"]}
    if set(values["T"]) != want_T:
        raise ValueError(f"calibration.json has T for n = {sorted(values['T'])}, config.yaml needs {sorted(want_T)}")
    already = [".".join(p) for p, _ in tg if _get(cfg, p) is not None]
    if already:
        raise ValueError(f"config.yaml already sets {', '.join(already)}: refusing to overwrite. "
                         "Calibration is frozen once; reset them to null by hand only if you mean to redo it.")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for p, v in tg:
        _set(lines, p, v)
    text = "".join(lines)
    new = yaml.safe_load(text)
    for p, v in tg:
        if _get(new, p) != v:
            raise RuntimeError(f"freeze check failed: {'.'.join(p)} reads back {_get(new, p)!r}, wanted {v!r}")
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(config_path)
    return tg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.freeze_calibration", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", required=True, help="path to calibration.json")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    try:
        written = freeze(a.calibration, a.config)
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for path, v in written:
        print(f"{'.'.join(path)} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
