"""Resolve the settings of one training run from config.yaml (plus the calibration overrides).

config.yaml is the single source of settings. The only overrides are --lr and --T, and they are
accepted ONLY while the matching calibration value in config.yaml is still null. A needed value
that is null with no override is an error: the run refuses to start.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import common
import data

_REPO_ROOT = Path(__file__).resolve().parent.parent

IMPLEMENTED_LEVELS = (1,)  # levels 2 and 3 are added in later stages


@dataclass(frozen=True)
class RunSettings:
    level: int
    eps: float | None          # None = no noise ("eps0" in file names)
    n: int
    k: int
    lr: float
    T: int
    C: float | None            # None at level 1 (no clipping)
    sigma: float | None        # None at level 1 (no noise)
    run_seed: int
    house_ids: tuple[str, ...]  # client i (partition id i) is house_ids[i]
    windows_per_round: int
    num_examples: int          # the weight every client reports (identical for all clients)
    client_num_cpus: float
    recordings_dir: Path
    logs_dir: Path

    @property
    def eps_label(self) -> str:
        return "0" if self.eps is None else f"{self.eps:g}"

    @property
    def run_name(self) -> str:
        return f"L{self.level}_eps{self.eps_label}_n{self.n}_nb{self.k}"


def parse_eps(s: str | float | None) -> float | None:
    if s is None:
        return None
    if isinstance(s, str) and s.strip().lower() in ("none", "null"):
        return None
    eps = float(s)
    if eps <= 0:
        raise ValueError(f"eps must be > 0 or 'none', got {s}")
    return eps


def _calibrated(name: str, cfg_value, override):
    """The config value if it is set; otherwise the override. Refuse both-set and neither-set."""
    if cfg_value is not None:
        if override is not None:
            raise ValueError(
                f"config.yaml already sets calibration {name} = {cfg_value}; "
                f"--{name} is only accepted while it is null. Remove --{name}.")
        return cfg_value
    if override is None:
        raise ValueError(
            f"calibration {name} is null in config.yaml and no --{name} override was given; refusing to run.")
    return override


def resolve(level: int, eps, n: int, k: int, *, lr: float | None = None, T: int | None = None,
            recordings_dir: str | Path | None = None, logs_dir: str | Path | None = None,
            config_path: str | Path = "config.yaml") -> RunSettings:
    cfg = common.load_config(config_path)
    exp, cal = cfg["experiment"], cfg["calibration"]
    eps = parse_eps(eps)

    if level not in exp["levels"]:
        raise ValueError(f"level must be one of {exp['levels']}, got {level}")
    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(f"level {level} is not implemented yet (only {IMPLEMENTED_LEVELS})")
    if level == 1 and eps is not None:
        raise ValueError("level 1 has no noise: use --eps none")
    if exp["clients_per_round"] != "all":
        raise ValueError("only clients_per_round: all is supported (every house takes part every round)")

    lr = float(_calibrated("lr", cal["lr"], lr))
    T = int(_calibrated("T", (cal["T"] or {}).get(n), T))
    if lr <= 0 or T < 1:
        raise ValueError(f"need lr > 0 and T >= 1, got lr={lr}, T={T}")

    # Level 1 uses neither clipping nor noise. Levels 2-3 will read calibration.C and dp.sigma here
    # (and refuse to run while they are null).
    C = sigma = None

    paths = cfg["paths"]
    rec = Path(recordings_dir if recordings_dir is not None else paths["recordings"])
    logs = Path(logs_dir if logs_dir is not None else paths["logs"])
    rec = rec if rec.is_absolute() else _REPO_ROOT / rec
    logs = logs if logs.is_absolute() else _REPO_ROOT / logs

    return RunSettings(
        level=level, eps=eps, n=n, k=k, lr=lr, T=T, C=C, sigma=sigma,
        run_seed=common.run_seed(n, k),
        house_ids=tuple(data.get_neighbourhood(n, k)),
        windows_per_round=int(exp["windows_per_round"]),
        num_examples=int(cfg["simulation"]["client_weight"]),
        client_num_cpus=float(cfg["simulation"]["client_num_cpus"]),
        recordings_dir=rec, logs_dir=logs,
    )
