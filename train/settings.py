"""Resolve the settings of one training run from config.yaml (plus the calibration overrides).

config.yaml is the single source of settings. The only overrides are --lr, --T, --C and
--clipping-range, and each is accepted ONLY while the matching calibration value in config.yaml is
still null. A needed value that is null with no override is an error: the run refuses to start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import common
import data

_REPO_ROOT = Path(__file__).resolve().parent.parent

IMPLEMENTED_LEVELS = (1, 2, 3)

# Pre-run clipping_range check (level 3 with noise): refuse a run whose expected number of values
# beyond the SecAgg+ clipping_range, over the whole run, is above this.
MAX_EXPECTED_CUTS = 0.01


class ClippingRangeTooSmall(ValueError):
    """The SecAgg+ clipping_range is too close to the noise shares: the run is refused before it starts."""


def expected_clipping_cuts(n: int, T: int, P: int, clipping_range: float, sigma: float | None) -> float:
    """Expected number of values with |x| > clipping_range over a whole level-3 run:

        n * T * P * 2 * Phi(-clipping_range / (sigma / sqrt(n)))

    Each of the n*T*P protected values carries noise N(0, (sigma/sqrt(n))^2). The update itself
    (clipped to L2 norm C) is ignored. sigma None or 0 (eps = none) gives 0. 2*Phi(-x) = erfc(x/sqrt 2).
    """
    if not sigma:
        return 0.0
    share = sigma / math.sqrt(n)
    return n * T * P * math.erfc(clipping_range / (share * math.sqrt(2.0)))


def min_clipping_range(n: int, T: int, P: int, sigma: float, max_cuts: float = MAX_EXPECTED_CUTS) -> float:
    """The smallest clipping_range (rounded up to 0.01) whose expected cut count is below max_cuts."""
    lo, hi = 0.0, 1.0
    while expected_clipping_cuts(n, T, P, hi, sigma) >= max_cuts:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if expected_clipping_cuts(n, T, P, mid, sigma) < max_cuts else (mid, hi)
    cr = math.ceil(hi * 100) / 100
    while expected_clipping_cuts(n, T, P, cr, sigma) >= max_cuts:   # guard against rounding
        cr += 0.01
    return round(cr, 2)


def check_clipping_range(s: "RunSettings") -> None:
    """Refuse a level-3 run with noise whose expected clipping_range cuts exceed MAX_EXPECTED_CUTS.
    Levels 1-2 and eps = none skip the check."""
    if s.level != 3 or not s.sigma:
        return
    P = common.num_params(common.Model())
    exp_cuts = expected_clipping_cuts(s.n, s.T, P, s.clipping_range, s.sigma)
    if exp_cuts > MAX_EXPECTED_CUTS:
        raise ClippingRangeTooSmall(
            f"[{s.run_name}] clipping_range {s.clipping_range:g} is too small for the noise shares "
            f"(sigma/sqrt(n) = {s.sigma / math.sqrt(s.n):.4g}): expected {exp_cuts:.4g} values beyond it over "
            f"the run (n*T*P*2*Phi(-clipping_range/(sigma/sqrt(n))), n={s.n}, T={s.T}, P={P}), above "
            f"{MAX_EXPECTED_CUTS}. The smallest clipping_range that brings it under {MAX_EXPECTED_CUTS} is "
            f"{min_clipping_range(s.n, s.T, P, s.sigma):g}. Refusing to start.")


@dataclass(frozen=True)
class RunSettings:
    level: int
    eps: float | None          # None = no noise ("eps0" in file names)
    n: int
    k: int
    lr: float
    T: int
    C: float | None            # None at level 1 (no clipping)
    sigma: float | None        # None at level 1; 0.0 at level 3 with eps = none
    delta: float
    run_seed: int
    house_ids: tuple[str, ...]  # client i (partition id i) is house_ids[i]
    windows_per_round: int
    num_examples: int          # the weight every client reports (identical for all clients)
    client_num_cpus: float
    recordings_dir: Path
    logs_dir: Path
    # SecAgg+ (level 3 only). None = Flower default.
    num_shares: int | None = None
    reconstruction_threshold: int | None = None
    clipping_range: float | None = None
    quantization_range: int | None = None
    modulus_range: int | None = None
    # TEST-ONLY hook: every client sends an all-zero gradient (used to measure the noise alone).
    test_zero_update: bool = False

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
    flag = "--" + name.replace("_", "-")
    if cfg_value is not None:
        if override is not None:
            raise ValueError(
                f"config.yaml already sets calibration {name} = {cfg_value}; "
                f"{flag} is only accepted while it is null. Remove {flag}.")
        return cfg_value
    if override is None:
        raise ValueError(
            f"calibration {name} is null in config.yaml and no {flag} override was given; refusing to run.")
    return override


def resolve(level: int, eps, n: int, k: int, *, lr: float | None = None, T: int | None = None,
            C: float | None = None, clipping_range: float | None = None,
            out_dir: str | Path | None = None,
            recordings_dir: str | Path | None = None, logs_dir: str | Path | None = None,
            config_path: str | Path = "config.yaml", rounds: int | None = None,
            test_zero_update: bool = False) -> RunSettings:
    """Settings for one run.

    out_dir: if given, recordings and logs go to out_dir/recordings and out_dir/logs (unless
        recordings_dir / logs_dir are given explicitly).
    rounds: run exactly this many rounds, ignoring calibration.T (timing test only).
    """
    cfg = common.load_config(config_path)
    exp, cal, dp, sa = cfg["experiment"], cfg["calibration"], cfg["dp"], cfg["secaggplus"]
    eps = parse_eps(eps)

    if level not in exp["levels"]:
        raise ValueError(f"level must be one of {exp['levels']}, got {level}")
    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(f"level {level} is not implemented yet (only {IMPLEMENTED_LEVELS})")
    if level == 1 and eps is not None:
        raise ValueError("level 1 has no noise: use --eps none")
    if level == 2 and eps is None:
        raise ValueError("level 2 (local DP) needs a number for --eps")
    if exp["clients_per_round"] != "all":
        raise ValueError("only clients_per_round: all is supported (every house takes part every round)")

    lr = float(_calibrated("lr", cal["lr"], lr))
    if rounds is not None:
        if T is not None:
            raise ValueError("give either rounds or T, not both")
        T = int(rounds)
    else:
        T = int(_calibrated("T", (cal["T"] or {}).get(n), T))
    if lr <= 0 or T < 1:
        raise ValueError(f"need lr > 0 and T >= 1, got lr={lr}, T={T}")

    delta = float(dp["delta"])
    Cv = sigma = None
    sec = {}
    if level in (2, 3):
        Cv = float(_calibrated("C", cal["C"], C))
        if Cv <= 0:
            raise ValueError(f"need C > 0, got {Cv}")
        cfg_sigma = None if eps is None else (dp.get("sigma") or {}).get(int(eps) if float(eps).is_integer() else eps)
        sigma = float(cfg_sigma) if cfg_sigma is not None else common.sigma_for(eps, Cv, delta)
    elif C is not None or clipping_range is not None:
        raise ValueError("--C / --clipping-range only apply to levels 2 and 3")
    if level == 2 and clipping_range is not None:
        raise ValueError("--clipping-range only applies to level 3 (SecAgg+)")
    if level == 3:
        cr = float(_calibrated("clipping_range", cal["clipping_range"], clipping_range))
        if sa.get("clipping_range") is not None and float(sa["clipping_range"]) != cr:
            raise ValueError(f"secaggplus.clipping_range ({sa['clipping_range']}) differs from "
                             f"calibration.clipping_range ({cr})")
        sec = dict(num_shares=int(sa["num_shares"]), reconstruction_threshold=int(sa["reconstruction_threshold"]),
                   clipping_range=cr, quantization_range=sa.get("quantization_range"),
                   modulus_range=sa.get("modulus_range"))
        if n < sec["num_shares"]:
            raise ValueError(f"SecAgg+ needs n >= num_shares ({sec['num_shares']}), got n={n}")

    paths = cfg["paths"]
    if out_dir is not None:
        out = Path(out_dir)
        recordings_dir = recordings_dir if recordings_dir is not None else out / "recordings"
        logs_dir = logs_dir if logs_dir is not None else out / "logs"
    rec = Path(recordings_dir if recordings_dir is not None else paths["recordings"])
    logs = Path(logs_dir if logs_dir is not None else paths["logs"])
    rec = rec if rec.is_absolute() else _REPO_ROOT / rec
    logs = logs if logs.is_absolute() else _REPO_ROOT / logs

    return RunSettings(
        level=level, eps=eps, n=n, k=k, lr=lr, T=T, C=Cv, sigma=sigma, delta=delta,
        run_seed=common.run_seed(n, k),
        house_ids=tuple(data.get_neighbourhood(n, k)),
        windows_per_round=int(exp["windows_per_round"]),
        num_examples=int(cfg["simulation"]["client_weight"]),
        client_num_cpus=float(cfg["simulation"]["client_num_cpus"]),
        recordings_dir=rec, logs_dir=logs,
        test_zero_update=bool(test_zero_update),
        **sec,
    )
