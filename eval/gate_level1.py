"""Level-1 gate: must PASS before the full grid. Driven by notebooks/gate_level1.ipynb, one step per cell:

    python -m eval.gate_level1 calibration --root ROOT   the frozen lr, T[n], C (exit 2 if not frozen yet)
    python -m eval.gate_level1 train       --root ROOT   the 9 level-1 runs (n x k), 27 recordings
    python -m eval.gate_level1 attack      --root ROOT   TS-Inverse + learned inverter on the 27 recordings
    python -m eval.gate_level1 score       --root ROOT   eval.score_all --blind same_date
    python -m eval.gate_level1 check       --root ROOT   PASS/FAIL per part and overall -> gate_level1.json

Every step skips work whose output already exists under ROOT (a Colab disconnect = run all cells again):
train: train.run.run_resumable skips complete runs; attack: attacks.pipeline skips finished attacks; score: skipped
if per_window.csv covers all 54 outputs and is newer than every one of them.

Layout under ROOT (MyDrive/smart-meter-fl in Colab), the same places notebooks/attack_ts_inverse.ipynb uses, so
the full grid reuses the level-1 work:
    ROOT/grid/recordings/L1_eps0_n{n}_nb{k}_r{round}.pt      (train.run --out ROOT/grid; logs in ROOT/grid/logs)
    ROOT/attacks/attack_outputs/<recording>__{tsinverse,inverter}.pt (+ .json diagnostics)
    ROOT/attacks/aux_cache/                                  one recording's aux set at a time
    ROOT/attacks/reproduce/reproduction.json                 attacks.reproduce_ts_inverse (attack_ts_inverse.ipynb e)
    ROOT/gate_level1/results/                                eval.score_all output (incl. per_window.csv)
    ROOT/gate_level1/gate_level1.json                        the gate verdict

Gate parts (config.yaml gate):
  (a) reproduction: the table2_config row's mean input and target sMAPE (paper scale, 0-1) are each at most
      gate.repro_factor x Table II (attacks.reproduce.paper_fcn_london_b1). Doing better than the paper passes.
  (b) inverter vs blind: d = sMAPE(blind) - sMAPE(inverter) for every (recording, matched house-window) of the
      27 recordings (eval.score_all per_window.csv); PASS if mean(d) > 2 * SE(d), SE = std(ddof 1) / sqrt(count).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

import common
from eval import metrics as M

ATTACK_NAMES = ("tsinverse", "inverter")
REPRO_ROW = "table2_config"     # their Table II configuration (attacks/reproduce_ts_inverse.py)


class NotFrozen(RuntimeError):
    """config.yaml has no frozen calibration yet."""


def layout(root: str | Path) -> dict:
    root = Path(root)
    return {"root": root, "grid": root / "grid", "recordings": root / "grid" / "recordings",
            "attack_outputs": root / "attacks" / "attack_outputs", "aux_cache": root / "attacks" / "aux_cache",
            "reproduction": root / "attacks" / "reproduce" / "reproduction.json",
            "results": root / "gate_level1" / "results", "gate_json": root / "gate_level1" / "gate_level1.json"}


# --------------------------------------------------------------------------- calibration and the 27 recordings

def frozen_calibration(cfg: dict) -> dict:
    """{"lr", "T": {n: T[n]}, "C"} from config.yaml calibration; NotFrozen names every value still null."""
    cal, ns = cfg["calibration"], cfg["experiment"]["n"]
    T = cal.get("T") or {}
    missing = [k for k in ("lr", "C") if cal.get(k) is None] + [f"T[{n}]" for n in ns if T.get(n) is None]
    if missing:
        raise NotFrozen(f"config.yaml calibration is not frozen yet: {', '.join(missing)} null. Run "
                        "notebooks/calibrate.ipynb, then `python -m train.freeze_calibration --calibration "
                        "calibration.json` and commit config.yaml before the level-1 gate.")
    return {"lr": float(cal["lr"]), "T": {int(n): int(T[n]) for n in ns}, "C": float(cal["C"])}


def level1_runs(cfg: dict) -> list[tuple[int, int]]:
    """The level-1 grid runs: every (n, k) of experiment.n x experiment.neighbourhoods_k."""
    return [(int(n), int(k)) for n in cfg["experiment"]["n"] for k in cfg["experiment"]["neighbourhoods_k"]]


def level1_settings(root: str | Path, config_path: str | Path = "config.yaml") -> list:
    """train.settings.RunSettings of the 9 level-1 runs, writing to ROOT/grid (the frozen lr and T[n])."""
    from train.settings import resolve
    cfg = common.load_config(config_path)
    frozen_calibration(cfg)
    return [resolve(1, "none", n, k, out_dir=layout(root)["grid"], config_path=config_path)
            for n, k in level1_runs(cfg)]


def level1_recordings(root: str | Path, config_path: str | Path = "config.yaml") -> list[Path]:
    """The 27 recording paths, exactly as train.run writes them (rounds 1, (T+1)//2, T of each run)."""
    from train.recorder import recording_path, rounds_to_record
    return [recording_path(s, r) for s in level1_settings(root, config_path) for r in rounds_to_record(s.T)]


# --------------------------------------------------------------------------- steps

def train_level1(root: str | Path, config_path: str | Path = "config.yaml", run=None) -> None:
    """The 9 runs through train.run.run_resumable (complete runs are skipped, partial ones redone)."""
    if run is None:
        from train.run import run_resumable as run
    for s in level1_settings(root, config_path):
        run(s)


def attack_level1(root: str | Path, config_path: str | Path = "config.yaml", runner=None) -> dict:
    """Both attacks on the 27 recordings through attacks.pipeline (finished attacks are skipped)."""
    from attacks import pipeline
    L, recs = layout(root), level1_recordings(root, config_path)
    missing = [p.name for p in recs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} level-1 recordings missing (run the train step): {missing}")
    return pipeline.attack_all(recs, L["attack_outputs"], L["aux_cache"],
                               **({} if runner is None else {"runner": runner}))


def attack_outputs(recordings: list, attack_out: str | Path) -> list[Path]:
    from attacks import pipeline
    return [pipeline.output_path(p, attack_out, a) for p in recordings for a in ATTACK_NAMES]


def scores_complete(results: str | Path, recordings: list, attack_out: str | Path) -> bool:
    """True if results/per_window.csv has rows for every (recording, attack) and is newer than every output."""
    pw = Path(results) / "per_window.csv"
    outs = attack_outputs(recordings, attack_out)
    if not pw.exists() or not all(p.exists() for p in outs):
        return False
    if pw.stat().st_mtime < max(p.stat().st_mtime for p in outs):
        return False
    try:
        df = pd.read_csv(pw, usecols=["recording", "attack"])
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return False
    have = set(zip(df["recording"], df["attack"]))
    return all((Path(p).stem, a) in have for p in recordings for a in ATTACK_NAMES)


def score_level1(root: str | Path, config_path: str | Path = "config.yaml") -> bool:
    """eval.score_all --blind same_date into ROOT/gate_level1/results; False if skipped (already complete)."""
    from eval import score_all
    L, recs = layout(root), level1_recordings(root, config_path)
    outs = attack_outputs(recs, L["attack_outputs"])
    missing = [p.name for p in outs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} attack outputs missing (run the attack step): {missing}")
    if scores_complete(L["results"], recs, L["attack_outputs"]):
        print(f"scores in {L['results']} are complete, skipping", flush=True)
        return False
    score_all.main(["--recordings", str(L["recordings"]), "--attacks", str(L["attack_outputs"]),
                    "--out", str(L["results"]), "--blind", "same_date"])
    return True


# --------------------------------------------------------------------------- gate checks

def _num(x):
    return None if x is None or not math.isfinite(x) else float(x)


def check_reproduction(repro: dict | None, ref: dict, factor: float) -> dict:
    """Part (a). repro: reproduction.json (None if missing); ref: Table II {input_smape, target_smape}."""
    th_in, th_tar = factor * float(ref["input_smape"]), factor * float(ref["target_smape"])
    out = {"pass": False, "row": REPRO_ROW, "factor": factor, "input_threshold": th_in, "target_threshold": th_tar,
           "input_smape": None, "target_smape": None, "reason": None}
    if repro is None:
        out["reason"] = "reproduction.json missing: run attacks.reproduce_ts_inverse (attack_ts_inverse.ipynb cell e)"
    elif repro.get("data_is_fake"):
        out["reason"] = "reproduction.json was made on the FAKE stub"
    elif REPRO_ROW not in repro.get("rows", {}):
        out["reason"] = f"reproduction.json has no {REPRO_ROW} row"
    else:
        a = repro["rows"][REPRO_ROW]["attack"]
        si, st = float(a["smape_input"]["mean"]), float(a["smape_target"]["mean"])
        out.update(input_smape=si, target_smape=st, input_pass=si <= th_in, target_pass=st <= th_tar)
        out["pass"] = bool(out["input_pass"] and out["target_pass"])
    return out


def check_inverter_vs_blind(per_window: pd.DataFrame | None, stems: list[str]) -> dict:
    """Part (b). per_window: eval.score_all's per_window.csv; stems: the 27 recording names (no .pt)."""
    out = {"pass": False, "mean": None, "se": None, "ratio": None, "count": 0, "missing": list(stems), "reason": None}
    if per_window is None:
        out["reason"] = "per_window.csv missing: run the score step"
        return out
    d = per_window[(per_window["attack"] == "inverter") & per_window["recording"].isin(stems)]
    out["missing"] = sorted(set(stems) - set(d["recording"]))
    m, se = M.mean_se(d["smape_blind"] - d["smape_attack"])
    ratio = m / se if math.isfinite(se) and se > 0 else math.nan
    out.update(mean=_num(m), se=_num(se), ratio=_num(ratio), count=int(len(d)))
    if out["missing"]:
        out["reason"] = f"{len(out['missing'])} recordings have no inverter scores"
    elif math.isnan(se):
        out["reason"] = "fewer than 2 paired samples"
    else:
        out["pass"] = bool(m > 2 * se)
    return out


def gate_check(root: str | Path, config_path: str | Path = "config.yaml") -> dict:
    """Both parts, printed and written to ROOT/gate_level1/gate_level1.json. Returns the verdict."""
    cfg = common.load_config(config_path)
    L = layout(root)
    recs = level1_recordings(root, config_path)
    repro = json.loads(L["reproduction"].read_text()) if L["reproduction"].exists() else None
    pw_path = L["results"] / "per_window.csv"
    pw = pd.read_csv(pw_path) if pw_path.exists() else None
    a = check_reproduction(repro, cfg["attacks"]["reproduce"]["paper_fcn_london_b1"], float(cfg["gate"]["repro_factor"]))
    b = check_inverter_vs_blind(pw, [p.stem for p in recs])
    verdict = {"pass": bool(a["pass"] and b["pass"]), "reproduction": a, "inverter_vs_blind": b,
               "calibration": frozen_calibration(cfg), "recordings": [p.name for p in recs]}
    L["gate_json"].parent.mkdir(parents=True, exist_ok=True)
    L["gate_json"].write_text(json.dumps(verdict, indent=1))

    def pf(x):
        return "PASS" if x else "FAIL"
    f = lambda x: "n/a" if x is None else f"{x:.3g}"  # noqa: E731
    print(f"(a) reproduction ({REPRO_ROW}, x{a['factor']:g} Table II): {pf(a['pass'])}   input sMAPE "
          f"{f(a['input_smape'])} <= {a['input_threshold']:.3g}, target sMAPE {f(a['target_smape'])} <= "
          f"{a['target_threshold']:.3g}" + (f"   [{a['reason']}]" if a["reason"] else ""))
    print(f"(b) inverter vs blind (same_date): {pf(b['pass'])}   mean(blind - inverter) {f(b['mean'])} sMAPE %, "
          f"SE {f(b['se'])}, mean/SE {f(b['ratio'])} (needs > 2), n = {b['count']}"
          + (f"   [{b['reason']}]" if b["reason"] else ""))
    print(f"overall: {pf(verdict['pass'])}")
    print(f"written: {L['gate_json']}")
    return verdict


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.gate_level1", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["calibration", "train", "attack", "score", "check"])
    p.add_argument("--root", required=True, help="MyDrive/smart-meter-fl in Colab")
    p.add_argument("--config", default="config.yaml", help=argparse.SUPPRESS)
    p.add_argument("--allow-fake", action="store_true", help=argparse.SUPPRESS)        # tests only
    a = p.parse_args(argv)
    try:
        cal = frozen_calibration(common.load_config(a.config))
    except NotFrozen as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.step == "calibration":
        print(f"lr = {cal['lr']:g}   C = {cal['C']:g}   T = {cal['T']}")
        return 0
    if a.step in ("train", "attack", "score"):
        import data
        if getattr(data, "FAKE", False) and not a.allow_fake:
            print("error: data.py is the FAKE stub; the gate needs the real London data", file=sys.stderr)
            return 2
    if a.step == "train":
        train_level1(a.root, a.config)
    elif a.step == "attack":
        res = attack_level1(a.root, a.config)
        if any(res["failed"].values()):
            return 1
    elif a.step == "score":
        score_level1(a.root, a.config)
    else:
        return 0 if gate_check(a.root, a.config)["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
