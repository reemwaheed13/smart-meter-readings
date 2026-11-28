"""Both attacks on a list of recordings, one recording at a time:
python -m attacks.pipeline --recordings-dir DIR --out ATTACK_OUT --aux-cache AUX_CACHE

Per recording: TS-Inverse (attacks.ts_inverse), then the learned inverter (attacks.learned_inverter), each in
its own process, both on the SAME aux set (attacks/aux_gradients.py, built in AUX_CACHE by the first attack and
reused by the second). An attack whose outputs already exist with the current settings is skipped without
starting a process (ts_inverse.completed / learned_inverter.completed). The recording's aux set is deleted only
once BOTH attacks on it are finished, so the Drive holds about one set at a time; after a failure it stays, and a
re-run reuses it and runs only the attack that is missing.

Used by notebooks/attack_ts_inverse.ipynb (every recording in the grid's recordings/) and by the level-1 gate
(eval.gate_level1, its 27 recordings).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from attacks import learned_inverter, ts_inverse

_REPO_ROOT = Path(__file__).resolve().parent.parent

# (name in the output file, module), in the order they run
ATTACKS = (("tsinverse", "attacks.ts_inverse"), ("inverter", "attacks.learned_inverter"))


def output_path(recording: str | Path, out: str | Path, attack: str) -> Path:
    """OUT/<recording stem>__<attack>.pt, where eval.score_all looks for it."""
    return (ts_inverse.output_paths(recording, out)[0] if attack == "tsinverse"
            else learned_inverter.output_paths(recording, out)[0])


def attack_done(recording: str | Path, out: str | Path, attack: str) -> bool:
    """True if `attack`'s outputs on `recording` exist in `out` with the settings in config.yaml."""
    if attack == "tsinverse":
        s = ts_inverse.load_settings()
        return ts_inverse.completed(recording, out, s, bool(s["inversion_model"]["enabled"])) is not None
    if attack == "inverter":
        return learned_inverter.completed(recording, out, learned_inverter.load_settings()) is not None
    raise ValueError(f"unknown attack {attack!r}")


def drop_aux(recording: str | Path, aux_cache: str | Path) -> list[Path]:
    """Delete the recording's aux set(s) from aux_cache; returns the deleted files."""
    stem, d = Path(recording).stem, Path(aux_cache)
    gone = sorted(d.glob(f"{stem}__aux_*.pt")) if d.exists() else []
    for f in gone:
        f.unlink()
    return gone


def run_subprocess(attack: str, recording: str | Path, out: str | Path, aux_cache: str | Path) -> bool:
    """One attack on one recording in its own process; prints its last progress line, or its tail on failure."""
    module = dict(ATTACKS)[attack]
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", module, "--recording", str(recording), "--out", str(out),
                        "--aux-cache", str(aux_cache)], cwd=_REPO_ROOT, capture_output=True, text=True)
    last = [line for line in r.stdout.splitlines() if line.startswith("[")][-1:]
    print(f"    {attack:10s} {time.time() - t0:6.0f}s " + (last[0] if last else ""), flush=True)
    if r.returncode:
        print("    !!! FAILED:\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:]), flush=True)
    return r.returncode == 0


Runner = Callable[[str, "str | Path", "str | Path", "str | Path"], bool]


def attack_all(recordings: list, out: str | Path, aux_cache: str | Path, runner: Runner = run_subprocess) -> dict:
    """Both attacks on every recording (skipping finished ones); drops each recording's aux set once both are
    finished. Returns {"failed": {attack: [recording names]}, "ran": [(name, attack)], "skipped": [...]}."""
    failed = {a: [] for a, _ in ATTACKS}
    ran, skipped = [], []
    for i, p in enumerate(recordings, 1):
        p = Path(p)
        print(f"[{i}/{len(recordings)}] {p.name}", flush=True)
        ok = {}
        for attack, _ in ATTACKS:
            if attack_done(p, out, attack):
                print(f"    {attack:10s} done, skipping", flush=True)
                skipped.append((p.name, attack))
                ok[attack] = True
                continue
            ran.append((p.name, attack))
            ok[attack] = runner(attack, p, out, aux_cache) and attack_done(p, out, attack)
            if not ok[attack]:
                failed[attack].append(p.name)
        if all(ok.values()):
            drop_aux(p, aux_cache)
    print("failed: " + " | ".join(f"{a} {failed[a] or 'none'}" for a, _ in ATTACKS), flush=True)
    return {"failed": failed, "ran": ran, "skipped": skipped}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m attacks.pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings-dir", required=True, help="attacks every L*_r*.pt in it")
    p.add_argument("--out", required=True, help="attack output directory (eval.score_all --attacks)")
    p.add_argument("--aux-cache", required=True, help="aux-set cache shared by the two attacks")
    a = p.parse_args(argv)
    recs = sorted(Path(a.recordings_dir).glob("L*_r*.pt"))
    print(f"{len(recs)} recordings in {a.recordings_dir}", flush=True)
    res = attack_all(recs, a.out, a.aux_cache)
    return 1 if any(res["failed"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
