"""Calibration rules (train.calibrate) and the freeze into config.yaml (train.freeze_calibration).

No Flower here: the end-to-end test replaces train.run.run_resumable with a stub that writes the logs
a level-1 run would write, from synthetic val-MSE curves and update norms.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest
import yaml

import common
from train import calibrate as cal
from train import freeze_calibration as fz
from train.accountant import total_eps
from train.run import norms_path
from train.settings import expected_clipping_cuts, MAX_EXPECTED_CUTS, resolve

REPO = Path(__file__).resolve().parents[1]
INF = math.inf


# --------------------------------------------------------------------------- pure rules

def test_final_val_mse():
    assert cal.final_val_mse([9, 9, 1, 2, 3], 3) == 2.0
    assert cal.final_val_mse([1, 2, float("nan")], 3) == INF
    assert cal.final_val_mse([1, INF, 2], 2) == INF
    assert cal.final_val_mse([1, 2], 3) == INF   # too short


def _pick(finals, grid=(0.01, 0.03, 0.1, 0.3, 1.0), below=(0.003, 0.001), above=(3.0, 10.0), max_ext=2):
    calls = []

    def final_of(lr):
        calls.append(lr)
        return finals[lr]
    return cal.pick_lr(list(grid), list(below), list(above), max_ext, final_of), calls


def test_pick_lr_interior_needs_no_extension():
    r, calls = _pick({0.01: 5, 0.03: 3, 0.1: 1, 0.3: 2, 1.0: INF})
    assert r["lr"] == 0.1 and r["extensions"] == [] and not r["at_grid_edge"]
    assert calls == [0.01, 0.03, 0.1, 0.3, 1.0]


def test_pick_lr_extends_below_until_interior():
    # 0.003 wins but is again the lowest tried, so 0.001 is run too; then 0.003 is interior.
    r, _ = _pick({0.001: 3, 0.003: 0.5, 0.01: 1, 0.03: 2, 0.1: 3, 0.3: 4, 1.0: 5})
    assert r["extensions"] == [0.003, 0.001] and r["lr"] == 0.003 and not r["at_grid_edge"]


def test_pick_lr_stops_after_max_extensions_and_records_edge():
    r, _ = _pick({0.001: 0.1, 0.003: 0.5, 0.01: 1, 0.03: 2, 0.1: 3, 0.3: 4, 1.0: 5})
    assert r["extensions"] == [0.003, 0.001] and r["lr"] == 0.001 and r["at_grid_edge"]
    assert r["tried"] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    assert set(r["final_val_mse"]) == {"0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"}
    r, _ = _pick({0.001: 0.1, 0.003: 0.5, 0.01: 1, 0.03: 2, 0.1: 3, 0.3: 4, 1.0: 5}, max_ext=1)
    assert r["extensions"] == [0.003] and r["lr"] == 0.003 and r["at_grid_edge"]


def test_pick_lr_extends_above():
    r, _ = _pick({0.01: 5, 0.03: 4, 0.1: 3, 0.3: 2, 1.0: 1, 3.0: INF})
    assert r["extensions"] == [3.0] and r["lr"] == 1.0 and not r["at_grid_edge"]


def test_pick_lr_tie_goes_to_smaller():
    r, _ = _pick({0.01: 5, 0.03: 1, 0.1: 1, 0.3: 2, 1.0: 3})
    assert r["lr"] == 0.03


def test_best_so_far_ignores_nan():
    assert cal.best_so_far([float("nan"), 3, 4, 2, float("nan"), 5]) == [INF, 3, 3, 2, 2, 2]


def test_stopping_round():
    # m = 10, 5, 4, 3.99, 3.99, ...: from t=3 the next 2 rounds improve by < 1%.
    curve = [10, 5, 4, 3.99, 3.995, 3.99, 3.99]
    assert cal.stopping_round(curve, window=2, min_improvement=0.01) == {"T": 3, "hit_cap": False}
    steady = [100 * 0.9 ** t for t in range(20)]
    assert cal.stopping_round(steady, window=5, min_improvement=0.01) == {"T": 20, "hit_cap": True}


def test_stopping_round_uses_best_so_far_not_raw_curve():
    # A spike does not stop it early: on the raw curve t=1 (10 -> 50) would already "stall";
    # best-so-far is 10, 8, 8, 6, 4, 4, 4, which first stalls over t=5..7.
    curve = [10, 8, 50, 6, 4, 4, 4]
    assert cal.stopping_round(curve, window=2, min_improvement=0.01) == {"T": 5, "hit_cap": False}


@pytest.mark.parametrize("x,want", [(12.18, 13), (13, 13), (0.1234, 0.13), (0.12, 0.12), (999.1, 1000),
                                    (1.001, 1.1)])
def test_round_up_sig(x, want):
    assert cal.round_up_sig(x) == pytest.approx(want)
    assert cal.round_up_sig(x) >= x


def test_median_norm():
    assert cal.median_norm([3, 1, 2]) == 2
    with pytest.raises(ValueError):
        cal.median_norm([])


def test_choose_clipping_range_checks_every_pair():
    P = common.num_params(common.Model())
    C, T, ns, eps = 1.0, {5: 300, 10: 300, 20: 300}, [5, 10, 20], [1.0, 8.0]
    sigma = {e: common.sigma_for(e, C, 1e-5) for e in eps}
    r = cal.choose_clipping_range(C, sigma, T, ns, eps, P)
    cr = r["clipping_range"]
    assert cr >= r["from"]["min_clipping_range"] + C
    assert len(r["checks"]) == 6 and all(c["ok"] for c in r["checks"])
    for c in r["checks"]:
        assert expected_clipping_cuts(c["n"], T[c["n"]], P, cr, sigma[c["eps"]]) <= MAX_EXPECTED_CUTS


def test_choose_clipping_range_refuses_a_failing_pair(monkeypatch):
    real = cal.expected_clipping_cuts
    monkeypatch.setattr(cal, "expected_clipping_cuts",
                        lambda n, T, P, cr, s: 1.0 if n == 20 else real(n, T, P, cr, s))
    sigma = {1.0: common.sigma_for(1.0, 1.0, 1e-5)}
    with pytest.raises(ValueError, match="fails the pre-run check"):
        cal.choose_clipping_range(1.0, sigma, {5: 100, 20: 100}, [5, 20], [1.0], 1000)


def test_refuse_if_calibrated():
    cfg = common.load_config(REPO / "config.yaml")
    cfg["calibration"].update(lr=None, C=None, clipping_range=None, T={5: None})
    cfg["dp"]["sigma"] = {1: None}
    cal.refuse_if_calibrated(cfg)
    cfg["dp"]["sigma"] = {1: 2.0}
    with pytest.raises(ValueError, match=r"dp.sigma\[1\]"):
        cal.refuse_if_calibrated(cfg)


# --------------------------------------------------------------------------- end to end (stubbed runs)

def _null_config(tmp_path: Path, max_rounds=60, lr_rounds=20) -> Path:
    """A copy of the real config.yaml (comments kept) with calibration nulled and short run lengths."""
    p = tmp_path / "config.yaml"
    shutil.copy(REPO / "config.yaml", p)
    cfg = yaml.safe_load(p.read_text())
    assert cfg["calibration"]["lr"] is None, "the repo config.yaml is already frozen: update this test"
    text = (p.read_text().replace("rounds: 200", f"rounds: {lr_rounds}")
            .replace("max_rounds: 1000", f"max_rounds: {max_rounds}").replace("window: 50", "window: 5"))
    p.write_text(text)
    return p


def _curve(lr: float, n: int, R: int) -> list[float]:
    """Synthetic val MSE: lr 0.1 is best, lr 1.0 diverges; larger n plateaus later."""
    if lr >= 1.0:
        return [1.0] * (R - 1) + [float("nan")]
    base = 0.01 + abs(math.log10(lr) + 1)   # minimum at lr = 0.1
    stop = 10 + n                           # plateau from round `stop`
    return [base + 1.0 / min(t, stop) for t in range(1, R + 1)]


def _norm(n: int, r: int, i: int) -> float:
    return 0.5 + 0.01 * i + 0.001 * r + 0.0001 * n


@pytest.fixture
def stub_runs(monkeypatch):
    done = []

    def fake_run_resumable(s):
        done.append((s.n, s.lr, s.T, s.logs_dir))
        s.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(s.logs_dir / f"{s.run_name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "val_mse"])
            for r, v in enumerate(_curve(s.lr, s.n, s.T), 1):
                w.writerow([r, v])
        with open(norms_path(s), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "house_id", "update_norm"])
            for r in range(1, s.T + 1):
                for i, h in enumerate(s.house_ids):
                    w.writerow([r, h, _norm(s.n, r, i)])
        return True
    monkeypatch.setattr(cal, "run_resumable", fake_run_resumable)
    return done


def test_calibrate_end_to_end_then_freeze(tmp_path, stub_runs):
    cfg_path = _null_config(tmp_path)
    out = tmp_path / "cal"
    res = cal.calibrate(out, cfg_path)
    v = res["values"]

    # lr: grid 0.01..1.0, 0.1 wins, no extension. Each run in its own directory, level-1 settings.
    assert v["lr"] == 0.1 and res["lr"]["extensions"] == [] and res["lr"]["final_val_mse"]["1"] == INF
    lr_runs = [d for d in stub_runs if d[0] == 20 and d[2] == 20]
    assert sorted(d[1] for d in lr_runs) == [0.01, 0.03, 0.1, 0.3, 1.0]
    assert len({d[3] for d in stub_runs}) == len(stub_runs)

    # T[n]: plateau from round 10 + n; window 5. 60-round cap.
    for n in (5, 10, 20):
        info = res["T"][str(n)]
        assert v["T"][str(n)] == cal.stopping_round(_curve(0.1, n, 60), 5, 0.01)["T"]
        assert not info["hit_cap"] and info["num_norms"] == n * v["T"][str(n)]

    # C: median over rounds 1..T[n] only, pooled over n.
    pooled = [_norm(n, r, i) for n in (5, 10, 20) for r in range(1, v["T"][str(n)] + 1) for i in range(n)]
    assert v["C"] == pytest.approx(cal.median_norm(pooled))

    # sigma, clipping_range, privacy totals at eps 1 and 8 for every n.
    assert set(v["sigma"]) == {"1", "8"}
    assert v["sigma"]["1"] == pytest.approx(common.sigma_for(1.0, v["C"], 1e-5))
    assert all(c["ok"] for c in res["clipping_range"]["checks"]) and len(res["clipping_range"]["checks"]) == 6
    tot = {(p["n"], p["eps_per_round"]): p["eps_total"] for p in res["privacy_totals"]}
    assert set(tot) == {(n, e) for n in (5, 10, 20) for e in (1.0, 8.0)}
    assert tot[(5, 1.0)] == pytest.approx(total_eps(v["C"], v["sigma"]["1"], v["T"]["5"], 1e-5))
    assert "test split is never read" in res["rules"]["data"]

    on_disk = json.loads((out / "calibration.json").read_text())
    assert on_disk["values"] == json.loads(json.dumps(v))

    # Freeze: comments kept, values read back by train.settings, refuses a second time.
    before = cfg_path.read_text()
    fz.freeze(out / "calibration.json", cfg_path)
    after = cfg_path.read_text()
    assert [l for l in before.splitlines() if "null" not in l] == \
           [l for l in after.splitlines() if l in before.splitlines() and "null" not in l]
    assert sum(a != b for a, b in zip(before.splitlines(), after.splitlines())) == 8
    assert len(before.splitlines()) == len(after.splitlines())
    c_line = next(l for l in after.splitlines() if l.startswith("  C: "))
    assert c_line.index("#") == next(l for l in before.splitlines() if l.startswith("  C: ")).index("#")
    s = resolve(3, 1, 5, 0, config_path=cfg_path)
    assert (s.lr, s.T, s.C, s.clipping_range) == (v["lr"], v["T"]["5"], v["C"], v["clipping_range"])
    assert s.sigma == v["sigma"]["1"]
    with pytest.raises(ValueError, match="--lr"):
        resolve(1, "none", 5, 0, lr=0.3, config_path=cfg_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fz.freeze(out / "calibration.json", cfg_path)
    assert cfg_path.read_text() == after

    # calibrate refuses a frozen config before running anything.
    n_runs = len(stub_runs)
    with pytest.raises(ValueError, match="already has calibration values"):
        cal.calibrate(tmp_path / "again", cfg_path)
    assert len(stub_runs) == n_runs


def test_calibrate_records_hit_cap_and_keeps_max_rounds(tmp_path, stub_runs):
    cfg_path = _null_config(tmp_path, max_rounds=12)   # plateaus start at round 15+: never stops
    res = cal.calibrate(tmp_path / "cal", cfg_path)
    for n in ("5", "10", "20"):
        assert res["T"][n]["hit_cap"] and res["values"]["T"][n] == 12


def test_freeze_refuses_wrong_n_and_leaves_config(tmp_path):
    p = tmp_path / "config.yaml"
    shutil.copy(REPO / "config.yaml", p)
    before = p.read_text()
    j = tmp_path / "calibration.json"
    j.write_text(json.dumps({"values": {"lr": 0.1, "T": {"5": 10, "10": 10}, "C": 1.0, "clipping_range": 13.0,
                                       "sigma": {"1": 4.0, "8": 0.6}}}))
    with pytest.raises(ValueError, match="needs"):
        fz.freeze(j, p)
    assert fz.main(["--calibration", str(j), "--config", str(p)]) == 2
    assert p.read_text() == before


def test_freeze_partial_config_refuses_everything(tmp_path):
    p = tmp_path / "config.yaml"
    shutil.copy(REPO / "config.yaml", p)
    p.write_text(p.read_text().replace("    8: null", "    8: 0.5"))
    before = p.read_text()
    j = tmp_path / "calibration.json"
    j.write_text(json.dumps({"values": {"lr": 0.1, "T": {"5": 10, "10": 10, "20": 10}, "C": 1.0,
                                       "clipping_range": 13.0, "sigma": {"1": 4.0, "8": 0.6}}}))
    with pytest.raises(ValueError, match="dp.sigma.8"):
        fz.freeze(j, p)
    assert p.read_text() == before


def test_calibrate_notebook_summary_reads_calibration_json(tmp_path, stub_runs):
    """Cell f of notebooks/calibrate.ipynb runs on a calibration.json written by calibrate()."""
    out = tmp_path / "cal"
    cal.calibrate(out, _null_config(tmp_path))
    nb = json.loads((REPO / "notebooks/calibrate.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("# f. Summary"))
    pytest.importorskip("pandas")
    exec(compile(cell, "calibrate.ipynb:f", "exec"), {"OUT": str(out)})
