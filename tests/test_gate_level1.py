"""Level-1 gate (eval/gate_level1.py) and the shared attack pipeline (attacks/pipeline.py). Fake data only:
tiny attacks on tiny recordings; no Flower run."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import common
import data
from attacks import learned_inverter, pipeline, ts_inverse
from attacks.recording import load_recording
from eval import gate_level1 as G
from eval import score_all
from train.env import machine_info

LR = 0.1
W = common.load_config()["experiment"]["windows_per_round"]
T_TEST = {5: 3, 10: 4, 20: 10}          # rounds {1,2,3}, {1,2,4}, {1,5,10}: 3 distinct rounds per run
REF = {"input_smape": 6.3e-06, "target_smape": 2.5e-06}


@pytest.fixture
def frozen_config(tmp_path):
    cfg = common.load_config()
    cfg["calibration"].update(lr=LR, T=dict(T_TEST), C=1.0)
    p = tmp_path / "config.frozen.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def make_recording(n: int, k: int, rnd: int, rec_dir):
    """A level-1 recording like train/ writes, from the fake data, named as train.run names it."""
    houses = data.get_neighbourhood(n, k)
    dates = data.valid_dates(houses, "train")[10:10 + W]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rnd)
        m = common.Model()
    true = data.get_windows(houses, dates, "train")
    views = torch.stack([common.local_update(m, true[h], LR) for h in range(n)])
    rec = {"level": 1, "eps": None, "n": n, "nb": k, "round": rnd, "server_view": views, "view_is_sum": False,
           "global_model": m.state_dict(), "true_windows": true, "house_ids": houses, "dates": dates, "lr": LR,
           "C": None, "sigma": None, "round_time_s": 0.0, "bytes_per_house": 1.0, "env": machine_info()}
    rec_dir.mkdir(parents=True, exist_ok=True)
    p = rec_dir / f"L1_eps0_n{n}_nb{k}_r{rnd}.pt"
    torch.save(rec, p)
    load_recording(p)
    return p


@pytest.fixture
def tiny_attacks(monkeypatch):
    """Both attacks with tiny settings (the same for the done-check and the run), run in-process."""
    ts_load, inv_load = ts_inverse.load_settings, learned_inverter.load_settings
    monkeypatch.setattr(ts_inverse, "load_settings", lambda over=None: ts_load(
        {"steps": 4, "aux_samples": 16, "inversion_model": {"epochs": 1}, **(over or {})}))
    monkeypatch.setattr(learned_inverter, "load_settings", lambda over=None: inv_load(
        {"epochs": 1, "aux_samples": 16, **(over or {})}))
    calls = []

    def runner(attack, p, out, aux, fail=()):
        calls.append((p.name, attack))
        if (p.name, attack) in fail:
            return False
        if attack == "tsinverse":
            ts_inverse.attack_recording(p, out, ts_inverse.load_settings(), True, aux)
        else:
            learned_inverter.attack_recording(p, out, learned_inverter.load_settings(), aux)
        return True
    return runner, calls


# --------------------------------------------------------------------------- calibration

def test_frozen_calibration_refuses_nulls_and_reads_values():
    cfg = common.load_config()
    cfg["calibration"].update(lr=None, T={5: 10, 10: None, 20: None}, C=None)
    with pytest.raises(G.NotFrozen, match=r"lr, C, T\[10\], T\[20\]"):
        G.frozen_calibration(cfg)
    cfg["calibration"].update(lr=0.3, T={5: 10, 10: 20, 20: 30}, C=2.0)
    assert G.frozen_calibration(cfg) == {"lr": 0.3, "T": {5: 10, 10: 20, 20: 30}, "C": 2.0}


def test_cli_stops_with_a_clear_error_if_not_frozen(tmp_path, capsys):
    cfg = common.load_config()
    cfg["calibration"].update(lr=None, C=None, T={5: None, 10: None, 20: None})
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert G.main(["calibration", "--root", str(tmp_path), "--config", str(p)]) == 2
    assert "not frozen yet" in capsys.readouterr().err


# --------------------------------------------------------------------------- the 27 recordings

def test_27_recordings_named_and_placed_where_score_all_expects_them(tmp_path, frozen_config):
    root = tmp_path / "drive"
    L = G.layout(root)
    recs = G.level1_recordings(root, frozen_config)
    assert len(recs) == 27 and len(set(recs)) == 27
    want = {f"L1_eps0_n{n}_nb{k}_r{r}.pt" for n, T in T_TEST.items() for k in (0, 1, 2)
            for r in {1, (T + 1) // 2, T}}
    assert {p.name for p in recs} == want
    assert all(p.parent == L["recordings"] for p in recs)
    # train.run --out ROOT/grid writes exactly these, and score_all finds each recording from each output
    assert [s.recordings_dir for s in G.level1_settings(root, frozen_config)] == [L["recordings"]] * 9
    for p in recs:
        for attack in G.ATTACK_NAMES:
            out = pipeline.output_path(p, L["attack_outputs"], attack)
            assert out.parent == L["attack_outputs"] and out.name == f"{p.stem}__{attack}.pt"
            assert score_all.recording_for(out, L["recordings"]) == (p.stem, attack, str(p))


def test_train_step_runs_the_9_level1_runs(tmp_path, frozen_config):
    seen = []
    G.train_level1(tmp_path, frozen_config, run=seen.append)
    assert [(s.level, s.eps, s.n, s.k, s.T, s.lr) for s in seen] == [
        (1, None, n, k, T_TEST[n], LR) for n in (5, 10, 20) for k in (0, 1, 2)]
    assert {s.logs_dir for s in seen} == {G.layout(tmp_path)["grid"] / "logs"}


# --------------------------------------------------------------------------- skip-if-exists

def test_attack_pipeline_skips_finished_and_drops_aux_only_after_both(tmp_path, tiny_attacks):
    runner, calls = tiny_attacks
    recs = [make_recording(2, 0, 1, tmp_path / "rec"), make_recording(2, 0, 2, tmp_path / "rec")]
    out, aux = tmp_path / "attack_outputs", tmp_path / "aux"

    # The inverter fails on the first recording: its aux set must stay; the second's is dropped.
    res = pipeline.attack_all(recs, out, aux, runner=lambda *a: runner(*a, fail={(recs[0].name, "inverter")}))
    assert res["failed"] == {"tsinverse": [], "inverter": [recs[0].name]}
    assert list(aux.glob(f"{recs[0].stem}__aux_*.pt")) and not list(aux.glob(f"{recs[1].stem}__aux_*.pt"))
    assert pipeline.attack_done(recs[0], out, "tsinverse") and not pipeline.attack_done(recs[0], out, "inverter")

    # Re-run: only the missing attack runs, then that recording's aux set goes too.
    calls.clear()
    res = pipeline.attack_all(recs, out, aux, runner=runner)
    assert calls == [(recs[0].name, "inverter")] and not any(res["failed"].values())
    assert not list(aux.glob("*__aux_*.pt"))

    # Everything finished: nothing runs.
    calls.clear()
    res = pipeline.attack_all(recs, out, aux, runner=runner)
    assert calls == [] and len(res["skipped"]) == 4
    for p in recs:
        for a in G.ATTACK_NAMES:
            assert pipeline.output_path(p, out, a).exists()


def test_attack_done_is_false_when_settings_change(tmp_path, tiny_attacks, monkeypatch):
    runner, _ = tiny_attacks
    p = make_recording(2, 0, 1, tmp_path / "rec")
    out = tmp_path / "attack_outputs"
    pipeline.attack_all([p], out, tmp_path / "aux", runner=runner)
    assert pipeline.attack_done(p, out, "inverter")
    inv_load = learned_inverter.load_settings
    monkeypatch.setattr(learned_inverter, "load_settings", lambda over=None: {**inv_load(over), "epochs": 2})
    assert not pipeline.attack_done(p, out, "inverter") and pipeline.attack_done(p, out, "tsinverse")


def test_scores_complete_and_per_window_csv(tmp_path, tiny_attacks):
    runner, _ = tiny_attacks
    rec_dir, out, res = tmp_path / "rec", tmp_path / "attack_outputs", tmp_path / "results"
    recs = [make_recording(2, 0, 1, rec_dir), make_recording(3, 1, 1, rec_dir)]
    pipeline.attack_all(recs, out, tmp_path / "aux", runner=runner)
    assert not G.scores_complete(res, recs, out)                       # nothing scored yet
    score_all.main(["--recordings", str(rec_dir), "--attacks", str(out), "--out", str(res), "--blind", "same_date"])
    pw = pd.read_csv(res / "per_window.csv")
    assert len(pw) == 2 * (2 + 3) * W                                  # attacks x houses x windows
    assert set(pw["recording"]) == {p.stem for p in recs} and set(pw["attack"]) == set(G.ATTACK_NAMES)
    # per-window sMAPE averages to score_all's per-house sMAPE
    ph = pd.read_csv(res / "per_house.csv")
    g = pw.groupby(["attack", "n", "house"])[["smape_attack", "smape_blind"]].mean().reset_index()
    m = ph.merge(g, on=["attack", "n", "house"], suffixes=("", "_w"))
    assert len(m) == len(ph) and np.allclose(m["smape_attack"], m["smape_attack_w"])
    assert np.allclose(m["smape_blind"], m["smape_blind_w"])

    assert G.scores_complete(res, recs, out)
    t = os.stat(res / "per_window.csv").st_mtime + 10                 # an output redone after scoring
    os.utime(pipeline.output_path(recs[0], out, "inverter"), (t, t))
    assert not G.scores_complete(res, recs, out)
    assert not G.scores_complete(res, recs + [rec_dir / "L1_eps0_n5_nb0_r1.pt"], out)


# --------------------------------------------------------------------------- (a) reproduction

def _repro(si, st, fake=False):
    return {"data_is_fake": fake, "rows": {"table2_config": {"attack": {"smape_input": {"mean": si, "std": 0.0},
                                                                         "smape_target": {"mean": st, "std": 0.0}}}}}


def test_reproduction_pass():
    a = G.check_reproduction(_repro(6.0e-05, 2.4e-05), REF, 10)
    assert a["pass"] and a["input_threshold"] == pytest.approx(6.3e-05) and a["target_threshold"] == pytest.approx(2.5e-05)
    assert G.check_reproduction(_repro(1e-9, 1e-9), REF, 10)["pass"]  # better than the paper passes


@pytest.mark.parametrize("repro,reason", [
    (_repro(6.4e-05, 1e-06), None),                     # input above 10x
    (_repro(1e-06, 2.6e-05), None),                     # target above 10x
    (None, "missing"),
    (_repro(1e-9, 1e-9, fake=True), "FAKE"),
    ({"rows": {"ts_inverse_b1": {}}}, "no table2_config"),
])
def test_reproduction_fail(repro, reason):
    a = G.check_reproduction(repro, REF, 10)
    assert not a["pass"]
    if reason:
        assert reason in a["reason"]


# --------------------------------------------------------------------------- (b) inverter vs blind

def _per_window(stems, diff, seed=0, n_per=20):
    g = np.random.default_rng(seed)
    rows = []
    for s in stems:
        for i in range(n_per):
            blind = 50 + g.normal()
            rows += [dict(recording=s, attack="inverter", smape_attack=blind - diff[i % len(diff)], smape_blind=blind),
                     dict(recording=s, attack="tsinverse", smape_attack=0.0, smape_blind=blind)]
    return pd.DataFrame(rows)


def test_inverter_vs_blind_pass():
    stems = ["a", "b", "c"]
    diff = np.random.default_rng(1).normal(1.0, 1.0, 20)
    b = G.check_inverter_vs_blind(_per_window(stems, diff), stems)
    d = np.tile(diff, 3)
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert b["pass"] and b["count"] == 60 and b["missing"] == []
    assert b["mean"] == pytest.approx(d.mean()) and b["se"] == pytest.approx(se)
    assert b["ratio"] == pytest.approx(d.mean() / se) and b["ratio"] > 2


def test_inverter_vs_blind_fail():
    stems = ["a", "b", "c"]
    diff = np.array([1.0, -1.0] * 10) + 0.05             # mean 0.05, SE ~0.13: not > 2 SE
    b = G.check_inverter_vs_blind(_per_window(stems, diff), stems)
    assert not b["pass"] and 0 < b["ratio"] < 2
    worse = G.check_inverter_vs_blind(_per_window(stems, -np.abs(diff)), stems)   # inverter worse than blind
    assert not worse["pass"] and worse["mean"] < 0
    # a recording without inverter scores fails the gate even with a strong effect elsewhere
    miss = G.check_inverter_vs_blind(_per_window(stems[:2], np.full(20, 5.0) + np.arange(20) * 0.01), stems)
    assert not miss["pass"] and miss["missing"] == ["c"]
    assert not G.check_inverter_vs_blind(None, stems)["pass"]


# --------------------------------------------------------------------------- the whole check

@pytest.mark.parametrize("good", [True, False])
def test_gate_check_writes_verdict(tmp_path, frozen_config, good, capsys):
    root = tmp_path / "drive"
    L = G.layout(root)
    stems = [p.stem for p in G.level1_recordings(root, frozen_config)]
    L["reproduction"].parent.mkdir(parents=True)
    L["reproduction"].write_text(json.dumps(_repro(1e-05, 1e-05)))
    L["results"].mkdir(parents=True)
    diff = np.random.default_rng(2).normal(0.0, 1.0, 20)
    diff = diff - diff.mean() + (2.0 if good else 0.0)       # bad: mean exactly 0
    _per_window(stems, diff).to_csv(L["results"] / "per_window.csv", index=False)
    rc = G.main(["check", "--root", str(root), "--config", str(frozen_config)])
    v = json.loads(L["gate_json"].read_text())
    assert rc == (0 if good else 1) and v["pass"] is good
    assert v["reproduction"]["pass"] and v["inverter_vs_blind"]["pass"] is good
    assert v["inverter_vs_blind"]["count"] == 27 * 20 and len(v["recordings"]) == 27
    out = capsys.readouterr().out
    assert "(a) reproduction" in out and "(b) inverter vs blind" in out
    assert f"overall: {'PASS' if good else 'FAIL'}" in out
