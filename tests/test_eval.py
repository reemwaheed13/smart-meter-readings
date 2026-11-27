"""Tests for Shifa's half: metrics, blind guess and the scoring pipeline (on the fake stub)."""
import numpy as np
import pandas as pd
import pytest
import torch

import common
import data
from eval import metrics as M
from eval.baseline import blind_guess
from eval.score_all import load_attack_output, score_one, summarise
from attacks.recording import load_recording

N, K = 5, 0


# ------------------------------------------------------------------ metrics
def test_smape_basics():
    a = np.array([1.0, 2.0, 0.0])
    assert M.smape(a, a) == 0
    assert M.smape(np.array([1.0]), np.array([0.0])) == pytest.approx(200)
    assert M.smape(np.zeros(3), np.zeros(3)) == 0


def test_matching_recovers_a_shuffle():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.1, 1, (4, 96))
    perm = [2, 0, 3, 1]
    assert np.allclose(M.match_windows(true[perm], true), true)


def _day(rise_slot, peak_slot, daytime=0.5, night=0.1):
    d = np.full(48, night)
    d[rise_slot:] = daytime
    d[peak_slot] = 2.0
    return d


def test_routine_detection():
    d = _day(rise_slot=14, peak_slot=38)
    assert abs(M.morning_rise(d) - 14) <= 1
    assert abs(M.evening_peak(d) - 38) <= 1
    w1, w2 = np.tile(_day(14, 38), 2), np.tile(_day(18, 40), 2)
    rise, peak = M.routine_errors(w1, w2)
    assert rise == pytest.approx(2.0, abs=0.5) and peak == pytest.approx(1.0)


def test_away_label():
    assert M.away_label(_day(14, 38, daytime=0.11)) is True
    assert M.away_label(_day(14, 38, daytime=0.5)) is False
    assert M.balanced_accuracy([True, False], [True, False]) == 1


def test_paired_test():
    att, blind = np.full(20, 5.0) + np.arange(20) * 0.01, np.full(20, 50.0)
    assert M.paired_test(att, blind)["holds"] is False           # attack far better: leak
    noise = np.random.default_rng(0).normal(0, 1, 20)
    assert M.paired_test(blind + noise, blind)["holds"] is True   # no real difference


# ------------------------------------------------------------------ blind guess
def test_blind_guess_shape_and_range():
    dates = data.valid_dates(data.get_neighbourhood(N, K), "train")[:4]
    for mode in ("weekday", "same_date"):
        g = blind_guess(dates, mode)
        assert g.shape == (4, 96) and np.isfinite(g).all()
        assert 0 <= g.min() and g.max() <= 1


# ------------------------------------------------------------------ scoring pipeline
def _recording(level, eps):
    hs = data.get_neighbourhood(N, K)
    dates = data.valid_dates(hs, "train")[:4]
    P = common.num_params(common.Model())
    return {
        "level": level, "eps": eps, "n": N, "nb": K, "round": 1,
        "server_view": torch.zeros(P) if level == 3 else torch.zeros(N, P),
        "view_is_sum": False, "global_model": common.Model().state_dict(),
        "true_windows": data.get_windows(hs, dates, "train"), "house_ids": hs, "dates": dates,
        "lr": 0.1, "C": None if level == 1 else 1.0, "sigma": None if level == 1 else 0.0,
        "round_time_s": 0.1, "bytes_per_house": 100.0,
        "env": {"python": "3", "flwr": "1.45.0", "torch": "2", "cpu_model": "x", "cpu_cores": 1,
                "gpu_visible": False},
    }


def test_perfect_attack_leaks_and_blind_attack_does_not(tmp_path):
    rows = []
    rec1 = _recording(1, None)
    true = rec1["true_windows"]
    rows += score_one(rec1, true[:, [3, 1, 0, 2]], "inverter")[0]          # perfect, shuffled
    rec3 = _recording(3, 1.0)
    g = torch.Generator().manual_seed(0)
    blind = torch.from_numpy(blind_guess(rec3["dates"])).repeat(N, 1)     # [4n, 96]
    blind = (blind + 0.02 * torch.randn(blind.shape, generator=g)).clamp(0, 1)
    rows += score_one(rec3, blind, "inverter")[0]                         # a noisy blind guess
    s = summarise(pd.DataFrame(rows), pd.DataFrame(score_one(rec1, true, "inverter")[1]
                                                     + score_one(rec3, blind, "inverter")[1]))
    l1 = s[s["level"] == 1].iloc[0]
    l3 = s[s["level"] == 3].iloc[0]
    assert l1["smape_attack"] == pytest.approx(0, abs=1e-4) and not l1["protection_holds"]
    # One-to-one matching against the truth lets 4n noisy guesses pick their best pairing, so a noisy
    # blind guess scores a little better than blind (a known, small scoring advantage; see README).
    assert l3["smape_improvement"] < 2 and l3["rise_err_holds"] and l3["away_acc_holds"]


def test_files_round_trip(tmp_path):
    rec = _recording(2, 8.0)
    torch.save(rec, tmp_path / "L2_eps8_n5_nb0_r1.pt")
    torch.save({"rebuilt": rec["true_windows"]}, tmp_path / "L2_eps8_n5_nb0_r1__inverter.pt")
    loaded = load_recording(tmp_path / "L2_eps8_n5_nb0_r1.pt")
    rebuilt = load_attack_output(tmp_path / "L2_eps8_n5_nb0_r1__inverter.pt")
    house_rows, day_rows = score_one(loaded, rebuilt, "inverter")
    assert len(house_rows) == N and len(day_rows) == N * 4 * 2
