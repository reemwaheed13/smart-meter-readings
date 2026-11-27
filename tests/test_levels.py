"""Stage 4 tests: level 2 (Flower LocalDpMod), level 3 (SecAgg+ + noise_share_mod), the
accountant, running cost and resumable runs.

Simulation tests run real Flower simulations on the data.py in the repo (currently the FAKE stub).
"""
from __future__ import annotations

import csv
import json
import math
import shutil

import numpy as np
import pytest
import torch
import yaml

import common
from attacks.recording import KEYS, load_recording
from train import accountant
from train.app import ClippingRangeExceeded
from train.costs import running_cost
from train.noise_share_mod import NoiseShareMod, noise_seed
from train.recorder import recording_path, rounds_to_record
from train import run as train_run
from train.run import is_complete, run, run_resumable
from train.settings import (MAX_EXPECTED_CUTS, ClippingRangeTooSmall, check_clipping_range,
                            expected_clipping_cuts, min_clipping_range, resolve)

N, K, LR, C, CR = 5, 0, 0.1, 1.0, 8.0
DELTA = common.load_config()["dp"]["delta"]
P = common.num_params(common.Model())
STEP = 2 * CR / 2 ** 22  # SecAgg+ quantisation step (Flower default quantization_range 2**22)


@pytest.fixture(scope="module")
def null_config(tmp_path_factory):
    """config.yaml with every calibration value nulled, so the overrides are allowed."""
    cfg = common.load_config()
    cfg["calibration"].update(lr=None, C=None, clipping_range=None,
                              T={n: None for n in cfg["experiment"]["n"]})
    cfg["dp"]["sigma"] = {1: None, 8: None}
    cfg["secaggplus"]["clipping_range"] = None
    p = tmp_path_factory.mktemp("cfg") / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _sim(tmp_path_factory, null_config, tag, level, eps, T=3, n=N, cr=CR, **kw):
    out = tmp_path_factory.mktemp(tag)
    s = resolve(level, eps, n, K, lr=LR, T=T, C=C, clipping_range=cr if level == 3 else None,
                out_dir=out, config_path=null_config, **kw)
    run(s)
    return s


def _recs(s):
    return {r: load_recording(recording_path(s, r)) for r in rounds_to_record(s.T)}


def _clean(rec, i):
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    return common.local_update(m, rec["true_windows"][i], rec["lr"])


@pytest.fixture(scope="module")
def l2(tmp_path_factory, null_config):
    return _sim(tmp_path_factory, null_config, "l2", 2, 1)


@pytest.fixture(scope="module")
def l2_rerun(tmp_path_factory, null_config):
    return _sim(tmp_path_factory, null_config, "l2b", 2, 1)


@pytest.fixture(scope="module")
def l3_none(tmp_path_factory, null_config):
    return _sim(tmp_path_factory, null_config, "l3none", 3, "none")


@pytest.fixture(scope="module")
def l3_zero_noise(tmp_path_factory, null_config):
    # Updates forced to zero (test-only hook), so the aggregate is the noise alone. A wide
    # clipping_range keeps the n=5 noise shares (std sigma/sqrt(5) ~ 2.2) inside it.
    return _sim(tmp_path_factory, null_config, "l3zero", 3, 1, T=2, cr=64.0, test_zero_update=True)


# --------------------------------------------------------------------------- settings

def test_settings_levels_2_3(null_config):
    s = resolve(2, 8, N, K, lr=LR, T=3, C=C, config_path=null_config)
    assert s.sigma == pytest.approx(common.sigma_for(8, C, DELTA))
    s = resolve(3, "none", N, K, lr=LR, T=3, C=C, clipping_range=CR, config_path=null_config)
    assert s.sigma == 0.0 and s.num_shares == 5 and s.reconstruction_threshold == 3 and s.clipping_range == CR
    with pytest.raises(ValueError, match="C"):
        resolve(2, 1, N, K, lr=LR, T=3, config_path=null_config)
    with pytest.raises(ValueError, match="clipping-range"):
        resolve(3, 1, N, K, lr=LR, T=3, C=C, config_path=null_config)
    with pytest.raises(ValueError):
        resolve(2, "none", N, K, lr=LR, T=3, C=C, config_path=null_config)


def test_settings_refuse_C_override_when_config_set(tmp_path, null_config):
    cfg = yaml.safe_load(null_config.read_text())
    cfg["calibration"]["C"] = 2.0
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="--C"):
        resolve(2, 1, N, K, lr=LR, T=3, C=1.0, config_path=p)
    assert resolve(2, 1, N, K, lr=LR, T=3, config_path=p).C == 2.0


def test_out_dir(tmp_path, null_config):
    s = resolve(1, "none", N, K, lr=LR, T=3, out_dir=tmp_path, config_path=null_config)
    assert s.recordings_dir == tmp_path / "recordings" and s.logs_dir == tmp_path / "logs"


# --------------------------------------------------------------------------- level 2

def _l2_noise(s, rec, i):
    """LocalDpMod's noise for house i in gradient space, reproduced from the seeded numpy RNG:
    the server computes (w - returned)/lr, so the recorded row is clip(g, C) - noise/lr."""
    np.random.seed(noise_seed(s.run_seed, rec["round"], i))
    std_w = s.lr * s.C * math.sqrt(2 * math.log(1.25 / s.delta)) / s.eps   # LocalDpMod's std, weight space
    parts = [np.random.normal(0, std_w, t.shape).astype(np.float32).reshape(-1)
             for t in rec["global_model"].values()]
    return -torch.from_numpy(np.concatenate(parts)).double() / s.lr


def test_level2_server_view_is_clean_gradient_plus_noise(l2):
    sigma = common.sigma_for(1, C, DELTA)
    for rec in _recs(l2).values():
        assert rec["server_view"].shape == (N, P) and rec["view_is_sum"] is False
        assert rec["sigma"] == pytest.approx(sigma) and rec["C"] == C
        for i in range(N):
            clean = common.clip(_clean(rec, i), C).double()
            noise = _l2_noise(l2, rec, i)
            clipped_part = rec["server_view"][i].double() - noise
            # float32 weights: (w - returned)/lr is exact to ~ulp(w)/lr
            assert torch.allclose(clipped_part, clean, atol=2e-4), (rec["round"], i)
            assert torch.linalg.vector_norm(clipped_part) <= C * (1 + 1e-3)
            assert float(noise.std()) == pytest.approx(sigma, rel=0.05)
            assert float((rec["server_view"][i].double() - clean).std()) == pytest.approx(sigma, rel=0.05)


def test_level2_fedsgd_step(l2):
    recs = _recs(l2)
    for r in (1, 2):
        start = common.flatten(recs[r]["global_model"])
        want = start - LR * recs[r]["server_view"].mean(dim=0)
        assert torch.allclose(common.flatten(recs[r + 1]["global_model"]), want, atol=1e-5)


def test_level2_rerun_identical(l2, l2_rerun):
    a, b = _recs(l2), _recs(l2_rerun)
    for r in a:
        for k in KEYS - {"round_time_s"}:
            x, y = a[r][k], b[r][k]
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), (r, k)
            elif isinstance(x, dict) and k == "global_model":
                assert all(torch.equal(x[j], y[j]) for j in x), (r, k)
            else:
                assert x == y, (r, k)


def test_recording_env(l2):
    env = _recs(l2)[1]["env"]
    assert env["flwr"] == "1.45.0" and env["cpu_cores"] >= 1 and isinstance(env["gpu_visible"], bool)


# --------------------------------------------------------------------------- noise_share_mod

def test_noise_share_mod_shares_sum_to_sigma():
    n, sigma, run_seed = 20, 3.0, 1234
    mod = NoiseShareMod(C=1.0, sigma=sigma, n=n, run_seed=run_seed, clipping_range=None)
    zeros = [np.zeros((64, 48), np.float32), np.zeros(64, np.float32)]
    shares = []
    for pid in range(n):
        out, cuts = mod.protect_arrays(zeros, server_round=1, partition_id=pid)
        assert [a.shape for a in out] == [a.shape for a in zeros] and out[0].dtype == np.float32
        flat = np.concatenate([a.reshape(-1) for a in out])
        assert flat.std() == pytest.approx(sigma / math.sqrt(n), rel=0.06)
        shares.append(flat)
    total = np.sum(shares, axis=0)
    assert total.std() == pytest.approx(sigma, rel=0.06)
    # reproducible, and different clients get different noise
    again, _ = mod.protect_arrays(zeros, 1, 0)
    assert np.array_equal(np.concatenate([a.reshape(-1) for a in again]), shares[0])
    assert not np.array_equal(shares[0], shares[1])


def test_noise_share_mod_clips_and_counts():
    big = [np.full(100, 10.0, np.float32)]                  # norm 100
    out, cuts = NoiseShareMod(C=1.0, sigma=None, n=4, run_seed=0, clipping_range=0.05).protect_arrays(big, 1, 0)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, rel=1e-5)
    assert cuts == 100                                      # every value 0.1 > 0.05
    _, cuts = NoiseShareMod(C=1.0, sigma=None, n=4, run_seed=0, clipping_range=0.2).protect_arrays(big, 1, 0)
    assert cuts == 0


# --------------------------------------------------------------------------- level 3

def test_level3_view_times_n_is_sum_of_clipped_updates(l3_none, capsys):
    """eps = none: server_view * n equals the sum of the clipped true updates up to SecAgg+
    quantisation (each house's stochastic rounding error is < one step). Confirms Flower returns
    the MEAN (view_is_sum False)."""
    worst = 0.0
    for rec in _recs(l3_none).values():
        assert rec["server_view"].shape == (P,) and rec["view_is_sum"] is False and rec["sigma"] == 0.0
        want = sum(common.clip(_clean(rec, i), C) for i in range(N))
        err = float((rec["server_view"] * N - want).abs().max())
        worst = max(worst, err)
        assert err < N * STEP, rec["round"]
        assert float((rec["server_view"] - want).abs().max()) > 100 * N * STEP  # it is not the sum
    with capsys.disabled():
        print(f"\n[level 3, eps none, n={N}] max |server_view*n - sum of clipped updates| = {worst:.3g} "
              f"(quantisation step {STEP:.3g}, bound n*step = {N * STEP:.3g})")


def test_level3_fedsgd_step_and_csv(l3_none):
    recs = _recs(l3_none)
    for r in (1, 2):
        want = common.flatten(recs[r]["global_model"]) - LR * recs[r]["server_view"]
        assert torch.allclose(common.flatten(recs[r + 1]["global_model"]), want, atol=1e-6)
    with open(l3_none.logs_dir / f"{l3_none.run_name}.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["clipping_range_cuts"] for r in rows] == ["0", "0", "0"]
    # SecAgg+: four stages of messages, more than one plain float32 update
    assert all(float(r["bytes_per_house"]) > 4 * P for r in rows)


def test_level3_noise_std_is_sigma(l3_zero_noise):
    sigma = common.sigma_for(1, C, DELTA)
    for rec in _recs(l3_zero_noise).values():
        total = rec["server_view"].double() * N
        assert float(total.std()) == pytest.approx(sigma, rel=0.05)
        assert abs(float(total.mean())) < 5 * sigma / math.sqrt(P)


def test_level3_clipping_range_violation_fails_the_run(tmp_path_factory, null_config):
    with pytest.raises(ClippingRangeExceeded, match="clipping_range"):
        _sim(tmp_path_factory, null_config, "l3cut", 3, "none", T=2, cr=1e-4)


# --------------------------------------------------------------------------- accountant / cost / resume

def test_accountant_hand_checked():
    # eps 1, delta 1e-5, C 1, T 10 (by hand):
    # ln(1.25/1e-5) = ln 125000 = 11.736069;  sigma = sqrt(2 * 11.736069) = sqrt(23.472138) = 4.844805
    # rho per round = C^2/(2 sigma^2) = 1/46.944276 = 0.0213019;  total rho = 10 * that = 0.213019
    # eps_total = 0.213019 + 2 sqrt(0.213019 * ln 1e5) = 0.213019 + 2 sqrt(0.213019 * 11.512925)
    #           = 0.213019 + 2 * 1.566034 = 3.345086
    sigma = common.sigma_for(1, 1.0, 1e-5)
    assert sigma == pytest.approx(4.844805, abs=1e-6)
    assert accountant.rho_per_round(1.0, sigma) == pytest.approx(0.0213019, abs=1e-7)
    assert accountant.total_eps(1.0, sigma, 10, 1e-5) == pytest.approx(3.345086, abs=1e-5)
    assert accountant.total_eps(1.0, 0.0, 10, 1e-5) == math.inf
    # rho does not depend on C once sigma = sigma_for(eps, C, delta)
    assert accountant.total_eps(3.0, common.sigma_for(1, 3.0, 1e-5), 10, 1e-5) == pytest.approx(3.345086, abs=1e-5)


def test_accountant_table(tmp_path):
    rows = accountant.table(1.0, 1e-5, [10], {2: [1, 8], 3: [1, 8, None]})
    assert [(r["level"], r["eps_per_round"]) for r in rows] == [(2, "1"), (2, "8"), (3, "1"), (3, "8"), (3, "none")]
    assert rows[0]["eps_total"] == pytest.approx(3.345086, abs=1e-5) and rows[4]["eps_total"] == math.inf
    p = accountant.write_table(tmp_path / "t.csv", rows)
    with open(p) as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == ["level", "eps_per_round", "sigma", "T", "eps_total"] and got[4]["eps_total"] == "inf"


def test_running_cost_skips_warmup(l3_none):
    rows = list(csv.DictReader(open(l3_none.logs_dir / f"{l3_none.run_name}.csv")))
    cost = running_cost(l3_none.logs_dir / f"{l3_none.run_name}.csv")
    assert cost["rounds"] == [2, 3]
    assert cost["median_round_time_s"] == pytest.approx(np.median([float(r["round_time_s"]) for r in rows[1:]]))


def test_resume(l3_none, tmp_path):
    assert is_complete(l3_none)
    assert run_resumable(l3_none) is False                 # complete: skipped, no simulation
    # A truncated recording makes the run partial.
    import shutil
    s = l3_none
    copy = type(s)(**{**s.__dict__, "recordings_dir": tmp_path / "recordings", "logs_dir": tmp_path / "logs"})
    shutil.copytree(s.recordings_dir, copy.recordings_dir)
    shutil.copytree(s.logs_dir, copy.logs_dir)
    assert is_complete(copy)
    recording_path(copy, 2).write_bytes(b"not a recording")
    assert not is_complete(copy)
    recording_path(copy, 2).unlink()
    assert not is_complete(copy)


# --------------------------------------------------------------------------- pre-run clipping_range check

def test_expected_clipping_cuts_formula():
    # n*T*P*2*Phi(-cr/(sigma/sqrt n)); 2*Phi(-x) = erfc(x/sqrt 2). Hand check at x = 2: 2*Phi(-2) = 0.0455003.
    sigma, n = 2.0 * math.sqrt(5), 5            # share std = 2.0
    assert expected_clipping_cuts(n, 10, P, 4.0, sigma) == pytest.approx(5 * 10 * P * 0.0455003, rel=1e-5)
    assert expected_clipping_cuts(n, 10, P, 4.0, 0.0) == 0.0 and expected_clipping_cuts(n, 10, P, 4.0, None) == 0.0
    # The Colab failure: C = 1, eps = 1, n = 5, T = 10, clipping_range 8.0 is 3.69 share stds:
    # 2*Phi(-3.6923) = 2.222e-4, times n*T*P = 520800 -> about 116 cuts expected.
    s1 = common.sigma_for(1, 1.0, DELTA)
    assert expected_clipping_cuts(5, 10, P, 8.0, s1) == pytest.approx(115.73, rel=1e-3)
    assert expected_clipping_cuts(5, 10, P, 16.0, s1) < 1e-6           # 7.38 stds: about 8e-8


def test_min_clipping_range_is_smallest_passing():
    s1 = common.sigma_for(1, 1.0, DELTA)
    for n, T in [(5, 10), (5, 200), (20, 50)]:
        cr = min_clipping_range(n, T, P, s1)
        assert expected_clipping_cuts(n, T, P, cr, s1) < MAX_EXPECTED_CUTS
        assert expected_clipping_cuts(n, T, P, cr - 0.01, s1) >= MAX_EXPECTED_CUTS
        assert cr == round(cr, 2)


def test_check_refuses_small_clipping_range(null_config):
    s = resolve(3, 1, 5, K, lr=LR, T=10, C=1.0, clipping_range=8.0, config_path=null_config)
    want_min = min_clipping_range(5, 10, P, s.sigma)
    with pytest.raises(ClippingRangeTooSmall) as e:
        check_clipping_range(s)
    msg = str(e.value)
    assert f"{expected_clipping_cuts(5, 10, P, 8.0, s.sigma):.4g}" in msg and f"{want_min:g}" in msg
    check_clipping_range(resolve(3, 1, 5, K, lr=LR, T=10, C=1.0, clipping_range=16.0, config_path=null_config))
    # eps = none and levels 1-2 skip the check, however small clipping_range is
    check_clipping_range(resolve(3, "none", 5, K, lr=LR, T=10, C=1.0, clipping_range=1e-4, config_path=null_config))
    check_clipping_range(resolve(2, 1, 5, K, lr=LR, T=10, C=1.0, config_path=null_config))
    check_clipping_range(resolve(1, "none", 5, K, lr=LR, T=10, config_path=null_config))


def test_refused_run_starts_nothing_and_deletes_nothing(null_config, tmp_path, capsys):
    s = resolve(3, 1, 5, K, lr=LR, T=10, C=1.0, clipping_range=8.0, out_dir=tmp_path, config_path=null_config)
    s.logs_dir.mkdir(parents=True)
    partial = s.logs_dir / f"{s.run_name}.csv"
    partial.write_text("round\n")
    with pytest.raises(ClippingRangeTooSmall):
        run_resumable(s)
    with pytest.raises(ClippingRangeTooSmall):
        run(s)
    assert partial.exists() and not s.recordings_dir.exists()
    rc = train_run.main(["--level", "3", "--eps", "1", "--n", "5", "--k", str(K), "--out", str(tmp_path),
                         "--lr", str(LR), "--T", "10", "--C", "1.0", "--clipping-range", "8.0"])
    assert rc == 2 and "smallest clipping_range" in capsys.readouterr().err


def test_clipping_range_logged_and_changed_value_redoes_run(l3_none, l2, tmp_path):
    t = json.load(open(l3_none.logs_dir / f"{l3_none.run_name}_test.json"))
    assert t["clipping_range"] == CR
    assert json.load(open(l2.logs_dir / f"{l2.run_name}_test.json"))["clipping_range"] is None
    copy = type(l3_none)(**{**l3_none.__dict__, "recordings_dir": tmp_path / "recordings", "logs_dir": tmp_path / "logs"})
    shutil.copytree(l3_none.recordings_dir, copy.recordings_dir)
    shutil.copytree(l3_none.logs_dir, copy.logs_dir)
    assert is_complete(copy)
    other = type(copy)(**{**copy.__dict__, "clipping_range": 2 * CR})
    assert not is_complete(other)                          # logged 8.0, current 16.0: redo
    p = copy.logs_dir / f"{copy.run_name}_test.json"
    del t["clipping_range"]
    p.write_text(json.dumps(t))                            # an older log without the key
    assert not is_complete(copy)
