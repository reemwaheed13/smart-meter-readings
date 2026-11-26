"""Stage 1 tests: Flower level-1 FedSGD, the recorder and the recording loader.

The simulation tests launch real Flower simulations (Ray backend) on the data.py in the repo
(currently the FAKE stub). Each simulation takes up to about a minute, mostly Ray start-up.
"""
from __future__ import annotations

import csv

import pytest
import torch
import yaml

import common
import data
from attacks.recording import KEYS, load_recording
from train.recorder import recording_path, rounds_to_record
from train.run import run
from train.settings import resolve

N, K, LR = 5, 0, 0.1


@pytest.fixture(scope="module")
def null_config(tmp_path_factory):
    """config.yaml with the calibration values nulled, so the --lr/--T overrides are allowed
    even after calibration fills the real config in."""
    cfg = common.load_config()
    cfg["calibration"]["lr"] = None
    cfg["calibration"]["T"] = {n: None for n in cfg["experiment"]["n"]}
    p = tmp_path_factory.mktemp("cfg") / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _simulate(tmp_path_factory, null_config, T, tag):
    out = tmp_path_factory.mktemp(tag)
    s = resolve(1, "none", N, K, lr=LR, T=T, recordings_dir=out / "recordings", logs_dir=out / "logs",
                config_path=null_config)
    run(s)
    return s


@pytest.fixture(scope="module")
def sim3(tmp_path_factory, null_config):
    return _simulate(tmp_path_factory, null_config, 3, "sim3")


@pytest.fixture(scope="module")
def sim3_rerun(tmp_path_factory, null_config):
    return _simulate(tmp_path_factory, null_config, 3, "sim3_rerun")


@pytest.fixture(scope="module")
def sim2(tmp_path_factory, null_config):
    return _simulate(tmp_path_factory, null_config, 2, "sim2")


def _recs(s):
    return {r: load_recording(recording_path(s, r)) for r in rounds_to_record(s.T)}


# --------------------------------------------------------------------------- unit

@pytest.mark.parametrize("T,want", [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, [1, 2, 4]),
                                    (10, [1, 5, 10]), (11, [1, 6, 11])])
def test_rounds_to_record(T, want):
    assert rounds_to_record(T) == want


def test_settings_refuse_null_without_override(null_config):
    with pytest.raises(ValueError, match="lr"):
        resolve(1, "none", N, K, T=3, config_path=null_config)
    with pytest.raises(ValueError, match="T"):
        resolve(1, "none", N, K, lr=LR, config_path=null_config)


def test_settings_refuse_override_when_config_set(tmp_path):
    cfg = common.load_config()
    cfg["calibration"]["lr"] = 0.05
    cfg["calibration"]["T"] = {5: 7, 10: None, 20: None}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="--lr"):
        resolve(1, "none", 5, 0, lr=0.1, config_path=p)
    with pytest.raises(ValueError, match="--T"):
        resolve(1, "none", 5, 0, T=3, config_path=p)
    s = resolve(1, "none", 5, 0, config_path=p)
    assert (s.lr, s.T) == (0.05, 7)


def test_settings_level1_needs_eps_none(null_config):
    with pytest.raises(ValueError):
        resolve(1, "8", N, K, lr=LR, T=3, config_path=null_config)


def test_load_recording_rejects_bad_files(tmp_path):
    P = common.num_params(common.Model())
    good = {
        "level": 1, "eps": None, "n": 2, "nb": 0, "round": 1,
        "server_view": torch.zeros(2, P), "view_is_sum": False,
        "global_model": common.Model().state_dict(), "true_windows": torch.zeros(2, 4, 96),
        "house_ids": ["a", "b"], "dates": ["2013-01-02"] * 4, "lr": 0.1, "C": None, "sigma": None,
        "round_time_s": 0.1, "bytes_per_house": 100.0,
        "env": {"python": "3.12.3", "flwr": "1.45.0", "torch": "2.15.0", "cpu_model": "x", "cpu_cores": 2,
                "gpu_visible": False},
    }
    torch.save(good, tmp_path / "ok.pt")
    load_recording(tmp_path / "ok.pt")
    for bad in ({"server_view": torch.zeros(3, P)}, {"true_windows": torch.zeros(2, 3, 96)},
                {"view_is_sum": True}, {"dates": ["2013-01-02"] * 3}, {"house_ids": ["a"]},
                {"env": {}}, {"level": 3, "C": 1.0, "sigma": 0.0}):
        torch.save({**good, **bad}, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            load_recording(tmp_path / "bad.pt")
    torch.save({k: v for k, v in good.items() if k != "sigma"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="missing"):
        load_recording(tmp_path / "bad.pt")


# --------------------------------------------------------------------------- simulation

def test_three_rounds_write_three_valid_recordings(sim3):
    files = sorted(sim3.recordings_dir.glob("*.pt"))
    assert [f.name for f in files] == [f"L1_eps0_n5_nb0_r{r}.pt" for r in (1, 2, 3)]
    for r, f in zip((1, 2, 3), files):
        rec = load_recording(f)
        assert set(rec) == KEYS
        assert (rec["level"], rec["eps"], rec["n"], rec["nb"], rec["round"]) == (1, None, N, K, r)
        assert rec["view_is_sum"] is False
        assert rec["house_ids"] == data.get_neighbourhood(N, K)
        assert rec["lr"] == LR and rec["C"] is None and rec["sigma"] is None
        assert rec["bytes_per_house"] > 4 * common.num_params(common.Model())  # float32 payload + framing
        assert set(rec["dates"]) <= set(data.valid_dates(rec["house_ids"], "train"))


def test_csv_log(sim3):
    with open(sim3.logs_dir / "L1_eps0_n5_nb0.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["round"]) for r in rows] == [1, 2, 3]
    assert list(rows[0]) == ["round", "val_mse", "round_time_s", "bytes_per_house", "is_warmup",
                             "clipping_range_cuts"]
    assert [r["is_warmup"] for r in rows] == ["True", "False", "False"]
    assert all(r["clipping_range_cuts"] == "" for r in rows)  # level 1: no SecAgg+
    assert all(float(r["val_mse"]) > 0 and float(r["round_time_s"]) > 0 for r in rows)
    assert (sim3.logs_dir / "L1_eps0_n5_nb0_test.json").exists()


def test_server_view_is_exactly_local_update(sim3):
    for rec in _recs(sim3).values():
        m = common.Model()
        m.load_state_dict(rec["global_model"])
        assert torch.equal(rec["true_windows"],
                           data.get_windows(rec["house_ids"], rec["dates"], "train"))
        for i in range(rec["n"]):
            want = common.local_update(m, rec["true_windows"][i], rec["lr"])
            assert torch.allclose(rec["server_view"][i], want, rtol=1e-6, atol=1e-8), f"house {i}"


def test_fedsgd_step(sim3):
    recs = _recs(sim3)
    for r in (1, 2):
        start = common.flatten(recs[r]["global_model"])
        want = start - LR * recs[r]["server_view"].mean(dim=0)
        got = common.flatten(recs[r + 1]["global_model"])
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-8)


def test_rerun_is_deterministic(sim3, sim3_rerun):
    a, b = _recs(sim3), _recs(sim3_rerun)
    assert a.keys() == b.keys()
    for r in a:
        for k in KEYS - {"round_time_s"}:
            x, y = a[r][k], b[r][k]
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), (r, k)
            elif k == "global_model":
                assert all(torch.equal(x[j], y[j]) for j in x), (r, k)
            else:
                assert x == y, (r, k)


def test_two_rounds_no_duplicate_recordings(sim2):
    files = sorted(f.name for f in sim2.recordings_dir.glob("*.pt"))
    assert files == ["L1_eps0_n5_nb0_r1.pt", "L1_eps0_n5_nb0_r2.pt"]
    for f in files:
        load_recording(sim2.recordings_dir / f)


def test_same_dates_across_T(sim2, sim3):
    """Dates depend only on run_seed and the round, so a T=2 and a T=3 run agree on rounds 1-2."""
    a, b = _recs(sim2), _recs(sim3)
    for r in (1, 2):
        assert a[r]["dates"] == b[r]["dates"]
        assert torch.equal(a[r]["server_view"], b[r]["server_view"])
