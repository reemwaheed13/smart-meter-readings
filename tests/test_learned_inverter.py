"""Learned inverter (attacks/learned_inverter.py) and the level-matched, cached aux set
(attacks/aux_gradients.py). Fake data only; tiny aux sets and epochs, no real attack is run."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import common
import data
from attacks import aux_gradients, learned_inverter
from attacks.recording import load_recording
from eval import score_all
from eval.baseline import blind_guess
from eval.score_all import load_attack_output
from train.env import machine_info
from train.noise_share_mod import NoiseShareMod

LR = 0.1
W = common.load_config()["experiment"]["windows_per_round"]
FAST = ["--aux-samples", "16", "--epochs", "1"]


def make_recording(level: int, n: int, tmp_path, *, sigma: float = 0.0, rnd: int = 1, nb: int = 0):
    """A recording like train/ writes, from the fake data, with the level's noise."""
    houses = data.get_neighbourhood(n, nb)
    dates = data.valid_dates(houses, "train")[10:10 + W]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        m = common.Model()
    true = data.get_windows(houses, dates, "train")
    C = None if level == 1 else 1.0
    g = torch.Generator().manual_seed(1)
    views = torch.stack([common.protect(common.local_update(m, true[h], LR), level, C, sigma, n, generator=g)
                         for h in range(n)])
    rec = {"level": level, "eps": None if level == 1 or not sigma else 1.0, "n": n, "nb": nb, "round": rnd,
           "server_view": views.mean(0) if level == 3 else views, "view_is_sum": False,
           "global_model": m.state_dict(), "true_windows": true, "house_ids": houses, "dates": dates, "lr": LR,
           "C": C, "sigma": None if level == 1 else float(sigma), "round_time_s": 0.0, "bytes_per_house": 1.0,
           "env": machine_info()}
    p = tmp_path / f"L{level}_eps{'0' if rec['eps'] is None else '1'}_n{n}_nb{nb}_r{rnd}.pt"
    torch.save(rec, p)
    return load_recording(p), p


def run_cli(p, out, *extra):
    assert learned_inverter.main(["--recording", str(p), "--out", str(out), "--aux-cache", str(out / "aux"),
                                  *extra]) == 0
    pt, js = learned_inverter.output_paths(p, out)
    return load_attack_output(pt), json.loads(js.read_text())


# --------------------------------------------------------------------------- the untrained inverter is the blind guess

@pytest.mark.parametrize("level,n", [(1, 2), (2, 2), (3, 3)])
def test_untrained_equals_same_date_blind_guess(tmp_path, level, n):
    rec, p = make_recording(level, n, tmp_path, sigma=0.5)
    rebuilt, diag = run_cli(p, tmp_path / "out", "--aux-samples", "8", "--epochs", "0")
    guess = torch.from_numpy(blind_guess(rec["dates"], "same_date"))              # [4, 96]
    want = guess.repeat(n, 1) if level == 3 else guess.unsqueeze(0).expand(n, -1, -1)
    assert rebuilt.shape == want.shape and torch.equal(rebuilt, want)
    assert diag["training"]["best_epoch"] == 0 and diag["training"]["history"] == []


# --------------------------------------------------------------------------- aux noise = the server view's noise

def test_aux_noise_std_matches_level2_and_level3_server_views(monkeypatch):
    """With every update zeroed, the server's view and an aux sample are pure noise: compare their stds."""
    P = common.num_params(common.Model())
    monkeypatch.setattr(common, "local_update", lambda model, windows, lr, **kw: torch.zeros(P))
    C, sigma, n, S = 1.0, 2.0, 4, 6
    state = common.Model().state_dict()

    # Level 2: what the server sees per house is common.protect(g, 2, ...) (LocalDpMod in gradient space).
    g = torch.Generator().manual_seed(0)
    server2 = torch.stack([common.protect(torch.zeros(P), 2, C, sigma, n, generator=g) for _ in range(S)])
    aux2, _ = aux_gradients.aux_samples(state, 2, C, n, LR, S, seed=3, sigma=sigma)
    # Level 3: the server's view is the mean of the n houses' noise_share_mod outputs.
    mod = NoiseShareMod(C=C, sigma=sigma, n=n, run_seed=1234)
    server3 = torch.stack([torch.from_numpy(np.mean([mod.protect_arrays([np.zeros(P, np.float32)], r, pid)[0][0]
                                                     for pid in range(n)], axis=0)) for r in range(1, S + 1)])
    aux3, _ = aux_gradients.aux_samples(state, 3, C, n, LR, S, seed=3, sigma=sigma)

    assert aux_gradients.noise_std(2, sigma, n) == sigma and aux_gradients.noise_std(3, sigma, n) == sigma / n
    assert aux_gradients.noise_std(1, sigma, n) == 0 and aux_gradients.noise_std(3, 0.0, n) == 0
    for server, aux, want in ((server2, aux2, sigma), (server3, aux3, sigma / n)):
        assert float(server.std()) == pytest.approx(want, rel=0.03)
        assert float(aux.std()) == pytest.approx(want, rel=0.03)
        assert float(aux.mean()) == pytest.approx(0.0, abs=0.05 * want)
    # Level 1 has no noise.
    aux1, _ = aux_gradients.aux_samples(state, 1, None, n, LR, 2, seed=3, sigma=sigma)
    assert torch.count_nonzero(aux1) == 0


def test_aux_noise_has_its_own_stream():
    rec_state = common.Model().state_dict()
    g0, w0 = aux_gradients.aux_samples(rec_state, 3, 1.0, 2, LR, 4, seed=9, sigma=0.0)
    g1, w1 = aux_gradients.aux_samples(rec_state, 3, 1.0, 2, LR, 4, seed=9, sigma=1.0)
    assert torch.equal(w0, w1)                       # changing sigma changes the noise only, not the windows
    assert not torch.equal(g0, g1)
    assert float((g1 - g0).std()) == pytest.approx(0.5, rel=0.05)    # sigma / n


# --------------------------------------------------------------------------- outputs, seeds, scoring

@pytest.mark.parametrize("level,n", [(1, 2), (2, 2), (3, 2)])
def test_output_shapes_range_and_weights_only(tmp_path, level, n):
    rec, p = make_recording(level, n, tmp_path, sigma=0.3)
    rebuilt, diag = run_cli(p, tmp_path / "out", *FAST)
    pt, _ = learned_inverter.output_paths(p, tmp_path / "out")
    assert set(torch.load(pt, weights_only=True)) == {"rebuilt"}
    assert tuple(rebuilt.shape) == ((n * W, 96) if level == 3 else (n, W, 96))
    assert rebuilt.dtype == torch.float32 and 0 <= float(rebuilt.min()) and float(rebuilt.max()) <= 1
    assert diag["rebuilt_shape"] == list(rebuilt.shape) and len(diag["training"]["history"]) == 1
    assert diag["seeds"]["run_seed"] == common.run_seed(n, 0)
    assert diag["aux"]["noise_std"] == aux_gradients.noise_std(level, rec["sigma"], n)
    assert set(diag) == {"recording", "attack", "level", "eps", "n", "nb", "round", "rebuilt_shape", "seeds",
                         "aux", "training", "fingerprint"}                          # diagnostics only, no scores


def test_same_seed_same_output_and_resume(tmp_path, capsys):
    _, p = make_recording(2, 2, tmp_path, sigma=0.3)
    r1, d1 = run_cli(p, tmp_path / "a", *FAST)
    r2, d2 = run_cli(p, tmp_path / "b", *FAST)
    assert torch.equal(r1, r2) and d1["training"] == d2["training"]
    capsys.readouterr()
    run_cli(p, tmp_path / "a", *FAST)
    assert "skipping" in capsys.readouterr().out
    _, d3 = run_cli(p, tmp_path / "a", "--aux-samples", "16", "--epochs", "2")    # settings changed: redone
    assert len(d3["training"]["history"]) == 2


def test_cli_then_score_all_same_date(tmp_path, monkeypatch):
    recs, out = tmp_path / "recordings", tmp_path / "attack_outputs"
    recs.mkdir()
    for level in (1, 3):
        _, p = make_recording(level, 2, recs, sigma=0.0)
        run_cli(p, out, *FAST)
    res = tmp_path / "results"
    monkeypatch.setattr("sys.argv", ["score_all", "--recordings", str(recs), "--attacks", str(out),
                                     "--out", str(res), "--blind", "same_date"])
    score_all.main()
    per_house = pd.read_csv(res / "per_house.csv")
    assert set(per_house["attack"]) == {"inverter"} and sorted(set(per_house["level"])) == [1, 3]
    assert len(per_house) == 4 and (res / "summary.md").exists()


# --------------------------------------------------------------------------- the aux cache

def test_aux_cache_reused_and_invalidated_by_sigma(tmp_path, monkeypatch):
    rec, p = make_recording(3, 2, tmp_path, sigma=0.5)
    cache = tmp_path / "cache"
    g1, w1, d1, i1 = aux_gradients.cached_aux_samples(rec, p, 8, cache)
    assert not i1["from_cache"]

    def boom(*a, **k):
        raise AssertionError("recomputed")
    monkeypatch.setattr(aux_gradients, "aux_samples", boom)
    g2, w2, d2, i2 = aux_gradients.cached_aux_samples(rec, p, 8, cache)          # no recomputation
    assert i2["from_cache"] and i2["path"] == i1["path"]
    assert torch.equal(g1, g2) and torch.equal(w1, w2) and d1 == d2
    with pytest.raises(AssertionError, match="recomputed"):                       # other sigma: a new key
        aux_gradients.cached_aux_samples({**rec, "sigma": 0.6}, p, 8, cache)
    monkeypatch.undo()
    g3, _, _, i3 = aux_gradients.cached_aux_samples({**rec, "sigma": 0.6}, p, 8, cache)
    assert not i3["from_cache"] and i3["path"] != i1["path"] and not torch.equal(g3, g1)


def test_ts_inverse_and_inverter_share_one_aux_set(tmp_path):
    from attacks import ts_inverse
    _, p = make_recording(2, 2, tmp_path, sigma=0.3)
    out, cache = tmp_path / "attack_outputs", tmp_path / "aux"
    assert ts_inverse.main(["--recording", str(p), "--out", str(out), "--aux-cache", str(cache),
                            "--steps", "2", "--aux-samples", "16", "--epochs", "1"]) == 0
    files = sorted(cache.glob("*.pt"))
    assert len(files) == 1
    assert learned_inverter.main(["--recording", str(p), "--out", str(tmp_path / "o2"), "--aux-cache", str(cache),
                                  *FAST]) == 0
    d2 = json.loads(learned_inverter.output_paths(p, tmp_path / "o2")[1].read_text())
    assert d2["aux"]["from_cache"] is True and d2["aux"]["path"] == str(files[0])
