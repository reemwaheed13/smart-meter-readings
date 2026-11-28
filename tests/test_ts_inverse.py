"""TS-Inverse port (attacks/ts_inverse.py), aux gradients, its output format and the reproduction script.

Fake data only (the data.py stub). No real attack is run: step counts and aux sizes are tiny.
Scoring, matching and the blind guess are Shifa's (eval/); these tests only use them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import common
import data
from attacks import aux_gradients, reproduce_ts_inverse, ts_inverse
from attacks.recording import load_recording
from eval import score_all
from eval.score_all import load_attack_output, score_one
from train.env import machine_info

LR = 0.1
W = common.load_config()["experiment"]["windows_per_round"]


def make_recording(level: int, n: int, tmp_path=None, *, seed: int = 0, rnd: int = 1, view_is_sum: bool = False):
    """A recording built like train/ does, from the fake data, without noise (sigma 0)."""
    houses = data.get_neighbourhood(n, 0)
    dates = data.valid_dates(houses, "train")[10:10 + W]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = common.Model()
    true = data.get_windows(houses, dates, "train")
    C = None if level == 1 else 1.0
    views = torch.stack([aux_gradients.house_view(m, true[h], level, C, LR) for h in range(n)])
    if level == 3:
        sv = views.sum(0) if view_is_sum else views.mean(0)
    else:
        sv = views
    rec = {"level": level, "eps": None, "n": n, "nb": 0, "round": rnd, "server_view": sv,
           "view_is_sum": view_is_sum, "global_model": m.state_dict(), "true_windows": true,
           "house_ids": houses, "dates": dates, "lr": LR, "C": C, "sigma": None if level == 1 else 0.0,
           "round_time_s": 0.0, "bytes_per_house": 1.0, "env": machine_info()}
    if tmp_path is None:
        return rec
    p = tmp_path / f"L{level}_eps0_n{n}_nb0_r{rnd}.pt"
    torch.save(rec, p)
    load_recording(p)              # it must pass the recording checks
    return rec, p


def _settings(**over):
    return ts_inverse.load_settings(over)


# --------------------------------------------------------------------------- matching loss with the true windows

@pytest.mark.parametrize("level", [1, 2, 3])
def test_true_windows_give_zero_matching_loss(level):
    n = 3
    rec = make_recording(level, n)
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    C = rec["C"]
    if level == 3:
        mean_view, _ = ts_inverse.level3_views(rec)
        loss = ts_inverse.grad_match_loss(m, rec["true_windows"], mean_view, 3, C, LR)
        scale = mean_view.abs().sum()
    else:
        losses = [ts_inverse.grad_match_loss(m, rec["true_windows"][h:h + 1], rec["server_view"][h], level, C, LR)
                  for h in range(n)]
        loss, scale = max(losses), rec["server_view"].abs().sum(1).min()
    assert loss.item() < 1e-6 * float(scale)
    # and a random guess is far from zero
    g = torch.Generator().manual_seed(0)
    fake = torch.rand(rec["true_windows"].shape, generator=g)
    target = ts_inverse.level3_views(rec)[0] if level == 3 else rec["server_view"][0]
    bad = ts_inverse.grad_match_loss(m, fake if level == 3 else fake[:1], target, level, C, LR)
    assert bad.item() > 0.05 * float(scale)


# --------------------------------------------------------------------------- level 1, one house, one round

def test_level1_loss_falls_and_beats_blind_guess():
    rec = make_recording(1, 1, seed=3)
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    settings = _settings()                      # the configured attack: their 500 * 4 * 10 = 20000 steps
    res = ts_inverse.invert(m, rec["server_view"][0], 1, None, LR, 1, W, settings, seed=123)
    assert res["steps"] == 20000
    trace = np.array(res["trace"])              # [step, grad loss, total]
    chunks = np.array_split(trace[:, 1], 4)
    means = [c.mean() for c in chunks]
    assert all(a > b for a, b in zip(means, means[1:])), means        # falls steadily
    assert trace[-1, 1] < 0.2 * trace[0, 1]
    rows, _ = score_one(rec, res["recon"], ts_inverse.ATTACK, blind_mode="same_date")   # the grid's scoring
    a, b = rows[0]["smape_attack"], rows[0]["smape_blind"]
    assert a < b, (a, b)


def test_invert_is_deterministic():
    rec = make_recording(1, 1)
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    s = _settings(steps=20)
    r1 = ts_inverse.invert(m, rec["server_view"][0], 1, None, LR, 1, W, s, seed=7)
    r2 = ts_inverse.invert(m, rec["server_view"][0], 1, None, LR, 1, W, s, seed=7)
    r3 = ts_inverse.invert(m, rec["server_view"][0], 1, None, LR, 1, W, s, seed=8)
    assert torch.equal(r1["recon"], r2["recon"]) and r1["trace"] == r2["trace"]
    assert not torch.equal(r1["recon"], r3["recon"])


def test_steps_follow_their_formula():
    s = _settings()
    assert ts_inverse.num_steps(s, 1) == 5000 and ts_inverse.num_steps(s, 4) == 20000
    assert ts_inverse.num_steps(_settings(steps=7), 4) == 7


# --------------------------------------------------------------------------- level 3: the x n sum check

def test_level3_sum_check_uses_times_n():
    n = 4
    rec = make_recording(3, n)
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    true_sum = torch.stack([common.clip(common.local_update(m, rec["true_windows"][h], LR), rec["C"])
                            for h in range(n)]).sum(0)
    mean_view, sum_view = ts_inverse.level3_views(rec)
    assert rec["view_is_sum"] is False
    assert torch.allclose(sum_view, rec["server_view"] * n)
    assert torch.allclose(sum_view, true_sum, rtol=1e-5, atol=1e-7)
    # The mean-scale fake view times n is the sum the server would see.
    assert torch.allclose(ts_inverse.fake_view(m, rec["true_windows"], 3, rec["C"], LR).detach() * n, true_sum,
                          rtol=1e-5, atol=1e-7)
    # Handled wrongly (the mean taken as the sum), the check fails.
    assert not torch.allclose(rec["server_view"], true_sum, rtol=1e-2)
    # A recording that does hold the sum (view_is_sum True) gives the same mean and sum views...
    rec_sum = make_recording(3, n, view_is_sum=True)
    m2, s2 = ts_inverse.level3_views(rec_sum)
    assert torch.allclose(m2, mean_view, rtol=1e-5, atol=1e-8) and torch.allclose(s2, sum_view, rtol=1e-5, atol=1e-7)
    # ...and multiplying it by n anyway (ignoring view_is_sum) fails the check.
    assert not torch.allclose(rec_sum["server_view"] * n, true_sum, rtol=1e-2)


# --------------------------------------------------------------------------- output shapes as eval.score_all reads them

def test_level3_flat_output_is_matched_across_houses():
    n = 3
    rec = make_recording(3, n)
    flat = rec["true_windows"].reshape(n * W, -1)
    rebuilt = flat[torch.randperm(n * W, generator=torch.Generator().manual_seed(1))]     # [4n, 96], shuffled
    rows, _ = score_one(rec, rebuilt, ts_inverse.ATTACK, blind_mode="same_date")
    assert all(r["smape_attack"] == pytest.approx(0.0, abs=1e-6) for r in rows)
    assert all(r["smape_blind"] > 0 for r in rows)


def test_paper_smape_is_their_definition():
    a = np.array([0.0, 1.0, 0.5, 0.0])
    b = np.array([0.0, 0.5, 0.5, 0.2])
    # 0/0 -> 0;  2*0.5/1.5;  0;  2*0.2/0.2 = 2   (a fraction on the 0-1 scale, not %)
    assert reproduce_ts_inverse.paper_smape(a, b) == pytest.approx((0 + 2 / 3 + 0 + 2) / 4)
    assert reproduce_ts_inverse.SMAPE_SCALE == "paper scale (0-1), reproduction only"


# --------------------------------------------------------------------------- regularisers and inversion model

def test_regularisers():
    flat = torch.full((2, 96), 0.3)
    assert float(ts_inverse.trend_l1(flat)) == pytest.approx(0.0, abs=1e-7)
    line = torch.linspace(0, 1, 96).repeat(2, 1)
    assert float(ts_inverse.trend_l1(line)) == pytest.approx(0.0, abs=1e-6)
    day = torch.rand(1, 48, generator=torch.Generator().manual_seed(0))
    assert float(ts_inverse.periodicity(torch.cat([day, day], 1), 48)) == 0.0
    q = torch.tensor([[0.1, 0.3, 0.7, 0.9]]).repeat(48, 1)
    assert float(ts_inverse.bounds(torch.full((3, 48), 0.5), q)) == 0.0               # inside every band
    # 0.95 is outside both bands by 0.05 and 0.25 -> (0.05 + 0.25) / 2
    assert float(ts_inverse.bounds(torch.full((3, 48), 0.95), q)) == pytest.approx(0.15)


def test_aux_samples_shapes_clipping_noise_and_seed():
    rec = make_recording(3, 3)
    sigma = 0.6
    g1, w1 = aux_gradients.aux_samples(rec["global_model"], 3, 1.0, 3, LR, 6, seed=5, sigma=sigma)
    g2, w2 = aux_gradients.aux_samples(rec["global_model"], 3, 1.0, 3, LR, 6, seed=5, sigma=sigma)
    assert g1.shape == (6, common.num_params(common.Model())) and w1.shape == (6, 3 * W, 96)
    assert torch.equal(g1, g2) and torch.equal(w1, w2)
    m = common.Model()
    m.load_state_dict(rec["global_model"])
    clean = torch.stack([common.clip(common.local_update(m, w1[0, h * W:(h + 1) * W], LR), 1.0) for h in range(3)]).mean(0)
    # The samples are no longer the clean view: they carry the level-3 server view's noise, std sigma / n.
    assert float((g1[0] - clean).std()) == pytest.approx(sigma / 3, rel=0.05)
    g0, w0 = aux_gradients.aux_samples(rec["global_model"], 3, 1.0, 3, LR, 6, seed=5, sigma=0.0)
    assert torch.equal(w0, w1) and torch.allclose(g0[0], clean)              # sigma 0: the clean view
    pool = set(data.attacker_houses())
    assert not pool & set(rec["house_ids"])
    g3, w3 = aux_gradients.aux_samples(rec["global_model"], 1, None, 3, LR, 4, seed=5, windows_per_house=1)
    assert w3.shape == (4, 1, 96)


def test_inversion_model_trains_and_predicts():
    rec = make_recording(1, 1)
    s = _settings(inversion_model={"epochs": 2})
    grads, wins = aux_gradients.aux_samples(rec["global_model"], 1, None, 1, LR, 24, seed=1)
    model, log = ts_inverse.train_inversion_model(grads, wins, s, seed=2)
    model2, _ = ts_inverse.train_inversion_model(grads, wins, s, seed=2)
    qi, qt = ts_inverse.predict_quantiles(model, rec["server_view"][0])
    qi2, _ = ts_inverse.predict_quantiles(model2, rec["server_view"][0])
    assert qi.shape == (48, 4) and qt.shape == (48, 4) and torch.equal(qi, qi2)
    assert log["samples"] == 24 and log["val_samples"] >= 1


# --------------------------------------------------------------------------- CLI: outputs, resume

def test_cli_level1_writes_outputs_and_resumes(tmp_path, capsys):
    rec, p = make_recording(1, 2, tmp_path)
    out = tmp_path / "attack_outputs"
    args = ["--recording", str(p), "--out", str(out), "--no-inversion-model", "--steps", "10"]
    assert ts_inverse.main(args) == 0
    pt, js = out / f"{p.stem}__tsinverse.pt", out / f"{p.stem}__tsinverse.json"
    rebuilt = load_attack_output(pt)                                    # weights_only=True
    assert set(torch.load(pt, weights_only=True)) == {"rebuilt"}
    assert rebuilt.shape == (2, W, 96) and 0 <= float(rebuilt.min()) and float(rebuilt.max()) <= 1
    diag = json.loads(js.read_text())
    assert diag["level"] == 1 and len(diag["inversions"]) == 2 and diag["inversion_model"] is None
    assert diag["rebuilt_shape"] == [2, W, 96] and diag["seeds"]["run_seed"] == common.run_seed(2, 0)
    assert all(x["steps"] == 10 and x["grad_loss_final"] >= 0 for x in diag["inversions"])
    assert set(diag) == {"recording", "attack", "level", "eps", "n", "nb", "round", "rebuilt_shape", "seeds",
                         "fingerprint", "inversion_model", "inversions"}           # diagnostics only, no scores
    capsys.readouterr()
    assert ts_inverse.main(args) == 0
    assert "skipping" in capsys.readouterr().out
    assert json.loads(js.read_text()) == diag
    # Other settings: redone, and the house parts are not reused.
    assert ts_inverse.main(args[:-1] + ["11"]) == 0
    assert json.loads(js.read_text())["inversions"][0]["steps"] == 11


def test_cli_level3_with_inversion_model(tmp_path):
    n = 2
    rec, p = make_recording(3, n, tmp_path)
    out = tmp_path / "attack_outputs"
    args = ["--recording", str(p), "--out", str(out), "--steps", "6", "--aux-samples", "12", "--epochs", "1"]
    assert ts_inverse.main(args) == 0
    pt, js = out / f"{p.stem}__tsinverse.pt", out / f"{p.stem}__tsinverse.json"
    diag = json.loads(js.read_text())
    assert diag["inversion_model"]["samples"] == 12
    chk = diag["server_view_times_n_check"]
    assert chk["view_is_sum"] is False
    assert chk["server_view_times_n_l1_norm"] == pytest.approx(float((rec["server_view"] * n).abs().sum()), rel=1e-5)
    assert chk["true_windows_l1"] < 1e-5 * chk["server_view_times_n_l1_norm"]
    assert load_attack_output(pt).shape == (n * W, 96)
    first = load_attack_output(pt)
    js.unlink()                                 # interrupted after the attack: parts are reused
    assert ts_inverse.main(args) == 0
    assert torch.equal(load_attack_output(pt), first)


def test_cli_outputs_go_through_score_all(tmp_path, monkeypatch):
    recs = tmp_path / "recordings"
    recs.mkdir()
    paths = [make_recording(1, 2, recs)[1], make_recording(3, 2, recs)[1]]
    out = tmp_path / "attack_outputs"
    assert ts_inverse.main(["--recording", *map(str, paths), "--out", str(out), "--no-inversion-model",
                            "--steps", "4"]) == 0
    res = tmp_path / "results"
    monkeypatch.setattr("sys.argv", ["score_all", "--recordings", str(recs), "--attacks", str(out),
                                     "--out", str(res), "--blind", "same_date"])
    score_all.main()
    per_house = __import__("pandas").read_csv(res / "per_house.csv")
    assert set(per_house["attack"]) == {"tsinverse"} and sorted(set(per_house["level"])) == [1, 3]
    assert len(per_house) == 4 and (res / "summary.md").exists()


# --------------------------------------------------------------------------- reproduction script

def test_reproduce_refuses_fake_data(tmp_path, capsys):
    assert reproduce_ts_inverse.main(["--out", str(tmp_path)]) == 2
    assert "FAKE" in capsys.readouterr().err


def test_reproduce_attacked_state_is_their_init():
    # Their model is built right after torch.manual_seed(seed): same layers, same order, same init.
    torch.manual_seed(10)
    ref = torch.nn.Sequential(torch.nn.Linear(48, 64), torch.nn.Sigmoid(), torch.nn.Linear(64, 64),
                              torch.nn.Sigmoid(), torch.nn.Linear(64, 48))
    ours = reproduce_ts_inverse.attacked_state(10)
    for a, b in zip(ref.parameters(), ours.parameters()):
        assert torch.equal(a, b)


def test_reproduce_runs_on_fake_with_tiny_settings(tmp_path):
    args = ["--out", str(tmp_path), "--allow-fake", "--seeds", "10", "43", "--steps", "5",
            "--aux-samples", "8", "--epochs", "1"]
    assert reproduce_ts_inverse.main(args) == 0
    r = json.loads((tmp_path / "reproduction.json").read_text())
    assert set(r["rows"]) == {"table2_config", "ts_inverse_b1", "ours_level1_b4"}
    assert r["rows"]["table2_config"]["windows"] == 1 and r["rows"]["ours_level1_b4"]["windows"] == 4
    assert r["paper_table2_fcn_london_b1"] == {"input_smape": 6.3e-06, "target_smape": 2.5e-06}
    assert set(r["rows"]["ts_inverse_b1"]["attack"]["smape_input"]) == {"mean", "std"}
    assert r["data_is_fake"] is True
