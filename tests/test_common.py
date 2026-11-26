import math

import pytest
import torch

import common
import data
from common import Model, local_update, num_params, protect, sigma_for


def _model(seed=0):
    common.set_seeds(seed)
    return Model()


def _windows(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(4, 96, generator=g)


def test_local_update_shape_and_model_unchanged():
    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    u = local_update(m, _windows(), lr=0.1)
    assert u.shape == (num_params(m),)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in m.parameters())


def test_local_update_matches_manual_gradient():
    m = _model()
    w = _windows()
    u = local_update(m, w, lr=0.1)
    loss = torch.nn.functional.mse_loss(m(w[:, :48]), w[:, 48:])
    loss.backward()
    manual = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
    assert torch.allclose(u, manual)


def test_same_seed_same_update():
    u1 = local_update(_model(123), _windows(7), lr=0.1)
    u2 = local_update(_model(123), _windows(7), lr=0.1)
    assert torch.equal(u1, u2)


def test_protect_noise_reproducible_with_generator():
    u = local_update(_model(), _windows(), lr=0.1)
    a = protect(u, 2, 1.0, 0.5, 5, generator=torch.Generator().manual_seed(common.run_seed(5, 1)))
    b = protect(u, 2, 1.0, 0.5, 5, generator=torch.Generator().manual_seed(common.run_seed(5, 1)))
    assert torch.equal(a, b)


def test_protect_level1_identity():
    u = torch.randn(1000)
    assert torch.equal(protect(u, 1, 0.1, 5.0, 10), u)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("sigma", [0, None])
def test_clipping_norm_at_most_C(level, sigma):
    C = 0.5
    u = torch.randn(10_000) * 10
    out = protect(u, level, C, sigma, 10)
    assert torch.linalg.vector_norm(out) <= C + 1e-6
    small = torch.randn(10_000) * 1e-4  # already inside the ball: unchanged
    assert torch.allclose(protect(small, level, C, sigma, 10), small)


def test_level3_noise_share_and_sum():
    sigma, n, P, trials = 2.0, 10, 10_416, 20
    g = torch.Generator().manual_seed(common.run_seed(n, 0))
    zero = torch.zeros(P)
    shares = torch.stack([
        torch.stack([protect(zero, 3, 1.0, sigma, n, generator=g) for _ in range(n)])
        for _ in range(trials)
    ])  # [trials, n, P]
    per_client_std = shares.std().item()
    sum_std = shares.sum(dim=1).std().item()
    assert per_client_std == pytest.approx(sigma / math.sqrt(n), rel=0.02)
    assert sum_std == pytest.approx(sigma, rel=0.02)


def test_level2_noise_std():
    sigma = 1.5
    g = torch.Generator().manual_seed(0)
    out = protect(torch.zeros(100_000), 2, 1.0, sigma, 10, generator=g)
    assert out.std().item() == pytest.approx(sigma, rel=0.02)


def test_sigma_for():
    assert sigma_for(None, 1.0, 1e-5) == 0
    assert sigma_for(1.0, 2.0, 1e-5) == pytest.approx(2.0 * math.sqrt(2 * math.log(1.25e5)))


def test_run_seed():
    assert common.run_seed(10, 2) == 10_020


def test_flatten_roundtrip():
    m = _model()
    sd = m.state_dict()
    flat = common.flatten(sd)
    assert flat.shape == (num_params(m),)
    back = common.unflatten(flat, sd)
    assert all(torch.equal(back[k], sd[k]) for k in sd)


def test_config_loads():
    cfg = common.load_config()
    assert cfg["experiment"]["windows_per_round"] == 4
    assert cfg["data"]["window_length"] == 96


def test_data_stub_interface():
    hs = data.get_neighbourhood(5, 1)
    assert hs == data.get_neighbourhood(5, 1) and len(hs) == 5
    assert not set(hs) & set(data.attacker_houses())
    assert len(data.attacker_houses()) == 500
    dates = data.valid_dates(hs, "train")[:4]
    w = data.get_windows(hs, dates, "train")
    assert w.shape == (5, 4, 96) and w.min() >= 0 and w.max() <= 1
    splits = [set(data.valid_dates(hs, s)) for s in ("train", "val", "test")]
    assert not (splits[0] & splits[1] or splits[1] & splits[2] or splits[0] & splits[2])
