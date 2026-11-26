"""Shared code for the Flower clients and the attacks.

The Flower clients (train/) and the learned inverter / TS-Inverse (attacks/) both import this
module, so the update an attacker simulates is computed by exactly the same code as the update
a house sends.
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_REPO_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- config

def load_config(path: str | Path = "config.yaml") -> dict:
    """Load config.yaml. A relative path that does not exist in the current directory is
    resolved against the repository root."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- model

INPUT_LEN = 48
HIDDEN = 64
OUTPUT_LEN = 48


class Model(nn.Module):
    """Linear(48,64) -> sigmoid -> Linear(64,64) -> sigmoid -> Linear(64,48). Loss: MSE."""

    def __init__(self, input_len: int = INPUT_LEN, hidden: int = HIDDEN, output_len: int = OUTPUT_LEN):
        super().__init__()
        self.fc1 = nn.Linear(input_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


loss_fn = nn.MSELoss()


def num_params(model: nn.Module) -> int:
    """P, the length of a flat update."""
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- flatten helpers
# Flat order is the state_dict key order (fc1.weight, fc1.bias, fc2.weight, ...), each
# tensor flattened row-major. The model has no buffers, so this is also parameters() order.

def flatten(state_dict: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    """state_dict -> flat tensor [P]."""
    return torch.cat([t.reshape(-1) for t in state_dict.values()])


def unflatten(flat: torch.Tensor, template: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """flat tensor [P] -> state_dict shaped like `template` (a state_dict or a model)."""
    if isinstance(template, nn.Module):
        template = template.state_dict()
    out, i = OrderedDict(), 0
    for k, t in template.items():
        n = t.numel()
        out[k] = flat[i:i + n].reshape(t.shape).to(t.dtype)
        i += n
    if i != flat.numel():
        raise ValueError(f"flat has {flat.numel()} values, template needs {i}")
    return out


# --------------------------------------------------------------------------- local update

def local_update(model: nn.Module, windows: torch.Tensor, lr: float, *, create_graph: bool = False) -> torch.Tensor:
    """One FedSGD step for one house. Returns the update as a flat tensor [P].

    windows: float tensor [4, 96], scaled 0-1. Inputs are windows[:, :48] and targets are
    windows[:, 48:].

    **The update is the gradient**, not a weight difference: it is the gradient of the mean
    MSE loss over the windows with respect to the model parameters, flattened in state_dict
    order (see `flatten`). The server applies

        model -= lr * aggregate

    so `lr` is *not* applied here. It is in the signature so every caller passes the
    calibrated lr through one path. TS-Inverse matches gradients, so the attacker compares
    against exactly this quantity.

    The model is not modified: no weights change and no `.grad` is written.

    create_graph=True keeps the graph so an attacker can differentiate the update with
    respect to `windows` (TS-Inverse gradient matching).
    """
    if windows.dim() != 2 or windows.shape[1] != INPUT_LEN + OUTPUT_LEN:
        raise ValueError(f"windows must be [B, {INPUT_LEN + OUTPUT_LEN}], got {tuple(windows.shape)}")
    x, y = windows[:, :INPUT_LEN], windows[:, INPUT_LEN:]
    params = list(model.parameters())
    loss = loss_fn(model(x), y)  # mean over windows and outputs
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    flat = torch.cat([g.reshape(-1) for g in grads])
    return flat if create_graph else flat.detach()


# --------------------------------------------------------------------------- protection

def clip(update: torch.Tensor, C: float) -> torch.Tensor:
    """Scale `update` so that its L2 norm is at most C. Does not modify the input."""
    norm = torch.linalg.vector_norm(update)
    scale = torch.clamp(C / (norm + 1e-12), max=1.0)
    return update * scale


def protect(update: torch.Tensor, level: int, C: float | None, sigma: float | None, n: int, *,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply the client-side protection for one house's flat update [P].

    level 1: return the update unchanged (no protection).
    level 2: clip to L2 norm C, add N(0, sigma^2) per coordinate (local DP, full noise).
    level 3: clip to L2 norm C, add N(0, (sigma/sqrt(n))^2) per coordinate. This is the
             house's noise share: the n shares sum to N(0, sigma^2) in the SecAgg+ total.
    sigma None or 0 means no noise (level 3 with eps = none).

    The generator should be seeded from run_seed so the noise is reproducible.
    """
    if level == 1:
        return update
    if level not in (2, 3):
        raise ValueError(f"unknown level {level}")
    if C is None:
        raise ValueError("levels 2 and 3 need a clipping norm C")
    out = clip(update, C)
    if sigma:
        std = sigma if level == 2 else sigma / math.sqrt(n)
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype, device=out.device)
        out = out + std * noise
    return out


def sigma_for(eps: float | None, C: float, delta: float) -> float:
    """Gaussian-mechanism noise std: sigma = C * sqrt(2 ln(1.25/delta)) / eps. Returns 0 when
    eps is None.

    Caveat: this classic Gaussian bound is only proven for eps < 1. For eps = 8 the value is a
    per-round *label* used to set the noise, not a guarantee. The reported privacy guarantee is
    the total from the zCDP accountant over all rounds.
    """
    if eps is None:
        return 0.0
    return C * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


# --------------------------------------------------------------------------- seeds

def run_seed(n: int, k: int) -> int:
    """run_seed = 1000*n + 10*k. Sets the model init, the per-round dates and the DP noise.
    The same run_seed is used at every level. The houses depend only on (n, k)."""
    return 1000 * n + 10 * k


def set_seeds(seed: int) -> None:
    """Seed torch, numpy and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
