"""The house (Flower client): one FedSGD gradient per round on its own 4 windows.

What the client returns depends on the level (see README, "Flower wiring"):

- level 1: the gradient g, flat [P] (common.protect at level 1 returns it unchanged).
- level 2: its NEW weights w - lr*g, one array per state_dict tensor. Flower's LocalDpMod then
  clips "returned minus received" (= -lr*g, the true update) to lr*C and adds its noise.
- level 3: the raw gradient g, flat [P]. noise_share_mod clips it and adds the noise share,
  then secaggplus_mod quantises and masks it.

Just before returning, the client seeds numpy's global generator from (run_seed, round,
partition id). LocalDpMod's noise (level 2) and SecAgg+'s stochastic rounding (level 3) both
draw from numpy's global generator, so this makes them reproducible.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from flwr.app import Context
from flwr.client import NumPyClient

import common
import data
from train.noise_share_mod import noise_seed
from train.settings import RunSettings


def weights_from_ndarrays(arrays: list[np.ndarray]) -> "OrderedDict[str, torch.Tensor]":
    """Global weights as sent by the server (one array per state_dict tensor, in state_dict order)."""
    keys = list(common.Model().state_dict().keys())
    if len(arrays) != len(keys):
        raise ValueError(f"expected {len(keys)} weight arrays, got {len(arrays)}")
    return OrderedDict((k, torch.from_numpy(np.array(a, copy=True))) for k, a in zip(keys, arrays))


class HouseClient(NumPyClient):
    def __init__(self, partition_id: int, s: RunSettings):
        self.pid = partition_id
        self.s = s
        self.house_id = s.house_ids[partition_id]

    def fit(self, parameters, config):
        s = self.s
        server_round = int(config["server_round"])
        weights = weights_from_ndarrays(parameters)
        model = common.Model()
        model.load_state_dict(weights)

        dates = str(config["dates"]).split(",")
        if len(dates) != s.windows_per_round:
            raise ValueError(f"expected {s.windows_per_round} dates, got {dates}")
        windows = data.get_windows([self.house_id], dates, "train")[0]  # [4, 96]

        update = common.local_update(model, windows, s.lr)              # flat [P] gradient
        if s.test_zero_update:                                          # TEST-ONLY hook
            update = torch.zeros_like(update)
        seed = noise_seed(s.run_seed, server_round, self.pid)

        if s.level == 1:
            gen = torch.Generator().manual_seed(seed)
            out = [common.protect(update, 1, s.C, s.sigma, s.n, generator=gen).numpy()]
        elif s.level == 2:
            new = common.flatten(weights) - s.lr * update               # w - lr*g
            out = [t.numpy() for t in common.unflatten(new, weights).values()]
        elif s.level == 3:
            out = [update.numpy()]                                      # noise_share_mod protects it
        else:
            raise ValueError(f"unknown level {s.level}")

        np.random.seed(seed)  # immediately before returning: LocalDpMod / SecAgg+ rounding draw from it
        # Same weight for every house: no client counts more in the aggregate.
        return out, s.num_examples, {"house_id": self.house_id}


def make_client_fn(s: RunSettings):
    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        return HouseClient(pid, s).to_client()
    return client_fn
