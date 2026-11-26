"""The house (Flower client): one FedSGD gradient per round on its own 4 windows."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from flwr.app import Context
from flwr.client import NumPyClient

import common
import data
from train.settings import RunSettings


def noise_seed(run_seed: int, server_round: int, partition_id: int) -> int:
    """Seed of the generator a house uses for its DP noise in one round. Derived from run_seed,
    so the noise is reproducible (unused at level 1: protect() returns the update unchanged)."""
    return int(np.random.SeedSequence([run_seed, server_round, partition_id]).generate_state(1)[0])


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
        model = common.Model()
        model.load_state_dict(weights_from_ndarrays(parameters))

        dates = str(config["dates"]).split(",")
        if len(dates) != s.windows_per_round:
            raise ValueError(f"expected {s.windows_per_round} dates, got {dates}")
        windows = data.get_windows([self.house_id], dates, "train")[0]  # [4, 96]

        update = common.local_update(model, windows, s.lr)              # flat [P] gradient
        gen = torch.Generator().manual_seed(noise_seed(s.run_seed, int(config["server_round"]), self.pid))
        update = common.protect(update, s.level, s.C, s.sigma, s.n, generator=gen)

        # Same weight for every house: no client counts more in the aggregate.
        return [update.numpy()], s.num_examples, {"house_id": self.house_id}


def make_client_fn(s: RunSettings):
    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        return HouseClient(pid, s).to_client()
    return client_fn
