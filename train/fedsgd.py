"""Server-side FedSGD strategy (a legacy flwr.server.strategy.Strategy).

It is a legacy Strategy on purpose: flwr 1.45.0's SecAggPlusWorkflow only drives a
LegacyContext's Strategy (configure_fit in its setup stage, aggregate_fit after unmasking), so
level 3 reuses this class unchanged.

Updates are gradients (see common.local_update). Each round:
    aggregate   = mean of the client updates (every client reports the same weight)
    new weights = old weights - lr * aggregate
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy

import common
import data
from train.settings import RunSettings


def to_ndarrays(state_dict) -> list[np.ndarray]:
    return [t.detach().cpu().numpy() for t in state_dict.values()]


class FedSGD(Strategy):
    def __init__(self, s: RunSettings, initial_state: "OrderedDict[str, torch.Tensor]"):
        super().__init__()
        self.s = s
        self.template = OrderedDict((k, v.clone()) for k, v in initial_state.items())
        self.initial_state = initial_state
        # One generator, seeded with run_seed, picks the training dates for every round.
        self.date_gen = torch.Generator().manual_seed(s.run_seed)
        self.train_dates = data.valid_dates(list(s.house_ids), "train")
        if len(self.train_dates) < s.windows_per_round:
            raise ValueError("not enough valid training dates")
        self.round_start: dict | None = None   # set by configure_fit
        self.last_round: dict | None = None    # set by aggregate_fit, read by the round runner
        self.current = common.flatten(initial_state).clone()

        val_dates = data.valid_dates(list(s.house_ids), "val")
        self.val_windows = data.get_windows(list(s.house_ids), val_dates, "val").reshape(-1, 96)

    # ------------------------------------------------------------------ helpers
    def model_at(self, flat: torch.Tensor) -> common.Model:
        m = common.Model()
        m.load_state_dict(common.unflatten(flat, self.template))
        return m

    def pick_dates(self) -> list[str]:
        idx = torch.randperm(len(self.train_dates), generator=self.date_gen)[: self.s.windows_per_round]
        return [self.train_dates[i] for i in sorted(idx.tolist())]

    @torch.no_grad()
    def val_mse(self) -> float:
        m = self.model_at(self.current)
        w = self.val_windows
        return float(common.loss_fn(m(w[:, :48]), w[:, 48:]))

    @torch.no_grad()
    def test_error_kwh(self) -> dict[str, float]:
        hs = list(self.s.house_ids)
        w = data.get_windows(hs, data.valid_dates(hs, "test"), "test").reshape(-1, 96)
        m = self.model_at(self.current)
        err = data.unscale(m(w[:, :48])) - data.unscale(w[:, 48:])
        return {"test_mae_kwh": float(err.abs().mean()), "test_rmse_kwh": float(err.pow(2).mean().sqrt())}

    # ------------------------------------------------------------------ Strategy API
    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(to_ndarrays(self.initial_state))

    def configure_fit(self, server_round, parameters: Parameters, client_manager):
        n = self.s.n
        client_manager.wait_for(n)
        clients = client_manager.sample(num_clients=n, min_num_clients=n)  # every house, every round
        if len(clients) != n:
            raise RuntimeError(f"round {server_round}: expected {n} clients, got {len(clients)}")

        start = common.flatten(OrderedDict(
            (k, torch.from_numpy(np.array(a, copy=True)))
            for k, a in zip(self.template.keys(), parameters_to_ndarrays(parameters))))
        dates = self.pick_dates()
        self.round_start = {"round": server_round, "weights": start, "dates": dates}
        config = {"server_round": server_round, "dates": ",".join(dates)}
        return [(c, FitIns(parameters, config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"round {server_round}: {len(failures)} client failures (dropouts are out of scope)")
        rs = self.round_start
        if rs is None or rs["round"] != server_round:
            raise RuntimeError("aggregate_fit called without a matching configure_fit")
        weights = {fr.num_examples for _, fr in results}
        if len(weights) != 1:
            raise RuntimeError(f"clients reported different weights {weights}")

        # Levels 1-2: one update per house; order the rows as house_ids.
        by_house = {}
        for _, fr in results:
            hid = str(fr.metrics["house_id"])
            (arr,) = parameters_to_ndarrays(fr.parameters)
            by_house[hid] = torch.from_numpy(np.array(arr, copy=True))
        if sorted(by_house) != sorted(self.s.house_ids) or len(results) != self.s.n:
            raise RuntimeError(f"round {server_round}: results do not cover the neighbourhood exactly once")
        server_view = torch.stack([by_house[h] for h in self.s.house_ids])  # [n, P]
        # (Level 3: SecAggPlusWorkflow hands every result the same unmasked aggregate, so this
        # branch will take the single [P] vector instead; see README "Flower wiring".)

        aggregate = server_view.mean(dim=0)
        new = rs["weights"] - self.s.lr * aggregate
        self.current = new
        self.last_round = {
            "round": server_round,
            "global_model": common.unflatten(rs["weights"].clone(), self.template),
            "server_view": server_view,
            "dates": list(rs["dates"]),
        }
        return ndarrays_to_parameters(to_ndarrays(common.unflatten(new, self.template))), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []  # no federated evaluation; validation is centralised on the server (round runner)

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None
