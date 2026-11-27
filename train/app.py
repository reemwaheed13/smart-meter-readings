"""Flower ServerApp / ClientApp for all three levels (see README, "Flower wiring").

Server: ServerApp(main) -> LegacyContext(strategy=FedSGD) -> DefaultWorkflow(fit_workflow=RoundRunner(inner)).
        Levels 1-2 use inner = flwr's default_fit_workflow. Level 3 uses SecAggPlusWorkflow.
Client: ClientApp(client_fn -> NumPyClient, mods=[...]). Level 1: no mods. Level 2: [LocalDpMod].
        Level 3: [secaggplus_mod, noise_share_mod] (noise_share_mod runs inside secaggplus_mod).
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

from flwr.app import Context
from flwr.client.mod import LocalDpMod, secaggplus_mod
from flwr.clientapp import ClientApp
from flwr.common.serde import recorddict_to_proto
from flwr.server import ServerConfig
from flwr.server.compat import LegacyContext
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, Key
from flwr.server.workflow.default_workflows import default_fit_workflow
from flwr.serverapp import Grid, ServerApp

import common
from train.client import make_client_fn
from train.env import machine_info
from train.fedsgd import FedSGD
from train.noise_share_mod import NoiseShareMod
from train.recorder import rounds_to_record, save_recording
from train.settings import RunSettings


class MeteredGrid:
    """Wraps the Grid and counts the bytes each house actually sends: the serialised protobuf size
    of every reply's content (RecordDict), summed per node. At level 3 this sums over all four
    SecAgg+ stages, because every stage goes through send_and_receive."""

    def __init__(self, grid: Grid):
        self._grid = grid
        self.bytes_by_node: dict[int, int] = defaultdict(int)

    def __getattr__(self, name):
        return getattr(self._grid, name)

    def reset(self) -> None:
        self.bytes_by_node = defaultdict(int)

    def send_and_receive(self, messages, *, timeout=None):
        replies = list(self._grid.send_and_receive(messages, timeout=timeout))
        for m in replies:
            if m.has_content():
                self.bytes_by_node[m.metadata.src_node_id] += recorddict_to_proto(m.content).ByteSize()
        return replies


CSV_COLUMNS = ["round", "val_mse", "round_time_s", "bytes_per_house", "is_warmup", "clipping_range_cuts"]


class ClippingRangeExceeded(RuntimeError):
    """Some values of the protected updates were beyond the SecAgg+ clipping_range, so SecAgg+
    would have cut them silently. The run is refused: raise calibration.clipping_range."""


class RoundRunner:
    """Fit workflow wrapper: times one round, reads the byte meter, logs val MSE, and writes the
    recordings. Timing covers only the inner Flower fit workflow (send, local updates, receive,
    aggregate), not the logging or recording done here afterwards."""

    def __init__(self, s: RunSettings, inner, strategy: FedSGD, meter: MeteredGrid):
        self.s, self.inner, self.strategy, self.meter = s, inner, strategy, meter
        self.record_rounds = set(rounds_to_record(s.T))
        self.csv_path = s.logs_dir / f"{s.run_name}.csv"
        s.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        self.recorded: list[str] = []

    def __call__(self, grid, context: LegacyContext) -> None:
        r = int(context.state.config_records[MAIN_CONFIGS_RECORD][Key.CURRENT_ROUND])
        self.meter.reset()
        t0 = time.perf_counter()
        self.inner(grid, context)
        round_time = time.perf_counter() - t0

        if self.strategy.last_round is None or self.strategy.last_round["round"] != r:
            raise RuntimeError(f"round {r} did not aggregate")
        sent = list(self.meter.bytes_by_node.values())
        if len(sent) != self.s.n:
            raise RuntimeError(f"round {r}: bytes measured for {len(sent)} houses, expected {self.s.n}")
        bytes_per_house = sum(sent) / len(sent)

        val = self.strategy.val_mse()
        cuts = self.strategy.last_round["clip_cuts"]   # None at levels 1-2
        with open(self.csv_path, "a", newline="") as f:
            # Round 1 is the warm-up (it includes Ray starting the client actors); the reported running
            # cost is the median round_time_s over rounds 2..T (train.costs.running_cost).
            csv.writer(f).writerow([r, f"{val:.8g}", f"{round_time:.6f}", f"{bytes_per_house:g}",
                                    r == 1, "" if cuts is None else cuts])
        print(f"[{self.s.run_name}] round {r}/{self.s.T} val_mse={val:.6g} "
              f"time={round_time:.3f}s bytes/house={bytes_per_house:g}"
              + ("" if cuts is None else f" clipping_range_cuts={cuts}"), flush=True)
        if cuts:
            raise ClippingRangeExceeded(
                f"[{self.s.run_name}] round {r}: {cuts} values of the protected updates exceed the SecAgg+ "
                f"clipping_range {self.s.clipping_range}; SecAgg+ would cut them. Raise calibration.clipping_range.")

        if r in self.record_rounds:
            self.recorded.append(str(save_recording(self.s, self.strategy.last_round, round_time, bytes_per_house)))
        if r == self.s.T:
            test = self.strategy.test_error_kwh()
            with open(self.s.logs_dir / f"{self.s.run_name}_test.json", "w") as f:
                json.dump({"round": r, **test, "clipping_range": self.s.clipping_range, "env": machine_info()},
                          f, indent=2)
            print(f"[{self.s.run_name}] test MAE={test['test_mae_kwh']:.4f} kWh "
                  f"RMSE={test['test_rmse_kwh']:.4f} kWh", flush=True)


def make_server_app(s: RunSettings) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        common.set_seeds(s.run_seed)  # before model init
        strategy = FedSGD(s, common.Model().state_dict())
        legacy = LegacyContext(context=context, config=ServerConfig(num_rounds=s.T), strategy=strategy)
        meter = MeteredGrid(grid)
        inner = secaggplus_workflow(s) if s.level == 3 else default_fit_workflow
        DefaultWorkflow(fit_workflow=RoundRunner(s, inner, strategy, meter))(meter, legacy)

    return app


def secaggplus_workflow(s: RunSettings) -> SecAggPlusWorkflow:
    """SecAgg+ with config.yaml's num_shares, reconstruction_threshold and clipping_range. Quantisation
    and modulus stay at Flower's defaults unless config.yaml sets them."""
    kw = {}
    if s.quantization_range is not None:
        kw["quantization_range"] = int(s.quantization_range)
    if s.modulus_range is not None:
        kw["modulus_range"] = int(s.modulus_range)
    return SecAggPlusWorkflow(num_shares=s.num_shares, reconstruction_threshold=s.reconstruction_threshold,
                              clipping_range=s.clipping_range, **kw)


def client_mods(s: RunSettings) -> list:
    if s.level == 1:
        return []
    if s.level == 2:
        # The client returns w - lr*g, so "returned minus received" is -lr*g. Clipping it to lr*C and
        # noising with sensitivity lr*C is, in gradient terms, clip to C + N(0, sigma_for(eps, C, delta)^2).
        lc = s.lr * s.C
        return [LocalDpMod(clipping_norm=lc, sensitivity=lc, epsilon=s.eps, delta=s.delta)]
    if s.level == 3:
        return [secaggplus_mod, NoiseShareMod(C=s.C, sigma=s.sigma, n=s.n, run_seed=s.run_seed,
                                              clipping_range=s.clipping_range)]
    raise ValueError(f"unknown level {s.level}")


def make_client_app(s: RunSettings) -> ClientApp:
    return ClientApp(client_fn=make_client_fn(s), mods=client_mods(s))


def backend_config(s: RunSettings) -> dict:
    return {"client_resources": {"num_cpus": s.client_num_cpus, "num_gpus": 0.0}}

