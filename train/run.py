"""Entry point: python -m train.run --level 1 --eps none --n 5 --k 0 [--lr 0.1 --T 10]

Launches a Flower simulation (flwr.simulation.run_simulation, Ray backend) with n clients.
--lr and --T are accepted only while calibration.lr / calibration.T[n] are null in config.yaml.
"""
from __future__ import annotations

import argparse
import sys

from flwr.simulation import run_simulation

from train.app import backend_config, make_client_app, make_server_app
from train.settings import RunSettings, resolve


def run(s: RunSettings) -> None:
    print(f"[{s.run_name}] houses={list(s.house_ids)} T={s.T} lr={s.lr} run_seed={s.run_seed}", flush=True)
    run_simulation(
        server_app=make_server_app(s),
        client_app=make_client_app(s),
        num_supernodes=s.n,
        backend_config=backend_config(s),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--eps", required=True, help="'none' or a number")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--lr", type=float, default=None, help="only while calibration.lr is null")
    p.add_argument("--T", type=int, default=None, help="only while calibration.T[n] is null")
    a = p.parse_args(argv)
    try:
        s = resolve(a.level, a.eps, a.n, a.k, lr=a.lr, T=a.T)
    except (ValueError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
