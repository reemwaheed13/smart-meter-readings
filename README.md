# What Does Private Federated Learning Leak in Small Neighbourhoods?

**Differential privacy and secure aggregation for smart-meter load forecasting, tested against time-series gradient inversion**

*In one line:* we measure how much household load data an honest-but-curious Flower server can reconstruct from
federated updates under no protection, local DP, and distributed DP with SecAgg+, in neighbourhoods of 5 to 20 houses.

CS 5803 Applied Cryptography · LUMS · Fall 2026
Team: Reem, Shifa-e-Fatima · Supervisor: Dr. Ali

---

## Why this matters

Smart meters record household electricity use every few minutes. That load curve reveals a lot about the people
living there: when they wake up, leave, come home or go on holiday, and which appliances they own.

Federated learning (FL) lets many meters train a shared electricity-forecasting model without uploading their raw
readings. Each meter trains locally and sends only a **model update** to the server. But model updates leak:
a *gradient inversion attack* lets the server work backwards from an update and reconstruct the load curve that
produced it.

This project measures how well the two standard protections stop such attacks on realistic smart-meter data, in the
small neighbourhoods (5 to 20 houses) where real deployments operate and where protection is most likely to break.

## The two protections

**Differential privacy (DP).** Each meter clips its model update and adds random noise to it before sending it.
The noise is added to the **model update the meter sends**, not to each individual reading. The amount of noise is
set by a privacy budget ε: smaller ε means more noise, more privacy, and a less accurate forecast.

**Secure aggregation (SecAgg).** Each meter hides its update by adding random **masks** that are built to cancel out
when the server adds all the updates together. The server therefore learns only the **total** of the neighbourhood's
updates, never any single meter's update. This is not normal encryption: nothing is decrypted. The masks simply
disappear in the sum.

**DP with SecAgg (distributed DP).** Because the server only sees the total, each meter needs to add only its
**share** of the noise. The shares add up to the full amount of protection in the total.

## Threat model

- **Attacker:** an **honest-but-curious server**. It runs the protocol correctly, but tries to learn private data
  from everything it receives: individual updates when there is no SecAgg, or only the neighbourhood total when there
  is.
- **Protected:** each household's load curve.
- **Out of scope:** a malicious server that deviates from the protocol, fake clients, and side-channel attacks.

## What is new

Combining DP and SecAgg to improve accuracy is already known, and FL with DP or SecAgg has been applied to household
load forecasting in simulation (Delgado Fernández et al., 2021), but without attacks. This project contributes:

1. **Attacks built for time-series data.** We test each protection level against TS-Inverse (Meijer et al., 2025),
   a gradient inversion attack designed for time-series forecasting models, and against a learned inverter
   (Wu et al., 2023), rather than image-based attacks.
2. **Attacks on the neighbourhood total.** The TS-Inverse authors attacked one house at a time. We test what the
   server can recover when it sees only the **total of 5, 10 or 20 houses**, as it does under SecAgg. The number of
   houses is a main experiment, not a side note.
3. **A distributed-noise step for Flower's SecAgg+.** Flower does not offer a step where each house adds its share
   of the DP noise before SecAgg+ masks its update. We implement it as a client mod, `noise_share_mod`.

## Plan for this semester

Everything runs on a PC, using [Flower](https://flower.ai) to simulate many houses on one machine.

1. **Forecasting task.** Train a small electricity-forecasting model on a public smart-meter dataset
   (London Smartmeter).
2. **Three protection levels, and only these three:**

   | Level | Implementation | ε |
   |---|---|---|
   | 1. No protection | Plain Flower FL | none |
   | 2. DP only | Flower local DP: each house clips its update and adds its own full noise on the client | 1, 8 |
   | 3. DP with SecAgg | Flower's SecAgg+, plus our `noise_share_mod`, where each house adds its share of the noise before masking | 1, 8, and none (SecAgg+ alone) |

3. **Attacks.** Run both attacks, TS-Inverse and the learned inverter, at every level, with neighbourhoods of 5, 10
   and 20 houses. We start with the learned inverter, and implement TS-Inverse from the paper if no public code is
   available.
4. **Measurements.** For each level, neighbourhood size and ε:
   - **Attack success, compared with blind guessing:** sMAPE of the reconstructed load curves; how well the
     attacker recovers routine timing; home/away detection; and, at level 3, how much each single house is
     exposed when the server sees only the neighbourhood total.
   - **Forecast error** in kWh.
   - **Running cost:** time per round and bytes sent per house.

Out of scope this semester: anything beyond a PC running Flower. The microcontroller and dropouts are future work
(see below).

## Repository layout

```
config.yaml        every setting (single source of truth; nothing hard-coded elsewhere)
common.py          shared model, local update, protection, seeds (used by clients AND attacks)
data.py            data interface (currently a FAKE random stub, to be replaced by the real loader)
requirements.txt   exact pinned versions (flwr pinned: SecAgg+ APIs change between versions)
data/              raw dataset files (gitignored, download locally)
train/             Flower FedSGD: client, strategy, server/client apps, recorder, entry point (train.run)
attacks/           learned inverter, TS-Inverse; recording.py loads and checks a recording
eval/              metrics and plots
recordings/        attack recordings (gitignored; they go to the shared Drive, never git)
logs/              per-round CSV logs and test error (gitignored)
tests/             pytest tests
```

## Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch
pip install -r requirements.txt
pytest
```

Install the CPU build of torch first, as shown, and then the requirements. requirements.txt installs
`flwr[simulation]`, which brings in Ray: Ray is the only simulation backend in flwr 1.45.0. pytest runs real Flower
simulations, so it takes about two minutes.

## Running

```bash
python -m train.run --level 1 --eps none --n 5 --k 0 --lr 0.1 --T 10
```

`--lr` and `--T` are accepted only while `calibration.lr` / `calibration.T[n]` are null in config.yaml. Once
calibration fills them in, the run takes them from config.yaml and refuses the flags. If a needed value is null
and no flag is given, the run refuses to start. Each run writes:

- `logs/L{level}_eps{eps}_n{n}_nb{k}.csv`, one row per round with `round, val_mse, round_time_s, bytes_per_house`.
  `val_mse` is the global model's MSE on the neighbourhood's val windows after the round. `round_time_s` is the wall
  time of Flower's fit step for the round (send, local updates, receive, aggregate). Round 1 also includes Ray
  starting the client actors, about 20–40 s. `bytes_per_house` is the measured serialised size of the message
  content each house sends in the round, averaged over houses.
- `logs/L{level}_eps{eps}_n{n}_nb{k}_test.json` with the test MAE and RMSE in kWh after round T.
- Recordings at rounds 1, ⌈T/2⌉ and T. The middle round is `(T+1)//2`, and duplicates are dropped, so T = 2
  writes rounds 1 and 2 only.

## Flower wiring (flwr 1.45.0)

All three levels share one server/client API path. Level 1 is that path with nothing added. This is the path that
`SecAggPlusWorkflow` and `secaggplus_mod` require (read from the installed source):

- `SecAggPlusWorkflow.__call__` refuses any context that is not a `LegacyContext`. It drives the
  `LegacyContext`'s legacy `Strategy`: `configure_fit` in its setup stage, and `aggregate_fit` after unmasking.
- `secaggplus_mod` reads the client's reply with `compat.recorddict_to_fitres`, i.e. the legacy FitRes layout
  that a `NumPyClient` produces.

**Simulation:** `flwr.simulation.run_simulation(server_app, client_app, num_supernodes=n)` on the Ray backend
(`train/run.py`). It is the only Python API for Flower's simulation runtime in 1.45.0. It prints a deprecation
notice in favour of the `flwr run` CLI, which runs the same `ServerApp`/`ClientApp` objects.

**Server** (`train/app.py`, `train/fedsgd.py`):
`ServerApp` with `@app.main()` → `LegacyContext(strategy=FedSGD, config=ServerConfig(num_rounds=T))` →
`DefaultWorkflow(fit_workflow=RoundRunner(inner))`, run on a `MeteredGrid` that wraps the `Grid`.

- `FedSGD` is a legacy `flwr.server.strategy.Strategy`. `configure_fit` samples all n clients and puts the 4
  dates in the fit config. `aggregate_fit` takes the mean of the updates and applies `w -= lr * mean`.
- `RoundRunner` times the inner fit workflow, then logs val MSE and writes the recordings.
- `MeteredGrid.send_and_receive` adds up the protobuf size of every reply's content per node.
- Level 1: `inner = flwr.server.workflow.default_workflows.default_fit_workflow`.

**Client** (`train/client.py`): `ClientApp(client_fn=..., mods=[])`. `client_fn` maps
`node_config["partition-id"]` i to `house_ids[i]` and returns a `NumPyClient`. Its `fit` returns
`([update [P]], weight, {"house_id": ...})`, after `common.protect(update, level, C, sigma, n)`. Every house
reports the same weight (`simulation.client_weight`).

**Where level 2 hooks in:** in the client's `common.protect` call. At level 1 it returns the update unchanged.
At level 2 it clips to C and adds N(0, σ²) from a run_seed generator, as the handshake says. Flower's
`LocalDpMod` cannot be used as-is. It clips the *difference* between the parameters the client returns and the
parameters it received, which assumes the client returns weights. Our client returns a gradient. Its noise also
comes from numpy's global RNG, not from run_seed.

**Where level 3 hooks in**, with no restructuring:
- Server: `inner = SecAggPlusWorkflow(num_shares, reconstruction_threshold, clipping_range, ...)` from
  config.yaml `secaggplus`.
- Client: `mods=[secaggplus_mod, noise_share_mod]`. The first mod is the outermost, so `noise_share_mod` sees
  the plaintext update before `secaggplus_mod` quantises and masks it. Clip and noise share must be applied
  exactly once.
- `FedSGD.aggregate_fit` then gets results whose parameters all hold the same unmasked vector. Flower divides
  it by the total weight, so with equal weights it is the **mean** of the n updates, not their sum. That vector
  becomes the `[P]` `server_view`.
- `MeteredGrid` already counts the bytes of all four SecAgg+ stages.

## Handshakes

These are the agreements both halves of the project code against. Change them only by agreement, and update this
section when you do.

### common.py

- `load_config(path="config.yaml")`
- `Model`: Linear(48,64) -> sigmoid -> Linear(64,64) -> sigmoid -> Linear(64,48). MSE loss.
- `num_params(model)` -> P (P = 10416)
- `local_update(model, windows, lr)`: windows is a float tensor [4, 96] scaled 0–1. Inputs are the first 48
  readings, targets the last 48. One FedSGD step: compute the mean MSE gradient over the 4 windows and return
  it as a flat tensor [P]. Do NOT modify the model in place. The update is defined as the gradient; the server
  applies model -= lr * aggregate. (TS-Inverse matches gradients.)
- `protect(update, level, C, sigma, n, *, generator=None)`:
  - level 1: return unchanged
  - level 2: clip to L2 norm C, add N(0, sigma^2) per coordinate
  - level 3: clip to L2 norm C, add N(0, (sigma/sqrt(n))^2) per coordinate (the noise share)
  - sigma None or 0 means no noise (level 3, eps = none)

  The generator comes from run_seed so the noise is reproducible.
- `sigma_for(eps, C, delta)` = C * sqrt(2*ln(1.25/delta)) / eps; returns 0 when eps is None.
  Caveat: the classic Gaussian bound is only proven for eps < 1, so eps = 8 is a per-round label
  and the zCDP accountant's total is the reported guarantee.
- `run_seed(n, k)` = 1000*n + 10*k, plus `set_seeds(seed)` for torch, numpy and random.
- `flatten` / `unflatten` helpers between state_dict and [P].

### config.yaml

config.yaml is the single source of settings. Nothing is hard-coded elsewhere. Values that calibration
produces (lr, T per n, C, clipping_range, sigma per eps) stay null until calibration fills them in.

### Recording format

One `torch.save` dict per run per attacked round at

```
recordings/L{level}_eps{eps}_n{n}_nb{k}_r{round}.pt
```

(`eps0` for level 1 and level-3 no-noise), with keys:

| Key | Content |
|---|---|
| `level`, `eps`, `n`, `nb`, `round` | run identifiers |
| `server_view` | `[n, P]` for levels 1–2; `[P]` for level 3, as returned by Flower's aggregate step, de-quantised |
| `view_is_sum` | bool |
| `global_model` | state_dict at the start of that round |
| `true_windows` | `[n, 4, 96]` |
| `house_ids` | list of n, same order |
| `dates` | 4 `"YYYY-MM-DD"` |
| `lr`, `C`, `sigma` | |
| `round_time_s` | |
| `bytes_per_house` | |

`view_is_sum` only matters at level 3, where `server_view` is the single `[P]` aggregate instead of one row per
house. It is always False at levels 1–2. At level 1, `C` and `sigma` are None (no clipping, no noise).
`attacks/recording.py: load_recording(path)` checks every key and shape. It loads with `weights_only=True`, so a
file from the shared Drive cannot run code.

### data.py interface

- `get_neighbourhood(n, k)` -> n house IDs (deterministic in n, k)
- `valid_dates(house_ids, split)` -> list of `"YYYY-MM-DD"`
- `get_windows(house_ids, dates, split)` -> tensor `[houses, dates, 96]` in 0–1
- `attacker_houses()` -> 500 IDs
- `unscale(x)`

The current data.py is a FAKE stub with random data, to be replaced by the real data.py.

### Seed convention

Houses depend only on (n, k). run_seed sets the model init, per-round dates and noise. The same run_seed is used
at every level.

## Future work

- **Dropouts:** what happens when some meters drop out during a round, since this is where SecAgg's protection may
  weaken.
- **Microcontroller cost:** running the protected client on a meter-class microcontroller (an EFR32MG24 / Arm
  Cortex-M33 development kit) and measuring time, memory and energy, including key agreement.
- **Side channels:** whether power or timing measurements of the on-device noise and mask generation leak the
  values the protection depends on.

These are planned as a follow-up directed research project.

## Key references

- Meijer et al. *TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series Forecasting Models.*
  IEEE SaTML 2025. [arXiv:2503.20952](https://arxiv.org/abs/2503.20952)
- Delgado Fernández et al. Privacy-preserving federated learning for residential short-term load forecasting. 2021.
- Wu et al. *Learning to Invert: Simple Adaptive Attacks for Gradient Inversion in Federated Learning.* UAI 2023.
- Abadi et al. *Deep Learning with Differential Privacy.* ACM CCS 2016.
- Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* ACM CCS 2017.
- Bell et al. *Secure Single-Server Aggregation with (Poly)Logarithmic Overhead* (SecAgg+). ACM CCS 2020.
- Kairouz et al. *The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation.*
  ICML 2021.
