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
   of the DP noise before SecAgg+ masks its update. We implement it.

## Plan for this semester

Everything runs on a PC, using [Flower](https://flower.ai) to simulate many houses on one machine.

1. **Forecasting task.** Train a small electricity-forecasting model on a public smart-meter dataset
   (London Smartmeter).
2. **Three protection levels:**

   | Level | Implementation |
   |---|---|
   | No protection | Plain Flower FL |
   | DP only | Flower's built-in local DP: each house adds its own full noise |
   | DP with SecAgg | Flower's built-in SecAgg+, plus our own step where each house adds its share of the noise before masking |

3. **Attacks.** Run both attacks against each level, with neighbourhoods of 5, 10 and 20 houses. We start with the
   learned inverter, and implement TS-Inverse from the paper if no public code is available.
4. **Measurements.** For each level, neighbourhood size and ε: how well the attacker reconstructs load curves, and
   how much forecast accuracy is lost.

## Repository layout

```
config.yaml        every setting (single source of truth; nothing hard-coded elsewhere)
common.py          shared model, local update, protection, seeds (used by clients AND attacks)
data.py            data interface (currently a FAKE random stub, to be replaced by the real loader)
requirements.txt   exact pinned versions (flwr pinned: SecAgg+ APIs change between versions)
data/              raw dataset files (gitignored, download locally)
train/             Flower clients, server, strategies
attacks/           learned inverter, TS-Inverse
eval/              metrics and plots
recordings/        attack recordings (gitignored; they go to the shared Drive, never git)
tests/             pytest tests
```

## Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build (optional)
pip install -r requirements.txt
pytest
```

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
