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
   | 2. DP only | Flower's built-in local DP (`LocalDpMod`): each house clips its update and adds its own full noise on the client | 1, 8 |
   | 3. DP with SecAgg | Flower's SecAgg+, plus our `noise_share_mod`, where each house adds its share of the noise before masking | 1, 8, and none (SecAgg+ alone) |

3. **Attacks.** Run both attacks, TS-Inverse and the learned inverter, at every level, with neighbourhoods of 5, 10
   and 20 houses. We start with the learned inverter, and use the authors' public code
   (github.com/Capsar/ts-inverse, CC BY-SA 4.0), adapted to our model and to summed updates.
4. **Measurements.** For each level, neighbourhood size and ε:
   - **Attack success, compared with blind guessing:** sMAPE of the reconstructed load curves; how well the
     attacker recovers routine timing; home/away detection; and, at level 3, how much each single house is
     exposed when the server sees only the neighbourhood total.
   - **Forecast error** in kWh.
   - **Running cost:** time per round and bytes sent per house. Round 1 is logged but flagged
     `is_warmup = True` (it includes Ray starting the client actors). The reported time per round is the
     **median `round_time_s` over rounds 2..T** (`train.costs.running_cost`).
   - **Privacy:** the total ε over all T rounds from the zCDP accountant (`train/accountant.py`).

Out of scope this semester: anything beyond a PC running Flower. The microcontroller and dropouts are future work
(see below).

## Repository layout

```
config.yaml        every setting (single source of truth; nothing hard-coded elsewhere)
common.py          shared model, local update, protection, seeds (used by clients AND attacks)
data.py            data interface: routes to data_real.py (default) or data_fake.py (SMART_METER_DATA=fake)
data_real.py       real London Smartmeter 2013 loader; `python data_real.py prepare` builds the cleaned .npz
data_fake.py       random stand-in with the same interface (the tests use it)
requirements.txt   exact pinned versions (flwr pinned: SecAgg+ APIs change between versions)
data/              raw dataset files (gitignored, download locally)
train/             Flower FedSGD: client, strategy, server/client apps, recorder, entry point (train.run)
attacks/           learned inverter, TS-Inverse (ts_inverse.py, reproduce_ts_inverse.py; CC BY-SA 4.0, see
                   LICENSE-THIRD-PARTY), aux_gradients.py (shared); recording.py loads and checks a recording
eval/              blind guess, scoring, result tables and figures (see "Evaluation")
recordings/        attack recordings (gitignored; they go to the shared Drive, never git)
logs/              per-round CSV logs and test error (gitignored)
tests/             pytest tests
notebooks/         colab_run.ipynb (the Stage 4 check), attack_ts_inverse.ipynb (both attacks); outputs on Google Drive
```

## Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch
pip install -r requirements.txt
pytest
```

Install the CPU build of torch first, as shown, and then the requirements. requirements.txt installs
`flwr[simulation]`, which brings in Ray: Ray is the only simulation backend in flwr 1.45.0. pytest runs real
(small) Flower simulations.

**The local environment is for writing code and running pytest only.** Every real run (smoke runs, timing test,
calibration and the experiment grid) happens in Colab through `notebooks/colab_run.ipynb`, with outputs on
Google Drive (see "Running in Colab"). Times measured on a local machine are not the project's running cost.

## Running

These are the commands the Colab notebook calls (shown here for reference; run them in Colab).

```bash
python -m train.run --level 1 --eps none --n 5 --k 0 --lr 0.1 --T 10
python -m train.run --level 2 --eps 1    --n 5 --k 0 --lr 0.1 --T 10 --C 1.0
python -m train.run --level 3 --eps 8    --n 5 --k 0 --lr 0.1 --T 10 --C 1.0 --clipping-range 16.0 --out runs/
```

`--lr`, `--T`, `--C` and `--clipping-range` are accepted only while `calibration.lr` / `calibration.T[n]` /
`calibration.C` / `calibration.clipping_range` are null in config.yaml. Once calibration fills them in, the run
takes them from config.yaml and refuses the flags. If a needed value is null and no flag is given, the run refuses
to start. σ is `dp.sigma[eps]` if set, otherwise `common.sigma_for(eps, C, delta)`.

`--out DIR` puts `recordings/` and `logs/` under DIR (default: the repo's `recordings/` and `logs/`). Runs are
**resumable**: a run whose recordings and both log files all exist in DIR, pass `load_recording`, cover rounds
1..T and logged the current clipping_range is skipped. A partial run, or a completed run whose test JSON logged a
different clipping_range (a JSON without the key counts as null, so an older level-3 run is redone), is deleted and
rerun from scratch.

**Pre-run clipping_range check (level 3 with noise).** Before anything starts, the run computes the expected number
of protected values beyond the SecAgg+ clipping_range over the whole run, n · T · P · 2·Φ(−clipping_range / (σ/√n)),
from the noise shares alone (the update, clipped to norm C, is ignored). If it is above 0.01 the run refuses to start,
deletes nothing, and names the expected count and the smallest clipping_range that brings it under 0.01
(`train.settings.check_clipping_range`). Levels 1–2 and eps = none skip the check. Example: C = 1, eps = 1, n = 5,
T = 10 has σ/√5 ≈ 2.17, so clipping_range 8.0 (3.7 stds) expects about 116 cuts and is refused; the smallest passing
value is 12.18. Each run writes:

- `logs/L{level}_eps{eps}_n{n}_nb{k}.csv`, one row per round with
  `round, val_mse, round_time_s, bytes_per_house, is_warmup, clipping_range_cuts`.
  `val_mse` is the global model's MSE on the neighbourhood's val windows after the round. `round_time_s` is the wall
  time of Flower's fit step for the round (send, local updates, receive, aggregate; at level 3 all four SecAgg+
  stages). `is_warmup` is True for round 1, which also includes Ray starting the client actors; the
  reported running cost is the median `round_time_s` over rounds 2..T. `bytes_per_house` is the measured serialised
  size of the message content each house sends in the round (all SecAgg+ stages at level 3), averaged over houses.
  `clipping_range_cuts` (level 3 only, empty otherwise) is the number of values, over all houses, of the protected
  updates with |x| > clipping_range. SecAgg+ would cut them silently, so **any count above 0 fails the run**
  (`ClippingRangeExceeded`) after the row is logged.
- `logs/L{level}_eps{eps}_n{n}_nb{k}_test.json` with the test MAE and RMSE in kWh after round T, the run's
  `clipping_range` (null at levels 1–2), and `env` (below).
- Recordings at rounds 1, ⌈T/2⌉ and T. The middle round is `(T+1)//2`, and duplicates are dropped, so T = 2
  writes rounds 1 and 2 only.

Every recording and test JSON stores `env`: the python, flwr and torch versions, the CPU model, the core count and
whether a GPU was visible (`train/env.py`).

**Privacy table:** `python -m train.accountant --out DIR [--C 1.0 --T 10]` writes `DIR/privacy_table.csv`
(`level, eps_per_round, sigma, T, eps_total`). Gaussian mechanism, sensitivity C, noise std σ, every house in every
round, no subsampling: ρ per round = C²/(2σ²), total ρ = T·ρ, ε_total = ρ + 2√(ρ ln(1/δ)). The same holds at
levels 2 and 3: at level 3 the sum carries total std σ, and one house moves the sum by at most C. eps = none gives
ε_total = ∞. The per-round ε (1 or 8) is a label that sets σ through the classic Gaussian bound, which is only
proven for ε < 1. The zCDP total is the reported guarantee.

**Timing test:** `python -m train.timing_test --out DIR --T <rounds per run> [--lr --C --clipping-range]` runs
50 rounds of level 3 (n = 20, eps = 1) and writes `DIR/timing.json`: the median `round_time_s` over rounds 2..50,
`bytes_per_house`, `env`, and `projected_total_s = median × T × 27` (the 27 level-3 runs). Fake data is fine
here: the cost does not depend on the readings.

## Running in Colab

`notebooks/colab_run.ipynb` does the whole Stage 4 check in one *Runtime → Run all*.

**One-time setup: the GitHub token.** The repo may be private, so the notebook clones it with a token that it
reads from a Colab secret. In Colab: left sidebar → key icon (*Secrets*) → *Add new secret*. Use the name
`GH_TOKEN` and, as the value, a GitHub personal access token with read access to
`reemwaheed13/smart-meter-readings` (a fine-grained token with *Contents: read-only* is enough). Then switch on
*Notebook access*. The notebook never prints the token, and it doesn't keep it in the clone's git remote. If the
secret is missing, the clone cell prints these instructions and tries an anonymous clone, which works only for a
public repo.

**Run all.** The cells, in order:

1. mount Google Drive and create `OUT = /content/drive/MyDrive/smart-meter-fl/stage4`. The same cell holds the
   only settings: `T_PROJECTION` (default 100), `COMPUTE_BUDGET_H` (default 12) and `FORCE_TIMING`;
2. clone `reemwaheed13/smart-meter-readings` into `/content/smart-meter-readings` with `GH_TOKEN`, or
   `git pull --ff-only` if it is already there;
3. `pip install flwr[simulation]==1.45.0 pyyaml==6.0.4` and nothing else, keeping Colab's own torch and
   numpy. Then print python, torch, numpy, flwr and ray versions, CPU model, core count and GPU (or "none"),
   and save them to `OUT/versions.json`;
4. `pytest -q`. If any test fails, the notebook stops here;
5. 10 smoke runs on the fake data, T = 10, `--lr 0.1 --C 1.0 --clipping-range 16.0` (each flag only where it
   applies and while config.yaml has null), into `OUT/smoke`: L1 none n5; L2 1 n5, 8 n5, 1 n20; L3 1 n5,
   8 n5, none n5, 1 n10, 1 n20, none n20. They are resumable: runs already complete on Drive are skipped,
   except a run whose logged clipping_range differs from the current one, which is redone. A clipping_range cut
   fails its run, with a banner. (At 8.0, L3 eps 1 n5 failed: its noise share std σ/√5 ≈ 2.17 puts 8.0 only
   3.7 stds out.);
6. print the smoke summary and save it to `OUT/smoke/smoke_summary.csv`. It has one row per run: level, eps,
   n, bytes_per_house, round-1 time, median `round_time_s` over rounds 2..T, clipping cut count, and test MAE
   and RMSE (kWh);
7. the timing test (`train.timing_test`: 50 rounds, L3, n = 20, eps = 1) → `OUT/timing.json`. It prints the
   median, the projected total = median × T × 27, and whether a T cap is needed against `COMPUTE_BUDGET_H`
   (and if so, the largest T that fits). T is `calibration.T[20]` if set, otherwise `T_PROJECTION`. The cell
   prints which one it used. It uses the same clipping_range, 16.0. A 50-round `timing.json` already on Drive is reused if it was made
   with that clipping_range, unless `FORCE_TIMING = True`;
8. zip `OUT/smoke`, `timing.json` and `versions.json` into **`OUT/stage4_results.zip`**: send this back.
   If any smoke run failed, the notebook ends with an error after the zip is written.

Every cell is safe to re-run. After a disconnect, just *Run all* again.

**Where outputs land** (all on Drive, under `OUT`): `versions.json`; `smoke/logs/`, `smoke/recordings/`,
`smoke/smoke_summary.csv`; `timing.json` (its run's own logs and recordings go to `logs/` and `recordings/`);
`stage4_results.zip`.

Colab's torch is a different build from the pinned CPU one, so its version is recorded in every recording's
`env`.

## TS-Inverse

`python -m attacks.ts_inverse --recording PATH [PATH ...] --out DIR` attacks recordings with TS-Inverse
(Meijer et al. 2025), ported from [github.com/Capsar/ts-inverse](https://github.com/Capsar/ts-inverse). Settings are
in config.yaml, `attacks.ts_inverse`, and are fixed: there is no λ tuning step. Run it in Colab through
`notebooks/attack_ts_inverse.ipynb`, after the real `data.py` is in.

**Licence.** `attacks/ts_inverse.py` and `attacks/reproduce_ts_inverse.py` contain ported code and are licensed
under CC BY-SA 4.0, with attribution to Capsar/ts-inverse and Meijer et al. 2025 (see `LICENSE-THIRD-PARTY`). No
other file contains their code, and none of our own modules import these two.

**What it does.** The fake windows' update is computed with `common.local_update(..., create_graph=True)` at the
recorded weights, so the matched quantity is exactly the training gradient:

| Level | Server view | Matched against |
|---|---|---|
| 1 | `server_view[h]`, one row per house | `local_update(fake_h)`, 4 fake windows per house |
| 2 | `server_view[h]` = ĝ_h (clip + noise) | `clip(local_update(fake_h), C)` |
| 3 | `server_view` = mean of the n protected updates | `mean_h clip(local_update(fake_h), C)`, 4n fake windows |

At level 3 the attack optimises against the **mean**, which keeps the gradient term on the paper's single-gradient
scale. The diagnostics JSON holds an L1 check against `server_view · n` (the sum) for the reconstruction and for
the true windows. Loss: Σ|fake − observed| + λ_P·periodicity(48) + λ_T·trend + λ_Q^obs·bounds(inputs) +
λ_Q^tar·bounds(targets), with λ_P = λ_T = 0.5 (the paper's London best) and bounds weights 1 / 0.1 (their code). The
bounds come from the quantile inversion model, which is **on everywhere, including the grid**. It is trained per
recording on 4096 attacker-pool samples at the recorded weights: the shared aux set (see "Aux set" below), with the
level's noise. `--no-inversion-model` exists only so the tests run fast.

**Outputs and scoring.** `python -m attacks.ts_inverse --recording PATH ... --out attack_outputs` writes, per
recording, `attack_outputs/<recording name>__tsinverse.pt` = `{"rebuilt": tensor}` in the format of "Attack output
format" below (0–1, `[n, 4, 96]` at levels 1–2, `[4n, 96]` at level 3). Next to it, `<recording name>__tsinverse.json`
holds diagnostics only: initial and final gradient loss and steps per inversion, the seeds, the level-3
`server_view · n` check and the inversion-model log. It holds no scores. Resume parts go in `attack_outputs/_work/`.
The grid is scored **only** by `python -m eval.score_all --blind same_date` (Shifa's matching, sMAPE, blind guess
and tables; see "Evaluation").

**Seeds and resume.** Every random draw (fake-window init, restarts, aux samples, inversion-model init and shuffling)
is seeded from (`common.run_seed(n, nb)`, round, house, restart), so reruns give identical output. A recording whose
`__tsinverse.pt` exists and whose `__tsinverse.json` holds the same settings is skipped. After an interruption,
finished houses and the trained inversion model are reused.

**Batch-1 reproduction gate.** `python -m attacks.reproduce_ts_inverse --out DIR` replicates their London FCN
setting. The attacked state, read from their code, is the **untrained** FCN right after `set_seed(seed)` with PyTorch's
default init. Nothing draws from torch's global RNG in between, so `torch.manual_seed(seed)` + `common.Model()` gives
the same weights. It attacks the gradient of one training window (warm-up 0, 1 batch, `update_model` False), with
seeds 10, 43, 28, 80, 71. It reports input and target sMAPE (mean ± std over the seeds) next to Table II's
6.3e-06 / 2.5e-06, in three rows: `table2_config` (their Table II attack, which their results notebook takes from
`ts-inverse_batch1_without_target_reconst_12-6-2024`: no inversion model and no regularisers), `ts_inverse_b1` (our
configured TS-Inverse, 1 window) and `ours_level1_b4` (**our setting**: level 1, 4 windows). It refuses the FAKE
`data.py`. Its sMAPE is on the **paper scale (0–1), reproduction only**: a fraction on the 0–1 values, input and target
halves apart, as in Table II. It is not the grid's score. Its matching (`eval.metrics.match_windows`) and blind guess
(`eval.baseline.blind_guess`, `same_date`) come from `eval/`, and MAE/RMSE are in kWh.

### TS-Inverse: what is ported and what changed

Ported (`attacks/ts_inverse.py`): their attack loop (`AttackTSInverseWorker.attack_batch`): L1 gradient loss, Adam at
lr 0.01, sign of the input gradient, clamping to [0, 1] every 2nd step, `ReduceLROnPlateau(0.1, patience
steps // 10)` on the total loss, U(0, 1) init and steps = 500 × windows × 10. Also their periodicity (period 48) and
trend (detached least-squares line) regularisers, the quantile-bounds loss, `ImprovedGradToInputNN_Quantile` (two
branches of residual blocks 768 → 512, dropout 0.05, quantiles 0.1/0.3/0.7/0.9) and its training (AdamW 1e-3, pinball
loss, batches of 32, 75 + 1 epochs, `ReduceLROnPlateau` on the held-out loss).

Changes from their code:

1. **Model and data plumbing:** our `common.Model` (the same layers as their `FCN_Predictor`) and `common.local_update`
   replace their model and loss call. The attack reads our recordings instead of their in-process training. Their
   Flower path is unfinished (their README TODO), and none of it is used.
2. **Levels 2–3:** the fake update is clipped to C inside the match. At level 3 the fake view is the mean over 4n
   fake windows in n fake houses. Neither exists in their code, which attacks one unclipped gradient.
3. **Aux data:** attacker-pool windows (`attacks/aux_gradients.py`, 4096 samples, fixed groups of 4 windows from one
   house and 4 dates, n houses at level 3, with the level's clipping and noise). They used the attacked series' own test/val splits, single-window gradients regrouped at
   random into batches every epoch. The held-out set for the plateau scheduler is the last 10 % of the aux samples
   (theirs was a separate split).
4. **Quantile-bounds pairing:** they pair the bounds the other way round (upper = q_j). With ordered quantile
   predictions that differs by a constant only, so the gradients match. We use the paper's form (Eq. 11).
5. **λ values:** fixed at λ_P = λ_T = 0.5 and bounds 1 / 0.1. Their code sweeps λ_P, λ_T ∈ {2, 1, 0.5}.
6. **Restarts:** 1, as in their code. If set above 1, the restart with the lowest gradient loss is kept (their code
   has no restarts).
7. **Seeding:** our seeded generators per draw instead of their global `set_seed`. Their dropout-mask options (TCN
   only) and one-shot targets (batch 1 only) are not ported.
8. **Scoring:** done by `eval.score_all` (Hungarian matching on sMAPE; they used a greedy match with an identity
   fallback), with the `same_date` attacker-pool blind guess. The paper-scale sMAPE is kept only in the reproduction.
9. **Reproduction data:** our London `data.py` (2013, standard tariff, min-max fitted on the attacker pool), one
   house chosen by `--house` (default: the first house of neighbourhood n = 5, k = 0), and the window drawn by a
   generator seeded with the seed. They used the first of 30 consumers in the Monash London file, min-max per series
   and split, and a shuffled loader draw.

## Aux set (shared by both attacks)

`attacks/aux_gradients.py` builds, per recording, 4096 attacker-pool samples (update, windows) at the recording's
weights, with the public protection settings (level, C, sigma, n). The attacker pool is disjoint from every
neighbourhood. With g = `common.local_update` of 4 windows of one attacker house on 4 random train dates:

| Level | Sample (what the server would see) | Noise std per coordinate |
|---|---|---|
| 1 | g | 0 |
| 2 | `common.protect(g, 2, C, σ, n)` = clip(g, C) + N(0, σ²) | σ |
| 3 | mean over n attacker houses of `common.protect(g_h, 3, C, σ, n)` = mean_h [clip(g_h, C) + N(0, σ²/n)] | **σ / n** |

Level 2 in training is Flower's `LocalDpMod` in weight space, which clips −lr·g to lr·C and adds
N(0, (lr·C·√(2 ln(1.25/δ))/ε)²). The server's g_hat = (w − returned)/lr is therefore clip(g, C) + N(0, σ²) with σ =
`sigma_for(eps, C, δ)`, the recording's sigma as long as `dp.sigma` in config.yaml stays null. Level 3: each house's
share is N(0, σ²/n), the n shares sum to N(0, σ²), and the server's view is their **mean**, so its noise is σ/n. SecAgg+'s
stochastic rounding (about 4e-6) is not modelled. The window draw and the noise use separate seeded streams, so
changing σ changes only the noise. The dates are drawn from those valid for every house in the sample. It reads only
through `data.py`, on the train split.

The set is seeded from `aux_seed(run_seed(n, nb), round)` and cached on disk (`--aux-cache DIR`, default
`OUT/_aux_cache`; in Colab `MyDrive/smart-meter-fl/attacks/aux_cache`). The key is the recording name, level, C, σ,
n, aux size, seed, windows per house, lr and the aux version, so TS-Inverse and the learned inverter on one recording
share one set, and changing σ (or anything else in the key) makes a new one. A set is about 0.17–0.3 GB (4096 ×
10,416 float32 gradients plus the windows), about 34 GB for the 162-recording grid, so the Colab notebook runs both
attacks per recording and deletes that recording's set once both have succeeded.

## Learned inverter

`python -m attacks.learned_inverter --recording PATH ... --out attack_outputs --aux-cache DIR` (our own code). A
network is **retrained for every recording** at its weights on its aux set. Its input is the update [P]: one house's
row at levels 1–2, `server_view` (the mean) at level 3. It outputs 4 windows at levels 1–2 and 4n at level 3, each
[96] in 0–1, in any order.

- **Prior:** each output slot starts from the attacker pool's same-date mean window for its date
  (`eval.baseline.blind_guess(dates, "same_date")`; at level 3 the 4 date priors repeated for the n houses). The
  network predicts a correction on top: out = clamp(prior + (2·sigmoid(z) − 1), 0, 1), with the head initialised to
  zero. An **untrained inverter outputs exactly the blind guess**, so anything better is learned from the update.
- **Network:** a residual MLP trunk (768, 512; Linear → BatchNorm → ReLU → dropout 0.05 → Linear → BatchNorm plus a
  projected skip), the same architecture as TS-Inverse's inversion model, written independently. Then a linear head
  to K × 96.
- **Loss:** MSE after Hungarian matching of the outputs to the true windows inside each sample (the update does not
  say which window is which; the matching is computed without gradient).
- **Training:** AdamW 1e-3, batches of 32, 75 epochs, `ReduceLROnPlateau` (0.1, patience 7) on the held-out loss: the
  same settings as TS-Inverse's inversion model. The last 10 % of the aux set is held out to pick the best epoch;
  epoch 0 (the blind guess) is a candidate too. No other tuning. Settings: config.yaml `attacks.learned_inverter`.
- **Seeds:** every draw (model init and dropout, shuffling) from (`run_seed(n, nb)`, round, house index −1 = the
  recording, purpose); the aux set from `aux_seed`, the same as TS-Inverse's.
- **Outputs:** `attack_outputs/<recording name>__inverter.pt` = `{"rebuilt": tensor}` ("Attack output format"), and
  `<recording name>__inverter.json` with diagnostics only: train and held-out loss per epoch, best epoch, seeds and
  the aux set used. No scores. Resumable: skipped if both exist with the same settings and aux key, redone otherwise.
  Scored with TS-Inverse in one table by `python -m eval.score_all --blind same_date`.

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
- Levels 1–2: `inner = flwr.server.workflow.default_workflows.default_fit_workflow`. Level 3:
  `inner = SecAggPlusWorkflow(...)`.

**Client** (`train/client.py`): `ClientApp(client_fn=..., mods=[])`. `client_fn` maps
`node_config["partition-id"]` i to `house_ids[i]` and returns a `NumPyClient`. Its `fit` returns
`(arrays, weight, {"house_id": ...})`. Every house reports the same weight (`simulation.client_weight` = 1000,
see level 3). Just before returning, the client seeds numpy's global generator from
`noise_seed(run_seed, round, partition id)`. LocalDpMod's noise (level 2) and SecAgg+'s stochastic rounding
(level 3) both draw from it, so reruns give identical files.

**Level 2: Flower's built-in local DP (`LocalDpMod`).** `mods=[LocalDpMod(clipping_norm=lr·C, sensitivity=lr·C,
epsilon=eps, delta=delta)]`. LocalDpMod clips "parameters returned minus parameters received" and then adds
N(0, std²) with std = sensitivity·√(2 ln(1.25/δ))/ε from numpy's global RNG (read from the flwr 1.45.0 source,
`flwr/supercore/differential_privacy.py: add_localdp_gaussian_noise_to_params`). So at level 2 the client
returns its **new weights** w − lr·g instead of the gradient, which makes "returned minus received" the true update
−lr·g. The server converts each reply back to gradient space, ĝ = (w − returned)/lr. Since the update is lr times
the gradient, clipping it to lr·C is clipping g to C, and the noise std lr·C·√(2 ln(1.25/δ))/ε becomes exactly
`sigma_for(eps, C, delta)` in gradient terms. `server_view [n, P]` holds the ĝ rows in `house_ids` order, and
the aggregate step is the same FedSGD step as level 1, `w -= lr · mean(ĝ)`.

**Level 3: Flower's SecAgg+ plus our `noise_share_mod`.**
- Server: `inner = SecAggPlusWorkflow(num_shares=5, reconstruction_threshold=3, clipping_range=...)`, with
  quantisation and modulus at Flower's defaults (2²² and 2³²). **`num_shares` = 5, not 4: flwr 1.45.0 bumps even
  share counts to odd, and 4 crashes the clients.** Its `setup_stage` turns an even `num_shares` (≠ n) into
  `num_shares + 1` for the neighbour ring *after* it has sent the even value to the clients, which then crash
  with an IndexError in `_share_keys`. 5 is what Flower itself would use for 4. `reconstruction_threshold`
  stays 3.
- Client: `mods=[secaggplus_mod, NoiseShareMod(C, sigma, n, run_seed, clipping_range)]`
  (`train/noise_share_mod.py`). The first mod is the outermost, so `noise_share_mod` sees the client's plaintext
  gradient. It clips it to C and adds the noise share N(0, (σ/√n)²) via `common.protect(update, 3, ...)`, seeded
  from (run_seed, round, partition id), before `secaggplus_mod` quantises and masks it. eps = none means clip only.
  It also counts the values with |x| > clipping_range and reports the count in the fit metrics, which SecAgg+
  forwards unmasked, so the server can log it and fail the run.
- SecAgg+ multiplies each update by `num_examples / max_weight` before clipping it to ±clipping_range and quantising
  it. `client_weight` is 1000 = Flower's default `max_weight`, so the factor is exactly 1: the quantisation step
  is 2·clipping_range/2²² (3.8e-6 for 8.0) and SecAgg+ clips at exactly clipping_range. (With weight 1 the step
  would be about 1000× coarser.)
- `FedSGD.aggregate_fit` gets results whose parameters all hold the same unmasked, de-quantised vector. Flower
  divides it by the total weight, so with equal weights it is the **mean** of the n protected updates, not their
  sum (read from the 1.45.0 source and confirmed by `test_level3_view_times_n_is_sum_of_clipped_updates`: with
  n = 5, eps = none, clipping_range 8.0, the max per-coordinate |server_view·n − sum of clipped true updates| was
  1.39e-05, against a quantisation step of 2·8.0/2²² = 3.81e-06 and the test's bound n·step = 1.91e-05). That
  vector is the `[P]` `server_view`, recorded with `view_is_sum = False`. **Attacks multiply it by n** to get the
  sum. The aggregate step is `w -= lr · server_view`.
- `MeteredGrid` counts the bytes of all four SecAgg+ stages.

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
| `env` | dict: `python`, `flwr`, `torch` (str), `cpu_model` (str), `cpu_cores` (int), `gpu_visible` (bool) |

`view_is_sum` only matters at level 3, where `server_view` is the single `[P]` aggregate instead of one row per
house. It is always False at levels 1–2. At level 3 it is also False, because Flower's SecAgg+ aggregate is the
**mean** of the n updates: attacks multiply `server_view` by n to get the sum. At level 2 `server_view` holds
the gradient-space rows ĝ = (w − returned)/lr (see "Flower wiring"). At level 3 with eps = none, `sigma` is 0.0.
`env` was added in Stage 4 by agreement. At level 1, `C` and `sigma` are None (no clipping, no noise).
`attacks/recording.py: load_recording(path)` checks every key and shape. It loads with `weights_only=True`, so a
file from the shared Drive cannot run code.

### data.py interface

- `get_neighbourhood(n, k)` -> n house IDs (deterministic in n, k)
- `valid_dates(house_ids, split)` -> list of `"YYYY-MM-DD"`
- `get_windows(house_ids, dates, split)` -> tensor `[houses, dates, 96]` in 0–1
- `attacker_houses()` -> 500 IDs
- `unscale(x)`

`data.py` only routes: with `SMART_METER_DATA` unset or `real` it loads `data_real.py`, with
`SMART_METER_DATA=fake` it loads `data_fake.py`. `tests/conftest.py` sets `fake`, so pytest never needs the
real data. `data.FAKE` says which one is loaded. The real loader reads the cleaned file
`data.processed_file` (default `data/processed/clean_2013.npz`, gitignored); the env var
`SMART_METER_PROCESSED` points it somewhere else, e.g. the copy on the shared Drive in Colab. If neither
exists it stops with an error rather than falling back to fake data. In the real data, for one n the
neighbourhoods k = 0, 1, 2 never share a house.

### Attack output format

One `torch.save({"rebuilt": tensor})` per recording per attack, scaled 0–1 like `true_windows`, at

```
attack_outputs/L{level}_eps{eps}_n{n}_nb{k}_r{round}__{attack}.pt     attack = inverter | tsinverse
```

(the recording's own name plus `__inverter` or `__tsinverse`). Shape `[n, 4, 96]` at levels 1–2 (row i =
house i's 4 windows, in any order) and `[4n, 96]` at level 3 (every window rebuilt from the aggregate, in any
order). Scoring pairs them with the true windows itself (Hungarian matching). Loaded with
`weights_only=True`.

### Seed convention

Houses depend only on (n, k). run_seed sets the model init, per-round dates and noise. The same run_seed is used
at every level.

## Evaluation

```bash
python data_real.py prepare        # once: data/raw/ -> data/processed/clean_2013.npz, then a summary
python -m eval.score_all           # recordings/ + attack_outputs/ -> results/summary.md, summary.csv, per_house.csv, per_day.csv
python -m eval.figures             # results/fig_attack_vs_houses_r{round}.png
python -m eval.exposure            # results/exposure.csv, fig_exposure.png (level 3)
python -m eval.forecast_table      # logs/*_test.json -> results/forecast.md, forecast.csv
```

`score_all` takes `--recordings`, `--attacks`, `--out` and `--blind weekday|same_date`.

**Definitions** (`eval/metrics.py`, `eval/baseline.py`):

- **Blind guess:** for each of the window's 2 days, the attacker pool's mean day for that weekday over its
  train days (`--blind same_date`: the pool's mean on that exact date, a stronger baseline).
- **sMAPE** in kWh, 0 to 200 %. Readings where both values are 0 count as a match.
- **Matching:** one-to-one, minimising total sMAPE (Hungarian), within a house's 4 windows at levels 1–2 and
  over all 4n windows at level 3. Caveat: because the matching sees the truth, an attacker that outputs many
  varied windows gets a small selection advantage (a noisy blind guess scores slightly below blind on sMAPE in
  `tests/test_eval.py`).
- **Routine timing:** morning rise = the largest jump between 04:00 and 12:00; evening peak = the highest
  reading between 16:00 and midnight; both on a 3-reading moving average. Error in hours, mean of the 2 days.
- **Home or away:** a day is "away" if mean use 09:00–17:00 is below 1.3 × the median 01:00–05:00 use.
  Accuracy per house, plus pooled balanced accuracy (away days are rarer).
- **Protection holds:** per score, the attack's improvement over the blind guess, paired per house over all
  houses and neighbourhoods, is at most 2 standard errors. `protection_holds` = this holds for sMAPE, rise
  error, peak error and home/away.
- **Per-house exposure (level 3):** Spearman correlation of attack success (blind sMAPE − attack sMAPE) with
  the house's share of the neighbourhood's use and with its RMSE distance from the blind guess.
- **Forecast cost:** test MAE and RMSE in kWh from each run's `_test.json`, mean ± SE over k.

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
