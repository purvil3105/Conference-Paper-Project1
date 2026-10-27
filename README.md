# Deadline-Aware Joint Random-Access Control for Dense NB-IoT Smart-Meter Networks

A runnable Python implementation of a **mass-conserving discrete-event NB-IoT
RACH simulator** with a **tabular Q-learning** controller that jointly selects an
**admission probability**, a **backoff setting**, and an **NPRACH configuration**,
compared against a **one-registry** suite of fixed / adaptive / single- and
dual-/full-joint-control baselines.

> **Scope.** This is an *NB-IoT-inspired, system-level* model evaluated under the
> assumptions in [`docs/model_assumptions.md`](docs/model_assumptions.md). It is
> **not** standards-compliant. No energy model, no deep-RL baseline, and no
> SLA/real-time/lightweight claim is asserted. Results from `run_pilot` with the
> default budget are a **pilot**, not the full research experiment. See the scope
> section below and the assumptions doc before quoting any number.

## What it does (and what it does not)

- Tracks **every request** through WAITING/BARRED → BACKOFF → ACCESS-IN-PROGRESS →
  COMPLETED/DROPPED, enforcing `generated = completed + dropped + pending` after
  every event batch.
- Separates **simulator truth** (ledger) from **controller observations**
  (receiver-side estimates) from **reward/evaluation** records.
- Distinguishes **preamble success** ≠ **random-access completion** ≠
  application delivery (the last is Level C and is **not** implemented).
- Enforces **resource feasibility** (a 3.75 kHz × 1 ms calendar; R_RACH + R_DATA ≤
  R_total) and **delayed control actions** (effective at the next NPRACH boundary
  after a modelled signalling delay).
- Trains tabular Q-learning with **feasible-action masks**, the **actual next
  observation**, and **frozen-policy** (ε=0) evaluation on **held-out** traces.
- Produces, from recorded logs only, the load (device-count) figures the project
  asks for: on-time service, preamble (access) success, collision, delay
  percentiles, drops, and resource use.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional editable install (provides the nbiot-* console commands):
pip install -e .
```
Developed and tested on Python 3.11.9 with the pinned versions in
`requirements.txt` (numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, pytest 8.3.4).

## Quick start

```bash
# 1. fast validation gates (human-readable)
python -m nbiot_rach.run_smoke_tests

# 2. full automated test suite
python -m pytest

# 3. the small reproducible end-to-end PILOT (smoke -> train -> eval -> figures)
python -m nbiot_rach.run_pilot            # ~10-15 min on a laptop

# scale up toward a real run:
python -m nbiot_rach.run_pilot --episodes 2000 --policy-seeds 0 1 2 3 4 --n-test 20
```

Individual stages:

```bash
python -m nbiot_rach.run_training  --episodes 150 --train-pop 30000 --policy-seeds 0 1
python -m nbiot_rach.run_evaluation --populations 10000 30000 50000 100000
python -m nbiot_rach.make_paper_outputs
```

## Outputs (written under `results/`)

```
results/
  checkpoints/   B{2..5}_s{seed}.npz/.json   frozen Q-tables + history + provenance
  logs/          training_phase.json, b0_tuning.json
  tables/        main_comparison.csv, main_comparison_per_run.csv,
                 load_sweep.csv, controller_cost.csv, metric_definitions.csv,
                 algorithm1.txt
  figures/       fig_on_time_vs_load, fig_preamble_success_vs_load,
                 fig_collision_vs_load, fig_p99_delay_vs_load,
                 fig_delay_percentiles_B5, fig_drops_vs_load,
                 fig_resource_vs_load, fig_on_time_event_vs_load,
                 fig_main_comparison_bar, fig_training_curves   (.png + .pdf)
  manifest.json  settings, git revision, library versions, pilot note
```

## Repository layout

```
nbiot_rach/
  config.py            scenarios, protocol, resource, reward, learning defaults
  rng.py               independent reproducible RNG streams
  traffic.py           periodic + event-triggered trace generation (Beta(3,4))
  actions.py           joint action registry + resource feasibility masks
  ledger.py            request ledger + mass-conservation invariants
  resource.py          3.75 kHz x 1 ms reservation calendar
  observation.py       controller observations + 150-bin state encoder
  reward.py            deadline-aware reward (+ ablation modes)
  environment.py       10-step discrete-event environment (step/reset)
  agents/              fixed, adaptive (occupancy-match), tabular Q-learning
  baseline_registry.py one source of truth for B0..B6
  metrics.py           cohort metrics, delays, resources, uncertainty
  training.py          train / validate / freeze / evaluate / cost
  experiment.py        high-level pipeline
  plotting.py          figures from logs
  run_*.py             CLI entry points
tests/                 conservation, deadlines, contention, backoff, action
                       delay, reward, Q-transition, resource, multimeter, repro
docs/                  model_assumptions.md, methodology_to_code.csv,
                       reviewer_resolution.csv
```

## Baselines (one registry — `nbiot_rach/baseline_registry.py`)

| ID | Method | Learns | Controls |
|----|--------|:---:|----------|
| B0 | Tuned fixed configuration | no | fixed tuple (validation-selected) |
| B1 | Load-aware occupancy-matching (adapted heuristic) | no | admission |
| B2 | Tabular Q, admission only | yes | p_A |
| B3 | Tabular Q, admission + backoff | yes | p_A, backoff |
| B4 | Tabular Q, admission + NPRACH | yes | p_A, NPRACH |
| B5 | **Full joint** tabular Q | yes | p_A, backoff, NPRACH |
| B6 | DQN | — | **not implemented** (no proxy curve permitted) |

## Reviewer concerns → implementation

Each DELCON reviewer concern is mapped to its code/test resolution and its
remaining manuscript task in
[`docs/reviewer_resolution.csv`](docs/reviewer_resolution.csv). The methodology →
code → test crosswalk is in
[`docs/methodology_to_code.csv`](docs/methodology_to_code.csv).

## Honest-reporting notes

- Every plotted value traces to a recorded run through the baseline registry;
  there are **no synthetic/proxy curves, no hard-coded performance, and no
  fabricated variability**.
- A well-tuned fixed admission can be strong on the on-time metric under generous
  deadlines; the learned controller's value and **limits** appear in the
  throughput–delay trade-off and heavier/burstier regimes. The pilot does not
  establish convergence or a deployment claim.
- Zero observed misses in a scenario is a sample outcome, not a guarantee; a zero
  denominator is reported as `NaN` ("not applicable"), never `0`.
