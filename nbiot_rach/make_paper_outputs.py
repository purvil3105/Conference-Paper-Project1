"""CLI: build figures and supporting tables from recorded result logs.

Generates the load-sweep figures (on-time service, preamble success, collision,
delay percentiles, drops, resource use), the reference-load comparison bar chart,
training curves, a metric-definitions table and an Algorithm-1 text listing.
All values trace back to the CSV logs produced by ``run_evaluation``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import generate_all
from .persistence import write_csv


METRIC_DEFINITIONS = [
    {"metric": "p_on_time", "definition": "On-time completions / deadline-evaluable cohort E(H). Primary metric.", "denominator": "|E(H)|"},
    {"metric": "p_miss", "definition": "1 - p_on_time (same information, not an independent gain).", "denominator": "|E(H)|"},
    {"metric": "preamble_success_prob", "definition": "Singleton preamble successes / transmitted preambles (attempt-conditioned).", "denominator": "transmitted preamble attempts"},
    {"metric": "collision_prob_devices", "definition": "Devices on collided choices / transmitted preambles.", "denominator": "transmitted preamble attempts"},
    {"metric": "collision_prob_choices", "definition": "Collided choices / occupied choices (choice-level).", "denominator": "occupied choices"},
    {"metric": "delay_median_s / p95 / p99", "definition": "Arrival-to-completion delay among COMPLETED requests (survivorship-biased; always report with service outcomes).", "denominator": "completed requests"},
    {"metric": "dropped_total", "definition": "Terminally dropped requests (retry-limit / overflow / expiry).", "denominator": "-"},
    {"metric": "resource_utilisation", "definition": "(NPRACH reserved + Msg3 data used) / R_total over all windows.", "denominator": "sum of per-window R_total"},
    {"metric": "unresolved", "definition": "Arrived, non-terminal, deadline not yet matured at H (reported separately, never counted as success).", "denominator": "-"},
]


def main():
    ap = argparse.ArgumentParser(description="Build NB-IoT RACH paper figures/tables")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    tables = out / "tables"
    figures = out / "figures"

    write_csv(tables / "metric_definitions.csv", METRIC_DEFINITIONS,
              columns=["metric", "definition", "denominator"])

    made = generate_all(tables, figures, logs_dir=out / "logs")
    print(f"generated {len(made)} figures in {figures}:")
    for m in made:
        print("  -", Path(m).name)
    # Algorithm-1 listing (code-to-method crosswalk aid)
    _write_algorithm1(out / "tables" / "algorithm1.txt")
    print("metric definitions ->", tables / "metric_definitions.csv")


def _write_algorithm1(path):
    text = """Algorithm 1. Training and frozen-policy evaluation (as implemented)

Input: traffic model, request-state ledger, feasible action registry A(s),
       state encoder, reward weights, alpha schedule, gamma, epsilon schedule
Initialise Q(s,a)=0 for all defined states and feasible actions   [agents/qlearning.py]
For each training episode:                                          [training.train_qlearning]
    Reset simulator, ledger, timers, counters, logs                 [environment.reset]
    While the episode has not terminated:
        (1) create arrivals (implicit: generated when t passes arrival)   [environment._eligible_requests]
        (2) process matured deadlines                               [environment._process_deadlines]
        (3) update retry timers; release eligible BACKOFF requests  [environment._run_opportunity]
        (4) observe permitted measurements; encode state s          [observation.ObservationModel]
        (5) select feasible action a (epsilon-greedy)               [agents/qlearning.select_action]
        (6) schedule a; apply at next NPRACH boundary after delay   [environment.step / _promote_pending_if_due]
        (7) simulate preamble contention + RAR + Msg3 + CR          [environment._contend]
        (8) record completions, drops, deadline events, resources   [ledger + resource.ResourceCalendar]
            Assert A(t)=C(t)+X(t)+B(t); abort run if violated        [ledger.assert_invariants]
        (9) compute reward from realised events                     [reward.compute_reward]
       (10) observe the ACTUAL next state s'; Q-update              [environment.step -> agents/qlearning.update]
            target = reward (terminal) else reward + gamma*max_{a' in A(s')} Q(s',a')
Freeze policy; set epsilon = 0                                      [agents/qlearning.freeze]
Evaluate on independent held-out traces; do not update Q            [training.evaluate_policy]
Report request-level service, resource use, uncertainty, cost       [metrics.episode_metrics]
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
