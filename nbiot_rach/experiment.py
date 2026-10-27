"""High-level experiment pipeline (train -> validate -> freeze -> test -> sweep).

Everything a script needs, so the scripts stay thin. All outputs are generated
from recorded runs through the single baseline registry (no proxy/synthetic
curves, no hard-coded performance numbers).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .config import (Config, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS,
                     population_sweep_configs)
from .actions import ActionRegistry
from .agents import FixedPolicy
from .baseline_registry import baseline_specs, make_policy
from .training import (train_qlearning, tune_fixed_action, evaluate_policy,
                       build_traces, measure_controller_cost)
from .metrics import aggregate, bootstrap_ci
from .persistence import (save_checkpoint, load_checkpoint, write_json, write_csv,
                          runtime_metadata, read_json)


@dataclass
class ExperimentSettings:
    out_dir: str = "results"
    train_pop: int = 30_000
    episodes: int = 150
    policy_seeds: tuple = (0, 1)
    baselines: tuple = ("B0", "B1", "B2", "B3", "B4", "B5")
    populations: tuple = (10_000, 20_000, 30_000, 50_000, 75_000, 100_000)
    val_seeds: tuple = VAL_SEEDS[:5]
    test_seeds: tuple = TEST_SEEDS[:5]
    val_every: int = 50
    label: str = "pilot"

    @property
    def base_cfg(self) -> Config:
        return Config(name=f"train_pop_{self.train_pop}", n_meters=self.train_pop)

    def paths(self):
        out = Path(self.out_dir)
        return {
            "root": out, "ckpt": out / "checkpoints", "tables": out / "tables",
            "logs": out / "logs", "figures": out / "figures",
        }


# ---------------------------------------------------------------- training
def run_training_phase(st: ExperimentSettings, verbose: bool = True) -> dict:
    cfg = st.base_cfg
    specs = baseline_specs()
    p = st.paths()
    trained = {}
    t0 = time.time()
    for bid in st.baselines:
        spec = specs[bid]
        if not spec.is_learning:
            continue
        for ps in st.policy_seeds:
            if verbose:
                print(f"[train] {bid} ({spec.name}) policy_seed={ps} "
                      f"episodes={st.episodes} pop={st.train_pop}")
            agent, mask, hist = train_qlearning(
                cfg, spec, episodes=st.episodes, policy_seed=ps,
                val_seeds=st.val_seeds, val_every=st.val_every, verbose=verbose)
            ck = p["ckpt"] / f"{bid}_s{ps}.npz"
            save_checkpoint(ck, agent, cfg, bid, ps, hist, mask)
            trained.setdefault(bid, []).append(str(ck))
            if verbose:
                print(f"       best val_p_on={hist.best_val:.4f} "
                      f"coverage={hist.coverage_state_action}/"
                      f"{hist.n_states*hist.n_actions_feasible} "
                      f"train_time={hist.wall_clock_s:.1f}s")
    dur = time.time() - t0
    write_json(p["logs"] / "training_phase.json",
               {"trained": trained, "settings": asdict(st),
                "wall_clock_s": dur, "runtime": runtime_metadata()})
    return trained


# -------------------------------------------------------- policy construction
def _b0_action(st: ExperimentSettings) -> int:
    cfg = st.base_cfg
    reg = ActionRegistry(cfg)
    val_traces = build_traces(cfg, st.val_seeds)
    aid, score = tune_fixed_action(cfg, reg, val_traces)
    write_json(st.paths()["logs"] / "b0_tuning.json",
               {"action_id": aid, "val_p_on": score,
                "pA": float(reg.pA[aid]), "backoff_ms": int(reg.backoff_ms[aid]),
                "nprach_config_id": int(reg.nprach_config_id[aid])})
    return aid


def build_eval_policies(st: ExperimentSettings):
    """Return {bid: [(policy, mask, policy_seed), ...]} for evaluation."""
    cfg = st.base_cfg
    specs = baseline_specs()
    reg = ActionRegistry(cfg)
    policies: dict[str, list] = {}
    b0_aid = None
    for bid in st.baselines:
        spec = specs[bid]
        if bid == "B0":
            b0_aid = _b0_action(st)
            pol, mask = make_policy(spec, cfg, reg, None, fixed_action_id=b0_aid)
            policies[bid] = [(pol, mask, 0)]
        elif bid == "B1":
            from .rng import RngBundle
            pol, mask = make_policy(spec, cfg, reg, RngBundle.create(0, 0))
            policies[bid] = [(pol, mask, 0)]
        elif spec.is_learning and spec.implemented:
            runs = []
            for ps in st.policy_seeds:
                ck = st.paths()["ckpt"] / f"{bid}_s{ps}.npz"
                if ck.exists():
                    agent, mask = load_checkpoint(ck, cfg)
                    runs.append((agent, mask, ps))
            if runs:
                policies[bid] = runs
    return policies, b0_aid


# ------------------------------------------------------------- reference eval
def evaluate_reference(st: ExperimentSettings, verbose: bool = True) -> dict:
    cfg = st.base_cfg
    specs = baseline_specs()
    policies, b0_aid = build_eval_policies(st)
    test_traces = build_traces(cfg, st.test_seeds)
    p = st.paths()

    per_run_rows = []          # one row per (baseline, policy_seed, test_trace)
    summary_rows = []          # one aggregated row per baseline
    cost_rows = []
    for bid, runs in policies.items():
        pooled = []
        for (pol, mask, ps) in runs:
            res = evaluate_policy(cfg, pol, mask, test_traces, policy_seed=ps)
            for r in res:
                r2 = dict(r); r2["baseline"] = bid; r2["name"] = specs[bid].name
                r2["policy_seed"] = ps
                per_run_rows.append(r2)
                pooled.append(r)
            if specs[bid].is_learning:
                cost = measure_controller_cost(pol)
                cost["baseline"] = bid; cost["policy_seed"] = ps
                cost_rows.append(cost)
        agg = aggregate(pooled)
        row = {"baseline": bid, "name": specs[bid].name,
               "n_runs": len(pooled)}
        for k in ("p_on_time", "p_miss", "p_on_time_periodic", "p_on_time_event",
                  "preamble_success_prob", "collision_prob_devices",
                  "delay_median_s", "delay_p95_s", "delay_p99_s",
                  "dropped_total", "late_completions", "resource_utilisation",
                  "cohort_size"):
            row[f"{k}_mean"] = agg.get(k, {}).get("mean", float("nan"))
            row[f"{k}_std"] = agg.get(k, {}).get("std", float("nan"))
        # bootstrap CI on the primary metric
        lo, hi = bootstrap_ci([r["p_on_time"] for r in pooled])
        row["p_on_time_ci_low"], row["p_on_time_ci_high"] = lo, hi
        summary_rows.append(row)
        if verbose:
            print(f"[eval ref] {bid} {specs[bid].name}: "
                  f"p_on={row['p_on_time_mean']:.4f}±{row['p_on_time_std']:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  drops={row['dropped_total_mean']:.0f}")

    write_csv(p["tables"] / "main_comparison.csv", summary_rows)
    write_csv(p["tables"] / "main_comparison_per_run.csv", per_run_rows)
    if cost_rows:
        write_csv(p["tables"] / "controller_cost.csv", cost_rows)
    return {"summary": summary_rows, "per_run": per_run_rows, "cost": cost_rows,
            "b0_action_id": b0_aid}


# --------------------------------------------------------------- load sweep
def evaluate_load_sweep(st: ExperimentSettings, verbose: bool = True) -> list[dict]:
    """Evaluate each baseline across the population (device-count) axis.

    Learning policies are trained once (at ``train_pop``) and frozen; they are
    evaluated unchanged at every population to test load generalisation.
    """
    policies, _ = build_eval_policies(st)
    specs = baseline_specs()
    p = st.paths()
    rows = []
    METRICS = ["p_on_time", "p_on_time_periodic", "p_on_time_event",
               "preamble_success_prob", "collision_prob_devices",
               "delay_median_s", "delay_p95_s", "delay_p99_s", "dropped_total",
               "late_completions", "resource_utilisation", "cohort_size",
               "n_requests"]
    for n in st.populations:
        cfg_n = Config(name=f"pop_{n}", n_meters=n)
        test_traces = build_traces(cfg_n, st.test_seeds)
        for bid, runs in policies.items():
            pooled = []
            for (pol, mask, ps) in runs:
                res = evaluate_policy(cfg_n, pol, mask, test_traces, policy_seed=ps)
                pooled.extend(res)
            agg = aggregate(pooled, keys=METRICS)
            row = {"baseline": bid, "name": specs[bid].name, "n_meters": n,
                   "n_runs": len(pooled)}
            for k in METRICS:
                row[f"{k}_mean"] = agg.get(k, {}).get("mean", float("nan"))
                row[f"{k}_std"] = agg.get(k, {}).get("std", float("nan"))
            lo, hi = bootstrap_ci([r["p_on_time"] for r in pooled])
            row["p_on_time_ci_low"], row["p_on_time_ci_high"] = lo, hi
            rows.append(row)
            if verbose:
                print(f"[sweep] pop={n:6d} {bid:3s} {specs[bid].name:18s} "
                      f"p_on={row['p_on_time_mean']:.3f} "
                      f"preamble={row['preamble_success_prob_mean']:.3f} "
                      f"p99={row['delay_p99_s_mean']:.2f}s")
    write_csv(p["tables"] / "load_sweep.csv", rows)
    return rows
