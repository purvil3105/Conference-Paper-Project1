"""Parallel full-budget experiment runner (plan §14 Phases 5-8).

Every training run (one baseline x seed x reward-variant) is independent, so we
fan them out across processes with a ProcessPoolExecutor. Evaluation is then
parallelised per population. Results are identical to a serial run (seeds are
unchanged); only the wall-clock differs. The run is **resumable**: a training
task whose checkpoint already exists is skipped.

What it runs:
  * CORE full budget:   B2-B5 x N seeds x E episodes (default 2000) at train_pop
  * REWARD-MODE ablation:   B5 under throughput_only + deadline_aware_no_resource
  * REWARD-WEIGHT ablation: B5 one-factor grid around the default weights
  * EVALUATION:         reference + device-count load sweep on held-out test
                        traces, plus an ablation table at the reference load
  * FIGURES/TABLES via make_paper_outputs

NOT included (documented as remaining work): the rich/oracle STATE ablation
(needs additional encoder code), and held-out generalisation scenarios.

Windows-safe: workers are top-level functions; the pool is created under the
__main__ guard; each worker pins OMP_NUM_THREADS=1 so processes do not contend.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import Config, RewardWeights, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS

# metrics collected in the load sweep (same set as experiment.evaluate_load_sweep)
METRICS = ["p_on_time", "p_on_time_periodic", "p_on_time_event",
           "preamble_success_prob", "collision_prob_devices",
           "delay_median_s", "delay_p95_s", "delay_p99_s", "dropped_total",
           "late_completions", "resource_utilisation", "cohort_size",
           "n_requests"]


# --------------------------------------------------------------- workers
def _train_task(t: dict) -> dict:
    """Train one (baseline, seed, reward-variant) run and checkpoint it."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from .baseline_registry import baseline_specs
    from .training import train_qlearning
    from .persistence import save_checkpoint, read_json

    ckpt = t["ckpt"]
    if t.get("resume", True) and os.path.exists(ckpt):
        try:
            meta = read_json(ckpt.replace(".npz", ".json"))
            return {"ckpt": ckpt, "bid": t["bid"], "seed": t["seed"],
                    "variant": t["variant"], "skipped": True,
                    "best_val": meta["history"].get("best_val", float("nan")),
                    "train_time": meta["history"].get("wall_clock_s", 0.0)}
        except Exception:
            pass  # fall through and retrain if metadata is unreadable

    cfg = Config(name=t["cfg_name"], n_meters=t["train_pop"],
                 reward=RewardWeights(**t["reward"]))
    spec = baseline_specs()[t["bid"]]
    agent, mask, hist = train_qlearning(
        cfg, spec, episodes=t["episodes"], policy_seed=t["seed"],
        val_seeds=tuple(t["val_seeds"]), val_every=t["val_every"], verbose=False)
    save_checkpoint(ckpt, agent, cfg, t["bid"], t["seed"], hist, mask)
    return {"ckpt": ckpt, "bid": t["bid"], "seed": t["seed"],
            "variant": t["variant"], "skipped": False,
            "best_val": hist.best_val, "coverage": hist.coverage_state_action,
            "train_time": hist.wall_clock_s}


def _eval_pop_task(t: dict) -> list[dict]:
    """Evaluate every baseline at one population on the held-out test traces."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from .actions import ActionRegistry
    from .agents import FixedPolicy, AdaptivePolicy
    from .baseline_registry import (baseline_specs, PINNED_BACKOFF_MS,
                                     PINNED_NPRACH_ID)
    from .persistence import load_checkpoint
    from .training import build_traces, evaluate_policy
    from .metrics import aggregate, bootstrap_ci

    cfg = Config(name=f"pop_{t['population']}", n_meters=t["population"])
    reg = ActionRegistry(cfg)
    specs = baseline_specs()
    test_traces = build_traces(cfg, tuple(t["test_seeds"]))
    rows = []
    for bid in t["baselines"]:
        spec = specs[bid]
        pooled = []
        if bid == "B0":
            pool_mask = spec.mask_fn(reg)
            pooled += evaluate_policy(cfg, FixedPolicy(t["b0_aid"]), pool_mask,
                                      test_traces, policy_seed=0)
        elif bid == "B1":
            pool_mask = spec.mask_fn(reg)
            pol = AdaptivePolicy(reg, pool_mask, PINNED_BACKOFF_MS, PINNED_NPRACH_ID)
            pooled += evaluate_policy(cfg, pol, pool_mask, test_traces, policy_seed=0)
        else:
            for ps in t["policy_seeds"]:
                ck = os.path.join(t["ckpt_dir"], f"{bid}_s{ps}.npz")
                if not os.path.exists(ck):
                    continue
                agent, mask = load_checkpoint(ck, cfg)
                pooled += evaluate_policy(cfg, agent, mask, test_traces, policy_seed=ps)
        agg = aggregate(pooled, keys=METRICS)
        row = {"baseline": bid, "name": specs[bid].name,
               "n_meters": t["population"], "n_runs": len(pooled)}
        for k in METRICS:
            row[f"{k}_mean"] = agg.get(k, {}).get("mean", float("nan"))
            row[f"{k}_std"] = agg.get(k, {}).get("std", float("nan"))
        lo, hi = bootstrap_ci([r["p_on_time"] for r in pooled])
        row["p_on_time_ci_low"], row["p_on_time_ci_high"] = lo, hi
        rows.append(row)
    return rows


def _eval_ablation_task(t: dict) -> dict:
    """Evaluate one trained ablation checkpoint at the reference load."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from .persistence import load_checkpoint
    from .training import build_traces, evaluate_policy
    from .metrics import aggregate, bootstrap_ci

    cfg = Config(name=f"ablation_{t['variant']}", n_meters=t["train_pop"])
    traces = build_traces(cfg, tuple(t["test_seeds"]))
    agent, mask = load_checkpoint(t["ckpt"], cfg)
    res = evaluate_policy(cfg, agent, mask, traces, policy_seed=t["seed"])
    agg = aggregate(res, keys=["p_on_time", "p_on_time_event", "delay_p99_s",
                               "dropped_total", "preamble_success_prob",
                               "resource_utilisation"])
    lo, hi = bootstrap_ci([r["p_on_time"] for r in res])
    out = {"group": t["group"], "variant": t["variant"], "seed": t["seed"],
           "p_on_time_ci_low": lo, "p_on_time_ci_high": hi, "n_runs": len(res)}
    for k, v in agg.items():
        out[f"{k}_mean"] = v["mean"]
    return out


# --------------------------------------------------------------- task builders
def _reward_variants() -> list[tuple[str, str, dict]]:
    """(group, variant_name, reward_kwargs) for the reward ablations."""
    base = dict(w_on=1.0, w_miss=1.0, w_drop=1.0, w_q=0.01, w_r=0.01,
                K=384.0, mode="full")
    out = []
    # reward-mode ablation
    out.append(("mode", "throughput_only", {**base, "mode": "throughput_only"}))
    out.append(("mode", "deadline_aware_no_resource",
                {**base, "mode": "deadline_aware_no_resource"}))
    # one-factor reward-weight sensitivity around the default
    out.append(("weight", "w_miss2", {**base, "w_miss": 2.0}))
    out.append(("weight", "w_drop2", {**base, "w_drop": 2.0}))
    out.append(("weight", "w_q0", {**base, "w_q": 0.0}))
    out.append(("weight", "w_q0.05", {**base, "w_q": 0.05}))
    out.append(("weight", "w_r0", {**base, "w_r": 0.0}))
    out.append(("weight", "w_r0.05", {**base, "w_r": 0.05}))
    return out


# --------------------------------------------------------------- orchestration
def run(episodes, train_pop, policy_seeds, baselines, populations, n_test,
        n_val, val_every, max_workers, out_dir, ablation_seeds, resume):
    from .actions import ActionRegistry
    from .training import tune_fixed_action, build_traces, measure_controller_cost
    from .persistence import (write_csv, write_json, runtime_metadata,
                              load_checkpoint)

    t_start = time.time()
    out = Path(out_dir)
    ckpt_dir = out / "checkpoints"
    abl_dir = out / "checkpoints_ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    abl_dir.mkdir(parents=True, exist_ok=True)
    test_seeds = list(TEST_SEEDS[:n_test])
    val_seeds = list(VAL_SEEDS[:n_val])

    base_reward = dict(w_on=1.0, w_miss=1.0, w_drop=1.0, w_q=0.01, w_r=0.01,
                       K=384.0, mode="full")

    # ---- build training task list --------------------------------------
    learn = [b for b in baselines if b in ("B2", "B3", "B4", "B5")]
    tasks = []
    for bid in learn:
        for ps in policy_seeds:
            tasks.append(dict(bid=bid, seed=ps, variant="core",
                              cfg_name=f"train_pop_{train_pop}", train_pop=train_pop,
                              reward=dict(base_reward), episodes=episodes,
                              val_seeds=val_seeds, val_every=val_every,
                              ckpt=str(ckpt_dir / f"{bid}_s{ps}.npz"), resume=resume))
    # ablations: train B5 under each reward variant
    abl_variants = _reward_variants()
    for (group, variant, rw) in abl_variants:
        for ps in ablation_seeds:
            tasks.append(dict(bid="B5", seed=ps, variant=variant,
                              cfg_name=f"ablation_{variant}", train_pop=train_pop,
                              reward=rw, episodes=episodes, val_seeds=val_seeds,
                              val_every=val_every,
                              ckpt=str(abl_dir / f"{variant}_s{ps}.npz"),
                              resume=resume))

    workers = max(1, min(max_workers, (os.cpu_count() or 2) - 1))
    print(f"[parallel] {len(tasks)} training tasks on {workers} workers "
          f"(episodes={episodes}, train_pop={train_pop}, seeds={list(policy_seeds)})")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_train_task, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            tag = "skip" if r.get("skipped") else f"{r.get('train_time', 0):.0f}s"
            print(f"  [train {done}/{len(tasks)}] {r['variant']:28s} {r['bid']} "
                  f"s{r['seed']}  best_val={r.get('best_val', float('nan')):.4f}  ({tag})")
    print(f"[parallel] training done in {time.time()-t_start:.0f}s")

    # ---- tune B0 once, then parallel evaluation per population ----------
    cfg_ref = Config(name=f"train_pop_{train_pop}", n_meters=train_pop)
    reg = ActionRegistry(cfg_ref)
    b0_aid, b0_score = tune_fixed_action(cfg_ref, reg, build_traces(cfg_ref, val_seeds))
    write_json(out / "logs" / "b0_tuning.json",
               {"action_id": b0_aid, "val_p_on": b0_score,
                "pA": float(reg.pA[b0_aid]), "backoff_ms": int(reg.backoff_ms[b0_aid]),
                "nprach_config_id": int(reg.nprach_config_id[b0_aid])})

    eval_tasks = [dict(population=p, baselines=list(baselines),
                       policy_seeds=list(policy_seeds), test_seeds=test_seeds,
                       b0_aid=b0_aid, ckpt_dir=str(ckpt_dir)) for p in populations]
    sweep_rows = []
    t_eval = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_eval_pop_task, t): t for t in eval_tasks}
        for fut in as_completed(futs):
            rows = fut.result()
            sweep_rows.extend(rows)
            p = rows[0]["n_meters"]
            b5 = next((r for r in rows if r["baseline"] == "B5"), rows[0])
            print(f"  [eval pop={int(p)//1000}k] done  "
                  f"B5 p_on={b5['p_on_time_mean']:.3f}")
    sweep_rows.sort(key=lambda r: (r["n_meters"], r["baseline"]))
    write_csv(out / "tables" / "load_sweep.csv", sweep_rows)

    # main comparison = sweep rows at the reference (train) population
    main_rows = [r for r in sweep_rows if int(r["n_meters"]) == train_pop]
    write_csv(out / "tables" / "main_comparison.csv", main_rows)
    print(f"[parallel] evaluation done in {time.time()-t_eval:.0f}s")

    # ---- controller cost (one per learning baseline, seed 0) -----------
    cost_rows = []
    for bid in learn:
        ck = ckpt_dir / f"{bid}_s{policy_seeds[0]}.npz"
        if ck.exists():
            agent, _ = load_checkpoint(str(ck), cfg_ref)
            c = measure_controller_cost(agent)
            c["baseline"] = bid
            cost_rows.append(c)
    if cost_rows:
        write_csv(out / "tables" / "controller_cost.csv", cost_rows)

    # ---- reward ablation evaluation at reference load ------------------
    abl_tasks = []
    # include the core 'full' reward B5 runs as the reference point
    for ps in ablation_seeds:
        ck = ckpt_dir / f"B5_s{ps}.npz"
        if ck.exists():
            abl_tasks.append(dict(group="mode", variant="full", seed=ps,
                                  ckpt=str(ck), train_pop=train_pop,
                                  test_seeds=test_seeds))
    for (group, variant, rw) in abl_variants:
        for ps in ablation_seeds:
            ck = abl_dir / f"{variant}_s{ps}.npz"
            if ck.exists():
                abl_tasks.append(dict(group=group, variant=variant, seed=ps,
                                      ckpt=str(ck), train_pop=train_pop,
                                      test_seeds=test_seeds))
    abl_results = []
    if abl_tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_eval_ablation_task, t): t for t in abl_tasks}
            for fut in as_completed(futs):
                abl_results.append(fut.result())
    # aggregate ablation across seeds
    from .metrics import aggregate as agg_fn
    abl_summary = []
    variants_seen = []
    for r in abl_results:
        key = (r["group"], r["variant"])
        if key not in variants_seen:
            variants_seen.append(key)
    for (group, variant) in variants_seen:
        rs = [r for r in abl_results if r["group"] == group and r["variant"] == variant]
        a = agg_fn(rs, keys=["p_on_time_mean", "p_on_time_event_mean",
                             "delay_p99_s_mean", "dropped_total_mean",
                             "preamble_success_prob_mean", "resource_utilisation_mean"])
        row = {"group": group, "variant": variant, "n_seeds": len(rs)}
        for k, v in a.items():
            row[k.replace("_mean_mean", "_mean")] = v["mean"]
        abl_summary.append(row)
    if abl_summary:
        write_csv(out / "tables" / "ablation_reward.csv", abl_summary)

    # ---- figures + manifest -------------------------------------------
    from . import make_paper_outputs
    import sys
    sys.argv = ["make_paper_outputs", "--out-dir", out_dir]
    make_paper_outputs.main()

    total = time.time() - t_start
    write_json(out / "manifest.json", {
        "label": "full_budget_parallel",
        "wall_clock_s": total,
        "settings": dict(episodes=episodes, train_pop=train_pop,
                         policy_seeds=list(policy_seeds), baselines=list(baselines),
                         populations=list(populations), n_test=n_test, n_val=n_val,
                         val_every=val_every, workers=workers,
                         ablation_seeds=list(ablation_seeds)),
        "b0_action_id": b0_aid,
        "ablations": {"reward_mode": True, "reward_weight": True,
                      "state_rich_oracle": False, "generalisation_scenarios": False},
        "runtime": runtime_metadata(),
        "note": ("Full-budget parallel run. State (rich/oracle) ablation and "
                 "held-out generalisation scenarios are NOT included (remaining "
                 "work). See docs/model_assumptions.md."),
    })
    print(f"\n[parallel] FULL-BUDGET run complete in {total/60:.1f} min -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Parallel full-budget NB-IoT RACH run")
    ap.add_argument("--out-dir", default="results_full")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--train-pop", type=int, default=30000)
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ablation-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--baselines", nargs="+",
                    default=["B0", "B1", "B2", "B3", "B4", "B5"])
    ap.add_argument("--populations", type=int, nargs="+",
                    default=[10000, 20000, 30000, 50000, 75000, 100000])
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    run(episodes=args.episodes, train_pop=args.train_pop,
        policy_seeds=tuple(args.policy_seeds), baselines=tuple(args.baselines),
        populations=tuple(args.populations), n_test=args.n_test, n_val=args.n_val,
        val_every=args.val_every, max_workers=args.max_workers, out_dir=args.out_dir,
        ablation_seeds=tuple(args.ablation_seeds), resume=not args.no_resume)


if __name__ == "__main__":
    main()
