"""Training, validation, freezing and evaluation orchestration (plan §9, §11, §14).

Strict train / validation / test separation:
  * training traces  (seeds 1001..)  drive Q updates and tuning;
  * validation traces (seeds 2001..) select checkpoints / fixed controls;
  * test traces      (seeds 3001..)  are touched ONLY at final evaluation.

Policies are frozen (epsilon = 0, no Q updates) before test evaluation. Traffic
traces are shared across policies for paired comparison. Controller memory and
decision latency are measured separately from simulator overhead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS
from .actions import ActionRegistry
from .traffic import generate_trace, Trace
from .rng import RngBundle
from .environment import NBIoTRachEnv
from .agents import QLearningAgent, FixedPolicy, AdaptivePolicy
from .agents.base import Policy
from .baseline_registry import BaselineSpec, make_policy
from .metrics import episode_metrics, aggregate


# --- trace cache -----------------------------------------------------------
def build_traces(cfg: Config, seeds) -> dict[int, Trace]:
    return {int(s): generate_trace(cfg, int(s)) for s in seeds}


# --- evaluation of one (frozen) policy -------------------------------------
def evaluate_policy(cfg: Config, policy: Policy, action_mask: np.ndarray,
                    traces: dict[int, Trace], policy_seed: int = 0,
                    check_invariants: bool = True) -> list[dict]:
    """Run ``policy`` (frozen) on each trace. Returns one metric dict per trace."""
    results = []
    for seed, trace in traces.items():
        rng = RngBundle.create(trace_seed=seed, policy_seed=policy_seed)
        env = NBIoTRachEnv(cfg, trace, rng, action_mask=action_mask,
                           check_invariants=check_invariants)
        env.run_episode(policy, train=False)
        m = episode_metrics(env.ledger, env.log, cfg, h_us=env.end_us)
        m["trace_seed"] = seed
        results.append(m)
    return results


# --- fixed-config tuning (B0) ----------------------------------------------
def tune_fixed_action(cfg: Config, registry: ActionRegistry,
                      val_traces: dict[int, Trace],
                      candidates=None) -> tuple[int, float]:
    """Select the best fixed action on validation traces (never on test)."""
    if candidates is None:
        candidates = []
        for pa in (0.01, 0.025, 0.05, 0.10, 0.25, 0.50):
            for bo in (0, 256, 1024, 4096):
                candidates.append((pa, bo, 4))   # NPRACH pinned to 48 choices
    feas = registry.feasible_mask()
    best_aid, best_score = None, -np.inf
    for (pa, bo, nc) in candidates:
        aid = registry.index_of(pa, bo, nc)
        res = evaluate_policy(cfg, FixedPolicy(aid), feas, val_traces,
                              check_invariants=False)
        score = float(np.nanmean([r["p_on_time"] for r in res]))
        if score > best_score:
            best_score, best_aid = score, aid
    return int(best_aid), float(best_score)


# --- Q-learning training with validation checkpointing ---------------------
@dataclass
class TrainHistory:
    episodes: list = field(default_factory=list)
    train_reward: list = field(default_factory=list)
    val_p_on: list = field(default_factory=list)
    val_episode: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    best_val: float = -np.inf
    best_episode: int = -1
    coverage_state_action: int = 0
    n_states: int = 0
    n_actions_feasible: int = 0
    wall_clock_s: float = 0.0


def train_qlearning(cfg: Config, spec: BaselineSpec, episodes: int,
                    policy_seed: int = 0,
                    train_seeds=TRAIN_SEEDS, val_seeds=VAL_SEEDS,
                    val_every: int | None = None,
                    verbose: bool = False):
    """Train one tabular-Q baseline. Returns (frozen_agent, action_mask, history)."""
    registry = ActionRegistry(cfg)
    action_mask = spec.mask_fn(registry)
    feasible = np.nonzero(action_mask)[0]
    val_every = val_every or cfg.learning.val_every

    agent_rng = RngBundle.create(trace_seed=0, policy_seed=policy_seed)
    agent = QLearningAgent(
        n_states=(len(cfg.congestion_bins) - 1) * (len(cfg.load_bins) - 1)
        * (len(cfg.trend_bins) - 1),
        n_actions=registry.n_actions, feasible=feasible, learning=cfg.learning,
        exploration_rng=agent_rng["exploration"], tiebreak_rng=agent_rng["tiebreak"])

    train_traces = build_traces(cfg, train_seeds)
    val_traces = build_traces(cfg, val_seeds)
    train_seed_list = list(train_traces.keys())

    hist = TrainHistory(n_states=agent.n_states, n_actions_feasible=int(feasible.size))
    best_q = agent.q.copy()
    best_visits = agent.visits.copy()
    t0 = time.time()

    for ep in range(episodes):
        agent.set_epsilon_for_episode(ep, episodes)
        seed = train_seed_list[ep % len(train_seed_list)]
        rng = RngBundle.create(trace_seed=seed, policy_seed=policy_seed)
        env = NBIoTRachEnv(cfg, train_traces[seed], rng, action_mask=action_mask,
                           check_invariants=False)
        total_r, _ = env.run_episode(agent, train=True)
        hist.episodes.append(ep)
        hist.train_reward.append(float(total_r))
        hist.epsilon.append(float(agent.epsilon))

        if (ep + 1) % val_every == 0 or ep == episodes - 1:
            # greedy (frozen-style) validation without committing to freeze
            saved_eps, saved_frozen = agent.epsilon, agent.frozen
            agent.epsilon, agent.frozen = 0.0, True
            res = evaluate_policy(cfg, agent, action_mask, val_traces,
                                  policy_seed=policy_seed, check_invariants=False)
            agent.epsilon, agent.frozen = saved_eps, saved_frozen
            val_score = float(np.nanmean([r["p_on_time"] for r in res]))
            hist.val_episode.append(ep)
            hist.val_p_on.append(val_score)
            if val_score > hist.best_val:
                hist.best_val = val_score
                hist.best_episode = ep
                best_q = agent.q.copy()
                best_visits = agent.visits.copy()
            if verbose:
                print(f"  [{spec.bid}] ep {ep+1}/{episodes} eps={agent.epsilon:.3f} "
                      f"train_r={total_r:.3f} val_p_on={val_score:.4f} "
                      f"(best {hist.best_val:.4f}@{hist.best_episode+1})")

    # restore best-validation checkpoint and freeze
    agent.q = best_q
    agent.visits = best_visits
    agent.freeze()
    hist.coverage_state_action = int(np.count_nonzero(agent.visits))
    hist.wall_clock_s = time.time() - t0
    return agent, action_mask, hist


# --- controller cost (memory + decision latency), measured separately ------
def measure_controller_cost(agent: QLearningAgent, n_probe: int = 20000) -> dict:
    feasible = agent.feasible
    rng = np.random.default_rng(777)
    states = rng.integers(0, agent.n_states, size=n_probe)
    t0 = time.perf_counter()
    for s in states:
        agent._greedy(int(s), feasible)
    dt = time.perf_counter() - t0
    q_bytes = agent.q.nbytes
    amap = agent.frozen_action_map()
    return {
        "q_table_bytes": int(q_bytes),
        "q_table_kib": q_bytes / 1024.0,
        "frozen_action_map_bytes": int(amap.nbytes),
        "n_states": int(agent.n_states),
        "n_feasible_actions": int(feasible.size),
        "decision_latency_us_mean": 1e6 * dt / n_probe,
        "n_probe": n_probe,
    }
