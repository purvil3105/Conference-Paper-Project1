"""One baseline registry for everything (methodology §H, plan §10).

Every table, figure and log reads its series from THIS registry, so the baseline
set is identical across text, tables and figures (reviewer concern: inconsistent
baseline sets).

  B0  Tuned fixed configuration            (non-learning)
  B1  Load-aware adaptive (occupancy match)(non-learning, adapted heuristic)
  B2  ACB/admission-only tabular Q         (learning; backoff+NPRACH pinned)
  B3  Admission + backoff tabular Q        (learning; NPRACH pinned)
  B4  Admission + NPRACH tabular Q         (learning; backoff pinned)
  B5  Full joint tabular Q                 (learning; all three controls)
  B6  Actual DQN                           (NOT IMPLEMENTED -- no proxy curve)

B6 is intentionally absent from all performance outputs: the methodology forbids
synthetic/proxy deep-RL curves. If a DQN is ever added it must be genuinely
trained, tuned, checkpointed and frozen before it may enter a figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .actions import ActionRegistry
from .config import Config, LearningConfig
from .agents import FixedPolicy, AdaptivePolicy, QLearningAgent
from .rng import RngBundle

# Pinned (validation-tuned) controls for the ablation masks. Documented as
# study choices; override via tune routines if desired.
PINNED_BACKOFF_MS = 0
PINNED_NPRACH_ID = 4


@dataclass
class BaselineSpec:
    bid: str
    name: str
    is_learning: bool
    description: str
    mask_fn: Callable[[ActionRegistry], np.ndarray]
    implemented: bool = True


def baseline_specs() -> dict[str, BaselineSpec]:
    def full(reg: ActionRegistry) -> np.ndarray:
        return reg.feasible_mask()

    def acb_only(reg: ActionRegistry) -> np.ndarray:
        return reg.control_mask(vary_pa=True, vary_backoff=False, vary_nprach=False,
                                fixed_backoff_ms=PINNED_BACKOFF_MS,
                                fixed_nprach_id=PINNED_NPRACH_ID)

    def acb_backoff(reg: ActionRegistry) -> np.ndarray:
        return reg.control_mask(vary_pa=True, vary_backoff=True, vary_nprach=False,
                                fixed_nprach_id=PINNED_NPRACH_ID)

    def acb_nprach(reg: ActionRegistry) -> np.ndarray:
        return reg.control_mask(vary_pa=True, vary_backoff=False, vary_nprach=True,
                                fixed_backoff_ms=PINNED_BACKOFF_MS)

    def joint(reg: ActionRegistry) -> np.ndarray:
        return reg.control_mask(vary_pa=True, vary_backoff=True, vary_nprach=True)

    return {
        "B0": BaselineSpec("B0", "Tuned-Fixed", False,
                           "Tuned fixed configuration from the shared registry", full),
        "B1": BaselineSpec("B1", "Adaptive-OccMatch", False,
                           "Load-aware occupancy-matching admission (adapted heuristic)", full),
        "B2": BaselineSpec("B2", "QL-ACB-only", True,
                           "Tabular Q, admission only (backoff+NPRACH pinned)", acb_only),
        "B3": BaselineSpec("B3", "QL-ACB+Backoff", True,
                           "Tabular Q, admission + backoff (NPRACH pinned)", acb_backoff),
        "B4": BaselineSpec("B4", "QL-ACB+NPRACH", True,
                           "Tabular Q, admission + NPRACH (backoff pinned)", acb_nprach),
        "B5": BaselineSpec("B5", "QL-Joint", True,
                           "Full joint tabular Q (admission + backoff + NPRACH)", joint),
        "B6": BaselineSpec("B6", "DQN", True,
                           "Actual DQN -- NOT IMPLEMENTED (no proxy curve permitted)",
                           full, implemented=False),
    }


def make_policy(spec: BaselineSpec, cfg: Config, registry: ActionRegistry,
                rng: RngBundle, fixed_action_id: int | None = None):
    """Instantiate a (fresh, untrained) policy for a baseline spec."""
    feasible_mask = spec.mask_fn(registry)
    feasible = np.nonzero(feasible_mask)[0]
    if spec.bid == "B0":
        aid = fixed_action_id if fixed_action_id is not None else int(feasible[-1])
        return FixedPolicy(aid), feasible_mask
    if spec.bid == "B1":
        return (AdaptivePolicy(registry, feasible_mask,
                               fixed_backoff_ms=PINNED_BACKOFF_MS,
                               fixed_nprach_id=PINNED_NPRACH_ID), feasible_mask)
    if spec.bid == "B6":
        raise NotImplementedError("B6 (DQN) is intentionally not implemented")
    # learning baselines B2-B5
    n_states = (len(cfg.congestion_bins) - 1) * (len(cfg.load_bins) - 1) * (len(cfg.trend_bins) - 1)
    agent = QLearningAgent(n_states=n_states, n_actions=registry.n_actions,
                           feasible=feasible, learning=cfg.learning,
                           exploration_rng=rng["exploration"],
                           tiebreak_rng=rng["tiebreak"])
    return agent, feasible_mask
