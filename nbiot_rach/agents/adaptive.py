"""B1: load-aware adaptive admission (occupancy-matching heuristic).

This is an explicitly-named occupancy-matching heuristic inspired by D-ACB, NOT a
faithful reproduction of the full D-ACB algorithm (plan §10). It sets

    p_A ~= min(1, n_choices / estimated_eligible_load)

using the *controller observation* idle-count load estimate (the same estimate
the Q-learning controller sees), with backoff and NPRACH pinned to
validation-tuned values. It then snaps to the nearest available action in the
shared registry.
"""
from __future__ import annotations

import numpy as np

from .base import Policy
from ..actions import ActionRegistry


class AdaptivePolicy(Policy):
    name = "adaptive"

    def __init__(self, registry: ActionRegistry, feasible: np.ndarray,
                 fixed_backoff_ms: int = 256, fixed_nprach_id: int = 4):
        self.registry = registry
        self.fixed_backoff_ms = int(fixed_backoff_ms)
        self.fixed_nprach_id = int(fixed_nprach_id)
        # candidate actions matching the pinned backoff + NPRACH config
        mask = (feasible
                & (registry.backoff_ms == self.fixed_backoff_ms)
                & (registry.nprach_config_id == self.fixed_nprach_id))
        self.cand = np.nonzero(mask)[0]
        if self.cand.size == 0:   # fall back to any feasible
            self.cand = np.nonzero(feasible)[0]
        self.cand_pA = registry.pA[self.cand]
        self.n_choices = int(registry.usable_choices[self.cand[0]])

    def select_action(self, state, feasible, train=False, features=None) -> int:
        if features is None:
            target = 1.0
        else:
            _congestion, load_over_ref, _trend = features
            est_eligible = max(load_over_ref * 48.0, 1e-6)
            target = min(1.0, self.n_choices / est_eligible)
        # snap to nearest available admission probability
        j = int(np.argmin(np.abs(self.cand_pA - target)))
        return int(self.cand[j])
