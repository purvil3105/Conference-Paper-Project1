"""Tabular Q-learning controller (methodology §G, plan §9).

  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_{a' in A(s')} Q(s',a') - Q(s,a) ]

with a zero continuation value for a terminal transition. Feasible-action masking
is applied to both action selection and the next-state max. Learning rate
``alpha = (1 + visit_count(s,a))^(-0.6)`` (per state-action). Exploration is
epsilon-greedy; frozen evaluation uses epsilon = 0 and never updates Q.

The next state used in the update is the ACTUAL next observation returned by the
environment (no separately constructed proxy).
"""
from __future__ import annotations

import numpy as np

from .base import Policy
from ..config import LearningConfig


class QLearningAgent(Policy):
    name = "qlearning"
    is_learning = True

    def __init__(self, n_states: int, n_actions: int, feasible: np.ndarray,
                 learning: LearningConfig, exploration_rng: np.random.Generator,
                 tiebreak_rng: np.random.Generator):
        self.n_states = n_states
        self.n_actions = n_actions
        self.feasible = np.asarray(feasible, dtype=np.int64)
        self.lc = learning
        self.q = np.full((n_states, n_actions), learning.q_init, dtype=np.float32)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        # infeasible actions are masked out of every max/argmax with -inf
        self._feas_mask = np.zeros(n_actions, dtype=bool)
        self._feas_mask[self.feasible] = True
        self.exp_rng = exploration_rng
        self.tie_rng = tiebreak_rng
        self.epsilon = learning.eps_start
        self.frozen = False

    # --- schedule ----------------------------------------------------------
    def set_epsilon_for_episode(self, episode: int, total_episodes: int) -> None:
        if self.frozen:
            self.epsilon = 0.0
            return
        decay_eps = max(int(self.lc.eps_decay_fraction * total_episodes), 1)
        if episode >= decay_eps:
            self.epsilon = self.lc.eps_end
        else:
            frac = episode / decay_eps
            self.epsilon = self.lc.eps_start + frac * (self.lc.eps_end - self.lc.eps_start)

    def freeze(self) -> None:
        self.frozen = True
        self.epsilon = 0.0

    def checksum(self) -> float:
        return float(np.sum(self.q.astype(np.float64)))

    # --- action selection --------------------------------------------------
    def _greedy(self, state: int, feasible: np.ndarray) -> int:
        qrow = self.q[state, feasible]
        best = qrow.max()
        winners = feasible[qrow >= best - 1e-12]
        if winners.size == 1:
            return int(winners[0])
        return int(winners[self.tie_rng.integers(0, winners.size)])

    def select_action(self, state: int, feasible: np.ndarray,
                      train: bool = False, features=None) -> int:
        feasible = np.asarray(feasible, dtype=np.int64)
        if train and not self.frozen and self.exp_rng.random() < self.epsilon:
            return int(feasible[self.exp_rng.integers(0, feasible.size)])
        return self._greedy(state, feasible)

    # --- update ------------------------------------------------------------
    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool,
               feasible_next: np.ndarray) -> None:
        if self.frozen:
            return
        self.visits[s, a] += 1
        alpha = (1.0 + self.visits[s, a]) ** (-self.lc.alpha_exponent)
        if terminal:
            target = r
        else:
            fn = np.asarray(feasible_next, dtype=np.int64)
            target = r + self.lc.gamma * float(self.q[s_next, fn].max())
        self.q[s, a] += np.float32(alpha * (target - self.q[s, a]))

    # --- persistence -------------------------------------------------------
    def frozen_action_map(self) -> np.ndarray:
        """uint16 greedy action per state (constant-time deployment lookup)."""
        amap = np.zeros(self.n_states, dtype=np.uint16)
        masked = np.where(self._feas_mask[None, :], self.q, -np.inf)
        amap[:] = np.argmax(masked, axis=1).astype(np.uint16)
        return amap

    def state_dict(self) -> dict:
        return {"q": self.q, "visits": self.visits, "feasible": self.feasible,
                "epsilon": self.epsilon, "frozen": self.frozen}

    def load_state_dict(self, d: dict) -> None:
        self.q = d["q"].astype(np.float32)
        self.visits = d["visits"]
        self.feasible = d["feasible"]
        self._feas_mask = np.zeros(self.n_actions, dtype=bool)
        self._feas_mask[self.feasible] = True
        self.epsilon = float(d.get("epsilon", 0.0))
        self.frozen = bool(d.get("frozen", False))
