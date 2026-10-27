"""B0: tuned fixed configuration (methodology §H, plan §10).

A single fixed action tuple selected on training/validation traces from the SAME
action registry. This is explicitly a *tuned* fixed configuration -- not a claim
that it represents "3GPP performance".
"""
from __future__ import annotations

import numpy as np

from .base import Policy


class FixedPolicy(Policy):
    name = "fixed"

    def __init__(self, action_id: int):
        self.action_id = int(action_id)

    def select_action(self, state, feasible, train=False, features=None) -> int:
        return self.action_id
