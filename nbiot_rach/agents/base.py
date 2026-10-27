"""Policy interface shared by every controller (one registry, fair comparison)."""
from __future__ import annotations

import numpy as np


class Policy:
    """Base policy. ``select_action`` must return a *feasible* action id.

    ``features`` is the controller observation tuple
    ``(congestion, load_over_ref, trend)`` for the window just measured; it is the
    only observation a deployable policy may read (simulator truth is reserved for
    reward/evaluation).
    """

    name = "base"
    is_learning = False

    def select_action(self, state: int, feasible: np.ndarray,
                      train: bool = False, features=None) -> int:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:  # no-op for non-learning policies
        pass

    def freeze(self) -> None:
        pass
