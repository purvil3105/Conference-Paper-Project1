"""Controller observations and finite-state encoding (methodology §E, plan §8).

Three interfaces are kept strictly separate:
  * simulator truth (the ledger) -- used only for reward and evaluation;
  * controller observations      -- receiver-side estimates only;
  * reward / evaluation records.

Compact state (150 bins = 5 x 5 x 6):
  1. Congestion proxy (5 bins): EWMA of  collided_choices / occupied_choices
     over the measurement window. A receiver-observation proxy, NOT the true
     number of colliding devices.
  2. Eligible-load estimate / 48 (5 bins): idle-count load estimate
     log(I/n)/log(1-1/n), corrected by the *effective* admission probability.
  3. Load trend (6 bins): current eligible-load estimate / previous estimate.

The idle/occupied/success classification is an *ideal-observation assumption*
(perfect preamble detection), documented as such. It is not a proven eNB
capability. A finite table does not make the environment Markov; the compact
controller operates on a quantised observation of a partially observed process.

The "rich" state additionally exposes known active/pending action context; the
"oracle" state exposes true pending load and mean request age -- clearly labelled
unavailable ground truth, for ablation only.
"""
from __future__ import annotations

import numpy as np

from .config import Config

_EPS = 1e-9


def estimate_attempted_load(idle_choices: float, n_choices: int,
                            population_cap: float) -> tuple[float, bool]:
    """Idle-count estimator of the number of attempting (admitted) devices.

    Returns ``(estimate, saturated)``. For ``0 < I < n`` the maximum-likelihood
    idle-count estimate is ``log(I/n) / log(1 - 1/n)``. ``I == 0`` is *saturated*
    (not an exact measurement): a documented population-based cap is returned.
    ``I == n`` means no attempts observed -> estimate 0.
    """
    n = float(n_choices)
    I = float(idle_choices)
    if I >= n:                     # empty observation
        return 0.0, False
    if I <= 0.0:                   # saturated
        return population_cap, True
    est = np.log(I / n) / np.log(1.0 - 1.0 / n)
    return float(est), False


def _bin_index(value: float, edges) -> int:
    """Index of the half-open bin [edges[i], edges[i+1]) containing value.

    Values on the last edge map to the top bin (last-edge equality rule).
    """
    edges = np.asarray(edges, dtype=np.float64)
    nb = edges.size - 1
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return nb - 1
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return min(max(idx, 0), nb - 1)


class ObservationModel:
    """Maintains EWMA/trend state and encodes the quantised controller state."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_congestion = len(cfg.congestion_bins) - 1
        self.n_load = len(cfg.load_bins) - 1
        self.n_trend = len(cfg.trend_bins) - 1
        self.n_states = self.n_congestion * self.n_load * self.n_trend
        # population cap used when the idle-count estimator saturates
        self.population_cap = float(cfg.n_meters)
        self.reset()

    def reset(self) -> None:
        self.ewma_congestion = 0.0
        self.has_congestion = False
        self.prev_load_est = None
        # per-window accumulators
        self._win_reset()
        self.last_features = (0.0, 0.0, 1.0)

    def _win_reset(self) -> None:
        self._occ_sum = 0            # occupied choices summed over opportunities
        self._col_sum = 0            # collided choices summed
        self._load_ests: list[float] = []
        self._saturated_any = False
        self._n_opps = 0

    def update_opportunity(self, occupied: int, collided: int, idle: int,
                           n_choices: int, active_pA: float) -> None:
        """Feed one NPRACH opportunity's detected occupancy statistics."""
        self._n_opps += 1
        self._occ_sum += occupied
        self._col_sum += collided
        # EWMA congestion proxy updated per opportunity
        if occupied > 0:
            ratio = collided / occupied
            if not self.has_congestion:
                self.ewma_congestion = ratio
                self.has_congestion = True
            else:
                a = self.cfg.ewma_alpha
                self.ewma_congestion = a * ratio + (1 - a) * self.ewma_congestion
        # idle-count load estimate, corrected by effective admission prob
        est, sat = estimate_attempted_load(idle, n_choices, self.population_cap)
        self._saturated_any = self._saturated_any or sat
        pA = max(active_pA, 1e-4)
        eligible_est = est / pA
        self._load_ests.append(eligible_est)

    def window_features(self) -> tuple[float, float, float]:
        """Compute (congestion, load_over_ref, trend) for the completed window."""
        congestion = self.ewma_congestion if self.has_congestion else 0.0
        if self._load_ests:
            load_est = float(np.mean(self._load_ests))
        else:
            load_est = 0.0
        load_over_ref = load_est / 48.0
        if self.prev_load_est is None or self.prev_load_est <= _EPS:
            trend = 1.0                      # explicit empty-history / zero rule
        else:
            trend = load_est / self.prev_load_est
        self.prev_load_est = load_est
        self.last_features = (congestion, load_over_ref, trend)
        self._win_reset()
        return congestion, load_over_ref, trend

    def encode(self, features: tuple[float, float, float]) -> int:
        congestion, load_over_ref, trend = features
        ci = _bin_index(congestion, self.cfg.congestion_bins)
        li = _bin_index(load_over_ref, self.cfg.load_bins)
        ti = _bin_index(trend, self.cfg.trend_bins)
        return (ci * self.n_load + li) * self.n_trend + ti

    def decode(self, state: int) -> tuple[int, int, int]:
        ti = state % self.n_trend
        rest = state // self.n_trend
        li = rest % self.n_load
        ci = rest // self.n_load
        return ci, li, ti
