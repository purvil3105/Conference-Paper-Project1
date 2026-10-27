"""AMI traffic generation: periodic meter reports + event-triggered requests.

Two separately parameterised components (methodology §C):
  (i)  periodic reports: period, uniform initial phase, optional jitter;
  (ii) event-triggered: onset, spread, affected-meter fraction, burst profile.

Each generated message gets its own request id (no silent coalescing). A single
meter may generate both a periodic and an event request in the same window; these
are distinct requests served in FIFO order (methodology §A, plan §7).

The Beta(3,4) burst profile is ONE scenario, not a claim about all AMI traffic.
Its CDF is  F(u) = 20u^3 - 45u^4 + 36u^5 - 10u^6  and we invert it by bisection,
mapping  t = onset + spread * u  (plan "Portability rules").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .rng import RngBundle
from .timeutils import s_to_us

# Traffic-class codes
CLASS_PERIODIC = 0
CLASS_EVENT = 1
CLASS_NAMES = {CLASS_PERIODIC: "periodic", CLASS_EVENT: "event"}


def beta34_cdf(u: np.ndarray | float) -> np.ndarray | float:
    """CDF of Beta(3,4) on [0,1]:  20u^3 - 45u^4 + 36u^5 - 10u^6."""
    u = np.asarray(u, dtype=np.float64)
    return 20 * u**3 - 45 * u**4 + 36 * u**5 - 10 * u**6


def beta34_inverse(p: np.ndarray, iters: int = 60) -> np.ndarray:
    """Invert the Beta(3,4) CDF by bisection (monotone on [0,1])."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    lo = np.zeros_like(p)
    hi = np.ones_like(p)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = beta34_cdf(mid)
        high = f > p
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
    return 0.5 * (lo + hi)


@dataclass
class Trace:
    """An immutable arrival trace (parallel arrays, sorted by arrival time)."""
    request_id: np.ndarray      # int64, 0..N-1
    meter_id: np.ndarray        # int64
    arrival_us: np.ndarray      # int64 microseconds
    deadline_us: np.ndarray     # int64 microseconds
    traffic_class: np.ndarray   # int8 (CLASS_PERIODIC / CLASS_EVENT)
    trace_seed: int
    config_name: str
    n_meters: int

    @property
    def n_requests(self) -> int:
        return int(self.request_id.size)

    def summary(self) -> dict:
        return {
            "trace_seed": self.trace_seed,
            "config_name": self.config_name,
            "n_meters": self.n_meters,
            "n_requests": self.n_requests,
            "n_periodic": int(np.sum(self.traffic_class == CLASS_PERIODIC)),
            "n_event": int(np.sum(self.traffic_class == CLASS_EVENT)),
            "first_arrival_s": float(self.arrival_us.min() / 1e6) if self.n_requests else float("nan"),
            "last_arrival_s": float(self.arrival_us.max() / 1e6) if self.n_requests else float("nan"),
        }


def generate_trace(cfg: Config, trace_seed: int) -> Trace:
    """Generate one reproducible periodic-plus-event arrival trace."""
    rng = RngBundle.create(trace_seed=trace_seed, policy_seed=0)["traffic"]
    period_us = s_to_us(cfg.reporting_period_s)
    window_us = cfg.arrival_window_us

    meter_ids: list[np.ndarray] = []
    arrivals: list[np.ndarray] = []
    deadlines: list[np.ndarray] = []
    classes: list[np.ndarray] = []

    # --- (i) periodic reports ---------------------------------------------
    # Uniform initial phase in [0, period). A meter reports in the window iff its
    # phase (plus optional jitter) lands in [0, arrival_window).
    phase = rng.uniform(0.0, cfg.reporting_period_s, size=cfg.n_meters)
    if cfg.phase_jitter_s > 0.0:
        phase = phase + rng.uniform(-cfg.phase_jitter_s, cfg.phase_jitter_s,
                                    size=cfg.n_meters)
    phase_us = np.floor(phase * 1e6 + 0.5).astype(np.int64)
    in_window = (phase_us >= 0) & (phase_us < window_us)
    p_meters = np.nonzero(in_window)[0].astype(np.int64)
    p_arr = phase_us[in_window]
    if p_meters.size:
        meter_ids.append(p_meters)
        arrivals.append(p_arr)
        deadlines.append(p_arr + s_to_us(cfg.periodic_deadline_s))
        classes.append(np.full(p_meters.size, CLASS_PERIODIC, dtype=np.int8))

    # --- (ii) event-triggered requests ------------------------------------
    if cfg.event_enabled and cfg.event_fraction > 0.0:
        n_evt = int(round(cfg.event_fraction * cfg.n_meters))
        if n_evt > 0:
            evt_meters = rng.choice(cfg.n_meters, size=n_evt, replace=False).astype(np.int64)
            onset_us = s_to_us(cfg.event_onset_s)
            spread_us = s_to_us(cfg.event_spread_s)
            if cfg.event_profile == "beta34":
                u = beta34_inverse(rng.uniform(0.0, 1.0, size=n_evt))
            elif cfg.event_profile == "uniform":
                u = rng.uniform(0.0, 1.0, size=n_evt)
            elif cfg.event_profile == "simultaneous":
                u = np.zeros(n_evt, dtype=np.float64)
            else:
                raise ValueError(f"unknown event_profile {cfg.event_profile!r}")
            e_arr = onset_us + np.floor(u * spread_us + 0.5).astype(np.int64)
            meter_ids.append(evt_meters)
            arrivals.append(e_arr)
            deadlines.append(e_arr + s_to_us(cfg.event_deadline_s))
            classes.append(np.full(n_evt, CLASS_EVENT, dtype=np.int8))

    if not meter_ids:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_c = np.zeros(0, dtype=np.int8)
        return Trace(empty_i.copy(), empty_i.copy(), empty_i.copy(), empty_i.copy(),
                     empty_c, trace_seed, cfg.name, cfg.n_meters)

    meter_id = np.concatenate(meter_ids)
    arrival_us = np.concatenate(arrivals)
    deadline_us = np.concatenate(deadlines)
    traffic_class = np.concatenate(classes)

    # Stable sort by (arrival, meter) so FIFO order within a meter is well defined.
    order = np.lexsort((meter_id, arrival_us))
    meter_id = meter_id[order]
    arrival_us = arrival_us[order]
    deadline_us = deadline_us[order]
    traffic_class = traffic_class[order]
    request_id = np.arange(meter_id.size, dtype=np.int64)

    return Trace(request_id, meter_id, arrival_us, deadline_us, traffic_class,
                 trace_seed, cfg.name, cfg.n_meters)
