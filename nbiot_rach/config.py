"""Central configuration: scenario, protocol, resource, reward and learning.

Every field is tagged in ``docs/model_assumptions.md`` as ``verified_value``,
``study_assumption`` or ``to_be_mapped``. Values here are *proposed defaults and
search ranges*, not proven-optimal settings. Final tuning choices must be made on
training/validation data only (never on the held-out test traces).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Sequence

import numpy as np

from .timeutils import ms_to_us, s_to_us


# --- Seed namespaces (Implementation plan §11) -----------------------------
TRAIN_SEEDS = tuple(range(1001, 1011))   # policy / exploration training seeds
VAL_SEEDS = tuple(range(2001, 2021))     # validation traffic traces
TEST_SEEDS = tuple(range(3001, 3051))    # held-out final test traffic traces


@dataclass(frozen=True)
class RewardWeights:
    """Deadline-aware reward weights (methodology §F).

    ``r = w_on*C_on/K - w_miss*M_new/K - w_drop*X_new/K - w_q*B_norm - w_r*R_norm``
    """
    w_on: float = 1.0
    w_miss: float = 1.0
    w_drop: float = 1.0
    w_q: float = 0.01
    w_r: float = 0.01
    K: float = 384.0            # fixed reference scale = 48 choices x 8 opportunities
    mode: str = "full"          # "full" | "deadline_aware_no_resource" | "throughput_only"


@dataclass(frozen=True)
class LearningConfig:
    """Tabular Q-learning hyperparameters (methodology §G, plan §9)."""
    gamma: float = 0.99                 # per 640 ms decision (~64 s effective horizon)
    alpha_exponent: float = 0.6         # alpha = (1 + visit_count)^(-exponent)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.80    # linear decay over first 80% of episodes
    q_init: float = 0.0                 # float32 table initialised to 0
    pilot_episodes: int = 100
    main_episodes: int = 2000
    val_every: int = 25


@dataclass(frozen=True)
class Config:
    """A single frozen experiment configuration (scenario + protocol + budget)."""

    name: str = "reference"

    # --- Population / scenario (plan §3) -----------------------------------
    n_meters: int = 50_000
    reporting_period_s: float = 900.0       # study_assumption
    phase_jitter_s: float = 0.0             # extra jitter on periodic phase
    # Event-triggered component
    event_enabled: bool = True
    event_onset_s: float = 30.0
    event_fraction: float = 0.05            # affected-meter fraction
    event_spread_s: float = 10.0            # requests spread over this window
    event_profile: str = "beta34"           # "beta34" | "uniform" | "simultaneous"
    # Deadlines (study_assumption, pending real utility requirement)
    periodic_deadline_s: float = 60.0
    event_deadline_s: float = 30.0
    # Arrival + observation window
    arrival_window_s: float = 120.0         # generate during [0, arrival_window)
    horizon_s: float = 180.0                # observe until every deadline matures
    warmup_s: float = 0.0
    # Deadline-expiry behaviour: record a miss but keep serving (plan §3)
    expiry_as_drop: bool = False
    # Finite queue capacity (0 = unbounded reference). Overflow => explicit drop.
    queue_capacity_per_meter: int = 0

    # --- NPRACH / protocol (plan §3, §5) -----------------------------------
    nprach_period_ms: float = 80.0
    nprach_start_offset_ms: float = 8.0
    preamble_repetitions: int = 1
    preamble_duration_ms: float = 5.6       # study_assumption (format-0-like)
    subcarrier_choices: tuple = (12, 24, 36, 48)   # usable CBRA choices per config id
    retry_limit: int = 10                   # total preamble transmissions (study rule)

    # --- Level B access-stage delays (study_assumption) --------------------
    # A request becomes COMPLETED only after the represented RA procedure.
    rar_delay_ms: float = 40.0
    msg3_delay_ms: float = 32.0
    contention_resolution_delay_ms: float = 40.0
    # Downlink RAR grant capacity per NPRACH opportunity (study_assumption):
    # the eNB can respond to a limited number of detected preambles per window.
    rar_grants_per_opportunity: int = 24

    # --- Resource calendar (plan §5) ---------------------------------------
    # Accounting unit = 3.75 kHz x 1 ms. 180 kHz carrier = 48 subcarriers.
    carrier_subcarriers: int = 48
    msg3_resource_units: float = 6.0        # units per Msg3 grant (study_assumption)

    # --- Control timing (plan §4) ------------------------------------------
    decision_interval_ms: float = 640.0     # eight 80 ms NPRACH periods
    # Base reference: action effective at the next NPRACH boundary (0 added delay,
    # so a_k governs its own window and aligns with its reward). The delay sweep
    # (delay_640 / delay_2560) adds signalling delay as a labelled sensitivity.
    action_signalling_delay_ms: float = 0.0

    # --- Observation / state (plan §8) -------------------------------------
    ewma_alpha: float = 0.2
    state_kind: str = "compact"             # "compact" | "rich" | "oracle"
    congestion_bins: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    load_bins: tuple = (0.0, 1.0, 5.0, 20.0, 100.0, np.inf)
    trend_bins: tuple = (0.0, 0.5, 0.8, 1.0, 1.25, 2.0, np.inf)

    # --- Reward / learning -------------------------------------------------
    reward: RewardWeights = field(default_factory=RewardWeights)
    learning: LearningConfig = field(default_factory=LearningConfig)

    # --- derived integer-microsecond fields (filled in __post_init__) -----
    def __post_init__(self):  # noqa: D401
        object.__setattr__(self, "nprach_period_us", ms_to_us(self.nprach_period_ms))
        object.__setattr__(self, "nprach_offset_us", ms_to_us(self.nprach_start_offset_ms))
        object.__setattr__(self, "decision_interval_us", ms_to_us(self.decision_interval_ms))
        object.__setattr__(self, "horizon_us", s_to_us(self.horizon_s))
        object.__setattr__(self, "arrival_window_us", s_to_us(self.arrival_window_s))
        object.__setattr__(self, "signalling_delay_us", ms_to_us(self.action_signalling_delay_ms))
        access = (self.rar_delay_ms + self.msg3_delay_ms
                  + self.contention_resolution_delay_ms)
        object.__setattr__(self, "access_stage_delay_us", ms_to_us(access))
        # Opportunities per decision window (used for reward scale K documentation)
        object.__setattr__(self, "opportunities_per_window",
                           int(round(self.decision_interval_ms / self.nprach_period_ms)))

    # convenience
    def with_overrides(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        # add derived
        for k in ("nprach_period_us", "nprach_offset_us", "decision_interval_us",
                  "horizon_us", "arrival_window_us", "signalling_delay_us",
                  "access_stage_delay_us", "opportunities_per_window"):
            d[k] = getattr(self, k)
        return d


# --- Scenario registry (plan §11 test groups) ------------------------------
def scenario_registry() -> dict[str, Config]:
    """Named scenarios. The reference case plus one-factor sweeps."""
    base = Config(name="reference")
    scen: dict[str, Config] = {"reference": base}

    # Debug populations
    scen["debug_100"] = base.with_overrides(name="debug_100", n_meters=100)
    scen["debug_1000"] = base.with_overrides(name="debug_1000", n_meters=1000)

    # Population sweep (10k / 50k / 100k) -- for the load (device-count) plots
    for n in (10_000, 20_000, 30_000, 50_000, 75_000, 100_000):
        scen[f"pop_{n}"] = base.with_overrides(name=f"pop_{n}", n_meters=n)

    # Event-fraction sweep
    for frac in (0.01, 0.05, 0.20, 1.00):
        scen[f"evt_{int(frac*100)}pct"] = base.with_overrides(
            name=f"evt_{int(frac*100)}pct", event_fraction=frac)

    # Held-out event widths / onsets (generalisation)
    scen["evt_width_1s"] = base.with_overrides(name="evt_width_1s", event_spread_s=1.0)
    scen["evt_width_20s"] = base.with_overrides(name="evt_width_20s", event_spread_s=20.0)
    scen["evt_onset_60s"] = base.with_overrides(name="evt_onset_60s", event_onset_s=60.0)
    scen["evt_uniform"] = base.with_overrides(name="evt_uniform", event_profile="uniform")
    scen["evt_simultaneous"] = base.with_overrides(
        name="evt_simultaneous", event_profile="simultaneous", event_spread_s=1.0)

    # Delay / holding-time sensitivities (base reference already uses 0 ms)
    scen["delay_640"] = base.with_overrides(name="delay_640",
                                            action_signalling_delay_ms=640.0)
    scen["delay_2560"] = base.with_overrides(name="delay_2560",
                                             action_signalling_delay_ms=2560.0)
    return scen


def population_sweep_configs(
    populations: Sequence[int] = (10_000, 20_000, 30_000, 50_000, 75_000, 100_000),
    base: Config | None = None,
) -> list[Config]:
    """Configs that differ only in ``n_meters`` (the device-count / load axis)."""
    base = base or Config(name="reference")
    return [base.with_overrides(name=f"pop_{n}", n_meters=n) for n in populations]
