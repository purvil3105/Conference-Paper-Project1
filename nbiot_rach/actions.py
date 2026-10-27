"""Joint action registry and resource-feasibility masking (methodology §D, plan §4).

An action is the tuple ``a = (p_A, backoff_ms, nprach_config_id)``:
  * ``p_A``  -- admission probability (NOT barring probability). An eligible
               request is admitted to contend with probability ``p_A``.
  * ``backoff_ms`` -- backoff window; on a failed attempt a timer is drawn
               uniformly from ``[0, backoff_ms]`` (ms), snapped to the next
               opportunity. An existing request's timer is never re-drawn when a
               later action changes the setting.
  * ``nprach_config_id`` -- selects a configured subcarrier count (12/24/36/48).

The Cartesian product is 9 x 4 x 4 = 144 candidate actions (``p_A = 0`` is kept
only as an optional "bar everything" diagnostic, excluded from the default set).

Resource cost is derived from the configuration-to-resource calendar
(3.75 kHz x 1 ms units); it is NOT an assumed linear NPRACH->NPUSCH rule.
``standards_status`` is "abstract" for every row: these are NB-IoT-inspired
candidate values, not a release-specific signalling map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

PA_VALUES = (0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0)
PA_DIAGNOSTIC = 0.0                       # "bar everything" diagnostic only
BACKOFF_MS = (0, 256, 1024, 4096)         # NB-IoT backoff-table entries
NPRACH_CONFIG_IDS = (1, 2, 3, 4)          # -> subcarrier_choices index


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    pA: float
    backoff_ms: int
    nprach_config_id: int
    usable_choices: int
    period_ms: float
    start_offset_ms: float
    repetitions: int
    resource_units: float        # NPRACH reservation per decision window
    standards_status: str        # always "abstract" here
    source_clause: str


class ActionRegistry:
    """The single source of truth for permitted action tuples."""

    def __init__(self, cfg: Config, include_diagnostic: bool = False):
        self.cfg = cfg
        pa_list = list(PA_VALUES)
        if include_diagnostic:
            pa_list = [PA_DIAGNOSTIC] + pa_list
        specs: list[ActionSpec] = []
        aid = 0
        opps = cfg.opportunities_per_window
        for pa in pa_list:
            for bo in BACKOFF_MS:
                for ncid in NPRACH_CONFIG_IDS:
                    choices = cfg.subcarrier_choices[ncid - 1]
                    # Reserved NPRACH units over one decision window:
                    #   choices (subcarriers) x preamble_duration_ms x reps x opps
                    units = (choices * cfg.preamble_duration_ms
                             * cfg.preamble_repetitions * opps)
                    specs.append(ActionSpec(
                        action_id=aid, pA=float(pa), backoff_ms=int(bo),
                        nprach_config_id=int(ncid), usable_choices=int(choices),
                        period_ms=cfg.nprach_period_ms,
                        start_offset_ms=cfg.nprach_start_offset_ms,
                        repetitions=cfg.preamble_repetitions,
                        resource_units=float(units),
                        standards_status="abstract",
                        source_clause="plan_sec4_abstract_grid"))
                    aid += 1
        self.specs = specs
        self.n_actions = len(specs)
        self.pA = np.array([s.pA for s in specs], dtype=np.float64)
        self.backoff_ms = np.array([s.backoff_ms for s in specs], dtype=np.int64)
        self.nprach_config_id = np.array([s.nprach_config_id for s in specs], dtype=np.int64)
        self.usable_choices = np.array([s.usable_choices for s in specs], dtype=np.int64)
        self.resource_units = np.array([s.resource_units for s in specs], dtype=np.float64)

    # --- feasibility -------------------------------------------------------
    def resource_budget_units(self) -> float:
        """R_total per decision window in 3.75 kHz x 1 ms units."""
        return float(self.cfg.carrier_subcarriers * self.cfg.decision_interval_ms)

    def feasible_mask(self) -> np.ndarray:
        """Actions whose NPRACH reservation fits the shared resource budget."""
        return self.resource_units <= self.resource_budget_units() + 1e-9

    def control_mask(self, vary_pa=True, vary_backoff=True, vary_nprach=True,
                     fixed_backoff_ms: int = 0, fixed_nprach_id: int = 4) -> np.ndarray:
        """Restrict the action set for the control-ablation baselines (B2-B5).

        A fixed control is pinned to a (validation-tuned) default value.
        """
        mask = self.feasible_mask()
        if not vary_backoff:
            mask &= (self.backoff_ms == fixed_backoff_ms)
        if not vary_nprach:
            mask &= (self.nprach_config_id == fixed_nprach_id)
        if not vary_pa:
            # pin pA to 1.0 (unrestricted) when admission is not a control
            mask &= np.isclose(self.pA, 1.0)
        return mask

    def index_of(self, pA: float, backoff_ms: int, nprach_config_id: int) -> int:
        for s in self.specs:
            if (abs(s.pA - pA) < 1e-12 and s.backoff_ms == backoff_ms
                    and s.nprach_config_id == nprach_config_id):
                return s.action_id
        raise KeyError((pA, backoff_ms, nprach_config_id))

    def to_rows(self) -> list[dict]:
        return [dict(action_id=s.action_id, pA=s.pA, backoff_ms=s.backoff_ms,
                     nprach_config_id=s.nprach_config_id,
                     usable_choices=s.usable_choices, period_ms=s.period_ms,
                     start_offset_ms=s.start_offset_ms, repetitions=s.repetitions,
                     resource_units=s.resource_units,
                     standards_status=s.standards_status,
                     source_clause=s.source_clause) for s in self.specs]
