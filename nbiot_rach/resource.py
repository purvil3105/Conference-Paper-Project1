"""Shared time-frequency resource calendar (methodology §D, plan §5).

Accounting unit = 3.75 kHz x 1 ms. Over one decision window the shared budget is

    R_total = carrier_subcarriers (48) x decision_interval_ms

NPRACH reserves part of this budget (derived from the active configuration's
subcarrier count, preamble duration, repetitions and opportunities -- NOT an
assumed linear rule). The remainder is available to data (Msg3 uplink grants).
We enforce  R_RACH + R_DATA_used <= R_total  and reject over-budget allocations.

Downlink RAR response capacity is accounted separately (per opportunity, in the
environment); it is not "free" merely because the objective concerns uplink
access (plan §5, item 4).
"""
from __future__ import annotations

from .config import Config


class ResourceCalendar:
    """Per-decision-window reservation calendar."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.total_units = float(cfg.carrier_subcarriers * cfg.decision_interval_ms)
        self.msg3_units = float(cfg.msg3_resource_units)
        self.reset_window(nprach_reserved_units=0.0)

    def reset_window(self, nprach_reserved_units: float) -> None:
        """Begin a new decision window with the given NPRACH reservation."""
        if nprach_reserved_units > self.total_units + 1e-9:
            raise ValueError(
                f"NPRACH reservation {nprach_reserved_units} exceeds "
                f"R_total {self.total_units}")
        self.rach_reserved = float(nprach_reserved_units)
        self.data_budget = self.total_units - self.rach_reserved
        self.data_used = 0.0
        self.msg3_grants = 0

    def try_msg3_grant(self) -> bool:
        """Reserve one Msg3 uplink grant if the data budget allows it."""
        if self.data_used + self.msg3_units <= self.data_budget + 1e-9:
            self.data_used += self.msg3_units
            self.msg3_grants += 1
            return True
        return False

    @property
    def reserved_units(self) -> float:
        """Total resources *reserved* this window (RACH + used data)."""
        return self.rach_reserved + self.data_used

    def r_norm(self) -> float:
        """Normalised resource use over the window: reserved / R_total."""
        if self.total_units <= 0:
            return 0.0
        return self.reserved_units / self.total_units
