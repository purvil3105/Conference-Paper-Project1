"""Integer time-tick helpers and a dependency-free percentile.

All simulation time is stored as integer microseconds (``int64``) so that
deadline comparisons are exact (no floating-point boundary ambiguity). Physical
durations declared in ms/s are converted to microseconds with explicit rounding
rules (round-half-up to the nearest microsecond).
"""
from __future__ import annotations

import numpy as np

US_PER_MS: int = 1_000
US_PER_S: int = 1_000_000


def ms_to_us(ms: float) -> int:
    """Convert milliseconds to integer microseconds (round half up)."""
    return int(np.floor(ms * US_PER_MS + 0.5))


def s_to_us(s: float) -> int:
    """Convert seconds to integer microseconds (round half up)."""
    return int(np.floor(s * US_PER_S + 0.5))


def us_to_s(us: int | np.ndarray) -> float | np.ndarray:
    """Convert integer microseconds to seconds (float)."""
    return np.asarray(us, dtype=np.float64) / US_PER_S


def ceil_to_grid(t_us: int, grid_us: int, offset_us: int = 0) -> int:
    """Smallest ``offset + k*grid`` (k>=0 integer) that is >= ``t_us``.

    Used to snap an action's effective time to the next NPRACH opportunity
    boundary (never retrospectively).
    """
    if t_us <= offset_us:
        return offset_us
    k = (t_us - offset_us + grid_us - 1) // grid_us
    return offset_us + k * grid_us


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile without external statistics packages.

    Matches numpy's default ('linear') method. Returns ``float('nan')`` for an
    empty input, since a percentile over an empty cohort is undefined.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    if arr.size == 1:
        return float(arr[0])
    rank = (q / 100.0) * (arr.size - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if lo == hi:
        return float(arr[lo])
    frac = rank - lo
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)
