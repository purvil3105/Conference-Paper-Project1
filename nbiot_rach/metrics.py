"""Request-level service metrics, delays, resource use and uncertainty.

Primary metric (methodology §I, plan §11): on-time completion fraction over the
deadline-evaluable cohort E(H). For the primary PAIRED comparison we use a fixed
arrival cohort and a horizon H large enough that every cohort deadline has
matured (arrivals < 120 s, deadlines <= 60 s => H = 180 s), so every policy is
scored on the same generated requests.

Reported alongside (never delay alone -- survivorship bias):
  * miss fraction (the complement, same information in another form);
  * completions, late completions, drops by reason, unresolved;
  * median / p95 / p99 arrival-to-completion delay among completed requests;
  * attempt-conditioned preamble success (exact denominator);
  * collision measures (collided devices / attempts; collided choices / occupied);
  * resource reservation/use;
  * backlog over time + backlog-drain proxy.

A zero denominator returns NaN ("not applicable"), never 0.
"""
from __future__ import annotations

import numpy as np

from .ledger import (RequestLedger, COMPLETED, DROPPED, WAITING, BACKOFF, ACCESS,
                     DROP_RETRY_LIMIT, DROP_OVERFLOW, DROP_EXPIRY)
from .traffic import CLASS_PERIODIC, CLASS_EVENT
from .timeutils import percentile


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def episode_metrics(ledger: RequestLedger, env_log, cfg, h_us: int | None = None) -> dict:
    h_us = cfg.horizon_us if h_us is None else h_us
    led = ledger
    arrived = led.arrival_us <= h_us
    deadline_elapsed = led.deadline_us <= h_us
    dropped_by_h = (led.status == DROPPED) & (led.drop_us <= h_us) & (led.drop_us >= 0)

    # Deadline-evaluable cohort E(H)
    cohort = arrived & (deadline_elapsed | (dropped_by_h & ~deadline_elapsed))
    n_cohort = int(np.count_nonzero(cohort))

    on_time_cohort = led.on_time & cohort
    p_on = _ratio(np.count_nonzero(on_time_cohort), n_cohort)
    p_miss = (1.0 - p_on) if not np.isnan(p_on) else float("nan")

    def class_p_on(cls):
        m = cohort & (led.traffic_class == cls)
        return _ratio(np.count_nonzero(led.on_time & m), np.count_nonzero(m))

    # completions & delays
    completed = led.status == COMPLETED
    delays_s = (led.completion_us[completed] - led.arrival_us[completed]) / 1e6
    n_completed = int(np.count_nonzero(completed))
    n_late = int(np.count_nonzero(completed & ~led.on_time))

    # unresolved: arrived, non-terminal, deadline NOT yet matured by H
    non_terminal = (led.status == WAITING) | (led.status == BACKOFF) | (led.status == ACCESS)
    unresolved = int(np.count_nonzero(arrived & non_terminal & ~deadline_elapsed))
    pending_missed = int(np.count_nonzero(arrived & non_terminal & deadline_elapsed))

    attempts = env_log.total_attempts
    preamble_ok = env_log.total_preamble_success

    metrics = {
        "config_name": cfg.name,
        "n_meters": cfg.n_meters,
        "n_requests": led.n,
        "cohort_size": n_cohort,
        # primary
        "p_on_time": p_on,
        "p_miss": p_miss,
        "p_on_time_periodic": class_p_on(CLASS_PERIODIC),
        "p_on_time_event": class_p_on(CLASS_EVENT),
        # service outcomes
        "completed": n_completed,
        "on_time": int(np.count_nonzero(led.on_time)),
        "late_completions": n_late,
        "dropped_total": int(np.count_nonzero(led.status == DROPPED)),
        "drops_retry_limit": int(np.count_nonzero(led.drop_reason == DROP_RETRY_LIMIT)),
        "drops_overflow": int(np.count_nonzero(led.drop_reason == DROP_OVERFLOW)),
        "drops_expiry": int(np.count_nonzero(led.drop_reason == DROP_EXPIRY)),
        "unresolved": unresolved,
        "pending_past_deadline": pending_missed,
        # delays among completed (seconds)
        "delay_median_s": percentile(delays_s, 50),
        "delay_p95_s": percentile(delays_s, 95),
        "delay_p99_s": percentile(delays_s, 99),
        "delay_mean_s": float(np.mean(delays_s)) if delays_s.size else float("nan"),
        # contention (attempt-conditioned, explicit denominators)
        "attempts_total": int(attempts),
        "preamble_success_total": int(preamble_ok),
        "preamble_success_prob": _ratio(preamble_ok, attempts),
        "collision_prob_devices": _ratio(env_log.total_collided_devices, attempts),
        "collision_prob_choices": _ratio(env_log.total_collided_choices,
                                          env_log.total_occupied_choices),
        "rar_fail_total": int(env_log.total_rar_fail),
        "msg3_fail_total": int(env_log.total_msg3_fail),
        # resources (reserved / used / budget over all windows)
        "nprach_units_sum": float(env_log.nprach_units_sum),
        "data_units_sum": float(env_log.data_units_sum),
        "budget_units_sum": float(env_log.budget_units_sum),
        "resource_utilisation": _ratio(
            env_log.nprach_units_sum + env_log.data_units_sum,
            env_log.budget_units_sum),
        # conservation check at H
        "conservation_ok": _conservation_ok(led, h_us),
    }
    # backlog-drain proxy: time after event onset until backlog <= 5% of peak
    metrics["peak_backlog"], metrics["backlog_drain_time_s"] = _backlog_drain(env_log)
    return metrics


def _conservation_ok(led: RequestLedger, h_us: int) -> bool:
    try:
        led.assert_invariants(h_us)
        return True
    except AssertionError:
        return False


def _backlog_drain(env_log):
    series = env_log.backlog_series
    if not series:
        return 0, float("nan")
    ts = np.array([s[0] for s in series])
    bl = np.array([s[1] for s in series])
    peak = int(bl.max())
    if peak == 0:
        return 0, float("nan")
    peak_t = ts[int(np.argmax(bl))]
    thresh = 0.05 * peak
    after = np.nonzero((ts > peak_t) & (bl <= thresh))[0]
    drain = float(ts[after[0]] - peak_t) if after.size else float("nan")
    return peak, drain


# --- aggregation across seeds / uncertainty --------------------------------
def aggregate(metric_dicts: list[dict], keys: list[str] | None = None) -> dict:
    """Mean / std / n across a list of per-run metric dicts (numeric keys only)."""
    if not metric_dicts:
        return {}
    if keys is None:
        keys = [k for k, v in metric_dicts[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in metric_dicts], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                      "n": int(vals.size)}
    return out


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    rng = rng or np.random.default_rng(12345)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_difference(values_a, values_b) -> dict:
    """Paired trace-level difference (A - B) with a bootstrap CI."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    d = a - b
    lo, hi = bootstrap_ci(d)
    return {"mean_diff": float(np.mean(d)) if d.size else float("nan"),
            "ci_low": lo, "ci_high": hi, "n": int(d.size)}
