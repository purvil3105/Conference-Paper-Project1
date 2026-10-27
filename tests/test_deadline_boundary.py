"""Deadline boundary: completion at d-1 tick and d is on time; d+1 tick is late.

We run a single uncontended request to obtain its real completion timestamp c,
then set the deadline to c-1us, c and c+1us and check the on-time / miss outcome
produced by the actual simulator pipeline.
"""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from .helpers import custom_trace


def _run_single(deadline_us, arrival_us=0):
    cfg = Config(n_meters=1, event_enabled=False, reporting_period_s=1e9)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0], [arrival_us / 1e6], [deadline_us / 1e6], n_meters=1)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(1.0, 0, 4)))
    return env.ledger


def test_completion_boundary():
    # 1) discover the true completion timestamp with a very loose deadline
    led = _run_single(deadline_us=10_000_000_000)
    c = int(led.completion_us[0])
    assert c > 0

    # 2) deadline exactly at completion -> on time, no miss
    led_eq = _run_single(deadline_us=c)
    assert led_eq.on_time[0] and not led_eq.miss_recorded[0]

    # 3) deadline one tick before completion -> late, miss recorded
    led_lo = _run_single(deadline_us=c - 1)
    assert (not led_lo.on_time[0]) and led_lo.miss_recorded[0]

    # 4) deadline one tick after completion -> on time, no miss
    led_hi = _run_single(deadline_us=c + 1)
    assert led_hi.on_time[0] and not led_hi.miss_recorded[0]


def test_sim_end_does_not_create_or_rescue_outcomes():
    # a request whose deadline matures while still pending is a miss, not a success
    cfg = Config(n_meters=1, event_enabled=False, reporting_period_s=1e9,
                 horizon_s=5.0)
    reg = ActionRegistry(cfg, include_diagnostic=True)
    tr = custom_trace([0], [0.0], [2.0], n_meters=1)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0),
                       include_diagnostic=True, check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(0.0, 0, 4)))  # barred: never served
    assert not env.ledger.on_time[0]
    assert env.ledger.miss_recorded[0]
    assert env.ledger.status[0] not in (3, 4)  # still pending (kept serving)
