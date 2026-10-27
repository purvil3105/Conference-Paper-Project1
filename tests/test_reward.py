"""Reward event accounting: each miss/drop/completion counted exactly once;
a miss and a later drop on the same request are distinct costs."""
import numpy as np

from nbiot_rach.config import Config, RewardWeights
from nbiot_rach.traffic import generate_trace
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from nbiot_rach.ledger import COMPLETED, DROPPED
from nbiot_rach.reward import RewardInputs, compute_reward
from .helpers import custom_trace


def _collect(env, pol):
    s = env.reset()
    tot = dict(c_on=0, c_total=0, m=0, x=0)
    while not env.done:
        a = pol.select_action(s, env.feasible_actions)
        s, r, term, info = env.step(a)
        tot["c_on"] += info.c_on
        tot["c_total"] += info.c_total
        tot["m"] += info.m_new
        tot["x"] += info.x_new
    return env.ledger, tot


def _assert_identities(led, tot):
    assert tot["c_total"] == int(np.count_nonzero(led.status == COMPLETED))
    assert tot["c_on"] == int(np.count_nonzero(led.on_time))
    assert tot["x"] == int(np.count_nonzero(led.status == DROPPED))
    assert tot["m"] == int(np.count_nonzero(led.miss_recorded))


def test_counting_identities_under_overload():
    # overload -> many drops (fast retry exhaustion before the deadline => no miss)
    cfg = Config(n_meters=50000, horizon_s=200.0)
    reg = ActionRegistry(cfg)
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 3001), RngBundle.create(3001, 0),
                       check_invariants=True)
    led, tot = _collect(env, FixedPolicy(reg.index_of(1.0, 0, 1)))
    _assert_identities(led, tot)
    assert int(np.count_nonzero(led.status == DROPPED)) > 0


def test_miss_then_drop_is_two_distinct_costs():
    # two meters permanently collide on a single subcarrier (no backoff); the
    # deadline (1 s) matures while they are still retrying -> a miss is recorded;
    # much later they exhaust the retry limit -> an explicit drop. One miss AND
    # one drop, each counted once.
    cfg = Config(n_meters=2, event_enabled=False, reporting_period_s=1e9,
                 subcarrier_choices=(1, 1, 1, 1), retry_limit=50, horizon_s=6.0)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0, 1], [0.0, 0.0], [1.0, 1.0], n_meters=2)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    led, tot = _collect(env, FixedPolicy(reg.index_of(1.0, 0, 1)))
    _assert_identities(led, tot)
    both = led.miss_recorded & (led.status == DROPPED) & (led.drop_us >= led.deadline_us)
    assert int(np.count_nonzero(both)) == 2       # both requests miss then drop
    assert tot["m"] == 2 and tot["x"] == 2


def test_throttled_admission_produces_misses():
    # heavy load with strong barring -> many requests wait past their deadline
    # (a miss) without ever being dropped.
    cfg = Config(n_meters=100000)
    reg = ActionRegistry(cfg)
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 3001), RngBundle.create(3001, 0),
                       check_invariants=True)
    led, tot = _collect(env, FixedPolicy(reg.index_of(0.001, 0, 4)))
    _assert_identities(led, tot)
    assert int(np.count_nonzero(led.miss_recorded)) > 0


def test_reward_modes():
    w = RewardWeights()
    inp = RewardInputs(c_on=10, c_total=12, m_new=3, x_new=2, b_norm=0.5, r_norm=0.4)
    full = compute_reward(inp, w)
    no_res = compute_reward(inp, RewardWeights(mode="deadline_aware_no_resource"))
    thru = compute_reward(inp, RewardWeights(mode="throughput_only"))
    assert np.isclose(thru, 12 / w.K)
    assert np.isclose(no_res, (10 - 3 - 2) / w.K)
    assert full < no_res            # resource/backlog penalties reduce the reward
