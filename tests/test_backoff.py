"""Backoff: zero and maximum timer range; retry only at an eligible opportunity;
no retroactive resampling of an existing request's timer."""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.ledger import BACKOFF
from nbiot_rach.agents import FixedPolicy
from nbiot_rach.timeutils import US_PER_MS
from .helpers import custom_trace


def _collide_once(backoff_ms):
    # one opportunity per window; two meters collide on a single subcarrier
    cfg = Config(n_meters=2, event_enabled=False, reporting_period_s=1e9,
                 subcarrier_choices=(1, 1, 1, 1), retry_limit=50,
                 decision_interval_ms=80.0, horizon_s=5.0)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0, 1], [0.0, 0.0], [4.0, 4.0], n_meters=2)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.reset()
    s = env.state()
    a = reg.index_of(1.0, backoff_ms, 1)
    _, _, _, info = env.step(a)   # one opportunity
    return env, info


def test_zero_backoff_retries_next_opportunity():
    env, info = _collide_once(0)
    led = env.ledger
    assert np.all(led.status == BACKOFF)
    # next_eligible equals the (first) opportunity time -> eligible next opportunity
    opp0 = env.cfg.nprach_offset_us
    assert np.all(led.next_eligible_us == opp0)


def test_max_backoff_range():
    env, info = _collide_once(4096)
    led = env.ledger
    opp0 = env.cfg.nprach_offset_us
    bmax = 4096 * US_PER_MS
    assert np.all(led.next_eligible_us >= opp0)
    assert np.all(led.next_eligible_us <= opp0 + bmax)


def test_no_retroactive_resample():
    # after a request is in BACKOFF with a far-future timer, changing the action's
    # backoff setting must not redraw its existing timer.
    env, info = _collide_once(4096)
    led = env.ledger
    saved = led.next_eligible_us.copy()
    # the next opportunity is at 88 ms; requests whose timer is beyond that cannot
    # be released there, so their timer must be untouched by a new action setting.
    next_opp = env.cfg.nprach_offset_us + env.cfg.nprach_period_us
    untouched = saved > next_opp
    a = env.registry.index_of(1.0, 256, 1)
    env.step(a)
    assert np.array_equal(led.next_eligible_us[untouched], saved[untouched])
    assert np.all(led.status[untouched] == BACKOFF)
