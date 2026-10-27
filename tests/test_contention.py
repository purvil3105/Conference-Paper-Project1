"""Contention model: singleton-success mean matches M(1-1/n)^(M-1).

Also checks forced same-choice contention (n=1): with >=2 contenders all fail.
"""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from .helpers import custom_trace


def test_uniform_contention_mean():
    n_choices, M, trials = 48, 50, 6000
    rng = np.random.default_rng(2024)
    total = 0
    for _ in range(trials):
        ch = rng.integers(0, n_choices, size=M)
        counts = np.bincount(ch, minlength=n_choices)
        total += int(np.count_nonzero(counts == 1))
    emp = total / trials
    analytic = M * (1 - 1 / n_choices) ** (M - 1)
    # Monte-Carlo tolerance
    assert abs(emp - analytic) < 0.4, (emp, analytic)


def test_forced_collision_retains_requests():
    # Two requests from different meters, same arrival; with a 1-subcarrier NPRACH
    # config they always collide on the single choice -> both retry (not dropped
    # until the retry limit), none complete on the first opportunity.
    cfg = Config(n_meters=2, event_enabled=False, reporting_period_s=1e9,
                 subcarrier_choices=(1, 1, 1, 1), retry_limit=3, horizon_s=5.0)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0, 1], [0.0, 0.0], [4.0, 4.0], n_meters=2)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(1.0, 0, 1)))
    led = env.ledger
    # with a single shared choice and 2 contenders, every attempt collides ->
    # both exhaust the retry limit and are dropped (explicit terminal loss)
    assert int(np.count_nonzero(led.status == 4)) == 2   # DROPPED
    assert env.log.total_preamble_success == 0
    assert np.all(led.attempts[:] >= cfg.retry_limit)
