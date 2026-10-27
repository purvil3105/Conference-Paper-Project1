"""Per-meter concurrency: one active procedure per meter; later messages FIFO."""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from nbiot_rach.ledger import COMPLETED
from .helpers import custom_trace


def test_single_meter_two_requests_served_fifo():
    # one meter generates two requests; with ample subcarriers both succeed, but
    # the second may only start after the first frees the meter (FIFO order).
    cfg = Config(n_meters=1, event_enabled=False, reporting_period_s=1e9,
                 horizon_s=10.0)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0, 0], [0.0, 0.1], [8.0, 8.0], n_meters=1)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(1.0, 0, 4)))
    led = env.ledger
    assert int(np.count_nonzero(led.status == COMPLETED)) == 2
    # request 0 arrives first (FIFO head); its completion precedes request 1's
    # first attempt (the meter was busy until request 0 finished).
    assert led.first_attempt_us[1] >= led.completion_us[0]


def test_no_two_concurrent_procedures_per_meter():
    # heavier: one meter, three requests; never more than one ACCESS at a time is
    # guaranteed structurally -- here we check all eventually complete in order.
    cfg = Config(n_meters=1, event_enabled=False, reporting_period_s=1e9,
                 horizon_s=20.0)
    reg = ActionRegistry(cfg)
    tr = custom_trace([0, 0, 0], [0.0, 0.05, 0.1], [18.0, 18.0, 18.0], n_meters=1)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(1.0, 0, 4)))
    led = env.ledger
    assert int(np.count_nonzero(led.status == COMPLETED)) == 3
    comp = led.completion_us
    assert comp[0] <= comp[1] <= comp[2]      # FIFO completion order
