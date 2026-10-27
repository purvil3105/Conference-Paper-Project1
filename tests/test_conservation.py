"""Mass conservation: A(t) = C(t) + X(t) + B(t) after every event batch."""
import numpy as np
import pytest

from nbiot_rach.config import Config
from nbiot_rach.traffic import generate_trace
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from nbiot_rach.ledger import ConservationError, COMPLETED, DROPPED


@pytest.mark.parametrize("n_meters", [500, 5000])
@pytest.mark.parametrize("tuple_", [(1.0, 0, 1), (0.05, 1024, 4), (0.25, 256, 2)])
def test_conservation_every_step(n_meters, tuple_):
    cfg = Config(n_meters=n_meters)
    reg = ActionRegistry(cfg)
    tr = generate_trace(cfg, 3001)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    pol = FixedPolicy(reg.index_of(*tuple_))
    s = env.reset()
    # invariant is asserted inside step(); also check explicitly each step
    while not env.done:
        a = pol.select_action(s, env.feasible_actions)
        s, r, term, info = env.step(a)
        env.ledger.assert_invariants(info.t_end_us)
    led = env.ledger
    A = led.n
    C = int(np.count_nonzero(led.status == COMPLETED))
    X = int(np.count_nonzero(led.status == DROPPED))
    B = led.pending_count(cfg.horizon_us)
    assert A == C + X + B


def test_overflow_is_an_explicit_drop():
    # finite per-meter queue capacity would create overflow drops; here we just
    # confirm no request is ever silently removed under heavy load.
    cfg = Config(n_meters=50000)
    reg = ActionRegistry(cfg)
    tr = generate_trace(cfg, 3001)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0), check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(1.0, 0, 1)))  # overload
    led = env.ledger
    terminal = int(np.count_nonzero((led.status == COMPLETED) | (led.status == DROPPED)))
    pending = led.pending_count(cfg.horizon_us)
    assert terminal + pending == led.n
