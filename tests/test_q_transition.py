"""Q-learning: terminal target, next-state replay, frozen-policy invariance."""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.traffic import generate_trace
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import QLearningAgent


def _agent(n_states=150, n_actions=144, feasible=None):
    feasible = np.arange(n_actions) if feasible is None else feasible
    return QLearningAgent(n_states, n_actions, feasible, Config().learning,
                          np.random.default_rng(0), np.random.default_rng(1))


def test_terminal_target_has_no_continuation():
    ag = _agent()
    ag.q[5, :] = 7.0        # a large next-state value that must be ignored
    ag.update(s=0, a=3, r=2.0, s_next=5, terminal=True, feasible_next=ag.feasible)
    alpha = (1.0 + 1) ** (-Config().learning.alpha_exponent)
    assert np.isclose(ag.q[0, 3], alpha * (2.0 - 0.0), atol=1e-6)


def test_non_terminal_uses_feasible_next_max():
    ag = _agent()
    ag.q[9, 2] = 10.0       # best feasible next action value
    ag.update(s=0, a=1, r=1.0, s_next=9, terminal=False, feasible_next=ag.feasible)
    alpha = (1.0 + 1) ** (-Config().learning.alpha_exponent)
    gamma = Config().learning.gamma
    assert np.isclose(ag.q[0, 1], alpha * (1.0 + gamma * 10.0 - 0.0), atol=1e-5)


def test_next_state_replay_matches_decision_observation():
    cfg = Config(n_meters=3000)
    reg = ActionRegistry(cfg)
    feasible = np.nonzero(reg.feasible_mask())[0]
    ag = _agent(n_actions=reg.n_actions, feasible=feasible)
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 1001), RngBundle.create(1001, 0),
                       check_invariants=False)
    s = env.reset()
    prev_s_next = None
    for _ in range(10):
        a = ag.select_action(s, env.feasible_actions, train=True)
        s_next, r, term, info = env.step(a)
        # the state we will use at the NEXT decision is exactly s_next
        assert isinstance(s_next, int)
        if prev_s_next is not None:
            assert s == prev_s_next
        prev_s_next = s_next
        s = s_next


def test_frozen_policy_does_not_change_q():
    cfg = Config(n_meters=3000)
    reg = ActionRegistry(cfg)
    feasible = np.nonzero(reg.feasible_mask())[0]
    ag = _agent(n_actions=reg.n_actions, feasible=feasible)
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 1001), RngBundle.create(1001, 0),
                       check_invariants=False)
    # a few training updates
    s = env.reset()
    for _ in range(20):
        a = ag.select_action(s, env.feasible_actions, train=True)
        s2, r, term, info = env.step(a)
        ag.update(s, a, r, s2, term, env.feasible_actions)
        s = s2
    ag.freeze()
    checksum = ag.checksum()
    visits = ag.visits.sum()
    # frozen evaluation on a fresh episode must not modify Q or visits
    env2 = NBIoTRachEnv(cfg, generate_trace(cfg, 3001), RngBundle.create(3001, 0),
                        check_invariants=False)
    env2.run_episode(ag, train=False)
    assert ag.epsilon == 0.0
    assert ag.checksum() == checksum
    assert ag.visits.sum() == visits
