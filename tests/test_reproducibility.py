"""Reproducibility: same seed + config + trace + policy reproduces the result."""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.traffic import generate_trace
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy
from nbiot_rach.baseline_registry import baseline_specs
from nbiot_rach.training import train_qlearning
from nbiot_rach.metrics import episode_metrics


def _run_once():
    cfg = Config(n_meters=8000)
    reg = ActionRegistry(cfg)
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 3001), RngBundle.create(3001, 0))
    env.run_episode(FixedPolicy(reg.index_of(0.1, 256, 3)))
    return env.ledger, episode_metrics(env.ledger, env.log, cfg)


def test_trace_generation_is_reproducible():
    cfg = Config(n_meters=20000)
    a = generate_trace(cfg, 3001)
    b = generate_trace(cfg, 3001)
    assert np.array_equal(a.arrival_us, b.arrival_us)
    assert np.array_equal(a.meter_id, b.meter_id)
    assert np.array_equal(a.deadline_us, b.deadline_us)


def test_episode_is_reproducible():
    led1, m1 = _run_once()
    led2, m2 = _run_once()
    assert np.array_equal(led1.completion_us, led2.completion_us)
    assert np.array_equal(led1.status, led2.status)
    assert m1["p_on_time"] == m2["p_on_time"]
    assert m1["preamble_success_prob"] == m2["preamble_success_prob"]


def test_training_is_reproducible():
    cfg = Config(n_meters=5000)
    spec = baseline_specs()["B5"]
    ag1, _, _ = train_qlearning(cfg, spec, episodes=8, policy_seed=0,
                                val_seeds=(2001, 2002), val_every=4)
    ag2, _, _ = train_qlearning(cfg, spec, episodes=8, policy_seed=0,
                                val_seeds=(2001, 2002), val_every=4)
    assert ag1.checksum() == ag2.checksum()
