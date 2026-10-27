"""Delayed control: old settings remain active until the logged effective time."""
import numpy as np

from nbiot_rach.config import Config
from nbiot_rach.rng import RngBundle
from nbiot_rach.environment import NBIoTRachEnv
from nbiot_rach.actions import ActionRegistry
from nbiot_rach.agents import FixedPolicy


def test_action_effective_after_delay():
    # 2.56 s signalling delay = 4 decision windows. The default active NPRACH
    # config has 48 choices (id 4); the agent selects id 1 (12 choices) at t=0.
    cfg = Config(n_meters=3000, action_signalling_delay_ms=2560.0, horizon_s=6.0)
    reg = ActionRegistry(cfg)
    env = NBIoTRachEnv(cfg, __import__("nbiot_rach.traffic", fromlist=["generate_trace"])
                       .generate_trace(cfg, 3001), RngBundle.create(3001, 0))
    s = env.reset()
    assert env.active_choices == 48          # default active config
    new = reg.index_of(1.0, 0, 1)            # 12 choices
    choices_by_window = []
    while not env.done:
        _, _, _, info = env.step(new)
        choices_by_window.append(info.active_choices)
    # windows 0..3 keep the old 48 choices; from window 4 the new 12 take effect
    assert choices_by_window[0] == 48
    assert choices_by_window[3] == 48
    assert choices_by_window[4] == 12
    assert choices_by_window[5] == 12


def test_zero_delay_effective_same_window():
    cfg = Config(n_meters=3000, action_signalling_delay_ms=0.0, horizon_s=6.0)
    reg = ActionRegistry(cfg)
    from nbiot_rach.traffic import generate_trace
    env = NBIoTRachEnv(cfg, generate_trace(cfg, 3001), RngBundle.create(3001, 0))
    env.reset()
    new = reg.index_of(1.0, 0, 1)            # 12 choices
    _, _, _, info = env.step(new)
    assert info.active_choices == 12          # effective within its own window
