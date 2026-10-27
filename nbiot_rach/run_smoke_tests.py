"""Fast, human-readable validation gates (plan §12). Exit code 0 = all pass.

These mirror the pytest suite but run quickly and print a report. They are a
pre-flight check, not a substitute for ``pytest``.
"""
from __future__ import annotations

import sys

import numpy as np

from .config import Config
from .traffic import generate_trace
from .rng import RngBundle
from .environment import NBIoTRachEnv
from .actions import ActionRegistry
from .agents import FixedPolicy
from .metrics import episode_metrics


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run() -> int:
    ok = True
    print("NB-IoT RACH smoke tests")

    # 1. conservation + endpoint on a small reference run
    cfg = Config(n_meters=2000)
    reg = ActionRegistry(cfg)
    tr = generate_trace(cfg, 3001)
    env = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0),
                       action_mask=None, check_invariants=True)
    env.run_episode(FixedPolicy(reg.index_of(0.25, 256, 4)))
    m = episode_metrics(env.ledger, env.log, cfg)
    led = env.ledger
    gen = led.n
    ok &= _check("conservation generated = completed + dropped + pending",
                 gen == m["completed"] + m["dropped_total"] + m["unresolved"]
                 + m["pending_past_deadline"],
                 f"{gen} = {m['completed']}+{m['dropped_total']}+"
                 f"{m['unresolved']}+{m['pending_past_deadline']}")
    ok &= _check("conservation invariant holds at H", m["conservation_ok"])

    # 2. no-arrival scenario: ratios undefined, no spurious completions
    cfg0 = Config(n_meters=10, event_enabled=False, reporting_period_s=1e9)
    tr0 = generate_trace(cfg0, 3001)
    env0 = NBIoTRachEnv(cfg0, tr0, RngBundle.create(3001, 0))
    env0.run_episode(FixedPolicy(reg.index_of(1.0, 0, 4)))
    m0 = episode_metrics(env0.ledger, env0.log, cfg0)
    ok &= _check("no-arrival: 0 requests and NaN on-time ratio",
                 tr0.n_requests == 0 and np.isnan(m0["p_on_time"]))

    # 3. analytical singleton contention: E[successes] = M (1-1/n)^(M-1)
    n_choices, M, trials = 48, 40, 4000
    rng = np.random.default_rng(99)
    succ = 0
    for _ in range(trials):
        ch = rng.integers(0, n_choices, size=M)
        counts = np.bincount(ch, minlength=n_choices)
        succ += np.count_nonzero(counts == 1)
    emp = succ / trials
    analytic = M * (1 - 1 / n_choices) ** (M - 1)
    ok &= _check("contention mean matches M(1-1/n)^(M-1)",
                 abs(emp - analytic) < 0.3,
                 f"emp={emp:.3f} analytic={analytic:.3f}")

    # 4. all-barred (diagnostic pA=0): every request remains pending (WAITING)
    envd = NBIoTRachEnv(cfg, tr, RngBundle.create(3001, 0),
                        include_diagnostic=True, check_invariants=True)
    aid0 = envd.registry.index_of(0.0, 0, 4)
    envd.run_episode(FixedPolicy(aid0))
    mb = episode_metrics(envd.ledger, envd.log, cfg)
    ok &= _check("all-barred (pA=0): no completions, no drops, all pending",
                 mb["completed"] == 0 and mb["dropped_total"] == 0
                 and mb["attempts_total"] == 0)

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
