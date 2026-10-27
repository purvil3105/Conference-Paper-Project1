"""Reproducible, *independent* random-number streams.

The implementation plan (§11) requires separating traffic, controller
exploration, contention choice, backoff, and channel randomness so that
different policies do not desynchronise a shared seed. We derive independent
``numpy.random.Generator`` streams from a ``(trace_seed, policy_seed)`` pair
using ``SeedSequence.spawn`` with a fixed stream ordering.

Keying environmental draws by request/attempt (the plan's "stronger pairing")
is achieved in practice by giving the *environment* streams a seed that depends
only on the traffic trace seed, so two policies replayed on the same trace see
the same arrival process and the same contention/backoff draw stream *until*
their decisions diverge. For the primary paired comparison we additionally share
the saved arrival trace itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed stream index ordering (do not reorder: changes reproducibility).
_STREAMS = (
    "traffic",      # 0 arrival timestamps / affected meters
    "contention",   # 1 subcarrier choice
    "admission",    # 2 Bernoulli admission draw
    "backoff",      # 3 backoff timer draw
    "channel",      # 4 reserved: detection / channel failures
    "exploration",  # 5 controller epsilon-greedy
    "tiebreak",     # 6 greedy tie-breaking
    "bootstrap",    # 7 statistics resampling
)
_STREAM_IDX = {name: i for i, name in enumerate(_STREAMS)}


@dataclass
class RngBundle:
    """A bundle of named, independent generators for one (trace, policy) run."""
    trace_seed: int
    policy_seed: int
    _gens: dict

    @classmethod
    def create(cls, trace_seed: int, policy_seed: int = 0) -> "RngBundle":
        # Environmental streams depend only on the trace seed; controller streams
        # depend on the policy seed. This keeps the exogenous process paired
        # across policies on the same trace.
        env_ss = np.random.SeedSequence([trace_seed, 0xE11A])
        ctl_ss = np.random.SeedSequence([policy_seed, 0xC7A1])
        env_children = env_ss.spawn(len(_STREAMS))
        ctl_children = ctl_ss.spawn(len(_STREAMS))
        gens = {}
        for name, i in _STREAM_IDX.items():
            if name in ("exploration", "tiebreak", "bootstrap"):
                gens[name] = np.random.default_rng(ctl_children[i])
            else:
                gens[name] = np.random.default_rng(env_children[i])
        return cls(trace_seed=trace_seed, policy_seed=policy_seed, _gens=gens)

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._gens[name]
