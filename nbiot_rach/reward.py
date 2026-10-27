"""Deadline-aware reward computation from realised events (methodology §F).

    r_t = w_on*C_on/K - w_miss*M_new/K - w_drop*X_new/K - w_q*B_norm - w_r*R_norm

Terms (all from realised events in the just-simulated interval):
  * C_on   -- requests newly completed on or before their deadline;
  * M_new  -- newly matured deadline misses NOT already counted as dropped;
  * X_new  -- newly terminally dropped requests;
  * B_norm -- true pending count / N_meters (offline signal, NOT an observation);
  * R_norm -- resources reserved this interval / interval budget;
  * K      -- fixed reference scale (384), independent of the chosen action.

Each penalty is applied at most once per event (miss / drop flags). A late request
that later drops incurs one miss cost AND one terminal-drop cost -- distinct costs.

Reward ablations (``mode``):
  * "throughput_only"           -> r = C_total / K  (all completions, incl. late)
  * "deadline_aware_no_resource"-> drop the w_q and w_r terms
  * "full"                      -> the full reward above
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RewardWeights


@dataclass
class RewardInputs:
    c_on: int               # newly completed on-time this interval
    c_total: int            # newly completed total (incl. late) this interval
    m_new: int              # newly recorded deadline misses this interval
    x_new: int              # newly dropped this interval
    b_norm: float           # true pending / N_meters
    r_norm: float           # reserved resource / budget


def compute_reward(inp: RewardInputs, w: RewardWeights) -> float:
    K = w.K
    if w.mode == "throughput_only":
        return w.w_on * inp.c_total / K
    if w.mode == "deadline_aware_no_resource":
        return (w.w_on * inp.c_on / K
                - w.w_miss * inp.m_new / K
                - w.w_drop * inp.x_new / K)
    # full reward
    return (w.w_on * inp.c_on / K
            - w.w_miss * inp.m_new / K
            - w.w_drop * inp.x_new / K
            - w.w_q * inp.b_norm
            - w.w_r * inp.r_norm)
