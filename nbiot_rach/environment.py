"""Discrete-event NB-IoT RACH environment (methodology §A/G, plan §6/7).

One ``step`` advances exactly one 640 ms decision window (eight 80 ms NPRACH
opportunities) following the methodology's causal order:

  (1) create arrivals            (handled implicitly: a request is "generated"
                                  once sim-time passes its arrival timestamp)
  (2) process matured deadlines
  (3) update retry timers / release eligible requests from BACKOFF
  (4) form the controller observation (only permitted measurements)
  (5) select a feasible action   (done by the agent, outside step)
  (6) schedule the action; apply only after its modelled signalling delay,
      snapped to the next NPRACH opportunity boundary (never retrospectively)
  (7) execute the access attempt + represented Level-B stages (RAR, Msg3, CR)
  (8) record completions, terminal drops, deadline events, resource use
  (9) compute reward from realised events in the interval
  (10) construct the ACTUAL next observation and return it

Endpoint (Level B): a request is COMPLETED only after the represented
random-access procedure (preamble success -> RAR grant -> Msg3 grant ->
contention-resolution delay). A singleton preamble outcome is a *preamble
success*, not automatically completion. This is NOT application-message delivery
(that would be Level C). All stage timings/capacities are study assumptions
(see docs/model_assumptions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .actions import ActionRegistry
from .ledger import (RequestLedger, WAITING, BACKOFF, ACCESS, COMPLETED, DROPPED,
                     DROP_RETRY_LIMIT, DROP_EXPIRY)
from .observation import ObservationModel
from .resource import ResourceCalendar
from .reward import RewardInputs, compute_reward
from .rng import RngBundle
from .traffic import Trace
from .timeutils import ceil_to_grid, US_PER_MS


@dataclass
class StepInfo:
    """Diagnostics returned alongside each step (plan: log everything)."""
    reward_inputs: RewardInputs
    c_on: int
    c_total: int
    c_late: int
    m_new: int
    x_new: int
    b_norm: float
    r_norm: float
    selected_action: int
    active_pA: float
    active_backoff_ms: int
    active_choices: int
    pending_effective_us: int
    window_index: int
    t_start_us: int
    t_end_us: int


@dataclass
class EpisodeLog:
    backlog_series: list = field(default_factory=list)   # (t_s, pending)
    reward_series: list = field(default_factory=list)
    action_series: list = field(default_factory=list)
    # episode-level contention / resource counters
    total_attempts: int = 0
    total_preamble_success: int = 0
    total_collided_devices: int = 0
    total_collided_choices: int = 0
    total_occupied_choices: int = 0
    total_opportunities: int = 0
    total_rar_fail: int = 0
    total_msg3_fail: int = 0
    nprach_units_sum: float = 0.0
    data_units_sum: float = 0.0
    budget_units_sum: float = 0.0


class NBIoTRachEnv:
    """Mass-conserving NB-IoT RACH discrete-event environment."""

    def __init__(self, cfg: Config, trace: Trace, rng: RngBundle,
                 action_mask: np.ndarray | None = None,
                 check_invariants: bool = True,
                 include_diagnostic: bool = False):
        self.cfg = cfg
        self.trace = trace
        self.rng = rng
        self.registry = ActionRegistry(cfg, include_diagnostic=include_diagnostic)
        self.obs_model = ObservationModel(cfg)
        self.calendar = ResourceCalendar(cfg)
        self.check_invariants = check_invariants
        feas = self.registry.feasible_mask()
        if action_mask is not None:
            feas = feas & action_mask
        self.action_mask = feas
        self.feasible_actions = np.nonzero(feas)[0]
        if self.feasible_actions.size == 0:
            raise ValueError("no feasible actions under the given mask")
        self.n_states = self.obs_model.n_states
        self.n_actions = self.registry.n_actions
        self._multi_meter_mask = self._compute_multi_meter_mask()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _compute_multi_meter_mask(self) -> np.ndarray:
        """True for requests whose meter generates more than one request."""
        counts = np.bincount(self.trace.meter_id, minlength=self.trace.n_meters)
        return counts[self.trace.meter_id] > 1

    def reset(self) -> int:
        self.ledger = RequestLedger(self.trace)
        self.obs_model.reset()
        self.t_us = 0
        self.window_index = 0
        interval = self.cfg.decision_interval_us
        self.n_windows = int(np.ceil(self.cfg.horizon_us / interval))
        # Decision windows fall on a 640 ms grid, so the simulation ends at the
        # first grid point at or beyond the horizon. Conservation / finalisation /
        # cohort evaluation all use this actual end time.
        self.end_us = self.n_windows * interval
        self.meter_busy = np.zeros(self.trace.n_meters, dtype=bool)
        # Active control (window-0 defaults: unrestricted admission, no backoff,
        # largest NPRACH config = most contention resources).
        default_aid = int(self.feasible_actions[0])
        # prefer pA=1, backoff=0, nprach=4 if feasible
        try:
            default_aid = self.registry.index_of(1.0, 0, 4)
            if not self.action_mask[default_aid]:
                default_aid = int(self.feasible_actions[0])
        except KeyError:
            pass
        self._set_active(default_aid)
        # scheduled-but-not-yet-effective actions: list of (effective_us, action_id),
        # appended in non-decreasing effective-time order (one decision per window).
        self.pending: list[tuple[int, int]] = []
        self.last_scheduled_eff_us = -1
        self.log = EpisodeLog()
        self.done = False
        # initial observation (cold start, no measurements yet)
        self._last_state = self.obs_model.encode((0.0, 0.0, 1.0))
        return self._last_state

    def _set_active(self, action_id: int) -> None:
        self.active_aid = int(action_id)
        self.active_pA = float(self.registry.pA[action_id])
        self.active_backoff_us = int(self.registry.backoff_ms[action_id]) * US_PER_MS
        self.active_backoff_ms = int(self.registry.backoff_ms[action_id])
        self.active_choices = int(self.registry.usable_choices[action_id])
        self.active_resource_units = float(self.registry.resource_units[action_id])

    # ------------------------------------------------------------- public API
    def feasible_action_set(self) -> np.ndarray:
        return self.feasible_actions

    def state(self) -> int:
        return self._last_state

    def step(self, action_id: int):
        """Apply ``action_id`` over one decision window. Returns
        ``(s_next, reward, terminal, StepInfo)``."""
        if self.done:
            raise RuntimeError("episode already terminated; call reset()")
        if not self.action_mask[action_id]:
            raise ValueError(f"action {action_id} not feasible under mask")
        cfg = self.cfg
        interval = cfg.decision_interval_us
        period = cfg.nprach_period_us
        offset = cfg.nprach_offset_us
        opps = cfg.opportunities_per_window
        t_start = self.t_us
        t_end = t_start + interval

        # (6) schedule the action; effective at next NPRACH boundary after delay.
        # Queued (not overwritten) so a delayed action from an earlier window still
        # takes effect at its logged effective time.
        eff = ceil_to_grid(t_start + cfg.signalling_delay_us, period, offset)
        self.pending.append((eff, int(action_id)))
        self.last_scheduled_eff_us = eff

        # Pre-compute NPRACH reservation for the whole window (choices active at
        # each opportunity), so the data budget is known before Msg3 grants.
        opp_times = offset + (self.window_index * opps + np.arange(opps)) * period
        reserved = 0.0
        for ot in opp_times:
            choices = self._choices_at(ot)
            reserved += choices * cfg.preamble_duration_ms * cfg.preamble_repetitions
        self.calendar.reset_window(nprach_reserved_units=reserved)

        # window-local reward accumulators
        c_on = c_total = c_late = m_new = x_new = 0

        for ot in opp_times:
            res = self._run_opportunity(int(ot))
            c_on += res[0]
            c_total += res[1]
            c_late += res[2]
            m_new += res[3]
            x_new += res[4]

        # On the final window, drain completions/deadlines up to the window end
        # (= end_us) so every event is counted in exactly one window's reward.
        terminal = (self.window_index + 1) >= self.n_windows
        if terminal:
            co, ct, cl = self._process_completions(t_end)
            mn, xn = self._process_deadlines(t_end)
            c_on += co
            c_total += ct
            c_late += cl
            m_new += mn
            x_new += xn

        # (8/9) resource + reward
        self.log.nprach_units_sum += self.calendar.rach_reserved
        self.log.data_units_sum += self.calendar.data_used
        self.log.budget_units_sum += self.calendar.total_units
        b_norm = self.ledger.true_pending_count(t_end) / max(self.trace.n_meters, 1)
        r_norm = self.calendar.r_norm()
        rin = RewardInputs(c_on=c_on, c_total=c_total, m_new=m_new, x_new=x_new,
                           b_norm=b_norm, r_norm=r_norm)
        reward = compute_reward(rin, cfg.reward)

        # advance decision boundary
        self.t_us = t_end
        self.window_index += 1

        # (3) release BACKOFF whose timer expired by t_end (so next obs reflects it)
        # (2 already handled inside opportunities)
        if self.check_invariants:
            self.ledger.assert_invariants(t_end)

        # (10) construct the ACTUAL next observation
        feats = self.obs_model.window_features()
        s_next = self.obs_model.encode(feats)
        self._last_state = s_next

        # logging
        pending_now = self.ledger.pending_count(t_end)
        self.log.backlog_series.append((t_end / 1e6, pending_now))
        self.log.reward_series.append(reward)
        self.log.action_series.append(int(action_id))

        terminal = self.window_index >= self.n_windows
        if terminal:
            # completions/deadlines were already drained up to end_us above
            self.done = True

        info = StepInfo(reward_inputs=rin, c_on=c_on, c_total=c_total, c_late=c_late,
                        m_new=m_new, x_new=x_new, b_norm=b_norm, r_norm=r_norm,
                        selected_action=int(action_id), active_pA=self.active_pA,
                        active_backoff_ms=self.active_backoff_ms,
                        active_choices=self.active_choices,
                        pending_effective_us=self.last_scheduled_eff_us,
                        window_index=self.window_index - 1,
                        t_start_us=t_start, t_end_us=t_end)
        return s_next, reward, terminal, info

    # --------------------------------------------------------------- internals
    def _effective_aid_at(self, t_us: int) -> int:
        """Action id in effect at ``t_us`` (pure lookup; does not mutate state).

        The latest-scheduled pending action whose effective time has arrived wins;
        otherwise the currently-active action.
        """
        aid = self.active_aid
        best_eff = -1
        for eff, a in self.pending:
            if eff <= t_us and eff > best_eff:
                best_eff = eff
                aid = a
        return aid

    def _choices_at(self, t_us: int) -> int:
        """Usable subcarrier choices active at time ``t_us`` (honours delay)."""
        return int(self.registry.usable_choices[self._effective_aid_at(t_us)])

    def _promote_pending_if_due(self, now: int) -> None:
        if not self.pending:
            return
        due = [(eff, a) for (eff, a) in self.pending if eff <= now]
        if due:
            # effective times are non-decreasing; the latest due action wins
            _, a_best = max(due, key=lambda x: x[0])
            self._set_active(a_best)
            self.pending = [(eff, a) for (eff, a) in self.pending if eff > now]

    def _run_opportunity(self, now: int):
        """Simulate one NPRACH opportunity at absolute time ``now`` (us).

        Returns ``(c_on, c_total, c_late, m_new, x_new)`` realised this opportunity.
        """
        led = self.ledger
        cfg = self.cfg

        # (6 cont.) promote a scheduled action that is now effective
        self._promote_pending_if_due(now)

        # (8a) finish completions timestamped <= now BEFORE finalising deadlines
        c_on, c_total, c_late = self._process_completions(now)

        # (2) finalise matured deadlines
        m_new, x_new_deadline = self._process_deadlines(now)

        # (3) release BACKOFF requests whose timer expired
        rel = (led.status == BACKOFF) & (led.next_eligible_us <= now)
        if np.any(rel):
            led.status[rel] = WAITING

        # (7) execute the access attempt
        x_new_retry = self._contend(now)

        return c_on, c_total, c_late, m_new, x_new_deadline + x_new_retry

    def _process_completions(self, now: int):
        led = self.ledger
        due = (led.status == ACCESS) & (led.completion_us >= 0) & (led.completion_us <= now)
        if not np.any(due):
            return 0, 0, 0
        idx = np.nonzero(due)[0]
        led.status[idx] = COMPLETED
        self.meter_busy[led.meter_id[idx]] = False
        on_time = led.on_time[idx]
        c_on = int(np.count_nonzero(on_time))
        c_total = idx.size
        c_late = c_total - c_on
        return c_on, c_total, c_late

    def _process_deadlines(self, now: int):
        led = self.ledger
        cfg = self.cfg
        due = ((led.deadline_us <= now) & (~led.miss_recorded)
               & (led.arrival_us <= now))
        if not np.any(due):
            return 0, 0
        # on time (or will complete on time): completion scheduled <= deadline
        will_on_time = (led.completion_us >= 0) & (led.completion_us <= led.deadline_us)
        dropped_before = (led.status == DROPPED) & (led.drop_us < led.deadline_us)
        miss = due & (~will_on_time) & (~dropped_before)
        idx = np.nonzero(miss)[0]
        led.miss_recorded[idx] = True
        m_new = idx.size
        x_new = 0
        if cfg.expiry_as_drop:
            # drop only still-pending WAITING/BACKOFF requests (do not interrupt
            # an in-progress procedure that is finishing late)
            dropable = idx[(led.status[idx] == WAITING) | (led.status[idx] == BACKOFF)]
            if dropable.size:
                led.status[dropable] = DROPPED
                led.drop_reason[dropable] = DROP_EXPIRY
                led.drop_us[dropable] = now
                led.drop_recorded[dropable] = True
                x_new = dropable.size
        return m_new, x_new

    def _eligible_requests(self, now: int) -> np.ndarray:
        led = self.ledger
        cand = ((led.status == WAITING) & (led.arrival_us <= now)
                & (led.next_eligible_us <= now)
                & (~self.meter_busy[led.meter_id]))
        if not np.any(cand):
            return np.empty(0, dtype=np.int64)
        # FIFO per meter among multi-request meters: keep only the head request
        multi = self._multi_meter_mask
        if np.any(multi):
            nonterm = ((led.status == WAITING) | (led.status == BACKOFF)
                       | (led.status == ACCESS))
            nm = nonterm & multi
            head = np.full(self.trace.n_meters, np.iinfo(np.int64).max, dtype=np.int64)
            if np.any(nm):
                np.minimum.at(head, led.meter_id[nm],
                              led.trace.request_id[nm].astype(np.int64))
            blocked = multi & (led.trace.request_id != head[led.meter_id])
            cand = cand & (~blocked)
        return np.nonzero(cand)[0]

    def _contend(self, now: int) -> int:
        """Admission + preamble contention + Level-B stages. Returns drops."""
        led = self.ledger
        cfg = self.cfg
        elig = self._eligible_requests(now)
        n_choices = self._choices_at(now)
        if elig.size == 0:
            # still record an empty observation for the state estimator
            self.obs_model.update_opportunity(0, 0, n_choices, n_choices, self.active_pA)
            self.log.total_opportunities += 1
            return 0

        pA = self._pA_at(now)
        # (admission) Bernoulli draw per eligible request
        u = self.rng["admission"].random(elig.size)
        admitted = elig[u < pA]           # barred requests remain WAITING
        if admitted.size == 0:
            self.obs_model.update_opportunity(0, 0, n_choices, n_choices, pA)
            self.log.total_opportunities += 1
            return 0

        # mark first attempt time + increment attempts for every transmitted preamble
        first = admitted[led.first_attempt_us[admitted] < 0]
        led.first_attempt_us[first] = now
        led.attempts[admitted] += 1
        self.log.total_attempts += admitted.size

        # (contention) uniform subcarrier choice
        choice = self.rng["contention"].integers(0, n_choices, size=admitted.size)
        counts = np.bincount(choice, minlength=n_choices)
        occupied = int(np.count_nonzero(counts >= 1))
        collided_choices = int(np.count_nonzero(counts >= 2))
        idle = n_choices - occupied
        singleton_choice = counts == 1
        success_mask = singleton_choice[choice]       # per admitted request
        success = admitted[success_mask]
        failed = admitted[~success_mask]               # collisions

        self.log.total_opportunities += 1
        self.log.total_occupied_choices += occupied
        self.log.total_collided_choices += collided_choices
        self.log.total_collided_devices += int(failed.size)
        self.log.total_preamble_success += int(success.size)
        self.obs_model.update_opportunity(occupied, collided_choices, idle, n_choices, pA)

        # (Level B) downlink RAR capacity, then Msg3 uplink grant
        rar_fail = np.empty(0, dtype=np.int64)
        msg3_fail = np.empty(0, dtype=np.int64)
        if success.size:
            cap = cfg.rar_grants_per_opportunity
            if success.size > cap:
                order = self.rng["channel"].permutation(success.size)
                granted_rar = success[order[:cap]]
                rar_fail = success[order[cap:]]
            else:
                granted_rar = success
            # Msg3 uplink grant from the shared data budget
            msg3_ok = []
            for r in granted_rar:
                if self.calendar.try_msg3_grant():
                    msg3_ok.append(r)
                else:
                    msg3_fail = np.append(msg3_fail, r)
            msg3_ok = np.asarray(msg3_ok, dtype=np.int64)
            # successful RA procedures -> ACCESS-IN-PROGRESS, completes after delay
            if msg3_ok.size:
                completion = now + cfg.access_stage_delay_us
                led.status[msg3_ok] = ACCESS
                led.completion_us[msg3_ok] = completion
                led.on_time[msg3_ok] = completion <= led.deadline_us[msg3_ok]
                self.meter_busy[led.meter_id[msg3_ok]] = True

        self.log.total_rar_fail += int(rar_fail.size)
        self.log.total_msg3_fail += int(msg3_fail.size)

        # all failed attempts (collision, no RAR, no Msg3) -> retry or drop
        all_failed = np.concatenate([failed, rar_fail, msg3_fail])
        if all_failed.size:
            return self._handle_failed(all_failed, now)
        return 0

    def _handle_failed(self, reqs: np.ndarray, now: int) -> int:
        led = self.ledger
        cfg = self.cfg
        over = led.attempts[reqs] >= cfg.retry_limit
        drop = reqs[over]
        retry = reqs[~over]
        x_new = 0
        if drop.size:
            led.status[drop] = DROPPED
            led.drop_reason[drop] = DROP_RETRY_LIMIT
            led.drop_us[drop] = now
            led.drop_recorded[drop] = True
            x_new = drop.size
        if retry.size:
            led.status[retry] = BACKOFF
            if self.active_backoff_us > 0:
                draw = self.rng["backoff"].integers(0, self.active_backoff_us + 1,
                                                     size=retry.size)
            else:
                draw = np.zeros(retry.size, dtype=np.int64)
            led.next_eligible_us[retry] = now + draw
        return x_new

    def _pA_at(self, now: int) -> float:
        return float(self.registry.pA[self._effective_aid_at(now)])

    # ------------------------------------------------------------ convenience
    def run_episode(self, policy, train: bool = False):
        """Run a full episode with a policy object exposing
        ``select_action(state, feasible)`` and (optionally) ``update(...)``.
        Returns the accumulated reward and the final ledger."""
        s = self.reset()
        total_r = 0.0
        while not self.done:
            a = policy.select_action(s, self.feasible_actions, train=train)
            s_next, r, terminal, info = self.step(a)
            if train and hasattr(policy, "update"):
                policy.update(s, a, r, s_next, terminal, self.feasible_actions)
            s = s_next
            total_r += r
        return total_r, self.ledger
