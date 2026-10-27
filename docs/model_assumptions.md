# Model assumptions and scope of claims

This simulator is an **NB-IoT-inspired, system-level model evaluated under the
stated assumptions below**. It is **not** a standards-compliant implementation.
Every parameter is tagged:

- `verified_value` — grounded in a cited 3GPP value *as a candidate value*
  (never as proof that a whole configuration or dynamic reconfiguration schedule
  is legal);
- `study_assumption` — a declared modelling choice, tested for sensitivity where
  relevant;
- `to_be_mapped` — a quantity that a standards-faithful mode would need to map to
  a specific release/clause before any compliance claim.

## 1. Service endpoint (what "completion" means)

The implemented level is **Level B** (plan §2): a request is `COMPLETED` only
after the represented random-access procedure —
**preamble success → RAR grant (downlink capacity) → Msg3 grant (uplink
resource) → contention-resolution delay**. Distinctions enforced in code and
metrics:

- a **singleton preamble outcome** is a *preamble success*, not completion
  (`environment._contend`, `metrics.preamble_success_prob`);
- **random-access completion** is the Level-B endpoint above (`status COMPLETED`);
- **application-message delivery** is **Level C and is NOT implemented**. No
  application-delivery or utility-SLA claim is made.

## 2. Parameter table

| Parameter | Value | Tag | Notes / sensitivity |
|---|---|---|---|
| Cell / carrier | 1 cell, 1×180 kHz uplink | study_assumption | no hidden parallel carriers |
| Release family | R15 FDD legacy NPRACH (candidate) | to_be_mapped | values are candidates, not a compliance map |
| Coverage | one good-coverage group, repetitions = 1 | study_assumption | deep-coverage repetition scenarios not validated |
| Population N | 10k / 20k / 30k / 50k / 75k / 100k | study_assumption | stress-test settings, not a capacity promise |
| Reporting period | 900 s | study_assumption | test 300 / 1800 s separately |
| Reporting phase | uniform in [0, period), jitter 0 | study_assumption | synchronised/jittered phases are stress tests |
| Event onset / fraction / spread | 30 s / 5% / 10 s | study_assumption | fractions 1/5/20/100% supported |
| Event burst profile | Beta(3,4) (CDF 20u³−45u⁴+36u⁵−10u⁶) | study_assumption | one scenario only; uniform/simultaneous supported |
| Periodic deadline | 60 s after generation | study_assumption | test 30 / 120 s |
| Event deadline | 30 s after generation | study_assumption | test 10 / 60 s; pending a real utility requirement |
| Arrival window / horizon | [0,120) s / 180 s | study_assumption | H chosen so every cohort deadline matures |
| NPRACH period | 80 ms | verified_value (candidate) | TS 36.331 lists 40/80/160 ms FDD; test 40/160 |
| NPRACH start offset | 8 ms | verified_value (candidate) | TS 36.331 lists 8 ms among offsets |
| Configured subcarriers | 12 / 24 / 36 / 48 | verified_value (candidate) | TS 36.331 values; ≠ guaranteed usable CBRA choices |
| Usable CBRA choices | = configured subcarriers | study_assumption | CBRA start-subcarrier mapping abstracted |
| Preamble repetitions | 1 | study_assumption | legal repetition settings + costs not swept here |
| Preamble duration | 5.6 ms (format-0-like) | study_assumption | used only for the resource calendar |
| Retry limit | 10 total preamble tx | study_assumption | explicit study rule; test 5 / 20 |
| Per-meter concurrency | 1 active procedure, FIFO | study_assumption | no silent coalescing; distinct request IDs |
| Queue capacity | unbounded (reference) | study_assumption | finite-capacity overflow = explicit drop |
| Deadline-expiry behaviour | record miss, keep serving | study_assumption | expiry-as-drop is a separate scenario |
| Channel | ideal reception / ideal detection | study_assumption | failure/detection models not added |
| RAR grants / opportunity | 24 | study_assumption | downlink response capacity (abstract) |
| Msg3 resource units / grant | 6 (of 3.75 kHz×1 ms) | study_assumption | uplink NPUSCH cost (abstract) |
| Resource unit | 3.75 kHz × 1 ms | study_assumption | accounting calendar unit |
| R_total / window | 48 subcarriers × 640 ms = 30720 units | study_assumption | shared budget; R_RACH+R_DATA ≤ R_total |
| Decision interval | 640 ms (8 × 80 ms) | study_assumption | plan §4 abstract reference |
| Action signalling delay | 0 ms (base, = next NPRACH boundary) | study_assumption | delay sweep 640 / 2560 ms |
| EWMA smoothing α | 0.2 | study_assumption | validate 0.1 / 0.4 |
| State bins | 5 (cong) × 5 (load) × 6 (trend) = 150 | study_assumption | bin edges frozen before test (config.py) |
| Reward weights | w_on=1, w_miss=1, w_drop=1, w_q=0.01, w_r=0.01 | study_assumption | tune on validation only; K=384 fixed |
| γ / α schedule / ε schedule | 0.99 / (1+visits)^-0.6 / 1.0→0.05 over 80% | study_assumption | plan §9 defaults |

TS 36.331 supports 40/80/160 ms NPRACH periodicities, 8 ms start offsets and
12/24/36/48 configured subcarriers **as candidate values**; this does not make
any particular full configuration, admission mechanism, or dynamic
reconfiguration schedule standards-compliant.

## 3. Observation model (ideal-observation assumption)

The controller's idle/occupied/collision classification assumes **perfect
preamble detection** at the eNB. This is an ideal-observation assumption, not a
proven NB-IoT receiver capability. The admission probability is an **abstract
admission controller**; it is **not** the R15 ACB/MIB/SIB14-NB signalling
procedure and must not be read as LTE ACB-factor semantics.

The compact state is a **quantised observation of a partially observed
environment**; a finite table does not establish the Markov property. Rich and
oracle states exist for ablation (`config.state_kind`); oracle features are
labelled unavailable ground truth and are never a deployable input.

## 4. Scope of claims (what we do NOT claim)

Until the corresponding checks are complete, the method is described as a
**candidate controller evaluated under stated simulation assumptions**:

- **"Standards-compliant"** — requires a release-specific action/signalling map
  (`standards_status` is `abstract` for every action). Not claimed.
- **"Real-time"** — requires measured latency on a stated platform. We report a
  measured decision latency on this machine only (`controller_cost.csv`); no
  real-time guarantee.
- **"Lightweight"** — we report measured Q-table memory and decision latency; the
  word itself is not asserted as a property.
- **"Energy-efficient"** — **no physical power/time model is implemented.** We do
  not report energy; only attempt/repetition counts under their own names.
- **"SLA guarantee"** — the deadline-aware reward is a *soft* objective; a zero
  observed miss rate is an empirical sample outcome, not a guarantee.
- **"Outperforms DRL"** — **no DQN/DDQN/PPO is implemented**; no deep-RL
  comparison or superiority is claimed (no proxy curves exist, by design).

## 5. Pilot vs full experiment

Results produced by `run_pilot` with the default small budget are a **pilot**
(≈150 episodes, 2 training seeds, 5 test traces), intended to demonstrate the
pipeline and the honest metric picture — **not** the full research experiment.
A full run (≥2000 episodes, ≥5 training seeds, more test traces, reward/state
ablations, held-out generalisation scenarios) is launched with the same scripts
by scaling the flags, and its scope/uncertainty must be reported accordingly.

## 6. What remains untested / open

- Standards-faithful admission/BI/NPRACH signalling and timing map (Level of
  compliance = none; abstract only).
- Level C application-delivery stages (data queue, packet size, scheduling,
  retransmissions).
- Channel failure/detection and deep-coverage repetition models.
- A genuinely trained DQN baseline (B6) — intentionally absent.
- Full reward-weight and state-design sensitivity sweeps at the full training
  budget; the pilot does not establish convergence.
