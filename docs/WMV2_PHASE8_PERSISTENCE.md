# WMv2 Phase 8 — Production Persistence & Longitudinal State

*Audit · architecture · execution path · state definitions · production design · limitations · forensic traces.*

This is one of the **three** Phase-8 documents (with `WMV2_PHASE8_VALIDATION.md` and
`WMV2_PHASE8_MIGRATION.md`). It covers what Phase 8 *is* and how it executes. Numbers, splits, and grading
live in the validation doc; schema/compat/integration live in the migration doc.

Phase 8 answers one question with executable code: **does the system genuinely remember and learn from
history, such that deleting or altering history measurably changes execution where history is causally
relevant?** It is not enough for values to remain stored. The whole phase is graded on the causal ablation.

---

## Part 0 — Ruthless audit of the pre-Phase-8 persistence surface

Audited the *actual code*, not the intended architecture. Findings per family (status ∈ {production-executable,
executable-unvalidated, executable-toy, partial, point-estimate-only, storage-only, benchmark-specific,
arbitrary, absent}):

| Family | Where it lived pre-8 | Posterior? | Updated by events? | StateDelta? | Survives restart? | Changes behavior? | Verdict |
|---|---|---|---|---|---|---|---|
| engagement / momentum | `reference/omnibehavior.py`, `wmv2_persistence_power.py` | point | inline formula | ✗ | ✗ (cache only) | ✗ (bypass predictor) | **benchmark-specific**, not in shared world |
| typed event history | `nonlinear/history.py` (`p7_*` logs) | ✗ | append-only | ✗ | ✗ (in-process) | features only | partial (features, no posterior, no durability) |
| rolling entity features | `swm/state/entity_history.py` | ✗ | streaming as-of | ✗ | ✗ | features only | partial |
| episodic memory | `swm/memory/*` | Beta-Binomial | leakage-safe retrieval | ✗ | ✗ | response-fn only | executable-unvalidated for WMv2 world |
| recipient/dyadic | `swm/decision/recipient_history.py` | point | as-of | ✗ | ✗ | decision-fn only | partial |
| actor-policy adaptation | `phase4_execution.apply_adaptation` | point | TD update | ✓ (in-world) | ✗ (single run) | ✓ | **explicitly documented as NOT the cross-run service** |
| trust / relationship | `transitions.RelationshipUpdateOperator` | point | in-world | ✓ | ✗ | ✓ | symmetric only, no violation/repair path-dependence |
| beliefs / resources | `transitions.*` | point/dist | in-world | ✓ | ✗ | ✓ | no sequential filter, no history |
| commitment / reputation / institutional-stage / habit / risk / learned-strategy | — | — | — | — | — | — | **absent as persistent families** |

**The two load-bearing gaps.**
1. **No cross-run persistence service.** Everything above is in-process or cache-file. `WMV2_PHASE4_ACTOR_POLICY.md`
   line 156 states `apply_adaptation` "is not a substitute for Phase 8's future cross-run persistence
   service." That service did not exist.
2. **The persistence WIN never touched a WorldState.** The `n=7074` OmniBehavior result
   (`wmv2_persistence_power.json`) was computed by a standalone longitudinal predictor
   (`A_persist = user_rate × (p_hot/p_cold)^(momentum−0.5)`, zero LLM, no WorldState). It is a real signal
   but it bypassed the shared architecture — exactly the anti-pattern the phase forbids.

Phase 8 closes both: a durable event-sourced service, and the winning signal re-derived by a **sequential
filter, materialized into WorldState, consumed by the existing ActorView→policy path**.

---

## Part 1 — Architecture (five planes, all present)

```
 real event histories                                    ┌─ CODE ─────────────────────────────────────┐
 (OmniBehavior / Enron / congress / Phase-2 evidence)    │ phase8_persistence  typed specs + contracts │
        │                                                │ phase8_events       immutable event log     │
        ▼                                                │ phase8_filtering    sequential inference     │
 [EVIDENCE plane]  phase8_events.EventLog                │ phase8_transitions  in-world transitions     │
   append-only · content-hash dedup · idempotent ·       │ phase8_service      cross-run store          │
   deterministic order · hash-verifiable · JSONL-durable │ phase8_materialize  WorldState bridge        │
   · corrections/retractions/identity relinks            │ phase8_pipeline     universal entry + abln   │
        │  events_as_of(t, filter|smooth)  (leakage gate)└────────────────────────────────────────────┘
        ▼
 [POSTERIOR plane]  phase8_filtering  (carries the prior forward, per event)
   DecayedBetaBernoulli (momentum) · AsymmetricTrust (gain≪loss + repair) · GaussianKalman ·
   forward-HMM stage · particle SMC   →  PersistentStatePosterior (+ lineage, ESS, resample)
        │  phase8_service.PersistentStore.replay(as_of)  →  checkpoint (integrity hash + lineage)
        ▼
 [WORLD-STATE plane]  phase8_materialize.materialize_persistent_state
   writes each posterior into the field its spec declares:
     engagement_propensity → entity.latent_state[phase4_policy_value:engage]
     trust                 → network.edge.trust     resource_level → entity.resources
     habit_strength        → entity.past_actions     reputation/risk → entity.beliefs[...]
     institutional_stage   → entity.latent_state[institutional_stage]
        │  each write emits an explicit PersistentStateDelta (no silent mutation)
        ▼
 [EXECUTION plane]  the EXISTING ActorViewBuilder → ActorPolicyModel → rollout
   habit family reads action_history · reinforcement family reads policy_state Q · reciprocity reads edge
   trust · feasibility reads resources/commitments/stage · utility reads beliefs
        →  changed action distribution · StateDelta · terminal readout
        │  phase8_transitions closes the loop: action/outcome → persistent update → next decision
        ▼
 SimulationResult   (no-abstention: weak history widens uncertainty + lowers grade, never blocks)
```

The design invariant that makes the causal ablation real: **Phase 8 never adds a new decision path.** It
writes into the exact WorldState fields the Phase-4 policy already reads. Remove the history → the filtered
posterior reverts to its prior → the materialized field changes → the `ActorView` projection changes → the
`ActorPolicyModel` action distribution changes → the terminal readout changes. Verified end-to-end in
`test_history_removal_changes_action_distribution` and `wmv2_phase8_ablations.py`.

---

## Part 2 — The execution path (universal entry)

`phase8_pipeline.simulate_with_persistence(question, as_of, horizon, context, actor_history, …)`:

```
NL question → compile_world → WorldExecutionPlan → build_world (WorldState)
  → ingest actor_history into the durable EventLog (leakage-safe, observed_time ≤ as_of)
  → PersistentStore.replay(as_of)  → sequential-filter posteriors
  → materialize_persistent_state(world, posteriors)  → WorldState fields + PersistentStateDelta
  → standard rollout (existing operators + phase8 persistence operators)  → terminal readout
  → SimulationResult (support grade lowered when history is thin; never abstains)
```

Live end-to-end run (`experiments/results/phase8/universal_path_trace.json`): the compiled `Alice` entity
received 6 ingested engagement events, filtered to **1 materialized persistent posterior** consumed by the
rollout; status `completed_with_degradation` (no-abstention preserved). The actor-decision path
(`materialize_and_decide`) drives the **real** `ActorViewBuilder` + `ActorPolicyModel` — no bypass.

---

## Part 3 — Persistent variable specifications (Part 1 of the mandate)

Every persistent variable is a typed `PersistentVariableSpec` (not a bare scalar). It declares semantics,
scope, support, posterior family, transition family + parameter source, observation model, causal
parents/children, evidence dependencies, update triggers, reset conditions, timescale, actor visibility,
memory accessibility, terminal sensitivity, identifiability, provenance, **and — enforced at registration —
the WorldState field it `materializes_into` plus the `consumed_by` mechanism**. A spec with no
materialization target or no consumer is *refused* (`test_ornamental_variable_is_refused`): storage that no
mechanism reads is not persistence.

Nine canonical families ship (all non-ornamental):

| variable | scope | posterior | transition family | materializes_into | consumed_by |
|---|---|---|---|---|---|
| `engagement_propensity` | actor | beta_bernoulli | reinforcement (forgetting) | `latent_state[phase4_policy_value:engage]` | reinforcement/habit policy + readout |
| `habit_strength` | actor | dirichlet_count | habit_accumulation | `past_actions` | habit policy family |
| `trust` | dyad | beta_bernoulli | trust_asymmetric | `network.edge.trust` | reciprocity policy + relationship_update |
| `relationship_strength` | relationship | beta_bernoulli | relationship_strengthen | `network.edge.strength` | social_proof policy |
| `reputation` | actor | beta_bernoulli | reputation_accrual | `beliefs[reputation]` | limited-depth policy |
| `commitment` | actor | categorical_stage | commitment_create/fulfill/violate | `commitments` | feasibility + obligation policy |
| `resource_level` | actor | gaussian_state | resource_flow | `resources` | feasibility + resource_update |
| `risk_tolerance` | actor | gaussian_state | risk_adaptation | `beliefs[risk_tolerance]` | risk_sensitive policy |
| `institutional_stage` | institution | categorical_stage | institutional_stage (HMM) | `latent_state[institutional_stage]` | feasibility + institutional_vote |

Path-dependence is first-class: trust records a `violation_count` (repaired-to-0.5 ≠ never-fell);
institutional stage records its `path` (reached-via-appeal ≠ reached-directly). See
`test_trust_repair_is_path_dependent`, `test_categorical_stage_records_appeal_path_dependence`.

---

## Part 4 — Event-sourced history & sequential inference

**The event log is the causal source of truth; a checkpoint is a derived artifact.** `PersistentEvent`
carries `event_time` (when it happened), `observed_time` (when it became knowable — the *filtering* gate),
`availability_time` (public visibility), source/evidence hashes, `identity_link_uncertainty` (probabilistic
attribution), parents, kind, and a content-hash `event_id`. The log is append-only, deduped by content id
(idempotent, retry-safe), deterministically ordered by `(event_time, seq, event_id)`, hash-chained
(tamper-evident), and JSONL-durable (survives process restart — the cross-run gap). Corrections, retractions,
and identity relinks are **new events** that supersede in the *effective* view; the original is never
overwritten.

Leakage control is explicit and separated: `events_as_of(t, mode="filter")` returns `observed_time ≤ t`
(production/forecasting — only what was knowable); `mode="smooth"` returns `event_time ≤ t` (retrospective
analysis only). A query without `as_of` is refused.

Sequential inference **carries the prior posterior forward** (it does not re-infer each step). The momentum
filter is a conjugate Beta-Bernoulli with exponential forgetting toward a hierarchical per-actor anchor:
after a run of 1s the posterior climbs above the anchor, after 0s it falls below — the winning persistence
signal as a proper filter, with `decay=1` recovering a memoryless accumulator. Every step records
prior→observation→posterior, log-likelihood, ESS, and resample decision.

---

## Part 5 — Production design

* **Durability / cross-run.** `PersistentStore` = `EventLog` + filter registry. State is produced *only* by
  `replay(as_of)`; nothing is hand-set. Checkpoints snapshot derived posteriors + event watermark + code/
  schema versions + lineage + an integrity hash. Deterministic restore, corruption detection (integrity
  mismatch → `CorruptionError`), typed schema migration (`MigrationError` on incompatible), rollback (a read
  of an earlier derived state — the log stays complete), and `compare` for diffs.
* **Determinism.** identical (log, as_of, seed) ⇒ identical checkpoint + integrity hash
  (`test_checkpoint_deterministic_replay_parity`).
* **Performance** (`wmv2_phase8_ablations.performance_bench`, 5000 events / 50 actors): ingest **77k
  events/s**, replay **3.3 ms/actor**, checkpoint save 0.12 s / restore 0.024 s, deterministic parity ✓,
  integrity ✓. Bounded, actor-scoped, time-bounded retrieval — a multi-year history is never loaded into a
  prompt.
* **No-abstention.** Thin/absent/uncertain history widens uncertainty and lowers the support grade; it never
  blocks. Only engineering failures raise (typed `execution_failed`).
* **LLM contract.** The LLM may interpret event text, propose event types / candidate variables / structural
  history hypotheses, and summarize old memories. It may **not** mint any persistent value, transition rate,
  decay/half-life, trust coefficient, identity-link probability, or action/outcome probability — those come
  from observed history, fitted longitudinal models, reference packs, hierarchical priors, or explicit broad
  uncertainty.

---

## Part 6 — Forensic traces (paired full-history vs history-removed)

Two human-readable traces (full machine-readable set: `experiments/results/phase8/forensic_traces.json`).
Both run the identical actor through the **real** `ActorViewBuilder` + `ActorPolicyModel` under two histories.

**Trace A — `user_164`, hot recent history (y_true = 1, 128 prior events, 4/5 recent acted, user anchor 0.740).**
```
full_history:    filter → engagement_propensity posterior 0.8010  (climbs above the 0.740 anchor via momentum)
                 materialized → latent_state[phase4_policy_value:engage] = 0.8010  (PersistentStateDelta emitted)
                 ActorView.policy_state reads it → action P(engage)=0.5409, P(wait)=0.4591
history_removed: filter([]) → posterior 0.7398 (== anchor, no momentum)
                 materialized 0.7398 → action P(engage)=0.5378
Δ posterior = +0.0612   ·   action distribution shifts   ·   view hash changes → history_changed_trajectory=True
```
The recent hot streak pushes the propensity **above** the persistent user level; removing it collapses the
propensity back onto the anchor and shifts the action distribution. History is causally consumed.

**Trace B — `user_010`, cold recent history (y_true = 0, 25 prior events, recent5=3 but anchor 0.310).**
```
full_history:    posterior 0.3362   →   materialized 0.3362   →   P(engage)=0.5173
history_removed: posterior 0.3105 (== anchor)   →   P(engage)=0.5160
Δ posterior = +0.0257   ·   view hash changes → history_changed_trajectory=True
```
Even a modest recent signal moves the materialized field and the projected view. The action-distribution
delta is smaller here because the blended policy mixes reinforcement with utility/random-utility families —
an honest reflection of magnitude, not a manufactured effect.

---

## Part 8 — Production completion (canonical integration, storage, runtime status)

A focused completion pass closed the remaining architectural/operational gaps so persistence is part of the
**canonical** production path — not a separate entry point.

**8.1 Canonical pipeline integration.** `pipeline.simulate(question, …, persistence=ctx, actor_history=…)`
is now the one product entry. When history or a compatible checkpoint exists it automatically: resolves
world/scenario/actor identity → loads the prior checkpoint + event-log watermark → ingests leakage-safe new
history → replays sequential filters → **selects causally-relevant families** → materializes into the
standard `WorldState` → exposes only actor-visible memory → runs the ordinary policy/mechanism rollout →
persists feedback → commits a new versioned checkpoint → returns lineage/versions/support-effect/limitations
in the standard `SimulationResult`. With no history it initializes broad priors and continues (no abstention).
`simulate_with_persistence` is retained as a thin compile-first wrapper over the shared `run_with_persistence`
— there is exactly one canonical execution path. Verified live across a process restart through SQLite.

**8.2 Default use of all causally-relevant families (`phase8_runtime`).** Every implemented family may be
selected and executed **by default** when causally relevant. `select_families` blocks ONLY `quarantined`/
`incompatible`; `exploratory`/`highly_speculative` families execute with broader uncertainty, a lower support
grade, and a sensitivity contribution — never behind an experimental flag. The distinction is *validated →
stronger support* vs *experimental → broader uncertainty*, not *usable vs disabled*. A load-bearing
experimental family lowers the overall support grade (`support_grade_effect`) and widens terminal
uncertainty. Uncertain transitions drop to the strongest available fallback tier (1 fitted → 7 competing
hypotheses) and are **never removed** (`resolve_transition_tier`).

**8.3 Runtime support + provenance enforcement.** Each `PersistentVariableSpec` carries an executable
`runtime_status` (validated at registration) + `supporting_evidence` + `transport_risk`.
`family_runtime_manifest` exposes, per used family: id, version, status, transition family, parameter source,
supporting dataset/paper, posterior uncertainty (mean+sd+ESS — never a silent point estimate), applicability,
transport risk, sensitivity contribution, affected state variables, downstream consumers, and limitations.

**8.4 Transactional storage (`phase8_storage`).** A typed `PersistentStorageBackend` with two
implementations: `JsonlBackend` (portable/testing) and `SqliteBackend` (**production**, WAL journal). SQLite
gives idempotent atomic event append (INSERT-OR-IGNORE on the content-hash PK), transactional atomic
checkpoint commit + rollback, multi-process writers + concurrent readers, automatic crash recovery (WAL),
integrity verification (watermark chain + event↔checkpoint consistency), compaction (bounded checkpoint
growth; events never pruned), and deterministic replay. `EventLog(backend=SqliteBackend(...))` swaps it in
with no other change. Throughput is lower than in-memory (durable fsync per event) — the honest cost of
transactional durability.

**8.5 Canonical identity resolution (`phase8_identity`).** Stable world/scenario/actor/dyad/edge ids
(deterministic, no wall clock); aliases, merges, institution renames, role-at-time, and **probabilistic
linkage** (several weighted hypotheses when uncertain — history is never forced onto one actor with false
certainty).

**8.6 Actor-visible memory in the canonical view.** `_expose_actor_memory` materializes only
recency/salience-weighted, strictly-before-`as_of` recalled traces into `entity.memory` (via the episodic
store) — the omniscient log keeps everything, the actor gets probabilistic, non-perfect recall.

---

## Part 7 — Limitations (honest)

* **Production-usable ≠ empirically validated.** All nine families are **production-usable and selected by
  default** (none quarantined/incompatible). But held-out validation is uneven: `engagement_propensity`
  (n=7074) and `habit_strength` (n=7290) have adequately-powered wins; `resource_level` is structurally
  sound (accounting-driven) but has no held-out dataset; `trust`/`relationship_strength`/`commitment`/
  `institutional_stage` are exploratory (Track B weak, Track C null, or no dataset); `reputation`/
  `risk_tolerance` are highly_speculative (broad priors, no dataset). Experimental families still execute —
  with broader uncertainty and a lower support grade — they are NOT disabled. See the validation doc's
  four-status grading and final family table.
* **Dyadic persistence is weak on Enron.** Track B beats the frequency baseline but not the base-rate
  baseline (AUROC ≈ 0.50); the 2001 collapse is a regime change that washes out dyadic momentum.
* **Institutional pass-persistence is a null.** Track C is not detectable vs the base rate and is
  underpowered — pass/fail is bill-driven, not chamber momentum.
* **The action-distribution effect is modest** even when the materialized field moves a lot, because the
  production policy is a blended mixture of families; the clean signal is in the materialized field / readout.
* **Transition parameters are reference-pack / broad priors** for the non-engagement families (asymmetric
  trust gain/loss, memory half-life, reputation rates). They are labeled as such and are *not* claimed
  empirically supported.
* **Hierarchical shrinkage is near-ornamental** for the engagement Brier on this cohort (ablation Δ = +1e-5):
  most users have enough events that shrinkage barely moves the anchor. Preserved as an honest finding.
* **Checkpoint size** grows with retained lineage (~2.7 MB / 50 actors); production would trim lineage tails
  or store deltas.
