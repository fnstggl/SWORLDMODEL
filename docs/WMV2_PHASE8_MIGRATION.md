# WMv2 Phase 8 — Migration & Integration Contract

*Schema changes · compatibility · checkpoint migration · Phase-4 (and parallel-branch) integration contract ·
reproducibility commands.*

Companion to `WMV2_PHASE8_PERSISTENCE.md` (architecture) and `WMV2_PHASE8_VALIDATION.md` (results). This doc
is for anyone building on Phase 8 or developing a parallel branch (Phase 4 actor-policy learning, Phase 11/12).

---

## 1 — Schema changes (additive, backwards-compatible)

Phase 8 is **purely additive**. It does **not** modify `contracts.py`, `compiler.py`,
`WorldExecutionPlan`, `WorldState`, `materialize.py`, `transitions.StateDelta`, or the rollout loop. This was
deliberate to avoid conflicts with parallel Phase 4/11/12 work.

New modules (all under `swm/world_model_v2/`):

| module | owns |
|---|---|
| `phase8_persistence` | `PersistentVariableSpec` registry, `PersistentStateKey/Posterior/View/Delta/Lineage`, `MemoryTrace`, Phase-4 contract |
| `phase8_events` | `PersistentEvent`, `EventLog` (durable, event-sourced) |
| `phase8_filtering` | sequential filters (Beta-Bernoulli / asymmetric-trust / Kalman / stage-HMM / particle) |
| `phase8_transitions` | in-world persistence operators (registered in the **canonical** operator registry) |
| `phase8_service` | `PersistentStore`, `PersistentCheckpoint`, migration/verify/restore/rollback/compare |
| `phase8_materialize` | posterior → WorldState field writer + `HistoryIngestor` |
| `phase8_pipeline` | `simulate_with_persistence`, `materialize_and_decide`, `run_history_ablation` |

**How persistent state reaches WorldState without a schema change:** it writes into fields the universal
schema already has — `entity.latent_state[...]` (the typed extension namespace), `entity.beliefs[...]`,
`entity.resources`, `entity.past_actions`, `entity.commitments`, and `network.edge.trust`/`.strength`. No new
`ENTITY_FIELDS` entry is required. The two in-world operators register through the existing
`transitions.register_operator` — no competing registry.

**Versioning.** `phase8-persistence-1.0` (contracts/specs), `phase8-events-1.0` + `ingestion_version 1.0`
(event log), `phase8-store-1.0` (checkpoints). Each object carries its `schema_version`.

---

## 2 — Compatibility

* **Old runs are unaffected.** Nothing in the pre-Phase-8 pipeline imports Phase 8; `pipeline.simulate()` is
  untouched and still functional. Phase 8 is opt-in via `simulate_with_persistence` or by registering the
  persistence operators on a rollout.
* **Old artifacts remain readable.** The prior persistence results (`wmv2_persistence_power.json`,
  `wmv2_omnibehavior_v2.json`, `omnibehavior_eval.json`) are preserved untouched; Phase 8 writes new files
  under `experiments/results/phase8/`.
* **Event kinds are a closed set** (`observation`/`correction`/`retraction`/`identity_link_change`/
  `revised_observation`/`provenance_update`/`policy_feedback`); an unknown kind is refused at construction.
* **Full test suite**: 28 Phase-8 tests pass; the only repo-wide failure is the pre-existing environmental
  `fastapi`-missing case in `test_state_world_model.py`, unrelated to Phase 8 (it does not import
  `world_model_v2` persistence). No regressions introduced.

---

## 3 — Checkpoint migration

A checkpoint is a *derived* artifact — the event log is the source of truth, so the safest "migration" is
always **replay from the log**. When only the checkpoint is available:

```python
store = PersistentStore(world_id, scenario_id)
cp   = PersistentCheckpoint.from_dict(json.load(open(path)))
store.verify(cp)      # CorruptionError if integrity hash mismatches; MigrationError if schema incompatible
cp   = store.migrate(cp)          # upgrades a known older 1.x schema; refuses an incompatible major version
posteriors = store.restore(cp)    # deterministic reconstruction
```

Guarantees:
* **Corruption** (tampered posterior/watermark) → `CorruptionError`, never a silent load
  (`test_corruption_is_detected`).
* **Incompatible schema** → typed `MigrationError`, never garbage (`test_incompatible_schema_raises_migration_error`).
* **Determinism** — identical (log, as_of, seed) reproduces the checkpoint + integrity hash bit-for-bit
  (`test_checkpoint_deterministic_replay_parity`); restore round-trips through disk
  (`test_checkpoint_restore_roundtrip`).
* **Rollback** returns an earlier derived state as a *read* — the immutable log is never truncated.

---

## 4 — Integration contract for Phase 4 (and parallel branches)

Phase 8 exposes a **stable, typed boundary** so Phase-4 actor-policy learning can read persistent posteriors
and emit outcomes back **without either branch rewriting the other's code**. Names map to existing WMv2
equivalents where they exist.

### 4.1 What Phase 4 READS

`phase8_persistence.persistent_features_for_policy(view: PersistentStateView) -> dict` returns a flat, typed
feature dict from an actor's **visible** persistent state (belief / trust / habit / reputation / resource /
risk / stage + per-feature uncertainty + provenance). Phase 4 never touches the event log or another actor's
private posteriors — it reads only this projection. It is additive: new persistent variables surface here
without changing Phase 4's signature.

In practice the materialized fields *already* flow into the Phase-4 `ActorView` with no new code, because
Phase 8 writes into the exact fields `ActorViewBuilder` projects:

| persistent variable | WorldState field | ActorView field | Phase-4 policy family that reads it |
|---|---|---|---|
| engagement_propensity | `latent_state[phase4_policy_value:engage]` | `policy_state` | `reinforcement_learning` / `ewa` |
| habit_strength | `past_actions` | `action_history` | `habit` |
| trust | `network.edge.trust` | `relationships[].trust` | `reciprocity` / `social_proof` |
| resource_level | `resources` | `resources` | feasibility |
| commitment | `commitments` | `commitments` | feasibility / `obligation` |
| reputation / risk_tolerance | `beliefs[...]` | `beliefs` | `limited_depth_reasoning` / `risk_sensitive` |
| institutional_stage | `latent_state[institutional_stage]` | `policy_state` / feasibility | feasibility gate |

### 4.2 What Phase 4 EMITS back

`phase8_persistence.PolicyFeedbackEvent(at, actor_id, action_name, outcome, reward, target_id, magnitude, …)`
with `outcome ∈ {reward, failure, sanction, promise_fulfilled, promise_violated, response_received,
institutional_decision, resource_change, trust_event, relationship_event}`. Phase 8 converts it into a
`PersistentEvent` (`kind="policy_feedback"`) appended to the log; the next `replay` sees the updated history,
so the next decision reflects it — the closed loop. The in-world `PersistenceUpdateOperator` applies the same
update live during a rollout (reinforcement of the engagement Q-value, asymmetric trust, reputation accrual).

### 4.3 Stable objects the contract guarantees

`PersistentStateKey`, `PersistentVariableSpec`, `PersistentStatePosterior`, `PersistentStateView`,
`PersistentEvent` (the `PersistentObservation`/`PersistentUpdateEvent` equivalent), `PersistentStateDelta`,
`HistoryWindow` (reused from `nonlinear/history.py`), `MemoryTrace`, `PersistentCheckpoint`,
`PersistentLineage`, `persistent_features_for_policy`, `PolicyFeedbackEvent`. Field additions are additive;
`schema_version` gates breaking changes.

### 4.4 Rules for parallel branches

* Do **not** import Phase-4-owned policy internals into Phase 8 (the contract is the only coupling).
* Do **not** depend on unmerged branches.
* If a shared file (`state.py`, `transitions.py`) ever must change, keep it minimal, typed,
  backwards-compatible, tested, and documented here. Phase 8 changed **none** of them.

---

## 4b — Production-completion additions (canonical path, storage, runtime status)

**Canonical usage (one entry point).** Persistence is now part of the ordinary `pipeline.simulate`:

```python
from swm.world_model_v2.pipeline import simulate
from swm.world_model_v2.phase8_pipeline import PersistenceContext
from swm.world_model_v2.phase8_service import PersistentStore
from swm.world_model_v2.phase8_events import EventLog
from swm.world_model_v2.phase8_storage import SqliteBackend

store = PersistentStore(world_id, scenario_id, log=EventLog(world_id, scenario_id,
                        backend=SqliteBackend("state.db")))
store.register_filter("engagement_propensity", <builder>)
res = simulate(question, llm=llm, as_of=..., horizon=...,
               persistence=PersistenceContext(store=store),
               actor_history={actor_id: [<events>]})     # loads checkpoint, replays, materializes, re-checkpoints
```
No `actor_history`/`persistence` ⇒ behaviour is byte-identical to before (no regression).
`simulate_with_persistence` remains as a compile-first wrapper over the shared `run_with_persistence`.

**Storage backends (`phase8_storage`).** `EventLog(path=...)` ⇒ `JsonlBackend` (portable/testing, unchanged
behaviour). `EventLog(backend=SqliteBackend(db))` ⇒ production WAL backend: atomic idempotent append,
transactional checkpoint commit/rollback, multi-process writers, crash recovery, `verify_integrity`,
`compact(keep_checkpoints=N)`. Checkpoints round-trip via `PersistentStore.commit_checkpoint` /
`load_latest_checkpoint`. **Migration note:** an existing JSONL log is read as-is; to move to SQLite, replay
the JSONL events into a `SqliteBackend`-backed log (the event log is the source of truth, so this is a
lossless re-ingest — idempotent by content id).

**Runtime-status contract (`phase8_runtime`).** Each family carries an executable `runtime_status`. Consumers
call `select_families(candidate_ids)` (blocks only `quarantined`/`incompatible`), `family_runtime_manifest`
(full per-family provenance for the result), `support_grade_effect` (grade/uncertainty impact of load-bearing
experimental families), and `resolve_transition_tier` (1→7 fallback; uncertain transitions are never removed).
`runtime_status_table()` produces the family status table. A new family registers via
`register_persistent_variable` with a `runtime_status` (defaults `exploratory`).

**Identity contract (`phase8_identity`).** `IdentityResolver.resolve(raw, as_of=…)` returns weighted
hypotheses (uncertain linkage preserved), with `world_id`/`scenario_id`/`dyad_id`/`edge_id` helpers.

## 5 — Reproducibility commands

All pure compute (no LLM) except the one live universal-path trace. Datasets auto-download to
`data/omnibehavior/` (HF `jiawei-ucas/OmniBehavior`, CC-BY-NC-SA 4.0) / reuse cached Phase-9 Enron + congress
artifacts.

```bash
# unit / integration / e2e / adversarial / migration tests
PYTHONPATH=. python -m pytest tests/test_wmv2_phase8.py -q                       # 28 tests

# Track A — reproduce the persistence win THROUGH the shared world (n=7074)
PYTHONPATH=. python -m experiments.wmv2_phase8_shared_world --n-users 140
#   → experiments/results/phase8/shared_world_trackA.json

# Tracks B (Enron dyadic) + C (Senate institutional)
PYTHONPATH=. python -m experiments.wmv2_phase8_tracks
#   → experiments/results/phase8/tracks_BC.json

# causal ablations + forensic traces + checkpoint/performance bench
PYTHONPATH=. python -m experiments.wmv2_phase8_ablations --n-users 140
#   → experiments/results/phase8/ablations.json, forensic_traces.json

# completion pass: OmniBehavior regression protection + canonical ablations + cross-run + storage bench
PYTHONPATH=. python -m experiments.wmv2_phase8_regression --n-users 140
#   → experiments/results/phase8/regression_canonical.json (established result NOT overwritten)

# completion pass: targeted habit held-out + honest family-evidence map
PYTHONPATH=. python -m experiments.wmv2_phase8_empirical --n-users 140
#   → experiments/results/phase8/empirical_completion.json
```

### Dataset acquisition / provenance

| dataset | id / source | acquisition | license |
|---|---|---|---|
| OmniBehavior | HF `jiawei-ucas/OmniBehavior` (`raw_user_data/en/`) | `swm.eval.omnibehavior_eval.download_users` (urllib, no token, smallest-first) → `data/omnibehavior/` | CC-BY-NC-SA 4.0 |
| Enron | CMU enron tarball → `experiments/results/phase9/enron_comm_edges.json` | Phase-9 cached artifact | public |
| Senate S117 | Voteview roll-calls → `experiments/results/phase9/congress_S117_bills.json` | Phase-9 cached artifact | public |

`data/` is git-ignored (repo policy); the cache re-downloads on demand. Machine-readable results under
`experiments/results/phase8/` **are** committed. Determinism: every experiment is seeded; checkpoints carry
an integrity hash; identical (log, as_of, seed) reproduce identical posteriors + terminal distributions.

---

## 6 — What a consumer should know

* Phase 8 is **production-usable for all nine families by default** (none quarantined/incompatible), but
  empirical validation is uneven: `engagement_propensity` + `habit_strength` are empirically supported;
  the rest are usable with broader uncertainty + a lower support grade (see the validation doc's final family
  table). Production-usable is not the same as validated — treat exploratory/highly_speculative persistent
  state as structurally sound and executable but empirically unproven, and read the per-family support status
  in the result's `phase8_family_manifests`.
* Persistence **lowers the support grade** when history is thin and **never abstains** (no-abstention
  contract). Recommendation eligibility is withheld where a longitudinal causal effect is unsupported, while
  the simulation still runs.
* The event log — not the checkpoint — is the source of truth. Prefer replay over trusting a stored
  checkpoint when both are available.
