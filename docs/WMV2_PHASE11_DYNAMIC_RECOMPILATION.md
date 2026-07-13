# WMV2 Phase 11 — Dynamic Recompilation: Audit, Architecture, Contracts, Migration

Phase 11 builds **production dynamic recompilation**: while a simulation runs, detect when the compiled causal
model is no longer adequate, determine the smallest causally sufficient revision, generate + score competing
revised plans, migrate compatible posterior world-state safely, retain structural uncertainty, and continue
the *same* timeline — never silently resetting history or fabricating certainty. It is graded honestly across
four independent statuses in `WMV2_PHASE11_VALIDATION.md`.

Everything is built on the audited canonical V2 path; Phase 11 adds one universal pipeline and does **not**
duplicate Phase 1/3/6/8/9/10.

## 1. Ruthless audit of the inherited recompilation surface (Phase 0)

Base: `claude/world-model-v2` @ `9f20591` (includes merged Phase 3 hardening #89 and Phase 8 persistence #90).

| Capability | Where | Classification |
|---|---|---|
| Evidence-conditioned plan revision | `evidence_recompile.recompile_with_evidence` | **executable but COMPILE-TIME only** — one-shot LLM-proposed structural revision from an evidence bundle; no runtime trigger, no state/particle/event migration, no lineage/rollback, not in the rollout loop |
| Full re-decomposition | `compiler.recompile(plan, new_evidence, reason)` | **executable but toy for continuation** — recompiles from scratch, bumps `version`/`parent_version`, but **discards** posterior particles + history (a reset, not a migration) |
| Typed plan diff | `evidence_recompile.PlanDiff/PlanDiffEntry` | executable; stringly-typed; reused + extended into `phase11.contracts.TypedPlanDiff` |
| Predictive surprise | `posterior.posterior_predictive_check`; Phase-8 `ParticleFilter` `obs_loglik`/`ess` | executable diagnostic; **reused** as the trigger signal |
| Posterior particles | `posterior.ParticlePosterior` (`Particle(world, weight, ancestry)`, `ess`, `resample`) | executable; **reused** for migration |
| Checkpoint / integrity / schema-migration | Phase-8 `PersistentStore`, `PersistentCheckpoint` (sha256, `verify`, `migrate`, `rollback_to`) | executable; **reused** patterns (canonical hashing, integrity, schema versioning) |
| Rollout loop | `rollout.RolloutEngine.run_branch` | executable; **no runtime recompile hook** (seam at `rollout.py:41`, after `clock.advance_to`) |
| Result contract + statuses | `result.SimulationResult`, `SIMULATION_STATUSES` (`completed`/`completed_with_degradation`/`clarification_required`/`execution_failed`) | executable; **reused** |
| Runtime trigger detection / fusion / scope inference / candidate scoring / state+event+particle migration / plan lineage / rollback / anti-thrashing / continued-rollout recompile | — | **ABSENT** — the Phase 11 deliverable |

Two load-bearing findings drove the design:
1. **The only prior "recompile" is compile-time or a reset.** Neither preserves the running posterior/history
   across a mid-simulation structural change. Phase 11 is the runtime layer they lack.
2. **`WorldExecutionPlan.plan_hash()` is too weak for identity** — it excludes `institutions`,
   `structural_hypotheses`, `version`, so an institution-rule-only revision hashes *identically* to its
   parent. Phase 11 adds `_serial.plan_content_hash` (folds in version + full inventories + P11 markers), used
   for lineage/oscillation/identity; without it the oscillation guard mis-fires on legitimate revisions.

Also reused, not rebuilt: `WorldState.clone()` (deepcopy — the only whole-world copy; there is no `WorldState`
serializer, so migration is deepcopy + typed surgery), `Event`/`EventQueue`, `StateDelta` (a *record*;
operators mutate the world in place), `register_relation/quantity/entity_extension` (writes are registry-gated).

## 2. One universal architecture (spec §7)

```
advance ensemble to next EXTERNAL observation
  → assimilate (Phase-3 posterior update — ALWAYS, recompile or not)
  → predictive diagnostics (surprise / impossibility / ESS / regime / drift)
  → trigger detection (eligible observations only; 16 families)
  → dependence-aware fusion + false-positive control
  → decision (proceed?) → scope selection (smallest causally sufficient)
  → candidate generation (current + minimal + alternative + full) → static validation
  → reproducible scoring (current plan included) → retain the plan MIXTURE
  → atomic migration (off-path build → verify invariants → activate | rollback)
  → emit typed recompile events → continue the same timeline
  → terminal distribution marginalised over the plan mixture + particles
```

One pipeline drives every domain (`controller.RecompilationController`); domains supply evidence + parameter
packs but never bypass the common trigger/scoring/migration/continuation path. Execution is via an injected
`ExecutionAdapter` so the controller runs on the real V2 substrate in production and on a numeric substrate in
validation (which still uses real `WorldState` particles, so migration invariants are genuinely exercised).

### The trigger discipline (critical)
An event **sampled by the running simulation that the active plan already represents** is executed normally
(mechanisms + `StateDelta`) and **never** triggers a recompile — however surprising. That is ordinary Phase-3
posterior updating, not model inadequacy. Only these are trigger-eligible (`triggers.observation_eligible`):
external / leakage-safe-historical-replay / internal-diagnostic observations, an observation **outside the
active plan's representational support**, or **verified** new structure. Consequently **a normal one-shot
forecast with no later external evidence performs zero recompiles** — an invariant enforced by the eligibility
gate and checked by tests + the negative-control episodes.

## 3. Versioned contracts (`phase11/contracts.py`, spec §8)

All are `@dataclass` + the `Versioned` mixin (semantic version + deterministic canonical serialization + a
self-verifying sha256 content hash + forward `migrate`), so every stored artifact self-detects corruption.

| Contract | Role |
|---|---|
| `RecompileObservation` | one observation: `origin` (gates eligibility), `representable`, residual, evidence ids/hashes, actor visibility, related entities/institutions/network region, contradictions, temporal validity |
| `RecompileTriggerEvidence` | one detector's finding: family, scope candidates, severity, persistence, expected impact, evidence independence, trigger probability, alternatives, thresholds version, cooldown, fingerprint |
| `RecompileDecision` | action ∈ 15 types, selected + deferred scope, expected value of recompilation, compute estimate, support grade, limitations |
| `PlanRevisionCandidate` | changed/unchanged components, causal explanation, supporting/contradictory evidence, mapping requirements, complexity, LLM provenance, static-validation result, `is_current_plan` |
| `TypedPlanDiff` / `TypedPlanDiffEntry` | typed change over 15 plan targets (outcome contract … terminal readouts) — not an untyped JSON diff |
| `MigrationPlan` | entity/split/merge/population/institution/network/edge/state-variable mappings, unit + posterior + parameter transforms, history/evidence retention, pending-event transforms, canceled reasons, orphaned state, rollback reference, invariants |
| `PlanLineageNode` / `PlanLineageEdge` | immutable lineage with plan hash/version, parents, trigger/migration ids, checkpoint hash, posterior weight, code/compiler/registry versions, status, failure reason |
| `RecompilationTrace` | the full audit: observations, diagnostics, trigger posterior, scope + alternatives, all candidates + scores + rejections, decision, migration report, before/after summaries, event-queue diff, plan mixture, lineage, continued rollout, terminal effect, cost/latency, checksums, emitted events |

## 4. Trigger detection, fusion, scope (spec §9–§11)

- **16 executable detectors** (`triggers.py`, asserted to cover every family): diagnostic-driven
  (unexplained-residual — fires only when extreme or part of a sustained run, *not* on a single in-support
  surprise; impossible-event; sustained-predictive-failure; particle-collapse; mechanism-regime-change;
  parameter-drift; precondition-failure) and evidence/structure-driven (new actor/institution, rule/authority/
  coalition/network/outcome-space change, exogenous shock, evidence contradiction). Each returns a typed
  probability from a transparent `severity·persistence·independence` squash — **never an LLM-minted number**.
- **Thresholds are versioned with provenance** and `residual_high` is **learned on the calibration split's
  unchanged-control residuals** (false-alarm-controlled 90th-percentile); test outcomes are never used.
- **Fusion** (`fusion.py`) collapses dependent/syndicated evidence to its strongest signal (counted once),
  noisy-ORs independent groups, lets one impossible event dominate many weak residuals, retains contradictions,
  and applies cooldowns / hysteresis / persistence / a false-alarm budget so a noisy observation cannot force a
  recompile. It classifies transient / drift / local-structural / global-structural.
- **Scope** (`scope.py`) selects the smallest causally sufficient scope via a typed causal-impact graph;
  global invalidations escalate to `outcome_contract`/`full_plan`, drift stays `parameter_only`, and high
  terminal sensitivity + *ambiguous* evidence retains a competing hypothesis (mixture) rather than downgrading
  a verified edit. Chosen scope **and** alternatives are recorded.

## 5. Candidate generation, scoring (spec §12–§13)

- Candidates always include the **current plan** (so "don't recompile" can win), a minimal deterministic
  revision at the scope, an alternative structural hypothesis when ambiguous, and a full-recompile candidate on
  global invalidation. Typed `PlanTransform` ops build a revised plan **copy** (copy-on-write). The LLM may
  propose ≤2 grounded qualitative alternatives (cites evidence ids; no numbers/rule-text/winner); a
  deterministic fallback keeps it runnable with no LLM. Static validation: schema, temporal (no future-dated
  rule / no time reversal), evidence grounding, unsupported-precision, causal reachability.
- Scoring (`scoring.py`) is a fixed weighted sum of **separately stored** components (residual reduction,
  evidence fit, structural plausibility, mechanism/institution/network consistency, continuity/migration
  completeness, complexity, compute, transport, calibration); the plan posterior is a softmax (Bayesian model
  weights). **The LLM never selects.** Candidates within a margin are retained as a normalized mixture, so
  structural uncertainty is preserved, not collapsed to top-1; `recompile_warranted` is False when the current
  plan wins.

## 6. Migration design (spec §14–§16)

Migration is causal, not serialization. On an additive structural revision, **all** source entities/quantities/
edges are preserved (deepcopy of each particle world + typed surgery) and new structure is added with **broad**
priors (a new actor gets a wide latent, no access to others' private history). Explicit handlers: actor
**split** (partition resources — no duplication — into uncertain components), actor **merge** (sum resources
once, retain provenance, represent identity uncertainty), institution-rule + network-edge migration (registers
relation types as needed). **Posterior particles** are migrated + renormalized with ESS reported; broad priors
for new variables prevent deterministic collapse / false certainty. **Pending events** are classified
(valid / deduped / superseded / canceled / dropped-past) with reason codes — guaranteeing **no duplicate
signatures and no event behind the migration time (no time reversal)**. Anything unmappable becomes a typed
**orphan/quarantine** record with a terminal-sensitivity estimate — no silent loss.

The **adopted structure then governs execution** via the `ExecutionAdapter.post_migration` hook (broad
uncertainty over revised/new components, §15) — this is precisely why recompilation recovers *beyond* what
posterior updating alone can do (a no-recompile run's stale particles cannot reach the new regime).

## 7. Lineage, checkpoint, rollback, anti-thrashing (spec §18–§19)

`lineage.py`: immutable plan lineage (nodes + edges) with cycle + A→B→A **oscillation** detection;
integrity-hashed `Checkpoint`s (sha256, Phase-8 style) persisted **atomically** (temp + `os.replace` — closing
the non-atomic-write gap the audit found in Phase-8 `save_checkpoint`); a `RecompileTransaction` with **atomic
activate-or-rollback** (candidate built off-path, invariants verified, then activated; any failure restores the
source snapshot so the only valid world is never partially mutated). Anti-thrashing: max recompiles/horizon,
oscillation refusal, fusion cooldowns/hysteresis/false-alarm budget. On a safety limit the controller continues
the best valid plan mixture with a **degraded** support grade — it does not silently stop forecasting.

## 8. Integration into the canonical path & execution records (spec §17, §20)

The controller drives a stepped rollout via the `ExecutionAdapter` (`advance`/`predict`/`assimilate`/
`terminal`/`post_migration`). A real V2 adapter binds `init_state → queue_builder_from_plan → operators_from_
plan → contract.project`; the seam is `rollout.py:41`. Each recompilation emits the ten typed records
(`recompile_triggered`, `recompile_candidate_generated`, `recompile_decision`, `plan_migrated`,
`event_canceled`, `event_remapped`, `plan_branch_added`, `plan_branch_pruned`, `recompile_completed`,
`recompile_failed`) — these record causal-model evolution and **do not themselves move terminal probabilities**;
those come only from continued world execution. Simulation time is monotonic (`SimulationClock.advance_to`
raises on reversal). No forecast abstention is introduced; support grade never *improves* through recompilation.

## 9. Compatibility seams to parallel phases

Phase 11 depends only on the canonical base. Where adjacent phases expose relevant contracts but are unmerged,
Phase 11 defines narrow interfaces rather than cherry-picking: the `ExecutionAdapter` is the seam to the full
rollout substrate; `plan_content_hash` and the `Versioned` mixin are the seam to a future plan-serialization/
Phase-8-store integration (the current `Checkpoint` holds the live deepcopy for rollback + an atomic serialized
digest; wiring the ensemble into `PersistentStore` for cross-process resume is documented remaining work).

## 10. Reproducibility

```
PYTHONPATH=. python -m experiments.wmv2_phase11_corpus     # frozen corpus + splits + preregistered gates
PYTHONPATH=. python -m experiments.wmv2_phase11_eval       # arms B0-B6 + ablations + metrics + gate scoring
PYTHONPATH=. python -m experiments.wmv2_phase11_traces     # full forensic traces
PYTHONPATH=. python -m pytest tests/test_wmv2_phase11.py   # 27 unit/integration/adversarial/determinism tests
```
Machine-readable artifacts under `experiments/results/phase11/`: `corpus.jsonl`, `splits.json`, `gates.json`,
`corpus_manifest.json`, `eval.json`, `forensic_traces.jsonl`, `forensic_index.json`.

## 11. Known limitations (honest)

- **Real-episode coverage is 2 real-grounded vs the 60 target** — the numeric-substrate arm (real `WorldState`
  particles, controlled process) exercises the full Phase 11 logic across 10 families × 9 domains, but is
  semi-synthetic. The real-record *replay* arm (actual roll-call/docket streams as observation sequences) is
  the declared remaining expansion; the corpus manifest reports `real=False` and is **not** relabelled.
- **Two gates fall short on the frozen test split**: trigger recall 0.76 (< 0.85) and scope exact/equivalent
  0.69 (< 0.75) — driven by the pure-diagnostic families (impossible / regime / contradiction) on the
  numeric substrate. Reported, not tuned away.
- **Migration to the persistent store is in-memory** (deepcopy snapshots for rollback + atomic digest);
  cross-process resumable checkpoints via Phase-8 `PersistentStore` are a documented seam, not yet wired.
- The **real V2 `ExecutionAdapter`** (compile a real plan per episode, run through `operators_from_plan`) is
  defined by contract; the validated runs use the numeric adapter. Consequently the empirical verdict is
  **empirically validated on the constructed corpus**, **not production eligible**.
