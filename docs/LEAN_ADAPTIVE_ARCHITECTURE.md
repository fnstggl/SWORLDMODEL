# World Model V2 — Lean-Adaptive Execution Profile

`execution_profile="lean_adaptive"` is an internal execution strategy of `world_model_v2` —
never a different product. `execution_profile="full_fidelity"` remains the permanent, unchanged
PR-#127 research-grade runtime (maximum-depth analysis, research, debugging,
structural-sensitivity studies, escalation target). Entry:

```python
from swm.world_model_v2.unified_runtime import simulate_world
simulate_world(question, as_of=..., llm=..., execution_profile="full_fidelity")   # research grade
simulate_world(question, as_of=..., llm=..., execution_profile="lean_adaptive")   # the default
```

Resolution: explicit argument > `SWM_EXECUTION_PROFILE` env > module default
(`unified_runtime.DEFAULT_EXECUTION_PROFILE` = **`lean_adaptive`** since the §25 default
switch, taken when all seven conditions passed on the complete five-question paired baseline —
evidence in `experiments/results/exp109_acceptance_final.json`; `full_fidelity` remains the
explicit research-grade option). Unknown names fail loudly. The profile is stamped into
`result.provenance["execution_profile"]`.

## The governing principle

Preserve every distinction that could materially change the decision or outcome. Remove every
computation that does not. Thirty particles remain thirty distinct possible worlds; only the
genuinely distinct actor decision situations inside them spend provider calls. The reuse
standard everywhere is **exact equality after deterministic projection onto the actor's
decision-relevant situation** — never byte-equality of whole worlds, never fuzzy similarity, no
LLM judge / embeddings / approximate matching anywhere in a cache path.

## Module map (all under `swm/world_model_v2/`)

| Module | Responsibility |
|---|---|
| `lean_runtime.py` | The orchestrator: same canonical funnel, lean stages swapped in |
| `lean_artifacts.py` | `RunSharedArtifacts` — compile once; hash/version/provenance/ownership; dependency-cascade invalidation; payloads frozen at registration |
| `lean_structural.py` | One primary model → one reversal critic → ≤1 challenger; no-reversal verdict = single-survivor convergence certificate; extra credible alternatives ⇒ `structurally_underidentified` + full-fidelity escalation offer |
| `lean_cohorts.py` | `LeanCohortHypothesizer` — same one-generation-per-actor memoization and `branch_index mod K` assignment law as full fidelity; deterministic paraphrase collapse (behavioral token multisets); reversal-focused cohort critic; bounded expansion; explicit under-modeled marker. No cohort probabilities exist |
| `lean_context.py` | `DecisionRelevantContext` — the deterministic projection + signature; `DecisionDependencySpec` with a non-narrowable mandatory floor; canonical fact ids; `DecisionContextDifference`; `DecisionEquivalenceCertificate`; the context-seeded replicate rng law |
| `lean_decision_cache.py` | `DecisionEquivalenceCache` — run-scoped, immutable validated templates, deep-copy reuse, receiving-branch revalidation, single-flight, failures never cached, `explain_equivalence()` |
| `lean_invalidation.py` | Frontier gate (`should_invoke`), deterministic prechecks, `PriorDecisionValidity` + material-change detection, duplicate-notification suppression, the five execution classifications, `AvoidedCallLedger` (§23 reasons) |
| `lean_cognition.py` | One-call bounded cognition — every stage explicit in one structured response; deterministic memory stages per branch; validation-failure escalation to the staged pipeline (recorded) |
| `lean_prompts.py` | `ActorContextSnapshot` (byte-stable cohort-baseline prefix → provider prompt-prefix caching) + `ActorDecisionDelta`; `effective_actor_view` losslessness reconstruction; measured char accounting |
| `lean_consequences.py` | `ConsequenceProgramCache` — provider response reused on exact (fingerprint, compile-prompt) keys; parse/validate/authority/boundary/binding/execution rerun per branch |
| `lean_routing.py` | `TieredRouter` — deterministic code first; light tier for low-risk language stages; strong tier pinned for consequential stages; recorded escalations; honest single-family manifest |
| `lean_particles.py` | `run_progressive_particles` — index-keyed batches on the same prepared run; all-conditions stopping (drift, interval, side-of-0.5, unresolved, truncation, structural agreement, reversal, action ranking); explicit recorded tolerances |
| `lean_stability.py` | Instability signal detection; ONE capped execution-replicate probe (behavioral replicate 1 on the same compiled world); replicate results reported, never averaged |
| `lean_controller.py` | `LeanActorController` — binds everything into the canonical actor runtime through three seams; research-first arming invariant |

## The three canonical seams (the only integration points)

1. **Operator seam** — `phase4_execution.ProductionActorPolicyOperator.run`: the causal-frontier
   gate (structural observability only; fails open; never de-lists actors).
2. **Decision seam** — `qualitative_actor.QualitativeActorPolicyRuntime.decide`: prechecks →
   prior-decision validity → projection → equivalence cache (single-flight) → one-call cognition,
   escalating to the staged full-fidelity pipeline on a recorded reason. The obstacle-revision
   call routes through the same seam. Everything downstream — `_resolve`, perceived/actual
   feasibility, revision, memory commit, §NAP posterior, persistence — runs unchanged per branch.
3. **Post seam** — `_post_execute`: persists the branch's standing decision + processed facts.

With no controller attached (full fidelity), all three seams are inert — pinned by test.

## What is shared vs never shared

Shared (immutable): validated decision templates (deep-copied on reuse), consequence-compile
provider responses (fully revalidated per branch), cohort templates, run-shared artifacts,
question-level compilation via the run's content-addressed call cache.

Never shared: mutable actor state, memory, event queues, world objects, institution state,
population state, `StateDelta`s, consequence execution results, branch weights.

## Correctness invariants kept from PR #127 (unweakened)

Canonical entry `unified_runtime.simulate_world`; evidence-sufficiency gating; escalating
retrieval; recurrence-aware priors; §NAP (no arbitrary numeric social reality, no silent
`generic_outcome_prior`); visible operator-rejection reasons; `ensure_outcome_pathway` on every
rollout entry (inside `prepare_persistence_run` — lean cannot bypass it); rollout retry; honest
unresolved statuses; deprecated-simulator quarantine; §19 strict actor integrity (no numeric
actors — the lean deterministic layer may conclude *no decision exists*, it can never *decide*
for a human).

## Mandatory gates (tests)

* **Isolated caching parity** (`tests/test_lean_integration.py::test_isolated_caching_parity_gate`):
  identical scripted worlds, caches on vs off → identical decisions/distributions/statuses/
  censuses/forecasts; only call counts differ.
* **Concurrency identity** (`::test_lean_sequential_equals_bounded_concurrent`): sequential ==
  `SWM_BRANCH_THREADS=4`; parallelism changes wall-clock only.
* **Research-first**: actor psychology cannot start before the research ledger completes
  (`LeanActorController.arm_actor_calls`).
* **Behavioral replicates**: `behavioral_replicates_per_decision_context = 1` by default; the
  replicate index is part of every context signature; particle→replicate assignment is
  deterministic.

## Escalation paths

Lean → full fidelity remains one argument away. A lean run self-reports when escalation is
warranted: `structurally_underidentified` (structural cap), stability signals
(`provenance.lean.stability_signals`), the execution-replicate spread limitation, and cohort
under-modeling markers.

## Benchmarks

`experiments/exp107_btf3_full_fidelity_post127.py` (baseline) and
`experiments/exp108_btf3_lean_adaptive.py` (lean arm) run the five frozen BTF-3 rows under the
sealed-replay frozen-background bundle (`experiments/btf3_frozen_bundle.py`) with real provider
token usage recorded per call. `experiments/exp109_compare_arms.py` emits the machine-readable
comparison.
