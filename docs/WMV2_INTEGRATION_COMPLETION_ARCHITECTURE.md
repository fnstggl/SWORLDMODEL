# World Model V2 — Integration Completion: Architecture

**Scope.** This run does **not** add a new runtime. It closes concrete *activation-chain
breaks* inside the one canonical runtime (`simulate_world` / `unified_runtime.py`) so that phases
which are **causally required** by a question actually instantiate and execute, instead of being
declared by the compiler and then silently dropped before execution. It also fingerprints the
runtime so the old Phase‑12 calibration corpus is correctly demoted to *diagnostic-only*.

The mandate's correct standard is **not** 100% raw activation of every phase on every question. It
is: **near‑100% recall of a phase on the questions that structurally require it, low false
activation on questions that do not, and a real causal effect when it does fire.** Manufacturing
100% by executing irrelevant phases, or adding empty calls to make manifests light up, is
explicitly forbidden and was not done.

## The one runtime (unchanged shape)

`compile_world` → `WorldExecutionPlan` (typed sections: entities, populations, institutions,
relations, quantities, latents, scheduled_events, accepted_mechanisms, actor_decisions,
structural_hypotheses) → `simulate_world` threads a single plan / world / event-queue /
StateDelta log / terminal through Phases 1–11. Event → operator dispatch is unchanged:
`op.applicable(world, ev)` → `op.run` → `StateDelta` appended. No phase-specific side pipeline
was added.

## Part A — activation-chain audit (per phase, compiler → runtime)

Full machine-readable audit: `experiments/results/integration/activation_chain_audit.json`.
The audit compiled real questions and inspected both the emitted plan fields and the runtime
instantiation contracts. Summary of where each chain breaks:

| Phase | Compiler emits | Runtime needs | Where the chain breaks |
|---|---|---|---|
| **4** actor policy | `actor_decisions` + `production_actor_policy` op | registered actor operators + a decision event | Conditional: emitted for clearly-strategic Qs; gap is Qs where the compiler never identifies a strategic actor / names the operator. |
| **6** mechanism registry | `accepted_mechanisms[operator=<family>]` | family operators (registered) fire iff named | Compiler rarely names a registry operator for general Qs; falls back to `generic_outcome_prior`. |
| **7** nonlinear | `accepted_mechanisms[nonlinear_*]` + a scheduled event | nonlinear operators (registered) | Operators registered but the compiler never names them and emits no matching event. |
| **9 pop** | `populations[{id,segments}]` | an operator that aggregates population state into the terminal | Emission is question-dependent **and** there is no causal *consumer* of population state. |
| **9 net** | `relations[{src,rel,dst}]` | layer-specific consumers | Relations land in `world.network` but distinct semantic layers aren't modeled and nothing consumes them causally. |
| **10** institutions | `institutions[{id, rules:[{kind,params}]}]` | rules whose `kind ∈ EXECUTABLE_RULE_KINDS`, an `institution_action` operator, and an `institutional_action` event carrying an `InstitutionRuntime` | **ROOT CAUSE (fixed this run):** emitted rule kinds (`voting_rule`/`confirmation_process`/`committee_vote`…) are **not** in `EXECUTABLE_RULE_KINDS` → `materialize` **drops** them → empty (ornamental) RuleSystem. |
| **11** recompilation | controller runs over observations | — | Runs on every execution; fires only on genuine surprise. Not yet validated on adversarial shock/migration scenarios. |

## What this run actually lands (verified) vs. defers (honest)

**Landed and verified end-to-end — Phase 10 rule executability.**
`swm/world_model_v2/integration_completion.py::normalize_institution_rules` maps the compiler's
rule kinds onto the closed `EXECUTABLE_RULE_KINDS` set (`voting_rule→quorum`,
`confirmation_process/committee_vote→procedure`, `veto→decision_right`, `funding→budget`,
`timeline→deadline`, …; anything unmapped → generic executable `procedure`). It **never drops a
rule**, preserves `params`, records the original kind in `ru["_original_kind"]`, and is
idempotent. It is conservative: it only makes an **already-declared** institution executable — it
never invents an institution. Wired **default-on** in `unified_runtime.py` (guarded by
`"phase10_institutions" not in drop`) right after populations/networks are threaded, and recorded
under `manifest["phase10_institutions"]["normalization"]`. Verified: on the Senate-confirmation
question, executable institution rules go **0 → 4** and the completeness diagnostics clear.

**Diagnostic infrastructure, not a claim of full activation.**
- `completeness_diagnostics(plan)` (Part J) flags `institution_declared_but_no_executable_rule`
  and `institution_declared_but_no_operator` — a deterministic post-compiler validator, recorded
  in the lineage.
- `infer_required_phases(plan)` gives an **independent structural** requirement judgment used only
  for the manifest `relevance` field and the validator. It is **not** the benchmark ground truth
  (those labels live in `experiments/integration_corpus.py`, Part B) and it fabricates nothing.

**Deferred, reported as failed gates with an exact continuation path (see the validation doc).**
Full *execute + StateDelta + matched-ablation* activation of Phases 6/7/9pop/9net to the ≥95%
recall / ≥90% causal-effect gates requires emitting new operators **and** their causal consumers
(population aggregation into the terminal; typed multilayer network consumers; nonlinear operator
+ matching event; registry operator selection). Doing that with integrity is a multi-run effort;
this run does **not** fake it. The Part-A audit gives each one an exact fix + acceptance test.

## Part O — runtime fingerprint / Phase‑12 invalidation

`swm/world_model_v2/runtime_fingerprint.py::runtime_fingerprint()` content-hashes every phase
version + the runtime commit. `corpus_status(hash)` returns `product_eligible` only if the corpus
was produced under the current fingerprint, else `diagnostic_only`. The old Phase‑3/Phase‑12
corpus predates these changes, so it is **diagnostic_only** and the existing calibrator is marked
**INCOMPATIBLE** (`experiments/results/integration/phase12_compatibility.json`).
`experiments/phase12_refit.py` now prints the required fingerprint and refuses to treat an
unstamped corpus as product-eligible.
