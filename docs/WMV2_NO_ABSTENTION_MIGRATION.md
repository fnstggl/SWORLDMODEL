# WMv2 No-Abstention Migration (Part A5)

*The production world model now attempts a simulation for **every coherent social question**. Weak
evidence, structural disagreement, transport risk, unfamiliar domains, missing production-eligible
parameter packs, and uncertain hidden variables no longer prevent a forecast — they widen priors, add
competing hypotheses, disperse the posterior, lower the **support grade**, add explicit limitations and
reduce recommendation eligibility. Forecast abstention as a first-class outcome is **removed**.*

This document is the repository-wide audit (A5): every production use of abstention/refusal semantics,
classified, with the migration applied and the historical record preserved.

## 1. The three axes that replace binary abstention

A single `abstain: bool` conflated three independent questions. They are now separated
(`swm/world_model_v2/result.py`):

| axis | values | meaning |
|---|---|---|
| **SimulationStatus** | `completed` · `completed_with_degradation` · `clarification_required` · `execution_failed` | *did the simulation run?* |
| **SupportGrade** | `empirically_supported` · `transfer_supported` · `exploratory` · `highly_speculative` | *how well-evidenced is the forecast?* |
| **RecommendationStatus** | `eligible` · `limited` · `withheld` · `not_requested` | *may we recommend an action?* |

Rules enforced by the contract:

- A forecast is **present whenever the simulation ran** (`completed` / `completed_with_degradation`).
  Epistemic weakness lives in `support_grade` + `limitations`, never in a refusal.
- `clarification_required` is reserved for a genuinely **incoherent** question — no simulable outcome
  contract exists even after generating competing interpretations. It must be **rare** (< 5% gate).
- `execution_failed` is an **engineering/architectural** failure (exception, unresolved reference, invalid
  plan, unbindable readout, missing operator, parser/retrieval/infrastructure failure). It carries a
  `failure_taxonomy` code and **may not be used to hide** compiler incompleteness, missing mechanism
  coverage, unresolved terminal bindings, or unsupported state paths.
- **Forecast abstention rate must be 0**: no coherent question that ran may withhold a probability for
  epistemic reasons.

## 2. What guarantees a forecast — the mechanism fallback hierarchy (A3)

`swm/world_model_v2/fallback.py`. For each required causal process the compiler chooses the highest
defensible tier; `no production-eligible mechanism` **never** becomes `no forecast`:

| tier | family | support grade |
|---|---|---|
| 1 | scenario-fitted + held-out validated | empirically_supported |
| 2 | domain-validated parameter pack | empirically_supported |
| 3 | cross-domain/population transfer-validated pack | transfer_supported |
| 4 | published empirical mechanism (widened transport uncertainty) | transfer_supported |
| 5 | reference-class estimated mechanism | exploratory |
| 6 | generic structural mechanism family (broad priors) | exploratory |
| 7 | multiple competing qualitative mechanism hypotheses | highly_speculative |

Tiers 6–7 are **real typed mechanisms** (`GenericOutcomeOperator`): they state causal assumptions, draw the
terminal outcome from a broad prior (a fixed wide Beta selected by a **qualitative** directional lean — the
number is never LLM-minted), produce a `StateDelta`, execute through `WorldState`, support sensitivity, and
are labeled exploratory/highly_speculative. The generic resolver is **always attached** as the terminal
safety net, so the readout binds and the option space is covered for every question. The overall support
grade is the **weakest** load-bearing tier.

## 3. Repository-wide classification of every abstention reference

Scope of the migration = the **production V2 path** (`swm/world_model_v2/` + `swm/facade.py`). The V1 /
baseline engines under `swm/engine/*` are frozen science (`product_eligible=False`) and keep their own
abstention semantics deliberately; they are **not** the product path and were not changed.

### 3a. Migrated — live refusal paths retired

| location | old behavior | new behavior |
|---|---|---|
| `compiler.py` `compile_world` | raised `CompileAbstention` on no readout / only-unvalidated mechanisms | never epistemically abstains; repairs the readout, attaches the fallback resolver, grades support |
| `materialize.py` `run_from_plan` / `check_readout_binding` | raised `MaterializeAbstention` on dangling readout / no executable mechanism | raises `CompilerExecutionError` (`terminal_readout_unbindable` / `missing_required_operator`) — an **engineering** failure, taxonomy'd |
| `facade.py` V2 path | returned `{abstain: True, …}` | returns `SimulationResult.as_dict()` — a forecast whenever it ran |
| `forward_ledger_v2.py` | boolean `abstained` gated scorability | three result axes; `row_produced_forecast()` reads old + new rows; weak grade never excludes a forecast |
| `calibration.py` | `decide_abstention` / `AbstentionDecision` decided whether to forecast | `grade_support()` grades but never gates; old policy **deprecated**, retained for its tests only |

### 3b. Deprecated-but-retained (compat shims, no live refusal)

`CompileAbstention` (compiler.py), `MaterializeAbstention` (materialize.py),
`decide_abstention`/`AbstentionDecision`/`ABSTAIN_GRADES` (calibration.py). Kept so old imports resolve and
their existing tests still pass; **not on the production path**. Each carries a DEPRECATED docstring.

### 3c. Legitimately retained — `abstain` as a domain action, NOT a forecast refusal

These are real-world actions/states and are correctly kept:

- `institutions.py` vote execution: `votes = {voter: 'yes'|'no'|'abstain'}` — a **voter** abstaining.
- `transitions.py` collective-vote operator: default per-actor ballot `'abstain'`.
- `actor_cognition.py`: the election action set includes `'abstain'` (a voter choosing not to vote); an
  actor may also `abstain` from **interpreting** an item it cannot parse — an actor-level micro-decision
  under uncertainty, surfaced as state, not a system-level forecast refusal.

### 3d. Documentation refreshed (stale abstention-model prose)

`state.py` (Provenance docstring + sensitivity comment) and `calibration.py` (module docstring) were
reworded from the old signal→abstain model to the no-abstention framing (high-sensitivity `assumed` fields
→ broad priors + lower support grade, not refusal).

## 4. Historical results are preserved (not rewritten)

- **No historical benchmark output was edited, deleted, hidden, weakened, or overwritten.** The Session-1
  Phase-1 validation (`docs/WMV2_COMPILER_VALIDATION.md`,
  `experiments/results/wmv2_compiler_generality.json`) still reports its original numbers — including the
  29.8% `CompileAbstention` rate that was a *valid, desired* outcome under the **old** semantics. The
  Session-1 harness `experiments/wmv2_compiler_generality.py` is likewise untouched.
- The new no-abstention validation lives in **new** files
  (`experiments/wmv2_phase1_no_abstention_generality.py`,
  `docs/WMV2_PHASE1_VALIDATION.md`) so the two epistemic regimes are never conflated.
- **Reading** old artifacts: `result.migrate_legacy_result()` re-labels a stored pre-migration
  `abstain=True` into the new axes **without editing the artifact** — a coherence-reason abstention →
  `clarification_required`; a "no executable mechanism" / "dangling readout" / "parse failure" abstention →
  `execution_failed` with the right taxonomy (those were engineering gaps, not forecast refusals). The
  original text is preserved under `provenance.migrated_from_legacy_abstention`.

## 5. Migration tests (intentional semantic changes, pinned)

The four tests that encoded the OLD abstention behavior were updated to assert the NEW behavior — a
deliberate semantic change, documented here, **not** a hidden regression:

| test | now asserts |
|---|---|
| `test_compiler_never_abstains_repairs_readout_and_falls_back` (was `…abstains_without_readout_or_mechanisms`) | no-readout & only-unknown-mechanism plans COMPILE, repair the readout, attach the fallback resolver, and produce a real distribution |
| `test_operatorless_mechanism_rejected_but_fallback_still_forecasts` (was `…rejected_at_compile`) | the unported mechanism is still rejected, but the generic resolver forecasts anyway |
| `test_dangling_readout_is_repaired_not_aborted` (was `…aborts`) | a dangling readout is repaired to the canonical `outcome`, run completes |
| `test_novel_scenario_compiles_and_runs_end_to_end` | fidelity plan `marginalized_with_uncertainty`; distribution over the DECLARED options |

New regression test `test_binary_option_polarity_is_order_invariant` locks the affirmative-first option
normalization (a `yes`-leaning question gives high P(affirmative) regardless of the order the LLM listed the
options). Ledger back-compat covered by `test_forward_ledger_v2.py` (old + new rows both score).

Full suite after migration: **all targeted suites green** (see `docs/WMV2_PHASE1_VALIDATION.md`); the two
repo-wide failures (`fastapi` not installed; gitignored `data/dataset_registry.json`) pre-exist and are
environmental, unrelated to this change.
