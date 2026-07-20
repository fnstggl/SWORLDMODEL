# Integration of PR #124 + PR #125 + PR #126 — preservation ledger and audit

This branch (`claude/world-model-pr-integration-b00zd3`) fully merges and supersedes:

| PR | Title | Head | Merge base vs `claude/world-model-v2` tip (c8c0eca) |
|---|---|---|---|
| #124 | Remove the validation gate: the compiled world always simulates and decides its own outcome | `145f79f` (claude/remove-validation-gate-oyz05w) | `48cd060` (pre-#123) |
| #125 | §NAP: remove arbitrary numerical social reality from the default runtime | `c681b78` (claude/world-model-v2-scalars-n3h7jo) | `c8c0eca` |
| #126 | Evidence-sufficiency gating + recurrence-aware priors | `31b488f` (claude/metaculus-prediction-backtests-msl431) | `48cd060` (pre-#123) |

#124 and #126 share history up to `a8fdac4` (the EXP-101/EXP-102 segment: provenance-aware
mechanism kernels, `_num()` null tolerance, the OpenRouter backend, exp101/102 artifacts). Both
predate PR #123's combined-runtime rearchitecture (structural ensembles, staged persistence
funnel), so their merges also reconcile with #123's shapes. #125 was built directly on the tip.

Merge order: base c8c0eca → merge #126 → merge #124 → merge #125 → semantic composition.
The order only provided a working sequence; **no overlapping file was resolved whole-file
ours/theirs**, and no capability was dropped because of order.

Marks used below: **U** preserved unchanged · **C** preserved and combined with another
implementation · **E** preserved through an equivalent consolidated implementation ·
**D** duplicate of another implementation · **R** intentionally removed (justified).

---

## 1. Preservation ledger — PR #124

| File / capability | Mark | Where it lives in the integration |
|---|---|---|
| compiler.py — execution-critical key ordering in the decompose prompt | U | `compiler.py` prompt (`KEY ORDER MATTERS`) |
| compiler.py — `_EXECUTION_CRITICAL_KEYS`, `_CONTINUATION_PROMPT`, `_recover_truncated_tail`, `truncated_reply`/`truncation_recovered_keys` provenance | U | `compiler.py` |
| compiler.py — run-everything principle (experimental mechanisms EXECUTE labeled: `calibration_status=experimental`, `uncertainty_widened`, exploratory grade; rejection only for unknown id / no operator) | U | `compiler.py` mechanism acceptance |
| Evidence text entering the compiler (`simulate_world(evidence=...)` → `compile_world`) | C | `unified_runtime.simulate_world` threads `evidence` through **both** structural modes (single-model ablation and ensemble candidate compilation) — #123's router did not exist in #124's tree |
| llm_actor.py — actor knowledge scoping prompt, `register_entity_extension("llm_persona_state")`, non-fatal `expected_reactions` writes | U | `llm_actor.py` |
| materialize.py — entity-type normalization (repair, never refuse) | U | `materialize.build_world` |
| materialize.py — `branch_thread_count()` / `run_branches()` (SWM_BRANCH_THREADS) | U | `materialize.py` |
| rollout.py — thread-parallel particle rollout | C | composed INTO #123's index-keyed `run_particle_range`: parallel via `run_branches` when no `particle_scope`; serial when the order-sensitive cross-model actor cache is active. Index keys and seed law identical either way |
| run_from_plan — `allow_experimental=True` + posterior injection | U | `materialize.run_from_plan` |
| phase3_pipeline.py / phase9_pipeline.py QUARANTINED banners | U | module docstrings |
| pipeline.py — quarantine docstring + deprecation warning on `simulate()` | C/D | docstring merged with #126's; the **inline** `warnings.warn` was a duplicate of #126's `@quarantined` decorator (same DeprecationWarning naming the same canonical entry) — the decorator is the one authority (test: `test_pipeline_simulate_is_quarantined_but_helpers_are_not`) |
| pipeline.py — `harden_general_path` (institution normalization + scheduled-reality + activation synthesis on every entry) + call inside `simulate()` | U | `pipeline.py` |
| result_from_run — truncation/hardening provenance keys | U | `pipeline.result_from_run` |
| phase8_pipeline.py — persistence parity: posterior injection, experimental execution, hypothesis stratification | C/E | ported into #123's staged funnel: `prepare_persistence_run` runs `_inject_posterior_rate` + `operators_from_plan(allow_experimental=True)`; stratification reimplemented **index-keyed** (`_stratified_queue_builder`) so progressive pilot/extension slices see exactly the assignment a monolithic roll would (pinned by `test_persistence_hypothesis_stratification_is_index_keyed`); finalize reports `structural_realized_mass` with the same keys as `_run_with_hypotheses` |
| qualitative_actor.py — knowledge-scoped hypothesize/decide prompts, PUBLIC CALENDAR section, `public_facts` config, SWM_ACTOR_MAX_CALLS, non-fatal anticipation writes | U | `qualitative_actor.py` (merged into #123's bounded-cognition config) |
| scheduled_facts.py — RECURRING EVENTS prompt block, `recurrence` field, `public_facts_lines`, `plan._scheduled_facts` | C | unioned with #126's influence schema in one extractor/operator |
| unified_runtime — `ev_text` fallback to caller evidence | C | `_condition_plan` (`_bundle_text(bundle, N) or str(evidence)[:N]`, both fidelity and temporal-compiler blocks) |
| swm/api/mechanisms.py + openrouter_backend.py + provenance kernels (shared segment) | U | identical in both #124/#126; merged once |
| docs/WMV2_CANONICAL_PATH.md, BACKTEST_FINDINGS.md updates | U | as merged |
| exp101/102/105 scripts, results, condensed traces, forecastbench quarantine banner | U | `experiments/` |
| tests: test_world_model_v2 run-everything expectations, test_mechanisms provenance tests | U | `tests/` |

Lines from #124 not present verbatim (24, audited): the pre-#123 monolithic
`run_with_persistence` body, the pre-merge `pipeline.py` docstring, the pre-#123 `run()` body and
the standalone `ev_text` line — all **E** (behavior preserved in the staged-funnel composition,
merged docstring, composed `run_particle_range`, composed `ev_text`), plus the duplicate inline
warning (**D**, above).

## 2. Preservation ledger — PR #125

Every one of #125's 45 files merged textually clean (its base was the integration base). The full
§NAP architecture is intact: `numeric_provenance.py` (ledger, approved/non-causal/rejected
classes, fitted-artifact eligibility, unresolved-mechanism records), `legacy_numeric_ablations.py`
(token-gated tables, no env door), the event_time rewrite (provenance-gated absorbing writers,
posterior-only residual process, branch terminal categories, honest bounds,
`simulated_scenario_frequency`), qualitative process records, observational persistence,
stance/coupling/action-magnitude removals, `unresolved`/`partially_resolved` statuses,
recommendation withholding, equal-weight-mixture demotion, enforcement tests (static scan, call
spies, mutation invariance, honest-unresolved, provenance completeness), six forensic traces,
deliverables and audit artifacts, and the rewritten test files.

Deltas against #125's exact lines (3, audited):

| Item | Mark | Explanation |
|---|---|---|
| `_fp_target_mass` ledger row applicability text + calibration `posterior_evidence_id` line | C | extended, not weakened: the row now names the **specific** prior behind the posterior (grounded estimate / recurrence / reference class / lean, evidence quality, retained effective N) via `plan._outcome_prior_provenance` — uniting #126's grounded prior with #125's acknowledged-remaining-assumption pattern |

Adaptations of #124/#126 mechanisms to §NAP (mechanisms preserved, provenance enforced):

* #126's no-null guard no longer manufactures a flat 0.5 (see §5.2 below).
* #126's `ensure_outcome_pathway` never bolts a resolver onto an honest-unresolved plan (§5.1).
* #126's broadened fact schema cannot over-trigger deterministic absorption (`strictly_entailing`, §5.3).

## 3. Preservation ledger — PR #126

| File / capability | Mark | Where it lives in the integration |
|---|---|---|
| `_quarantine.py` — `quarantined()` decorator, CANONICAL_ENTRY, warn-once semantics | U | `swm/world_model_v2/_quarantine.py`; now the **single** quarantine authority on `pipeline.simulate` |
| evidence_orchestrator — resolution-criterion-aware `_query_terms`, extra stopwords | U | `evidence_orchestrator.py` |
| evidence_orchestrator — escalated strategy (2x window, forced Wikipedia, decisive reformulation) | C | + an official-source query variant (announcement/statement/press-release terms), per the integration contract |
| evidence retry trigger (<3 claims → escalate, keep the better bundle) | E | consolidated into `gather_evidence_with_escalation` — ONE authority called by both the single-model path and the ensemble shared-bundle path (the ensemble had **no** retry before) |
| `evidence_sufficiency_signal` | C | tolerant of both bundle generations (#123's V2 replay bundle has no `documents`); same starvation semantics, pinned by #126's own tests |
| sufficiency gate + starved warning + limitation surfacing | E | `_evidence_sufficiency_block` + `_apply_result_guards`, called per structural model in the ensemble AND in the single-model path |
| `MAX_LLM_EFFECTIVE_N`, `QUALITY_MAX_N`, `grounded_estimate_prior`, `estimate_reference_base_rate`, 4-tier `build_outcome_rate_prior` | U | `phase3_priors.py`; prior provenance additionally stashed on the plan and named in the §NAP ledger row |
| `OUTCOME_EVENT_OPERATORS`, `_instantiate_operator`, `ensure_outcome_pathway` | C | rewritten as the composed authority (see §5.1): also counts event-time absorbing channels + the absorption monitor, respects §NAP honest-unresolved, and now guards `run_from_plan` too (previously only the persistence funnel) |
| pathway invariant in the persistence funnel + provenance/limitations | C | `prepare_persistence_run` computes it; the handle carries it; `finalize_persistence_run` surfaces it |
| rollout retry (fresh seed on an intermittent empty rollout) | E | `_project_terminal_guarded` (single-model) + the `_finalize_model` re-roll (ensemble), sharing the `_no_forecast` predicate; honest unresolved results are exempt |
| no-silent-None guarantee | C | `_apply_result_guards` — §NAP-composed ladder (§5.2). The **flat 0.5 rung is intentionally removed** (see removals) |
| `simulate_world_stable` (mean-of-K, opt-in) + `_used_probability` | U | `unified_runtime.py`; unresolved runs return None from `_used_probability` and are excluded from the mean — honest by construction |
| scheduled-facts influence schema (pattern_strength / outcome_influence / influence_strength), back-compat, `entailment_nudge`, accumulating log-odds `fact_entailment` | C | unioned with #124's recurrence metadata; plus the `strictly_entailing` boundary (§5.3) |
| `@quarantined` on `pipeline.simulate` + module docstring | C | merged docstring keeps both PRs' content |
| exp103/104/105-rerun/106 experiments + findings + checkpointing, .gitignore trace rules | U | `experiments/`, `.gitignore` |
| 8 test files (evidence sufficiency/targeting, grounded prior, mean-of-K, outcome pathway, quarantine, scheduled recurrence, mechanisms) | U | all pass unchanged |

Lines from #126 not present verbatim (71, audited): the inline evidence-retry block, the inline
sufficiency/fallback/rollout-retry/no-null blocks, and `ensure_outcome_pathway`'s
docstring/branches — all **E** in the five consolidated helpers above; plus `pattern_strength`
inline expression (**E**, hoisted to a local), and the flat-0.5 rung (**R**, §5.2).

## 4. Intentional removals (complete list)

1. **#124's inline `warnings.warn` in `pipeline.simulate`** — original: PR #124,
   `swm/world_model_v2/pipeline.py::simulate`. Why it cannot remain: it duplicates the
   DeprecationWarning already emitted by #126's `@quarantined(simulate)` (two warnings for one
   call; two competing quarantine authorities). Where the behavior lives: the decorator (warn-once
   naming `unified_runtime.simulate_world`, plus pinned `__quarantined__`/`__use_instead__`
   metadata). Test: `tests/test_quarantine.py::test_pipeline_simulate_is_quarantined_but_helpers_are_not`
   and `::test_quarantined_decorator_warns_once_and_passes_through`.
2. **#126's flat `0.5` last-resort fallback** (`fb = grounded_fallback_mean if ... else 0.5`) —
   original: PR #126, `unified_runtime.simulate_world`. Why it cannot remain: it directly violates
   #125's core invariant ("Absence of a justified number means the mechanism is unresolved — never
   0.5"), one of the two implementations had to yield on the same responsibility, and the task's
   governing rule keeps §NAP enforcement while preserving the operational capability. Where the
   intended behavior lives: `_apply_result_guards` still guarantees a structured non-None result
   for every coherent question — posterior-backed EXECUTION-DEGRADED fallback; explicitly-opened
   generic-prior door (`SWM_ALLOW_GENERIC_PRIOR`, the pre-existing §28 mechanism #125 kept) for a
   deliberately prior-driven labeled forecast; otherwise an explicit `unresolved` result with the
   missing mechanism named and the prior mean recorded as a labeled non-headline diagnostic
   (ledger-rejected `llm_estimated`). Tests:
   `tests/test_integration_cross_pr.py::test_guard_ladder_posterior_backed_fallback`,
   `::test_guard_ladder_prior_requires_the_explicit_door`,
   `::test_guards_never_touch_honest_unresolved_or_partially_resolved`.

No other production line from any PR was removed without an equivalent consolidated implementation
(audited line-by-line; see §6).

## 5. Semantic conflict resolutions (composition, not selection)

### 5.1 Outcome-pathway repair (one authority: `materialize.ensure_outcome_pathway`)
Composes #126's rollout-viability invariant, #124's structure-decides-the-outcome, and #125's
readout-not-resolver + honest-unresolved semantics:
1. census counts resolver-contract pairs AND event-time absorbing channels (dated absorbing facts,
   absorbing institutional decisions, provenance-approved first-passage residual processes) and
   requires the absorption monitor for event-time plans;
2. accidental losses (dropped writer / dropped monitor) are repaired by re-instantiating the
   DECLARED operator — structure is restored, never replaced by a prior;
3. a plan with §NAP unresolved-mechanism records is honest-unresolved-by-design: never "repaired"
   with a resolver;
4. an event-time plan with an accidental total loss records a plan-level unresolved mechanism
   (named, preserved) instead of bolting a terminal resolver onto a readout-not-resolver contract;
5. a resolver-contract plan with total loss gets the canonical `resolve_outcome` back (posterior
   attached when present; without one, `GenericOutcomeOperator` itself enforces the §28/§NAP
   refusal gate).
Applied at BOTH rollout entries (persistence prepare + `run_from_plan`). Transient empty rollouts
additionally get one recorded retry (below); nothing ever silently returns None; nothing pretends
a structurally broken run succeeded.

### 5.2 Generic/prior fallback (one authority: `unified_runtime._apply_result_guards`)
Output modes are now explicit and mutually distinguishable: normal simulated forecast ·
evidence-supported posterior result · deliberately prior-driven forecast (explicit door, loudly
labeled) · execution-degraded result (posterior-backed) · partially resolved · unresolved. The
API always returns a valid structured result; a fallback never masquerades as a simulated
structural outcome and never bypasses provenance (posterior fallback is manifest-registered;
refused prior fallback is manifest-**rejected** and non-headline).

### 5.3 Recurrence (one subsystem, three layers preserved)
* extraction/metadata (#124): recurring-pattern hunting, `recurrence` cadence + past instances,
  `recurring_event` kind — retained;
* actor calendar visibility (#124): `plan._scheduled_facts` → `QualitativeConfig.public_facts` →
  PUBLIC CALENDAR prompt section — retained;
* influence (#126): pattern_strength/outcome_influence/influence_strength with the accumulating,
  per-fact-capped log-odds `fact_entailment` (raises AND lowers compose) — retained;
* grounded continuous prior (#126): recurrence-aware outside-view estimate, quality-capped
  (`sourced` vs `model_memory`), recurrence transport floor — retained, provenance-stashed;
* provenance (#125): only an evidence-cited, STRICTLY entailing dated fact absorbs
  deterministically at its real date; a model-knowledge recurrence conditions the prior and the
  actors but cannot absorb (recorded unresolved when it claimed strict entailment). The new
  `strictly_entailing` boundary (direct entailment / confirmed_scheduled / influence ≥ 0.9) keeps
  #126's broadened schema from over-triggering either absorption or §NAP unresolved records.
Contrary evidence and modeled disruptions still move the outcome (lowering facts accumulate
negatively; evidence updates the posterior; actors see the calendar).

### 5.4 Evidence retries (one authority: `gather_evidence_with_escalation`)
The original stochastic re-roll capability survives inside a genuinely different escalation
(fresh seed AND: 2x lookback window, forced Wikipedia on every requirement, decisive-fact
reformulation from the resolution criterion, official-source query). Applied on the single-model
path AND the ensemble shared bundle (which previously had no retry).

### 5.5 Hypothesis handling
Intra-plan structural hypotheses stratify particles on every funnel (#124 parity), index-keyed for
progressive slices; hypothesis identity and realized mass are reported. Under §NAP the ±0.2/±0.4
numeric lean shift on a posterior stays quarantined (structures differ structurally, and via
LEAN_BETA only behind the explicit prior door) — #125's structural-ensemble level continues to
serve per-model conditionals + robust ranges with `partially_resolved` semantics.

## 6. Audit method and results

* **Commit/file/overlap matrix**: recorded in the session inspection report; tri-overlap files
  (materialize, pipeline, unified_runtime) plus pairwise overlaps were resolved hunk-by-hunk.
* **Symbol audit**: every `def`/`class`/CONSTANT introduced by each segment exists in the
  integration — PR124: 5/5, PR126: 57/57, shared: 11/11, PR125: 112/112.
* **Test audit**: every test function added by any PR exists — PR126: 28/28, shared: 2/2,
  PR125: 62/62 (+ #124's modified assertions); #125's rewrites of the three legacy-machinery test
  files are inherited exactly (its own documented contract change).
* **Reverse line audit** (PR head → integration, production scope): PR124 357 added lines / 24 not
  verbatim; PR126 779/71; shared 137/0; PR125 2392/3 — every non-verbatim line is classified above
  (E/C/D/R); none is an unexplained loss.
* **Enforcement**: the §NAP static scan, call spies, mutation invariance, honest-unresolved and
  provenance-completeness tests all pass over the combined tree — with #124's and #126's code
  present and executing.
* **Full suite** (hermetic environment — provider credentials unset so the model-family pool
  stays honest monoculture): integration **1857 passed, 1 skipped, 3 failed**; baseline
  `c8c0eca` under the identical environment: **1818 passed, 1 skipped, 3 failed** — the SAME
  three failures both sides (`test_dataset_registry_is_valid_and_honest`: missing
  `data/dataset_registry.json`; `test_default_family_pool_registers_deepseek_single_strong_family`:
  requires a DEEPSEEK_API_KEY in the environment; `test_predict_and_rollout_are_distinct`:
  missing optional `fastapi`). **Zero regressions introduced by the integration; +39 net new
  passing tests.**

## 7. Known limitations (honest)

* #125's own honest limits stand: no accuracy claim, no backtest/calibration of the combined
  runtime; remaining acknowledged assumptions are unchanged except the posterior's prior, which is
  now the better-grounded #126 construction (still LLM-mediated where history is not sourced —
  recorded per-run in the ledger row's `prior_provenance`).
* The recurrence hint parameter of `build_outcome_rate_prior(recurrence=...)` remains a caller
  affordance (as in #126); the default path grounds recurrence through the estimator prompt and
  the calendar layer rather than a wired hint.
* `SWM_BRANCH_THREADS` parallelism applies when no cross-model actor cache is active; ensemble
  pilots with the shared actor cache stay serial (the cache's `enter_branch` contract is
  order-sensitive).
* EXP-101..106 experiment artifacts reflect the code as it was when each ran; they are preserved
  as provenance, not re-executed against the merged runtime.
