# WMv2 Phase 1 — Current Compiler-Path Audit (Part B)

*A stage-by-stage trace of the production Phase-1 path as it stands, with the abstention/refusal semantics
found repo-wide and how each was resolved. Companion to `WMV2_NO_ABSTENTION_MIGRATION.md` (the
classification table) and `WMV2_PHASE1_ARCHITECTURE.md` (the design).*

## The single production path (traced)

| # | stage | file:symbol | behavior |
|---|---|---|---|
| 1 | entry | `facade.forecast(architecture="world_model_v2")` → `pipeline.simulate` | the ONE door; contaminated legacy execution raises rather than shipping |
| 2 | decompose | `compiler.compile_world` → `_DECOMPOSE_PROMPT` + `llm` | qualitative structure only; bounded parse retry; `_salvage_json` recovers truncated replies |
| 3 | coherence | `compile_world` (`coherent:false` ∧ no outcome) | → `ClarificationRequired` (rare); otherwise proceed |
| 4 | outcome | `_coerce_outcome` + `_affirmative_first` | malformed contract repaired, never refused; affirmative option normalized to `options[0]` |
| 5 | readout | `_repair_readout` | declared quantity kept; else routed to canonical `outcome` the resolver writes |
| 6 | mechanisms | registry vet + `select_tier` + fallback | accepted (executable) / rejected (recorded) / experimental (not executed); generic resolver ALWAYS attached |
| 7 | causal sufficiency | `_fidelity_plan` | high-sensitivity explicit, low-sensitivity marginalized-with-uncertainty (kept), uncertain kept |
| 8 | grade | `overall_support_grade` | weakest load-bearing tier; `degraded` if fallback/transfer/repair used |
| 9 | materialize | `materialize.build_world` + `InitialStateModel` | provenance-typed fields (never fabricated `observed`); latents as distributions; correlated |
| 10 | rollout | `RolloutEngine` + `_run_with_hypotheses` | event-driven on real calendar time; structural hypotheses stratified as competing particles |
| 11 | readout bind | `check_readout_binding` | technically unbindable → `CompilerExecutionError` (engineering failure), never abstain |
| 12 | project | `OutcomeContract.project` over terminal states | native terminal distribution; unresolved mass reported, not counted |
| 13 | result | `pipeline.result_from_run` | `SimulationResult` — forecast present whenever the simulation ran |

## Abstention/refusal semantics found repo-wide, and disposition

Full classification table in `WMV2_NO_ABSTENTION_MIGRATION.md` §3. Summary of what the audit found in the
**production V2 path**:

- **3 live forecast-refusal paths** (`CompileAbstention` in the compiler, `MaterializeAbstention` in
  materialize, `abstain:True` from the facade) — all retired. The compiler no longer abstains for epistemic
  reasons; materialize failures are now `CompilerExecutionError` (taxonomy'd engineering failures); the
  facade returns a `SimulationResult`.
- **1 scorability gate** (`forward_ledger_v2.abstained`) — migrated to the three result axes with a
  back-compat reader.
- **1 grading policy** (`calibration.decide_abstention`) — deprecated; replaced by `grade_support` which
  grades but never gates.
- **Legitimate domain `abstain`** (a voter/actor choosing not to act, in `institutions.py`,
  `transitions.py`, `actor_cognition.py`) — correctly retained; these are actions/states, not forecast
  refusals.
- **Stale abstention-model prose** (`state.py`, `calibration.py` docstrings) — reworded to the
  no-abstention framing.

## Where forecasts previously stopped, and why they no longer do

The Session-1 Phase-1 validation (`WMV2_COMPILER_VALIDATION.md`, preserved unedited) measured, under the
OLD semantics, a **29.8% CompileAbstention rate** and a **51% end-to-end execution rate** — i.e. roughly
half of coherent questions produced no forecast. The audit traced those stops to four causes, each now
resolved on the general path:

| old stop | old outcome | now |
|---|---|---|
| LLM readout pointed at an entity.field no mechanism writes | `MaterializeAbstention` (dangling readout) | `_repair_readout` → canonical `outcome`; resolver writes it |
| no registry mechanism resolved to an executable operator | `CompileAbstention` (no executable mechanism) | fallback hierarchy always attaches `generic_outcome_prior` |
| plan typed but readout unbound at horizon | unresolved terminal mass → abstain | generic resolver writes the readout iff unset; option space covered |
| complex multi-party decomposition overran `max_tokens` | `parser_failure` → no forecast | `_salvage_json` recovers the outcome-contract prefix; question still forecasts |

The genuinely-technical residue (LLM returns literally nothing parseable; a true runtime exception) remains
`execution_failed` with a taxonomy code — an **engineering** failure, honestly reported, never an epistemic
abstention. The measured rates under the new semantics are in `WMV2_PHASE1_VALIDATION.md`.

## Grep-proof generality (B14)

No scenario-level branch exists in `swm/world_model_v2/` — no `if election / if email / if viral / …`.
Enforced two ways: the unit gate `test_no_scenario_branches_in_v2_source` and the validation harness's
`_no_keyword_router` static check (reported as a B13 gate). Every domain flows through the identical
compile→materialize→rollout→readout path.
