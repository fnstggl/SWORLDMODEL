# §NAP — No Arbitrary Numeric Reality: final deliverables

## 1. Exact starting commit

`c8c0eca` — the merge commit of PR #123 (`claude/worldmodel-v2-core-arch-73s7eq`), head of
`claude/world-model-v2` at the time of this work.

## 2. Old vs new architecture

**Old (post-PR #123).** The default runtime already refused the ordinary generic Beta terminal
draw (§28) and routed actors through strict qualitative cognition. But the numerical world-model
layer still served by default:

* LLM process labels (dormant…imminent) → 0.15…0.85 `pathway_progress:*` bars (sd=0.15,
  unknown → 0.5) consumed by first-passage hazards at sampled weights 1.0/0.25/0.35 and by the
  binary residual chain at weight 0.6;
* stance labels → unfitted hazard ratios (0.55…2.10) shrunk by reliability (0.6), capability
  (0.75/0.4), graded control (0.25…1.0), live capacity (0.4+0.6·cap), split by 0.6, sampled per
  branch — always serving because `intention_hr_pack.json` never existed;
* the unfitted coupling layer (`COUPLING_PRIORS`): pathway step 0.04, persistence survival
  coins 0.75/0.85, attrition 0.0007/day, capacity psychology 0.85/0.6/0.35 with numeric
  ripeness/exhaustion/bandwagon stance rules;
* `ACTION_PATHWAY_EFFECTS` (accept +1.0 … exit −1.0) generating binding feasibility
  prohibitions via the ≥0.5 threshold, and gating novel-action ontology anchoring;
* the event-time forced-forecast ladder: posterior → 40-world keyword family rate → ungated
  lean-Beta; LLM fact confidence as a Bernoulli parameter; LLM mode priors normalized into
  first-passage intensity shares; hypothesis-lean ±0.2/±0.4 posterior nudges; the ungated
  `StructuralProcessPriorOperator` broad-prior draw; equal-weight structural averaging served
  as the headline probability.

**New.** A numerical value may affect production execution only under an approved provenance
class (`observed_measurement`, `explicit_user_input`, `institutional_rule`,
`physical_identity_or_conservation`, `empirically_fitted_and_validated` passing the artifact
gate, `derived_deterministic`). Everything else is a qualitative typed state, an actor-mediated
decision, a competing structural hypothesis, or an explicit `unresolved` record. Concretely:

* typed process records `{state, waiting_on, basis}` replace progress bars; ungrounded stays
  `ungrounded`, never 0.5;
* stances are qualitative records conditioning each actor's own situated cognition; a
  commitment binds feasibility only when its basis is a literal instrument
  (law/treaty/contract/formal rule) and only with the instrument's own stated prohibitions;
* outcomes resolve through: evidence-cited dated facts (deterministic at their real dates),
  institutional decisions executing their real rules, scenario-generated actor-mediated
  mechanisms (PRs #119–#123), or ONE residual process parameterized only by the evidence-updated
  posterior; otherwise the mechanism is recorded unresolved;
* branch terminals classify `resolved_yes / resolved_no / censored_by_real_horizon /
  unresolved_mechanism`; unresolved mass is never normalized away; results report honest bounds,
  the resolved-conditional distribution, and `simulated_scenario_frequency` semantics;
* materially disagreeing structural models are served as per-model conditionals + robust ranges
  (`partially_resolved`), never averaged into a headline; the equal-weight mixture survives only
  as a labeled diagnostic;
* persistence is observational: a provisional end-state confirms iff it still holds when its
  criterion window completes in the simulated world; collapse only via a modeled breaking
  mechanism (`break_provisional_state`) — no survival coin;
* every run carries a `numeric_causal_inputs` manifest (approved-and-consumed / approved-unused
  / rejected / compute-safety / analysis-display / ablation-only) merged from the plan ledger
  and every branch world's ledger;
* recommendations are withheld whenever unresolved mass exists (Phase 13 §31 + §NAP gate).

## 3. Machine-readable audit

`artifacts/no_arbitrary_numeric_reality/numeric_causality_audit.json` — every numeric-causal
finding with file/symbol/value/concept/class/default-reachability/verdict/replacement, plus the
entry-point call-graph trace and search patterns used.

## 4. Removed or quarantined production consumers

| Consumer | Disposition |
|---|---|
| `mode_graph.PROCESS_STATE_LEVELS` + `declare_pathway_processes` (+ event_time `declare_contested_mode_channels`) | removed; tables buried in `legacy_numeric_ablations` |
| `mode_graph.STANCE_ORIENTATION / RELIABILITY_SHRINK / CAPABILITY_SHRINK / CONTROL_WEIGHTS / ENDOGENOUS_STANCE_SPLIT / combine_stances / _stance_hr / pathway_orientation` | removed; buried |
| `event_time.INTENTION_HR_PRIORS / _hr_table / mode_hazard_ratio / sampled HR machinery / endogenous split / consume channels` | removed; buried |
| `event_time._calibrated_target / _fp_target_mass` family-rate and lean-Beta rungs | deleted; posterior-only, ledger-registered |
| `world_dynamics.COUPLING_PRIORS / sampled_coupling / StanceReviewOperator / capacity+attrition` | removed; buried |
| `world_dynamics.PersistenceCheckOperator` survival coin | replaced observational |
| `phase4_policy.ACTION_PATHWAY_EFFECTS / actions_advancing_pathway / stance_action_alignment` | removed; buried; numeric-baseline stance term zeroed |
| `resolution_criteria._STANCE_WEIGHT` + `actor_intentions` aggregate + prohibition derivation | removed; literal-instrument binding replaces it |
| `semantic_consequences.derive_pathway_summaries` (0.95/0.45/0.30/0.15/stage-fraction) | token-gated legacy ablation; no production caller |
| `phase4_execution._apply_pathway_effects` | legacy-mode-only; now sources its tables from the quarantine under the token |
| `fallback._LEAN_SHIFT / _apply_lean_shift` | removed/raising; buried |
| `phase_consumers.StructuralProcessPriorOperator` LEAN_BETA draw | suppressed by default (+unresolved record) |
| `phase_consumers.AggregateOutcomeOperator` family-rate rung | suppressed by default (+unresolved record) |
| `phase_consumers.PopulationAggregationOperator` 0.5/0.2 heterogeneity defaults | refuses assumed priors; observed/derived only |
| `phase_consumers.NetworkDiffusionOperator` `_LAYER_TRANSMISSIBILITY` | suppressed by default (+unresolved record) |
| `temporal_runtime` stance-watch thresholds (0.30/0.70/0.08) + attrition call | removed |
| scheduled-fact confidence → Bernoulli | replaced: evidence-cited facts absorb deterministically; model-knowledge facts recorded unresolved |
| LLM mode `prior` elicitation + z-normalized intensity shares | removed; support counts only |
| structural equal-weight headline | demoted to labeled diagnostic; per-model conditionals primary |

## 5. Replacement mechanisms (through the existing generated architecture)

1. **Scenario-generated concrete state** — typed process records (`process_state:<pathway>`,
   string-valued) + the generated scenario schemas' typed objects/facts (PR #119/#121).
2. **Scenario-generated events and transitions** — evidence-cited scheduled facts absorbing at
   real dates; institutional decisions as absorbing writers; the generated temporal model's
   decision triggers (PR #120).
3. **Institutional/communication/physical mechanisms** — unchanged PR #123 machinery (kernel
   ports, mechanism invocations, outside-world provenance-gated arrivals).
4. **Actor-mediated decisions** — the strict qualitative actor (PR #123) with stances and
   literal binding commitments in its own view; stance changes through its own cognition.
5. **Competing structural hypotheses** — the structural ensemble's separate conditionals and
   robust ranges, no averaged headline under material disagreement.
6. **Explicit unresolved uncertainty** — `record_unresolved_mechanism` /
   `plan_record_unresolved`, branch terminal category `unresolved_mechanism`, statuses
   `unresolved` / `partially_resolved`, honest bounds, withheld recommendations.

## 6. Default-route call-graph proof

Static + runtime enforcement in `tests/test_numeric_reality_enforcement.py`:
AST scan proves no production module references any quarantined symbol and no module-level
import of the quarantine exists; raising call-spies prove default event-time execution touches
neither `_beta_sample` nor any legacy table; mutation invariance proves default output is
byte-identical when every buried constant is multiplied by 1000. The default spine
(facade → `simulate_world` → structural ensemble → `_condition_plan` → conversion →
persistence funnel → `result_from_run`) is traced in the audit JSON.

## 7. Runtime provenance manifest examples

See `forensic_traces/case*.json` → `numeric_causal_inputs`. Example (case 1): consumed —
`absorbing_fact:entailed_fact:signing` (observed_measurement, claim c_sign_1, value 1.0);
rejected — `family_fallback_rate` (keyword_family_rate), `lean_beta_target` (lean_beta).

## 8. Enforcement test results

Full suite: 1753 tests → 1770 collected after the rewrite; result at time of writing:
all green except two pre-existing environment failures unrelated to this change
(`test_agent_engine::test_dataset_registry_is_valid_and_honest` — missing
`data/dataset_registry.json` in the repo; `test_state_world_model::test_predict_and_rollout_are_distinct`
— missing optional `fastapi` dependency), both verified to fail identically on the starting
commit. The §NAP enforcement file covers: static reference scan, module-level import scan,
event-time source assertions, no-env-door check, raising call-spies over default execution,
default suppression of structural-process/network-diffusion draws, mutation invariance,
unresolved/partially-resolved result behavior with withheld recommendations, manifest
completeness, fitted-artifact eligibility gate, and no-numerical-psychology source checks.
PR #123's invariants (no ActorPolicyModel / no UtilityInference on strict default; truncation
on actor failure) remain enforced by `tests/test_core_arch_invariants.py`, unchanged.

## 9. Forensic trace summaries

`forensic_traces/summary.json`:

| Case | Terminal | Distribution | Unresolved | Notes |
|---|---|---|---|---|
| 1 negotiation, signed proposal | completed | yes 1.0 | 0 | evidence-cited fact absorbs at its real date |
| 2 public political commitment | **unresolved** | — | 1.0 | statement is not binding; no mechanism → named missing mechanism; recommendation withheld |
| 3 institutional vote | completed | yes 0.69 | 0 | real threshold rule over posterior member votes |
| 4 personal communication | completed | reply 0.52 | 0 | posterior-only residual; simulated_scenario_frequency |
| 5 launch + 14-day persistence | completed | yes 0.84 | 0 | observational persistence; modeled rollback collapses ⅓ of branches; no survival coin |
| 6 no defensible mechanism | **unresolved** | — | 1.0 | no Beta/family draw; family+lean registered rejected |

## 10. Exact remaining empirically unsupported assumptions

Still on a default path, explicitly recorded (audit `verdict: remaining_assumption`):

1. The evidence posterior's prior is a broad unfitted lean-Beta (phase3) with reference-class
   transport discounts (`TRANSPORT_RETAINED`, `MAX_EFFECTIVE_N=40`); the posterior serves only
   with ≥1 effective as-of observation and its ledger entry carries this caveat.
2. `TIMING_REGIMES` (immediate…months duration bands) and `LATENT_STATE_CHECK_FACTOR`
   (attention cadence ×3/×5/×6) in the generated temporal layer — behavior-timing structure;
   terminal timing claims carry `timing_support_classification`.
3. `bounded_cognition` memory retrieval-failure bands (0.02/0.15/0.35…).
4. `phase8_persistence` priors (mean 0.5, sd 0.29; per-variable terminal sensitivities) —
   inert without a durable actor checkpoint.
5. `phase9_temporal._EVENT_LOGODDS` trust-edge shifts (downstream diffusion now refuses, so
   these currently modulate edge state only).
6. `registry/applicability.py` family-matching heuristic weights over the cited-coefficient
   registry (the coefficients themselves are cited; the matcher is heuristic).
7. LLM role-play itself: particle frequencies are labeled `simulated_scenario_frequency`, not
   calibrated probability.

## 11. Explicit classification

* **Mechanically implemented:** yes (all changes above, with tests).
* **Default production path:** yes — the provenance gate, unresolved semantics, and
  quarantines are the default; no flag enables them.
* **Backtested:** **no.** No claim of predictive accuracy is made.
* **Calibrated:** **no.** All served shares are simulated-scenario frequencies.
* **Statistically superior:** **no** (not claimed; requires held-out backtesting).
* **Prospectively validated:** **no.**
* **Exploratory-launch ready:** yes, with the honest-status contract
  (unresolved/partially_resolved results and withheld recommendations are expected outputs).
* **Consequential-recommendation ready:** **no** — recommendations are withheld under
  unresolved mass, truncation, under-modeling, and family monoculture, by design.

## Final self-audit answers (from the final code and traces)

1. Can a qualitative process label become a production number? **No** (no label→number map
   exists; `declare_typed_processes` writes strings; enforcement scans).
2. Can a process stage become generic percentage completion? **No** (`derive_pathway_summaries`
   token-gated; no `pathway_progress` quantities exist).
3. Can a public stance directly multiply a hazard? **No** (no stance→hazard channel; HR tables
   buried; conversion reports `stance_hazard_channel: removed_quarantined`).
4. Can reliability, control or capability become an unfitted coefficient? **No** (qualitative
   record fields only).
5. Can an action name carry a universal numeric world effect? **No** (table buried; consequence
   flow is the generated causal boundary).
6. Can a hand-authored action magnitude change feasibility? **No** (binding prohibitions come
   only from literal instruments' own stated prohibitions).
7. Can an LLM mint a mode probability or structural-model weight? **No** (mode prompt requests
   no weights; support counts only; ensemble critics strip minted probabilities; no headline
   average under disagreement).
8. Can a missing mechanism become a broad prior draw? **No** (generic/institution/aggregate/
   structural-process/diffusion draws all suppressed with unresolved records).
9. Can event-time fallback manufacture target outcome mass? **No** (family and lean-Beta rungs
   deleted; posterior-only, ledger-registered; otherwise unresolved).
10. Can unresolved branches be normalized away? **No** (distribution carries
    `unresolved_mechanism` mass; bounds reported; sums preserved).
11. Can a tiny keyword-selected family pack resolve an unrelated question? **No** (fails
    `fitted_artifact_eligible`; registered as rejected in every conversion manifest).
12. Can compute exhaustion substitute numerical psychology? **No** (PR #123 invariant
    preserved: `ActorDecisionUnavailable` truncates the branch).
13. Does every consumed real-world causal number have approved provenance? **Yes** (ledger
    registration at consumption; manifest in every result; provenance-completeness test).
14. Does default execution remain fully scenario-generated, actor-mediated, event-driven and
    structurally ensembled? **Yes** (PR #119–#123 architecture untouched and now the ONLY
    consequence route; enforcement suite for those PRs still green).
15. When the system does not know, does it actually say it does not know? **Yes**
    (`unresolved` status: "Outcome unresolved under the current model", missing mechanisms
    named, recommendations withheld — forensic cases 2 and 6).
