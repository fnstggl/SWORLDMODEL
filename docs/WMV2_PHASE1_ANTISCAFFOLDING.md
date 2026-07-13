# WMv2 Phase 1 — Anti-Scaffolding Answers (B17)

*Direct answers to the "is this real or is it scaffolding?" challenges, each backed by a code reference and
a reproducible artifact. The claim under test: a single generic pipeline produces an honest, executable,
terminal-state forecast for an arbitrary coherent social question — no domain hard-coding, no LLM-minted
probabilities, no hidden refusals.*

### Q1 · Is this keyword-routing a hand-written branch per domain?
**No.** There is no `if election / if email / if viral / …` anywhere in `swm/world_model_v2/`. All 16
domains flow through the identical `compile_world → build_world → run_from_plan → project` path. Enforced
two ways: the unit gate `test_no_scenario_branches_in_v2_source` (`tests/test_world_model_v2.py`) and the
harness static check `_no_keyword_router` reported as the B13 gate `no_keyword_router`
(`experiments/results/wmv2_phase1_no_abstention_generality.json`). The per-domain forecast rates in that
JSON come from one compiler, not sixteen.

### Q2 · Does the LLM mint the final probability?
**No.** The LLM proposes only **qualitative** structure — actors, relationships, mechanism *names* from the
registry, a directional `outcome_lean` (one of five words), per-hypothesis leans, and sensitivities.
Terminal probabilities come from a typed mechanism: `GenericOutcomeOperator` (`fallback.py`) draws a
per-particle base rate `p ~ Beta(a,b)` where `(a,b)` is a **fixed wide** pair selected by the qualitative
lean (`LEAN_BETA`), then `outcome ~ Bernoulli(p)`. `AgentDecisionOperator` **ignores** an LLM-minted `p`
unless an explicit experimental opt-in is set — proved by `test_llm_cannot_mint_probabilities_by_default`
(`tests/test_wmv2_tier_a_fixes.py`). The B13 gate `llm_prob_injection_rate == 0` measures that no accepted
mechanism minted a probability on the real run.

### Q3 · Does a genuinely novel, unseen question actually run?
**Yes.** The 104 validation questions are natural language only — **no scripted target plans**; the compiler
builds its own plan for each. `test_novel_scenario_compiles_and_runs_end_to_end` uses a
negotiation/ratification class never named in any implementation example and asserts a native terminal
distribution. The per-domain `forecast_rate` in the validation JSON is measured on held-out questions the
code has never seen.

### Q4 · Is the fallback hierarchy real, or decoration that never fires?
**Real and load-bearing.** The ablation `no_fallback_hierarchy`
(`experiments/wmv2_phase1_ablations.py`) removes the generic resolver + resolve_outcome event; the
forecast/complete rate **collapses** and questions turn into `execution_failed:missing_required_operator`.
Removing readout repair (`no_readout_repair`) turns questions into
`execution_failed:terminal_readout_unbindable`. See `experiments/results/wmv2_phase1_ablations.json`
(`contributions_vs_full`). If the fallback were decoration, removing it would change nothing.

### Q5 · Are structural hypotheses real competing particles or a cosmetic label?
**Real.** `materialize._run_with_hypotheses` stratifies particles across hypotheses by prior and overrides
the terminal resolver's lean **per hypothesis**, so competing structures produce genuinely different
terminal outcomes; `structural_posterior` is the materialized particle mass. The ablation
`no_structural_hyps` drives structural entropy to **0** and narrows terminal dispersion — proof the
component does work. Forensic traces (`docs/WMV2_PHASE1_FORENSIC_TRACES.md`) show the per-hypothesis leans
and the resulting `structural_disagreement`.

### Q6 · Is the readout actually read from terminal world states, or hardcoded?
**Read from terminal states.** `OutcomeContract.project` reads the readout variable off each branch's
terminal `WorldState`; the result carries `readout == "terminal_states"` and `n_deltas > 0` (a real
StateDelta history). The forensic traces include a sampled `delta_log` per trace so the causal history is
inspectable. A readout that cannot bind is an `execution_failed` (engineering), never a silent constant.

### Q7 · Are the provenance statuses honest, or is fabricated data stamped "observed"?
**Honest.** No LLM-proposed entity field enters the world as `observed`; compiler proposals are `inferred`
with method `compiler:proposal:*` (`test_compiler_proposals_are_not_stamped_observed`). The B13 gate
`provenance_status_rate == 1.0` and `unsupported_precision_rate < 0.02` measure this on the real run.

### Q8 · Is it just returning 0.5 for everything (a degenerate constant)?
**No.** The forecast tracks the lean: with more particles, `strong_yes → ~0.71`, `neutral → ~0.50`,
`strong_no → ~0.30` (Beta means), and the polarity is order-invariant —
`test_binary_option_polarity_is_order_invariant`. On the **general path** many forecasts are *near* 0.5
because the honest posture, absent a held-out-validated domain mechanism and with competing structural
hypotheses, is a broad prior — correctly graded `exploratory`/`highly_speculative` with wide dispersion and
explicit limitations, **not** a hardcoded constant. Sharpening those forecasts requires domain parameter
packs (tiers 1–4) and evidence assimilation (Phase 3), which are out of Phase-1 scope; Phase 1's claim is
generality + no-abstention + honesty, which the varying-by-lean behavior and the support grades demonstrate.

### Q9 · Where does a weak/unknowable question stop being a forecast?
**It doesn't stop.** Weak evidence, transport risk, unfamiliar domain, missing parameter pack, uncertain
latents → broader priors + competing hypotheses + lower `support_grade` + `limitations` + reduced
`recommendation_status`, but still a forecast (`forecast_abstention_rate == 0`). Only a genuinely
incoherent question → `clarification_required` (gate < 5%), and only a technical failure →
`execution_failed` (gate < 10%, taxonomy'd). Those two are measured and bounded, not used as cover for
epistemic weakness (see `WMV2_NO_ABSTENTION_MIGRATION.md`).

### Q10 · Is the whole thing reproducible, or does it drift run to run?
**Reproducible.** Each compilation is keyed by `plan_hash` (digests question, contract, mechanisms,
hypotheses, compiler version) and persisted; provenance stamps the code commit, prompt hash, evidence
bundle hash, and seed. Harnesses are resumable via per-question caches; no `Date.now()`/`Math.random()`
nondeterminism enters a stored artifact (timestamps injected, RNG seeded). Same question + same cache →
identical result.
