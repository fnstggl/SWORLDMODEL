# Audit B — the 10 most dangerous DEFAULT-ON findings (World Model V2)

Default path traced: `facade.forecast(architecture="world_model_v2")` → `unified_runtime.simulate_world`
(structural_mode **ensemble** default) → `structural_runtime` → per-model `phase8_pipeline.prepare/slice/finalize`
→ `materialize.operators_from_plan` with **SWM_ACTOR_POLICY=hybrid_relevant_actor_policy** and
**SWM_CONSEQUENCES=generated_actor_mediated_world** as the unset-env defaults → temporal-runtime rollout →
`EventTimeContract`/`OutcomeContract.project`. Every finding below is reachable on that path with no opt-in.

## 1. Tier-3 actors are numeric psychology, executed into the terminal
`qualitative_actor.py:917` + `actor_selection.py:151`. Any declared entity without an authority/stance/
resource/relation signal in the compiled plan gets tier 3 → `ActorPolicyModel` (broad-pack utility softmax,
fixed family weights, `posterior.sample`). The sampled action **executes**, its consequences compile through
the causal boundary, and its branch mass lands in F(deadline). The report lists them
(`actors_routed_numerically`) but nothing quantifies how much terminal mass ran through fitted psychology.
This is the single largest standing numeric-actor substitution: it is *by design* (hybrid mode), but tier 3
currently means "plan lacked metadata about this actor", not "this actor is genuinely routine".

## 2. LLM failure → numeric decision, not truncation
`qualitative_actor.py:947` (`llm_failed_or_unparseable`). After primary retries + fallback model families,
a Tier-1 consequential actor's decision is taken by the numeric policy and **executed**. The row is excluded
from the counted action distribution, but the branch — with a broad-prior softmax decision driving Putin or a
CEO — stays inside the terminal probability. The honest-truncation contract already exists 30 lines away
(`phase4_execution.py:663` records `temporally_truncated:actor_llm_budget_exhausted` and refuses to invent
behavior); parse/transport failure should use the same contract.

## 3. Budget exhaustion has TWO doors; only one is honest
`ProductionActorPolicyOperator.run` and `GeneratedActorInvocationOperator.run` gate before invoking and
record temporal truncation (correct). But `qualitative_actor.py:918` converts `llm_budget_exhausted` into a
numeric decision for any caller that reaches `decide()` un-gated (the individual-reaction route at
`individual_reaction.py:231`, the phase-3 bridge, any future seam), and `engine.decide:758` returns None on
a mid-loop budget hit — which the caller then **mislabels** `llm_failed_or_unparseable` → numeric. The
numeric door must be deleted; both seams should emit the truncation record.

## 4. SWM_ACTOR_LLM_BUDGET=240 vs the 200-particle event-time floor
`qualitative_actor.py:707` (240 calls/run, all branches × actors × decisions) collides with
`event_time.py:1254` (event-time contracts floor `n_particles` at 200). One decision per branch exhausts the
budget at ~branch 120–240; every later branch's decisions become truncations. Because truncated branches
project as *censored* mass (see #5), the binary answer quietly drifts toward NO as the budget runs out —
a compute artifact wearing a probability's clothes. Budget must scale with particles × expected decisions,
and truncated-branch share belongs next to the number.

## 5. Truncated branches are counted as clean NO mass
`event_time.py:263` (`EventTimeContract.project`): P(yes) = F(deadline); every unabsorbed branch counts
toward NO / `none_of_the_options_by_horizon` — including branches stopped by `safety_max_events`,
invocation caps, or LLM-budget truncation. `pipeline.result_from_run:98` flips the status to
`temporally_truncated` and caps support, but the **distribution itself** blends compute-exhausted worlds in
as if they causally survived to the horizon. Truncated branches need separate projection (report over
completed branches + truncated share) or hazard-state imputation.

## 6. Institution members are Bernoulli coins on the absorbing path
`phase_consumers.py:117` (`CollectiveThresholdDecisionOperator`): member votes = iid draws from ONE
propensity (posterior draw, or **LEAN_BETA broad prior when no posterior**), Binomial against the declared
threshold — "member correlation not modeled". `activation_synthesis.py:279` schedules this event for every
declared institution, and `convert_binary_to_event_time`/`convert_to_event_time` mark it the **absorbing
writer**: for institutional questions this coin IS the answer. The decision holders are typically declared
entities — the qualitative runtime could actually simulate their votes (generated-mode
`run_institutional_aggregation` already counts real per-holder records when schema mechanisms drive it).

## 7. The generic-prior ladder still pins the binary answer's total mass
`event_time.py:514/701` (`_calibrated_target`/`_fp_target_mass`): the binary residual chain's per-particle
target absorbed-mass = posterior → `family_hazard_pack` pooled rate → **LEAN_BETA(lean)**. With no `as_of`,
an empty bundle, or no admissible claims (all default-reachable — evidence failure "never blocks"), and with
the pack file being a **cwd-relative path** (`experiments/replay_vault_v3/...` — silently absent outside the
repo root), the answer's total probability mass is a 40-world population average (mostly sports questions)
or an ignorance Beta shaped by an LLM-proposed lean. The simulation then only shapes *when/how*, not
*how much*. Same ladder in `AggregateOutcomeOperator` (`phase_consumers.py:291`) and the always-attached
`GenericOutcomeOperator` terminal safety net (`compiler.py:493`, live for continuous contracts and whenever
event-time conversion throws — which is swallowed at `unified_runtime.py:441` into a lineage note only).

## 8. Template inner lives on hypothesis failure
`qualitative_actor.py:233` (`_fallback_hypotheses`): if the hypothesis-generation call fails or parses
empty, the actor's persistent hidden state for the entire run becomes one of three stock personalities
(steady_confident / private_doubt / depleted_delegating). Decisions are still LLM-made, but conditioned on
generic psychology shared across every actor and scenario that hits this path. Recorded only inside
`revision_log`/assumptions — no result-level surface. Should retry fallback families and, on total failure,
run stateless rather than instantiate template minds.

## 9. Fixed context windows silently amputate long cascades
`QualitativeConfig` (`qualitative_actor.py:706-712`) + `build_prompt`: last **10** observed events (220
chars each), last **6** actions, last **8** memories (of 12 stored), menu ≤14, rules [:8], relationships
[:8], beliefs [:14]; hypothesizer evidence last 10 events / 2400 chars. In deep cascades the causally
decisive early events leave the prompt with no salience selection and **no record of what was dropped** —
the actor's psychology changes as a pure function of buffer position. Needs salience-weighted selection +
`n_events_dropped` provenance.

## 10. Plan-level actor caps decide who exists
`fidelity.py:55-141` (missing_entities [:10], relations [:16], actions [:12]) and
`resolution_criteria.py:134` (intentions grounded for first **8** entities, 12 stances kept). The 9th+
entity gets no grounded stance → no hazard ratio, no tier signal → numeric tier 3 (or invisible to hazards
entirely). Combined with the absent never-invoked-entity census (`generated_world.py` counts invocations but
never reports declared-but-never-invoked actors), whole actors can be identified and then silently not
simulated. Caps need recorded clip counts and causal-signal-ranked (not positional) selection.

### Honorable mentions (default-on, lower blast radius)
- `INTENTION_HR_PRIORS` (`event_time.py:73`): stance→hazard multipliers 0.55×–2.10×, unfitted (no pack in
  repo), with reliability/capability shrinks — a numeric psychology channel deliberately split against the
  behavioral channel (`endogenous_stance_split` prior median 0.6, unfitted).
- `ensure_first_passage_state` (`event_time.py:753`): no fitted survival curve → hardcoded 0.5 total
  absorption mass per mode over the window.
- `direct_targets[:16]` (`generated_world.py:344`): 17th+ recipient of one event silently dropped.
- Frontier LLM extension only when deterministic base <6 actors (`generated_world.py:1027`).
- Equal-weight ensemble mixture remains the headline number even when `incomplete=true` (labeled).
- Cwd-relative packs (`family_hazard_pack`, `family_survival_pack`, `actor_decision_calibration`): running
  from any other directory silently downgrades to broad priors / unvalidated labels.

### What is already honest (keep as pinned contracts)
`ProductionActorPolicyOperator.run` + `GeneratedActorInvocationOperator.run` budget gates (truncation, never
substitution); `run_branch_temporal` safety_max_events truncation; mechanism runtime `unresolved` statuses
(success never assumed); `unresolved_share` option-space accounting in `OutcomeContract.project`;
`structurally_underidentified` generation-ceiling marker; quarantined `legacy_ablations` (token-gated);
`AgentDecisionOperator` uniform path (dead on default — compiler swaps it out); persona blend (opt-in env,
`llm_probability_minting` stamped); individual-reaction route failing loud with no backend.
