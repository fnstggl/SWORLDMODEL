# Action-first event-time architecture (world-model-v2)

This build closes the gap named in the PR #100/#102 review threads: **the actors' simulated actions
and the timing probabilities were partly disconnected** — a grounded refusal multiplied the deal
hazard directly, while the phase-4 policy choosing that actor's weekly actions never read the
refusal, and the hazard clock was almost entirely the fitted family curve. The system had two
parallel channels; the causal model demanded one chain:

```
quote (evidence)
  → stance(actor, mode)                 classification only — never LLM-minted magnitudes
  → conditions the actor's ACTION POLICY (phase-4 utility 'stance' component + binding commitments)
  → chosen actions change PATHWAY-PROCESS state (pathway_progress:* quantities)
  → other actors react (same loop, every cadence round)
  → hazard rounds CONSUME the process state (relative/multiplicative, bounded)
  → the event predicate eventually becomes true; absorption is OBSERVED (never resolved)
  → deadline probability, timing curve, mode-of-resolution = three readouts of the same trajectories
```

## The layers (all universal — no scenario branching anywhere)

1. **Question contract** — the resolution-criterion parser produces the absorbing predicate,
   polarity, deadline (`resolution_criteria`).
2. **Mode & pathway graph** (`mode_graph`) — canonical end-state decomposition:
   * `canonical_modes`: K independent elicitation passes reconciled with the compiler's structural
     hypotheses — id canonicalization (time indices stripped, shortest consensus name), token-overlap
     clustering, **majority vote across sources**, averaged priors, a consensus/agreement score in
     lineage. Compile variance in the mode set became a measured number instead of silent
     nondeterminism.
   * 13-pathway registry, actor-driven AND world-driven: cooperative_agreement, unilateral_action,
     institutional_procedure, operational_execution, competitive_interaction | threshold_crossing,
     diffusion_adoption, market_aggregation, physical_process, stochastic_external,
     resource_depletion, cascade_failure, scheduled_transition. **Stance logic is one mechanism
     family, not the organizing principle**: a hurricane has no stance; under `aggregation` stances
     shrink to near-irrelevance; under `none` they have no effect; world-driven modes consume the
     nonlinear/population state instead.
   * **Decision structures replace "most-opposed binds"**: each mode carries
     `{rule, approvers, stages}` and the stance-combination law is DERIVED — unanimity/weakest-link
     (veto logic: treaties), majority/weighted_coalition (the weighted center: bills), hierarchy/
     unilateral (the controller binds: resignations, launches), strongest_actor (contests),
     cumulative_pressure, aggregation, none. "Most-opposed binds" survives only as the unanimity
     case.
   * **Graded control** replaces `controls_pathway: bool`: sole_authority, veto, agenda_setting,
     partial_implementation, coalition_member, operational_capability, informal_influence, none —
     documented log-effect weights (a president may want a bill but lack the votes).
3. **Stances** (`resolution_criteria.ground_actor_intentions`) — **mode-scoped**:
   `stance(actor, mode)` via `target_mode`. Russia simultaneously *actively_pursuing*
   russian_victory, *committed_to_prevent* ukrainian_victory, *conditionally open* to a ceasefire —
   three stances, three different bindings. Per (actor, mode) the most specific stance wins; a
   stance targeting another mode never binds this one. Each stance carries reliability + capability
   (can they act on it) + graded control. Written three places: `plan._intention_stances` (hazard
   combiner), entity `stances` field (ActorView → policy), entity `commitments` (binding
   prohibitions for high-reliability categorical stances — the feasibility channel).
4. **Behavior** (`phase4_policy` / `phase4_execution`) —
   * `ACTION_PATHWAY_EFFECTS`: ontology-level signed effects of every action on every pathway
     process (accept +1.0 cooperative; reject −0.7; exit −1.0; escalate −0.5 cooperative,
     +0.4 unilateral; mobilize +0.5 unilateral; approve +0.8 institutional; launch +0.9
     operational; …). Ontology actions, never scenario keywords.
   * a `stance` utility component (broad-prior coefficient 1.5, documented) scores every candidate
     action by `pathway_orientation(stances, pathway) × effect` — consistency with one's own stated
     public commitments. Targeted prevent-stances contribute 0 to per-actor pathway orientation
     (opposing the rival's victory ≠ opposing military resolution).
   * decision opportunities now carry REAL candidate sets (pathway- and actor-type-derived), so a
     weekly review can express accept/reject/escalate/mobilize/approve instead of act/wait.
   * `_apply_pathway_effects`: an EXECUTED action writes a bounded step (0.04 × effect, clamped
     [0.05, 0.95]) onto every declared `pathway_progress:*` quantity it affects.
5. **Hazards** (`event_time`) — every hazard round consumes: its mode's OWN pathway process
   (weight 1.0), every other declared process at 0.25 (spillover: a collapsing battlefield forces
   parties to the table), world-driven couplings (nonlinear_state, population aggregates, 0.35) for
   non-actor modes, plus the pre-existing bounded channels. Consumption is RELATIVE (multiplicative,
   no-effect-centered, total clamp ×[0.25, 4]) for every hazard parameterization — the absolute
   blend destroys per-round hazards. The direct stance→hazard multiplier is log-split
   (`ENDOGENOUS_STANCE_SPLIT = 0.6`) whenever the behavioral channel is live, so the stance's
   effect is not double-counted between the direct and the behavioral route.
6. **Readout** — unchanged from PR #102: absorption observed, never resolved; binary = F(deadline);
   the timing CDF, censored mass, and mode×time marginal come from the same trajectories.

## Effect sizes: priors → measurements

* `experiments/replay_v3/fit_intention_hr.py` builds the labeled statement→hazard-change corpus:
  archived paired-date news per resolved market, LLM stance CLASSIFICATION only, effect MEASURED
  from the archived price path (implied-hazard inversion λ = −ln(1−p)/(T−t), median post/pre ±7d).
  Rows feed `fit_intention_hazard_ratios` (partial pooling toward no-effect); the resulting
  `intention_hr_pack.json` replaces the documented priors wholesale at load. Until it is run in a
  networked environment, provenance reports `documented_priors_unfitted` — honest.
* the resolution-time proxy is v2: sticky 0.9-crossing (a spike that collapses below 0.5 was not
  the event), early-close resolution timestamps preferred, scheduled-window denominator.

## The frozen event-time benchmark

`build_event_time_vault.py` freezes a FUTURE-WINDOW set of open markets (no outcomes exist at
freeze), SHA-256 sealed; `score_event_time_vault.py` refuses to run before the window closes,
verifies the seal, opens ONCE, and scores censoring-aware CRPS (`event_time.crps_first_passage`)
against the market-implied constant-hazard baseline + interval coverage + Brier-at-deadline —
when-type and deadline-type questions scored as one object. **Until that vault is built (network)
and scored (after its window), every event-time performance statement is development-split
evidence.**

## Evidence breadth

`requirements_from_plan` now derives per-actor recent-statement requirements (the stance grounding
substrate), declared-quantity measurement requirements (the order-of-battle class), and a
scheduled-calendar requirement, all from the plan's own structure; orchestrator caps raised
(3→8 retrieved requirements, 8→16 claim docs).

## Validation status (honest)

* 1142 tests pass (2 pre-existing environment failures outside world-model-v2). New: 19
  action-chain tests (including the end-to-end two-world proof that simulated behavior changes
  absorbed mass), 10 fitting/scoring tests, reworked decision-structure/mode-scoping tests.
* `experiments/replay_v3/offline_event_time_demo.py` runs the FULL post-evidence production path
  (canonical modes → process grounding → stance grounding → trajectory depth → event-time
  conversion → materialize → rollout → readout) with elicitations PINNED to the previous live
  compile of "When will the Russia-Ukraine conflict end?" — this environment's network policy
  blocks the LLM API and all evidence hosts, so the pinned-elicitation run is the strongest
  validation possible here; the rollout itself is LLM-free and production-exact. Cross-domain
  fixtures (Senate bill / product launch / inflation threshold) exercise majority, hierarchy, and
  world-driven aggregation structures on the same engine.
* Remaining known distance from maximal fidelity, ranked: (1) effect sizes still priors until
  `fit_intention_hr.py` runs on real data; (2) stances are static within a run — Phase 11 regime
  shifts do not yet rewrite stance records mid-trajectory; (3) capability is a 3-level
  classification, not a resource-conditioned quantity; (4) the pathway-process state is
  one scalar per pathway — no per-dyad negotiation state, no spatial battlefield state; (5) the
  frozen event-time vault has not yet been built/scored (needs network + its window).
