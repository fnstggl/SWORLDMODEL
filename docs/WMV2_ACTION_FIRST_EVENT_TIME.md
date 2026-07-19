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

`requirements_from_plan` derives per-actor recent-statement requirements (the stance grounding
substrate), declared-quantity measurement requirements (the order-of-battle class), and a
scheduled-calendar requirement, all from the plan's own structure; orchestrator caps raised
(3→8 retrieved requirements, 8→16→24 claim docs).

**The free-source stack (no paid APIs anywhere).** More Google RSS alone is NOT the answer — RSS
caps ~20 items per query and sometimes ignores after:/before: server-side. The layered design:

1. **GDELT 2.0 DOC API** (`GdeltDocConnector`) — keyless, thousands of outlets, PRECISE
   startdatetime/enddatetime windows, updated every 15 minutes since 2017. The BREADTH layer: the
   same requirement-driven facet queries (an actor's statements, a quantity's measurements, the
   calendar) fan out to dozens of dated articles — battlefield reporting, aid packages, domestic
   politics, alliance commitments arrive without any scenario-specific code. Published rate limit
   honored (module-wide ≥6.5s pacing; the plain-text advisory is classified http_error).
2. **As-of Wikipedia revisions** (existing connector, widened to 2 scoped entities per
   requirement) — the structured DEPTH layer: timeline articles, "List of military aid…",
   order-of-battle pages, negotiation-history pages carry exactly the battlefield/aid/negotiation
   state a war question needs, in one server-side-dated fetch each.
3. **Google News RSS** (existing) — the recency/headline layer.

Live measurement on the Ukraine question's facets: **~13 archived items → 127 dated items across
8 requirement-driven facets** (terminal, Putin/Zelenskyy/Trump statements, battlefield, aid
schedules, negotiations, calendar). Next free increments, each a slot not a redesign: Wayback CDX
snapshots of institutional pages (ISW, ministries), and Wikidata SPARQL for structured
composition/capacity facts.

## The world-dynamics layer (`world_dynamics.py`) — the five ranked gaps, closed

The five limitations the first build reported as-is are now engineering, not architecture:

1. **Effect sizes** — the statement→hazard-change measurement is PLACEBO-CONTROLLED (the same
   post/pre implied-hazard ratio at non-statement dates normalizes out the secular drift the naive
   ratio credits to the statement) and PATHWAY-STRATIFIED (a refusal moves a deal differently than
   a bill; strata pool toward the level estimate). Pack staleness is stamped into every conversion
   report (`hr_pack`: source / fitted_at / n_rows / stratified). The fit still needs one networked
   run — that is data acquisition, not design.
2. **Stances are DYNAMIC within a run** (`StanceReviewOperator`): grounded stances are initial
   conditions. At the trajectory cadence each actor's stances are reviewed against the causal state
   their own and their rivals' actions produced — RIPENESS (a rival mode nearing completion softens
   the loser's shared-pathway refusal: the collapsing battlefield opens talks), WINNING (hardens
   against concessions), EXHAUSTION (drained capacity softens pursue-stances), BANDWAGON (a
   succeeding shared process erodes weak opposition). One level per review, cooldown hysteresis,
   every change a StateDelta naming its rule. The policy reads stances live (behavior changes next
   decision); hazard rounds RE-DERIVE their stance hazard ratio from current records
   (stance-hash-keyed re-sampling) — h(t) genuinely shifts mid-trajectory.
3. **Process state is richer than one scalar per pathway**: contested (non-shared) pathways carry
   PER-MODE channels (`mode_progress:<pathway>:<mode>`) — an actor's actions advance THEIR pursued
   modes and suppress rivals' (sampled contested_suppression); shared processes weight PRINCIPALS
   (the decision structure's approvers) above bystanders; and **capability is a depletable capacity
   resource** — initialized from grounding, burned by effortful actions AND by CONTESTED ATTRITION
   (while ≥2 rivals pursue the same contested pathway, every cadence round drains both sides by a
   sampled coupling — wars of attrition exhaust by DURATION, not decision count), read live by the
   stance combiner and the exhaustion rule. Attrition → stance → behavior → hazard is one loop.
4. **Coupling magnitudes are PRIOR DISTRIBUTIONS, sampled per branch** (pathway_step, endogenous
   split, own/cross/world consume weights, contested suppression, persistence survival) — the
   structural-constant uncertainty reaches the terminal CDF; `fit_coupling_pack` replaces them from
   scored vault trajectories; `experiments/replay_v3/sensitivity_harness.py` sweeps them in-repo.
5. **The vault protocol is PROVEN offline** (`test_wmv2_vault_protocol.py`): build+seal, opens-only-
   after-window, never-rebuilt-in-place, time gate, tamper gate, single-open gate — all exercised
   end-to-end against a stubbed market API. The freeze itself needs one networked run.

Plus the deeper simulation gaps found in this pass:

* **PERSISTENCE SEMANTICS**: a criterion requiring the end-state to HOLD ("no active hostilities
  for ≥30 consecutive days") makes near-misses REAL — hazard success writes a PROVISIONAL
  absorption; the world pauses in the candidate end-state; a persistence check confirms (the
  criterion completes) or COLLAPSES it (the temporary ceasefire that fails now actually happens in
  trajectories, knocking the process back). Named near-miss states stopped being annotations.
* **CATEGORICAL UNIFICATION**: >2-option questions run the same first-passage machinery — the
  distribution over the question's own options is the absorbed_by marginal with honest
  none-of-the-options-by-horizon mass. When / deadline / categorical are now three readouts of one
  simulated object.
* **ACTOR PERCEPTION**: actors observe the PUBLIC process state (pathway/mode progress, population
  aggregates, nonlinear state) as beliefs — they were deciding blind to the world they acted in.
  Outcome/readout machinery (absorption stamps, sampled coefficients) never enters a view.
* **Timing resolution** scales with the horizon (up to 40 rounds per mode).

## Validation status (honest)

* Full suite green (2 pre-existing environment failures outside world-model-v2). New:
  world-dynamics (16), vault-protocol/fitter-rigor (6), action-chain (19), fitting/scoring (10).
* `experiments/replay_v3/offline_event_time_demo.py` runs the FULL post-evidence production path
  with elicitations PINNED to the previous live compile (this environment's network policy blocks
  the LLM API and all evidence hosts; the rollout itself is LLM-free and production-exact).
  Cross-domain fixtures (Senate bill / product launch / inflation threshold) exercise majority,
  hierarchy, and world-driven aggregation structures on the same engine.

## Is this the core vision? Assessment

The core vision: answers EMERGE from simulated causal worlds — real named actors with grounded,
evolving intentions and finite capacity; institutions with real rules executing inside the
trajectory; non-actor mechanisms alongside them; timing, deadline probability and mode of
resolution read out of one simulated object; every number either fitted, sampled from a documented
prior, or measured — never minted by an LLM mid-run.

**Architecturally the system now matches that vision.** What still separates it from the vision's
full realization is DATA and SCALE, not design:

1. the effect-size and coupling packs are unfitted until the networked fitting runs execute
   (interfaces, placebo controls, stratification all ready);
2. the frozen event-time vault is unbuilt until one networked freeze (protocol proven);
3. evidence per question is still thin relative to a world war (breadth now scales with plan
   structure, but archived-source depth is bounded by the connectors);
4. state granularity remains abstracted (no spatial battlefield, no per-dyad ledger beyond
   principals/contested channels, populations only when declared) — the honest next increments,
   each of which now has a natural slot in the mode-graph/process-channel design;
5. stance updates are rule-based (four documented universal rules); a fitted stance-transition
   model (from longitudinal statement corpora) is the data-driven replacement, and the
   StanceReviewOperator is exactly where it plugs in.
