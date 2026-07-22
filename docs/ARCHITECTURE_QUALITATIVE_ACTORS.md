# Persistent Qualitative LLM Actors — architecture, audit, and evaluation

**The hypothesis under test (plain language).** A person's next decision is predicted better by
an LLM that *inhabits a persistent, person-specific qualitative worldview* — believes it is that
person, holds their private reality as text ("I think time favors me… I distrust my advisers'
optimism…"), interprets each new event through that worldview, updates it, and **chooses one
action** — than by reducing the person to numerical utility variables and predefined behavioral
equations. Probabilities are produced *afterward*, by counting the actions independently chosen
across many hidden-state hypotheses and samples — never by asking any single model call for a
distribution, and never by blending the choice with broad-prior utility scores.

Separation of concerns:

```
LLM actor       — what the person believes, fears, wants; how they read the moment; what they DO
simulation      — what information reaches them; what is possible; whether the attempt executes;
                  what consequences follow; what others observe
statistics      — how often each action was chosen across branches; uncertainty; calibration;
                  whether this beat the numerical baseline
```

Numbers exist outside the simulated mind (particle weights, aggregation, calibration, world
mechanisms, gating). They are not the mind's internal representation.

---

## 1. Audit of commit `cc17199` (persona-blended numeric policy) against this hypothesis

`cc17199` is preserved, renamed honestly as the **`persona_blended_numeric_policy`** baseline
(arm B). It does not implement the hypothesis. Violations, requirement by requirement:

| # | Requirement | Where `cc17199` violates it |
|---|---|---|
| 1 | No numerical cognition schema | The schema *demands* numbers: `inclination` per action, `confidence` 0..1, numeric `belief_updates` deltas clamped ±0.15 (`belief += 0.15` is exactly the prohibited form). The parser abstains unless numeric appraisals parse. |
| 2 | The LLM chooses an action | The LLM never chooses. It rates every option; `ActorPolicyModel.decide` + `UtilityInference` + policy families produce the anchor; `PersonaCalibration.blend` mixes; `posterior.sample` picks. |
| 3 | Real qualitative actor particles | No hidden-reality hypotheses exist. "Particles" are the numeric posterior worlds; cognition runs once on a *representative* particle; `PersonaCognition` is per-decision, not a persistent alternative reality. |
| 4 | Branch-specific decisions | Cross-particle prompt caching + representative-particle cognition give every branch the same cognition; `particle_prompts>1` **log-pools inclinations before deciding** (explicitly prohibited); `decide_and_execute_particles` executes one pooled action in every world. |
| 5 | Distribution from observed decisions | The distribution is a softmax of self-reported inclinations log-pooled with the numeric posterior. Nothing is counted; no clustering layer exists. |
| 6 | External calibration only | "Calibration" is a hand-chosen blend weight (0.5) and temperature prior. No raw-vs-calibrated pair of the required kind; no `unvalidated` labeling of an aggregated empirical distribution. |
| 7 | Persistent qualitative updates | Persistence is a 300-char reflection, scalar belief deltas, and short reaction strings — not a structured qualitative state with per-revision provenance. |
| 8 | Causal relevant-actor selection | An additive heuristic (stances +0.35, network +0.15, …) is primary. No authority/veto/implementation/resource/information analysis; no recorded question-specific reasons. |
| 9 | Single-individual mode | Absent — worse, the heuristic requires stances/capacity/network, so a dinner-question target scores ~0.2 and receives **no** cognition at the default scope. |
| 10 | Novel actions with executable meaning | Unknown names execute as `record_action` only; no mechanism compilation; no `novel_action_unmodeled` marking. |

**Q2 — does the current runtime permit different actions in different world particles?**
Structurally yes on the production rollout path: `WorldModelV2Run.run` gives each particle world
its own queue and `run_branch`, and `ProductionActorPolicyOperator.run` calls
`decide(None, [world], …)` per branch — branches *can* choose differently, and per-branch state
lives in each (deep-copied) world. Two things defeated it: (a) the Phase-3→4 bridge
`decide_and_execute_particles` pools one posterior and executes one action across all worlds;
(b) `cc17199`'s own representative-particle cognition + cross-branch prompt cache made every
branch share one cognition.

**Q3 — required refactor.** Per-branch qualitative state stored *in the world* (entity
`latent_state`, branch-isolated by particle deep-copies); deterministic per-branch hypothesis
assignment; a qualitative runtime whose `decide()` receives one branch world and returns a
**chosen** action (a degenerate observed-choice posterior recording alternatives — never
touching `UtilityInference`, policy-family scoring, or the blend); a post-rollout aggregation
layer counting choices across branch traces (cluster → raw distribution → external calibrator);
a mode router bound in `operators_from_plan`; a causal `RelevantActorSelector` with dynamic
promotion; a `NovelActionCompiler`; no cross-branch cognition caches in qualitative mode.

**Q4 — reusable from `llm_actor.py`.** The information-boundary prompt discipline and its
leakage tests; the abstaining-parse pattern (adapted: numerics are *rejected*, not clamped);
novel-target visibility validation; the `_post_execute` write-back hook; the
`operators_from_plan` binding seam; the budget/threading pattern; scripted-backend test
patterns; `action_menu` (as an options list, not a rating target).

**Q5 — baseline-only parts.** `PersonaCalibration` (softmax/log-pool), the
inclination/confidence schema, `LLMActorPolicyModel`'s blend, representative-particle pooling,
and the additive heuristic as *primary* selection. These remain runnable as arm B and are never
used by `persistent_qualitative_llm_policy` to select an action.

### 1b. What the numeric policy actually consumes from the rich `ActorView` (field audit)

Materially consumed by the numeric choice: `stances` (the stance utility term — the one strong
coefficient), `commitments[binding/prohibits]` + `resources` + `authority` +
`observed_evidence_ids`/belief *keys* (feasibility), `action_history` (habit count),
`policy_state[phase4_policy_value:*]` (RL/EWA), `relationships` (only Σ strength × 0.05),
`workload`/`attention`/`expected_reactions`/`beliefs` (only *presence*, gating family weights).

Never influencing the choice: `goals` (unread by the policy), the *content* of
`observed_events`, `remembered_events`, `beliefs_about_actors`, `preferences`, `incentives`,
`risk_beliefs`, `information_credibility`, `network_position`, and every uncertainty record —
`SubjectiveConsequenceModel` reads parameter-pack values and uses the view only as a provenance
hash. That is the architecture limitation this work addresses; it is not a data-and-scale gap.

---

## 2. The five policy modes

```
numeric_policy                    — the untouched Phase-4 numeric pipeline (arm A)
persona_blended_numeric_policy    — cc17199, honestly renamed (arm B, experimental baseline)
stateless_llm_policy              — qualitative loop with NO persistent state (arm C, ablation)
persistent_qualitative_llm_policy — the hypothesis (arm D)
hybrid_relevant_actor_policy      — arm D for consequential actors, numeric for routine actors,
                                    aggregate/mechanistic models untouched (arm E; the default)
```

Selected by `SWM_ACTOR_POLICY` (or programmatically). **Default-on wiring:** when an LLM backend
is supplied to the core V2 funnel, `materialize.operators_from_plan` binds
`hybrid_relevant_actor_policy`; with no backend, `numeric_policy`. Modes C/D/E never call
`ActorPolicyModel.decide`, `UtilityInference`, family scoring, or the persona blend to choose a
Tier-1 action. **Core-architecture update (docs/WMV2_CORE_ARCHITECTURE.md): on the DEFAULT
production path (`integrity="qualitative_strict"`), total LLM failure, budget exhaustion,
parse failure and Tier-3 routing NEVER produce a numeric decision — the §19.1 ladder (schema
salvage → same-family retries → comparable-family transitions → safe extraction) either yields
the actor's own decision or raises `ActorDecisionUnavailable`, and the branch STOPS with a
first-class truncation status. Routine actors are promoted to full qualitative actors rather
than handed a hidden numeric psychology. The numeric policy survives only as the explicitly
named baseline arm (`integrity="numeric_baseline_explicit"`) and the offline test-comparison
arena (`SWM_ALLOW_NUMERIC_BASELINE=1`, set in tests/conftest.py, never in production). The
marked-fallback description below documents that explicit baseline arm.**

## 3. Qualitative actor particles (`QualitativeActorState`)

Each Tier-1/2 actor gets **K mutually distinguishable qualitative hypotheses** about their
hidden reality, generated at first contact from evidence available at the simulation timestamp
(LLM-hypothesized; offline fallback produces labeled assumption-based variants). Hypothesis
`k = branch_index mod K` is assigned deterministically, persisted in that branch's world, and
**never leaks between branches** (worlds are independent deep copies).

State sections (primarily qualitative text or categorical records — scalars are rejected by the
parser): `identity_and_role`, `core_worldview`, `current_goals`, `fears_and_failure_conditions`,
`current_private_beliefs`, `beliefs_about_others`, `relationships`, `personal_condition`,
`organizational_pressures`, `commitments_and_identity_constraints`, `important_memories`,
`unresolved_uncertainties`, `evidence_basis`, `assumptions` — plus an append-only
`revision_log` carrying per-revision provenance (event, time, which sections changed).

## 4. The decision loop (per branch, per decision event)

1. Build the fail-closed `ActorView` (unchanged builder — the boundary holds by construction).
2. Load the branch's persistent qualitative state (initialize from the assigned hypothesis).
3. One LLM call: the persistent state + actor-local view + the new event + known feasible
   actions as *options* + the standing instructions ("you are the actor, not an analyst; you
   know only what is written here; maintain continuity; update only what the event justifies;
   choose what you would actually do — a supplied action, a modified action, a novel action,
   delegate, gather information, delay, or intentionally nothing").
4. Structured qualitative output: `actor_state_update` (revised sections, memories to preserve,
   unresolved uncertainties), `situation_interpretation`, `anticipated_reactions` (subjective —
   never treated as world truth), `decision` (act_or_wait, chosen_action, target, timing,
   observability, intended_effect, linked_actions), `novel_action_proposal`,
   `alternatives_considered`, `decision_summary`. **No numeric self-reports anywhere.**
5. The chosen action maps to a `TypedAction` (menu match, modified action, or
   `NovelActionCompiler`); deterministic validation (authority, resources, institutions,
   targets, commitments) decides what reality permits — a perceived-infeasible choice gets one
   bounded revision round; an actually-infeasible attempt executes as `action_blocked`.
6. The posterior recorded for the branch is the **observed choice** (`{chosen: 1.0}` with
   alternatives and full provenance; `llm_probability_minting: False` — nothing was minted, a
   decision was made). Total LLM failure ⇒ numeric fallback, `decision_source:
   numeric_fallback`, excluded from pure qualitative aggregation.
7. `_post_execute` persists the revised state, memories, and expectations onto the branch world
   on the same `StateDelta`, with a revision-log entry.

## 5. Distribution, clustering, calibration

After the rollout, `aggregate_actor_decisions` counts choices across branches:

```
raw_frequency(cluster) = Σ weight(branch selecting cluster) / Σ weight(all qualitative branches)
```

`ActionClusterer` (versioned `cluster-1.0`) groups semantically equivalent choices — exact
(name, target) first, then the compiled ontology anchor for novel actions; every original
selected action, its particle id, hypothesis, seed, evidence hash, cluster assignment, and
fallback status are preserved. `ActorPolicyCalibrator` then applies fitted actor→role→domain→
reference calibration *to the aggregated distribution* when a fitted pack exists; otherwise the
raw distribution is returned labeled **`unvalidated`**. Both `raw` and `calibrated` are always
kept. A blend weight or temperature prior is not calibration.

## 6. Relevant-actor selection and dynamic promotion

`RelevantActorSelector` assigns tiers question-specifically from the compiled plan's causal
structure — direct decision authority (institution `decision_right` holders), scheduled
`actor_decisions` participants, pathway principals, veto/blocking rules, stance-carrying
capability actors, resource/implementation control, persuasive access to principals (network
edges), and *reaction-is-the-question* (single-individual mode → automatic Tier 1, no stance/
network/capacity precondition). Every assignment records its reasons. The additive heuristic
survives only as a fallback for bare worlds. During the rollout, an event participant outside
the tier map is re-scored at event time (promotion recorded in `world.uncertainty_meta`).

## 7. Single-individual mode

`IndividualReactionSimulator` routes "how will this person react to X?" questions through the
same core architecture: the target is automatically Tier 1; K qualitative hidden-state
hypotheses are built from the supplied relationship history; the stimulus is delivered exactly
as the person would experience it; each particle independently interprets, reacts internally,
and chooses an observable response; responses are aggregated and calibrated (or labeled
`unvalidated`). No formal stances, institutional capacity, or network degree are required.

## 8. Novel actions must mean something

`NovelActionCompiler` attempts a bounded translation of a novel proposal into executable
mechanics: target resolution (visible entities/institutions only), communication mechanisms
(message delivery, reaction scheduling), institutional submissions, resource costs (effortful
class), delayed effects (timing), observability (public/private), and — where the proposal's
intended effect matches a declared causal pathway — the ontology anchor whose validated pathway
effects it inherits. Compiled mechanisms are validated before execution. If no causal reading
is supported, the action still executes as a record but the branch is explicitly marked
**`novel_action_unmodeled`** in the trace and delta — never a silent no-op.

## 9. Evaluation

`experiments/actor_policy_benchmark.py` runs frozen historical decision cases (evidence cut at
the prediction timestamp; leakage-checked) under all five arms and reports next-action
accuracy, top-k, log loss, Brier, calibration error, novel-action coverage, cost, and latency —
plus the two required demonstrations (multi-actor geopolitical; single-individual reaction).
Honest-claims ladder used in the report: implemented / mechanically verified / plausible /
backtested / calibrated / statistically better than baseline / still speculative.
