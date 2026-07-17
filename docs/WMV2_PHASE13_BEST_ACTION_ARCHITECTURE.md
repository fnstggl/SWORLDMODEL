# World Model V2 — Phase 13: Universal Best-Action Layer (Architecture)

Phase 13 is the universal decision layer for World Model V2. It transforms a decision-maker's
information, authority, resources, goals, constraints, and uncertainty into a validated feasible
action space, matched counterfactual simulations through the **canonical** unified runtime, robust
expected-utility / risk / regret analysis, and either a recommended action/policy, a Pareto frontier,
or a principled abstention — never a fabricated scalar.

It does **not** fork the simulator, does not create a parallel decision pipeline, and does not
directly mutate terminal probabilities. Every action becomes a canonical event.

Package: `swm/world_model_v2/phase13/`. Public API: `swm/world_model_v2/phase13/api.py`.

## 1. Canonical integration

The runtime-integration audit is machine-readable at `artifacts/phase13/runtime_audit.json`.

| Canonical seam | Phase 13 use |
| --- | --- |
| `compiler.WorldExecutionPlan` | `MatchedEvaluator.from_plan` compiles the decision world through the same plan contract |
| `materialize.build_world` / `operators_from_plan` / `queue_builder_from_plan` | the evaluator builds particles and the operator set from the plan — the plan's phase operators fire on every rollout |
| `rollout.RolloutEngine` | `crn.MatchedRolloutEngine` subclasses it — same event loop, StateDelta and follow-up semantics; only the RNG routing changes |
| `events.Event` + `register_event_type` | actions enter through the registered `decision_action` event; nothing bypasses the queue |
| `transitions.TransitionOperator` | `DecisionActionOperator` executes decision events through the canonical propose→validate→apply funnel |
| `transitions.observable_view` | the belief-state boundary for policies |
| `institutions.RuleSystem.validate_action` | the same executable rules the rollout enforces gate feasibility |
| Phase 11 recompilation | unchanged; structural change during a rollout triggers the canonical path |

`api._evaluator` accepts either a compiled `WorldExecutionPlan` (production path) or a dict of raw
runtime pieces (`initial/queue_builder/operators/contract`) for controlled tasks. Both drive the same
`MatchedEvaluator`.

## 2. Decision contract (`contracts.py`)

`DecisionProblem` is the complete typed contract, compiled before any action generation:
decision-maker + role + authority + controllable resources; as-of / horizon / decision points;
information sets (observable / private); candidate actions and `generated_action_permission`;
prohibitions; hard/soft/chance `ConstraintSpec`; `UtilitySpec`; `RiskSpec`; implementation/switching
costs; reversibility; information-gathering and human-approval flags.

The forecast question is never the objective. If `utility.provenance == "underspecified"` or no
stakeholder utility is supplied, the layer returns a Pareto frontier + missing-preference report +
abstention (`underspecification()` drives this) rather than inventing a scalar.

`DecisionResult` carries everything a caller needs to trust or challenge the recommendation: evaluated
actions/policies, feasibility verdicts (including rejections), the paired counterfactual block, VOI,
search diagnostics, the CRN manifest, causal-claim label, empirical-validation classification, support
grade, abstention status, active-phase census, provenance hashes, cost and latency. Never a bare
string.

## 3. Utility (`utility.py`) — decomposed, never one hidden scalar

Typed multi-stakeholder utility: per-stakeholder utility functions, weights, floors (minimum
guarantees, lexicographic), and rights (noncompensable predicates — a violation excludes the action
regardless of aggregate gain). Aggregations: weighted-sum, maximin, Nash social welfare, plus
distribution-level objectives (CVaR / chance-constrained / minimax-regret) applied in `robust.py`. The
result always shows the utility decomposition and a Pareto frontier for multi-stakeholder contracts.

## 4. Action ontology (`ontology.py`) — typed transformations, not a catalog

An action is `(actor, authority_basis, operation, object, params, timing, …)` where `operation`
comes from an **extensible registry** seeded with the nine cross-domain families (resources, time,
information, relationships, negotiation, institutional, operations, policy-control, meta). Domain
adapters register new operations through `register_operation` — there is no global switch statement.
`semantic_key()` deduplicates wording-only variants while preserving diverse families and deliberate
content variants.

## 5. Affordance-based generation (`affordances.py`) — the world proposes

Candidates are built from the actual world: authority-derived operations, controllable-quantity grids,
resource transfers along funding/control edges, institutional procedures naming the maker as a holder,
network-reachable contacts, disclosable private information, re-timing of pending events, plus the
mandatory baselines (do-nothing, defer, gather-information) and user candidates. A constrained LLM
proposer is one source among many; every proposal — LLM or not — passes deterministic validation and
the feasibility engine. The LLM may not set beliefs, outcomes, hidden state, or mint authority (those
are rejected with typed reasons, recorded).

## 6. Feasibility (`feasibility.py`) — typed verdicts, never silent drops

Checks authority, resources, timing, institutional rules (the canonical executable rules),
prohibitions, preconditions, network access, reversibility, and mutual exclusivity — each rejection a
typed reason code. Feasibility is state-dependent: the same precondition re-checks inside the rollout
when the action's event fires, so an action that becomes infeasible mid-policy fails loudly there too.

## 7. Intervention semantics (`interventions.py`) — action → intervention → event → StateDelta

The four-way distinction is structural. `to_intervention` produces a canonical `Intervention` whose
`apply()` only **schedules** the registered `decision_action` event (time-family operations also
re-time existing queue events — queue surgery, never state surgery). `DecisionActionOperator` executes
it through propose→validate→apply, re-checking preconditions and institutional rules at fire time,
drawing implementation failure from the action's own CRN stream, and emitting follow-up events
(`message_delivered`, `information_published`, `decision_opportunity`, `measurement`) that the plan's
own operators react to — that is what makes other-actor response real rather than a static delta. No
direct terminal-probability mutation; no Phase-13-only hidden state.

## 8. Matched counterfactual rollouts (`crn.py`, `counterfactual.py`)

Every compared alternative branches from the **same** posterior particle set with common random
numbers. `StreamRNG` derives an independent deterministic substream per named purpose
(`sha256(root_seed | stream_name)`), so an intervention that inserts an extra event consumes only its
own stream and unrelated exogenous shocks stay identical across arms. `verify_pairing` proves it: a
no-op arm reproduces the reference's exogenous trace exactly. Structural hypotheses are stratified by
particle index (identical assignment across arms). The report is built from paired differences
`U(action, particle) − U(reference, particle)` with the variance reduction **measured**, not asserted.

## 9. Robust evaluation (`robust.py`) — never rank on the mean alone

Per action: expected utility, distribution quantiles, CVaR, P(improvement) / P(material improvement) /
P(harm), constraint-violation probabilities, expected and minimax regret, per-structural-hypothesis
values with a fragility flag (an action winning under only one supported hypothesis is reported as
such), implementation cost, reversibility. The ranking objective is selected by the contract's
`RiskSpec` (expected / cvar / lower_confidence / minimax_regret / worst_hypothesis) and recorded.

## 10. Sequential policies (`policies.py`) & strategic reasoning (`strategic.py`)

A `Policy` maps the decision-maker's observable belief state (from `observable_view`) to an action;
`ContingentPlan` encodes observation-triggered branches and stop rules. Policies execute through
`PolicyExecutionOperator` at `decision_opportunity` events — the chosen action enters as a canonical
`decision_action` follow-up, same funnel as a one-step action. Conditioning on hidden state is
structurally impossible. `strategic.py` adds bounded iterated-best-response / level-k / quantal-response
for decisions where opponents optimize, with convergence reported, never silently claimed.

## 11. Value of information (`voi.py`) & search (`search.py`)

VOI computes EVPI/EVSI from the matched utility matrix (never confounded with world luck); it
recommends gathering when net EVSI exceeds the best immediate commitment's margin. Search compiles the
decision structure and selects the optimizer — exhaustive (small finite), successive-halving racing
(medium), coarse-to-fine hierarchical (large structured), policy-rollout (sequential) — with
feasibility respected during generation and a Part-19 correctness harness that measures the optimality
gap against exhaustive truth.

## 12. Public API (`api.py`) & safety (`abstain.py`)

One canonical interface: `recommend_action`, `evaluate_actions`, `optimize_policy`,
`value_of_information`. Simulation, recommendation, approval, and execution are separate — the API only
simulates and recommends; `human_approval_required` is stamped on every result. The system abstains
(with what is needed) when authority is unclear, utility is underspecified, a prohibited-harm marker
matches, or no substantive feasible action survives. Prohibited/coercive/deceptive actions targeting
protected or vulnerable groups are rejected.

## 13. Message action layer (cold-email optimizer)

The message optimizer (`swm/decision/`) is the concrete communication instance of the action layer:
L1 optimizes a strategy vector against a graded reply objective; L2 constructs the message move-by-move
by beam search (the world model selects, the LLM proposes single moves); L4 is a four-axis LLM critic
(coherent / annoying / AI-sounding / fabricated-vs-facts) with a deterministic numeric fact guard, a
redundancy detector, and a rewrite-or-prune repair loop that guarantees the gate's verdict and the
shipped text agree. `convenience_selling` is a first-class **negative objective lever** — performing
easiness / benefit-assurance backfires with a status-conscious skeptic (strong negative interaction
with `status_orientation` and `skepticism`), so the optimizer never builds convenience-selling in the
first place. The constructed message and its naive contrasts become typed Phase-13 actions evaluated
through the canonical matched-counterfactual funnel (see `experiments/exp090_thiel_phase13.py`).

Reply elasticities are fit and graded on 19,714 real ChangeMyView persuasion outcomes (held-out grade
A, ECE ~0.02); transporting that calibration to cold email is an explicit, stamped assumption.
