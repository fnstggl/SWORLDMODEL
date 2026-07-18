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

## 13. Outreach action layer (v4 — reply-first, three separated judges)

The communication instance of the decision layer was rebuilt after a diagnosed failure (see
FORENSICS §10): the original path optimized messages against numeric recipient traits an LLM had
invented — a closed loop that produced debate-bait with no sender identity. The corrected
architecture (`swm/decision/`):

**Action level first** (`experiments/exp092_thiel_action_first.py`): before any wording, the
decision compares real routes — cold email, cold text, permission-ask, full memo, warm introduction
via an operator, operator-forwarded memo, routing through an adjacent partner, waiting for pilot
evidence, not contacting yet. Each action is an **arrival context** for the same behavioral engine;
Beckett-side path assumptions (P(operator agrees), delay discounts) are explicit, uncertain, and
reported. The best action is often not a better sentence.

**Behavioral engine = qualitative persona ensemble** (`persona_response.py`, built on the
qualitative-actor discipline of `swm/world_model_v2/qualitative_actor.py`): the recipient is
rendered as a **qualitative dossier** — evidence quotes, incentives, dispositions as text, never
invented numeric traits (universal: resolver evidence for public figures, user-supplied context for
private individuals). There is no single recipient model: **competing inbox-reality hypotheses**
(assistant-screens 0.35 / intros-only 0.25 / reads-own-bursts 0.15 / evidence-first 0.15 /
ignores-all-cold 0.10) each run their own first-person simulations — "You are X … this arrives …
what do you actually do?" — choosing ONE categorical outcome per draw from a **valenced vector**
(no_response / dismissive[cost] / curious / requests_material / refers / meeting). Probabilities are
**counted choices**, never asked-for numbers. A winner that leads under only one hypothesis is
flagged fragile; indistinguishable arms are reported as within-noise; the output is
"best-supported among tested", never "best possible".

**Structural prior + gates**: the conjunctive response **funnel** (`response_funnel.py`:
open × understand × believe × relevant × worth × easy, valenced objective P(positive) −
0.25·P(negative)) is the offline objective and stage diagnosis — one failed gate multiplies
through; clarity cannot buy back missing identity. The deterministic **content contract**
(`outreach_contract.py`) requires identity / thesis / evidence-with-provenance / relevance / tiny
next step, flags diligence-bait asks and unanchored extraordinary claims, and supplies the
plain-human baseline every optimized candidate must beat under the system's own evaluator.
Register gates (four-axis critic, numeric fact guard, redundancy, cold-read critic) filter; they
are never the objective. A **caricature guard** clamps combat-flavored situational levers to ≤0 and
shrinks persona-derived elasticities by evidence confidence.

**DEFAULT search method — the reply-first beat planner** (`reply_first.py`; wired as
`optimize_cold_outreach(method="reply_first")`, the default; `experiments/exp095_thiel_reply_first.py`).
Both earlier methods (whole-draft generate-and-rank, line-level iterative editing) still searched
for "the text an imagined recipient scores highest", and more search just exploited the imaginary
judge: jargon compounds, competing benchmark numbers, a fabricated biography line the judge liked.
The planner inverts the direction of design:

1. **Desired replies** — start from the exact positive replies wanted, in the recipient's own
   typed voice ("Interesting. Send it." / "What's the main technical insight?"), each mapped to an
   outcome category.
2. **Backward requirements** — for each target reply: what must the recipient have JUST READ for
   that reply to be their natural next keystroke? Decomposed into worthwhile / surprising /
   believable / noticed / effortless, grounded in the dossier and ONLY the sender facts.
3. **Beat structures** — a message is functional beats (recognition, identity, surprising_idea,
   evidence, relevance, request), not sentences. Several orderings are instantiated as complete
   drafts under hard writing rules: no jargon compounds (say what the thing DOES), **at most one
   number** translated into reader consequence with provenance in the same sentence, a request
   typed the way a real person types, 45–85 words.
4. **Beat search before wording search** — necessity tests (drop each beat), merges, and
   request-shape swaps, compared as complete drafts.
5. **Wording last** — one capped iterative-editor pass inside the winning structure (fabrication
   guard on).

**Three separated judges** certify every candidate; invention and certification are never the same
role on the same prompt:

* **Truth judge** (hard gate, unchanged — the judge that caught the fabricated "skipping
  Princeton" line): numeric fact guard (`allowed_numbers`) + deterministic content contract +
  fabricated-vs-facts sentence judge, failing **closed** when unverifiable.
* **Human-language judge** (hard gate; `language_judge.py`): would a sharp busy person actually
  TYPE each sentence? Harsh on formal-bot register (ceremonious permission constructions),
  unglossed jargon compounds, more than one big number, performed politeness. Carries the
  **preference-learning hook**: real human A-vs-B choices (`record_preference`) are folded into
  the judge prompt as calibration examples, so the automated critic gradually inherits the human
  editor's demonstrated taste instead of its own opinion.
* **Outcome judge** (rank only, never a gate-opener): the qualitative persona ensemble runs
  **blind** — shuffled anonymous labels, no authorship information — and ranks ONLY candidates
  that already passed both gates, so persona appeal can never resurrect a fabrication or
  bot-register text. Gate pooling prefers strictly-clean candidates (zero flags) over
  flagged-but-high-score ones, and a flagged finalist gets one targeted repair pass — judge
  flags become edits, accepted only if the repair re-passes BOTH gates. Its counted distributions stay in machine-readable traces; **no simulated
  reply percentage is ever reported to a human**. When the count gap is within noise the label is
  "best-supported candidate under the current assumptions; no reliable distinction between
  finalists" — and the stated cure is real outreach outcomes, not more simulation.

The planner emits exactly **ONE recommended message** (ties broken by language score, then
brevity, then deterministic order) — a slate of near-equals is decision fatigue, not honesty; the
non-selected finalists appear only as labeled ordinal notes. Every LLM call is traced
(stage, full prompt, full response) so a run is auditable end to end.

**Second search method — the iterative editor** (`iterative_editor.py`,
`experiments/exp094_thiel_iterative_editor.py`; retained as the planner's wording subroutine and as
a standalone comparison path via `method="slate"`): an exacting human editor's loop, mechanized.
Strategy-diverse seeds → whole-message diagnosis → per-location materially-different alternatives
(keep / rewrite / shorten / reframe / merge / **delete** / insert) → an independent judge comparing
**complete email variants** in context (never isolated sentences) → an 8-axis whole-email rescore
that **rejects locally-better lines that worsen the message** → endgame sweeps (per-line deletion,
reorder, add-beat, shorten, replace-ask, new opening, reframe) → a small beam with informed rewrite
+ crossover to escape local optima. Every step lands in a machine-readable edit trace. The internal
8-axis score is the editor's compass only; the same register bias applies everywhere (em dashes
discouraged as overused, allowed when genuinely best; sign-off dashes fine — a soft critic penalty,
no deterministic stripping).

**Calibration honesty**: additive persuasion elasticities are fit and graded on 19,714 real
ChangeMyView outcomes (held-out grade A, ECE ≈ 0.02) — that grade applies to the additive
persuasion model only. The funnel magnitudes are structural priors; the persona ensemble is a
model-based judgment (uncalibrated LLM role-play). All three labels are stamped on outputs; real
outreach outcomes accumulate through the prospective ledger.
