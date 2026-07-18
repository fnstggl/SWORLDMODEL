# Scenario-Generated Action Layer (Phase 13 v2 — production generated mode)

The fixed Phase 13 operation catalog is **legacy**. In generated mode the system no longer
chooses a loosely fitting verb from a global registry, attaches free-form params, executes a
few hardcoded semantics, and records everything else as a `past_actions` stamp. The audit
that condemned that path is machine-readable at
`artifacts/phase13/action_language/audit_phase13_catalog.json` (61 findings; 65 of the 100
catalog verbs executed as history-only records; family-based search pruned on a generic
scalar; user actions coerced or rejected).

## The production chain

```
decision problem + generated scenario world (scenario_schema.py / generated_world.py)
→ explicit contract: goal, constraints, authority, resources, horizon, risk (contracts.py)
→ ScenarioActionLanguage generated from THAT world        (scenario_actions/language.py)
→ concrete actions / contingent plans — no verb labels    (scenario_actions/candidates.py)
→ deterministic feasibility + authority validation,
  per world hypothesis, revalidated at execution          (scenario_actions/feasibility.py)
→ compiled ONCE into scenario-native direct effects
  (generated-world kernel ops; mind-writes, terminal
  writes, undeclared vocabulary rejected statically)      (scenario_actions/compiler.py)
→ executed through the canonical runtime: plan-step
  events apply the precompiled ops; the generated
  control plane routes observations; affected actors
  react through their own persistent simulations          (scenario_actions/execution.py)
→ matched full-world counterfactual rollouts (CRN)        (phase13/counterfactual.py)
→ scenario goal contract read from evolved records:
  counted frequencies + real quantities, lexicographic
  gates + Pareto — no minted utilities or progress bars   (scenario_actions/goals.py)
→ trajectory diagnosis: earliest causal break, typed      (scenario_actions/diagnosis.py)
→ diagnosis-directed revision of the ACTION, ancestry
  preserved, rerun through the same matched worlds        (scenario_actions/generated_search.py)
→ robust recommendation, Pareto set, or abstention        (scenario_actions/report.py)
```

## Entry points

- `phase13.api.recommend_action(problem, world_context)` — **default**: a generated world
  context routes here automatically (`mode="auto"`); `mode="legacy_fixed_v1"` is the only
  door to the old catalog (baselines/ablations/frozen tests). A generated world can never
  silently fall back to fixed-v1; a missing scenario schema raises, classified structurally
  under-modeled.
- `phase13.api.evaluate_actions(problem, actions, world_context)` — supplied candidates
  only; differs from `recommend_action` ONLY in candidate provenance; never mutates the
  caller's `DecisionProblem`.
- `scenario_actions.api.discover_best_action(goal, context, problem=…)` — goal-backward
  discovery (the 12-step pipeline below).
- `scenario_actions.api.evaluate_proposed_actions(goal, ["free-text action", …], context,
  problem=…)` — arbitrary natural-language candidates, preserved verbatim, compiled against
  the scenario language, run through the full world. An action absent from every source
  file executes if its scenario semantics compile; otherwise it is partially modeled with
  every unresolved step exposed (scaffold events preserve exact content and are counted),
  or rejected as unmodeled — never a history-only fake execution.
- `scenario_actions.api.optimize_policy_generated` — contingent plans whose steps carry
  observation-predicates evaluated ONLY on the decision-maker's observable projection
  (visible records + delivered information; hidden state is structurally out of reach).

## The action language (per decision, per world)

`ScenarioActionLanguage`: decision maker, verified controllable objects, verified authority
sources (contract ∪ schema role ∪ institution holderships — never invented), information
boundaries, channels, institutions + procedures, real resources with live holdings,
deadlines, timing opportunities anchored to scenario records, relevant actors,
scenario-native action DIMENSIONS (open-ended example axes, never a menu), valid
combinations, the direct-effect compiler contract (the 7 semantically-empty kernel ops +
this schema's vocabulary), feasibility rules, and unresolved affordances (every claim the
deterministic validator could not ground — surfaced, not trusted). With no LLM the language
degrades LOUDLY to a deterministic schema projection.

## Kernel discipline

The kernel (`generated_world.KERNEL_OPS`, 7 ops) stays semantically empty: records,
relations, semantic events, conserved transfers, versioned schema extension. Substantive
meanings (launching, hiring, persuading, sanctioning…) exist only as scenario-generated
types. Static compiler gates reject: non-kernel ops, mind-writes (`_MIND_WRITE`),
numeric-minting fields, undeclared vocabulary (one repair round with the valid ids),
institutional decision records written by non-holders, and direct terminal-outcome writes
(an op that would itself satisfy an outcome predicate on a record type the maker lacks sole
authority over). Downstream consequences travel exclusively through observation delivery
and affected actors' own simulations.

## Goal-backward discovery (the generalized reply-first insight)

desired world states → backward requirements (what must be true just before success) →
causal levers (direct / other-actor-voluntary / institutional / outside-boundary) →
strategy structures from THREE independent generators (goal-backward, forward-affordance,
orthogonal) → concrete plans (exact steps, content, terms, timing, contingencies, stop
rules) → SIX independent critics (omission, feasibility/authority, mechanism,
domain-reality, goal-gaming, implementation) whose findings map to structural gates or
surfaced flags — a critic's dislike never eliminates → compile → screen → matched simulate
→ diagnose → revise (ancestry preserved; a revision that worsens forbidden-state frequency
vs its parent is rejected) → rerun → blind adjudication (shuffled labels, no provenance;
cannot override the deterministic comparison; disagreement is surfaced). Diversity is
measured and reported (strategy classes, targets, timings, channels, sequence lengths,
still-missing classes per the omission critic).

## Evaluation honesty

No universal progress scalar exists on this path. Ranking is lexicographic over typed
evidence: forbidden-state hits → success frequency (counted over matched, feasibility-
masked particles) → near-misses → declared-direction real quantities; worst-hypothesis
support breaks ties under `risk.robustness="worst_hypothesis"`. Unstated preferences yield
the Pareto set plus the exact missing preference, never a minted weighted sum. The final
claim is always *best-supported among the considered feasible actions under the stated
goal, constraints, world hypotheses, and simulation support*. Implementation uncertainty is
never an invented `failure_prob`: it is world mechanisms, user assumptions (labeled), or an
unresolved marker.

## Message composition (PR #115 preserved)

When a plan step is a consequential person-to-person message, the general planner decides
whether/to whom/why; an injected realizer may hand the exact wording to the reply-first
planner (`swm/decision/reply_first.py`) whose truth/language/outcome gates stay intact; the
realized text embeds into the step's `exact_content` and the full world simulation judges
what happens after it is sent. Messaging competes against non-message strategies inside the
same comparison; nothing became email-shaped.

## Enforcement

- `tests/test_scenario_action_layer.py` — end-to-end integration (offline, canonical engine).
- `tests/test_scenario_action_invariants.py` — the §18 invariants as executable tests.
- `tests/test_scenario_action_enforcement.py` — AST gate: the package may not import the
  legacy registry, may not carry a resurrected global verb list, and the kernel stays ≤ 8
  storage-mechanics ops.
- `tests/test_scenario_cross_domain.py` — 15+ materially different scenario fixtures
  (test-only; production has no domain branches), including randomized type names generated
  at test time.
- `benchmarks/phase13/scenario_action_acceptance.py` — regenerates
  `artifacts/phase13/action_language/acceptance_report.json` from executed checks.
- Live probes: `experiments/exp096_scenario_action_probes.py` →
  `artifacts/phase13/action_language/probes/` (architecture probes, not accuracy claims).

## What remains unvalidated

Simulated counterfactuals are `simulated_mechanism_counterfactual` claims; no backtest,
calibration, or prospective validation of THIS layer's recommendations exists yet. Actor
reactions are LLM role-play (unvalidated label rides on generated worlds). The legacy
catalog remains for frozen tests and explicit baselines only.
