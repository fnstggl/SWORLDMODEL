# Generated actor-mediated world — demos + matched evaluation (honest report)

**What this measures.** The production default (`generated_actor_mediated_world`) against
its three baselines on real DeepSeek runs: does each scenario generate its OWN semantics, do
actions become scenario-typed direct effects, do other actors respond only through their own
persistent invocations, and is the answer read from generated records via frozen predicates —
with the invariants `human_reactions_written_directly == 0` and `fixed_ontology_uses == 0`
holding in pure runs? Backend: `deepseek-v4-flash` for schema compilation, decisions, and
direct-effect compilation (all untrusted, all validated). Artifacts:
`experiments/results/generated_*.json`. Architecture + the 15-question audit:
`docs/GENERATED_WORLD.md`.

## 1. The matched 4-mode evaluation

One settlement scenario (two leaders, two mediated rounds, 8 particles, seed 23), identical
worlds and kickoffs; C and D share ONE pre-compiled generated schema so the actor-policy axis
is isolated. The generated arms read their answers from frozen predicates over generated
records; the baselines read their historical bar contract.

| Arm | Architecture × actors | Distribution | ops applied | events | delivered | invoked | declined | wall |
|---|---|---|---|---|---|---|---|---|
| A | legacy scalar × persistent | no_deal 1.000 | 0 | 0 | 0 | 0 | 0 | 30m |
| B | fixed-v1 catalog × persistent | no_deal 1.000 | 864 | 0 | 0 | 0 | 0 | 50m |
| C | generated × stateless | **settlement_reached 0.375** / no_settlement 0.625 | 72 | 54 | 54 | 50 | 6 | 17m |
| D | generated × persistent (**production default**) | no approval by deadline 1.000 | 91 | 70 | 63 | 59 | 13 | 21m |

- **The generated arms are the only ones whose distributions come from actor-mediated
  recursion over scenario-generated state**: 50–59 control-plane invocations per run, each
  a persistent-LLM decision on a delivered observation, with 6–13 deliberate declines
  (waiting is real behavior, not a missing handler). Invariants held in both:
  `human_reactions_written_directly = 0`, `fixed_ontology_uses = 0`.
- **Branch divergence is genuine** in arm C: three of eight particles reached a settlement
  record satisfying the frozen predicate; the others stalled with signed-by-one or lapsed
  deadlines — inspectable event-by-event in the world planes.
- **Do not read C > D as an actor-policy ranking**: one scenario, one seed, n=8; D's
  persistent hypotheses declined to act twice as often (13 vs 6), which at this scale can
  swing the binary either way. The arms demonstrate the CONSEQUENCE architecture; the
  50-case corpus remains the actor-policy evidence.
- **Cost**: generated runs are 17–21 min (vs 50 min fixed-v1) — the cascade is
  budget-bounded (5 invocations/actor) instead of chain-till-the-call-budget.

## 2. The five demos (all through the real funnel, DeepSeek end to end)

**A — corporate product + rebrand.** The schema compiler generated
`anchor_partner_signing`, `experiences_marketplace`, `marketplace_public_launch`,
`rebrand_completion`… (no Meridian/Airbnb-like type exists in repository code) and the
answer is read from those records against the frozen predicate. (One draw compiled a
degraded schema and served fixed-v1, STAMPED — kept in the logs as an honest example of the
loud degradation path; the committed artifact is a pure generated run.)

**B — board cascade** (5 actors, no menus anywhere): the CEO privately approaches the CFO
("meet privately with Arman to understand and neutralize his concerns"); the chair
pre-coordinates with the CEO; the fund director deliberately WAITS and monitors; the
formerly peripheral independent director proactively calls the chair; members then write
real `director_vote_record`s and the outcome is deterministic arithmetic over them — the
final answer is literally "fewer than 3 'approve' votes recorded by 2023-12-05". 120 ops,
95 events, 100 invocations, 7 declines.

**C — individual communication.** The reply schema (`trip_cancellation_by_friend`,
`jordan_expresses_reaction`) was generated for the question; all six samples decided in
pure generated mode with hypothesis-specific stances ("without offering an easy out",
"quietly lower expectations"); `reply_now` 1.0 with the reply text as scenario events.

**D — novel cross-domain (semiconductor supplier qualification).** Generated types
(`corrective_action_report`, requalification records…) nowhere in the repo; the
distribution SPLIT — "does not requalify" 0.75 / "Q3 order placed on or before
2024-01-13" 0.25 — from 59 invocations across supplier/customer/quality-owner cascades.

**E — non-human boundary (launch anomaly).** The schema declared the physics UNRESOLVED in
engineering language ("whether the turbopump hardware … is actually flightworthy cannot be
resolved by assertion") and the simulation never let anyone assert it: humans decided
(root-cause investigation, window management, range-safety consultation — 56 events, 93
deliveries), the flightworthiness stayed unestablished, and the frozen predicate resolved
"does NOT fly inside the window" 1.0.

## 3. What the fallback counters show (and why they matter)

A large share of demo-B/D events used the schema-scoped `unmodeled_actor_action`
scaffolding: the direct-effect compiler could not map every subtle political action onto
declared types, so the EXACT decision text was preserved, delivered, and recursed on — with
every such use counted as a fallback. This is the honest boundary of current compile
quality: the causal loop never breaks, and the artifacts show precisely how much of each
run's semantics were modeled vs carried as content. The development history in this branch's
commits is itself the measurement: wrapper-shaped ops, invented vocabulary, and undeclared
fields were each surfaced BY the loud-validation design and fixed without ever silently
corrupting a world.

## What is mechanically guaranteed (1336 tests, incl. the 34 required invariants)

- Production mode has **no global catalogs**: records/events validate only against the
  branch's generated schema; two entirely different domains run on one kernel; scenario
  types and events need no repository registration; the schema extends during rollout,
  versioned and branch-isolated.
- **Control ≠ world**: `ctrl_*` scheduler tasks never appear in the semantic history;
  delivery updates one actor's information state and triggers reconsideration through the
  control plane; the actor may wait, pick an affordance example, or invent an action; no
  generic acknowledge/ignore menu exists anywhere in production mode.
- **No social causality in code**: the kernel exposes no mind-write operation; compiler
  attempts are quarantined and counted; another actor's response comes only from their own
  policy; sender expectations stay subjective actor-local state.
- **Visibility is enforced**: public events reach all persons (with rule-declared channel
  delays and representations), private events reach only participants; different recipients
  can receive different representations of one source event.
- **Recursion is canonical and bounded**: cascades run through the production RolloutEngine
  queue; (actor, event) dedup + per-actor invocation budgets + cascade-depth caps terminate
  loudly; institutional arithmetic counts actual member decision records and an empty tally
  decides nothing.
- **Predicates over generated records** resolve the question; no op can mint probability/
  forecast fields; the `ACTION_PATHWAY_EFFECTS × pathway_step` writer and the fixed-v1
  handlers raise/never run under the production mode; both remain runnable as explicit
  baselines.
- **Degradation is stamped, never silent**: a schema-less world under the generated default
  serves fixed-v1 with `actual_mode`, `degraded`, `fixed_ontology_uses` and the reason on
  the report; pure generated evaluations exclude degraded runs.

## Honest limitations

- **No predictive-improvement claim.** These runs demonstrate the causal ARCHITECTURE.
  Consequence/trajectory accuracy against frozen historical intermediate facts — and the
  post-cutoff forward corpus from the earlier reports — remain the declared measurement
  work. Richer generated worlds are not evidence of better forecasts.
- **Schema quality is backend-bound.** Validation + auto-repair + critic make the schema
  SAFE, not insightful; missing decisive mechanisms remain possible and are only partially
  caught by the critic. Quarantine and fallback counts in the artifacts show how much of
  each run's semantics the compiler failed to model.
- **Populations and non-human mechanisms**: population responses are represented and
  opened, not resolved, without fitted models; physical questions without validated
  mechanisms stay labeled `unresolved` (demo E) — the simulation refuses to let anyone
  assert the physics.
- **Cost**: schema compilation adds 2–4 calls per run; every actor invocation costs a
  decide + a direct-effect compile call; cascades multiply this (budget-bounded). Wall
  times per artifact are in the JSONs.

## Reproduction

```
PYTHONPATH=. python experiments/generated_world_demo.py smoke        # offline
DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/generated_world_demo.py demoA
…demoB / demoC / demoD / demoE / armA / armB / armC / armD / combine
```
