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

RESULTS_PLACEHOLDER

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
