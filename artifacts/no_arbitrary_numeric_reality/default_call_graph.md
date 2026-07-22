# Default production call graph — where numbers used to enter, and what gates them now

## The default spine

```
swm/facade.py forecast(architecture="world_model_v2")
  └─ unified_runtime.simulate_world                       (no flags)
       └─ structural_runtime.simulate_structural_ensemble  (DEFAULT; single-model is an ablation)
            ├─ ensemble_compiler (independent structural generation + critics — critics are
            │   FORBIDDEN from minting probability/prior/weight fields, unchanged)
            ├─ per model: world_boundary + outside_world   (arrival rates provenance-gated, PR #123)
            ├─ per model: U._phase3_block                  (evidence posterior; ≥1 effective obs
            │   or absent — its lean-anchored prior is a RECORDED remaining assumption)
            ├─ per model: U._condition_plan
            │    ├─ resolution criterion parsing           (qualitative + literal deadline/window)
            │    ├─ canonical modes                        (support counts; NO priors)
            │    ├─ ground_process_states                  (qualitative {state, waiting_on, basis})
            │    ├─ declare_typed_processes                (string quantities; NO progress bars)
            │    ├─ ground_actor_intentions                (qualitative stances; literal binding
            │    │   instruments only; NO share quantity)
            │    ├─ temporal_compiler                      (scenario-generated triggers/channels)
            │    └─ event_time conversion:
            │         convert_binary_to_event_time / convert_to_event_time
            │           ├─ evidence-cited facts → deterministic absorbing events
            │           │   (observed_measurement; ledger-approved)
            │           ├─ institutional decisions → absorbing writers (institutional_rule)
            │           ├─ residual process ONLY from the evidence posterior
            │           │   (ledger-approved derived_deterministic)
            │           ├─ family survival/hazard packs → REGISTERED REJECTED (ineligible)
            │           └─ everything else → plan_record_unresolved (branch mass stays
            │               unresolved_mechanism at readout)
            ├─ per model: phase8_pipeline.run_with_persistence → materialize.run_from_plan
            │    ├─ queue_builder: skips refused first-passage specs (None from the gate)
            │    ├─ operators: HazardRound (provenance-gated success_prob only),
            │    │   FirstPassage (posterior-parameterized only), AbsorptionMonitor,
            │    │   PersistenceCheck (OBSERVATIONAL — no survival coin),
            │    │   institutional/aggregate consumers (posterior-only; family rung suppressed),
            │    │   StructuralProcessPrior + NetworkDiffusion (suppressed → unresolved),
            │    │   PopulationAggregation (observed/derived dists only),
            │    │   generated_world / causal_boundary     (typed consequences, unchanged)
            │    └─ EventTimeContract.project → branch terminal categories + bounds +
            │        unresolved mass (never normalized away)
            ├─ per model: pipeline.result_from_run → resolution_report + numeric_causal_inputs
            │   manifest + unresolved/partially_resolved statuses + withheld recommendations
            └─ _assemble_ensemble_result:
                 ├─ material disagreement → per-model conditionals + robust range,
                 │   NO headline average (partially_resolved)
                 ├─ agreement → shared conclusion (mixture served, labeled)
                 └─ all models unresolved → `unresolved`
phase13 api (separate entry): §31 withholding + §NAP unresolved-mass gate
```

## Where each removed number used to enter (before → after)

| Entry point | Before | After |
|---|---|---|
| `_condition_plan` | `ground_process_states` → 0.15…0.85 bars, sd 0.15, unknown 0.5; `declare_actor_capacity` 0.85/0.6/0.35 | qualitative typed records; no capacity resource |
| conversion (when/categorical) | per-mode first-passage intensity = LLM prior share × family curve × sampled stance HR × endogenous split, consuming progress at sampled weights | institutional/fact/generated absorbing writers, else `mode_transition:<id>` unresolved |
| conversion (binary) | residual chain: posterior → family rate (40 worlds) → LEAN_BETA; fact confidence as Bernoulli; progress consumption at 0.6 | posterior-only residual (ledger-registered); evidence-cited facts deterministic; else `residual_outcome_process` unresolved |
| rollout hazards | `_consume_state_hazard` ×2^weight clamps [0.25,4]; live stance-HR re-derivation; `sampled_coupling` draws | modulation = 1.0; no consume channels; no couplings |
| stance dynamics | numeric ripeness/winning/exhaustion/bandwagon rules over bars + capacity at 0.30/0.70 thresholds | the actor's own cognition at its real decision triggers |
| persistence | Bernoulli(sampled 0.75/0.85) | observational confirm/collapse via modeled breaking mechanisms |
| feasibility | `actions_advancing_pathway(min_effect=0.5)` prohibition sets from hand-authored magnitudes | literal instruments' own `explicit_prohibitions` only |
| terminal aggregation | equal-weight mixture served as headline | per-model conditionals + robust range under disagreement; mixture diagnostic only |
| Phase 13 | §31 truncation/under-modeled gates | + §NAP unresolved-mass gate (withheld) |
