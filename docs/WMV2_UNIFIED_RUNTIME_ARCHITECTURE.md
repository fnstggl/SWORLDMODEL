# WMv2 Unified Runtime — Architecture and Audit

*One canonical, maximum-capacity, default-on World Model V2 runtime that threads Phases 1–11 through a single
question→terminal path. Every claim here is backed by code and by the machine-readable artifacts under
`experiments/results/unified/`. Prior positive/negative/null results are preserved.*

## Part A — ruthless integration audit (verified in code at base commit 7d57dac)

`experiments/results/unified/integration_audit.json` records the pre-unification state:

- **Public facade → `pipeline.simulate`** (`swm/facade.py:87`) — the lightweight path with **no numeric
  posterior**. It was the ONLY V2 production path.
- **Single execution funnel:** `materialize.run_from_plan → rollout.RolloutEngine.run_branch` (one event queue,
  one `StateDelta` protocol). Phases 4/6/10 reach it by registering operators into `transitions._OPERATORS` and
  fire only if the compiled plan names them.
- **Orphans (no production caller):** Phase 3 posterior (`simulate_with_posterior`), Phase 9
  populations/networks (`simulate_with_populations_networks`), Phase 11 dynamic recompilation
  (`RecompilationController.run` — referenced only in its own module + tests). **Phase 7 nonlinear was
  CLI-only** (no runtime importer, though `nonlinear/operators.py` already defined real `TransitionOperator`s).
- **Phase 8 persistence** was the one non-core phase already wired (via `pipeline.simulate(persistence=…)`).
- **One plan type** (`compiler.WorldExecutionPlan`), one V2 `WorldState`/`StateDelta`
  (`state.py`+`transitions.py`); the only competitors are frozen v1 types under `swm/state`, `swm/api` (not
  reachable from the facade).
- **No unified orchestrator existed.**

## The old fragmented call paths

```
facade.forecast(world_model_v2)          -> pipeline.simulate            (no posterior)      [PUBLIC]
experiments/tests                        -> simulate_with_posterior      (P1+P2+P3)          [orphan]
experiments/tests                        -> simulate_with_populations_networks (P9)          [orphan]
phase11 tests                            -> RecompilationController.run   (P11)              [orphan]
nonlinear CLI                            -> nonlinear/__main__            (P7)               [CLI-only]
pipeline.simulate(persistence=…)         -> run_with_persistence          (P8)               [wired]
```

## The new canonical path (`swm/world_model_v2/unified_runtime.py::simulate_world`)

```
simulate_world(question, as_of, horizon, intervention, user_context, prior_checkpoint,
               compute_budget, seed, llm, execution_policy, trace_level)
  → Phase 1  compile_world                         → the ONE shared WorldExecutionPlan
  → Phase 2  gather_evidence (strict as-of)         → bundle → recompile_with_evidence → attach observations
  → Phase 3  tag_claims → build_outcome_rate_prior → infer_posterior → materialize particles ONTO the plan
  → Phase 9  populations/networks instantiated into the plan when the compiler declares them
  → Phase 11 RecompilationController over the as-of observations, on the SAME plan lineage
  → Phase 8  run_with_persistence terminal rollout THROUGH THE ONE FUNNEL, which fires the
             Phase 4 actor-policy, Phase 6/7 registry, and Phase 10 institution operators the plan names
  → one terminal WorldState distribution → raw_probability
  → SimulationResult + active-component manifest + plan lineage + Phase-12 incompatibility marker
```

The facade's `world_model_v2` branch now calls `simulate_world` (a static test,
`test_facade_routes_v2_to_unified_runtime_not_lightweight_pipeline`, fails if the lightweight bypass is
reintroduced). The legacy `pipeline.simulate` remains only as an internal compatibility helper.

## Default-on semantics

The ordinary caller passes **no** `use_posterior` / `enable_persistence` / `with_networks` /
`use_institutions` / `nonlinear` / `dynamic_recompile` / `maximum_capacity` flags — a test
(`test_no_phase_opt_in_flags_in_signature`) enforces their absence. Every completed phase is available
automatically; the compiler selects causally-relevant subsystems; each omission is recorded with a reason
(`causally_irrelevant` vs `unavailable`). `execution_policy={'drop_phases':[…]}` exists ONLY for the causal-
ablation harness, not for normal callers.

## Shared contracts

One `WorldExecutionPlan`, one `WorldState`/`StateDelta` lineage, one event queue (`RolloutEngine`), one terminal
projection. Phases mutate the same plan in place (`plan.posterior_rate_particles`, `plan.structural_posterior`,
`plan.population_latent_specs`); Phase 11 revises the same plan lineage; no probability-level phase ensembling.

## What became integrated in this run

| capability | before | after |
|---|---|---|
| Public V2 path | `pipeline.simulate` (no posterior) | `simulate_world` (unified) |
| Phase 3 posterior | orphan / experiments-only | **default-on**, materialized onto the plan |
| Phase 11 recompilation | orphan / tests-only | **default-on** loop over as-of observations |
| Phase 7 nonlinear | CLI-only | **runtime-registered** operators (fire when the plan names them) |
| Phase 8 persistence | opt-in arg | default-on terminal rollout |
| Phase 9 populations/networks | orphan pipeline | instantiated into the plan when the compiler declares them (see limitations) |

## Legacy compatibility

`pipeline.simulate`, `simulate_with_posterior`, `simulate_with_populations_networks`,
`simulate_with_persistence`, and the nonlinear CLI remain callable for isolated experiments, offline fitting,
diagnostic ablations, and backward-compatible artifact replay. They are no longer the public production path.

## Phase 12 invalidation

The unified runtime changes the forecast distribution, so the **pre-unification Phase-12 calibrator is marked
INCOMPATIBLE** (`res.provenance["calibration_compatibility"]`; `phase12_serve.compatible_with(...,
phase11_present=True)` refuses it). After this PR merges, Phase 12 must regenerate its corpus from the unified
runtime and refit: `PYTHONPATH=. python experiments/phase12_refit.py --regen`. **This run does not refit or
finalize Phase 12 calibration** (per the mandate).

## Honest integration-depth limitations (see the validation doc for ablation evidence)

- P1/P2/P3/P8/P11 execute on every eligible question through the one funnel; ablation shows P2/P3/P8 materially
  change the terminal.
- P4/P6/P10 fire when the compiler selects their operators (conditional, unchanged).
- P7 nonlinear operators are now runtime-registered but the compiler still rarely selects them for general
  forecasting questions — availability is fixed; selection frequency is a continuation item.
- P9 populations/networks are instantiated only when the compiler emits `PopulationSpec`/network objects, which
  it does not yet do for most general questions — a real remaining wiring gap, honestly recorded per run.
