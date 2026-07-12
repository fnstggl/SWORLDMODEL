# WMv2 Reproducibility Commands

All runs are deterministic under fixed seeds. Long runs are resumable via committed caches. Every result
file records `llm_calls`, `est_cost_usd`, `runtime_s`. LLM arms need `DEEPSEEK_API_KEY`; pure-compute arms
need none.

## Environment

```bash
pip install pytest                      # the swm core is dependency-free; tests need pytest
export DEEPSEEK_API_KEY=...             # only for LLM arms (compiler, BehaviorBench, historical)
export PYTHONPATH=.
```

## Tests (all)

```bash
python -m pytest tests/ -q             # 720+ tests; excludes fastapi-dependent test_state_world_model/test_agent_engine
```

## Build the machine-readable mechanism registry

```bash
python -m swm.world_model_v2.registry.build_registry   # writes registry/data/{registry,packs}.json
```

## Empirical benchmarks (each writes experiments/results/*.json)

```bash
# Phase 7 — nonlinear diffusion (Higgs); pure compute, resumable cohort cache
#   downloads SNAP higgs-twitter to data/higgs/ on first run
python -m experiments.wmv2_higgs_nonlinear_run --n-sample 4000 --particles 30

# Phase 4 — learned actor policy (BehaviorBench); pulls HF game data, no LLM
python -m experiments.wmv2_behaviorbench_policy_run

# Phase 8 — persistence at power (OmniBehavior); downloads cohort to data/omnibehavior/
python -m experiments.wmv2_persistence_power --n-users 140

# Phase 12 — calibration validation (forecasting corpus); pure compute
python -m experiments.wmv2_calibration_validation

# Phase 13 — best-action (matched CF + Upworthy randomized A/B); pure compute
python -m experiments.wmv2_best_action

# Phase 1 — compiler generality (real LLM, 104 held-out NL questions); resumable per-question cache
python -m experiments.wmv2_compiler_generality --jury-sample 24

# Phase 15 — historical forecasting (real LLM + as-of retrieval); resumable per-question cache
python -m experiments.wmv2_historical_benchmark --limit 60
```

## Determinism & resumability

- Seeds are fixed in every runner (`--seed` where applicable; defaults committed).
- Delete a benchmark's cache dir under `experiments/results/<name>/` to force a clean re-run.
- The registry store verifies a sha256 integrity hash on load and refuses a corrupted registry.
- `Date.now()`/randomness are seeded; no wall-clock nondeterminism in the scored paths.

## Data provenance (licenses)

- SNAP higgs-twitter (public research dataset)
- BehaviorBench moblab/game_behavior (CC-BY-NC-ND — benchmark use only)
- OmniBehavior / Kuaishou (CC-BY-NC-SA — benchmark use only)
- Upworthy Research Archive (CC-BY)
- Manifold / Polymarket resolved markets (public APIs)
