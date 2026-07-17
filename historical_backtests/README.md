# Historical Backtests — the permanent WMv2 backtesting laboratory

Reusable infrastructure for running the **complete production World Model V2 simulation**
(`swm.world_model_v2.unified_runtime.simulate_world` — never a reduced path) against historically
resolved questions, with period-bounded open-weight models served through OpenRouter.

## Scientific status: TIER_B_PROVIDER_PINNED_POST_RELEASE

Every question **opens strictly after the historical model's public release**, so the answer
cannot be in the weights (`model_release < question_open <= forecast_cutoff < resolution`).
OpenRouter cannot prove the serving endpoint is byte-identical to the open-weight checkpoint or
was never silently updated — that is why this is **Tier B, not Tier A**. The model interface is
designed so the identical frozen benchmark can be rerun against a hash-verified dedicated
deployment for Tier A.

## Two benchmark classes — never confuse them

- **Reusable regression benchmark** (`calibration` + `validation` splits): rerun on every WMv2
  version, compare architectures, diagnose failures. Results are labeled
  `REUSABLE_DEVELOPMENT_BACKTEST`. Rerunnable forever through the outcome-isolated harness.
- **Rotating sealed holdout** (`rotating_locked` split): outcomes opened **once** per runtime
  (append to `locked_access_log.json`), results labeled `ROTATING_SEALED_HOLDOUT`; after opening,
  the split is `CONSUMED` — it moves into the regression library and a NEW holdout must be
  designated for the next scientific claim.

**Stated limitation (do not weaken it):** mechanical outcome isolation prevents the forecasting
code and model from reading answers. It cannot make developers forget public historical outcomes
they have inspected. Reusable cases are valid for regression testing and engineering comparison;
they are **not** permanently pristine scientific holdouts. Strong improvement claims require a
newly selected rotating holdout or the live forward vault.

## Outcome isolation (mechanical)

- `framework/resolution_store.py` **raises at import** unless `REPLAY_SCORER=1`.
- The forecaster (`framework/runner.py`) never imports it (sentinel-tested), reads only the sealed
  question vault (no outcomes, no post-cutoff prices, no answer-encoding filenames), and receives
  only frozen pre-cutoff evidence capsules.
- Scoring runs in a **separate process**; every outcome access is appended to
  `resolution_vault/outcome_access_ledger.jsonl`.

## Layout

```
framework/    vault_build, evidence_build, packs (walk-forward), runner, qualify,
              baselines, scorer, metrics, freeze, resolution_store (scorer-only)
models/       historical_model_registry.json + loader + temporal gate
benchmark_versions/<id>/   sealed question vault, composition report, freeze manifest
evidence_archives/<id>/    frozen capsules (archived bytes only), sealed per capsule
fitted_packs/              survival corpus cache + monthly walk-forward snapshots
resolution_vault/          sealed outcomes + outcome-access ledger (REPLAY_SCORER only)
results/<id>/runtime_<commit>/   append-only forecast ledger (+ per-call provider audit)
tools/        run_benchmark, build_evidence, compare_runs, discover_historical_models
tests/        enforcement battery (mocked; no network, no outcomes)
```

## Commands

```bash
# rerun the reusable regression benchmark on the current WMv2 commit
python -m historical_backtests.tools.run_benchmark \
    --benchmark openrouter_llama31_v1 --runtime current --split reusable_regression

# forecast the rotating holdout (outcome opening is scorer-side, one-time)
python -m historical_backtests.tools.run_benchmark \
    --benchmark openrouter_llama31_v1 --runtime current --split rotating_holdout

# score (separate process; REPLAY_SCORER required)
REPLAY_SCORER=1 python -m historical_backtests.framework.scorer \
    --benchmark openrouter_llama31_v1 --run runtime_<commit>            # dev splits
REPLAY_SCORER=1 python -m historical_backtests.framework.scorer \
    --benchmark openrouter_llama31_v1 --run runtime_<commit> --open-locked  # ONE TIME

# compare two runtimes
python -m historical_backtests.tools.compare_runs openrouter_llama31_v1 RUN_A RUN_B
```

Results are **append-only and immutable**: one directory per (benchmark version × runtime
commit); old runs are never mutated to reflect new code.

## Creating a new rotating sealed holdout (without exposing outcomes)

1. Run `vault_build.py` for a NEW benchmark version id with a later question window
   (`REPLAY_SCORER=1`; the builder writes outcomes only into the sealed resolution vault).
2. Freeze capsules + packs + `freeze.py` manifest **before** any forecasting.
3. Forecast with the runner; open outcomes once with `--open-locked`; mark CONSUMED in the
   benchmark's README stanza; designate the next version.

## Adding a historical model (two-stage; never at execution time)

1. `python -m historical_backtests.tools.discover_historical_models` — writes audit candidates
   from OpenRouter + HF metadata. **Discovery evidence is not temporal proof.**
2. Verify the release timestamp against the developer's primary sources; record every training
   stage cutoff (base / continued-pretraining / instruction / preference) or mark
   `training_stage_temporal_boundary_unverified`; add the registry entry with an exact provider +
   quantization + HF revision; set `approval_status`. Unapproved entries cannot execute.
   Each approved model gets its OWN benchmark version (questions opening after ITS release) and
   its own result namespace — never pool rows across models into one primary score.
