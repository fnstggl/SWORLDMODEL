# WMv2 OpenRouter Historical Backtest — RESULTS (openrouter_llama31_v1, runtime_6b2288f)

**Tier:** `TIER_B_PROVIDER_PINNED_POST_RELEASE`. Model `meta-llama/llama-3.1-70b-instruct`
(released 2024-07-23, HF rev `1605565b…`), served **WandB / bf16**, pinned, no fallbacks. Every
one of 100 questions opened strictly after the release; `model_release < question_open <= cutoff <
resolution` proven per row. Not Tier A: OpenRouter cannot prove the serving weights are
byte-identical to the checkpoint.

## Execution accounting

- Entrypoint: `swm.world_model_v2.unified_runtime.simulate_world` for **every** row (sentinel
  tests prove the legacy simplified path is never reachable).
- 100 questions × 4 cutoffs = **400 expected complete runs; 400 attempted; 358 qualified**
  full-system (89.5%). 42 preserved as visible failures: 29 relevant-phase-blocked, 9 no
  first-passage readout, 4 execution_failed. Every qualified row: 11 PhaseExecutionRecords, 200
  particles, thousands of actor-action StateDeltas, first-passage terminal readout — recorded in
  the ledger's `full_run_proof`.
- Total OpenRouter spend: **$4.17** (primary + all baseline arms). Provider/quantization
  consistency failures: 0.

## The scientific result (rotating locked opened ONCE + all-scored paired tests, n=358, 72 clusters)

| Arm | Brier | paired vs WMv2 (Brier diff, 95% CI) |
|---|---|---|
| **Complete WMv2** | **0.302** | — |
| Direct same-model Llama | 0.421 | **−0.119 [−0.200, −0.042]** — WMv2 better, CI excludes 0 |
| Constant 0.5 | 0.250 | **+0.052 [+0.020, +0.087]** — WMv2 WORSE, CI excludes 0 |
| Market price @ cutoff | 0.280 | +0.022 [−0.038, +0.078] — no difference |
| Analogical (same model) | 0.280 | — |
| Observer panel (same model) | 0.387 | — |
| Call-matched ensemble | 0.349 | — |

Rotating-locked split alone (n=145): Brier 0.313, AUROC 0.52, ECE 0.25, accuracy@0.5 0.48,
event-time CRPS 0.130, base rate 0.52. Capability-normalized skill vs direct: **+0.283**.

## Honest verdict

**The complete World Model V2 simulation adds real, statistically-significant predictive value
over the SAME model forecasting directly** (Brier 0.302 vs 0.421; paired CI excludes zero;
+28% capability-normalized skill). The direct Llama-3.1-70B is severely overconfident — it slams
predictions to 0.001 or 0.5 — and the simulation's distributional, evidence-grounded readout
corrects much of that.

**But WMv2 does NOT beat a naive constant 0.5 (it is significantly worse) and does NOT beat the
market.** AUROC ≈ 0.52 is near-chance discrimination; ECE 0.25 shows it remains overconfident.
So: the simulation is a large improvement over naive use of this historical model, and NOT yet a
skilled forecaster in absolute terms on this benchmark. Both facts are reported; neither is hidden.

## Why (diagnostics, not excuses)

1. **Evidence was thin.** The archived-news (Wayback) layer returned **0 items across all 400
   rows** — every capsule was Wikipedia-revision-only, and 58/400 had zero items. The scrub
   rejected 52 contaminated items (working as intended), but the surviving evidence was too sparse
   to sharpen most forecasts. This is the single biggest fixable gap and is logged in
   `evidence_archives/.../_build_stats.json`.
2. **Effect-size and coupling packs ran on documented priors** (`insufficient_pre_cutoff_fit_data`
   recorded per row) — no fitted statement→hazard corpus exists yet.
3. **A 2023-cutoff 70B model is a weak reasoner** relative to frontier models; capability-
   normalized skill (which controls for this) is the fair lens, and it is positive.

## Reusability + scientific status

Results are append-only under `results/openrouter_llama31_v1/runtime_6b2288f/`; rerun any future
WMv2 commit with `tools/run_benchmark.py` and diff with `tools/compare_runs.py`. Per protocol:
the dev splits are `REUSABLE_DEVELOPMENT_BACKTEST` (rerunnable); the rotating-locked split is now
**CONSUMED** (opened once, `locked_access_log.json`) — a new rotating holdout must be designated
for the next scientific claim. Mechanical isolation bound the code and model; it cannot unsee
public history for the developer, which is exactly why the consumed-holdout discipline exists.

Full per-row table and case studies (5 most-accurate, 5 largest-error, 5 WMv2-beats-direct, 5
WMv2-trails-direct): `results/openrouter_llama31_v1/runtime_6b2288f/final_report.md` +
`case_details/`.
