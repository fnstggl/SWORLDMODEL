# WMv2 OpenRouter Historical Backtest — RESULTS (openrouter_llama31_v1, runtime_6b2288f)
*Generated 2026-07-18 14:31 UTC*

## 1–4. Model, provider, tier, entrypoint
1. Historical model: **meta-llama/Llama-3.1-70B-Instruct** (OpenRouter `meta-llama/llama-3.1-70b-instruct`, HF rev `1605565b47bb…`), knowledge cutoff 2023-12-31, released 2024-07-23.
2. Provider/quantization: **WandB / bf16** — pinned, no fallbacks, per-call audit ledgers.
3. Tier: **TIER_B_PROVIDER_PINNED_POST_RELEASE** — OpenRouter/WandB cannot prove the serving endpoint is byte-identical to HF revision 1605565b… or that it was never silently updated. Tier A requires a dedicated hash-verified deployment of the same frozen benchmark.
4. Entrypoint: `swm.world_model_v2.unified_runtime.simulate_world` for EVERY row (sentinel tests prove the legacy simplified path is never used).

## 5–9. Execution accounting
5. Full-run proof per row: PhaseExecutionRecords for the complete phase contract, operator delta census, particle counts, terminal source — stored in every ledger row; qualification gates in `framework/qualify.py`.
6. Questions selected: **100** (4 cutoffs each).
7. Expected complete runs: **400**.
8. Attempted: **400**; qualified full-system runs: **358**.
9. Incomplete/disqualified: **42** — causes: {'relevant_phase_blocked': 29, 'no_first_passage_readout(binary deadline questions must route event-time)': 9, 'simulation_status=execution_failed': 4}.

## 10–12. Scores
10. **Rotating locked test** (opened ONCE): n=358, Brier **0.3021**, log-loss 0.9381, AUROC 0.5341, ECE 0.2104, CRPS 0.132, 80% coverage 0.605.
    Baselines (same model, same evidence): 
    - analogical: Brier 0.2797 | paired diff 0.0224 CI95 [-0.0317, 0.0758]
    - call_matched_ensemble: Brier 0.3485 | paired diff -0.0464 CI95 [-0.1168, 0.0199]
    - constant_half: Brier 0.25 | paired diff 0.0521 CI95 [0.0203, 0.0865]
    - direct_same_model: Brier 0.4213 | paired diff -0.1192 CI95 [-0.2, -0.0416]
    - market_price_at_cutoff: Brier 0.2799 | paired diff 0.0222 CI95 [-0.038, 0.0777]
    - observer_panel: Brier 0.3867 | paired diff -0.0846 CI95 [-0.1617, -0.0139]

    Dev splits (REUSABLE_DEVELOPMENT_BACKTEST): n=213, Brier 0.2944, AUROC 0.5221; skill vs direct 0.2471.

## 13. Results by causal scale
- broad_aggregate: n=52, Brier 0.3348, AUROC 0.5629
- institutional_process: n=16, Brier 0.2228, AUROC 0.746
- mixed_scale: n=8, Brier 0.1035, AUROC None
- multi_actor_strategic: n=137, Brier 0.2943, AUROC 0.4829
- single_decision_maker: n=68, Brier 0.3034, AUROC 0.4792
- small_group_decision: n=77, Brier 0.33, AUROC 0.5864

## 14. Leakage census
- Evidence: archived-bytes only (Wayback capture proofs + Wikipedia revids); contamination scrub counts in `evidence_archives/_build_stats.json`; capsules sealed before simulation; deterministic query generation (no frontier model anywhere in the case-dependent path).

## 15. Verdict

**World Model V2 IMPROVED on the direct same-model baseline: paired Brier diff -0.1192 (95% CI [-0.2000, -0.0416], excludes 0); capability-normalized skill +0.283.**

## Locked-row table

| question | cutoff | outcome | WMv2 | direct | market | dominant mode | median t | qualified |
|---|---|---|---|---|---|---|---|---|
| No change in Fed interest rates after June 2025 meeting? | 2025-02-23 | 1 | 0.56 | 0.00 | 0.61 | resolution | 2025-06-07 | Y |
| Will Oscar Piastri be the 2025 Drivers Champion? | 2025-04-10 | 0 | 0.35 | 0.05 | 0.23 | resolution | 2025-12-07 | Y |
| Will Lando Norris be the 2025 Drivers Champion? | 2025-04-08 | 1 | 0.91 | 0.05 | 0.46 | entailed_fact:other | 2025-11-30 | Y |
| Israel x Hamas ceasefire before July? | 2025-04-04 | 0 | 0.61 | 0.00 | 0.55 | resolution | 2025-06-06 | Y |
| No change in Fed interest rates after July 2025 meeting? | 2025-04-02 | 1 | 0.99 | 0.50 | 0.41 | entailed_fact:scheduled_meeting | 2025-07-29 | Y |
| Fed decreases interest rates by 25 bps after July 2025 meeti | 2025-04-08 | 0 | 0.40 | 0.50 | 0.55 | resolution | 2025-07-30 | Y |
| Liberals win majority in Canadian election? | 2025-03-26 | 0 | 0.62 | 0.00 | 0.38 | resolution | 2025-04-22 | Y |
| Will the next Government of Canada be a Liberal majority? | 2025-04-01 | 0 | 0.58 | 0.00 | 0.51 | resolution | 2025-04-23 | Y |
| Will the next Government of Canada be a Liberal minority? | 2025-03-30 | 1 | 0.83 | 0.00 | 0.15 | entailed_fact:term_expiry | 2025-03-30 | Y |
| US military action against Iran before July? | 2025-04-13 | 1 | 0.40 | 0.00 | 0.20 | institutional:US_military | 2025-06-30 | Y |
| Israel military action against Iran before July? | 2025-04-11 | 1 | 0.38 | 0.50 | 0.28 | resolution | 2025-06-30 | Y |
| Trump x Ukraine mineral deal signed before May? | 2025-04-05 | 1 | 0.40 | 0.00 | 0.30 | resolution | 2025-04-30 | Y |
| Russia x Ukraine ceasefire before October? | 2025-05-10 | 0 | 0.32 | 0.20 | 0.39 | institutional:United Nations | 2025-09-30 | Y |
| Trump declassifies UFO files in 2025? | 2025-05-23 | 1 | 0.45 | 0.00 | 0.26 | resolution | 2025-12-31 | Y |
| TikTok sale announced in 2025? | 2025-05-24 | 1 | 0.62 | 0.00 | 0.29 | resolution | 2025-12-10 | Y |
| Will Andrew Cuomo win the 2025 NYC mayoral election? | 2025-05-22 | 0 | 0.90 | 0.00 | 0.86 | entailed_fact:election | 2025-11-04 | Y |
| Will Zohran Mamdani win the 2025 NYC mayoral election? | 2025-05-17 | 1 | 0.40 | 0.00 | 0.07 | resolution | 2025-11-04 | Y |
| Will Jannik Sinner win the 2025 US Open? | 2025-05-25 | 0 | 0.98 | 0.00 | 0.34 | entailed_fact:other | 2025-09-07 | Y |
| No change in Fed interest rates after September 2025 meeting | 2025-05-27 | 0 | 0.57 | 0.50 | 0.69 | resolution | 2025-09-06 | Y |
| Fed decreases interest rates by 25 bps after September 2025  | 2025-05-27 | 1 | 0.41 | 0.00 | 0.26 | resolution | 2025-09-17 | Y |
| Will Lee Jae-myung win 50-55% of the vote in the South Korea | 2025-05-12 | 0 | — | 0.00 | 0.45 | — | — | N: no_first_passage_readout(binary deadline |
| Will Lee Jae-myung win 45-50% of the vote in the South Korea | 2025-05-12 | 1 | 0.60 | 0.00 | 0.18 | resolution | 2025-05-30 | Y |
| Will Tesla launch a driverless Robotaxi service before July? | 2025-06-03 | 0 | 0.82 | 0.00 | 0.18 | entailed_fact:other | 2025-06-03 | Y |
| Israel x Hamas ceasefire before August? | 2025-06-17 | 0 | 0.99 | 0.60 | 0.39 | entailed_fact:scheduled_meeting | 2025-06-20 | Y |
| Fed decreases interest rates by 25 bps after October 2025 me | 2025-07-05 | 1 | 0.98 | 0.00 | 0.49 | entailed_fact:scheduled_meeting | 2025-10-28 | Y |
| No change in Fed interest rates after October 2025 meeting? | 2025-07-09 | 0 | 0.99 | 0.00 | 0.39 | entailed_fact:scheduled_meeting | 2025-10-28 | Y |
| Israel x Hamas ceasefire by July 15? | 2025-06-29 | 0 | 0.38 | 0.00 | 0.37 | institutional:Israeli government | — | Y |
| Will Artificial Intelligence be TIME's Person of the Year fo | 2025-07-25 | 0 | 0.90 | 0.00 | 0.26 | entailed_fact:other | 2025-12-01 | Y |
| Will lighter perform an airdrop by December 31? | 2025-07-15 | 1 | 0.34 | 0.00 | 0.62 | resolution | — | Y |
| Will Curtis Sliwa drop out? | 2025-07-16 | 0 | — | 0.00 | 0.24 | — | — | N: simulation_status=execution_failed |
| Will Mamdani get over 50% of the vote in the general mayoral | 2025-07-17 | 1 | 0.43 | 0.00 | 0.38 | resolution | 2025-11-03 | Y |
| Will Ken Paxton win the 2026 Texas Republican Primary? | 2025-08-26 | 1 | 0.12 | 0.60 | 0.62 | institutional:Texas Republican Party | — | Y |
| Israel x Hamas ceasefire by August 31? | 2025-07-22 | 0 | 0.26 | 0.60 | 0.62 | institutional:Israeli government | — | Y |
| Tesla launches unsupervised full self driving (FSD) by June  | 2025-08-19 | 1 | 0.53 | 0.00 | 0.35 | resolution | 2026-06-01 | Y |
| Houthi strike on Israel by August 31? | 2025-07-29 | 1 | 0.41 | 0.05 | 0.18 | resolution | — | Y |
| US government shutdown by October 1? | 2025-09-07 | 1 | 1.00 | 0.30 | 0.32 | entailed_fact:expiration | 2025-09-30 | N: relevant_phase_blocked:phase10_instituti |
| Will the Government shutdown end November 16 or later? | 2025-10-12 | 0 | 0.98 | 0.00 | 0.16 | institutional:US_Congress | 2025-11-14 | Y |
| Will the Government shutdown end November 4-7? | 2025-10-12 | 0 | 0.49 | 0.00 | 0.07 | resolution | 2025-11-14 | Y |
| Will the Government shutdown end November 12-15? | 2025-10-12 | 1 | 1.00 | 0.00 | 0.06 | institutional:US_Congress | 2025-11-14 | Y |
| US government shutdown Saturday? | 2025-11-25 | 1 | 0.94 | 0.00 | 0.28 | entailed_fact:expiration | 2026-01-31 | Y |
| No change in Fed interest rates after June 2025 meeting? | 2025-03-15 | 1 | 0.57 | 0.00 | 0.32 | resolution | 2025-05-31 | Y |
| Will Oscar Piastri be the 2025 Drivers Champion? | 2025-06-06 | 0 | 0.22 | 0.05 | 0.58 | institutional:FIA | 2025-12-07 | Y |
| Will Lando Norris be the 2025 Drivers Champion? | 2025-06-01 | 1 | 0.94 | 0.05 | 0.34 | entailed_fact:other | 2025-11-24 | Y |
| Israel x Hamas ceasefire before July? | 2025-04-24 | 0 | 0.58 | 0.00 | 0.56 | resolution | 2025-06-11 | Y |
| No change in Fed interest rates after July 2025 meeting? | 2025-04-20 | 1 | 0.99 | 0.50 | 0.34 | entailed_fact:scheduled_meeting | 2025-07-29 | Y |
| Fed decreases interest rates by 25 bps after July 2025 meeti | 2025-05-05 | 0 | 1.00 | 0.50 | 0.41 | entailed_fact:scheduled_meeting | 2025-07-29 | N: relevant_phase_blocked:phase10_instituti |
| Liberals win majority in Canadian election? | 2025-04-03 | 0 | 0.58 | 0.00 | 0.61 | resolution | 2025-04-23 | Y |
| Will the next Government of Canada be a Liberal majority? | 2025-04-08 | 0 | 0.59 | 0.00 | 0.61 | resolution | 2025-04-24 | Y |
| Will the next Government of Canada be a Liberal minority? | 2025-04-03 | 1 | 0.84 | 0.00 | 0.15 | entailed_fact:other | 2025-04-03 | Y |
| US military action against Iran before July? | 2025-04-29 | 1 | 0.24 | 0.00 | 0.24 | institutional:US_military | 2025-06-30 | Y |
| Israel military action against Iran before July? | 2025-04-26 | 1 | 0.40 | 0.50 | 0.32 | resolution | 2025-06-30 | Y |
| Trump x Ukraine mineral deal signed before May? | 2025-04-11 | 1 | 0.38 | 0.00 | 0.29 | resolution | 2025-04-30 | Y |
| Russia x Ukraine ceasefire before October? | 2025-06-12 | 0 | 0.30 | 0.20 | 0.14 | resolution | — | Y |
| Trump declassifies UFO files in 2025? | 2025-07-09 | 1 | 0.21 | 0.00 | 0.17 | institutional:US_Government | — | Y |
| TikTok sale announced in 2025? | 2025-07-12 | 1 | 0.44 | 0.00 | 0.33 | institutional:ByteDance | 2025-12-31 | Y |
| Will Andrew Cuomo win the 2025 NYC mayoral election? | 2025-06-30 | 0 | 0.35 | 0.00 | 0.10 | resolution | — | Y |
| Will Zohran Mamdani win the 2025 NYC mayoral election? | 2025-06-19 | 1 | 0.43 | 0.00 | 0.13 | resolution | 2025-11-04 | Y |
| Will Jannik Sinner win the 2025 US Open? | 2025-06-19 | 0 | 0.92 | 0.00 | 0.42 | entailed_fact:other | 2025-09-07 | Y |
| No change in Fed interest rates after September 2025 meeting | 2025-06-23 | 0 | 0.98 | 0.50 | 0.53 | entailed_fact:scheduled_meeting | 2025-09-17 | Y |
| Fed decreases interest rates by 25 bps after September 2025  | 2025-06-21 | 1 | 0.90 | 0.00 | 0.41 | entailed_fact:scheduled_meeting | 2025-09-16 | Y |
| Will Lee Jae-myung win 50-55% of the vote in the South Korea | 2025-05-17 | 0 | 0.59 | 0.00 | 0.47 | resolution | 2025-05-30 | Y |
| Will Lee Jae-myung win 45-50% of the vote in the South Korea | 2025-05-17 | 1 | — | 0.00 | 0.16 | — | — | N: no_first_passage_readout(binary deadline |
| Will Tesla launch a driverless Robotaxi service before July? | 2025-06-10 | 0 | 0.79 | 0.00 | 0.14 | entailed_fact:other | 2025-06-10 | Y |
| Israel x Hamas ceasefire before August? | 2025-06-27 | 0 | 0.74 | 0.30 | 0.54 | institutional:Israeli government | 2025-07-30 | Y |
| Fed decreases interest rates by 25 bps after October 2025 me | 2025-07-26 | 1 | 0.56 | 0.00 | 0.46 | resolution | 2025-10-20 | Y |
| No change in Fed interest rates after October 2025 meeting? | 2025-08-05 | 0 | 0.54 | 0.00 | 0.35 | resolution | 2025-10-21 | Y |
| Israel x Hamas ceasefire by July 15? | 2025-07-03 | 0 | 0.30 | 0.00 | 0.36 | institutional:Israeli government | — | Y |
| Will Artificial Intelligence be TIME's Person of the Year fo | 2025-08-31 | 0 | 0.97 | 0.00 | 0.30 | entailed_fact:other | 2025-11-25 | Y |
| Will lighter perform an airdrop by December 31? | 2025-08-09 | 1 | 0.60 | 0.00 | 0.63 | resolution | 2025-12-04 | Y |
| Will Curtis Sliwa drop out? | 2025-08-11 | 0 | — | 0.00 | 0.21 | — | — | N: simulation_status=execution_failed |
| Will Mamdani get over 50% of the vote in the general mayoral | 2025-08-07 | 1 | 1.00 | 0.00 | 0.47 | entailed_fact:election | 2025-11-03 | N: relevant_phase_blocked:phase10_instituti |
| Will Ken Paxton win the 2026 Texas Republican Primary? | 2025-10-28 | 1 | 0.55 | 0.70 | 0.37 | institutional:Texas Republican Party | 2026-05-25 | Y |
| Israel x Hamas ceasefire by August 31? | 2025-07-31 | 0 | 0.29 | 0.00 | 0.33 | institutional:Israeli government | — | Y |
| Tesla launches unsupervised full self driving (FSD) by June  | 2025-09-26 | 1 | 0.70 | 0.00 | 0.44 | institutional:Tesla Board of Directors | 2026-06-29 | Y |
| Houthi strike on Israel by August 31? | 2025-08-04 | 1 | 0.41 | 0.05 | 0.18 | resolution | — | Y |
| US government shutdown by October 1? | 2025-09-12 | 1 | 1.00 | 0.20 | 0.37 | entailed_fact:expiration | 2025-09-30 | N: relevant_phase_blocked:phase10_instituti |
| Will the Government shutdown end November 16 or later? | 2025-10-20 | 0 | — | 0.00 | 0.45 | — | — | N: simulation_status=execution_failed |
| Will the Government shutdown end November 4-7? | 2025-10-20 | 0 | 0.62 | 0.00 | 0.14 | resolution | 2025-11-03 | Y |
| Will the Government shutdown end November 12-15? | 2025-10-19 | 1 | 1.00 | 0.00 | 0.09 | institutional:US_Congress | 2025-11-14 | Y |
| US government shutdown Saturday? | 2025-12-10 | 1 | 0.99 | 0.00 | 0.33 | entailed_fact:expiration | 2026-01-31 | Y |
| No change in Fed interest rates after June 2025 meeting? | 2025-04-05 | 1 | 0.98 | 0.00 | 0.33 | entailed_fact:scheduled_meeting | 2025-06-17 | Y |
| Will Oscar Piastri be the 2025 Drivers Champion? | 2025-08-01 | 0 | 0.34 | 0.20 | 0.63 | institutional:FIA | 2025-12-07 | Y |
| Will Lando Norris be the 2025 Drivers Champion? | 2025-07-24 | 1 | 0.42 | 0.05 | 0.38 | resolution | 2025-12-07 | Y |
| Israel x Hamas ceasefire before July? | 2025-05-15 | 0 | 0.64 | 0.50 | 0.55 | resolution | 2025-06-17 | Y |
| No change in Fed interest rates after July 2025 meeting? | 2025-05-09 | 1 | 0.99 | 0.50 | 0.51 | entailed_fact:scheduled_meeting | 2025-07-29 | Y |
| Fed decreases interest rates by 25 bps after July 2025 meeti | 2025-05-31 | 0 | 0.99 | 0.00 | 0.18 | entailed_fact:scheduled_meeting | 2025-07-29 | Y |
| Liberals win majority in Canadian election? | 2025-04-11 | 0 | 0.60 | 0.00 | 0.64 | resolution | 2025-04-26 | Y |
| Will the next Government of Canada be a Liberal majority? | 2025-04-14 | 0 | 0.52 | 0.00 | 0.58 | resolution | 2025-04-26 | Y |
| Will the next Government of Canada be a Liberal minority? | 2025-04-07 | 1 | 0.54 | 0.00 | 0.14 | resolution | 2025-04-26 | Y |
| US military action against Iran before July? | 2025-05-16 | 1 | 0.99 | 0.00 | 0.10 | entailed_fact:scheduled_vote | 2025-06-15 | Y |
| Israel military action against Iran before July? | 2025-05-11 | 1 | 0.24 | 0.50 | 0.21 | resolution | — | Y |
| Trump x Ukraine mineral deal signed before May? | 2025-04-16 | 1 | 0.45 | 0.00 | 0.15 | resolution | 2025-04-30 | Y |
| Russia x Ukraine ceasefire before October? | 2025-07-16 | 0 | 0.29 | 0.20 | 0.10 | resolution | — | Y |
| Trump declassifies UFO files in 2025? | 2025-08-25 | 1 | 0.46 | 0.00 | 0.17 | resolution | 2025-12-31 | Y |
| TikTok sale announced in 2025? | 2025-08-30 | 1 | 0.40 | 0.00 | 0.18 | institutional:ByteDance | 2025-12-31 | Y |
| Will Andrew Cuomo win the 2025 NYC mayoral election? | 2025-08-08 | 0 | 0.40 | 0.00 | 0.09 | resolution | 2025-11-04 | Y |
| Will Zohran Mamdani win the 2025 NYC mayoral election? | 2025-07-22 | 1 | 0.99 | 0.50 | 0.71 | entailed_fact:election | 2025-11-04 | Y |
| Will Jannik Sinner win the 2025 US Open? | 2025-07-13 | 0 | 0.92 | 0.00 | 0.45 | entailed_fact:other | 2025-09-07 | Y |
| No change in Fed interest rates after September 2025 meeting | 2025-07-19 | 0 | 0.55 | 0.50 | 0.48 | resolution | 2025-09-11 | Y |
| Fed decreases interest rates by 25 bps after September 2025  | 2025-07-17 | 1 | 0.66 | 0.50 | 0.40 | resolution | 2025-09-05 | Y |
| Will Lee Jae-myung win 50-55% of the vote in the South Korea | 2025-05-23 | 0 | — | 0.00 | 0.57 | — | — | N: no_first_passage_readout(binary deadline |
| Will Lee Jae-myung win 45-50% of the vote in the South Korea | 2025-05-23 | 1 | — | 0.00 | 0.24 | — | — | N: no_first_passage_readout(binary deadline |
| Will Tesla launch a driverless Robotaxi service before July? | 2025-06-16 | 0 | 0.83 | 0.00 | 0.27 | entailed_fact:other | 2025-06-30 | Y |
| Israel x Hamas ceasefire before August? | 2025-07-08 | 0 | 0.85 | 0.60 | 0.66 | institutional:Israeli government | 2025-07-30 | Y |
| Fed decreases interest rates by 25 bps after October 2025 me | 2025-08-16 | 1 | 0.61 | 0.00 | 0.56 | resolution | 2025-10-15 | Y |
| No change in Fed interest rates after October 2025 meeting? | 2025-08-31 | 0 | 0.61 | 0.50 | 0.43 | resolution | 2025-10-18 | Y |
| Israel x Hamas ceasefire by July 15? | 2025-07-06 | 0 | 0.28 | 0.00 | 0.47 | institutional:Israeli government | — | Y |
| Will Artificial Intelligence be TIME's Person of the Year fo | 2025-10-08 | 0 | 0.90 | 0.00 | 0.32 | entailed_fact:other | 2025-12-01 | Y |
| Will lighter perform an airdrop by December 31? | 2025-09-02 | 1 | 0.57 | 0.00 | 0.89 | resolution | 2025-12-20 | Y |
| Will Curtis Sliwa drop out? | 2025-09-05 | 0 | 0.38 | 0.00 | 0.24 | resolution | — | Y |
| Will Mamdani get over 50% of the vote in the general mayoral | 2025-08-27 | 1 | 0.97 | 0.00 | 0.56 | entailed_fact:election | 2025-11-03 | Y |
| Will Ken Paxton win the 2026 Texas Republican Primary? | 2025-12-29 | 1 | 0.56 | 0.60 | 0.62 | institutional:Texas Republican Party | 2026-05-25 | Y |
| Israel x Hamas ceasefire by August 31? | 2025-08-09 | 0 | 0.33 | 0.30 | 0.10 | institutional:Israeli government | — | Y |
| Tesla launches unsupervised full self driving (FSD) by June  | 2025-11-04 | 1 | 0.96 | 0.00 | 0.53 | entailed_fact:deadline | 2025-12-31 | Y |
| Houthi strike on Israel by August 31? | 2025-08-11 | 1 | 0.45 | 0.05 | 0.15 | resolution | — | Y |
| US government shutdown by October 1? | 2025-09-18 | 1 | 0.99 | 0.20 | 0.36 | institutional:US_Congress | 2025-09-30 | Y |
| Will the Government shutdown end November 16 or later? | 2025-10-28 | 0 | 1.00 | 0.00 | 0.46 | institutional:US_Congress | 2025-11-14 | Y |
| Will the Government shutdown end November 4-7? | 2025-10-28 | 0 | 0.46 | 0.00 | 0.19 | institutional:US Congress | 2025-11-14 | Y |
| Will the Government shutdown end November 12-15? | 2025-10-26 | 1 | 0.99 | 0.00 | 0.14 | institutional:US_Congress | 2025-11-14 | Y |
| US government shutdown Saturday? | 2025-12-25 | 1 | 0.94 | 0.00 | 0.27 | entailed_fact:expiration | 2026-01-31 | Y |
| No change in Fed interest rates after June 2025 meeting? | 2025-04-25 | 1 | 0.56 | 0.00 | 0.45 | resolution | 2025-06-08 | Y |
| Will Oscar Piastri be the 2025 Drivers Champion? | 2025-09-27 | 0 | 0.42 | 0.05 | 0.67 | resolution | 2025-12-07 | Y |
| Will Lando Norris be the 2025 Drivers Champion? | 2025-09-16 | 1 | 0.89 | 0.05 | 0.20 | entailed_fact:other | 2025-11-24 | Y |
| Israel x Hamas ceasefire before July? | 2025-06-04 | 0 | 0.58 | 0.00 | 0.34 | resolution | 2025-06-22 | Y |
| No change in Fed interest rates after July 2025 meeting? | 2025-05-27 | 1 | 0.99 | 0.50 | 0.80 | entailed_fact:scheduled_meeting | 2025-07-29 | Y |
| Fed decreases interest rates by 25 bps after July 2025 meeti | 2025-06-27 | 0 | 0.91 | 0.00 | 0.21 | entailed_fact:scheduled_vote | 2025-07-30 | Y |
| Liberals win majority in Canadian election? | 2025-04-18 | 0 | 0.62 | 0.00 | 0.62 | resolution | 2025-04-26 | Y |
| Will the next Government of Canada be a Liberal majority? | 2025-04-20 | 0 | 0.59 | 0.00 | 0.60 | resolution | 2025-04-27 | Y |
| Will the next Government of Canada be a Liberal minority? | 2025-04-11 | 1 | 0.59 | 0.00 | 0.13 | resolution | 2025-04-24 | Y |
| US military action against Iran before July? | 2025-06-01 | 1 | 0.41 | 0.00 | 0.11 | institutional:US_military | 2025-06-30 | Y |
| Israel military action against Iran before July? | 2025-05-25 | 1 | 0.32 | 0.00 | 0.28 | resolution | — | Y |
| Trump x Ukraine mineral deal signed before May? | 2025-04-21 | 1 | 0.38 | 0.00 | 0.55 | resolution | 2025-04-30 | Y |
| Russia x Ukraine ceasefire before October? | 2025-08-19 | 0 | 0.28 | 0.30 | 0.21 | resolution | — | Y |
| Trump declassifies UFO files in 2025? | 2025-10-11 | 1 | 0.42 | 0.00 | 0.14 | resolution | 2025-12-31 | Y |
| TikTok sale announced in 2025? | 2025-10-18 | 1 | 0.41 | 0.00 | 0.29 | institutional:ByteDance | 2025-12-31 | Y |
| Will Andrew Cuomo win the 2025 NYC mayoral election? | 2025-09-16 | 0 | 0.90 | 0.00 | 0.17 | entailed_fact:election | 2025-11-04 | Y |
| Will Zohran Mamdani win the 2025 NYC mayoral election? | 2025-08-24 | 1 | 0.47 | 0.50 | 0.84 | resolution | 2025-11-04 | Y |
| Will Jannik Sinner win the 2025 US Open? | 2025-08-07 | 0 | 0.98 | 0.00 | 0.47 | entailed_fact:other | 2025-09-07 | Y |
| No change in Fed interest rates after September 2025 meeting | 2025-08-15 | 0 | 0.93 | 0.50 | 0.23 | entailed_fact:scheduled_meeting | 2025-09-17 | Y |
| Fed decreases interest rates by 25 bps after September 2025  | 2025-08-12 | 1 | 0.93 | 0.00 | 0.73 | entailed_fact:scheduled_meeting | 2025-09-17 | Y |
| Will Lee Jae-myung win 50-55% of the vote in the South Korea | 2025-05-28 | 0 | 0.63 | 0.00 | 0.50 | resolution | 2025-06-02 | Y |
| Will Lee Jae-myung win 45-50% of the vote in the South Korea | 2025-05-28 | 1 | — | 0.00 | 0.40 | — | — | N: no_first_passage_readout(binary deadline |
| Will Tesla launch a driverless Robotaxi service before July? | 2025-06-22 | 0 | 0.62 | 0.00 | 0.14 | resolution | 2025-06-29 | Y |
| Israel x Hamas ceasefire before August? | 2025-07-18 | 0 | 0.75 | 0.60 | 0.46 | institutional:Israeli government | 2025-07-30 | Y |
| Fed decreases interest rates by 25 bps after October 2025 me | 2025-09-06 | 1 | 0.93 | 0.00 | 0.58 | entailed_fact:other | 2025-10-28 | Y |
| No change in Fed interest rates after October 2025 meeting? | 2025-09-26 | 0 | 0.59 | 0.00 | 0.17 | resolution | 2025-10-23 | Y |
| Israel x Hamas ceasefire by July 15? | 2025-07-10 | 0 | 0.24 | 0.00 | 0.28 | institutional:Israeli government | — | Y |
| Will Artificial Intelligence be TIME's Person of the Year fo | 2025-11-14 | 0 | 0.41 | 0.00 | 0.34 | institutional:TIME | 2025-12-31 | Y |
| Will lighter perform an airdrop by December 31? | 2025-09-27 | 1 | 0.56 | 0.00 | 0.84 | resolution | 2025-12-12 | Y |
| Will Curtis Sliwa drop out? | 2025-10-01 | 0 | — | 0.00 | 0.15 | — | — | N: simulation_status=execution_failed |
| Will Mamdani get over 50% of the vote in the general mayoral | 2025-09-17 | 1 | 1.00 | 0.00 | 0.64 | entailed_fact:election | 2025-11-03 | N: relevant_phase_blocked:phase10_instituti |
| Will Ken Paxton win the 2026 Texas Republican Primary? | 2026-03-02 | 1 | 0.76 | 0.60 | 0.79 | institutional:Texas Republican Primary | 2026-05-25 | Y |
| Israel x Hamas ceasefire by August 31? | 2025-08-19 | 0 | 0.65 | 0.30 | 0.23 | institutional:Israeli government | 2025-08-30 | Y |
| Tesla launches unsupervised full self driving (FSD) by June  | 2025-12-12 | 1 | 0.55 | 0.00 | 0.58 | resolution | 2026-06-11 | Y |
| Houthi strike on Israel by August 31? | 2025-08-17 | 1 | 0.34 | 0.05 | 0.14 | resolution | — | Y |
| US government shutdown by October 1? | 2025-09-24 | 1 | 1.00 | 0.30 | 0.58 | entailed_fact:expiration | 2025-09-30 | N: relevant_phase_blocked:phase10_instituti |
| Will the Government shutdown end November 16 or later? | 2025-11-05 | 0 | 0.34 | 0.00 | 0.30 | institutional:US Government | — | Y |
| Will the Government shutdown end November 4-7? | 2025-11-05 | 0 | 0.39 | 0.00 | 0.17 | institutional:US_Congress | 2025-11-14 | Y |
| Will the Government shutdown end November 12-15? | 2025-11-03 | 1 | 0.99 | 0.00 | 0.18 | institutional:US_Congress | 2025-11-14 | Y |
| US government shutdown Saturday? | 2026-01-09 | 1 | 0.29 | 0.00 | 0.23 | institutional:US Government | — | Y |

## Cost

Total OpenRouter spend this run: **$4.17** across 3049 primary calls + baselines.
