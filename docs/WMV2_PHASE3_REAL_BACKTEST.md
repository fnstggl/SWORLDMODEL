# WMv2 Phase 3 — REAL Resolved Historical Backtest
*Validation only. This run does not redesign Phase 3, does not start Phase 4/9, and does not weaken any prior result. Every number below is a field of the committed machine-readable artifact `experiments/results/phase3/real_backtest.json`; failures and regressions are preserved, not hidden.*
## Verdict
**Phase 3 harms forecasting.**
> Phase 3 harms forecasting (a primary paired loss CI, Brier or log-loss, is entirely > 0 with neither < 0 => confirmed regression, no improvement)

**PRELIMINARY** — scored on **23** resolved questions (< 30). Treat as directional, not definitive.

## What this measures (and what it refuses to count)
The key comparison is the **identical production path with the Phase-3 posterior consumed vs. ignored** — same questions, same frozen `EvidenceBundleV2`, same compiled plan, same qualitative claim tags, same seed, same outcome contract. The ONLY thing that varies is whether the Phase-3 particle posterior is materialized onto the plan before rollout. Accuracy is scored against the **realized, resolved outcome** (not a synthetic one). Posterior *movement* ("the probability changed") is explicitly **not** counted as improvement — only lower loss against the real outcome is.

Production path per question:

```
historical question
  -> compile_world (Phase-2 CODE)
  -> gather_evidence  [strict as-of: Google News RSS after:/before:, per-doc temporal verification, claim-level leakage audit]  => ONE frozen EvidenceBundleV2 (reused by all arms)
  -> tag_claims [qualitative, no numbers]  => frozen tags (reused by all arms)
  -> infer_posterior [particle posterior]
  -> materialize onto plan  <== the only thing that varies between the two key arms
  -> rollout terminal
  -> score vs realized outcome
```

## Arms
| arm | what it is |
|---|---|
| `prior_only` | reference-class prior mean; no evidence assimilation at the terminal |
| `phase2_no_posterior` | Phase-2 evidence path; Phase-3 posterior computed but **not** consumed |
| `phase3_posterior` | Phase-3 posterior-conditioned terminal |
| `point_estimate` | posterior collapsed to its scalar mean (anti-scalar ablation) |
| `market` | crowd / prediction-market implied probability where reliably available |

## Aggregate scores (vs realized outcome, lower is better except directional accuracy)
| arm | n | Brier ↓ | log-loss ↓ | ECE ↓ | directional acc ↑ | mean p |
|---|---|---|---|---|---|---|
| `prior_only` | 23 | 0.2514 | 0.6960 | 0.1087 | 0.3913 | 0.5000 |
| `phase2_no_posterior` | 23 | 0.2581 | 0.7103 | 0.1431 | 0.5217 | 0.4955 |
| `phase3_posterior` | 23 | 0.3118 | 0.8481 | 0.3154 | 0.4348 | 0.5002 |
| `point_estimate` | 23 | 0.3016 | 0.8193 | 0.3667 | 0.4348 | 0.5027 |
| `market` | 0 | — | — | — | — | — |

## The key paired comparison — Phase 3 vs. Phase 2 (posterior consumed vs. ignored)
- per-question: **Phase-3 better on 8**, Phase-2 better on 11, tie on 4
- paired bootstrap (n=23, arm_a=phase3, arm_b=phase2; **negative = Phase-3 lowers loss**):
  - mean Brier difference **0.0538**, 95% CI **[-0.0018, 0.1138]**
  - mean log-loss difference **0.1378**, 95% CI **[0.0014, 0.2905]**
  - P(bootstrap Brier difference < 0) = **0.029**

### Per-question Brier deltas (aggregate wins cannot hide a per-question regression)
| qid | outcome | p(phase3) | p(phase2) | Brier phase3 | Brier phase2 | Δ (ph3−ph2) | verdict |
|---|---|---|---|---|---|---|---|
| `trump_2024` | 1 | 0.4667 | 0.6000 | 0.2844 | 0.1600 | 0.1244 | phase2_better |
| `harris_2024` | 0 | 0.5000 | 0.5500 | 0.2500 | 0.3025 | -0.0525 | phase3_better |
| `biden_nominee` | 0 | 0.7925 | 0.6226 | 0.6281 | 0.3876 | 0.2404 | phase2_better |
| `uk_labour` | 1 | 0.6027 | 0.6438 | 0.1578 | 0.1269 | 0.0310 | phase2_better |
| `shutdown_oct24` | 0 | 0.8182 | 0.5152 | 0.6695 | 0.2654 | 0.4040 | phase2_better |
| `shutdown_dec24` | 0 | 0.7595 | 0.5949 | 0.5768 | 0.3539 | 0.2229 | phase2_better |
| `fed_sep24` | 1 | 0.4000 | 0.4000 | 0.3600 | 0.3600 | 0.0000 | tie |
| `fed_nov24` | 1 | 0.4200 | 0.4200 | 0.3364 | 0.3364 | 0.0000 | tie |
| `fed_dec24` | 1 | 0.4242 | 0.4242 | 0.3315 | 0.3315 | 0.0000 | tie |
| `fed_jan25` | 0 | 0.3099 | 0.4225 | 0.0960 | 0.1785 | -0.0825 | phase3_better |
| `btc_100k` | 1 | 0.5333 | 0.6000 | 0.2178 | 0.1600 | 0.0578 | phase2_better |
| `recession_24` | 0 | 0.5949 | 0.5063 | 0.3539 | 0.2563 | 0.0976 | phase2_better |
| `nvda_split` | 1 | 0.5000 | 0.5606 | 0.2500 | 0.1931 | 0.0569 | phase2_better |
| `sp500_6000` | 1 | 0.5758 | 0.5758 | 0.1799 | 0.1799 | 0.0000 | tie |
| `gpt5_2024` | 0 | 0.5915 | 0.4507 | 0.3499 | 0.2031 | 0.1467 | phase2_better |
| `gpt5_2025` | 1 | 0.2532 | 0.3924 | 0.5577 | 0.3692 | 0.1885 | phase2_better |
| `apple_intel` | 1 | 0.7000 | 0.6000 | 0.0900 | 0.1600 | -0.0700 | phase3_better |
| `gaza_ceasefire24` | 0 | 0.2125 | 0.3625 | 0.0452 | 0.1314 | -0.0863 | phase3_better |
| `gaza_ceasefire25` | 1 | 0.4625 | 0.3750 | 0.2889 | 0.3906 | -0.1017 | phase3_better |
| `assad_fall` | 1 | 0.1176 | 0.3235 | 0.7786 | 0.4577 | 0.3210 | phase2_better |
| `ru_ua_cf24` | 0 | 0.2462 | 0.4154 | 0.0606 | 0.1726 | -0.1119 | phase3_better |
| `india_t20` | 1 | 0.6761 | 0.5211 | 0.1049 | 0.2293 | -0.1244 | phase3_better |
| `real_ucl` | 1 | 0.5479 | 0.5205 | 0.2044 | 0.2299 | -0.0255 | phase3_better |

## Manual leakage audit (stratified)
Across the **23** questions with an evidence trace, the strict as-of retrieval admitted **0** documents dated after `as_of` (out of 261 dated documents in the sampled traces) and fired **0** claim-level leakage flags. A hand audit of a stratified sample (`nvda_split` 2024-05, `fed_sep24` 2024-09, `btc_100k` 2024-11, `assad_fall` 2024-11, `gaza_ceasefire25` 2025-01) confirmed every admitted document's publication date precedes both `as_of` and the resolution event. **The regression below is therefore not a leakage artifact** — if anything, clean evidence makes the negative result more credible. Full per-document dates and temporal status are in `WMV2_PHASE3_REAL_BACKTEST_TRACES.md`.

## Integrity / reproducibility
- questions attempted: **24**, completed: **23**, scored: **23**, harness errors: **1**
- within-run numeric-posterior reproducibility (same frozen inputs → identical hash): **1.000**
- posterior consumed when evidence present: **1.000**
- retrieval date (UTC): **2026-07-13T15:20:24Z**, seed **0**

## Honest reading
This is a real resolved-outcome backtest on the production path, not a synthetic recovery test. Per-question deltas are reported precisely so an aggregate number cannot mask an individual regression. Per the acceptance rule, Phase 3 is **NOT** declared empirically validated unless the paired Brier CI lies entirely below zero. Phase 3 harms forecasting.

## Reproduce
```
PYTHONPATH=. python experiments/wmv2_phase3_real_backtest.py
PYTHONPATH=. python experiments/wmv2_phase3_real_backtest_render.py
```
