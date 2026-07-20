# EXP-108 — the §8-9 deadline-aware prior, LEAN, decides the strategy

## The question: is the rich PR#127 pipeline worth it? (Data says no.)

Propagation check on the Knesset question through the FULL PR#127 rich pipeline (numeric actors):
**1707 seconds (28 min), 189 LLM calls, result = p=None / status "under_modeled"** ("no validated causal
mechanism resolves the outcome" — the §NAP unresolved path). The grounded §8-9 prior (Knesset base rate
0.05) did NOT propagate: §NAP returns no number rather than surfacing the grounded prior.

## The lever, run LEAN (forecast = build_outcome_rate_prior mean, ~2 calls/q), same 25 questions

| forecaster | Brier | acc | AUC | mean-p YES / NO | cost/q |
|---|---|---|---|---|---|
| **lean §8-9 deadline prior** | **0.253** | **0.64** | **0.657** | 0.408 / 0.286 | ~2 calls / sec |
| rich PR#127 pipeline (EXP-107) | 0.310 | 0.48 | 0.413 | 0.444 / 0.513 | 189 calls / 28 min |
| thin mechanism kernel | 0.352 | 0.58 | 0.521 | — | 1 call |
| constant base-rate (0.40) | 0.240 | — | — | — | 0 |
| FutureSearch SOTA | 0.176 | — | — | — | — |

## Findings

1. **The §8-9 deadline-aware outside-view prior is the lever, and it works.** Lean, it fixes the
   anti-discrimination EXP-107 exposed (AUC 0.413 -> 0.657; mean-p YES now ABOVE NO), beats the rich
   pipeline (0.253 vs 0.310) and the thin kernel (0.352), at ~1% of the cost. Occurrence questions land
   0.12-0.26; recurrences/essentially-decided stay 0.84-0.86. No hardcoded pessimism.
2. **The rich PR#127 pipeline is NOT worth it for general forecasting.** 100x more expensive, less
   accurate, and its §NAP unresolved path buries the grounded prior as "under_modeled" (no number). The
   value was never the actor/structural rollout.
3. **Still short of SOTA** (0.253 vs 0.176) and a hair above a constant on raw Brier (0.253 vs 0.240) — but
   it is now a genuine discriminating forecaster, not an anti-discriminative one. Remaining error is mostly
   stage-classification misses (e.g. a "recurring_due" tanker-transit that resolved NO).

## Recommendation

Adopt the LEAN grounded-outside-view path (deadline-aware prior + evidence) as the forecasting architecture
for general questions. The rich actor/structural rollout stays for what it is actually for (auditable
entity/counterfactual simulation), not calibrated point forecasts. Next accuracy lever within the lean path:
sharpen stage classification + fold in the escalated evidence (imminence/blockage) that the lean prior does
not yet read.
