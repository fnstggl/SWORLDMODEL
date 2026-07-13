# WMV2 Phase 6 — Validation & Failures

All numbers regenerate from `experiments/wmv2_phase6_fits.py` (real-data fits), the committed registry, and
`experiments/wmv2_phase6_report.py`. Failures are **append-only and never deleted**.

## 1. Real-data held-out validation (this run)

| Family / pack | Dataset (n) | Split | Metric | Result | Baseline | Verdict |
|---|---|---|---|---|---|---|
| content_response_click | Upworthy Archive (~4.8k tests) | random 60/40 (in-dist) | pairwise acc vs causal CTR | **0.738** | random 0.5 | PASS |
| content_response_click | Upworthy Archive | time-forward (train early 60% → late 40%) | pairwise acc | **0.719** | random 0.5 | PASS (transfer) |
| content_response_click (ablation) | Upworthy Archive | random | pairwise (no-population) | 0.564 | 0.738 with population | population is load-bearing |
| attrition_dropout_hazard | IBM Telco (7032) | random 20% test | Brier | **0.141** | base-rate 0.198 | PASS; ECE 0.031 |
| attrition_dropout_hazard | IBM Telco | month-to-month → long-contract | Brier vs target base | 0.071 | 0.063 | **FAIL (negative transfer, preserved)** |
| response_occurrence_hazard | StackExchange (2500) | random 20% test | Brier | 0.249 | 0.246 | **NULL (preserved)** |
| argument_persuasion_success | CMV (1200) | random 20% test | Brier | 0.220 | 0.224 | **NULL (preserved)** |

Prior committed validations (unchanged, re-indexed): `exposure_response_hazard` (Higgs, held-out tied the
fitted logistic; nonlinearity ablation excludes 0), `engagement_momentum_persistence` (OmniBehavior, Brier
−0.0065 held-out + person-disjoint transfer), `social_preference_population` (BehaviorBench, LOGO transfer
passes, in-distribution loses to per-game histogram, public-goods misfit).

## 2. Published-estimate validation (Tier-4, NOT local)

The 6 `domain_restricted` packs carry the **study's own randomized/meta-analytic estimate** (recorded as
`kind="published_estimate"`, passed=True) — this is a verified causal effect in its original context, NOT a
held-out validation of our transport. Status discipline: these are explicitly **not** locally/transfer
validated or production-eligible; each family's `promotion_blockers("production_eligible")` is non-empty
(pinned by `test_published_estimate_cannot_reach_production_without_local_validation`).

## 3. Ablations (Part 16)

- **Selection ablations** (`wmv2_phase6_selection_eval.json`): names-only (0% Tier 1-4, 0% valid) → before
  (100% Tier 1-4 but 22.7% valid — false coverage) → no-applicability (90.9% Tier 1-4, 100% valid) → full
  Phase-6 (86.4% Tier 1-4, 100% valid). Removing applicability slightly raises Tier-1-4 by ignoring scenario
  fit — i.e. applicability correctly *demotes* mismatched-scenario families to honest lower tiers.
- **Mechanism ablation** (content_response): population heterogeneity 0.564 → 0.738 pairwise — the mechanism
  materially changes the prediction (not ornamental).
- **Diffusion ablation** (prior, preserved): nonlinear vs linear hazard Brier −0.00253 [−0.0034, −0.0017] on
  Higgs — nonlinearity is load-bearing.

## 4. Calibration

Telco attrition ECE 0.031 (well-calibrated); StackExchange 0.040; CMV 0.038 (calibrated but ≈ base rate).
Reliability curves are in `wmv2_phase6_fits.json` per dataset.

## 5. Preserved negative results (`wmv2_phase6_failures.json`, 7 records)

1. **Hawkes self-excitation** — held-out 24–72h count MAE 1098.9 > Poisson 973.0 → **quarantined**.
2. **Telco cross-subpopulation transfer** — month-to-month hazard does not beat the long-contract base rate
   (Δ+0.008 [0.004, 0.012]).
3. **StackExchange response** — surface features do not beat base rate on held-out.
4. **CMV persuasion** — surface features ≈ base rate (consistent with small persuasion effects,
   cf. Kalla-Broockman 2018).
5. **BehaviorBench public-goods** — FS→PG mapping misfit (W1 0.292), preserved.
6. **social_preference_population in-distribution** — loses to the per-game histogram baseline.
7. **simple_contagion (Higgs)** — linear hazard significantly worse than log-linear (comparator).

None was deleted, relabeled, or hidden. A null is graded as a null; a failed transfer blocks production but
does not erase the local held-out win.

## 6. Cost & latency

The Phase-6 fits + registry build + selection eval + traces + report run in seconds of pure-Python CPU (no
GPU, no external calls at build time); the only network use was **core-agent primary-source verification**
(WebSearch/WebFetch) during pack creation, not at runtime. Per-question compiler selection is O(families ×
processes) dictionary scoring — negligible next to rollout.
