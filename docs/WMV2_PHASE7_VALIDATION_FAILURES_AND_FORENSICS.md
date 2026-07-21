# WMV2 Phase 7 — Validation, Failures & Forensics

Numbers here are copied from the committed machine-readable artifacts (`wmv2_phase7_validation.json`,
`wmv2_phase7_historical_backtests.json`, `wmv2_phase7_ablations.json`, `wmv2_phase7_counterfactuals.json`,
`wmv2_phase7_failures.json`, `wmv2_phase7_forensic_traces.json`). Where a claim and an artifact disagree, the
artifact is authoritative.

All fits are **validation-only selected** and **parsimony-first**: the nonlinear form is chosen on validation,
the test set is scored once, and if the nonlinear form does not beat the simpler form on held-out (paired
bootstrap CI includes 0), the **simpler form is kept**. No form is selected using test performance.

---

## A. Component-level identical-split comparison (Part 8, `wmv2_phase7_validation.json`)

Test-set Brier (lower better); selection done on validation only.

| dataset (category) | base | constant | logistic (Phase 6) | logistic+interaction | gam | gam+interaction | selected → promoted |
|---|---|---|---|---|---|---|---|
| **telco** (persistence) | 0.266 | 0.1907 | 0.1404 | 0.1401 | 0.1352 | **0.1351** | gam+interaction → **gam+interaction (BEAT)** |
| **stackexchange** (participation) | 0.576 | 0.2396 | **0.2391** | 0.2389 | 0.2415 | 0.2411 | logistic → **logistic (null preserved)** |
| **cmv** (persuasion) | 0.651 | 0.2262 | **0.2099** | 0.2103 | 0.2109 | 0.2107 | logistic → **logistic (null preserved)** |

Telco held-out paired ΔBrier (gam+interaction − logistic) = **−0.0054, CI [−0.0082, −0.0027]** (excludes 0).
Calibration (ECE) also improves for the GAM (0.0211 vs 0.0258). This is a genuine nonlinear win driven by the
real declining tenure→churn hazard (churn rate 0.485 at 0yr → 0.017 at 6yr). StackExchange and CMV: the GAM is
*worse* than logistic on held-out — the nonlinear smooths overfit features with no real nonlinear signal, and
parsimony correctly keeps logistic. **The Phase-6 nulls are preserved, not overturned.**

Upworthy content response (impression-weighted Brier, `wmv2_phase7_historical_backtests.json`): global-CTR
baseline 0.000129 ≪ linear headline 0.000613 ≈ nonlinear headline 0.000609. Headline features are null;
the pooled/global baseline dominates. Honest negative.

---

## B. End-to-end historical backtests through WorldState + StateDelta (mandatory directive)

`wmv2_phase7_historical_backtests.json`. Primary comparison per backtest = **full Phase-7 form vs the prior
Phase-6 form**, identical initial state / as-of cutoff / seeds / particles / horizon; only the mechanism form
differs. Every arm executes through the real rollout engine (StateDeltas emitted, terminal read from state).

| category | dataset | Phase-6 arm | Phase-7 arm | result |
|---|---|---|---|---|
| **persistence** | telco churn | logistic (Brier 0.1430) | GAM (Brier **0.1375**) | **Phase 7 wins end-to-end**, paired ΔBrier −0.0055 CI [−0.0085, −0.0024]; beats constant (0.1904) too. Leakage-free (`total_charges` dropped). |
| **diffusion** | baby names 1880–2008 | linear_growth (RMSE 0.0968) | logistic_growth (RMSE **0.0288**) | **Phase 7 beats Phase 6** on trajectory RMSE, paired −0.068 CI [−0.130, −0.015]. **Honest caveat:** naive persistence (RMSE 0.0174) is competitive — post-peak decline is unmodeled by a growth-only mechanism. |
| **content** | Upworthy A/B | linear headline (0.000613) | nonlinear headline (0.000609) | **honest null** — neither beats the pooled/global CTR baseline (0.000129). |

**Execution proof.** The baby-name trajectory is stepped year-by-year: `NonlinearStateStepOperator` emits a
`StateDelta` per forecast year and schedules the next year as a follow-up event (e.g. Michael: 36 steps
executed; Robert: 74). The telco backtest runs each held-out customer through `NonlinearMechanismOperator`
over 40 branches and reads P(churn) as the terminal-churn frequency (the contract-projection readout).

**Aggregate (honest).** Phase 7 beats the prior Phase-6 form in **2/3 categories** (persistence, diffusion);
against *all* required baselines including naive persistence, only **persistence (telco)** is an unambiguous
clean win; diffusion beats the Phase-6 growth form but not naive persistence; content is a preserved null. The
lift is **real but category-specific — not a universal simulation-accuracy claim.**

---

## C. Ablations (Part 23, `wmv2_phase7_ablations.json`)

Telco ablation ladder (test Brier, incremental vs `linear_only`):

| arm | Brier | vs linear (paired) |
|---|---|---|
| linear_only | 0.1404 | — |
| gam_single_smooth (tenure only) | ~0.1360 | negative (helps) |
| gam_no_interaction | ~0.1352 | negative (helps) |
| full_phase7 (gam + interaction) | 0.1351 | negative (helps) |

The tenure smooth carries most of the gain; the interaction adds a small increment. On StackExchange/CMV every
nonlinear arm is ≥ linear (the components are ornamental there — correctly not promoted).

## D. Counterfactual sensitivity (Part 24, `wmv2_phase7_counterfactuals.json`)

All four sweeps obey their form's invariants: telco tenure↑→churn↓ (declining hazard); Hill exposure↑→saturating
increase; repeated exposure→fatigue decline; adoption-share↑→growth increment→0 at L. Counterfactual coherence
is a sanity check, **not** a substitute for held-out validation (stated in the artifact).

## E. Numerical stability (Part 17)

Forms guard division-by-zero to finite limits and convert genuine overflow (`x**n`, `log≤0`) into a recorded
`FormError` rather than crashing the rollout; `safety.safe_prob`/`safe_rate` clamp only where mathematically
justified (probability∈[0,1], rate≥0) and **record every clamp that bit** so "stable only because clamped"
shows up in the ledger. Self-exciting branching α≥1 is refused (explosive). `StabilityMonitor` caps endogenous
event generation (event-storm guard). Covered by `tests/test_wmv2_phase7_forms.py` and `_execution.py`.

---

## F. Failures & quarantines (Part 26, `wmv2_phase7_failures.json`, append-only)

| id | family | type | disposition |
|---|---|---|---|
| `p7_hawkes_preserved` | hawkes_self_excitation | quarantine | **quarantined** (MAE 1098.9 > Poisson 973.0 on Higgs — preserved, never re-promoted) |
| `p7_stackexchange_null` | response_occurrence_hazard | null_improvement | retained_linear |
| `p7_cmv_null` | argument_persuasion_success | null_improvement | retained_linear |
| `p7_upworthy_content_null` | content_response_click | null_improvement | retained_linear |
| `p7_telco_transport` | attrition_dropout_hazard | transfer_failure | preserved (nonlinear does not transport across contract types) |
| `p7_babyname_decline_unmodeled` | bass_diffusion | extrapolation_failure | preserved (growth-only cannot capture post-peak decline) |

The Hawkes quarantine is asserted still-present by `tests/test_wmv2_phase7_adversarial.py`
(`test_hawkes_stays_quarantined_in_registry`). No null is relabeled a success.

## G. Transport / extrapolation (Part 15)

Telco cross-contract transfer (`wmv2_phase7_validation.json → telco_transfer`): fit on month-to-month
customers, test on 1yr/2yr contracts. GAM Brier 0.0728 vs logistic 0.0696; paired Δ **+0.0032, CI
[0.0018, 0.0044]** — the nonlinear form is *worse* out-of-group because the fitted tenure spline extrapolates
across barely-overlapping tenure support. Verdict: **domain_restricted** (the honest transport failure the
spec demands, not hidden).

---

## H. Forensic traces (Part 29, `wmv2_phase7_forensic_traces.json`)

Each trace is produced by *actually running the operator* in a real WorldState; the StateDeltas and follow-up
events in the artifact are the real objects the rollout saw.

1. **`telco_attrition_gam`** — question ("will cust_042 churn?") → family `attrition_dropout_hazard` →
   candidates {logistic, logistic_interaction, gam, gam_interaction} → selected `gam_interaction` →
   observed-feature context → WorldState reads cust_042's features → event `nonlinear_transition` → GAM
   computes P(churn) (raising short-tenure risk above the logistic's constant slope) → **StateDelta**
   `quantities[churn]: None → <value>` → terminal churn read by downstream retention logic. All 24
   anti-scaffolding questions answered in the artifact.
2. **`diffusion_logistic_saturation`** — adoption trajectory → `logistic_growth` stepped through WorldState →
   a StateDelta per year + a scheduled next-year event → terminal share bounded by L (a linear form would
   overshoot past L).
3. **`posterior_jensen_gap`** — Hill saturation with an uncertain Phase-3 half-saturation `k` → per-particle
   propagation gives the posterior-correct `E[f(X)]` vs the biased `f(E[X])`; the operator writes the
   posterior-correct value and stamps `jensen_gap` on the StateDelta.

## I. Cost & latency

No LLM calls anywhere in Phase-7 fitting/validation/backtests (`llm_calls: 0`, `est_cost_usd: 0`). Runtimes:
validation ~13s, historical backtests ~23s, traces ~1s, registry build <1s (see each artifact's `_meta`).
