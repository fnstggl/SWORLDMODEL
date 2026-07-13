# WMv2 Phase 3B — Architecture and Repair
*The repairs implemented in this run, how they are fit and frozen, and their DEV-set behavior (optimistic; the honest number is the locked test in `WMV2_PHASE3B_REAL_VALIDATION.md`).*

## Repairs implemented (production code)
1. **Real reference-class priors** (`swm/world_model_v2/phase3b_reference_priors.py`, Part D) — data-backed as-of base rates (FOMC action, incumbent-party retention, shutdown frequency, index/threshold crossings, release slip rates, ceasefire hazards, corporate actions) with provenance, transport-risk-widened. Replaces the generic 0.50.
2. **Calibrated rate posterior** (`swm/world_model_v2/phase3b_repair.py`, Parts C/E/F) — the real `DirectionalRateModel` likelihoods tempered by `gamma` (shrinkage), mixed with a flat no-information model (`no_info_mix`), optionally flattened by `post_temp`. Fights the over-concentration diagnosed in Failure §A.
3. **Convex safe blend + evidence-quality gate** (Parts F/L) — the repaired forecast is a frozen convex combination (in logit space) of the Phase-2 terminal and the calibrated Phase-3 rate, with weight w∈[0,1] so **Phase-2 is never inverted**; below a support threshold it FALLS BACK to Phase-2. The system can conclude “this evidence does not justify moving Phase-2.” Pre-registered robustness constraints (gamma≤0.7, w_phase2≥0.5) were set from the diagnosis before the test.

## Frozen parameters (fit on DEV train, selected on DEV validation)
- rate calibration: use_ref_prior **False**, gamma **0.7000**, no_info_mix **0.0000**, post_temp **1.0000**
- blend: p_final = sigmoid(w·logit(p₂) + (1−w)·logit(p₃_cal)), w_phase2 = **0.5000** (convex, Phase-2 never inverted)
- gate: fall back to Phase-2 when NON-NEUTRAL effective observations < **4**
- dev split: train **12** / validation **11** questions (event-family-disjoint, no family crosses the split)

## DEV-set scores (OPTIMISTIC — repaired is fit here)
| arm | Brier | log-loss | ECE |
|---|---|---|---|
| prior_only | 0.2507 | 0.6946 | 0.1444 |
| phase2 | 0.2592 | 0.7125 | 0.2597 |
| phase3_current | 0.2525 | 0.6942 | 0.2490 |
| phase3_repaired (dev-fit) | 0.2356 | 0.6637 | 0.1537 |

Gate modes on dev: {'gate_phase2_fallback': 13, 'blended': 10}

> DEV improvement is expected by construction (the repair is fit here). It is reported only to show the repair behaves; the acceptance decision is made ONLY on the untouched locked test.
