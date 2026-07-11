# WMv2 standing evaluation report

## Reference World A — Enron individual message response (first real held-out V2 benchmark)

**Setup:** 60k messages, subject+direction reply reconstruction (rate 3.4%, delay p50 1.3h); leak-free
features (strictly-prior messages only); TIME-FORWARD (n=400, base@7d 11.5%) and PERSON-DISJOINT (n=400,
base@7d 2.2%) splits; mechanisms fitted on train only; arms I0/I1/I4–I8; paired bootstrap CIs vs I1;
20 machine-readable causal traces committed. LLM arms I2/I3: **not yet run** (flagged, costed subsample).

**Run 1 (INVALID, preserved):** header-only reply reconstruction found 0 replies → degenerate. Fixed.

**Run 2 (negative, preserved):** full V2 lost badly (Brier@7d 0.179 vs I1 0.094, Δ=+0.085 CI[0.062,0.109]),
error growing with horizon — diagnosed as the named temporal-fidelity failure: per-bucket hazard applied
per check-opportunity, compounding ("30 days ≠ 30 identical guesses"). Fix: per-opportunity h solves
1−(1−h)^n = H_bucket.

**Run 3 (corrected, FINAL for this round), time-forward Brier@7d:**

| arm | Brier@7d | Δ vs I1 (paired CI95) | verdict |
|---|---|---|---|
| I1 fitted statistical model | **0.0941** | — | **the bar** |
| I0 base rate | 0.1084 | +0.014 [0.002, 0.029] | I1's mechanisms are REAL |
| I4 V2 no latent | 0.0954 | +0.001 [−0.003, 0.005] | ≈ I1 |
| I5 V2 latent, no rollout | 0.1008 | +0.007 [−0.002, 0.016] | ns |
| I6 V2 no relationship | 0.0997 | +0.006 [−0.002, 0.014] | ns |
| **I7/I8 full V2** | **0.0964** | **+0.002 [−0.002, 0.007]** | **matches, does NOT beat** |

Horizon degradation eliminated (0.090→0.096 flat) — the integration fix is validated. Person-disjoint: all
arms collapse to the 2.2% base rate (tiny positives; nothing differentiates; V2 arms trivially but
significantly worse by ~0.0007 — noise-scale miscalibration on near-zero-rate strangers).

## Keep / revise / disable

| Component | Decision | Evidence |
|---|---|---|
| Fitted mechanisms (hazard, recipient/relationship rates, workload, hour/weekday) | **KEEP** | I1 beats base rate, Δ=+0.014, CI excludes 0 |
| Hazard-integration correction | **KEEP** | eliminated horizon-growing error (0.179→0.096) |
| Event-driven rollout | **NO EVIDENCE** (keep as research) | matches the closed-form model at ~10× compute; no lift |
| Latent attention state | **NO EVIDENCE** | I5/I7 spreads within noise |
| Relationship history inside the sim | **NO EVIDENCE** | I7−I6 ns |
| LLM message-content policy (I2/I3) | **UNRUN** | the open question: semantic content is the one signal I1 cannot see |

**Status label: architecture-validated + first-benchmark NO-EVIDENCE-OF-LIFT.** The V2 runtime now
*reproduces* a fitted statistical model through a real event-driven world (a nontrivial correctness result —
run 2 shows how easily simulation gets this wrong), but the defining claim (structured simulation adds
held-out predictive value) remains **undemonstrated**. Do not merge PR #75.

## Portfolio status

| Benchmark | Status |
|---|---|
| Enron (Ref. World A) | run (above); next: I2/I3 LLM arms — content is the untested signal |
| Upworthy / ForecastBench / crowd / BehaviorBench / OmniBehavior(repair) / Higgs | staged per `WMV2_BENCHMARK_MAP.md`, unrun for V2 |
| Forward ledger V2 wiring | pending |
