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

## Round 2 — MAX-CAPACITY (content ON, LLM policy ON): `V2_METADATA_TEMPORAL` renamed, content tested

The prior "full V2" (I7/I8) never read message content — renamed **`V2_METADATA_TEMPORAL`** (arm E5). The
max-capacity round adds a content-conditioned recipient policy through the typed boundary (exact email →
DeepSeek `deepseek-chat` → bounded multiplier on the fitted base → particle/event rollout) and ablates
every experimental mechanism leave-one-out against the full arm E10. 1,460 API calls, ~$0.42, particles=24.
Full detail + 20 forensic traces: `docs/WMV2_ENRON_MAXCAP_FORENSIC.md`,
`experiments/results/wmv2_enron_maxcap.json`.

**Time-forward, apples-to-apples Brier@7d (all arms on the same n=120 LLM subsample), paired CIs:**

| arm | Brier@7d | Δ vs E1 (paired CI95) | verdict |
|---|---|---|---|
| **E1 fitted metadata** | **0.0577** | — | **the bar** |
| E2 non-LLM text BoW | 0.0855 | worse | text alone < metadata |
| E3 raw one-shot LLM | 0.1638 | — | miscalibrated; **worse than BoW** (E3−E2 +0.078 [0.020,0.139]) |
| E5 V2_METADATA_TEMPORAL | 0.0584 | +0.001 ns | ≈ E1 |
| E6 content only | 0.0663 | — | content alone < metadata |
| **E10 MAX-CAPACITY** | **0.0622** | **+0.004 [−0.005, +0.016]** | **matches, does NOT beat** |

Mechanism leave-one-out (E10 − Ex, all NS): event rollout −0.005 [−0.020,+0.012], latent +0.0004
[−0.002,+0.003], relationship −0.008 [−0.020,+0.002]. Content effect E10 vs E5 +0.004 [−0.003,+0.013] NS.
Person-disjoint: everything collapses to the 2% base; E10 vs E1 ns; relationship effect exactly 0 (no prior
relationship exists for held-out persons).

**The one positive is methodological:** the raw LLM read loses (Brier 0.164, worse than BoW), but wrapping
that same read as a bounded multiplier on the fitted base inside the typed world rescues it to parity
(E10 vs E3 −0.10 [−0.15,−0.06]). The boundary's value here is **calibration discipline, not lift.**

## Round 3 — STRUCTURED ACTOR COGNITION (the scalar bottleneck audited, replaced, and tested)

Audit proved round 2's E10 carried content through ONE scalar (`reply_propensity`) into a two-action
fitted hazard → renamed **`V2_SCALAR_CONTENT`**, demoted to baseline **C0**. Replacement (universal
`swm/world_model_v2/actor_cognition.py`): typed 12-dim interpretation → TRAIN-fitted calibration layer →
typed actions (reply_now/reply_later/clarify/delegate/ignore) → correlated hidden actor state → dynamic
attention → relationship transitions, all through the shared runtime. Universality proven by a negotiation-
domain acceptance test. Full audit + results: `docs/WMV2_ACTOR_COGNITION_AUDIT.md`.

**Time-forward Brier@7d (n=120 identical rows):** E1 fitted metadata **0.0577** | C1 structured
interpretation **0.0587** (+0.001 [−0.005,+0.006] ns) | C0 scalar world 0.0618 | C6 max structured actor
0.0640 (+0.006 ns) | every ladder rung (typed actions, hidden state, dynamics, relationship) **NS**.
The fitted layer learned real semantics (thread_continuity +0.36, task_ownership +0.31, obligation +0.22,
social intent −0.21; metadata anchor discounted to 0.68) — the semantic channel now demonstrably carries
information — yet the score lands on the SAME number as metadata alone. **Three architectures of increasing
cognitive fidelity (none / scalar / structured) all match E1: the ceiling on this task is metadata signal +
label noise, not the content channel's richness.** Lead preserved: person-disjoint C1 AUROC 0.966 /
PR-AUC 0.333 vs E1 0.788 / 0.091 (n=60, ~1–2 positives — a lead to test at scale, not a claim).

## Keep / revise / disable

| Component | Decision | Evidence |
|---|---|---|
| Fitted mechanisms (hazard, recipient/relationship rates, workload, hour/weekday) | **KEEP** | I1 beats base rate, Δ=+0.014, CI excludes 0 |
| Hazard-integration correction | **KEEP** | eliminated horizon-growing error (0.179→0.096) |
| Event-driven rollout | **NO EVIDENCE** (keep as research) | matches the closed-form model at ~10× compute; no lift |
| Latent attention state | **NO EVIDENCE** | I5/I7 spreads within noise |
| Relationship history inside the sim | **NO EVIDENCE** | I7−I6 ns |
| LLM message-content policy (E3/E6/E10) | **RUN → NO EVIDENCE OF LIFT** | E10 vs E1 +0.004 ns; E10 vs E5 (content) +0.004 ns; raw LLM (E3) *worse* than BoW. Boundary adds calibration discipline, not predictive lift |

**Status label: architecture-validated + first-benchmark NO-EVIDENCE-OF-LIFT.** The V2 runtime now
*reproduces* a fitted statistical model through a real event-driven world (a nontrivial correctness result —
run 2 shows how easily simulation gets this wrong), but the defining claim (structured simulation adds
held-out predictive value) remains **undemonstrated**. Do not merge PR #75.

## Portfolio status — ALL FIVE PORTFOLIO BENCHMARKS RUN (see `WMV2_PORTFOLIO_VERDICT.md` for the full verdict)

| Benchmark | Structural test | Result |
|---|---|---|
| Enron (Ref. World A, 3 rounds) | content/actor cognition | matches fitted metadata, never beats (all rounds) |
| **BehaviorBench (Ref. World B)** | **interaction, heterogeneity** | **first significant mechanism positives: interaction −0.088..−0.161, latent heterogeneity essential; train histogram still wins aggregate; V2 ≫ LLM arms** |
| OmniBehavior (Ref. World C, repaired) | persistence | burstiness real in train (lift 1.96); no held-out effect at n=48 (underpowered); direct LLM below chance |
| Higgs (Ref. World D) | network, rollout | network features sig. real; mechanistic contagion sig. WORSE than fitted logistic; rollout ns |
| Upworthy (Ref. World E) | population heterogeneity | surface-fitted + population best (p@1 0.467); LLM interpretation dims sig. HURT (−0.060); population helps pairwise ordering only |
| ForecastBench deadline subset (27% coverage) | temporal/institutions | crowd unbeaten (0.152); fitted hazard exponent = identity (temporal adds exactly 0); text-only LLM catastrophic |
| Forward ledger V2 wiring | — | pending |

**Cross-portfolio pattern:** raw LLM < structured simulation ≤ fitted statistical baseline, everywhere.
The two surviving mechanisms (interaction, population heterogeneity) are exactly the ones a per-outcome
fit cannot express — cross-condition transfer is the trailhead. Do not merge PR #75.
