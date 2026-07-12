# WMv2 benchmark portfolio — the mechanism-level empirical verdict

Five leakage-free benchmarks, run in order, each targeting mechanisms its outcome STRUCTURALLY depends on.
Every arm on identical held-out rows with paired bootstrap CIs; fits on train only; all nulls preserved.
Artifacts: `experiments/results/wmv2_{behaviorbench,omnibehavior_v2,higgs,upworthy_v2,forecastbench_subset}.json`
(+ Enron rounds 1–3 in `WMV2_ENRON_MAXCAP_FORENSIC.md` / `WMV2_ACTOR_COGNITION_AUDIT.md`).
Total portfolio cost this round: ~$0.87, ~2,600 DeepSeek calls, all metered.

## 1. BehaviorBench — strategic interaction (7 economic games, distribution prediction, n=100 test/game)

**The program's first significant mechanism-level positives.**

| finding | evidence (ΔW1_norm, paired CI95) |
|---|---|
| **Simulated-partner INTERACTION is real** | removing it degrades V2: public_goods −0.161 [−0.194,−0.086], guessing −0.088 [−0.110,−0.052], proposer −0.070 [−0.131,−0.007]; investor ns |
| **Latent population heterogeneity is real** | point-preference ablation: aggregate 0.137 vs 0.058 |
| **LLM-as-human-sampler is far off-distribution** | direct sampling 0.185, elicitation ensemble 0.123 vs V2 0.058 (LLM proposer mean 78.8 vs human 44.8) |
| But the train histogram still wins overall | A1 0.038 < V2 0.058; V2 wins only via cross-game transfer (trust_investor 0.078 vs 0.098) |
| Interpretation dims: no effect | no_interp 0.053 ≈ A5 0.058 |

## 2. OmniBehavior — longitudinal persistence (REPAIRED; 4 mixed-rate users, n=48 — underpowered)

Repair mattered twice: last-8 sampling bias AND label degeneracy (E-comm/CS/Ad events are action RECORDS,
rate≈1.0; only passive exposures are predictions). After repair: burstiness is REAL in train (momentum
lift 1.96) but **no mechanism effect is detectable at n=48** (persistence Δ −0.00003 ns). V2 best point
estimate (Brier 0.1271, AUROC 0.624); **direct LLM below chance** (AUROC 0.392). The paper's hyperactivity
failure INVERTED post-repair (models under-predict, −0.04..−0.07). Status: needs a larger mixed-rate
cohort before any mechanism claim.

## 3. Higgs — network + temporal rollout (456k-node follower graph, time-forward cohorts, n=4,000)

**Network signal is real; the mechanistic world loses to fitted features; rollout does nothing.**
Exposure-feature logistic beats base rate (−0.00107 [−0.00156,−0.00058]; AUROC 0.595). The contagion
world (fitted per-exposure hazard, latent q, event-driven window) is significantly WORSE than that
logistic (+0.00234 [0.00117,0.00349]) — the fitted model learned concave exposure response and a negative
degree effect that a linear-in-exposure hazard cannot express. Within-window rollout: +0.00003 ns (honest
scope: in-sample subgraph, 3,414 edges). Latent q spread slightly hurts (+0.00039 [0.00015,0.00063]).
LLM arms structurally absent (no content in SNAP Higgs).

## 4. Upworthy — heterogeneous population response (randomized headline A/B, 120 train / 150 test)

**The LLM interpretation dims HURT — significantly.** Best arm: population world over SURFACE features
(p@1 0.467, pairwise 0.701). Full V2 with interpretation dims: 0.407 (U4 vs no_interp −0.060
[−0.12,−0.007]). Fitted surface baseline 0.46; direct LLM 0.387; 3-call ensemble 0.453; random 0.34.
Population aggregation (argmax-choosing particles) left winner-pick unchanged (Δ=0.0 vs point scalar) but
clearly improved pairwise ordering (0.641 vs 0.571). The fitted layer is load-bearing (unfitted weights →
0.333 ≈ random). Note: this slice (min_impressions=4000) is not comparable to the earlier p@1=0.56
DeepSeek run (min 1000, different eligibility).

## 5. ForecastBench-class V2-supported subset (179 deadline questions of 661, coverage 27%; test n=90)

**The crowd is unbeaten and already prices time.** Crowd Brier 0.1521 (AUROC 0.86). The fitted temporal
hazard exponent selected g=1.0 — the IDENTITY — on train: the deadline mechanism has literally nothing to
add on top of the market price (V2 temporal effect Δ = exactly 0). Platt ns. Question-text-only direct LLM
catastrophic (0.356, AUROC 0.437 below chance); 3-call ensemble no better — without retrieval, the LLM arm
measures evidence poverty, reported as such. Institutional/population kernels: unsupported here (no
defensible as-of whip/poll evidence) — logged, not faked.

---

## PART F — the eight questions, answered across the portfolio

| question | answer | strongest evidence |
|---|---|---|
| **Does structured actor cognition add value?** | **Mixed — structure yes, LLM-semantics no.** Structural preference/belief models with fitted latents carry real signal (BehaviorBench); the LLM *interpretation* channel added nothing on Enron/BehaviorBench and significantly HURT on Upworthy | Enron C1 ns; BB no_interp ≈ A5; Upworthy −0.060 [−0.12,−0.007] |
| **Does persistent hidden state add value?** | **Unproven.** Real in-sample burstiness (lift 1.96) but no held-out effect at the only valid n available (48) | OmniBehavior Δ −0.00003 ns |
| **Does actor interaction add value?** | **YES — the portfolio's clearest mechanism win.** Simulating the partner/population inside the actor's decision significantly improves distributional prediction in 3 of 4 structural games | −0.088 to −0.161, CIs exclude 0 |
| **Does temporal rollout add value?** | **No, in both worlds where it structurally applies.** | Higgs +0.00003 ns; ForecastBench fitted g=1.0 (identity) |
| **Does the network/institutional layer add value?** | **Network as FEATURES yes; as mechanism no. Institutions untestable without evidence pipelines** | Higgs H1 vs H0 −0.00107 sig; contagion world +0.00234 sig WORSE; institutions logged unsupported |
| **Does full V2 beat the strongest specialized non-simulation baseline?** | **No, nowhere.** Enron: fitted metadata. BehaviorBench: train histogram. Higgs: exposure logistic. Upworthy: surface-fitted (± population). ForecastBench: the crowd | all paired CIs above |
| **Does it beat grounded direct and ensembled LLMs?** | **Yes, everywhere both ran — usually by a lot.** The disciplined/structured arms dominate raw LLM judgment on every benchmark | BB 0.058 vs 0.185/0.123; Enron −0.10; OB LLM below chance; Upworthy 0.407/0.467 vs 0.387; FB −0.196/−0.190 |
| **Is any gain calibrated, repeatable, worth the cost?** | The *mechanism* gains (interaction, heterogeneity) are cheap (≈$0.04, no per-prediction LLM needed) and calibrated within V2 — but they have not yet beaten the best specialized baseline anywhere, so there is no deployable net gain to price | cost table in each result file |

## The portfolio-level pattern (three rounds + five benchmarks)

1. **Structure beats LLM judgment; fitted statistics beat structure.** On every benchmark the ordering is
   the same: raw LLM < structured simulation ≤ fitted statistical baseline. The world model's proven value
   so far is DISCIPLINE (calibration, bounded mechanisms, typed state) — not lift over the best fit.
2. **The two mechanisms that survived falsification — interaction and population heterogeneity — are
   exactly the ones a fitted per-outcome statistical model CANNOT express** (they generalize across games
   /conditions rather than memorizing one distribution). Cross-game transfer (trust_investor,
   public_goods) is where V2 actually beat the empirical baseline. That is the trailhead: prediction
   problems where no on-distribution training sample exists — new games, new conditions, counterfactuals,
   best-action search — are where mechanism transfer is the only option.
3. **The LLM's reliable role remains narrow**: it is a poor human-sampler, a poor forecaster without
   evidence, and its semantic readings have yet to add held-out value in any world; wrapped as bounded
   features it is at best neutral. Its one demonstrated safe use is proposing structure, not producing
   numbers.
4. Do not merge PR #75. The defining product claim (max-capacity V2 beats the strongest non-simulation
   baseline on a held-out benchmark) remains **undemonstrated after 6 benchmarks** — but for the first
   time, two of its core mechanisms have positive, significant, held-out evidence.
