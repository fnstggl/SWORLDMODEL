# WMv2 Production Acceptance Gates (Phase 17)

*Honest pass/partial/fail against every applicable gate. "Partial" and "fail" are stated plainly — the
purpose of this round was to fix real bottlenecks, not to disguise findings.*

## ARCHITECTURE gates

| gate | status | evidence |
|---|---|---|
| arbitrary questions compile without scripted plans | **PASS** | 104 held-out NL questions, 16 domains, real LLM, no scripted plans; 70% compile, 0 crashes (`WMV2_COMPILER_VALIDATION.md`) |
| actor-specific information boundaries enforced | PASS | `observable_view` boundary; tested (`test_world_model_v2.py`) |
| typed actions update one shared world | PASS | StateDelta per transition; endogenous action→event chains (`test_wmv2_tier_a_fixes.py`) |
| mechanisms execute with provenance | PASS | 47 executable families; provenance-honest (100% in generality run) |
| persistent state affects future actions | **PASS** | OmniBehavior persistence Δ−0.0065 held-out + person-disjoint transfer Δ−0.027 |
| interaction changes outcomes where applicable | PARTIAL | BehaviorBench interaction significant (prior); A4 endogenous events enable chains, tested; not yet a benchmarked multi-actor thread |
| institutions execute real rules | PARTIAL | validate_action + run_vote + closed rule-kind registry (7 kinds); institutional DECISION dynamics not yet executable families (compiler abstains on them) |
| uncertainty includes state+parameter+structural | PASS | `decompose_uncertainty` separates all three; structural hypotheses in `run_filtered` |
| dynamic recompilation works | PARTIAL | `recompile()` versions + chains; trigger detection is interface-level, not validated on adversarial regime-change |
| terminal readout mandatory | PASS | contract refuses to run without readout; option-space coverage guards no-op worlds |

## MECHANISM REGISTRY gates

| gate | status | evidence |
|---|---|---|
| ~40 real families | **PASS** | 47 families, ALL with executable transitions (zero empty) |
| 100-200 parameter packs OR documented path with substantial coverage | PARTIAL | 6 packs on distinct empirical contexts; documented path via `ingestion.py`; NOT 100+ (honest gap) |
| no empty production entries | PASS | `empty_entries == []` in the registry summary |
| applicability scoring | PASS | `applicability.py` real per-axis scoring, wired into compile |
| transport limits | PASS | every family `transport_risk` + citation limits |
| validation history | PASS | ValidationRecords with passed/failed |
| failed results preserved | **PASS** | Hawkes quarantined (failed), public_goods failure recorded, all nulls kept |

## EMPIRICAL gates

| gate | status | evidence |
|---|---|---|
| interaction effect replicated | PARTIAL | BehaviorBench interaction significant prior round; this round's universal policy shows small interaction gain (0.099 vs 0.108) |
| population heterogeneity effect replicated | PASS | FS mixture (heterogeneity) beats selfish point (0.099 vs 0.125) |
| nonlinear diffusion improves over prior linear | **PASS** | Higgs: nonlinear Δ−0.00253 vs linear (CI excl 0); closes gap to fitted logistic |
| persistence tested at adequate power | **PASS** | n=7074, power 0.993, effect CI excludes 0 |
| compiler tested on held-out domains | **PASS** | 16 domains, 104 questions |
| historical benchmark completed | PARTIAL | resumable pipeline built + run on a verified subset; NOT the 1000-question target (remaining work documented) |
| forward ledger active | PASS | append-only versioned locks, tested |
| calibration acceptable | PARTIAL | machinery cuts ECE on miscalibrated data; on the crowd corpus no miscalibration to fix (honest null) |
| abstention meaningful | **PASS** | 30% principled abstention in the generality run + signal-driven abstention policy |
| best-action benchmarks executed | PASS (result negative) | matched-CF mechanics validated; real-intervention decision lift negligible (Upworthy) |

## PRODUCT-CLAIM gates (does full V2 beat the alternatives?)

| comparison | verdict |
|---|---|
| beats grounded direct LLM | **YES** where both run (BehaviorBench 0.099 vs 0.185; historical B1 weak) |
| beats direct ensemble | YES (BehaviorBench 0.099 vs 0.123) |
| beats observer panel | not re-run this round (prior: structure ≥ panel) |
| beats generic analogical forecasting | not tested this round |
| beats crowds where present | **NO** — crowd unbeaten (ForecastBench + historical benchmark) |
| beats specialized models where present | **NO** — specialist/histogram wins in-distribution everywhere |
| adds value in cold-start and transfer | **YES, partially** — persistence transfers to new people (Δ−0.027); FS policy transfers to 3/4 held-out games; nonlinear diffusion ties the fitted ceiling |
| adds real decision lift | **NO** on the available randomized-intervention benchmark |

## Overall

**Do NOT merge PR #75 on implementation-completeness alone.** The defining product claim (full V2 beats the
strongest non-simulation baseline on an ordinary in-distribution benchmark) remains **undemonstrated** — the
crowd and task-specific fits are still unbeaten in-distribution. What IS newly demonstrated, with
adequately-powered held-out evidence: **persistence works out of sample and transfers to new people**,
**nonlinear diffusion closes the gap to the fitted ceiling**, **the general compiler runs end-to-end on
arbitrary questions with a real LLM**, and **the semantic channel is correctly quarantined by evidence**.
The architecture's real value remains transfer and discipline, not in-distribution lift — now with more
positive, significant, preserved evidence than any prior round.
