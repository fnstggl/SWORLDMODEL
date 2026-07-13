# WMv2 Phase 3B — Real Validation (Locked Test)
*The untouched, event-family- and temporally-disjoint held-out test. Run ONCE after all parameters were frozen. This is the only number that decides acceptance.*

Completed **34** / 35 questions (retrieval 2026-07-13T16:54:29Z, seed 0).

## Per-arm scores (vs realized outcome)
| arm | n | Brier ↓ | log-loss ↓ | dir acc ↑ | ECE ↓ |
|---|---|---|---|---|---|
| `prior_only` | 34 | 0.2335 | 0.6590 | 0.3824 | 0.2452 |
| `phase2` | 34 | 0.2639 | 0.7576 | 0.6176 | 0.2234 |
| `phase3_current` | 34 | 0.2142 | 0.6456 | 0.6176 | 0.2522 |
| `phase3_repaired` | 34 | 0.2430 | 0.6824 | 0.6765 | 0.2007 |

## Key paired comparisons (negative ⇒ improves vs Phase-2)
- **repaired Phase-3 vs Phase-2**: mean Brier diff **-0.0209** 95% CI **[-0.0426, -0.0041]**; mean log-loss diff **-0.0752** 95% CI **[-0.1853, -0.0086]**
- current Phase-3 vs Phase-2 (for reference): mean Brier diff **-0.0497** 95% CI **[-0.0794, -0.0184]**
- per-question: repaired better **12**, Phase-2 better **5**, tie **17**

## Domain breakdown (Brier)
| domain | n | Phase-2 | repaired |
|---|---|---|---|
| elections | 6 | 0.2515 | 0.2492 |
| econ | 3 | 0.2750 | 0.2495 |
| macro | 2 | 0.5525 | 0.4315 |
| finance | 7 | 0.2319 | 0.2091 |
| tech | 5 | 0.2927 | 0.2709 |
| geopolitics | 3 | 0.2309 | 0.2121 |
| sports | 4 | 0.2608 | 0.2577 |
| science | 3 | 0.1848 | 0.1714 |
| politics | 1 | 0.1575 | 0.1575 |

## Per-question deltas (repaired − Phase-2 Brier)
| qid | y | p₂ | p_repaired | ΔBrier | verdict |
|---|---|---|---|---|---|
| `modi_pm` | 1 | 0.6375 | 0.6230 | 0.011 | phase2_better |
| `afd_first_de` | 0 | 0.5385 | 0.5482 | 0.011 | phase2_better |
| `fr_nfp` | 1 | 0.3182 | 0.3182 | 0.000 | tie |
| `mx_sheinbaum` | 1 | 0.6375 | 0.6256 | 0.009 | phase2_better |
| `jp_ldp_maj` | 0 | 0.6000 | 0.5624 | -0.044 | repaired_better |
| `ven_maduro` | 1 | 0.6379 | 0.6379 | 0.000 | tie |
| `ecb_jun24` | 1 | 0.3182 | 0.3182 | 0.000 | tie |
| `boe_aug24` | 1 | 0.7333 | 0.7333 | 0.000 | tie |
| `boj_jul24` | 1 | 0.4625 | 0.5390 | -0.076 | repaired_better |
| `cpi_sep24` | 1 | 0.0375 | 0.2107 | -0.303 | repaired_better |
| `unemp_jul24` | 1 | 0.5775 | 0.5101 | 0.061 | phase2_better |
| `gold_2500` | 1 | 0.5333 | 0.5333 | 0.000 | tie |
| `us10y_5pct` | 0 | 0.5400 | 0.5400 | 0.000 | tie |
| `eth_5000` | 0 | 0.4304 | 0.4304 | 0.000 | tie |
| `nvda_3t` | 1 | 0.5190 | 0.5920 | -0.065 | repaired_better |
| `eth_etf` | 1 | 0.4364 | 0.5401 | -0.106 | repaired_better |
| `grok2` | 1 | 0.4507 | 0.4507 | 0.000 | tie |
| `llama3` | 1 | 0.5455 | 0.5455 | 0.000 | tie |
| `tiktok_law` | 1 | 0.3521 | 0.3544 | -0.003 | repaired_better |
| `tesla_robotaxi` | 1 | 0.5333 | 0.5333 | 0.000 | tie |
| `iran_israel` | 1 | 0.5811 | 0.6281 | -0.037 | repaired_better |
| `taiwan` | 0 | 0.3699 | 0.3699 | 0.000 | tie |
| `guyana` | 0 | 0.6167 | 0.6009 | -0.019 | repaired_better |
| `nba_celtics` | 1 | 0.5250 | 0.5125 | 0.012 | phase2_better |
| `euro_spain` | 1 | 0.5250 | 0.5371 | -0.011 | repaired_better |
| `masters_schef` | 1 | 0.4750 | 0.4875 | -0.013 | repaired_better |
| `nhl_panthers` | 1 | 0.4375 | 0.4375 | 0.000 | tie |
| `disney_peltz` | 0 | 0.4930 | 0.4930 | 0.000 | tie |
| `paramount` | 1 | 0.5395 | 0.6574 | -0.095 | repaired_better |
| `boeing_ceo` | 1 | 0.5082 | 0.5082 | 0.000 | tie |
| `moon_crew` | 0 | 0.3770 | 0.3770 | 0.000 | tie |
| `starliner` | 0 | 0.4430 | 0.4430 | 0.000 | tie |
| `hurricane_c5` | 1 | 0.5352 | 0.5806 | -0.040 | repaired_better |
| `cannabis` | 0 | 0.3968 | 0.3968 | 0.000 | tie |

## Pre-registered acceptance gates (Part K — frozen before the test opened)
- G1_brier_not_worse: **PASS**
- G2_logloss_not_worse: **PASS**
- G3_one_primary_CI_favorable: **PASS**
- G4_no_significant_regression: **PASS**
- G5_ece_not_materially_worse: **PASS**

**Locked-test verdict (by the pre-registered gates): PHASE3B_IMPROVES**.

## Critical honest caveats (read before citing the verdict)
1. **The UNREPAIRED current Phase-3 also beat Phase-2, and by MORE on Brier** (current 0.2142 vs repaired 0.2430 vs Phase-2 0.2639). So the headline is *Phase-3 evidence assimilation beat Phase-2 on this set*; the REPAIR's specific contribution is **calibration** (best ECE 0.2007 vs 0.2234/0.2522), **directional accuracy** (best 0.6765), and **catastrophic-regression safety** (gate/blend), NOT extra Brier over leaving Phase-3 alone.
2. **Arm orderings are INCONSISTENT across the three datasets** — committed backtest (Phase-3 worst), fresh dev capture (Phase-3 ≈ Phase-2), locked test (Phase-3 best). This is strong evidence of **high variance / low power** (n=34 < the 75-question target). The effect is small relative to the noise.
3. **prior_only is competitive on Brier** (0.2335) but has poor directional accuracy (0.3824); it looks good only by never taking a strong position. Phase-3 genuinely beats it on both Brier and direction.
4. **No domain regressed** for the repaired arm and there were **no catastrophic per-question regressions** (the largest repaired-vs-Phase-2 loss was ~0.06 Brier), because the gate/convex-blend caps downside — a real robustness property.
5. **Not production-validated.** The pre-registered gates pass on the untouched test, which is a genuine positive, but underpower + cross-set instability mean this is **promising, not conclusive**. Recommend the repaired (gated/blended) arm as the safer default CANDIDATE pending an adequately-powered (≥75) test; Phase-2 remains the conservative fallback.


---

# FINAL REPORT — brutally honest answers
1. **Original regression — variance, harness, or architecture?** Substantially **retrieval/sample variance**. The committed regression reproduced exactly from the frozen forecasts, but a fresh re-run of the identical production path flipped it (fresh Phase-3 Brier 0.2525 vs Phase-2 0.2592). No harness/scoring error was found (scoring reproduced bit-for-bit). A real architectural weakness (over-responsiveness/miscalibration) coexists but is not, by itself, a stable net-harm on this set.
2. **Was evidence double-counted?** No, not additively. Mechanism is **override, not addition** (the posterior REPLACES the terminal rate). The two forecasts are redundant (corr(logit p₂,p₃)=0.7136) off the shared bundle.
3. **Was the generic outcome-rate posterior harmful?** On the committed run yes; on re-run it was ~neutral-to-slightly-helpful. It is **over-confident** (dev ECE 0.2490 vs prior 0.1444) and can catastrophically regress on surprise events. It is retained only as a **calibrated, gated, subordinate** signal.
4. **Which observation models were misspecified?** The directional model's fixed sens/spec (0.85 for 'strong') concentrate too fast; a handful of weak directional claims move the terminal 10-30 pts. Repaired by global likelihood shrinkage (gamma=0.7). Per-claim-class hierarchical fits remain a documented dependency.
5. **Did real reference priors improve results?** **No** on this dev set — reference priors were built (Part D) with provenance but were **not selected by validation** (use_ref_prior=False). Honest negative; retained as an ablation, not in the frozen path.
6. **Did fitted likelihoods improve results?** Global shrinkage (gamma=0.7) is the only 'fitting' done to the likelihood; it modestly improves dev calibration. A full hierarchical likelihood refit was not performed (no labeled corpus) — documented dependency.
7. **Which representation worked best?** On DEV, the **calibrated rate posterior mean** blended 50/50 with the Phase-2 terminal; the raw scalar terminal posterior was worst-calibrated. Typed causal-latent representations (Part C) were not built to real data this run.
8. **Did scenario-specific latent inference outperform generic evidence voting?** Not tested at scale — typed causal latents were not fit to data this run (documented dependency). The generic rate posterior remains the signal, now calibrated and gated.
9. **Did repaired Phase-3 beat Phase-2 on the untouched final test?** Locked verdict: **PHASE3B_IMPROVES**. Repaired better on 12 / Phase-2 better on 5 / tie 17 of 34.
10. **Paired differences + CIs (locked):** Brier diff **-0.0209** CI **[-0.0426, -0.0041]**; log-loss diff **-0.0752** CI **[-0.1853, -0.0086]**. (Negative ⇒ repaired improves.) Current Phase-3 vs Phase-2 Brier diff **-0.0497** CI **[-0.0794, -0.0184]**.
11. **Which domains improved?** (directional, underpowered) elections, econ, macro, finance, tech, geopolitics, sports, science.
12. **Which domains regressed?** (directional, underpowered) none.
13. **Which acceptance gates passed?** G1_brier_not_worse, G2_logloss_not_worse, G3_one_primary_CI_favorable, G4_no_significant_regression, G5_ece_not_materially_worse.
14. **Which failed?** none.
15. **Software implemented?** Yes — reference priors, calibrated-posterior repair module, blend+gate, offline fit, locked-test harness are committed with tests passing.
16. **Executes end-to-end?** Yes — the repaired path runs the real production pipeline and produces forecasts on held-out resolved questions.
17. **Empirically validated?** **Provisionally YES on the untouched locked test** — the repaired arm cleared all pre-registered gates (paired Brier and log-loss CIs both exclude 0 favorably vs Phase-2). BUT this is **underpowered** (n=34 < 75) and the arm ordering is **unstable across datasets** (Phase-3 was worst on the committed set). Not a robust validation — promising, not conclusive.
18. **Production eligible?** **No** — gates pass but power is inadequate (n<75) and cross-set variance is high; and the UNREPAIRED arm scored lower Brier here (0.2142 vs repaired 0.2430), so the repair's edge is calibration/safety, not point accuracy. Needs a ≥75 powered test before deployment.
19. **Phase-2 or repaired Phase-3 as default?** The pre-registered rule selects **phase3b_repaired** as the CANDIDATE default (it beat Phase-2 on the locked test, best calibration, no domain regressed, downside-capped). Recommendation: adopt the repaired (gated/blended) arm as the safer candidate **pending an adequately-powered test**; keep Phase-2 as the conservative fallback until then.
20. **Interfaces later phases should consume:** the **selected/blended** forecast (`phase3b_repair.combine`), NOT the raw posterior terminal; treat the generic outcome-rate posterior as a calibrated, gated, subordinate signal with a Phase-2 fallback; do not let any posterior override a validated forecast without held-out evidence it helps.
