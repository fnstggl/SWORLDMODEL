# WMv2 Phase 12 — Validation

*Every number here is copied from a committed artifact under `experiments/results/phase12/`. This run is
**PROVISIONAL**: Phase 11 (dynamic recompilation) is absent from the base branch, the corpus (148 rows) is far
below the 1,000 target, and the full-force single path is not yet integrated. Negative and null results are
preserved, not hidden.*

## Corpus and splits (Parts B/C/D)

- **148** real maximum-capacity posterior forecasts (93 phase3acc + 34 phase3b + 23 diagnostic), **9 domains**,
  base rate 0.73 YES. Each is a genuine `simulate_with_posterior` terminal with provenance + active-component
  manifest. `maximum_capacity_available=False` (Phase 11 absent; Phases 8/9 not on the path).
- Immutable event-family splits (manifest hash `1b0e1aa1740d0fdc`): **calibration 75 / validation 32 / test 41**.
  Fit on calibration only; select on validation only; **test scored once**. No test outcome used in any fit.

## Two questions answered separately (mandate requirement)

1. **Are the probabilities statistically reliable?** Reasonably — raw test ECE **0.178**; no calibrator beat the
   raw probabilities on validation, so **identity was selected** (correct, honest negative).
2. **Is the simulator adding predictive signal vs non-simulation baselines?** **No.** On the test split the full
   V2 forecast does **not** beat the base rate, the grounded direct LLM, or the ensemble; the ensemble is best,
   and V2's **discrimination is materially weaker** (AUROC 0.58 vs 0.84). Calibration cannot fix this —
   calibration ≠ discrimination.

## Per-arm metrics — test split (n=41)

| arm | Brier ↓ | log-loss ↓ | AUROC ↑ | dir ↑ |
|---|---|---|---|---|
| base rate (domain) | 0.2176 | 1.4445 | 0.599 | 0.707 |
| prior / no-evidence (0.5) | 0.2500 | 0.6931 | 0.500 | 0.293 |
| Phase-2 evidence-only | 0.2512 | 0.6960 | 0.546 | 0.488 |
| **full V2 raw** | 0.2407 | 0.6706 | 0.583 | 0.512 |
| **full V2 calibrated** (identity) | 0.2407 | 0.6706 | 0.583 | 0.512 |
| grounded direct LLM | 0.2191 | 0.6275 | 0.835 | 0.659 |
| grounded direct ensemble | **0.2101** | **0.6032** | **0.845** | 0.659 |

Paired bootstrap (full V2 raw − baseline; positive ⇒ V2 worse): vs base rate **+0.023** CI [−0.077, 0.119];
vs direct LLM **+0.022** CI [−0.072, 0.104]; vs ensemble **+0.031** CI [−0.060, 0.110]; vs Phase-2 evidence-only
**−0.010** CI [−0.050, 0.031]. The CIs span 0 at n=41, but the point estimates and the large AUROC gap
consistently favour the direct baselines. **Preserved negative: the simulator does not beat grounded LLM
forecasting on this corpus.**

## Calibration (Parts E/F/G/H)

- **Selected calibrator: `identity`.** Platt, beta, and isotonic were fit on the calibration split and scored
  on validation; none beat identity on BOTH proper scores, so the identity mapping was kept (the mandated
  behaviour when no method reliably improves). Test raw = test calibrated (Brier 0.241, ECE 0.178).
- Conditioned/hierarchical families (domain / horizon / support / domain×support) were compared on test; none
  improved over the global/identity result. Calibration uncertainty (bootstrap) is returned per test row.
- **Gate G4: PASS** (identity correctly selected; calibrated not worse on either proper score).

## Support grading (Part I)

Fitted monotonic reliability model (expected error from pre-outcome features), grades assigned without the
outcome. Test Brier by grade:

| grade | Brier |
|---|---|
| empirically_supported | 0.197 |
| transfer_supported | 0.234 |
| exploratory | 0.186 |
| highly_speculative | **0.331** |

**`highly_speculative` clearly separates as the worst**, but the fine 4-level ordering is **NOT strictly
monotonic** on the 41-row test (exploratory beats supported). **Gate G5: FAIL (preserved).** Support grading is
**not** claimed empirically validated at 4-level resolution; the extremes separate. No test peeking was used to
force the ordering.

## Uncertainty decomposition + sensitivity (Parts J/K)

- Quantitative decomposition (posterior law-of-total-variance + leave-one-evidence-group-out) for **100%** of
  rich-trace rows; **synthetic recovery PASSES** (correctly attributes an injected dominant evidence source,
  −0.11 mean shift). On real data the epistemic variance is dominated by broad posterior spread with little
  attributable to individual evidence groups or structure — an honest reflection of weak evidence
  informativeness. **Gates G6, G7: PASS.**

## Critic (Part L)

The critic flags V2-vs-direct/ensemble disagreement and **never overwrites** the simulation number. On test,
**warned rows have higher error** (mean Brier 0.261 warned vs 0.230 unwarned, **error-lift +0.031**, 14 warned /
27 unwarned) — the warnings genuinely predict elevated error. Example (forensic traces): `indonesia_2024`, V2
raw 0.33 (outcome YES) while the direct ensemble said 0.82 — the critic flags the 0.52 disagreement; V2 was
confidently wrong there. **Gate G8: PASS.**

## Acceptance-gate scorecard (Part Q — graded honestly, thresholds not lowered)

| Gate | Status |
|---|---|
| G1 full-force integration | **PARTIAL** — max-capacity posterior path runs through one pipeline, but the facade default is the simpler `pipeline.simulate` and Phases 8/9/11 are not on the forecast path |
| G2 no forecast abstention | **PASS** — 0% abstention; all completed forecasts scored; grades never suppress a probability |
| G3 data governance | **PASS** — disjoint event-family splits; 0 test-outcome leakage; manifest hashed |
| G4 calibration | **PASS** — identity correctly selected; not worse on either proper score |
| G5 support grading | **FAIL (preserved)** — extremes separate; fine 4-level ordering not monotonic on n=41 |
| G6 uncertainty decomposition | **PASS** — 100% coverage; synthetic recovery correct; no prose percentages |
| G7 sensitivity | **PASS** — leave-one-evidence-group-out, not LLM opinion |
| G8 critic | **PASS** — cannot overwrite; warnings predict +0.031 higher Brier |
| G9 predictive comparison | **REPORTED (negative)** — full V2 does NOT beat base rate / direct LLM / ensemble |
| G10 causal integration | **PARTIAL** — in-path components (evidence, posterior) show terminal effect; out-of-path phases marked not-wired |
| G11 scale | **PARTIAL** — 148 real forecasts / 9 domains (< 1,000 / 10 target); resumable pipeline provided |
| G12 reproducibility | **PASS** — artifacts hashed; deterministic offline recomputation; corpus manifest recorded |

## Causal ablations (Part N)

On the forecast path: evidence-conditioning (Phase-2 vs prior) and posterior consumption (V2 vs Phase-2) both
change the terminal. **Phases 8 (persistence), 9 (populations/networks), nonlinear (7), and 11 (dynamic
recompilation) are NOT on the question→forecast path** — ablating them would be a no-op, so their removal effect
is recorded as **"not applicable — not wired"**, NOT "verified zero effect". This is the honest state; wiring
them into one path is a documented dependency.

## Four separate statuses

1. **Software implemented — YES.** Calibrator registry+selection, fitted support-grade model, uncertainty
   decomposition + sensitivity, critic, serving wiring, monitoring/compat gate, resumable refit — all real code
   with 12 passing tests.
2. **Executes end-to-end — YES.** On the real max-capacity posterior path, the calibrated result contract is
   populated (no longer ornamental).
3. **Empirically validated — PARTIAL.** Validated: identity-calibration selection (negative), critic
   warning→error-lift (positive), uncertainty-decomposition synthetic recovery, data governance. Not validated:
   4-level support-grade monotonicity (extremes only), and **predictive superiority over baselines is NOT
   achieved (negative)**.
4. **Production eligible — NO.** Provisional (Phase 11 absent), underpowered (148 < 1,000), not full-force
   (Phases 8/9/11 off path), and full V2 does not beat grounded LLM baselines.

## Answers to the 24 required questions

1. Complete max-capacity simulator executes by default? **No** — the facade default is `pipeline.simulate`
   (no posterior); the max-capacity posterior path is a separate entry.
2. All causally-relevant completed subsystems wired in? **No** — Phase 4/registry/institutions_v2 yes; Phases
   8/9/nonlinear/11 no.
3. Calibrators improve held-out reliability? **No net improvement** — identity selected (raw already ~calibrated).
4. Preserve/improve proper scoring? **Yes** (identity ⇒ unchanged; no worsening).
5. Support grading separates reliable from speculative? **Partially** — extremes yes, 4-level no.
6. Uncertainty quantitatively decomposed? **Yes** (validated on synthetic).
7. Sensitivities measured not narrated? **Yes** (leave-one-evidence-group-out).
8. Critic identifies genuine failure modes? **Yes** (warnings predict +0.031 higher Brier).
9. Critic unable to overwrite the simulation? **Yes**.
10. Calibrated full V2 beats grounded direct LLM? **No** (0.241 vs 0.219 Brier; AUROC 0.58 vs 0.84).
11. Beats the direct ensemble? **No** (ensemble best at 0.210).
12. Beats the observer panel? **Not run separately** (ensemble is the strongest baseline evaluated).
13. Beats analogical forecasting? **Not run** (documented dependency).
14. Beats crowds where present? **No genuine as-of market data** in this corpus (documented).
15. Beats specialized ceilings? **Not applicable** (no specialized ceiling on this general corpus).
16. Which subsystems add held-out value? **Evidence conditioning + posterior** add a small edge over the prior;
    none add value over the direct LLM.
17. Which are inactive/ornamental/harmful? **Calibration is inactive on this data (identity)**; Phases
    8/9/nonlinear/11 are not on the path (inactive by wiring).
18. Speculative forecasts still scored? **Yes** (0% abstention).
19. Forecast abstention zero? **Yes**.
20. Phase 12 software complete? **Yes** (code + tests).
21. Phase 12 empirically validated? **Partially** (see status 3).
22. Phase 12 production eligible? **No** (see status 4).
23. What remains before Phase 13? Integrate Phase 11 + full-force single path; refit; reach scale; achieve
    predictive parity with the direct baselines or clearly document why the simulator is retained despite it.
24. What remains before the Phase 15 historical benchmark? A distinct, larger, leakage-safe locked benchmark;
    post-Phase-11 max-capacity corpus; genuine as-of market/crowd baselines.

## Continuation manifest (resumable)

- **Phase 11 refit:** once Phase 11 lands and the corpus is regenerated from the post-Phase-11 max-capacity
  path (set `maximum_capacity_available=True`), run `PYTHONPATH=. python experiments/phase12_refit.py --regen`.
  `phase12_serve.compatible_with(bundle, phase11_present=True)` already refuses the provisional bundle.
- **Scale:** extend `experiments/phase12_baselines.py` / the capture pipeline to reach ≥1,000 rows across ≥10
  domains; the corpus builder and splits are resumable and re-hash automatically.
- **Full-force:** wire Phases 8/9/11 into one question→forecast path, then re-run ablations (they are currently
  no-ops).

## Reproduce

```
PYTHONPATH=. python experiments/phase12_corpus.py        # corpus + immutable splits (hash)
PYTHONPATH=. python experiments/phase12_calibrate.py     # fit/select/evaluate calibration
PYTHONPATH=. python experiments/phase12_grade.py         # fit + validate support grading
PYTHONPATH=. python experiments/phase12_uncertainty.py   # decomposition + sensitivity + synthetic recovery
PYTHONPATH=. python experiments/phase12_baselines.py     # direct-LLM + ensemble baselines (network)
PYTHONPATH=. python experiments/phase12_evaluate.py      # arms, ablations, comparisons, gates
PYTHONPATH=. python experiments/phase12_traces.py        # forensic traces
PYTHONPATH=. python experiments/phase12_refit.py         # resumable post-Phase-11 refit
python -m pytest tests/test_wmv2_phase12_calibration.py -q
```
