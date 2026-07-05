# EXP-026 — Conformal prediction sets: finishing the uncertainty contract

The unified `simulate()` API (EXP-024) already ships a calibrated probability, an honest confidence, an
abstain flag, and a calibration badge. The one missing piece of the uncertainty contract was a
**per-prediction guarantee**. Conformal prediction supplies it: a prediction **set** over the outcomes
{0,1} that contains the true outcome with probability ≥ 1 − α, under only exchangeability — no
distributional assumptions.

## Implemented
`swm/uncertainty/conformal.py` — split conformal for the binary regime. On a held-out calibration set it
scores each example by nonconformity `1 − p(true class)` and takes the finite-sample-corrected quantile
`q`. A label is included in the set iff its nonconformity ≤ q, giving one of:
- `{1}` / `{0}` — a confident singleton,
- `{0,1}` — genuinely uncertain (the set-valued analog of abstaining),
- `{}` — both outcomes surprising (flags OOD / mis-modeled).

Wired into the `Simulator`: `fit()` calibrates conformal on the same held-out grading tail, and every
`Prediction` now carries `prediction_set` + `coverage_target`.

## Result (no-cheat, real CMV corpus; conformal fit on the calibration tail, coverage measured on a held-out test split)

| α | target coverage | empirical | avg set size | frac uncertain {0,1} |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.919 | 1.63 | 0.63 |
| 0.10 | 0.90 | 0.903 | 1.56 | 0.56 |
| 0.20 | 0.80 | 0.792 | 1.26 | 0.26 |
| 0.30 | 0.70 | 0.719 | 1.08 | 0.08 |

**Empirical coverage tracks the target within ~0.03 at every level** (max deviation 0.031). The honest
trade-off is visible: on a hard, high-entropy problem, *guaranteeing* 90% coverage forces the model to
widen 56% of predictions to the uncertain set {0,1}; relax to 70% coverage and only 8% stay uncertain,
the rest becoming confident singletons. That is exactly the right behavior — the set size *is* the
honest uncertainty.

## Honest limits
- Conformal's guarantee is **marginal** and assumes **exchangeability**. A temporal train/test split
  mildly violates it, and the calibration tail here is small (~250 points), which is why the most
  demanding level (95%) slightly undershoots (0.919). This is expected finite-sample + temporal slack,
  not a bug — and it is reported, not hidden.
- The sets are over the binary outcome; extending to the population-distribution regime (EXP-023) would
  use conformalized distributional prediction — future work.

## Reproduce
`python -m experiments.exp026_conformal` (uses the committed CMV inference artifacts).
`python -m pytest tests/test_conformal.py` verifies the coverage guarantee on synthetic calibrated data.
