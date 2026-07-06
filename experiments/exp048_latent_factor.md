# EXP-048 — Modelling the correlation structure beats shrinking it (the estimation frontier)

The binding constraint the whole project keeps hitting: correlated variables (party ≈ ideology ≈
religiosity) get **double-counted**, so naive estimators are confidently wrong. EXP-041's pooled logistic
*shrinks* the collinear dummies — it damps the damage but can't tell "these two are the same latent axis,
count it once" from "both are independently weak." This tests the first-principles alternative:
**decompose** the correlated variables into a few orthogonal **latent value factors** and estimate the
outcome on those, so redundant variables collapse into one axis counted once. Double-counting becomes
impossible by construction.

## The build
`swm/variables/latent_factor_readout.py` — PCA (power iteration + deflation, pure Python) on the centered
one-hot demographic covariance yields K orthogonal factors; a small logistic per question estimates the
outcome on the K factor scores (+ the EXP-041 n-adaptive pooling toward the marginal). K is chosen on a
**train-internal** hold-out (no test leakage).

## Result (OpinionQA individual prediction, all 11 variables, no-cheat)

| estimator | log-loss ↓ | accuracy ↑ |
|---|---|---|
| NB (independence — double-counts) | 0.7397 | 0.641 |
| PooledLogistic (shrinkage, EXP-041) | 0.6104 | 0.669 |
| **LatentFactor (decorrelate, K=3)** | **0.5957** | **0.682** |

**Modelling the correlation structure beats shrinking it** — a 2.4% log-loss cut and +1.3 points accuracy
over the pooled logistic, and it wins on **data-poor questions** too (0.5719 vs 0.5801, n<25). K was
selected leakage-free at **3**: the 11 demographics really carry only ~3 latent axes, and estimating 3
orthogonal effects generalizes better than shrinking ~40 collinear dummies.

**The factors are the value axes the thesis is about** (top loadings):
- **Factor 1** — gender × partisanship: male/married/republican/conservative vs female/democrat (a
  traditional–conservative axis);
- **Factor 2** — religiosity: female/protestant/attends vs secular;
- **Factor 3** — socioeconomic-family: married/high-income/post-graduate.

So the readout literally maps each person onto a **latent value profile** and estimates the outcome from
it — the grounded "map the person's real latent variables and simulate" made concrete, with an
interpretable, auditable value decomposition instead of 40 opaque dummy weights.

## Why this matters for the bottleneck
This is a *structural* fix, not a bigger hammer:
- **No double-counting by construction** — orthogonal factors can't share credit; the redundancy in
  party/ideology/religiosity is resolved into one axis, not damped twice.
- **Data-efficient** — 3 effective parameters instead of ~40, so it estimates better exactly where data
  is thin (the realistic regime), which shrinkage can only partially rescue.
- **Self-tuning + interpretable** — K is picked from the data (no per-variable hand-tuning), and the axes
  are readable value dimensions, which is what a decision-support system needs.

It confirms the first-principles claim the project has been circling: **the frontier is estimation quality
— model the structure, don't just regularize it.** LatentFactor is now the best self-tuning estimator in
the stack.

## Honest limits
- The win over shrinkage is real but modest (2.4% log-loss); on OpinionQA the demographic signal is weak
  overall (everything sits near the marginal), so the ceiling is low. The gap should widen with richer,
  more-correlated variable sets (the person-side VariableMap's ~50 slots) where double-counting is worse.
- PCA is linear; genuinely non-linear value interactions would need a non-linear factor model.
- Factors are estimated on demographics only here; the deeper version factorizes the *full* VariableMap
  (grounded + inferred together) and weights each input by reliability — the next estimation build.

## Reproduce
`python -m experiments.exp048_latent_factor` → `experiments/results/exp048_latent_factor.json`.
`python -m pytest tests/test_latent_factor_readout.py`.
