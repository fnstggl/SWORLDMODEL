# EXP-072 — does modeling MORE pressuring variables help or hurt? The thesis, tested no-cheat.

The disagreement this settles: is "model every relevant variable" right, or does adding variables hurt? The
honest answer depends entirely on **how each variable's weight is set**. A weight is a *causal elasticity*
(`∂outcome/∂variable`); you never know it exactly, so the question is what you do with that ignorance. This
tests three weighting schemes on real data as variables are piled on.

## The machinery (implements the weighting answer)

- **`swm/variables/bayes_logistic.py`** — a logistic with a **Laplace posterior over the weights**: every
  variable's weight comes with a posterior SD (`weight_sd` = "how sure are we of this weight"), coefficients
  are shrunk toward a **prior mean** (the LLM/literature elasticity), and `predict_dist` **integrates over
  weight uncertainty** so an unknown weight *widens* the prediction instead of biasing it.
  `variance_contribution` is the triage: a variable's share of outcome variance ≈ `weight² · Var(feature)`,
  so only the few high-leverage weights need precise calibration.
- **`swm/eval/fidelity_ladder.py`** — add variables one at a time, score held-out accuracy at each rung, under
  NAIVE / CALIBRATED / (properly-tuned) shrinkage.

## Result A — high-dimensional real data (OpinionQA: demographics → binary opinion, 19 data-rich questions)

Add 12 demographic attributes (each a one-hot block) one at a time; averaged over questions, held-out:

| # variables | naive (free weights) | calibrated shrinkage | **properly-tuned shrinkage** |
|---|---|---|---|
| 1 | 0.580 | 0.578 | 0.588 |
| 6 | 0.596 | 0.574 | **0.564** |
| 12 | **0.656** | 0.609 | **0.573** |
| **harm from adding all 12 (full − best log-loss)** | **+0.111** | +0.066 | **+0.015** |

- **NAIVE reproduces "more variables hurt"** (+0.111 log-loss — it overfits the correlated dummies).
- **Properly-tuned shrinkage makes adding variables essentially harmless** (+0.015), and **up to 6 variables
  it actively helps** (0.588 → 0.564). More pressuring variables improve then plateau — they only hurt when
  the weights are un-calibrated.

## Result B — low-signal domain (ChangeMyView persuasion) is variance-triage in action

On CMV the real variables themselves are weakly identified (signal-to-noise |w|/sd ≈ 0.38, barely above the
noise variables' 0.39), so neither more real variables nor injected noise moves the held-out log-loss
(~0.63, near the base rate). That is the **triage principle** measured: when the outcome is mostly
irreducible, variable *count* is moot — the ceiling is information, not fidelity. And the weight report shows
the model *knows* it hasn't pinned these weights down.

## The reconciliation

"Less is more" was an artifact of adding variables with **un-calibrated point weights** — noise compounds. Add
each variable with a **calibrated weight + shrinkage + integrated uncertainty**, and a useless variable
auto-shrinks to ~zero weight (harmless) while a useful one earns its weight. **The binding constraint is
weight calibration, not variable count** — so "model every pressuring variable" is right, *provided each
carries a calibrated weight with uncertainty.* The next lever (visible here: strong-shrink starts worse at
k=1 by over-shrinking the one strong variable, then wins at k=12) is **empirical-Bayes / n-adaptive
shrinkage** — tune the prior precision per data size, so the model neither under- nor over-trusts each weight.

Run: `python -m experiments.exp072_fidelity_ladder`. Tests: `test_bayes_logistic.py`, `test_fidelity_ladder.py`.
