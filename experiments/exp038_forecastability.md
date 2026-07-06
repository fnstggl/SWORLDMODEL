# EXP-038 — Forecastability / triage score (Tetlock #1: say what you can forecast)

A general forecaster must know *where effort pays* — skip the trivial and the hopeless, concentrate on
the Goldilocks zone. This learns a score in [0,1] estimating how reliably a question's DIRECTION can be
called, from as-of features only (lean magnitude, recent volatility, days-to-resolution, a news
result-cue, and optional driver agreement/strength), fit no-cheat on TRAIN question outcomes and
evaluated on held-out TEST.

## Setup (no-cheat)
The score is `P(the lean call is correct | as-of features)`, a `LogisticReadout` fit on TRAIN questions
whose direction resolved. The target is honest resolution-call correctness — did the belief resolve on
its lean side — over **all** questions, *not* conditioned on movement (conditioning on movement biases
toward surprises for confident beliefs and breaks selective forecasting; see the fix below). Evaluated on
215 held-out Kalshi questions.

## Result (215 held-out Kalshi questions)
**The score genuinely separates forecastable from not.** Sorting held-out questions into score quartiles:

| Quartile | mean score | resolution accuracy |
|---|---|---|
| Q1 (least forecastable) | 0.797 | **0.849** |
| Q2 | 0.931 | 0.981 |
| Q3 | 0.967 | **1.000** |
| Q4 | 0.988 | 0.964 |

Least-vs-most-forecastable gap **+0.151** (Q1 0.849 → Q3 1.00). Selective forecasting is monotone where
it matters — dropping the bottom quartile lifts directional accuracy from 0.949 (full coverage) to 0.981
(top 75%/50%). Triage sends 213/215 to FORECAST at 0.953 accuracy and correctly withholds the near-0.5
uncertain ones to HEDGE.

## The honest findings
1. **The signal is dominated by the lean magnitude.** Confident beliefs (|p−0.5| large) resolve toward
   their side; near-0.5 questions are genuinely coin-flips and the score correctly rates them low (~0.80)
   with correspondingly lower accuracy (~0.85). This is EXP-036's lean-direction result turned into an
   explicit, learned *self-assessment* — the system now says which calls it trusts.
2. **Q4 dips below Q3 (0.964 vs 1.00).** The very-highest-score bucket is not the most accurate — a mild
   non-monotonicity from the score over-crediting extreme leans that occasionally reverse near
   resolution. The score is a reliable *ordering* (Q1≪Q2≈Q3≈Q4), not a perfectly monotone accuracy map.
3. **On this efficient-market data almost everything is forecastable** (base directional accuracy 0.95),
   so the abstain bucket is nearly empty. The score's value shows on the *inefficient / near-0.5 tail* —
   exactly where a general model needs to hold back. Its real test is questions with no liquid market
   (where directional accuracy is far from 0.95); this validates the mechanism on the data we have.

## What it means for the architecture
Triage (Tetlock's first commandment) is now a concrete, learned layer: `ForecastabilityScorer.score()`
gives a calibrated "can we call this?" number and `.triage()` returns FORECAST / HEDGE / ABSTAIN. It
composes with the `simulate()` abstention layer (EXP-024) — that one declines out-of-envelope *inputs*;
this one declines low-signal *questions*. Together the system can say what it can and can't usefully
forecast, and spend effort where it pays.

## Honest limits
- Validated on market questions, where nearly everything is directionally forecastable; the score's
  discrimination will matter more on no-market questions with a wider forecastability spread.
- The learned weights lean heavily on |lean|; the driver-agreement/strength features are available but
  were flat on this data (single-view driver inference). Multi-view drivers may add discrimination.

## Reproduce
`python -m experiments.exp038_forecastability` → `experiments/results/exp038_forecastability.json`.
`python -m pytest tests/test_forecastability.py`.
