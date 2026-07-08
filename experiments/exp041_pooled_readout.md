# EXP-041 — The estimator that makes "map more variables" monotone (Part 2 of the north-star build)

EXP-040 found grounded simulation beats the composite, *but* that naive Bayes double-counts correlated
variables (party ≈ ideology) and overfits thin per-question data — "map more variables" only helped after
hand-tuning shrinkage. This builds and validates the fix as a **reusable component**
(`swm/variables/pooled_readout.py`): a correlation-aware, partially-pooled readout.

## The two fixes
1. **Correlation-aware** — a logistic over the full one-hot variable vector shares credit among collinear
   dummies (ridge distributes weight) instead of NB's independence assumption multiplying them.
2. **Partial pooling** — each question's per-person model is blended with its marginal by an n-adaptive
   weight `w_q = n_q/(n_q+τ)`; data-rich questions trust their fit, data-poor ones shrink to their base
   rate. `τ` is one global hyperparameter tuned by empirical Bayes on a train-internal hold-out.

## Setup (no-cheat)
OpinionQA, respondents split train/test; three estimators over the EXP-040 variable-richness ladder;
`τ` tuned only on train. 6,991 held-out answers.

## Result — individual log-loss (accuracy) by variable richness

| variables | NB | Logistic (no pooling) | **PooledLogistic** |
|---|---|---|---|
| marginal (0) | 0.6137 (.658) | 0.6137 (.658) | 0.6137 (.658) |
| + party (1) | 0.6062 (.675) | 0.6288 (.671) | **0.6014 (.675)** |
| + party, ideology, religion (3) | 0.6363 (.668) | 0.7077 (.654) | **0.6006 (.680)** |
| all 11 | 0.7354 (.646) | 0.8603 (.648) | **0.6060 (.674)** |

**Pooling is what makes grounding robust.** As variables are added, **NB and the un-pooled logistic both
overfit-collapse** (0.61 → 0.74 and → 0.86 — the un-pooled logistic is *worse* than NB, because a
per-question logistic over ~30 dummies on ~38 respondents is pure variance). **The pooled readout does
not collapse** — it stays 0.61 → 0.606, best at 3 variables (0.6006). Correlation-awareness alone is not
enough; correlation-awareness **plus partial pooling** is.

**The decisive win is on data-poor questions** (100 questions with train n < 25, the regime that
dominates real-world forecasting):

| | independent logistic | **pooled logistic** |
|---|---|---|
| data-poor log-loss | 0.8507 | **0.6042** |

A **29% log-loss reduction** — partial pooling rescues exactly the thin-data questions where a from-scratch
per-question fit is hopeless, by borrowing strength from the population prior.

## Honest findings
1. **Partial pooling, not correlation-awareness, is the load-bearing fix.** The un-pooled logistic is the
   *worst* estimator when variables are many and data is thin. What saves grounding is shrinking toward
   the prior in proportion to how little data a question has.
2. **The pooled readout matches — does not beat — a fully hand-tuned NB on the aggregate** (best pooled
   0.6006 vs EXP-040's hand-tuned-α NB 0.5854). Its advantage is that it **self-tunes** (one global τ, no
   per-variable hand-tuning) and is **robust** (never collapses), and it **dominates on data-poor
   questions** — the honest, deployable estimator, not a new accuracy ceiling on data-rich questions.
3. This is the estimator the north star needs: it turns "add more grounded variables" from a liability
   into a safe operation, which is the precondition for scaling variable grounding at all.

## Honest limits
- Pooling is toward the question **marginal** (polarity-free), not a full hierarchical cross-question
  coefficient model — OpinionQA's per-question answer polarity blocks naive coefficient pooling. A
  latent-trait (IRT-style) pooling of the *person* across questions is the deeper version.
- Binary questions only (the cache is all binary); multi-class needs the one-vs-rest extension.

## Reproduce
`python -m experiments.exp041_pooled_readout` → `experiments/results/exp041_pooled_readout.json`.
`python -m pytest tests/test_pooled_readout.py`.
