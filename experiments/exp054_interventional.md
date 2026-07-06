# EXP-054 — The interventional KPI: can the model pick the causally-better intervention?

The simulation audit's sharpest finding: every headline KPI scores predictive *reconstruction* of a
marginal; none scores *intervention* — "what happens if I do X." This builds the one that does. The
Upworthy Research Archive is **randomized** headline A/B tests, so the observed CTR difference between arms
IS the causal effect of the headline — choosing one is a real `do(x)`, and its realized CTR is an unbiased
outcome (uniform randomization → off-policy value is exact). This is the first KPI in the project that
tests the thesis directly.

## Setup (no-cheat)
A headline→CTR model (pure-Python ridge on lexical features: length, question/number/caps, curiosity and
emotional words) trained on TRAIN experiments; on 1,476 held-out experiments it PICKS the arm it predicts
best. Scored on policy value / regret and CATE-sign, not reconstruction.

## Result

| policy | realized CTR |
|---|---|
| oracle (always the best arm) | 0.0207 |
| **model's pick** | **0.01679** |
| random / observed policy (mean arm) | 0.01638 |
| worst arm | 0.01251 |

- **Uplift over random: +0.041 pp** — the model's chosen headlines do beat a random pick, but barely.
- **Fraction of achievable uplift captured: 9.5%** (of the best−mean gap).
- **CATE-sign accuracy: 0.493** — pairwise causal ranking is **at chance** (0.5).

## The honest finding — and why it matters more than any prior result
**Our models are essentially unable to pick the causally-better intervention.** Simple lexical features
capture ~9.5% of the achievable headline uplift and rank arms within an experiment no better than a coin
flip. This is a humbling but *crucial* result, because it exposes exactly what the audit warned:

- The marginal KPIs looked excellent (EXP-050 population MAE 0.0045, EXP-047 market-consistency 0.57), yet
  on the **interventional** task — the one that defines a "what-would-actually-happen" engine — the same
  class of model is near-useless. **Reconstruction accuracy did not transfer to intervention skill.** A
  system can nail the marginal and still not know which lever to pull.
- Within an A/B test the arms are near-identical wordings of one story, so the causal CTR difference lives
  in subtle semantics our lexical features cannot see (the same ceiling EXP-044 hit) — the interventional
  version of "the frontier is semantic, not lexical." An LLM/semantic headline model (à la EXP-047) is the
  natural next attempt, now with a *causal* scoreboard to hold it to.

## What it changes
This stands up **KPI-A** as a permanent, honest scoreboard: **policy value / regret + CATE-sign on
randomized interventions.** It should headline results going forward instead of MAE-on-share (which the
audit showed is near-tautological marginal recovery). The number to beat is **9.5% of achievable uplift**
and **0.49 CATE-sign** — a low bar that makes real interventional progress legible.

## Honest limits
- Lexical features only (no semantics); the point is the *KPI*, not this model's skill.
- Contamination caveat: the archive is public since 2021; treated as a **mechanism** benchmark (does the
  model pick the causally-better arm), not a leakage-free skill number.
- Off-policy value is exact here only because assignment was uniform-random; on observational data it would
  need IPS/doubly-robust correction.

## Reproduce
`python -m experiments.exp054_interventional` → `experiments/results/exp054_interventional.json`
(committed parsed cache; raw CSV gitignored — see `experiments/datasets_upworthy.py`).
`python -m pytest tests/test_interventional.py`.
