# EXP-028 — Per-individual opinion prediction on OpinionQA (the SOTA benchmark's own task)

The best individual-simulation work (Generative Agent Simulations of 1,000 People) predicts each
person's **individual survey answers**. This runs that same task on the standard public benchmark —
**OpinionQA / Pew American Trends Panel** — through our architecture, closing the gap EXP-022 flagged as
"blocked on data access."

## Data
`RiverDong/OpinionQA` (HF mirror of Santurkar et al. / Pew ATP): ~295k (respondent, question) rows, each
a respondent's demographics + a question + **the answer that person actually gave**. We subsample 6,000
respondents (23,340 answers, 368 questions, 15 waves); the parsed cache is committed under
`experiments/results/exp028_oqa/` for reproducibility.

## Method (the architecture, applied per person)
Map each respondent's demographics → the **10 latent value-variables** the world model already uses
(religiosity, traditionalism, individualism, … — the EXP-023 value dims; `swm/variables/demographic_values.py`),
then predict their answer to a question by **value-similarity to other people who answered it** (softmax
over cosine of value profiles). No-cheat: split **respondents** train/test; a test respondent is a person
the model never saw, and their own answers are never used — only their inferred value profile.

## Result (predict a held-out person's answer; 6,825 test (person, question) pairs; mean over 5 seeds)
| tier | accuracy | log loss |
|---|---|---|
| marginal — the population answer ("average American") | 0.657 | 0.617 |
| demographic KNN — similarity on raw one-hot demographics | 0.676 | 0.626 |
| **value-similarity — the 10 inferred value-variables** | **0.680** | **0.614** |

- **Individual value modeling beats the population marginal by +0.023 accuracy — on all 5 seeds** (and
  is on par or better on log loss). Knowing a person's inferred values predicts *their* answer better
  than predicting the average American.
- **The 10-dim value abstraction beats raw one-hot demographics on calibration** (log loss 0.614 vs
  0.626, every seed) while matching its accuracy — the interpretable value compression *dominates* the
  ~50-dim raw encoding where it matters (calibrated probabilities), not just ties it.

## Why this matters
- It runs the **SOTA benchmark's own task — individual survey-response prediction — on standard public
  data**, no-cheat, and shows the VariableMap value architecture beats both the population baseline and
  raw demographics. This is the per-individual complement to EXP-023's population-level result.
- The winning representation is the **same 10 value-variables** used for country-level opinion (EXP-023):
  one interpretable latent value profile serves both the aggregate and the individual regime.

## Honest limits
- **Absolute gains are modest** (+0.023 accuracy over a strong 0.66 marginal; log loss ≈ marginal). Pew
  opinion answers are high-entropy and the marginal is hard to beat — the *relative* consistency (5/5
  seeds) and the calibration win over raw demographics are the signal.
- **Values are mapped from demographics by a transparent heuristic**, not inferred from free text or an
  interview — so this tests the value *architecture*, not deep per-person inference (that is EXP-025).
  Richer per-respondent inference (their own text, or their other answers as context) is the next lift.
- **Not directly comparable to the 1,000-person ~85%**: their metric is normalized to human test–retest
  on GSS with full interviews; this is raw accuracy on Pew ATP from demographics. We match the *task*,
  not the setup — this is comparability on the public benchmark, not a claim of parity.

## Reproduce
Download the parquet (fetch note in `experiments/datasets_opinionqa.py`) or use the committed parsed
cache → `python -m experiments.exp028_individual_opinion`. `python -m pytest tests/test_demographic_values.py`
covers the demographic→value mapping.
