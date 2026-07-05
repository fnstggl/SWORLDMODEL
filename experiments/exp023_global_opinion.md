# EXP-023 — Does inferred variable-mapping predict POPULATION OPINION? (the aggregate on-thesis test, on a standard benchmark)

The persuasion test (EXP-021) showed the VariableMap thesis pays off for a hard *individual* outcome.
This is its mirror at **population scale**, on a **standard benchmark** the best social-simulation work
is measured against: **GlobalOpinionQA** (Durmus et al., 2023 — World Values Survey + Pew Global
Attitudes, response distributions per country). The question: can we predict a *country's* distribution
over answers to a survey question from **LLM-inferred country value-variables alone** — never using that
country's own survey responses?

This is the aggregate analog of the core thesis. A country is an ENTITY; its value-variables
(religiosity, traditionalism, individualism, trust-in-institutions, openness-to-change, national pride,
economic-left, social-progressivism, hierarchy-respect, survival↔self-expression) are **inferred from
world knowledge, not from this survey** (so non-circular); the question is the ACTION that activates
some of those values.

## Setup (no-cheat / no-leakage)
- **1,591** GlobalOpinionQA questions with ≥3 country distributions; **60** countries given inferred
  value-profiles by a value-inference agent (world-knowledge only, blind to the survey).
- **Split the *countries* train/test** (70/30). For each **unseen test country** and each question,
  predict its answer distribution using **only the train countries' distributions** on that question,
  weighted by value-similarity (softmax over cosine similarity of the 10-dim value profiles). *The test
  country's own opinions are never used* — the only thing linking it to the prediction is its inferred
  value profile.
- **Hyperparameters (softmax temperature β, hybrid shrink w) tuned on a train/val split of TRAIN**,
  never on test.
- Tiers: **base** = unweighted global mean of train countries; **values** = value-similarity weighted;
  **hybrid** = shrink values toward the global mean.
- Metric: cross-entropy CE(actual‖pred) and total-variation distance, averaged over ~11k test
  (country, question) pairs. Lower is better.

## Result (test = unseen countries; 11,170 pairs; seed 0)
| tier | cross-entropy | total-variation |
|---|---|---|
| base (cross-country global mean) | 1.2070 | 0.1787 |
| **values (inferred value-similarity)** | **1.1886** | **0.1615** |
| hybrid | 1.1886 | 0.1615 |
| *oracle floor (country's own entropy)* | *1.0604* | *0* |

**Inferring a country's value-variables and predicting its opinion distribution from *similar*
countries beats the cross-country global mean by +0.018 cross-entropy and cuts total-variation
0.179 → 0.162 (~10% relative) — closing ~13% of the entire global-mean→oracle gap, using only the
inferred value profile and never the country's own survey answers.** The tuner picked β≈4, w≈1.0
(values-only), i.e. the value model needed no shrink toward the mean to win.

### Robustness (6 seeds, each re-tuned no-leakage on its own train/val split)
| | mean | min | max | sign |
|---|---|---|---|---|
| CE gain vs global mean | **+0.0254** | +0.0184 | +0.0347 | **6/6 positive** |
| TV gain vs global mean | **+0.0234** | +0.0172 | +0.0330 | **6/6 positive** |

The effect is positive on every split — a real cross-country signal in the inferred values, not a
lucky partition.

## Why this matters
- It extends the core thesis from the individual regime (persuasion, EXP-021) to the **aggregate /
  population regime on a standard, externally-defined benchmark** — the same family OpinionQA and the
  1,000-person work report on. The mechanism is identical: **infer the latent variables acting on the
  entity, then predict from them**, with the inference blind to the outcome.
- It is genuinely no-cheat at the *entity* level: an unseen country is predicted purely from its
  inferred value profile plus other countries' data. This is the population version of "predict a
  person you've never surveyed from their inferred variables."
- It closes EXP-022's most important open item — a standard survey benchmark — that was previously
  blocked on data access.

## Honest limits
- **Modest absolute size.** +0.018 CE / ~10% TV is a real but not dramatic gain; the global mean is
  already strong because most countries answer many questions similarly, and GlobalOpinionQA's
  distributions are high-entropy (oracle floor 1.06 vs base 1.21 — only 0.15 nats are *reducible* at
  all). Within that reducible band the inferred values capture ~13%.
- **60 countries, value-profiles only.** We infer 10 value-variables per country, not per person; this
  is population-level, not the individual-level GSS prediction the 1,000-person work does. It shows the
  architecture transfers to the aggregate benchmark, not that we match per-person SOTA.
- **Value-similarity transfer, not a generative simulation.** This tier predicts by weighting similar
  entities; it is the calibrated-readout end of the stack, not the multi-actor trajectory engine.
  Pairing inferred values with the simulation engine is future work.
- Country names are matched exactly to GlobalOpinionQA; the survey CSV is data-access-gated (fetch note
  in `datasets_globalopinion.py`), so the committed `exp023_country_values.json` makes the model
  reproducible even without re-downloading the survey.

## Reproduce
Download the GlobalOpinionQA table to `data/global_opinions.csv` (HF; see
`experiments/datasets_globalopinion.py`), then `python -m experiments.exp023_global_opinion`. The
inferred country value-profiles are committed at `experiments/results/exp023_country_values.json`;
`python -m pytest tests/test_global_opinion.py` verifies the scoring primitives and (when the CSV is
present) that inferred values beat the global mean.
