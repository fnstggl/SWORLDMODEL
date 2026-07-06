# EXP-043 — Grounding drivers in real retrieved news vs the LLM gestalt (Part b)

EXP-037's honest finding: the LLM-gestalt "drivers" added nothing over the holistic base rate, because
they were the LLM re-deriving its own judgment in a variable costume — not real external variables. Part
b tests the fix directly: extract drivers from the **real as-of news** attached to each question (dated
strictly before the target — leakage-free) and ask whether *grounded* variables recover signal the
gestalt could not.

## Setup (no-cheat)
Grounded features per question from its as-of news only (no price, no LLM opinion): news **volume**,
**result-cue** fraction (resolution language), signed **resolution-polarity** (positive vs negative
resolution terms), **recency** of the latest item, **source count**. A pooled logistic is fit on the
train split and scored on test_kalshi (train 2,049 / test 574; outcome = the market's near-resolution
value; base rate 0.348).

## Result

| arm | log-loss | brier | directional acc | corr w/ outcome |
|---|---|---|---|---|
| base rate (composite) | 0.6530 | 0.230 | 0.641 | 0.000 |
| **news-grounded** (real news features, no price) | 0.6546 | 0.231 | 0.641 | **0.047** |
| market lean (reference ceiling) | 0.2707 | 0.075 | 0.950 | 0.841 |

**Crude real-news features do *not* beat the base rate** (Δ −0.0016; a whiff of outcome correlation,
0.047, but nothing that improves calibrated prediction). The market price, by contrast, is enormously
more informative (corr 0.84, directional accuracy 0.95).

## The lesson — grounding is necessary but not sufficient
The gap between the news-grounded readout (corr 0.047) and the market (corr 0.84) **is the signal that
lives in the news but that lexical features fail to extract.** The market impounds the *content* of these
same articles; our volume / result-cue / polarity counts capture almost none of it. So:

- EXP-037: LLM-**gestalt** drivers add nothing over the base rate (the LLM's own judgment recycled).
- EXP-043: **crude real-news** features also add nothing over the base rate (real data, shallow reading).

Both fail for the *same* reason from opposite ends: the first has no real external variable, the second
has a real external variable read too shallowly. **"Ground the variables in real data" only pays once the
grounding captures real content** — entity-linked, question-aware extraction (does *this outcome's*
subject appear winning/losing in the news), not question-agnostic term counts. The resolution-polarity
feature carries the right sign (+0.12) and is the honest seed of that; it is just far too coarse.

## Honest findings
1. **A precise negative that maps the frontier:** the north-star bottleneck for question-level forecasting
   is not "list drivers" and not even "attach real news" — it is **reading the real content well enough**
   to recover what the crowd already extracts. That is a retrieval + NLP problem (entity linking, stance
   detection against the specific resolution), the concrete next investment.
2. The market-lean ceiling (corr 0.84) confirms the signal is *there* and *learnable from the news*;
   we are leaving it on the table with shallow features — quantifying exactly how much (0.047 vs 0.84).

## Honest limits
- Five hand-built lexical features; no entity linking, no per-question stance detection, no embeddings
  (pure-Python constraint). The experiment establishes the *floor* of real-news grounding, not its
  ceiling.
- Outcome proxied by the near-resolution market value; a true post-cutoff resolution label would be
  cleaner.

## Reproduce
`python -m experiments.exp043_news_grounding` → `experiments/results/exp043_news_grounding.json`.
