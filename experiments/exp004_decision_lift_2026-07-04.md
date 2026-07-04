# EXP-004 — Decision lift, not just calibration

The question a buyer actually asks. EXP-002/003 proved the predictions are *calibrated* and beat
baselines on log loss. But calibration is not what anyone pays for. The buyer asks: **"if I act on
this model instead of my current method, do I capture more of the outcome?"** This measures that,
on the same contamination-free HN predictions (rounds 3-4, n=228), framed as the real operator
decision under limited attention.

## Setup (no new data, no cheating)
Operator can act on only the top-K% of candidates (review / feature / send / spend). Rank by each
method; measure what fraction of the actual **winners** (score >= 40 — the tail where EXP-003
showed the edge is real) you capture. Compare: model vs random vs the strong **author-aware
segment baseline** vs a perfect **oracle**.

## Result

| act on top | model (claude) | author-aware baseline | random | oracle |
|---|---|---|---|---|
| 5%  | 16.7% | 11.1% | 4.9% | 61.1% |
| 10% | 27.8% | 16.7% | 9.9% | 100% |
| **20%** | **50.0%** | 33.3% | 19.6% | 100% |
| 30% | 55.6% | 38.9% | 29.4% | 100% |

(hit-capture = fraction of the 18 actual >=40 winners caught in the top-K)

**Headline @ top-20%: the model captures 50% of the winners vs 20% random (2.5x) vs 33% for the
author-aware baseline.** Lift over random +30pp; lift over the strong baseline +17pp.

## Honest significance
- **Over random: large and unambiguous** (2.5x at top-20%, consistent across all K).
- **Over the strong author-aware baseline: directionally positive but NOT yet significant** —
  bootstrap 95% CI on the lift is [-12%, +45%], P(no lift) = 0.18. With only 18 hits at n=228 the
  tail is too data-starved to nail the lift-over-baseline precisely. Same constraint EXP-003 hit.
- On *engagement* capture (total score, not winner count) the baseline ties the model (~40-45% at
  top-20-30%), because 1-2 mega-hits dominate the sum — the heavy tail makes total-score capture
  high-variance. **Winner-count capture (hit-capture) is the stabler, more honest decision metric.**

## What this establishes
The product's value proposition is now *measured*, not just asserted: **ranking by the model finds
the majority of winners in a fraction of the review effort.** That is the sentence a growth /
content / RevOps team buys. The same math is the email wedge — "here are N contacts/variants, act
on the top-K most likely to respond" — where the higher reply base rate makes the lift easier to
bank and to prove.

## What it does not establish, and the resulting next step
Lift over a *strong* baseline needs more tail events to reach significance. Every experiment now
converges on the same binding constraint: **outcome volume in the tail.** The unlock is not more
title-reasoning — it is running `auto_loop.py` on a schedule to accrue a live, compounding
scorecard, and/or pointing the loop at a wedge where the operator's own outcomes stream in
(email/text). Decision lift over random is bankable today; decision lift over the best baseline is
a data-volume problem with a known mechanism.
