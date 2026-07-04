# EXP-006 — Beat the prediction market at a fair horizon (no cheating). Result: not yet, and WHY.

Directive: make "beat prediction markets like Manifold" a core KPI; run no-cheat backtests and
iterate. This is the honest outcome and diagnosis. **Headline: at a fair information horizon the
market beats the swarm (Brier 0.178 vs 0.260). The binding constraint is INFORMATION STALENESS,
not reasoning — and no reasoning-side iteration (diverse ensembles, extremizing) closes it. The
only fix is as-of information retrieval; and strategically this confirms the audit — a liquid
market on public events is the one place a model is designed to lose.**

## The fair test (fixing the earlier unfair one)
The earlier "we lose to Manifold" compared our cold prediction to the market's CLOSING price
(near-hindsight). This fixes it: `manifold_harness.py` snapshots the market probability at a fixed
lead **T = creation + 48h** (reconstructed from the bet history), the predictor forecasts at T
**blind to the market price**, and both are scored against the eventual resolution. 140 resolved
binary markets that resolved AFTER the model's Jan-2026 cutoff (contamination-free), >=8 bettors,
>=3-day life. Predictions by an 8-agent swarm, question-text only, **no web search** (a lookup
would surface the post-cutoff outcome).

## Result
| segment | n | model Brier | market@48h Brier | model beats market (head-to-head) |
|---|---|---|---|---|
| ALL | 140 | 0.260 | **0.178** | 41% |
| thin (<25 bettors) | 106 | 0.272 | 0.175 | 40% |
| deep (>=25 bettors) | 34 | 0.221 | 0.189 | 44% |
| **market-uncertain (0.25-0.75)** | 93 | 0.247 | 0.234 | **52%** |

Iteration — diverse ensemble (base-rate view + mechanism view) + extremizing: Brier 0.266 (no
better than the single base-rate view, 0.260); extremizing raised the uncertain-subset win-rate to
55% but worsened Brier. **Reasoning-side iteration did not close the gap.**

## Why we lose — error analysis (the important part)
The worst losses are not misjudgments; they are missing information:
- "US-Iran peace agreement before <date>" — resolved YES; model 0.04, market 0.86.
- "Venezuela June-2026 earthquake death toll 2,000+" — YES; model 0.08, market 0.94.
- "ECB raises rates June 11 2026" — YES; model 0.12, market 0.82.
- "Elon Musk net worth > $1T on June 12" — YES; model 0.13, market 0.69.

In every case the market at its 48h horizon had **months of mid-2026 news the model cannot access**
(training cutoff Jan 2026). On "US-Iran deal" the market knew negotiations were underway; the model
structurally cannot. This is an **information deficit, not a reasoning deficit** — which is exactly
why diverse reasoning ensembles don't help.

Corroboration: on the **market-uncertain** subset (where the outcome was NOT yet information-
determined at 48h, so neither side has decisive news) the model is at near-parity with the market
(0.247 vs 0.234) and wins the head-to-head ~52%. The market's overall edge comes almost entirely
from the ~40 markets that were already information-determined by 48h.

## What would actually move this KPI
1. **As-of information retrieval** (the audit's section-6 prescription): feed the predictor news up
   to T but not past T. This is the ONLY thing that can close an information gap. It is hard to do
   leakage-free (naive web search returns the post-resolution outcome); it needs a timestamp-bounded
   news source or a snapshotted corpus. This is the real next experiment if the KPI is retained.
2. **Segment to winnable questions:** far-horizon / structural / counterfactual questions where 48h
   of news is not decisive — close to where the model is already at parity.

## Honest strategic read (brutally realistic, per the audit)
"Beat liquid prediction markets on public resolvable events" is the **hardest bar in forecasting**
and, per ForecastBench/Metaculus AIB, frontier models generally lose it — because markets are
purpose-built to aggregate exactly the current information the model lacks. Setting it as *the*
core KPI points the company at the one arena engineered to be unbeatable by a static-knowledge
model. The system's demonstrated edge (EXP-002/003/004) is on **counterfactual, no-market-exists
questions** — will *this* HN post land, will *this* email get a reply — where no aggregator has
priced the answer. Recommendation: keep "match the market on information-symmetric (uncertain)
questions" as a calibration KPI (we're ~there), pursue as-of retrieval to push it, but make the
company's core KPI the counterfactual decision-lift where the moat actually is.

## Reproduce
`python -m experiments.manifold_harness fetch --target 140 --lead-hours 48` →
`split --k 8` → 8 blind predictor agents write `data/mf_pred_agent*.json` →
`score --preds "data/mf_pred_agent*.json"`. Raw market data gitignored; harness + this doc committed.
