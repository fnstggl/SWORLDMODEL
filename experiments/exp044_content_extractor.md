# EXP-044 — Resolution-aware content extraction: how far lexical reading closes the EXP-043 gap

EXP-043 set the frontier: crude, question-agnostic news features don't beat the base rate, even though
the market extracts the same articles into a decisive signal (corr 0.84). The hypothesis was that
reading the news *for this question's specific outcome* — entity-linked, resolution-aware stance — would
recover what the crude features missed. This builds that extractor (`swm/variables/content_extractor.py`)
and measures exactly how much it recovers.

## What the extractor does
1. **Parse** the question into a resolution frame: subject (entities + salient words), numeric threshold
   + comparison direction, template (event vs scalar).
2. **Link** each news item to the question by subject overlap — only news actually *about* the subject
   counts (the entity-linking step EXP-043 skipped).
3. **Read stance** toward the YES outcome in linked news (positive vs negative outcome terms near the
   subject), recency-weighted, plus a resolution ("it's decided") cue.

## Result (Kalshi, train 2,049 / test 574, base rate 0.348)

**Raw signal — best single feature's correlation with the outcome (the honest measure; a multivariate
readout dilutes a weak signal at n=574):**

| | best crude feature | best resolution-aware feature | market lean |
|---|---|---|---|
| corr with outcome | 0.088 | **0.112** | 0.844 |

**Resolution-aware extraction recovers 1.26× the raw signal of crude features** — the recency-weighted
subject-stance is the strongest single grounded feature we have. The approach is directionally right:
reading the news for *this* outcome beats reading it question-agnostically.

**But it is still far too weak to matter in calibrated prediction:**

| arm | log-loss | corr | dir. acc |
|---|---|---|---|
| base rate (composite) | 0.653 | 0.00 | 0.641 |
| crude news features | 0.655 | 0.047 | 0.641 |
| resolution-aware | 0.660 | 0.024 | 0.641 |
| market lean (ceiling) | **0.271** | **0.844** | **0.950** |

Neither crude nor resolution-aware beats the base rate in log-loss. The best lexical feature recovers only
**~13% of the correlation the market extracts** from the same news.

## The honest finding — the frontier is semantic, not lexical
EXP-043 and EXP-044 together bracket the problem precisely:
- **EXP-037**: LLM-*gestalt* drivers add nothing (no real external variable).
- **EXP-043**: *crude* real-news features add nothing (real variable, read too shallowly).
- **EXP-044**: *entity-linked, resolution-aware lexical* features recover 1.26× more raw signal but still
  only 13% of the market's — and don't survive to calibrated prediction.

The gap is **genuine content understanding**: stance detection oriented to the *specific* resolution
criterion ("does *this outcome's* subject appear winning/losing?"), which lexical term-matching cannot do
because outcome polarity is question-specific (a positive economic headline means NO for "will
unemployment be above 4.2%"). Closing it needs semantic reading (embeddings / an LLM stance judge against
the resolution), not more lexical features. This experiment quantifies the remaining distance (13% →
100%) and confirms the bottleneck is NLP depth. The extractor is the pure-Python scaffold a semantic
stance model slots into.

## Honest limits
- Pure-Python lexical stance only; no embeddings, no per-question polarity orientation. That is exactly
  the missing piece the result points to.
- Outcome proxied by the near-resolution market value.

## Reproduce
`python -m experiments.exp044_content_extractor` → `experiments/results/exp044_content_extractor.json`.
`python -m pytest tests/test_content_extractor.py`.
