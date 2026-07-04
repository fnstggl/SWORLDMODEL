# EXP-008 — Leakage-safe as-of retrieval: does it help, and can it beat the market?

Domain: Manifold binary event markets (the only domain with a real market price at a fixed horizon).
Test set = the 140 EXP-006 markets that resolved AFTER the Jan-2026 model cutoff, each with a
reconstructed market price at horizon T (48h after creation) and a hidden 0/1 resolution.

## The leakage-safe retrieval source

`swm/retrieval/corpus.py` — `TimestampedCorpus`. Every sibling market is a `Document` timestamped at
its **close/resolution date**. For a target market created at C, `reference_class(question, as_of=C)`
returns only siblings that **resolved strictly before C** — a market that resolved before the target
even existed cannot encode the target's outcome. The gate is structural, not a convention:

- a `Document` with no numeric timestamp fails construction;
- `as_of(t)` returns only docs with `timestamp < t`;
- every retrieval path ends in `_assert_no_future`, which raises `LeakageError` if any returned doc
  is `>= t`. A future document cannot escape even if the similarity ranking is buggy.

`tests/test_corpus_leakage.py` proves it: a future document whose text is an exact match for the
query (so similarity would rank it first if reachable) is never returned, and the guard raises on a
smuggled future doc. **Any retrieval method that could pull post-resolution info raises rather than
leak.**

## Four predictors (identical markets, no web search, no post-as_of data)

1. **no retrieval** — the blind LLM forecast (question text only; the pooled EXP-006 swarm).
2. **LLM + as-of retrieval** — #1 shrunk toward the leakage-safe reference-class YES rate (weight
   `n/(n+8)`).
3. **state-model + retrieval** — a `swm.state.state.Posterior` seeded at the population base rate
   (worth 4 pseudo-markets) and `.observe()`'d with each retrieved sibling's resolution. This is the
   *actual* state machinery, no LLM.
4. **market price @T** — the reconstructed market@T (`data/mf_truth` `market_at_T`).

## Results

```
EXP-008  Manifold event forecasting, n=140 markets, base YES rate 0.471.
leakage-safe reference class: median 2 siblings/market, 41 markets with zero (fall back to prior).

   method                        logloss   brier    ece  uplift@20  beats mkt
   1 no retrieval (LLM)           0.7714  0.2660 0.1245     0.2071       39%
   2 LLM + as-of retrieval        0.6993  0.2516 0.1100     0.1000       36%
   3 state-model + retrieval      0.6876  0.2475 0.0620    -0.0071       30%
   4 market price @T              0.5269  0.1780 0.0701     0.4571        0%  (reference)

== retrieval-rich subset (>=5 siblings, n=63) ==
   1 no retrieval (LLM)         brier 0.2722  logloss 0.8177
   2 LLM + as-of retrieval      brier 0.2450  logloss 0.6850
   3 state-model + retrieval    brier 0.2403  logloss 0.6727
   4 market price @T            brier 0.1465  logloss 0.4491
```

## Does retrieval improve the metrics?

**Calibration: yes, clearly.** Adding leakage-safe retrieval to the LLM cut log loss 0.7714 → 0.6993,
Brier 0.2660 → 0.2516, and ECE 0.1245 → 0.1100. The pure state-model+retrieval went further — the
best ECE in the table (0.0620, roughly *half* the LLM's) and the best non-market log loss (0.6876).
On the retrieval-rich subset (≥5 siblings) the gain is larger (Brier 0.2722 → 0.2403), exactly as
expected: retrieval helps most where a real reference class exists.

**Decision lift / ranking: no — it hurts.** uplift@20 fell from +0.2071 (no retrieval) to +0.1000
(LLM+retrieval) to −0.0071 (state-model). Shrinking toward the reference-class base rate flattens the
top-of-ranking, which is what uplift@k measures. This is the honest tradeoff: **leakage-safe
retrieval buys calibration at the cost of discrimination** in this thin-corpus regime. If the product
KPI is "which markets should I bet," retrieval as implemented here is a net negative; if the KPI is
"give me a well-calibrated probability," it is a net positive.

## Can it beat the market? No — and that is the point, not a failure.

The market@T dominates every predictor on Brier (0.1780 vs best-model 0.2475) and log loss, and
**nothing beats it** (best model is closer to the outcome on only 39% of markets; the state model on
30%). This is not a tuning problem. It is structural, and it confirms the audit's central positioning:

> A leakage-safe retriever is *forbidden by construction* from touching the post-creation
> information that a liquid market aggregates. The market@T price at 48h has absorbed two days of new
> bets, news, and crowd updates; our reference class is limited to markets that fully resolved before
> the target was even created. We are not losing on reasoning — we are losing on **information we are
> not allowed to see**, exactly the staleness constraint diagnosed in EXP-006.

This is the same conclusion reached three independent ways now (EXP-006 direct, EXP-007 title-vs-
context, EXP-008 retrieval-vs-market): **the moat is not open, liquid, market-covered prediction.**
It is (a) calibration, where retrieval genuinely helps, and (b) counterfactual / no-market / private-
entity questions where there is no market@T to lose to.

## Honest limitations

- **Thin corpus.** All 140 markets sit in one month (created 2026-06-02 … 06-29), so the median
  reference class is 2 siblings and 41 markets have zero (they fall back to the population prior).
  The calibration gains would likely be larger with a deep multi-year resolved-market corpus; the
  ranking loss might shrink with a richer, more topically-matched reference class. The *method* is
  the deliverable; the magnitude is corpus-limited and reported as such.
- **Reference class ≈ base rate.** With so few siblings the retrieved signal is close to a global
  base rate, which is precisely why it improves calibration but flattens ranking. A larger corpus
  with tighter topical matching is the obvious next build.
- **No market leg for HN.** EXP-008 is Manifold-only because HN has no market price. The complementary
  HN retrieval question was already answered in EXP-007 (title+context did NOT beat title-only there,
  a different thin-signal regime).

## Verdict

Leakage-safe as-of retrieval is implemented, tested against leakage, and measured. It **improves
calibration (Brier/log loss/ECE)**, **hurts decision lift** in this thin-corpus regime, and **cannot
beat the market** — for the structural reason that it is not allowed to see what the market sees.
Keep it as a calibration layer; do not expect it to win open market-covered forecasts; point the
product at the no-market questions where there is nothing to lose to.
