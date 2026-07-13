# WMv2 Historical Forecasting Benchmark — Method (Phase 15)

Harness: `experiments/wmv2_historical_benchmark.py` (resumable). Corpus:
`experiments/results/backtest_corpus.json` via `swm/eval/forecasting_corpus.py`.

## The product simulation

```
historical question at time T
  → retrieve ONLY evidence available before T (as-of Google-News RSS, dated, gated)
  → typed EvidenceBundle (content+bundle hash, actor visibility) + leakage audit
  → compile a world from scratch (the ONE general compiler, no per-question predictor)
  → infer hidden state → simulate → read terminal probability
  → calibrate + abstain
  → compare with the eventual resolution
```

The SAME general path serves every question — no benchmark-specific model is trained per question, and the
benchmark adapter only loads input / defines the externally-given resolution / scores.

## Corpus & leakage defenses

- Manifold + Polymarket resolved binary markets, crowd probability reconstructed at a fair as-of lead
  (40% of market life), all `cutoff_clean` (resolved AFTER the model's training cutoff, 2024-07-01 — the
  model cannot have memorized the outcome).
- Per-question immutable as-of timestamp; resolution outcome + resolution time recorded.
- Evidence layer enforces the as-of gate (zero slack) and the leakage auditor scans every retrieved item
  for resolution terms / future dates / retrospective language / duplicates and grades the timestamp basis.
- Google-News `before:`/RSS dates are used as DISCOVERY only; the bundle re-checks each item's parsed date
  against as-of and quarantines failures. Retrieval failures degrade to evidence-poor (logged), never leak.

## Baselines (product-realistic — what a consumer could use for an arbitrary new question)

| id | arm |
|---|---|
| B0 | domain base rate (train split) |
| B1 | grounded one-shot LLM with the as-of evidence bundle |
| B6 | full V2 (compile → materialize → rollout → terminal readout), calibrated + abstaining |
| B7 | crowd/market probability as of T (where one genuinely existed) |

B2 (call-matched ensemble), B3 (observer panel), B4 (analogical retrieval), B5 (generic calibrated stack)
are implemented as arms on the base branch; this run reports the decision-relevant B0/B1/B6/B7 set and logs
the rest as available-not-run (honest scope, not hidden).

**Specialized ceiling baselines** (task-specific ML, empirical histogram, the crowd/market) are reported
SEPARATELY and are NOT treated as the product bar — they answer "what could extensive exact-task training
achieve," not "what could a consumer use for an arbitrary new question."

## Metrics

Brier, log loss, AUROC, base rate, mean prediction — for the FULL set, the V2-SUPPORTED subset, and the
ABSTAINED subset SEPARATELY; per-domain Brier; directional accuracy with the threshold frozen on train +
always-majority baseline + class balance; calibration buckets (50-60, 60-70, 70-80, 80-90, 90-100) with
observed frequency vs mean confidence; paired bootstrap CIs (V2 vs crowd / grounded / base).

## Coverage & abstention are first-class

V2 does not answer every question. Its abstention rate and the reasons are reported, and the crowd Brier on
the ABSTAINED subset is recorded (so we can see whether V2 abstains on the hard questions). A subset is
NEVER relabeled as the completed benchmark; the remaining-work field states exactly how many of the
corpus's cutoff-clean questions were scored and how to reach the 1000-question target (expand the corpus —
`build_corpus` pulls up to 4500 — and raise `--limit`; the per-question cache makes it resumable).

## Honest scope of this run

This corpus is prediction-market questions — heavily elections/sports/crypto/one-off events. These are
exactly the domains where V2's executable mechanism library is thinnest (the compiler-generality run
measured election 12% / coalition 20% e2e). So V2 will have LOW coverage here and correctly abstain on
most; the crowd is the specialized ceiling and is expected to dominate. This is the honest product-realistic
picture for market-style questions, and it is reported as such — not disguised. The domains where V2's
mechanisms ARE validated (diffusion, engagement/persistence, messaging, negotiation) are covered by their
own held-out benchmarks (Higgs, OmniBehavior, Enron, BehaviorBench), not by this market corpus.
