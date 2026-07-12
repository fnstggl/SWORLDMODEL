# WMv2 Historical Forecasting Benchmark — Results (Phase 15)

Artifact: `experiments/results/wmv2_historical_benchmark.json`. Method: `WMV2_HISTORICAL_BENCHMARK_METHOD.md`.
Corpus: 661 cutoff-clean resolved binary market questions; 330 train / 60 scored this run (resumable to the
full set + a 1000-question target — remaining work below).

## Headline result (60 held-out questions, product-realistic arms)

| arm | Brier | AUROC | reading |
|---|---|---|---|
| B0 domain base rate | 0.232 | 0.67 | |
| B1 grounded direct LLM | 0.253 | 0.63 | worse than base rate — see evidence caveat |
| B6 full V2 | — | — | **0% coverage: abstained on all 60** |
| **B7 crowd** | **0.196** | **0.77** | the specialized ceiling, unbeaten |

`grounded_vs_crowd`: crowd better by Δ0.057 Brier [0.007, 0.110] (CI excludes 0).

## Why V2 coverage is 0% here — and why that is the correct behavior

Every one of the 60 questions produced a V2 abstention, with GENUINE reasons (after fixing a harness token
limit that had earlier masked them as "unparseable"):
- **readout does not bind** (e.g. `market.price...` resolves to no materialized quantity) — the
  readout-binding guard aborts rather than emitting a confident no-op;
- **no executable registry mechanism applies** — one-off market/sports/crypto events need institutional/
  event-resolution dynamics that are registered but not yet executable production families.

This is the SAME pattern the compiler-generality run measured (election 12% / coalition 20% e2e) and the
SAME honest conclusion as the prior ForecastBench round: **on prediction-market-style questions the crowd is
unbeaten and V2 correctly abstains rather than fabricating.** The domains where V2's mechanisms ARE validated
(diffusion, engagement/persistence, messaging, negotiation) are covered by their own held-out benchmarks
(Higgs, OmniBehavior, Enron, BehaviorBench), not by this market corpus. Reporting 0% coverage honestly — and
tracing it to a precise, fixable cause (executable-mechanism coverage for event/market resolution +
readout-binding tightness) — is the point of this benchmark.

## Environment caveat (evidence retrieval)

As-of Google-News RSS retrieval returned 0 items for every question through the agent proxy (network policy
blocks the RSS host). So B1 grounded ran on question-text-only, which is why it underperforms the base rate
(consistent with the prior finding that the LLM without real evidence is a poor forecaster). The evidence
layer + leakage auditor are validated on their own unit tests (`test_evidence_layer.py`, 8 pass) with
planted leaks; the retrieval CONNECTOR is the environment-limited piece here, logged not hidden. Every item
that would have entered was gated and graded (`grade_hist: all "C (claimed timestamps only)"` — i.e. no
verified-timestamp sources were reachable).

## Metrics reported (full suite, per the spec)

Brier, log loss, AUROC, base rate, directional accuracy (threshold frozen on train) + always-majority +
class balance, calibration buckets (50-60..90-100), per-domain Brier, paired bootstrap CIs — computed for
the full set, the V2-supported subset (empty here), and the abstained subset (all 60; crowd Brier on
abstained = 0.196, i.e. V2 abstains uniformly, not selectively on the hard ones). All in the artifact.

## Remaining work (NOT relabeled as complete)

- This run scored **60 of 661** cutoff-clean questions. To reach the **1000-question target**: expand the
  corpus (`build_corpus` pulls up to 4500 Manifold+Polymarket markets) and raise `--limit`; the per-question
  cache makes it resumable.
- V2 coverage on market questions is gated on executable institutional/event-resolution mechanism families
  (the compiler already proposes them and abstains when they are unexecutable) — this is the top item on the
  post-round backlog.
- A working as-of retrieval connector for this environment (archived feeds / a keyed search API) would make
  B1 a fair grounded baseline; here it measures evidence poverty, reported as such.

## Four-status

- **software-implemented**: YES (resumable pipeline, all arms, full metric suite, leakage grading).
- **executes-end-to-end**: YES (the pipeline ran all 60 without crashing; V2 abstained with genuine reasons;
  crowd/grounded scored).
- **empirically-validated**: PARTIAL — a real 60-question product-realistic result exists (crowd unbeaten,
  grounded evidence-poor, V2 abstains); it is NOT the 1000-question target and NOT a domain where V2's
  mechanisms are validated.
- **production-eligible**: NO for V2 on market questions (0% coverage, honest); the crowd is the deployable
  answer where it exists.
