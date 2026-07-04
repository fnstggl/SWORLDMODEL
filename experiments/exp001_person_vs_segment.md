# EXP-001 — Does per-person inference beat the segment mean?

**The go/no-go experiment for the individual-regime thesis** (docs/individual-hidden-state-inference.md §7).
Runnable via `python -m experiments.exp001_run --db data/events.db`.

## Question

On held-out, *time-forward* message outcomes, does

```
p(reply | per-person posterior, message features)
```

beat

```
p(reply | segment mean, message features)
```

on **log loss** and **calibration (ECE)** — and does the gap **grow** as per-person evidence is added?

If the answer is no at realistic data sparsity, individual modeling is dead weight for this task and
we ship the segment model. That outcome is cheap, early, and honest — by design.

## Data

Any corpus in the normalized event store (`swm/ingestion/store.py`) with derived reply labels:
- **Own Gmail** (real, consented, contamination-free; modest N)
- **Design-partner outbound export** (CSV/JSON importer)
- **Own text threads** (iMessage chat.db importer) — highest per-person frequency, best for this test

Label: `y = 1` if the recipient replied within the channel window (email: 7 days; text: 1 day).
Sends to lists/no-reply addresses and auto-replies (heuristics in the importer) are excluded.

## Split (leakage rules, audit E)

- **Temporal:** choose T at the 80th percentile of send timestamps. Train/fit on sends `< T`,
  evaluate on sends `≥ T`. All persona posteriors and pooled rates are computed **as-of each send**
  (the store's as-of reads make future information physically unavailable).
- **Report two slices:** seen-recipient/future-time and unseen-recipient/future-time.
- No feature may derive from the label window (e.g., "they later unsubscribed").

## The ablation ladder (each rung adds one evidence source)

| Rung | Model | What it isolates |
|---|---|---|
| L0 | global base rate | the floor |
| L1 | + message features (logistic) | does *content* matter at all |
| L2 | + segment pooled reply rate | the aggregate regime — **the bar to beat** |
| L3 | + per-person pooled rate (hierarchical shrinkage) | is there individual signal in outcomes alone |
| L4 | + persona style/latency factors from *their own text* | does their language add signal |
| L5 | + operator correction (correct-a-guess pseudo-obs) | is elicited tacit knowledge worth anything |
| L6 | + VOI answer | is the one surgical question worth asking |

L5/L6 require live operator input, so at backtest time they run only on contacts where corrections
exist; the harness reports N per rung and refuses to compare rungs on non-overlapping populations.

## Metrics & decision rule

Primary: **log loss** and **ECE** (10-bin), per rung, with 1,000-resample bootstrap CIs.
Secondary: Brier, uplift@k (k=20%) for the ranking use-case.

- **GO** if L3 beats L2 on log loss with bootstrap p < 0.05 *and* ECE does not degrade.
- **NO-GO** for individual modeling if L3 ≤ L2; ship L2, revisit when more per-person data accrues.
- Each higher rung is priced by its marginal log-loss improvement — that number is literally
  "what one operator correction / one VOI question is worth."

## Noise floor

Estimate irreducible variance from repeat structure: for recipients with ≥10 sends, the variance of
their empirical reply rate bounds the best achievable Brier. Report every rung *against* this floor
so "we're not better" is distinguishable from "no one could be."

## Contamination note

All data here is private and post-dates nothing the model was trained on (features are engineered,
not LLM-recalled; the optional LLM trait extractor sees only the person's text, never the label).
If an LLM-derived feature is added later, run the redaction probe (audit E.4) before trusting it.
