# EXP-027 — Learned feature discipline: win without hand-picking traits

EXP-025's person-level win came with an asterisk: it needed **hand-picking** `intellectual_humility`;
the full 23-trait dense model overfit at n≈110 authors. Hand-picking doesn't scale and risks snooping.
This experiment removes the hand — the model selects its own features on the training authors only — and
asks whether it recovers the win *and* discovers the right traits.

## What was built
- `swm/transition/sparse_readout.py`:
  - `SparseLogisticReadout` — elastic-net (L1+L2) logistic by proximal gradient; L1 drives coefficients
    to exactly zero (learned selection).
  - `ScreenedLogisticReadout` — filter method: keep the top-*k* features by |correlation with y| on the
    **training** data, then a dense L2 logistic on those; *k* tuned on an inner train/val split. Far more
    stable than L1 at small *n*, and leakage-free (selection sees only training labels).

## Result (CMV person-level, 160 authors, 6 seeds; predict an unseen author's above-median persuasion rate)
| model (all given the full 23 traits unless noted) | log-loss gain vs base | accuracy |
|---|---|---|
| dense L2, all 23 traits (the EXP-025 overfitter) | −0.082 | 0.55 |
| elastic-net L1, all 23 traits | −0.032 | 0.51 |
| **screened (top-k by train corr, k tuned no-leakage)** | **+0.045** | **0.62** |
| hand-picked `intellectual_humility` (EXP-025 oracle) | +0.045 | 0.63 |

**The learned screener matches the hand-picked oracle (+0.045 log loss) without being told which trait
to use** — and it does so by choosing, on its own, the persuasion-relevant traits (times selected across
6 seeds): **certainty_disposition 6/6, intellectual_humility 2/6, politeness_disposition 1/6**. It
converges on k≈1.5 of 23 traits.

## What this establishes
- The EXP-025 win is **not an artifact of hand-picking**: a disciplined model recovers it from the full
  trait set, and independently rediscovers the same persuasion-theory traits (humility ↑, certainty ↓,
  politeness ↑) that the correlations flagged.
- **Discipline > raw capacity at small n.** Dense-all-23 and L1 both *lose* to the base rate; the win
  requires either sparsity-constrained selection (screening) or a strong prior. More inference helps;
  more free parameters do not.

## Honest limits
- **L1 elastic-net underperforms here** (−0.032): at ~110 training authors it scatters across correlated
  traits rather than converging. The stable win comes from the simpler correlation filter with a sparse
  grid (k ≤ 3) — the right prior at this sample size. L1 should become competitive with more authors.
- The screener's *k* and grid encode a sparsity prior; that prior is justified in-sample (k=5,8 overfit,
  shown), not free. At larger *n* the grid should widen.

## Reproduce
`python -m experiments.exp027_learned_discipline` (uses the committed EXP-025 persona signals).
`python -m pytest tests/test_sparse_readout.py` covers the L1 zeros, screening selection, and no-leakage
k-tuning on synthetic data.
