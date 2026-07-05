"""Split-conformal prediction for the binary-outcome regime — a finite-sample coverage guarantee.

The calibration badge (ECE) says the probabilities are *on average* well-calibrated. Conformal adds the
stronger, per-prediction contract the uncertainty story was missing: a PREDICTION SET over the outcomes
{0,1} that is guaranteed to contain the true outcome with probability >= 1 - alpha, under only the
exchangeability of the calibration data — no distributional assumptions.

Split conformal (Vovk; Angelopoulos & Bates): on a held-out calibration set, score each example by its
nonconformity s_i = 1 - p_model(true class). Take q = the ceil((n+1)(1-alpha))/n empirical quantile of
those scores. For a new instance, include a label in the prediction set iff its nonconformity <= q:

    include 1  iff (1 - p) <= q
    include 0  iff  p       <= q            (nonconformity of label 0 is 1 - (1-p) = p)

The set is then one of:
  {1}    confident positive       {0}    confident negative
  {0,1}  genuinely uncertain (the honest "could be either" — the set-valued analog of abstaining)
  {}     both outcomes surprising (rare; flags an out-of-distribution or mis-modeled instance)

`coverage()` verifies the guarantee empirically on a test split (should land near 1 - alpha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _quantile_level(n: int, alpha: float) -> float:
    """The finite-sample-corrected quantile rank for split conformal."""
    return min(1.0, math.ceil((n + 1) * (1 - alpha)) / max(1, n))


@dataclass
class ConformalBinary:
    alpha: float = 0.1                      # target miscoverage; 1 - alpha is the coverage guarantee
    q: float = 1.0                          # nonconformity threshold learned on calibration data
    n_cal: int = 0

    def fit(self, p_list, y_list) -> "ConformalBinary":
        """Calibrate on held-out (predicted P(y=1), true y) pairs."""
        scores = []
        for p, y in zip(p_list, y_list):
            p = min(1 - 1e-9, max(1e-9, p))
            scores.append(1 - (p if int(y) == 1 else (1 - p)))     # 1 - prob(true class)
        scores.sort()
        self.n_cal = len(scores)
        if self.n_cal == 0:
            self.q = 1.0
            return self
        lvl = _quantile_level(self.n_cal, self.alpha)
        idx = min(self.n_cal - 1, max(0, int(math.ceil(lvl * self.n_cal)) - 1))
        self.q = scores[idx]
        return self

    def predict_set(self, p: float) -> list[int]:
        p = min(1 - 1e-9, max(1e-9, p))
        s = []
        if p <= self.q:            # nonconformity of label 0 is p
            s.append(0)
        if (1 - p) <= self.q:      # nonconformity of label 1 is 1 - p
            s.append(1)
        return s

    def coverage(self, p_list, y_list) -> dict:
        """Empirical coverage + mean set size on a test split — the guarantee is coverage >= 1 - alpha."""
        n = len(y_list)
        if n == 0:
            return {"coverage": None, "avg_set_size": None, "target": round(1 - self.alpha, 3), "n": 0}
        covered = sizes = uncertain = empty = 0
        for p, y in zip(p_list, y_list):
            st = self.predict_set(p)
            covered += int(int(y) in st)
            sizes += len(st)
            uncertain += int(len(st) == 2)
            empty += int(len(st) == 0)
        return {"coverage": round(covered / n, 4), "avg_set_size": round(sizes / n, 4),
                "target": round(1 - self.alpha, 3), "frac_uncertain": round(uncertain / n, 4),
                "frac_empty": round(empty / n, 4), "n": n}
