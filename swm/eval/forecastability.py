"""Forecastability / triage score — say what the model can and can't usefully forecast (Tetlock #1).

A general forecaster must know where effort pays: skip the trivial (already resolved) and the hopeless
(pure noise near 0.5 with no signal), and concentrate on the "Goldilocks zone." This learns a score in
[0,1] estimating how RELIABLY a question's direction can be called, from as-of features:

  - lean magnitude |p−0.5| — confident beliefs resolve toward their side (EXP-036: 0.85 vs 0.80);
  - recent volatility — churny series are less predictable;
  - days-to-resolution — closer to resolution the lean is more decisive;
  - a resolution/result cue in the current news;
  - (when available) driver agreement + max strength from the question engine.

The score is fit to predict the realized directional CORRECTNESS of the lean call (no-cheat, on train),
so a high score genuinely means "we tend to be right here." It drives triage: FORECAST when high, HEDGE
in the middle, ABSTAIN when low — and it is validated by selective forecasting (keeping high-score
questions raises accuracy, EXP-038).
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.transition.readout import LogisticReadout


def forecastability_features(lean, volatility, days_to_res=None, result_cue=0.0,
                             driver_agreement=0.5, driver_strength=0.5):
    """As-of features for how reliably this question's direction can be called."""
    dtr = 1.0 if days_to_res is None else max(0.0, min(1.0, days_to_res / 60.0))
    return [abs(lean), min(1.0, volatility / 0.1), dtr, result_cue,
            abs(driver_agreement - 0.5) * 2, driver_strength]


FEATURE_NAMES = ["abs_lean", "volatility", "days_to_res", "result_cue", "driver_agreement", "driver_strength"]


@dataclass
class ForecastabilityScorer:
    model: LogisticReadout = None                 # type: ignore
    forecast_hi: float = 0.62                     # >= -> FORECAST; <= abstain_lo -> ABSTAIN; else HEDGE
    abstain_lo: float = 0.5

    def fit(self, examples, epochs=300):
        """examples: (features, was_direction_correct in {0,1}). Learns P(call is correct | features)."""
        X = [f for f, _ in examples]
        y = [int(c) for _, c in examples]
        if len(set(y)) == 2:
            self.model = LogisticReadout(epochs=epochs, l2=1.0).fit(X, y)
        return self

    def score(self, features) -> float:
        return self.model.predict_proba(features) if self.model else 0.5

    def triage(self, features) -> str:
        s = self.score(features)
        return "forecast" if s >= self.forecast_hi else ("abstain" if s <= self.abstain_lo else "hedge")
