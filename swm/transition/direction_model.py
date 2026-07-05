"""Direction model — forecasting WHICH WAY a belief moves (leakage-safe).

EXP-033/035 said direction is unforecastable, but that used only the price's own shape and tried
momentum. A leakage-safe diagnostic found a real structural driver: a belief's LEAN (distance from 0.5)
predicts the direction of its future move at roughly its calibration rate (~70% for confident beliefs) —
because a question resolves toward the side it currently favors. That is a directional signal (beats a
coin flip) even though it is NOT a point-forecast edge over the martingale (a 0.7 belief already *is* its
expected value). For questions with no liquid market it is the whole forecast.

This is a classifier — P(belief moves up over the horizon | as-of features) — not a drift term in the MC
rollout (which would conflate direction with variance and wreck the point/CRPS). Features are strictly
as-of: the lean, its magnitude, momentum, a resolution/result cue in the current news, and (when
parseable) days-to-resolution. Trained on the sign of realized multi-step moves in the TRAIN split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout

_RESULT = re.compile(r"\b(win|wins|won|winner|loses|lost|defeat|victory|elected|concede|exit poll|"
                     r"results?|announced|confirms?|resigns?|projected|declared|clinch|beats?)\b", re.I)


def direction_features(prices, *, news=None, days_to_res=None):
    """As-of features for the direction of the next multi-step move."""
    p = prices[-1]
    lean = p - 0.5
    k = min(len(prices) - 1, 5)
    mom = (p - prices[-1 - k]) / k if k > 0 else 0.0
    cue = 0.0
    if news:
        txt = " ".join((n.get("title", "") + " " + n.get("description", "")) for n in news[:8])
        cue = min(1.0, len(_RESULT.findall(txt)) / 3.0)
    dtr = 1.0 if days_to_res is None else max(0.0, min(1.0, days_to_res / 60.0))
    return [lean, abs(lean), lean * abs(lean), mom, cue, dtr, lean * cue]


FEATURE_NAMES = ["lean", "abs_lean", "lean_sq", "momentum", "result_cue", "days_to_res", "lean_x_cue"]


@dataclass
class DirectionModel:
    model: LogisticReadout = None                 # type: ignore
    flat: float = 0.02                            # |move| below this is "no move" (excluded from training)

    def fit(self, examples, epochs=300):
        """examples: list of (features, future_move). Trains P(up) on non-flat moves."""
        X, y = [], []
        for f, move in examples:
            if abs(move) >= self.flat:
                X.append(f); y.append(1 if move > 0 else 0)
        if len(set(y)) == 2:
            self.model = LogisticReadout(epochs=epochs, l2=1.0).fit(X, y)
        return self

    def p_up(self, features) -> float:
        return self.model.predict_proba(features) if self.model else 0.5

    def direction(self, features):
        """+1 (up), -1 (down), or 0 (no confident call) for the coming multi-step move."""
        p = self.p_up(features)
        return 1 if p > 0.55 else (-1 if p < 0.45 else 0)
