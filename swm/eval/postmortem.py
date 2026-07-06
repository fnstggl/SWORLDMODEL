"""Live post-mortem loop — a self-scoring, self-recalibrating forecast track record (Tetlock #8 + #10).

Two jobs a deployed forecaster must do and we hadn't:
  1. **Leakage-free skill.** Every backtest on dated data risks contamination (the model recalling
     outcomes — the Halawi trap). A forecast LOGGED before its resolution and SCORED after is
     contamination-free *by construction* — there is no future to leak. This log enforces that: skill is
     computed only over forecasts whose `made_at` strictly precedes their `resolves_at`.
  2. **Self-recalibration (perpetual beta).** As forecasts resolve, fit a calibration map on the
     resolved (forecast, outcome) pairs and apply it to future forecasts — so the system's probabilities
     get better the longer it runs, using only PAST resolved forecasts to correct FUTURE ones (no leak).

Deployed, this is the only honest source of a forecasting-skill number and the mechanism that closes
Tetlock's loop: measure → learn → recalibrate → measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.transition.readout import LogisticReadout


def _platt_fit(rows):
    if len(rows) < 12 or len({o for _, o in rows}) < 2:
        return None
    return LogisticReadout(epochs=300, l2=0.5).fit([[_logit(p)] for p, _ in rows],
                                                   [o for _, o in rows])


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


@dataclass
class PostMortemLog:
    forecasts: dict = field(default_factory=dict)     # fid -> {p, made_at, resolves_at, outcome, meta}
    _platt: LogisticReadout = None                    # type: ignore  recalibration map

    def log(self, fid, p, made_at, resolves_at, meta=None):
        self.forecasts[fid] = {"p": float(p), "made_at": made_at, "resolves_at": resolves_at,
                               "outcome": None, "meta": meta or {}}

    def resolve(self, fid, outcome):
        if fid in self.forecasts:
            self.forecasts[fid]["outcome"] = int(outcome)

    def _resolved(self, before=None, recalibrated=False):
        rows = []
        for f in self.forecasts.values():
            if f["outcome"] is None or f["made_at"] is None or f["resolves_at"] is None:
                continue
            if f["made_at"] >= f["resolves_at"]:          # forecast must PRE-date its resolution (no leak)
                continue
            if before is not None and f["resolves_at"] >= before:
                continue
            p = self.recalibrate(f["p"]) if recalibrated else f["p"]
            rows.append((p, f["outcome"]))
        return rows

    def skill(self, before=None, recalibrated=False) -> dict:
        rows = self._resolved(before, recalibrated)
        if len(rows) < 10:
            return {"n": len(rows), "note": "too few resolved forecasts to score"}
        p = [min(1 - 1e-6, max(1e-6, x)) for x, _ in rows]
        y = [o for _, o in rows]
        nf = [(pi, yi) for pi, yi in zip(p, y) if abs(pi - 0.5) > 0.02]
        da = sum(int((pi > 0.5) == (yi == 1)) for pi, yi in nf) / max(1, len(nf))
        return {"n": len(rows), "base_rate": round(sum(y) / len(y), 4), "brier": round(brier_score(y, p), 4),
                "log_loss": round(log_loss(y, p), 4), "ece": round(expected_calibration_error(y, p), 4),
                "directional_accuracy": round(da, 4), "leakage_free": True}

    def fit_recalibration(self, before):
        """Self-validating Platt scaling (perpetual beta, do-no-harm).

        Fit on forecasts resolved strictly before `before`, but only DEPLOY the map if it improves
        calibration on a held-out slice of that same past window. If the track record is already
        calibrated (nothing to fix) or too small/skewed to fit reliably, fall back to identity — so
        recalibration can never make a well-calibrated forecaster worse. Uses only past resolved
        forecasts; never touches anything at or after `before` (no leak).
        """
        rows = self._resolved(before)
        self._platt = None
        if len(rows) < 20:
            return self
        cut = int(len(rows) * 0.8)
        fit_rows, val = rows[:cut], rows[cut:]
        cand = _platt_fit(fit_rows)
        if cand is None or len(val) < 15:                 # need enough held-out evidence to trust the guard
            return self
        yv = [o for _, o in val]
        raw_ece = expected_calibration_error(yv, [min(1 - 1e-6, max(1e-6, p)) for p, _ in val])
        cal_ece = expected_calibration_error(yv, [cand.predict_proba([_logit(p)]) for p, _ in val])
        # deploy only on a MEANINGFUL held-out improvement: a razor-thin edge on a tiny validation
        # slice is noise and won't transfer (distribution shift), so we require a real margin.
        if cal_ece + 0.02 < raw_ece:
            self._platt = cand
        return self

    def recalibrate(self, p):
        return self._platt.predict_proba([_logit(p)]) if self._platt else p
