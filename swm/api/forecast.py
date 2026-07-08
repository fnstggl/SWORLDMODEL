"""The forecasting front door — ALWAYS run the honest latent-state simulation.

Closes the loop: one entry point that turns any binary future question into a genuine latent-state simulation
(base-rate-anchored, honest-uncertainty, time-accurate), with the live grounder wired in so metric questions
measure their current value instead of guessing it. This is the deployable forecaster the 660-question backtest
validated (calibrated ≤ base rate, real discrimination on modelable domains) — not the discredited readout.

    from swm.api.forecast import live_forecaster
    f = live_forecaster()                       # latent sim + DeepSeek + live grounding router
    f("Will Bitcoin be above $80k on 2026-09-01?", resolve="2026-09-01")

`calibration_temp` applies the overconfidence correction fit on the backtest (temperature scaling toward the
base rate); recalibrate per deployment as more resolved questions accrue — the flywheel.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import math

from swm.api.latent_forecast import latent_forecast

DEFAULT_CALIBRATION_TEMP = 0.7          # mild overconfidence correction (backtest: the raw sim is a bit sharp)


def _to_ts(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return float(calendar.timegm(_dt.datetime.strptime(str(v)[:19 if "T" in str(v) else 10], fmt).timetuple()))
        except Exception:
            continue
    return None


def _now_ts():
    return float(calendar.timegm(_dt.datetime.utcnow().timetuple()))


def _cal(p, temp):
    if temp is None or temp == 1.0:
        return p
    z = math.log(min(1 - 1e-6, max(1e-6, p)) / (1 - min(1 - 1e-6, max(1e-6, p))))
    return 1 / (1 + math.exp(-max(-35, min(35, temp * z))))


class Forecaster:
    """Wraps the latent-state simulation as the system's forecasting front door."""

    def __init__(self, llm=None, grounder=None, *, n=4000, calibration_temp=DEFAULT_CALIBRATION_TEMP):
        self.llm, self.grounder, self.n, self.temp = llm, grounder, n, calibration_temp

    def __call__(self, question, *, resolve=None, as_of=None, horizon_days=None) -> dict:
        llm = self.llm
        if llm is None:
            from swm.api.resilient_llm import resilient_chat_fn
            llm = resilient_chat_fn(system="You are a careful superforecaster. Reply with ONLY compact JSON.",
                                    max_tokens=700)
        as_of_ts = _to_ts(as_of) or _now_ts()
        resolve_ts = _to_ts(resolve) or (as_of_ts + (horizon_days or 90) * 86400)
        p, spec = latent_forecast(question, as_of_ts, resolve_ts, llm, n=self.n, grounder=self.grounder)
        if p is None:
            return {"question": question, "error": "could not compile a simulation"}
        return {"question": question, "p_yes": round(_cal(p, self.temp), 4), "p_yes_raw": round(p, 4),
                "kind": spec.kind, "base_rate": round(spec.base_rate, 3),
                "grounded": (spec.raw or {}).get("_grounded"),
                "horizon_days": round((resolve_ts - as_of_ts) / 86400, 1),
                "n_drivers": len(spec.drivers)}


def live_forecaster(*, n=4000, calibration_temp=DEFAULT_CALIBRATION_TEMP) -> Forecaster:
    """The deployable forecaster: latent-state simulation + DeepSeek + the live grounding router (Coinbase/web),
    so a metric question grounds its current value and simulates forward from the real present."""
    grounder = None
    try:
        from swm.api.live_grounding import live_router
        grounder = live_router()
    except Exception:
        grounder = None
    return Forecaster(grounder=grounder, n=n, calibration_temp=calibration_temp)
