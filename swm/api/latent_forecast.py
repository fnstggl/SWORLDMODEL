"""Latent-state forecaster — the re-architecture. A genuine simulation, not a regression.

Every fix from the backtest post-mortem, built in from first principles:

  1/2/3. LATENT STATE + TIME-ACCURATE TRANSITIONS. The outcome is read off a STATE that EVOLVES from now to
     the resolution date over the REAL elapsed time (1 day of horizon = 1 day of diffusion; a year = a year).
     - metric question (a threshold on a measurable number: price, rate, %, count): the number is a latent
       variable at its grounded current value, evolving as a driftless diffusion with volatility scaled by
       √(horizon) — P(YES) = fraction of simulated paths past the threshold at resolution. A real world-model
       roll-forward, time-accurate.
     - event question: the latent log-odds start at the base-rate anchor and are moved by driver shocks whose
       magnitude DECAYS over the horizon (evidence about a far event is less decisive) — P(YES) is the
       Monte-Carlo mean of the sampled outcome probability.
  4. HONEST UNCERTAINTY BY CONSTRUCTION. Every driver carries uncertainty (wide for ungrounded guesses, narrow
     for grounded facts); integrating over it regresses the forecast toward the anchor. The LLM CANNOT assert
     zero uncertainty — the sd floor is enforced here, not trusted from the model.
  5. BASE-RATE / OUTSIDE-VIEW ANCHOR. The forecast STARTS at the reference-class base rate; with no real
     evidence it stays there. A coin flip (base rate 0.5, no drivers) returns exactly 0.5 — by construction.
  6. HONEST IGNORANCE. No drivers ⇒ the base rate. The model expresses "I don't know" as the outside view,
     never as false precision.
  7/8. ONE calibrated readout for every question (the simulation), closing the loop with the backtest that
     tunes its few honest-uncertainty hyperparameters.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
from dataclasses import dataclass, field

from swm.api.retrieval_grounding import parse_json_lenient

SECONDS_PER_YEAR = 365.25 * 86400
STRENGTH_PUSH = 1.2          # max log-odds a single fully-grounded "decisive" driver contributes
TAU_EVIDENCE = 1.5          # yr; driver magnitude decays exp(-H/τ) — a far-off event regresses to base rate
GROUNDED_SD = (0.4, 0.2)    # (frac of |mag|, floor) for a grounded driver's honest uncertainty
GUESS_SD = (1.0, 0.7)       # much wider for an ungrounded guess — integrated out toward the anchor
BASE_SHRINK = 0.75          # regularize the LLM's base-rate logit toward 0.5 (its base rates are noisy)
CV_SD = {"high": 0.06, "med": 0.16, "low": 0.30}   # current-value uncertainty by the model's stated confidence
VOL_FLOOR = 0.18            # min annual vol — the LLM's vol estimate is unreliable; never sim a too-sharp path
# how much to TRUST a metric threshold sim vs fall back to the base rate, by confidence in the (ungrounded)
# current value. Low when the value is a guess — the honest posture without live grounding; a live grounder
# would justify high trust.
METRIC_TRUST = {"high": 0.7, "med": 0.4, "low": 0.2}


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


@dataclass
class LatentSpec:
    base_rate: float = 0.5
    kind: str = "event"
    current_value: float = None
    threshold: float = None
    direction: str = ">"
    annual_vol_pct: float = None
    grounded_conf: str = "low"
    metric_name: str = None                         # what the metric IS (for live grounding of its value)
    drivers: list = field(default_factory=list)     # [{direction:+/-1, strength:0..1, grounded:bool}]
    raw: dict = None


def build_prompt(question, as_of_ts, horizon_years):
    date = _dt.datetime.utcfromtimestamp(int(as_of_ts)).strftime("%Y-%m-%d")
    days = int(horizon_years * 365.25)
    return (
        f"You are a careful SUPERFORECASTER assembling the inputs to a SIMULATION — you do NOT state whether it "
        f"happens.\nTODAY IS {date}. The question resolves in about {days} days. Use ONLY information available "
        f"as of today; do not use knowledge of anything after it.\n\nQUESTION: {question}\n\n"
        f"Give the inputs to simulate it, as JSON:\n"
        f'1. "base_rate": the OUTSIDE-VIEW reference-class base rate — across all similar questions, what '
        f"fraction resolve YES? A fair coin is 0.5; a specific unlikely event is low. When genuinely unsure, "
        f"stay near 0.5. This is the anchor.\n"
        f'2. "kind": "metric" if YES means a measurable NUMBER crosses a threshold (price, rate, %, count, '
        f'index); otherwise "event".\n'
        f'   If metric also give: "current_value" (its value AS OF TODAY), "threshold", "direction" (">" or '
        f'"<"), "annual_vol_pct" (how many percent this number typically moves in a year), "grounded_conf" '
        f'("high"/"med"/"low" — how sure you are of current_value today).\n'
        f'   If event also give: "drivers": up to 5 factors that move it off the base rate, each '
        f'{{"factor","direction" (+1 toward YES, -1 toward NO),"strength" (0-1, how decisive),"grounded" '
        f"(true only if based on a concrete known fact as of today, false if a guess)}}.\n"
        f'Return ONLY compact JSON with keys base_rate, kind, current_value, threshold, direction, '
        f"annual_vol_pct, grounded_conf, drivers.")


def parse_latent(txt):
    r = parse_json_lenient(txt)
    if not r or r.get("base_rate") is None:
        return None
    try:
        br = min(0.98, max(0.02, float(r["base_rate"])))
    except Exception:
        return None
    drivers = []
    for d in (r.get("drivers") or [])[:6]:
        try:
            drivers.append({"direction": 1.0 if float(d.get("direction", 0)) >= 0 else -1.0,
                            "strength": min(1.0, max(0.0, float(d.get("strength", 0)))),
                            "grounded": bool(d.get("grounded", False))})
        except Exception:
            continue

    def _num(k):
        try:
            return float(r[k])
        except Exception:
            return None
    return LatentSpec(base_rate=br, kind=("metric" if r.get("kind") == "metric" else "event"),
                      current_value=_num("current_value"), threshold=_num("threshold"),
                      direction=("<" if str(r.get("direction")) == "<" else ">"),
                      annual_vol_pct=_num("annual_vol_pct"), metric_name=r.get("metric") or r.get("metric_name"),
                      grounded_conf=str(r.get("grounded_conf", "low")).lower(), drivers=drivers, raw=r)


def simulate_latent(spec: LatentSpec, horizon_years, *, n=3000, seed=0):
    """Monte-Carlo the latent state forward over the REAL horizon → P(YES). Anchored + honest by construction."""
    rng = random.Random(seed)
    hy = max(1e-4, float(horizon_years))

    if (spec.kind == "metric" and spec.current_value is not None and spec.threshold is not None
            and spec.annual_vol_pct is not None and spec.current_value > 0):
        x0, thr = spec.current_value, spec.threshold
        cv_sd = CV_SD.get(spec.grounded_conf, 0.22) * abs(x0)
        ann_vol = max(abs(spec.annual_vol_pct) / 100.0, VOL_FLOOR)      # floor: the LLM's vol is unreliable
        sigma = min(ann_vol * math.sqrt(hy), 6.0)                      # TIME-ACCURATE: vol grows with √horizon
        yes = 0
        for _ in range(n):
            start = max(1e-9, x0 + rng.gauss(0, cv_sd))                 # integrate current-value uncertainty
            drift = rng.gauss(0, 0.5 * sigma)                          # trend uncertainty (markets can drift)
            term = start * math.exp(-0.5 * sigma * sigma + drift + sigma * rng.gauss(0, 1))
            hit = term > thr if spec.direction == ">" else term < thr
            yes += 1 if hit else 0
        p_metric = yes / n
        trust = METRIC_TRUST.get(spec.grounded_conf, 0.25)             # shrink to base rate when the value is a guess
        return trust * p_metric + (1 - trust) * spec.base_rate

    # event: latent log-odds anchored at the (regularized) base rate, moved by decaying driver shocks
    L0 = BASE_SHRINK * _logit(spec.base_rate)                          # LLM base rates are noisy -> shrink to 0.5
    decay = math.exp(-hy / TAU_EVIDENCE)                                # far-off event -> evidence fades -> base rate
    acc = 0.0
    for _ in range(n):
        L = L0
        for d in spec.drivers:
            mag = d["direction"] * d["strength"] * STRENGTH_PUSH * decay
            fr, fl = GROUNDED_SD if d["grounded"] else GUESS_SD          # honest uncertainty, never zero
            L += rng.gauss(mag, fr * abs(mag) + fl)
        acc += _sig(L)
    return acc / n                                                     # E[sigmoid(L)] — integrates uncertainty


def latent_forecast(question, as_of_ts, resolve_ts, llm, *, n=3000, seed=0, grounder=None, metric_grounder=None):
    """Compile the honest simulation inputs (one LLM call, no outcome stated) and run the latent-state sim.
    CLOSE THE LOOP: when a `grounder` is supplied and the question is a metric threshold, MEASURE the current
    value live (Coinbase/web via the router) instead of trusting the LLM's guess — which lets the metric sim be
    both confident AND correct (trust=high), the lever the backtest could not use without leaking. Ungroundable
    ⇒ stays at the LLM's as-of estimate with its honest low trust."""
    hy = max(1e-4, (float(resolve_ts) - float(as_of_ts)) / SECONDS_PER_YEAR)
    spec = parse_latent(llm(build_prompt(question, as_of_ts, hy)))
    if spec is None:
        return None, None
    if spec.kind == "metric" and metric_grounder is not None:
        try:                                                  # AS-OF grounding: real price + realised vol at as_of
            g = metric_grounder.ground_metric(question, spec.metric_name, as_of_ts)
            if g is not None and g.get("value") is not None:
                spec.current_value, spec.grounded_conf = float(g["value"]), "high"
                if g.get("annual_vol_pct"):
                    spec.annual_vol_pct = float(g["annual_vol_pct"])
                spec.raw = {**(spec.raw or {}), "_grounded": g}
        except Exception:
            pass
    elif grounder is not None and spec.kind == "metric" and spec.metric_name:
        try:                                                  # live present-grounding (router)
            gv = grounder.ground(spec.metric_name, question=question)
            if gv is not None and gv.value is not None:
                spec.current_value, spec.grounded_conf = float(gv.value), "high"
                spec.raw = {**(spec.raw or {}), "_grounded": {"variable": spec.metric_name,
                                                              "value": gv.value, "source": gv.source}}
        except Exception:
            pass
    return simulate_latent(spec, hy, n=n, seed=seed), spec
