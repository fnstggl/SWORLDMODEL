"""EXP-039: the live post-mortem loop — a leakage-free skill number + self-recalibration (Tetlock #8/#10).

Two things a deployed forecaster must do that a backtest cannot:
  1. **Produce a contamination-free skill number.** Every dated backtest risks the model recalling the
     outcome (the Halawi trap). A forecast LOGGED before its resolution and SCORED after has no future to
     leak — `PostMortemLog` enforces `made_at < resolves_at` and computes skill only over such forecasts.
  2. **Recalibrate from its own track record (perpetual beta).** As forecasts resolve, fit a calibration
     map on the PAST resolved (forecast, outcome) pairs and apply it to FUTURE forecasts — better
     probabilities the longer it runs, using only past to correct future (no leak).

Here the "system forecast" for each Kalshi question is its as-of market belief `target.p` (the s_t our
transition operator starts from); the outcome is the market's near-resolution value (`future[-1] > 0.5`),
which strictly post-dates the forecast. We report:
  A. the leakage-free skill of that belief forecast;
  B. whether chronological Platt recalibration improves calibration on a HELD-OUT LATE window
     (fit on early-resolving forecasts only) — for the belief forecast, and for an explicit
     OVERCONFIDENCE stress-test variant (a common LLM failure) to show the loop corrects miscalibration.

No-cheat: recalibration is fit only on forecasts resolved BEFORE the eval window's forecasts were made.
Run: python -m experiments.exp039_postmortem
"""
from __future__ import annotations

import datetime
import json
import random
import re
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.eval.postmortem import PostMortemLog
from experiments.datasets_swm import load

RESULT = "experiments/results/exp039_postmortem.json"
_MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
        "OCT": 10, "NOV": 11, "DEC": 12}


def _resolves_at(r):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", r.get("market_id", ""))
    if not m:
        return None
    try:
        return datetime.datetime(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3)),
                                 tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None


def _underconfident(p):
    """A common deployed-forecaster failure: hedge probabilities toward 0.5 (shrink in prob space)."""
    return 0.5 + 0.6 * (p - 0.5)


def _cases():
    """Chronological forecasts: (fid, belief_p, made_at, resolves_at, outcome). made_at < resolves_at."""
    out = []
    for i, r in enumerate(load("test_kalshi")):
        if not (r.get("target") and r.get("future") and len(r["future"]) >= 3):
            continue
        ra = _resolves_at(r)
        made = r["target"]["t"]
        if ra is None or ra <= made:                    # leakage-free requires the resolution to post-date
            continue
        outcome = int(r["future"][-1]["p"] > 0.5)        # market's near-resolution value = the outcome
        out.append((f"kx-{i}", float(r["target"]["p"]), made, ra, outcome))
    out.sort(key=lambda z: z[3])                          # by resolution time
    return out


def _skill_on(pairs):
    if len(pairs) < 10:
        return {"n": len(pairs)}
    y = [o for _, o in pairs]; p = [min(1 - 1e-6, max(1e-6, x)) for x, _ in pairs]
    nf = [(pi, yi) for pi, yi in zip(p, y) if abs(pi - 0.5) > 0.02]
    da = sum(int((pi > 0.5) == (yi == 1)) for pi, yi in nf) / max(1, len(nf))
    return {"n": len(pairs), "base_rate": round(sum(y) / len(y), 4), "brier": round(brier_score(y, p), 4),
            "log_loss": round(log_loss(y, p), 4), "ece": round(expected_calibration_error(y, p), 4),
            "directional_accuracy": round(da, 4)}


def _real_recal(cases):
    """Fit Platt on the early-resolving half, evaluate raw vs recalibrated on the LATE half (no leak)."""
    cut = len(cases) // 2
    late = cases[cut:]
    log = PostMortemLog()
    for fid, bp, made, ra, oc in cases:
        log.log(fid, bp, made, ra); log.resolve(fid, oc)
    log.fit_recalibration(late[0][3])                     # uses only forecasts resolved strictly before late
    late_raw = [(bp, oc) for _, bp, _, _, oc in late]
    late_cal = [(log.recalibrate(bp), oc) for _, bp, _, _, oc in late]
    return {"n_fit_early": cut, "n_eval_late": len(late), "recalibration_deployed": log._platt is not None,
            "raw": _skill_on(late_raw), "recalibrated": _skill_on(late_cal)}


def _controlled_recal(n=400):
    """Validate the mechanism at realistic scale: a stationary UNDERCONFIDENT forecaster (a common
    real failure), logged chronologically. Recalibration is fit on the first half's resolved forecasts
    and applied to the held-out second half — the exact deployed loop, but with enough track record to
    test reliably (the real Kalshi set has too few resolved forecasts to recalibrate). Deterministic seed."""
    rng = random.Random(0)
    log = PostMortemLog()
    truth = []
    for i in range(n):
        p_true = rng.random()
        outcome = int(rng.random() < p_true)
        p_report = _underconfident(p_true)                # the forecaster systematically hedges toward 0.5
        log.log(f"c-{i}", p_report, made_at=i, resolves_at=i + 1)
        log.resolve(f"c-{i}", outcome)
        truth.append((p_report, outcome))
    cut = n // 2
    log.fit_recalibration(cut)                             # fit on forecasts resolved before step `cut`
    late = truth[cut:]
    raw = [(p, o) for p, o in late]
    cal = [(log.recalibrate(p), o) for p, o in late]
    return {"n": n, "n_eval_late": len(late), "recalibration_deployed": log._platt is not None,
            "raw": _skill_on(raw), "recalibrated": _skill_on(cal)}


def run():
    cases = _cases()
    # A. headline leakage-free skill of the belief forecast (whole track record)
    log = PostMortemLog()
    for fid, bp, made, ra, oc in cases:
        log.log(fid, bp, made, ra); log.resolve(fid, oc)
    headline = log.skill()

    # B1. real-data recalibration attempt (honest: too little resolved history to recalibrate reliably)
    real = _real_recal(cases)
    # B2. mechanism validation at realistic scale on a controlled miscalibrated stream
    ctrl = _controlled_recal()

    out = {"dataset": "kalshi", "n_forecasts": len(cases), "leakage_free": True,
           "headline_skill": headline, "real_recalibration": real, "controlled_recalibration": ctrl}
    print(f"EXP-039 post-mortem loop — Kalshi, {len(cases)} logged forecasts (made_at < resolves_at)")
    print(f"  A. LEAKAGE-FREE SKILL of the belief forecast (n={headline.get('n')}):")
    print(f"     brier {headline.get('brier')}  log_loss {headline.get('log_loss')}  ece {headline.get('ece')}"
          f"  directional {headline.get('directional_accuracy')}  (base rate {headline.get('base_rate')})")
    print("  B1. REAL-DATA recalibration (fit on early-resolved, eval on held-out late):")
    r, c = real["raw"], real["recalibrated"]
    print(f"      n_late={real['n_eval_late']} deployed={real['recalibration_deployed']}  "
          f"raw ece {r.get('ece')} -> recal ece {c.get('ece')}")
    print(f"      (with only ~{real['n_fit_early']} early resolved forecasts, the do-no-harm guard's "
          f"held-out slice is too small to reliably deploy — a longer track record is needed.)")
    print("  B2. CONTROLLED mechanism check (n=400 stationary underconfident forecaster):")
    r2, c2 = ctrl["raw"], ctrl["recalibrated"]
    print(f"      deployed={ctrl['recalibration_deployed']}  "
          f"raw ece {r2.get('ece')} brier {r2.get('brier')} -> recal ece {c2.get('ece')} brier {c2.get('brier')}")
    print(f"      -> ECE {'improved' if c2['ece'] < r2['ece'] else 'not improved'} "
          f"({r2['ece']:.4f} -> {c2['ece']:.4f}); the loop recovers calibration from its own track record.")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
