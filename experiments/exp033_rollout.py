"""EXP-033: multi-step belief rollout — does the temporal dynamics hold up over a horizon? (no-cheat)

The one-step operator (EXP-030) wins at t+1 with a known event. This asks the honest question the whole
"forecast forward" ambition rests on: how fast does error grow as we roll the belief forward days/weeks
with NO knowledge of future events, and does anything beat persistence (the martingale) beyond t+1?

No-cheat: the endogenous per-step dynamics are fit on TRAIN market trajectories; we then roll forward on
held-out TEST markets' FUTURE arrays (untouched daily prices after the target). The event-informed tier
uses the EXP-030 LLM impact for step 1 only (the one step where we have news); all later steps are
endogenous — exactly the real constraint that future news is unknown.

Tiers, reported at each horizon h (MAE vs the actual future price):
  persistence    : flat (belief stays at its last value) — the efficient-market null
  momentum       : linear extrapolation of the recent slope
  endogenous     : learned per-step drift/mean-reversion, rolled forward
  event_informed : endogenous + the LLM event impact at step 1
Also: does the Monte-Carlo uncertainty band widen correctly (coverage at each horizon)?
Writes JSON. Run: python -m experiments.exp033_rollout
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

from swm.transition.rollout import MultiStepRollout, _traj_features
from experiments.datasets_swm import load

RESULT = "experiments/results/exp033_rollout.json"
HORIZON = 10


def _load_impacts():
    imp = {}
    paths = glob.glob("data/swm_impact_[0-9]*.json") or glob.glob("experiments/results/exp030_swm/swm_impact.json")
    for fp in paths:
        for r in json.loads(Path(fp).read_text()):
            if isinstance(r, dict) and "id" in r:
                imp[r["id"]] = float(r.get("impact", 0.0)) * float(r.get("confidence", 1.0))
    return imp


def _seq(rec):
    s = [h["p"] for h in rec.get("history", [])]
    if rec.get("target"):
        s.append(rec["target"]["p"])
    s += [f["p"] for f in rec.get("future", [])]
    return s


def run():
    imp = _load_impacts()
    train = [r for r in load("train") if r.get("history") and r.get("target")]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target")
            and len(r.get("future", [])) >= HORIZON]
    roll = MultiStepRollout().fit([_seq(r) for r in train if len(_seq(r)) >= 3])
    # tune the step-1 impact scale on train's t+1 (no test leakage)
    tr_scale = min((0.05, 0.1, 0.15, 0.2), key=lambda s: statistics.mean(
        abs((r["history"][-1]["p"] + s * imp.get(f"tr_{i}", 0.0)) - r["target"]["p"])
        for i, r in enumerate(train[:640])))

    # accumulate MAE per horizon per tier, and band coverage
    tiers = ["persistence", "momentum", "endogenous", "event_informed"]
    mae = {t: [0.0] * HORIZON for t in tiers}
    cover = [0] * HORIZON
    n = 0
    for i, r in enumerate(test):
        start = [h["p"] for h in r["history"]] + [r["target"]["p"]]
        p0 = start[-1]
        fut = [f["p"] for f in r["future"][:HORIZON]]
        # momentum slope
        k = min(len(start) - 1, 5)
        slope = (start[-1] - start[-1 - k]) / k if k > 0 else 0.0
        endo = roll.rollout(start, HORIZON, first_step_impact=0.0, impact_scale=tr_scale)
        ev = roll.rollout(start, HORIZON, first_step_impact=imp.get(f"te_{i}", 0.0), impact_scale=tr_scale)
        for h in range(HORIZON):
            a = fut[h]
            mae["persistence"][h] += abs(p0 - a)
            mae["momentum"][h] += abs(min(1, max(0, p0 + slope * (h + 1))) - a)
            mae["endogenous"][h] += abs(endo[h]["mean"] - a)
            mae["event_informed"][h] += abs(ev[h]["mean"] - a)
            cover[h] += int(ev[h]["lo"] <= a <= ev[h]["hi"])
        n += 1

    per_h = {t: [round(mae[t][h] / n, 4) for h in range(HORIZON)] for t in tiers}
    coverage = [round(cover[h] / n, 3) for h in range(HORIZON)]
    # summary: at which horizons does any model beat persistence?
    beats = {t: [per_h[t][h] < per_h["persistence"][h] - 1e-9 for h in range(HORIZON)] for t in tiers}
    out = {"dataset": "kalshi", "n_test": n, "horizon": HORIZON, "impact_scale": tr_scale,
           "mae_per_horizon": per_h, "band_coverage_per_horizon": coverage,
           "event_informed_beats_persistence_at": [h + 1 for h in range(HORIZON) if beats["event_informed"][h]],
           "endogenous_beats_persistence_at": [h + 1 for h in range(HORIZON) if beats["endogenous"][h]]}
    print(f"EXP-033 multi-step rollout (Kalshi) — n={n}, horizon={HORIZON} days")
    print(f"  MAE by horizon (day):   " + " ".join(f"h{h+1:>2}" for h in range(HORIZON)))
    for t in tiers:
        print(f"  {t:<14} " + " ".join(f"{per_h[t][h]:.3f}" for h in range(HORIZON)))
    print(f"  band coverage (target 0.80): " + " ".join(f"{coverage[h]:.2f}" for h in range(HORIZON)))
    print(f"  event_informed beats persistence at horizons (days): {out['event_informed_beats_persistence_at'] or 'NONE'}")
    print(f"  endogenous beats persistence at horizons (days): {out['endogenous_beats_persistence_at'] or 'NONE'}")
    drift = round(per_h["persistence"][HORIZON - 1] / max(1e-9, per_h["persistence"][0]), 2)
    print(f"  drift: persistence MAE grows {per_h['persistence'][0]} (h1) -> {per_h['persistence'][HORIZON-1]} "
          f"(h{HORIZON}) = {drift}x")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
