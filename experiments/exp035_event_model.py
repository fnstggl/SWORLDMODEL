"""EXP-035: future-event model + distributional multi-step rollout — the honest "simulate forward" test.

EXP-033 proved persistence is unbeatable on the POINT forecast (efficient belief = martingale). So the
real target is a CALIBRATED PREDICTIVE DISTRIBUTION: place variance at the right times, widen correctly
with horizon, carry known directional signal. We score that properly — CRPS (a proper scoring rule that
rewards sharp AND calibrated distributions) and interval coverage — not just MAE.

No-cheat: the event model (drift + heteroskedastic variance) is fit on TRAIN market trajectories; rolled
forward on held-out TEST futures (Kalshi). The known-event direction at step 1 is the EXP-030 LLM impact.

Tiers, per horizon h:
  persistence_point : the martingale point (CRPS = |p_t - actual|) — no distribution
  constant_band     : persistence mean + constant per-step variance (homoskedastic MC)
  event_model       : learned drift + heteroskedastic variance + decaying known-event drift (MC)
Metrics: CRPS (lower better), 80% interval coverage (target 0.80), and MAE of the mean (sanity).
Writes JSON. Run: python -m experiments.exp035_event_model
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics
from pathlib import Path

from swm.transition.event_model import EventModel
from experiments.datasets_swm import load

RESULT = "experiments/results/exp035_event_model.json"
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
    return s + [f["p"] for f in rec.get("future", [])]


def _crps(samples, a):
    """MC CRPS estimator: E|X-a| - 0.5 E|X-X'|. samples sorted asc."""
    n = len(samples)
    e1 = sum(abs(x - a) for x in samples) / n
    # E|X-X'| via sorted-array closed form: sum_i (2i-n+1) x_i * 2 / n^2
    e2 = sum((2 * i - n + 1) * x for i, x in enumerate(samples)) * 2.0 / (n * n)
    return e1 - 0.5 * e2


def run(const_sigma=None):
    imp = _load_impacts()
    train = [r for r in load("train") if r.get("history") and r.get("target")]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target")
            and len(r.get("future", [])) >= HORIZON]
    # fit on a train-fit split; calibrate sigma_mult on a train-val split (no test leakage)
    fit_seqs = [_seq(r) for r in train[:520] if len(_seq(r)) >= 3]
    val = [r for r in train[520:640] if len(r.get("future", [])) >= 5]
    em = EventModel().fit(fit_seqs)
    best_m, best_c = 1.0, 1e9
    for m in (0.6, 0.75, 0.9, 1.0, 1.2):
        em.sigma_mult = m
        c = 0.0; k = 0
        for r in val:
            s = [h["p"] for h in r["history"]] + [r["target"]["p"]]
            f = em.forecast(s, 5, impact=0.0, n_samples=150, seed=1)
            for h in range(5):
                c += _crps(f[h]["samples"], r["future"][h]["p"]); k += 1
        if k and c / k < best_c:
            best_c, best_m = c / k, m
    em.sigma_mult = best_m
    # constant band sigma from train residuals (per-step std)
    if const_sigma is None:
        ds = []
        for r in train:
            s = _seq(r)
            ds += [s[i] - s[i - 1] for i in range(1, len(s))]
        const_sigma = statistics.pstdev(ds) if len(ds) > 2 else 0.03

    tiers = ["persistence_point", "constant_band", "event_no_impact", "event_model"]
    crps = {t: [0.0] * HORIZON for t in tiers}
    cover = {t: [0] * HORIZON for t in tiers}
    mae = {t: [0.0] * HORIZON for t in tiers}
    n = 0
    for i, r in enumerate(test):
        start = [h["p"] for h in r["history"]] + [r["target"]["p"]]
        p0 = start[-1]
        fut = [f["p"] for f in r["future"][:HORIZON]]
        em_f = em.forecast(start, HORIZON, impact=imp.get(f"te_{i}", 0.0), n_samples=300, seed=i)
        em_ni = em.forecast(start, HORIZON, impact=0.0, n_samples=300, seed=i)          # ablate the LLM event
        rng = random.Random(1000 + i)
        cb = [sorted(min(1, max(0, p0 + const_sigma * math.sqrt(h + 1) * rng.gauss(0, 1)))
                     for _ in range(300)) for h in range(HORIZON)]
        for h in range(HORIZON):
            a = fut[h]
            crps["persistence_point"][h] += abs(p0 - a); mae["persistence_point"][h] += abs(p0 - a)
            crps["constant_band"][h] += _crps(cb[h], a); mae["constant_band"][h] += abs(sum(cb[h]) / len(cb[h]) - a)
            lo, hi = cb[h][int(0.1 * len(cb[h]))], cb[h][int(0.9 * len(cb[h]))]
            cover["constant_band"][h] += int(lo <= a <= hi)
            crps["event_no_impact"][h] += _crps(em_ni[h]["samples"], a)
            mae["event_no_impact"][h] += abs(em_ni[h]["mean"] - a)
            cover["event_no_impact"][h] += int(em_ni[h]["lo"] <= a <= em_ni[h]["hi"])
            crps["event_model"][h] += _crps(em_f[h]["samples"], a); mae["event_model"][h] += abs(em_f[h]["mean"] - a)
            cover["event_model"][h] += int(em_f[h]["lo"] <= a <= em_f[h]["hi"])
        n += 1

    perh = {t: {"crps": [round(crps[t][h] / n, 4) for h in range(HORIZON)],
                "coverage": [round(cover[t][h] / n, 3) for h in range(HORIZON)],
                "mae": [round(mae[t][h] / n, 4) for h in range(HORIZON)]} for t in tiers}
    mean_crps = {t: round(statistics.mean(perh[t]["crps"]), 4) for t in tiers}
    out = {"dataset": "kalshi", "n_test": n, "horizon": HORIZON, "const_sigma": round(const_sigma, 4),
           "sigma_mult": round(em.sigma_mult, 3), "mean_crps": mean_crps, "per_horizon": perh,
           "event_beats_persistence_crps": mean_crps["event_model"] < mean_crps["persistence_point"],
           "event_beats_constant_crps": mean_crps["event_model"] < mean_crps["constant_band"],
           "llm_impact_crps_gain": round(mean_crps["event_no_impact"] - mean_crps["event_model"], 4)}
    print(f"EXP-035 event model + distributional rollout (Kalshi) — n={n}, horizon={HORIZON}, "
          f"sigma_mult {em.sigma_mult}")
    print(f"  mean CRPS (lower=better):  persistence {mean_crps['persistence_point']}  "
          f"constant_band {mean_crps['constant_band']}  event_no_impact {mean_crps['event_no_impact']}  "
          f"event_model {mean_crps['event_model']}")
    print(f"  CRPS by horizon:  " + " ".join(f"h{h+1:>2}" for h in range(HORIZON)))
    for t in tiers:
        print(f"    {t:<18} " + " ".join(f"{perh[t]['crps'][h]:.3f}" for h in range(HORIZON)))
    print(f"  event-model 80% coverage: " + " ".join(f"{perh['event_model']['coverage'][h]:.2f}" for h in range(HORIZON)))
    print(f"  constant-band 80% coverage: " + " ".join(f"{perh['constant_band']['coverage'][h]:.2f}" for h in range(HORIZON)))
    print(f"  event beats persistence on CRPS: {out['event_beats_persistence_crps']}; "
          f"beats constant band: {out['event_beats_constant_crps']}")
    print(f"  LLM-impact channel CRPS gain (event_model vs event_no_impact): {out['llm_impact_crps_gain']:+.4f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
