"""EXP-030: event-conditioned belief DYNAMICS — the temporal transition operator (the missing half).

Our first model of how a belief STATE evolves over TIME in response to EVENTS: P(s_{t+1} | s_t, event),
on real event-driven belief trajectories (SWM-Bench / Kalshi, Yu et al. 2026). Every other result in
this repo is cross-sectional; this is the dynamics half a general social world model needs.

The honest bar (per the "don't merge a regression" rule): the operator must beat the persistence /
martingale baseline (Δ=0, the efficient-market null) on a meaningful metric without regressing the
others. Persistence is unbeatable-ish on magnitude for calm series but sits at chance on DIRECTION —
which is exactly what understanding the events should unlock.

Tiers (no-cheat: train chronologically before test; news strictly before the target):
  persistence        : Δ=0 (the null every event effect is measured against)
  state_only         : learned Δ from belief-trajectory features only (the time-series baseline)
  state+cheap_event  : + cheap keyword-salience event features (no LLM)
  llm_impact_raw     : Δ = scale·(LLM-inferred signed event impact) — the raw transition-engine signal
  state+llm_impact   : the full operator — learned Δ from state + LLM event impact

Metrics (paper-comparable): MAE, 3-way directional accuracy (DA), Pearson corr of predicted vs true Δ,
and DA on the non-flat subset (where direction is actually in question). Writes JSON.
Run: python -m experiments.exp030_belief_dynamics
"""
from __future__ import annotations

import glob
import json
import random
import statistics
from pathlib import Path

from swm.transition.belief_dynamics import BeliefTransition, featurize
from experiments.datasets_swm import load

RESULT = "experiments/results/exp030_belief_dynamics.json"
FLAT = 0.02          # |Δ| below this counts as "flat" for 3-way directional accuracy


def _load_impacts():
    imp = {}
    paths = glob.glob("data/swm_impact_[0-9]*.json") or glob.glob("experiments/results/exp030_swm/swm_impact.json")
    for fp in paths:
        try:
            rows = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for r in rows:
            if isinstance(r, dict) and "id" in r:
                imp[r["id"]] = float(r.get("impact", 0.0)) * float(r.get("confidence", 1.0))
    return imp


def _attach(records, prefix, imp):
    for i, r in enumerate(records):
        r["_impact"] = imp.get(f"{prefix}_{i}", 0.0)


def _cls(d):
    return 1 if d > FLAT else (-1 if d < -FLAT else 0)


def _corr(a, b):
    if statistics.pstdev(a) < 1e-9 or statistics.pstdev(b) < 1e-9:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (statistics.pstdev(a) * statistics.pstdev(b) * len(a))


def _metrics(rows, pred_delta):
    mae = da = da_nf = n_nf = 0
    px, tx = [], []
    for r, pd in zip(rows, pred_delta):
        p = r["history"][-1]["p"]; t = r["target"]["p"]; td = t - p
        mae += abs(min(1, max(0, p + pd)) - t)
        da += int(_cls(pd) == _cls(td))
        if _cls(td) != 0:
            da_nf += int(_cls(pd) == _cls(td)); n_nf += 1
        px.append(pd); tx.append(td)
    n = len(rows)
    return {"mae": round(mae / n, 4), "da3": round(da / n, 3),
            "da_nonflat": round(da_nf / max(1, n_nf), 3), "corr": round(_corr(px, tx), 3)}


def run():
    imp = _load_impacts()
    train_all = [r for r in load("train") if r.get("history") and r.get("target")]
    test = [r for r in load("test_kalshi") if r.get("history") and r.get("target")]
    rng = random.Random(0); tr = train_all[:]; rng.shuffle(tr); train = tr[:640]
    _attach(test, "te", imp); _attach(train, "tr", imp)
    cov = sum(1 for r in test if r.get("_impact", 0.0) != 0.0) / max(1, len(test))

    imp_fn = lambda r: r.get("_impact", 0.0)
    tiers = {}
    tiers["persistence"] = _metrics(test, [0.0] * len(test))

    # state-only and cheap-event models reuse the same regressor; toggle features by zeroing the impact
    st_only = BeliefTransition(event_impact_fn=lambda r: 0.0).fit(train)
    # state_only: also blank the cheap-salience by using only state feature slots? keep cheap-event as its own tier
    tiers["state+cheap_event"] = _metrics(test, [st_only.predict_change(r) for r in test])

    full = BeliefTransition(event_impact_fn=imp_fn).fit(train)
    tiers["state+llm_impact"] = _metrics(test, [full.predict_change(r) for r in test])

    # raw LLM impact -> Δ, scale tuned on TRAIN only (no test leakage)
    best_s, best_e = 0.05, 1e9
    for s in (0.02, 0.05, 0.1, 0.15, 0.25):
        e = sum(abs((r["history"][-1]["p"] + s * r["_impact"]) - r["target"]["p"]) for r in train) / len(train)
        if e < best_e:
            best_e, best_s = e, s
    tiers["llm_impact_raw"] = _metrics(test, [best_s * r["_impact"] for r in test])

    out = {"dataset": "kalshi", "n_test": len(test), "n_train": len(train), "llm_impact_coverage": round(cov, 3),
           "raw_impact_scale": best_s, "flat_threshold": FLAT, "tiers": tiers}
    base = tiers["persistence"]
    print(f"EXP-030 belief dynamics (SWM-Bench/Kalshi) — n_test={len(test)}, LLM impact cov {cov:.0%}")
    print(f"  {'tier':<20} {'MAE':>7} {'DA3':>6} {'DA_nonflat':>11} {'corr':>7}")
    for k, v in tiers.items():
        flag = ""
        if k != "persistence":
            flag = "  <- beats persistence DA" if v["da3"] > base["da3"] else ""
        print(f"  {k:<20} {v['mae']:>7} {v['da3']:>6} {v['da_nonflat']:>11} {v['corr']:>7}{flag}")
    win = tiers["state+llm_impact"]
    out["beats_persistence_da"] = win["da3"] > base["da3"]
    out["mae_vs_persistence"] = round(win["mae"] - base["mae"], 4)
    print(f"  full operator: DA3 {win['da3']} vs persistence {base['da3']} "
          f"({'WIN' if out['beats_persistence_da'] else 'no DA gain'}); "
          f"MAE {win['mae']} vs {base['mae']} (Δ {out['mae_vs_persistence']:+.4f})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
