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
    # tune the raw-impact scale and the gate_scale on TRAIN only (no test leakage)
    best_s, best_e = 0.05, 1e9
    for s in (0.02, 0.05, 0.1, 0.15, 0.25):
        e = sum(abs((r["history"][-1]["p"] + s * r["_impact"]) - r["target"]["p"]) for r in train) / len(train)
        if e < best_e:
            best_e, best_s = e, s
    best_g, best_ge = 0.5, 1e9
    for g in (0.2, 0.35, 0.5, 0.7):
        m = BeliefTransition(event_impact_fn=imp_fn, gate_by_impact=True, gate_scale=g).fit(train)
        e = sum(abs(m.predict_belief(r) - r["target"]["p"]) for r in train) / len(train)
        if e < best_ge:
            best_ge, best_g = e, g

    st_only = BeliefTransition(event_impact_fn=lambda r: 0.0).fit(train)
    full = BeliefTransition(event_impact_fn=imp_fn).fit(train)
    gated = BeliefTransition(event_impact_fn=imp_fn, gate_by_impact=True, gate_scale=best_g).fit(train)

    def tierset(rows):
        return {
            "persistence": _metrics(rows, [0.0] * len(rows)),
            "state+cheap_event": _metrics(rows, [st_only.predict_change(r) for r in rows]),
            "llm_impact_raw": _metrics(rows, [best_s * r["_impact"] for r in rows]),
            "state+llm_impact": _metrics(rows, [full.predict_change(r) for r in rows]),
            "gated_llm_impact": _metrics(rows, [gated.predict_change(r) for r in rows]),
        }

    tiers = tierset(test)
    # event-driven subset: transitions where the LLM judged a real event (|impact| >= τ) — the paper's
    # "attributed subset", where dynamics should win over the persistence null.
    tau_ev = 0.15
    ev = [r for r in test if abs(r.get("_impact", 0.0)) >= tau_ev]
    tiers_ev = tierset(ev) if len(ev) >= 20 else {}

    base = tiers["persistence"]
    out = {"dataset": "kalshi", "n_test": len(test), "n_train": len(train),
           "llm_impact_coverage": round(cov, 3), "raw_impact_scale": best_s, "gate_scale": best_g,
           "flat_threshold": FLAT, "n_event_driven": len(ev), "tiers_all": tiers, "tiers_event_driven": tiers_ev}
    print(f"EXP-030 belief dynamics (SWM-Bench/Kalshi) — n_test={len(test)}, LLM impact cov {cov:.0%}, "
          f"gate_scale {best_g}")
    print(f"  --- FULL test set (n={len(test)}) ---")
    print(f"  {'tier':<20} {'MAE':>7} {'DA3':>6} {'DA_nonflat':>11} {'corr':>7}")
    for k, v in tiers.items():
        print(f"  {k:<20} {v['mae']:>7} {v['da3']:>6} {v['da_nonflat']:>11} {v['corr']:>7}")
    if tiers_ev:
        b2 = tiers_ev["persistence"]
        print(f"  --- EVENT-DRIVEN subset (|impact|>={tau_ev}, n={len(ev)}) ---")
        for k, v in tiers_ev.items():
            print(f"  {k:<20} {v['mae']:>7} {v['da3']:>6} {v['da_nonflat']:>11} {v['corr']:>7}")
        out["event_subset_gated_beats_persistence"] = {
            "mae": tiers_ev["gated_llm_impact"]["mae"] <= b2["mae"],
            "da3": tiers_ev["gated_llm_impact"]["da3"] > b2["da3"]}
    out["full_gated_mae_vs_persistence"] = round(tiers["gated_llm_impact"]["mae"] - base["mae"], 4)
    out["full_gated_da_vs_persistence"] = round(tiers["gated_llm_impact"]["da3"] - base["da3"], 4)
    print(f"  VERDICT (gated operator vs persistence): full-set MAE Δ {out['full_gated_mae_vs_persistence']:+.4f}, "
          f"DA3 Δ {out['full_gated_da_vs_persistence']:+.4f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
