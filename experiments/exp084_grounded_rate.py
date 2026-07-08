"""EXP-084: GROUND the per-series growth rate — the dynamics analog of state grounding.

EXP-083's pooled operator beats persistence with a growing edge, but LOSES to local linear extrapolation at
short horizon, because it climbs at a POOLED average rate, not each technology's own. The fix is exactly what
state grounding did for a variable's value: keep the transferable STRUCTURE (the saturation curvature, learned
across many technologies) but MEASURE the entity-specific SCALE (its growth rate) from its own recent
trajectory. `TransitionOperator.ground_gain` projects the series' recent observed Δ onto the pooled drift to
recover a per-series gain γ; rolling with Δ = γ·d_pool(x) climbs at the technology's OWN velocity while still
bending to saturation.

A 5-yr window's rate is most informative NEAR-term, though — a temporarily-flat window would wrongly predict a
long-run stall. So the deployed operator BLENDS: it trusts the grounded rate early and relaxes toward the
pooled rate over the horizon (`gain_relax`), the blend rate tuned on the TRAIN technologies only (never the
held-out test set). The claim: the grounded-rate operator wins (or ties) at EVERY horizon — vs persistence, vs
linear extrapolation (the EXP-083 short-horizon gap, closed), AND vs the pooled operator.

Run: python -m experiments.exp084_grounded_rate
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.simulation.transition_operator import (TransitionOperator, persistence_rollout,
                                                quadratic_self_basis)

ADOPT = "experiments/results/exp074/adoption.json"
RESULT = "experiments/results/exp084_grounded_rate.json"
WINDOW = 5
HORIZONS = [1, 3, 5, 8, 12, 16]
RELAX_GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]


def _skill(e, base):
    return round(1 - e / base, 4) if base > 0 else None


def _eval(op, techs, tech_list, relax):
    """Per-horizon errors for grounded-blend / pooled / persistence / linear over a set of technologies."""
    acc = {h: {"gr": 0.0, "pool": 0.0, "pe": 0.0, "lin": 0.0, "cov": 0.0, "n": 0.0, "gains": []}
           for h in HORIZONS}
    for t in tech_list:
        s = techs[t]
        for h in HORIZONS:
            a = acc[h]
            for i in range(WINDOW, len(s) - h):
                start, truth = {"adopt": s[i]["adopt"]}, s[i + h]["adopt"]
                gain = op.ground_gain(s[:i + 1], window=WINDOW)         # leakage-free: history up to origin only
                gr = op.rollout(start, h, n=250, seed=i, gain=gain, gain_relax=relax)["adopt"]
                pool = op.rollout(start, h, n=250, seed=i)["adopt"]["mean"]
                slope = s[i]["adopt"] - s[i - 1]["adopt"]
                lin = min(1.0, max(0.0, s[i]["adopt"] + slope * h))
                a["gr"] += (gr["mean"] - truth) ** 2
                a["pool"] += (pool - truth) ** 2
                a["pe"] += (persistence_rollout(start, ["adopt"], h)["adopt"]["mean"] - truth) ** 2
                a["lin"] += (lin - truth) ** 2
                a["cov"] += 1 if gr["p05"] <= truth <= gr["p95"] else 0
                a["gains"].append(gain["adopt"])
                a["n"] += 1
    return acc


def run() -> dict:
    raw = json.loads(Path(ADOPT).read_text())
    techs = {}
    for name, pts in raw.items():
        s = [(int(y), float(v) / 100.0) for y, v in pts]
        s = [(y, f) for y, f in s if 0.02 <= f <= 0.98]
        if len(s) >= 10:
            techs[name] = [{"adopt": f} for _, f in s]
    names = sorted(techs)
    train_techs = [names[i] for i in range(len(names)) if i % 2 == 0]
    test_techs = [names[i] for i in range(len(names)) if i % 2 == 1]
    op = TransitionOperator(names=["adopt"], basis=quadratic_self_basis, los=[0.0], his=[1.0]).fit(
        [techs[t] for t in train_techs])

    # --- tune the grounded->pooled blend rate on the TRAIN technologies ONLY (never the held-out test set) ---
    tune = {relax: _eval(op, techs, train_techs, relax) for relax in RELAX_GRID}
    tot_mse = {relax: sum(acc[h]["gr"] for h in HORIZONS) / sum(acc[h]["n"] for h in HORIZONS)
               for relax, acc in tune.items()}
    best_relax = min(tot_mse, key=tot_mse.get)

    # --- evaluate the tuned blend on the HELD-OUT test technologies ---
    ev = _eval(op, techs, test_techs, best_relax)
    rows = {}
    for h in HORIZONS:
        a = ev[h]
        rows[h] = {"grounded_skill_vs_persistence": _skill(a["gr"], a["pe"]),
                   "grounded_skill_vs_linear": _skill(a["gr"], a["lin"]),
                   "grounded_skill_vs_pooled": _skill(a["gr"], a["pool"]),
                   "pooled_skill_vs_linear": _skill(a["pool"], a["lin"]),      # the EXP-083 gap, for contrast
                   "grounded_rmse": round((a["gr"] / a["n"]) ** 0.5, 4), "coverage90": round(a["cov"] / a["n"], 3),
                   "median_gain": round(sorted(a["gains"])[len(a["gains"]) // 2], 3), "n": int(a["n"])}
    wins = all(r["grounded_skill_vs_persistence"] > 0 and r["grounded_skill_vs_linear"] >= -0.02
               and r["grounded_skill_vs_pooled"] >= -0.02 for r in rows.values())
    res = {"thesis": "grounding the per-series rate (keep pooled curvature, measure own velocity, relax to "
                     "pooled long-horizon) wins/ties at EVERY horizon vs persistence, linear extrapolation, "
                     "AND the pooled operator",
           "train_technologies": len(train_techs), "test_technologies": len(test_techs), "window": WINDOW,
           "blend_relax_tuned_on_train": best_relax, "wins_every_horizon": wins,
           "skill_by_horizon_years": rows}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-084  grounded per-series RATE (dynamics analog of state grounding), adoption HELD-OUT techs")
    print(f"  fit shape on {len(train_techs)} techs, forecast {len(test_techs)} held-out; rate grounded from each "
          f"tech's last {WINDOW}yr, blend relax={best_relax} (tuned on TRAIN techs)")
    print(f"  {'h(yr)':>5s} {'vs_persist':>11s} {'vs_linear':>10s} {'vs_pooled':>10s} | "
          f"{'(pooled vs_lin)':>15s} {'cov90':>6s} {'medγ':>6s}")
    for h, r in rows.items():
        print(f"  {h:>5d} {r['grounded_skill_vs_persistence']:>11.3f} {r['grounded_skill_vs_linear']:>10.3f} "
              f"{r['grounded_skill_vs_pooled']:>10.3f} | {r['pooled_skill_vs_linear']:>15.3f} "
              f"{r['coverage90']:>6.2f} {r['median_gain']:>6.2f}")
    print(f"  => grounded-rate wins/ties at EVERY horizon vs all three baselines: {wins}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
