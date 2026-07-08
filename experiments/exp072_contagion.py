"""EXP-072: a real contagion/tipping test — where a coupled dynamic should finally beat simple baselines.

EXP-070/071 found the individual<->institution couplings tie strong simple baselines (SCOTUS ideology, FOMC
inertia). The honest conclusion: a coupled/shared-world model earns its place only where (a) endogenous
cross-agent feedback is strong AND (b) simple baselines are weak. That is exactly the CONTAGION/TIPPING
regime — and baby-name popularity is its cleanest real instance: names spread by pure imitation (bandwagon),
saturate, and then CRASH (fashion fatigue). Persistence and trend both fail hard at the turning points.

Data: real SSA-derived baby-name shares (481 names, 1880-2008), each a fashion S-curve-then-decline.
We forecast a name's share H=10 years ahead from its as-of trajectory only (leakage-free), with three models:

  PERSISTENCE : share stays put (the baseline that beat us on FOMC/GSS).
  TREND       : extrapolate the recent slope linearly.
  CONTAGION   : the COUPLED bandwagon+saturation dynamic — momentum carries the name, but its growth is
                dragged down by its own level (fatigue/over-exposure), so a high-flying name DECELERATES,
                PEAKS, and REVERSES. This is the mean-field coupling (growth depends on current prevalence),
                the exact thing that ties on marginal-dominated data but should WIN when the process is
                genuinely a cascade. Two params (momentum persistence rho, fatigue lambda) fit on TRAIN
                names, scored on held-out TEST names.

The decisive cut: performance AT TURNING POINTS (as-of within +-3yr of the name's peak), where persistence
and trend are structurally wrong. If CONTAGION wins there — and overall — the coupled dynamic finally earns
its place on real data.
Run: python -m experiments.exp072_contagion
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = "experiments/results/exp072/baby_names.json"
RESULT = "experiments/results/exp072_contagion.json"
H = 10                    # forecast horizon (years)


def _series(d):
    ys = sorted(int(y) for y in d)
    return ys, [d[str(y)] * 100 for y in ys]     # percent units


def _contagion_roll(p0, g0, rho, lam, steps):
    """The coupled bandwagon+saturation dynamic: momentum decays, and the current LEVEL drags growth down
    (fatigue) -> rise, peak, reverse. Growth depends on prevalence => a genuinely coupled (non-separable)
    forecast, unlike persistence/trend."""
    p, g = p0, g0
    for _ in range(steps):
        g = rho * g - lam * p
        p = max(0.0, p + g)
    return p


def _samples(names):
    """(name, as_of index) forecast points with >=6yr history and the H-ahead truth available."""
    out = []
    for nm, d in names.items():
        ys, ps = _series(d)
        if len(ys) < 6 + H:
            continue
        peak_i = max(range(len(ps)), key=lambda i: ps[i])
        for i in range(4, len(ys) - H):
            if ys[i] + H not in [y for y in ys]:      # need contiguous H-ahead
                pass
            # require the +H year to exist in the (dense) series
            if (i + H) < len(ys) and ys[i + H] == ys[i] + H:
                g = (ps[i] - ps[i - 4]) / 4.0          # recent annual growth
                out.append({"p": ps[i], "g": g, "truth": ps[i + H],
                            "turning": abs(i - peak_i) <= 3, "rising": g > 0.02})
    return out


def _score(samples, predict):
    err = [abs(predict(s) - s["truth"]) for s in samples]
    return sum(err) / len(err) if err else float("nan")


def run():
    names = json.loads(Path(DATA).read_text())
    keys = sorted(names)
    cut = int(0.6 * len(keys))
    train = {k: names[k] for k in keys[:cut]}
    test = {k: names[k] for k in keys[cut:]}
    tr, te = _samples(train), _samples(test)

    persistence = lambda s: s["p"]
    trend = lambda s: max(0.0, s["p"] + s["g"] * H)

    # fit contagion (rho, lambda) on TRAIN
    best = None
    for rho in (0.2, 0.4, 0.6, 0.8, 0.95):
        for lam in (0.02, 0.05, 0.1, 0.2, 0.35):
            mae = _score(tr, lambda s, r=rho, l=lam: _contagion_roll(s["p"], s["g"], r, l, H))
            if best is None or mae < best[0]:
                best = (mae, rho, lam)
    _, rho, lam = best
    contagion = lambda s: _contagion_roll(s["p"], s["g"], rho, lam, H)

    def block(samples, label):
        return {"label": label, "n": len(samples),
                "persistence": round(_score(samples, persistence), 4),
                "trend": round(_score(samples, trend), 4),
                "contagion": round(_score(samples, contagion), 4)}

    overall = block(te, "ALL test points")
    turning = block([s for s in te if s["turning"]], "TURNING POINTS (near peak)")
    rising = block([s for s in te if s["rising"]], "RISING (g>0.02%/yr)")
    stable = block([s for s in te if not s["turning"] and not s["rising"]], "STABLE")

    def skill(b):
        base = min(b["persistence"], b["trend"])
        return round((base - b["contagion"]) / base, 4) if base else 0.0

    out = {"data": "SSA-derived baby-name shares, 481 names 1880-2008; forecast H=10yr ahead, leakage-free",
           "contagion_params": {"rho": rho, "lambda": lam}, "horizon_years": H,
           "ALL": overall, "TURNING_POINTS": turning, "RISING": rising, "STABLE": stable,
           "contagion_skill_vs_best_simple": {"all": skill(overall), "turning_points": skill(turning),
                                              "rising": skill(rising), "stable": skill(stable)},
           "verdict": ("CONTAGION (the coupled dynamic) beats persistence AND trend at the turning points"
                       if turning["contagion"] < min(turning["persistence"], turning["trend"]) else
                       "contagion does not clear the simple baselines even at turning points")}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-072  real contagion/tipping test — baby-name fashion cascades (481 names, H=10yr)")
    print(f"  contagion (coupled bandwagon+fatigue) params: rho={rho}, lambda={lam}")
    for b in (overall, rising, turning, stable):
        best_simple = min(b["persistence"], b["trend"])
        win = "CONTAGION WINS" if b["contagion"] < best_simple else "simple baseline wins"
        print(f"  {b['label']:28s} (n={b['n']:4d}) MAE  persist={b['persistence']:.3f}  "
              f"trend={b['trend']:.3f}  contagion={b['contagion']:.3f}   -> {win}")
    sk = out["contagion_skill_vs_best_simple"]
    print(f"  contagion skill vs best simple baseline: all {sk['all']:+}, turning {sk['turning_points']:+}, "
          f"rising {sk['rising']:+}, stable {sk['stable']:+}")
    print(f"  VERDICT: {out['verdict']}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
