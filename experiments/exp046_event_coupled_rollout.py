"""EXP-046: event-coupled population rollout — compositional dynamics + gated period shocks.

EXP-045 forecast the COMPOSITIONAL component of opinion change (the population's demographic mix evolves,
each cell carries a stance) and beat persistence, but explicitly left out PERIOD effects — the
within-cohort swings that events cause (a court ruling, a war, an economic shock move everyone, not just
the new cohorts). This couples the two:

    S(t) = S(last) + Δcompositional(last→t) + Δperiod(last→t)

EXP-045 assumed Δperiod = 0. Here we forecast it. The period signal is the composition-removed residual
r(y) = S(y) − ĝ(demographics at y): everything the cross-sectional demographic model does NOT explain —
the aggregate footprint of events over time. We ask the honest question: is that residual FORECASTABLE
(does opinion have a low-frequency secular drift a damped velocity can project), and does distributing the
shock by each cell's RESPONSIVENESS (the EXP-042 operator — some groups move with the times more than
others) help beyond a uniform shock?

Arms (rolling-origin, leakage-free — fit only on years < t; period velocity from PAST residuals only):
  1. persistence                 — S(last) (the martingale)
  2. compositional (EXP-045)      — S(last) + [ĝ(t) − ĝ(last)]   (Δperiod = 0)
  3. coupled_uniform             — + a damped projection of the aggregate period velocity
  4. coupled_gated               — + the period velocity distributed by per-cell responsiveness
                                    (estimated globally, pooled across items), so the shock depends on
                                    WHICH cells are in the population at t (composition × event coupling)

Decisive: does adding forecastable period dynamics beat compositional-only, and does responsiveness-gating
beat a uniform shock? Reports MAE by horizon + change directional accuracy. Writes JSON.
Run: python -m experiments.exp046_event_coupled_rollout
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.datasets_gss import load
from experiments.exp045_population_rollout import ATTRS, _grounded_model, _predict_share, _share

RESULT = "experiments/results/exp046_event_coupled_rollout.json"
ITEMS = ["cappun", "gunlaw", "grass", "abany", "letdie1", "fepol", "fefam", "homosex", "premarsx",
         "natheal", "natenvir", "natfare", "natcrime", "nateduc", "natrace"]
MIN_TRAIN_YEARS = 4
DAMP = 0.5                 # period-velocity damping (anti-overshoot; EXP-045 saw raw trend overshoot)
RECENT = 8                 # window (years of data points) for the period-velocity estimate
GATE_ATTR = "age"          # cell axis for responsiveness gating (impressionable-years: younger move more)


def _slope(series):
    """OLS slope of (year, value) points."""
    if len(series) < 2:
        return 0.0
    xs = [x for x, _ in series]; ys = [y for _, y in series]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 1e-9 else 0.0


def _gate_weights(by_year, cutoff, years, _cache={}):
    """Global, pooled responsiveness of each GATE_ATTR group: how much its opinion drifts per year
    relative to the population average, averaged over items and years before `cutoff`. >1 = more
    responsive. Memoized by cutoff (depends only on the past, not the current item) — leakage-free."""
    if cutoff in _cache:
        return _cache[cutoff]
    train_years = [y for y in years if y < cutoff]
    grp_vel, all_vel = defaultdict(list), []
    for item in ITEMS:
        ys = [y for y in train_years if _share(by_year[y], item) is not None]
        if len(ys) < MIN_TRAIN_YEARS:
            continue
        agg = _slope([(y, _share(by_year[y], item)) for y in ys])
        all_vel.append(abs(agg))
        groups = {r["demo"].get(GATE_ATTR, "unknown") for y in ys for r in by_year[y]}
        for g in groups:
            gs = []
            for y in ys:
                rows = [r for r in by_year[y] if r["demo"].get(GATE_ATTR) == g and item in r["answers"]]
                if len(rows) >= 15:
                    gs.append((y, sum(r["answers"][item] for r in rows) / len(rows)))
            if len(gs) >= MIN_TRAIN_YEARS:
                grp_vel[g].append(abs(_slope(gs)))
    base = (sum(all_vel) / len(all_vel)) if all_vel else 1.0
    w = {g: max(0.3, min(2.5, (sum(v) / len(v)) / base)) if v and base > 1e-9 else 1.0
         for g, v in grp_vel.items()}
    _cache[cutoff] = w
    return w


def run():
    recs = load()
    by_year = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r)
    years = sorted(by_year)

    methods = ("persistence", "compositional", "coupled_uniform", "coupled_gated")
    err = {m: defaultdict(list) for m in methods}
    dir_hit = {m: [0, 0] for m in methods}
    n_eval = 0

    for item in ITEMS:
        item_years = [y for y in years if _share(by_year[y], item) is not None]
        shares = {y: _share(by_year[y], item) for y in item_years}
        for t in item_years:
            train_years = [y for y in item_years if y < t]
            if len(train_years) < MIN_TRAIN_YEARS:
                continue
            last = train_years[-1]; horizon = t - last; actual = shares[t]
            train_rows = [r for y in train_years for r in by_year[y]]
            model, vocab = _grounded_model(train_rows, item)
            g_t = _predict_share(model, vocab, by_year[t])
            g_last = _predict_share(model, vocab, by_year[last])
            if g_t is None or g_last is None:
                continue
            # period residual series (composition removed), from PAST years only
            resid = [(y, shares[y] - _predict_share(model, vocab, by_year[y])) for y in train_years]
            recent = resid[-RECENT:] if len(resid) > RECENT else resid
            v_period = _slope(recent)
            # gating: distribute the period velocity by the year-t composition's mean responsiveness
            gw = _gate_weights(by_year, t, years)
            comp = by_year[t]
            gate = sum(gw.get(r["demo"].get(GATE_ATTR, "unknown"), 1.0) for r in comp) / max(1, len(comp))

            compositional = shares[last] + (g_t - g_last)
            preds = {
                "persistence": shares[last],
                "compositional": compositional,
                "coupled_uniform": compositional + DAMP * v_period * horizon,
                "coupled_gated": compositional + DAMP * v_period * gate * horizon,
            }
            hb = "1-3y" if horizon <= 3 else ("4-7y" if horizon <= 7 else "8y+")
            for m, p in preds.items():
                p = min(1.0, max(0.0, p))
                err[m]["all"].append(abs(p - actual)); err[m][hb].append(abs(p - actual))
                if abs(actual - shares[last]) > 0.005:
                    dir_hit[m][1] += 1
                    dir_hit[m][0] += int((p > shares[last]) == (actual > shares[last]))
            n_eval += 1

    def _mae(m, b):
        v = err[m][b]; return round(sum(v) / len(v), 4) if v else None
    buckets = ["all", "1-3y", "4-7y"]
    mae = {m: {b: _mae(m, b) for b in buckets} for m in methods}
    dacc = {m: round(dir_hit[m][0] / dir_hit[m][1], 4) if dir_hit[m][1] else None for m in methods}
    cu = mae["coupled_uniform"]["all"]; cg = mae["coupled_gated"]["all"]; co = mae["compositional"]["all"]
    out = {"dataset": "GSS", "n_items": len(ITEMS), "n_evaluations": n_eval, "damping": DAMP,
           "mae_by_horizon": mae, "change_directional_accuracy": dacc,
           "period_helps_over_compositional": min(cu, cg) < co,
           "gating_helps_over_uniform": cg < cu,
           "best_method": min(methods, key=lambda m: mae[m]["all"])}

    print(f"EXP-046 event-coupled population rollout (GSS) — {len(ITEMS)} items, {n_eval} forecasts, damp={DAMP}")
    print(f"  MAE of predicted share (lower better), by horizon:")
    print(f"    {'method':<20}{'all':>9}{'1-3y':>9}{'4-7y':>9}")
    for m in methods:
        print(f"    {m:<20}" + "".join(f"{mae[m][b]!s:>9}" for b in buckets))
    print("  change directional accuracy: " + "  ".join(f"{m} {dacc[m]}" for m in methods))
    print(f"  -> period dynamics help over compositional-only: {out['period_helps_over_compositional']}; "
          f"responsiveness-gating helps over uniform: {out['gating_helps_over_uniform']}")
    print(f"  -> best method: {out['best_method']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
