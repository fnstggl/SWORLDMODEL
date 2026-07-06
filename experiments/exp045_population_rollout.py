"""EXP-045: multi-step population-over-time rollout — can a GROUNDED population beat persistence?

The untested axis of the thesis. EXP-033 showed a market's belief is a near-martingale — nothing beats
"it stays put" over multiple steps. EXP-042 tested a single event step. This tests the real claim:
simulate a POPULATION FORWARD over years and predict opinion CHANGE, on real longitudinal data (GSS,
1972-2024, 15 attitude items, individual demographics each wave).

The grounded forward simulation predicts the *compositional* component of opinion change: a population's
demographics shift over time (cohort replacement, aging, rising education), and each demographic cell
carries a stance. So
    Ŝ_grounded(t) = S(last) + [ ĝ(demographics at t) − ĝ(demographics at last) ]
where ĝ(·) is a correlation-aware demographic→attitude model fit on all PRIOR years, applied to a given
year's real demographic composition. It adds the demographic-driven change onto the persistence level —
a true bottom-up population roll-forward. (Future demographics are near-deterministic — cohorts age, the
census projects them — so using year-t composition is fair; only ATTITUDES are held out.)

Rolling-origin, leakage-free: for each item and each test year t, fit only on years < t; predict S(t);
compare to the actual held-out share. Baselines: persistence S(last), and a linear trend extrapolation.
Reports MAE by method and by horizon (years ahead), plus directional accuracy of the change.
Run: python -m experiments.exp045_population_rollout
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.datasets_gss import load
from swm.transition.readout import LogisticReadout
from swm.variables.pooled_readout import encode, onehot_vocab

RESULT = "experiments/results/exp045_population_rollout.json"
ATTRS = ["age", "sex", "race", "region", "degree", "party", "ideology", "relig", "attendance", "marital"]
MIN_TRAIN_YEARS = 4        # need enough history to fit the grounded model + a trend
MAX_FIT_ROWS = 6000        # cap rows per grounded fit for speed (sampled deterministically)


def _share(rows, item):
    ys = [r["answers"][item] for r in rows if item in r["answers"]]
    return (sum(ys) / len(ys)) if ys else None


def _grounded_model(train_rows, item):
    rows = [r for r in train_rows if item in r["answers"]]
    if len(rows) > MAX_FIT_ROWS:
        rows = rows[:: max(1, len(rows) // MAX_FIT_ROWS)][:MAX_FIT_ROWS]
    y = [r["answers"][item] for r in rows]
    if len(set(y)) < 2:
        return None, None
    vocab = onehot_vocab(rows, ATTRS)
    X = [encode(r["demo"], ATTRS, vocab) for r in rows]
    return LogisticReadout(l2=1.0, epochs=120).fit(X, y), vocab


def _predict_share(model, vocab, year_rows):
    """Aggregate the grounded model over a year's real demographic composition -> predicted share."""
    if model is None:
        return None
    ps = [model.predict_proba(encode(r["demo"], ATTRS, vocab)) for r in year_rows]
    return sum(ps) / len(ps) if ps else None


def run():
    recs = load()
    by_year = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r)
    years = sorted(by_year)
    items = ["cappun", "gunlaw", "grass", "abany", "letdie1", "fepol", "fefam", "homosex", "premarsx",
             "natheal", "natenvir", "natfare", "natcrime", "nateduc", "natrace"]

    err = {m: defaultdict(list) for m in ("persistence", "linear_trend", "grounded_forward")}
    dir_hit = {m: [0, 0] for m in ("persistence", "linear_trend", "grounded_forward")}
    n_eval = 0

    for item in items:
        item_years = [y for y in years if _share(by_year[y], item) is not None]
        shares = {y: _share(by_year[y], item) for y in item_years}
        for ti, t in enumerate(item_years):
            train_years = [y for y in item_years if y < t]
            if len(train_years) < MIN_TRAIN_YEARS:
                continue
            last = train_years[-1]
            horizon = t - last
            actual = shares[t]
            # baselines
            p_persist = shares[last]
            # linear trend over train years
            xs = train_years; ysh = [shares[y] for y in xs]
            mx = sum(xs) / len(xs); my = sum(ysh) / len(ysh)
            den = sum((x - mx) ** 2 for x in xs)
            slope = (sum((x - mx) * (yv - my) for x, yv in zip(xs, ysh)) / den) if den > 1e-9 else 0.0
            p_trend = min(1.0, max(0.0, my + slope * (t - mx)))
            # grounded forward: persistence level + demographic-composition-driven change
            train_rows = [r for y in train_years for r in by_year[y]]
            model, vocab = _grounded_model(train_rows, item)
            g_t = _predict_share(model, vocab, by_year[t])
            g_last = _predict_share(model, vocab, by_year[last])
            if g_t is None or g_last is None:
                continue
            p_grounded = min(1.0, max(0.0, p_persist + (g_t - g_last)))

            preds = {"persistence": p_persist, "linear_trend": p_trend, "grounded_forward": p_grounded}
            hb = "1-3y" if horizon <= 3 else ("4-7y" if horizon <= 7 else "8y+")
            for m, p in preds.items():
                err[m]["all"].append(abs(p - actual))
                err[m][hb].append(abs(p - actual))
                if abs(actual - p_persist) > 0.005:            # a real move to call
                    dir_hit[m][1] += 1
                    dir_hit[m][0] += int((p > p_persist) == (actual > p_persist))
            n_eval += 1

    def _mae(m, b):
        v = err[m][b]
        return round(sum(v) / len(v), 4) if v else None

    buckets = ["all", "1-3y", "4-7y", "8y+"]
    mae = {m: {b: _mae(m, b) for b in buckets} for m in err}
    dacc = {m: round(dir_hit[m][0] / dir_hit[m][1], 4) if dir_hit[m][1] else None for m in dir_hit}
    gf, pe = mae["grounded_forward"]["all"], mae["persistence"]["all"]
    out = {"dataset": "GSS", "n_items": len(items), "n_evaluations": n_eval,
           "mae_by_horizon": mae, "change_directional_accuracy": dacc,
           "grounded_beats_persistence": gf < pe, "grounded_vs_persistence_mae": round(pe - gf, 4)}

    print(f"EXP-045 population-over-time rollout (GSS) — {len(items)} items, {n_eval} rolling-origin forecasts")
    print(f"  MAE of predicted population share (lower better), by horizon:")
    print(f"    {'method':<18}{'all':>9}{'1-3y':>9}{'4-7y':>9}{'8y+':>9}")
    for m in ("persistence", "linear_trend", "grounded_forward"):
        print(f"    {m:<18}" + "".join(f"{mae[m][b]!s:>9}" for b in buckets))
    print(f"  directional accuracy of the CHANGE vs persistence-anchor: "
          + "  ".join(f"{m} {dacc[m]}" for m in dacc))
    print(f"  -> grounded forward beats persistence (all-horizon MAE): {out['grounded_beats_persistence']} "
          f"(Δ {out['grounded_vs_persistence_mae']:+.4f})")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
