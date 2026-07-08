"""EXP-073: the historical-event no-cheat backtest — does the CALIBRATED multi-variable simulation beat the
baselines, and does MORE fidelity buy more skill?

The decisive test of the vision, on the one longitudinal case where a forward simulation has beaten
persistence before (EXP-045, GSS opinion 1972–2024). Now run through the new machinery:
  - the forecaster is a grounded compositional rollout whose per-cell opinion model is `CalibratedWeights`
    (per-variable priors + empirical-Bayes shrinkage + integrated weight uncertainty);
  - scored by the new `event_backtest` harness (SKILL vs persistence / linear-trend / base-rate), no-cheat
    (train only on years ≤ the origin; predict future years);
  - run at TWO fidelities — FEW variables vs ALL demographic variables — to test the thesis directly: with
    proper calibration, does adding pressuring variables raise the skill?

Forecast (EXP-045 form, anchored on persistence so the model supplies the CHANGE, not the level):
    Ŝ(t) = S(origin) + [ ĝ(composition at t) − ĝ(composition at origin) ]
where ĝ is the calibrated demographic→opinion model fit on rows ≤ origin.

Run: python -m experiments.exp073_event_backtest
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.datasets_gss import load
from swm.eval.event_backtest import Question, backtest
from swm.variables.calibrated_weights import CalibratedWeights, uninformative_prior

RESULT = "experiments/results/exp073_event_backtest.json"
ITEMS = ["cappun", "gunlaw", "grass", "abany", "homosex", "premarsx", "fefam", "fepol", "natheal",
         "natenvir", "natfare", "nateduc", "natrace", "natcrime", "letdie1"]
ATTRS_FULL = ["age", "sex", "race", "region", "degree", "party", "ideology", "relig", "attendance",
              "marital", "income"]
ATTRS_FEW = ["party", "age"]
ORIGIN = 2006
TRAIN_CAP = 1600


def _item_rows(rows, item):
    out = []
    for r in rows:
        a = dict(r["answers"])
        if item in a:
            out.append({"year": r["year"], "demo": r["demo"], "y": int(a[item])})
    return out


def _share(rows):
    ys = [r["y"] for r in rows]
    return sum(ys) / len(ys) if ys else None


def _vocab(rows, attrs):
    v = {}
    for r in rows:
        for a in attrs:
            k = (a, r["demo"].get(a, "unknown"))
            v.setdefault(k, len(v))
    return v


def _encode(demo, attrs, vocab):
    x = [0.0] * len(vocab)
    for a in attrs:
        j = vocab.get((a, demo.get(a, "unknown")))
        if j is not None:
            x[j] = 1.0
    return x


def _fit_g(train, attrs, seed=0):
    if len(train) > TRAIN_CAP:
        train = random.Random(seed).sample(train, TRAIN_CAP)
    vocab = _vocab(train, attrs)
    X = [_encode(r["demo"], attrs, vocab) for r in train]
    y = [r["y"] for r in train]
    priors = [uninformative_prior(f"{a}={lv}") for (a, lv) in vocab]
    g = CalibratedWeights(priors, temper_grid=(1.0, 4.0), epochs=60, eb_epochs=60).fit(X, y, tune=True, seed=seed)
    return g, vocab


def _pred_share(g, vocab, attrs, year_rows):
    if not year_rows:
        return None
    return sum(g.predict(_encode(r["demo"], attrs, vocab)) for r in year_rows) / len(year_rows)


def _forecasts_for(rows_by_item, attrs, items=None):
    """Return {qid: forecast} and the Question list (baselines) for a given fidelity."""
    forecasts, questions = {}, []
    for item in (items if items is not None else ITEMS):
        irows = rows_by_item[item]
        by_year = {}
        for r in irows:
            by_year.setdefault(r["year"], []).append(r)
        years = sorted(by_year)
        origins = [y for y in years if y <= ORIGIN]
        if not origins:
            continue
        o = origins[-1]
        train = [r for r in irows if r["year"] <= o]
        s_o = _share(by_year[o])
        if s_o is None or len(set(r["y"] for r in train)) < 2:
            continue
        base = sum(r["y"] for r in train) / len(train)
        # linear trend over the training years
        ty = [y for y in years if y <= o]
        ts = [_share(by_year[y]) for y in ty]
        slope = 0.0
        if len(ty) >= 2:
            mx = sum(ty) / len(ty); my = sum(ts) / len(ts)
            den = sum((x - mx) ** 2 for x in ty) or 1.0
            slope = sum((x - mx) * (s - my) for x, s in zip(ty, ts)) / den
        g, vocab = _fit_g(train, attrs, seed=hash(item) & 255)
        sp_o = _pred_share(g, vocab, attrs, by_year[o])
        for t in [y for y in years if y > o]:
            s_t = _share(by_year[t])
            sp_t = _pred_share(g, vocab, attrs, by_year[t])
            if s_t is None or sp_t is None:
                continue
            fc = min(1.0, max(0.0, s_o + (sp_t - sp_o)))
            qid = f"{item}@{t}"
            forecasts[qid] = fc
            questions.append(Question(qid, s_t, {"persistence": s_o, "base_rate": base,
                                                 "linear_trend": min(1.0, max(0.0, s_o + slope * (t - o)))},
                                      asof=str(o), resolved=str(t), meta={"horizon": t - o}))
    return forecasts, questions


def run() -> dict:
    rows = load()
    rows_by_item = {item: _item_rows(rows, item) for item in ITEMS}
    out = {}
    for label, attrs in (("few_vars", ATTRS_FEW), ("all_vars", ATTRS_FULL)):
        fc, qs = _forecasts_for(rows_by_item, attrs)
        card = backtest(qs, lambda q: fc[q.qid])
        out[label] = {"n_attrs": len(attrs), "scorecard": card}

    res = {"data": "GSS opinion 1972–2024 (rolling origin @%d, no-cheat)" % ORIGIN, "items": len(ITEMS),
           "few_vars": out["few_vars"], "all_vars": out["all_vars"],
           "thesis": {"few_skill_vs_persistence": out["few_vars"]["scorecard"]["skill_vs"].get("persistence"),
                      "all_skill_vs_persistence": out["all_vars"]["scorecard"]["skill_vs"].get("persistence"),
                      "more_variables_raised_skill":
                          out["all_vars"]["scorecard"]["skill_vs"].get("persistence", -9) >
                          out["few_vars"]["scorecard"]["skill_vs"].get("persistence", 9)}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-073  no-cheat event backtest (GSS opinion, rolling origin @%d)" % ORIGIN)
    for label in ("few_vars", "all_vars"):
        c = out[label]["scorecard"]
        print(f"  {label:9s} ({out[label]['n_attrs']:2d} vars): {c['n']} forecasts, MAE {c['model_loss']:.4f}  "
              f"| skill vs persistence {c['skill_vs'].get('persistence'):+.3f}  "
              f"trend {c['skill_vs'].get('linear_trend'):+.3f}  base {c['skill_vs'].get('base_rate'):+.3f}  "
              f"| beats all: {c['beats_all_baselines']}")
    t = res["thesis"]
    print(f"  -> more calibrated variables raised skill vs persistence: {t['more_variables_raised_skill']} "
          f"({t['few_skill_vs_persistence']:+.3f} → {t['all_skill_vs_persistence']:+.3f})")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
