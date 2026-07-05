"""EXP-023: population opinion prediction (GlobalOpinionQA) via inferred country value-variables.

The aggregate/population regime, on a standard benchmark comparable to the best social-simulation
work. Test the VariableMap thesis at population scale: does conditioning on LLM-INFERRED country
value-variables (religiosity, individualism, traditional↔secular, … — inferred from world knowledge,
NOT from this survey, so non-circular) predict a country's opinion distribution better than the
cross-country global mean?

No-cheat / no-leakage design: split COUNTRIES into train/test. For each TEST (unseen) country and each
question, predict its answer distribution using ONLY train countries' distributions on that question,
weighted by value-similarity to the test country. The test country's own opinions are never used —
the only thing tying it to the prediction is its inferred value profile.

  base    : global mean = unweighted mean of train countries' distributions on the question
  values  : value-similarity weighted mean (softmax over cosine similarity of value profiles)
  hybrid  : mix of the two (shrink the value model toward the global mean)

Metric: cross-entropy CE(actual || pred) = −Σ_o actual[o]·log pred[o], averaged over test
(country, question) pairs (lower is better); also mean total-variation distance.

Run: python -m experiments.exp023_global_opinion
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from experiments.datasets_globalopinion import load

# inferred country value-profiles: prefer the fresh data/ copy, fall back to the committed artifact
VALUES = "data/go_country_values.json"
VALUES_COMMITTED = "experiments/results/exp023_country_values.json"
RESULT = "experiments/results/exp023_global_opinion.json"
VDIMS = ["religiosity", "traditionalism", "individualism", "trust_institutions", "openness_change",
         "national_pride", "economic_left", "social_progressive", "hierarchy_respect",
         "survival_vs_selfexpression"]


def _vec(v):
    return [float(v.get(d, 0.5)) for d in VDIMS]


def _standardize(vals):
    keys = list(vals)
    M = [_vec(vals[k]) for k in keys]
    n, d = len(M), len(VDIMS)
    mu = [sum(M[i][j] for i in range(n)) / n for j in range(d)]
    sd = [max(1e-6, math.sqrt(sum((M[i][j] - mu[j]) ** 2 for i in range(n)) / n)) for j in range(d)]
    return {k: [(_vec(vals[k])[j] - mu[j]) / sd[j] for j in range(d)] for k in keys}


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _ce(actual, pred):
    return -sum(a * math.log(max(1e-9, p)) for a, p in zip(actual, pred))


def _tv(actual, pred):
    return 0.5 * sum(abs(a - p) for a, p in zip(actual, pred))


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1 / len(v)] * len(v)


def _entropy_floor(recs, valued):
    """Mean entropy of the actual valued-country distributions = irreducible CE (oracle that knows
    the country's own distribution). Contextualizes how much of the achievable gap values close."""
    ent, n = 0.0, 0
    for r in recs:
        for c, p in r["dists"].items():
            if c in valued:
                pp = _norm(p)
                ent += -sum(x * math.log(max(1e-9, x)) for x in pp); n += 1
    return ent / max(1, n)


def _evaluate(recs, sv, context_c, eval_c, beta, hybrid_w):
    """Predict eval_c countries from context_c countries only. Returns per-tier CE/TV metrics."""
    metrics = {m: {"ce": 0.0, "tv": 0.0, "n": 0} for m in ("base", "values", "hybrid")}
    for r in recs:
        d = r["dists"]
        tr = [(c, d[c]) for c in d if c in context_c]
        if len(tr) < 3:
            continue
        nopt = len(r["options"])
        gmean = _norm([sum(p[o] for _, p in tr) / len(tr) for o in range(nopt)])
        for c in d:
            if c not in eval_c or c not in sv:
                continue
            actual = _norm(d[c])
            sims = [_cos(sv[c], sv[cc]) for cc, _ in tr]
            mx = max(sims)
            w = [math.exp(beta * (s - mx)) for s in sims]
            sw = sum(w) or 1.0
            vpred = _norm([sum(w[i] * tr[i][1][o] for i in range(len(tr))) / sw for o in range(nopt)])
            hpred = _norm([hybrid_w * vpred[o] + (1 - hybrid_w) * gmean[o] for o in range(nopt)])
            for name, pred in (("base", gmean), ("values", vpred), ("hybrid", hpred)):
                metrics[name]["ce"] += _ce(actual, pred); metrics[name]["tv"] += _tv(actual, pred)
                metrics[name]["n"] += 1
    return metrics


def run(seed=0):
    recs = load()
    vpath = VALUES if Path(VALUES).exists() else VALUES_COMMITTED
    vals = json.loads(Path(vpath).read_text())
    countries = set(vals)
    sv = _standardize(vals)
    rng = random.Random(seed)
    clist = sorted(countries)
    rng.shuffle(clist)
    cut = int(0.7 * len(clist))
    train_c, test_c = set(clist[:cut]), set(clist[cut:])
    # tune beta, hybrid_w on a val split of TRAIN (fit predicts val), never on test
    tl = sorted(train_c); rng.shuffle(tl); fcut = int(0.7 * len(tl))
    fit_c, val_c = set(tl[:fcut]), set(tl[fcut:])
    best, best_ce = (6.0, 0.5), 1e9
    for beta in (2, 4, 6, 10, 16):
        for hw in (0.3, 0.5, 0.7, 1.0):
            m = _evaluate(recs, sv, fit_c, val_c, beta, hw)
            ce = m["hybrid"]["ce"] / max(1, m["hybrid"]["n"])
            if ce < best_ce:
                best_ce, best = ce, (beta, hw)
    beta, hybrid_w = best
    metrics = _evaluate(recs, sv, train_c, test_c, beta, hybrid_w)   # final: train predicts test
    out = {"n_questions": len(recs), "n_countries": len(countries), "n_test_countries": len(test_c),
           "beta": beta, "hybrid_w": hybrid_w, "tuned_on": "train/val split (no test leakage)", "tiers": {}}
    for m, v in metrics.items():
        out["tiers"][m] = {"cross_entropy": round(v["ce"] / v["n"], 4),
                           "total_variation": round(v["tv"] / v["n"], 4), "n_pairs": v["n"]}
    print(f"GlobalOpinionQA: {len(recs)} questions, {len(countries)} valued countries, "
          f"test-country pairs {metrics['base']['n']}")
    for m in ("base", "values", "hybrid"):
        t = out["tiers"][m]
        print(f"  {m:<8} cross-entropy {t['cross_entropy']}  total-variation {t['total_variation']}")
    g = out["tiers"]["base"]["cross_entropy"] - out["tiers"]["hybrid"]["cross_entropy"]
    out["values_gain_ce"] = round(g, 4)
    floor = _entropy_floor(recs, countries)
    base_ce = out["tiers"]["base"]["cross_entropy"]
    out["oracle_entropy_floor"] = round(floor, 4)
    out["gap_closed_frac"] = round(g / max(1e-9, base_ce - floor), 4)
    print(f"  Δ inferred-values (hybrid) vs global-mean: {g:+.4f} cross-entropy "
          f"({'values help' if g > 0 else 'no help'})")
    print(f"  oracle entropy floor {floor:.4f}; inferred values close "
          f"{out['gap_closed_frac'] * 100:.1f}% of the global-mean→oracle gap")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
