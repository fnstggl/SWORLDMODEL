"""EXP-028: per-INDIVIDUAL opinion prediction on OpinionQA (the SOTA benchmark's own task).

Generative Agent Simulations of 1,000 People predict each person's individual survey answers. This runs
the same task on the standard public benchmark — OpinionQA / Pew ATP — through our architecture: map a
respondent's demographics to the latent VALUE-VARIABLES (religiosity, traditionalism, individualism, …,
the EXP-023 value dims), then predict their answer to a question by value-similarity to OTHER people who
answered it. No-cheat: split RESPONDENTS train/test; a test respondent is a person the model never saw,
and their own answers are never used — only their inferred value profile ties them to the prediction.

Tiers (predict a held-out person's answer distribution for each question they answered):
  marginal          : the population answer distribution for that question (the standard OpinionQA
                      baseline — "the average American").
  demographic_knn   : similarity-weighted vote over train respondents, similarity on RAW one-hot
                      demographics (does knowing the person help at all?).
  value_similarity  : similarity-weighted vote, similarity on the 10 inferred VALUE-VARIABLES (does the
                      architecture's value abstraction retain/improve the signal at 10 interpretable
                      dims vs raw demographics?).

Metric: accuracy (argmax = the person's actual choice) and log loss over answer options, over held-out
(respondent, question) pairs. beta (softmax temperature) tuned on a train/val split of TRAIN only.
Writes experiments/results/exp028_individual_opinion.json. Run: python -m experiments.exp028_individual_opinion
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from swm.eval.metrics import log_loss
from swm.variables.demographic_values import VALUE_DIMS, value_vector
from experiments.datasets_opinionqa import load

RESULT = "experiments/results/exp028_individual_opinion.json"
_DEMO_FIELDS = ["race", "region", "age", "sex", "education", "marital", "religion", "attendance",
                "party", "ideology", "income", "citizen"]


def _onehot_space(recs):
    vals = defaultdict(set)
    for r in recs:
        for f in _DEMO_FIELDS:
            vals[f].add(r["demo"].get(f, "unknown"))
    cols = [(f, v) for f in _DEMO_FIELDS for v in sorted(vals[f])]
    idx = {c: i for i, c in enumerate(cols)}
    return idx


def _onehot(demo, idx):
    v = [0.0] * len(idx)
    for f in _DEMO_FIELDS:
        c = (f, demo.get(f, "unknown"))
        if c in idx:
            v[idx[c]] = 1.0
    return v


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def _predict_weighted(qtrain, vec, veckey, beta, nopt):
    """Softmax(sim)-weighted vote over train respondents who answered this question."""
    sims = [(_cos(vec, t[veckey]), t["ans"]) for t in qtrain]
    mx = max(s for s, _ in sims)
    w = [math.exp(beta * (s - mx)) for s, _ in sims]
    agg = [0.0] * nopt
    for wi, (_, ans) in zip(w, sims):
        agg[ans] += wi
    return _norm(agg)


def _marginal(qtrain, nopt):
    agg = [1.0] * nopt                     # Laplace smoothing
    for t in qtrain:
        agg[t["ans"]] += 1
    return _norm(agg)


def _evaluate(by_q, test, beta, tiers=("marginal", "demographic_knn", "value_similarity")):
    acc = {t: 0 for t in tiers}; ll = {t: 0.0 for t in tiers}; n = 0
    for r in test:
        qt = by_q.get(r["qid"])
        if not qt or len(qt) < 5:
            continue
        nopt = r["n_opt"]; a = r["answer_idx"]
        preds = {}
        if "marginal" in tiers:
            preds["marginal"] = _marginal(qt, nopt)
        if "demographic_knn" in tiers:
            preds["demographic_knn"] = _predict_weighted(qt, r["_demo"], "demo", beta, nopt)
        if "value_similarity" in tiers:
            preds["value_similarity"] = _predict_weighted(qt, r["_val"], "val", beta, nopt)
        for t in tiers:
            p = preds[t]
            acc[t] += int(max(range(nopt), key=lambda i: p[i]) == a)
            ll[t] += -math.log(max(1e-9, p[a]))
        n += 1
    return {t: {"acc": round(acc[t] / n, 4), "log_loss": round(ll[t] / n, 4)} for t in tiers}, n


def run(seed=0):
    recs = load()
    idx = _onehot_space(recs)
    for r in recs:
        r["_val"] = value_vector(r["demo"])
        r["_demo"] = _onehot(r["demo"], idx)
    uids = sorted({r["uid"] for r in recs})
    rng = random.Random(seed); rng.shuffle(uids)
    cut = int(0.7 * len(uids)); train_u, test_u = set(uids[:cut]), set(uids[cut:])
    train = [r for r in recs if r["uid"] in train_u]
    test = [r for r in recs if r["uid"] in test_u]
    by_q = defaultdict(list)
    for r in train:
        by_q[r["qid"]].append({"val": r["_val"], "demo": r["_demo"], "ans": r["answer_idx"]})

    # tune beta on a val split of TRAIN (no test leakage)
    tl = sorted(train_u); rng.shuffle(tl); fc = int(0.7 * len(tl))
    fit_u, val_u = set(tl[:fc]), set(tl[fc:])
    by_q_fit = defaultdict(list)
    for r in train:
        if r["uid"] in fit_u:
            by_q_fit[r["qid"]].append({"val": r["_val"], "demo": r["_demo"], "ans": r["answer_idx"]})
    val_rows = [r for r in train if r["uid"] in val_u]
    best_beta, best_ll = 8.0, 1e9
    for beta in (4, 8, 12, 20, 30):
        m, _ = _evaluate(by_q_fit, val_rows, beta, tiers=("value_similarity",))
        if m["value_similarity"]["log_loss"] < best_ll:
            best_ll, best_beta = m["value_similarity"]["log_loss"], beta

    tiers, n = _evaluate(by_q, test, best_beta)
    out = {"n_respondents": len(uids), "n_questions": len(by_q), "n_test_pairs": n,
           "beta": best_beta, "tuned_on": "train/val split (no test leakage)", "tiers": tiers}
    print(f"EXP-028 individual opinion (OpinionQA/Pew ATP) — {len(uids)} respondents, {len(by_q)} questions, "
          f"{n} held-out (person,question) pairs; beta {best_beta}")
    for t in ("marginal", "demographic_knn", "value_similarity"):
        print(f"  {t:<18} accuracy {tiers[t]['acc']}  log loss {tiers[t]['log_loss']}")
    ga = round(tiers["value_similarity"]["acc"] - tiers["marginal"]["acc"], 4)
    gl = round(tiers["marginal"]["log_loss"] - tiers["value_similarity"]["log_loss"], 4)
    out["value_vs_marginal_acc"] = ga; out["value_vs_marginal_logloss_gain"] = gl
    print(f"  Δ value_similarity vs marginal (population): {ga:+.4f} accuracy, {gl:+.4f} log loss "
          f"({'individual value modeling helps' if gl > 0 else 'no help'})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
