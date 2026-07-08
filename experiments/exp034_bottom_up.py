"""EXP-034: bottom-up simulate-and-aggregate vs top-down — does simulating individuals beat one number?

The core hypothesis: to predict a GROUP's collective opinion, is it better to simulate each heterogeneous
individual (from their VariableMap / value profile) and AGGREGATE, or to model the aggregate directly?
Park et al.'s generative-agents bet on the former. We test it no-cheat on OpinionQA.

For each demographic GROUP and question, predict the group's answer distribution three ways:
  top_down_global : the population marginal (the "average person" — ignores who's in the group)
  top_down_group  : the group's demographic-cell marginal from TRAIN (a group-level aggregate)
  bottom_up       : simulate each group member individually (value-similarity prediction from OTHER
                    people, EXP-028) and average their predicted distributions
Compared to the group's TRUE distribution (its members' actual answers). No-cheat: split respondents;
individual predictions use only TRAIN respondents; the group's own answers are never used to predict.

Metric: mean total-variation distance to the true group distribution (lower is better), overall and on
the most DISTINCTIVE groups (those whose true opinion is farthest from the global average — where
simulating who's in the group should matter most). Writes JSON. Run: python -m experiments.exp034_bottom_up
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from swm.variables.demographic_values import value_vector
from experiments.datasets_opinionqa import load

RESULT = "experiments/results/exp034_bottom_up.json"
GROUP_KEYS = ["ideology", "party", "religion", "age"]     # group respondents by these demographic cells


def _cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return d / (na * nb)


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def _tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def _dist(answers, nopt):
    v = [0.0] * nopt
    for a in answers:
        v[a] += 1
    return _norm(v)


def run(seed=0):
    recs = load()
    for r in recs:
        r["_val"] = value_vector(r["demo"])
    uids = sorted({r["uid"] for r in recs})
    rng = random.Random(seed); rng.shuffle(uids)
    cut = int(0.7 * len(uids)); train_u, test_u = set(uids[:cut]), set(uids[cut:])
    train = [r for r in recs if r["uid"] in train_u]
    test = [r for r in recs if r["uid"] in test_u]

    by_q = defaultdict(list)                       # qid -> [(val_vec, ans)] train
    q_marginal = {}                                # qid -> global train distribution
    q_nopt = {}
    tmp = defaultdict(list)
    for r in train:
        by_q[r["qid"]].append((r["_val"], r["answer_idx"]))
        tmp[r["qid"]].append(r["answer_idx"]); q_nopt[r["qid"]] = r["n_opt"]
    for q, ans in tmp.items():
        q_marginal[q] = _dist(ans, q_nopt[q])

    # group-cell marginals from TRAIN (top_down_group)
    cell_marg = defaultdict(lambda: defaultdict(list))     # (key,val) -> qid -> [ans]
    for r in train:
        for k in GROUP_KEYS:
            cell_marg[(k, r["demo"].get(k, "unknown"))][r["qid"]].append(r["answer_idx"])

    beta = 8
    def predict_individual(vec, qid, nopt):
        nb = by_q.get(qid)
        if not nb or len(nb) < 5:
            return None
        sims = [(_cos(vec, v), a) for v, a in nb]
        mx = max(s for s, _ in sims)
        w = [math.exp(beta * (s - mx)) for s, _ in sims]
        agg = [0.0] * nopt
        for wi, (_, a) in zip(w, sims):
            agg[a] += wi
        return _norm(agg)

    # group TEST respondents by each cell; predict each (group,question) distribution
    groups = defaultdict(list)                     # (key,val) -> [test rows]
    for r in test:
        for k in GROUP_KEYS:
            groups[(k, r["demo"].get(k, "unknown"))].append(r)

    tv = {"top_down_global": [], "top_down_group": [], "bottom_up": []}
    distinctive = {"top_down_global": [], "bottom_up": []}
    for gkey, rows in groups.items():
        byq = defaultdict(list)
        for r in rows:
            byq[r["qid"]].append(r)
        for qid, grows in byq.items():
            if len(grows) < 8 or qid not in q_marginal:
                continue
            nopt = grows[0]["n_opt"]
            true = _dist([r["answer_idx"] for r in grows], nopt)
            gm = q_marginal[qid]
            # bottom-up: average individual predictions of the group members
            preds = [p for r in grows if (p := predict_individual(r["_val"], qid, nopt))]
            if not preds:
                continue
            bu = _norm([sum(p[o] for p in preds) / len(preds) for o in range(nopt)])
            # top-down group cell marginal
            cm = cell_marg[gkey].get(qid)
            gdist = _dist(cm, nopt) if cm and len(cm) >= 5 else gm
            tv["top_down_global"].append(_tv(true, gm))
            tv["top_down_group"].append(_tv(true, gdist))
            tv["bottom_up"].append(_tv(true, bu))
            if _tv(true, gm) >= 0.10:               # distinctive group-question (far from global avg)
                distinctive["top_down_global"].append(_tv(true, gm))
                distinctive["bottom_up"].append(_tv(true, bu))

    out = {"n_group_questions": len(tv["bottom_up"]), "seed": seed,
           "mean_tv": {k: round(statistics.mean(v), 4) for k, v in tv.items()},
           "distinctive_n": len(distinctive["bottom_up"]),
           "distinctive_mean_tv": {k: round(statistics.mean(v), 4) for k, v in distinctive.items()},
           "bottom_up_beats_global": statistics.mean(tv["bottom_up"]) < statistics.mean(tv["top_down_global"]),
           "bottom_up_beats_group_marginal": statistics.mean(tv["bottom_up"]) < statistics.mean(tv["top_down_group"])}
    print(f"EXP-034 bottom-up vs top-down — OpinionQA, {out['n_group_questions']} (group,question) pairs")
    print("  mean total-variation to the true group distribution (lower better):")
    for k, v in out["mean_tv"].items():
        print(f"    {k:<18} {v}")
    print(f"  on DISTINCTIVE groups (n={out['distinctive_n']}, true far from global avg):")
    for k, v in out["distinctive_mean_tv"].items():
        print(f"    {k:<18} {v}")
    print(f"  bottom-up beats global marginal: {out['bottom_up_beats_global']}; "
          f"beats group-cell marginal: {out['bottom_up_beats_group_marginal']}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
