"""EXP-040: does grounded-variable simulation beat the composite? (the north-star test)

The whole thesis in one experiment: predict a real SOCIAL OUTCOME (how a population answers an opinion
question) two ways —
  - the COMPOSITE / crowd: model the aggregate directly (the question's marginal answer rate — "the
    average person"), the analog of reading a market's single number;
  - GROUNDED SIMULATION: map each individual's REAL variables (their actual demographics + attitudes —
    party, ideology, religion, age, income, ...) and simulate their answer, then aggregate.

and ask the two questions the north star hinges on:
  Q1. Does simulating individuals from their real variables beat the crowd composite? (individual answer
      log-loss / accuracy — the per-person simulation's quality.)
  Q2. Does adding MORE real variables monotonically improve the simulation? (the "map ALL the variables"
      claim — richness levels: marginal -> +party -> +party,ideology,religion -> all 11.)
  Q3. Does the grounded bottom-up composition beat the top-down aggregate on DISTINCTIVE subgroups —
      where who-is-in-the-group matters — as EXP-034 saw, now with real per-person variables not
      value-similarity? (aggregate TV to the true subgroup share.)

No-cheat: respondents split train/test by a stable hash; the per-person model is fit on TRAIN
respondents' (attributes -> answer) only; TEST respondents' answers are never seen when predicting. This
is the honest test of whether mapping+simulating real variables beats modeling the aggregate — with real
variables and a real outcome, not a market price.
Run: python -m experiments.exp040_grounded_simulation
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.datasets_opinionqa import load

RESULT = "experiments/results/exp040_grounded_simulation.json"
ATTRS = ["party", "ideology", "religion", "attendance", "age", "education", "income",
         "race", "sex", "region", "marital"]
LEVELS = {                              # variable-richness ablation (the "map more variables" axis)
    "marginal (composite)": [],
    "+party": ["party"],
    "+party,ideology,religion": ["party", "ideology", "religion"],
    "all 11 variables": ATTRS,
}
ALPHA = 4.0                             # NB smoothing pseudocount -> shrink toward attribute-independence


def _split(recs, test_frac=0.3, salt=0):
    tr, te = [], []
    for r in recs:
        h = (hash((salt, r["uid"])) % 1000) / 1000.0
        (te if h < test_frac else tr).append(r)
    return tr, te


class _QModel:
    """Shrinkage naive Bayes for ONE question: P(answer | person's real variables), fit on train rows.
    Falls back to the question marginal when a variable carries no signal (Laplace shrink to independence),
    so adding an uninformative variable does no harm — the honest "map more variables" behavior."""

    def __init__(self, rows, n_opt):
        self.n_opt = n_opt
        self.n = len(rows)
        self.count_a = [0.0] * n_opt
        self.count_la = defaultdict(lambda: [0.0] * n_opt)   # (attr,level) -> per-answer counts
        self.count_l = defaultdict(float)                    # (attr,level) -> total
        for r in rows:
            a = r["answer_idx"]
            self.count_a[a] += 1
            for attr in ATTRS:
                key = (attr, r["demo"].get(attr, "unknown"))
                self.count_la[key][a] += 1
                self.count_l[key] += 1
        tot = max(1.0, sum(self.count_a))
        self.p_a = [(c + 1.0) / (tot + n_opt) for c in self.count_a]     # smoothed marginal (composite)

    def predict(self, demo, use_attrs, alpha=ALPHA):
        logp = [math.log(pa) for pa in self.p_a]
        for attr in use_attrs:
            key = (attr, demo.get(attr, "unknown"))
            tot_l = self.count_l.get(key, 0.0)
            for a in range(self.n_opt):
                # P(level | answer=a) shrunk toward P(level) (independence) by `alpha` pseudocounts.
                # Heavier shrink protects against thin/correlated variables double-counting (the reason
                # naive all-11 overfits ~38 respondents/question — party and ideology are collinear).
                p_l = (tot_l + 1.0) / (self.n + self.n_opt)
                p_la = (self.count_la[key][a] + alpha * p_l) / (self.count_a[a] + alpha)
                logp[a] += math.log(p_la) - math.log(p_l)
        m = max(logp)
        ex = [math.exp(z - m) for z in logp]
        s = sum(ex)
        return [e / s for e in ex]


def _tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _eval_ll(models, rows, use, alpha):
    ll, correct, tot = 0.0, 0, 0
    for r in rows:
        p = models[r["qid"]].predict(r["demo"], use, alpha)
        ll += -math.log(min(1 - 1e-9, max(1e-9, p[r["answer_idx"]])))
        correct += int(max(range(len(p)), key=lambda i: p[i]) == r["answer_idx"])
        tot += 1
    return ll / max(1, tot), correct / max(1, tot), tot


def run():
    recs = load()
    tr, te = _split(recs)
    by_q_tr = defaultdict(list)
    for r in tr:
        by_q_tr[r["qid"]].append(r)
    n_opt = {r["qid"]: r["n_opt"] for r in recs}
    models = {q: _QModel(rows, n_opt[q]) for q, rows in by_q_tr.items() if len(rows) >= 12}
    te = [r for r in te if r["qid"] in models]

    # tune shrinkage on a TRAIN-internal holdout (leakage-free: never touches test) — the honest way to
    # let many variables help without overfitting. This is the estimation quality the thesis really needs.
    tr_fit, tr_val = _split(tr, test_frac=0.3, salt=1)
    by_qf = defaultdict(list)
    for r in tr_fit:
        by_qf[r["qid"]].append(r)
    fit_models = {q: _QModel(rows, n_opt[q]) for q, rows in by_qf.items() if len(rows) >= 12}
    tr_val = [r for r in tr_val if r["qid"] in fit_models]
    ALPHA_GRID = [4.0, 10.0, 20.0, 40.0, 80.0]
    shrink_sweep = {a: round(_eval_ll(fit_models, tr_val, ATTRS, a)[0], 4) for a in ALPHA_GRID}
    best_alpha = min(shrink_sweep, key=shrink_sweep.get)

    # Q1/Q2: individual simulation quality at each variable-richness level (at the tuned shrinkage)
    per_level = {}
    for name, use in LEVELS.items():
        ll, acc, tot = _eval_ll(models, te, use, best_alpha)
        per_level[name] = {"n": tot, "log_loss": round(ll, 4), "accuracy": round(acc, 4), "n_vars": len(use)}

    # Q3: aggregate subgroup share — grounded bottom-up vs top-down composite, on DISTINCTIVE subgroups
    full = LEVELS["all 11 variables"]
    subs = defaultdict(list)                 # (qid, attr, level) -> test rows
    for r in te:
        for attr in ("ideology", "party", "religion"):
            subs[(r["qid"], attr, r["demo"].get(attr, "unknown"))].append(r)
    tv_top, tv_bot, tv_top_d, tv_bot_d, nd = [], [], [], [], 0
    for (q, attr, lev), rows in subs.items():
        if len(rows) < 6:
            continue
        m = models[q]; no = m.n_opt
        true = [0.0] * no
        for r in rows:
            true[r["answer_idx"]] += 1
        true = [c / len(rows) for c in true]
        top = list(m.p_a)                                    # composite: the question marginal
        bot = [0.0] * no                                     # bottom-up: mean of grounded per-person sims
        for r in rows:
            p = m.predict(r["demo"], full)
            for a in range(no):
                bot[a] += p[a] / len(rows)
        tv_top.append(_tv(top, true)); tv_bot.append(_tv(bot, true))
        if _tv(top, true) > 0.10:                            # distinctive: subgroup far from the marginal
            tv_top_d.append(_tv(top, true)); tv_bot_d.append(_tv(bot, true)); nd += 1

    agg = {"n_subgroups": len(tv_top),
           "top_down_composite_TV": round(sum(tv_top) / len(tv_top), 4),
           "bottom_up_grounded_TV": round(sum(tv_bot) / len(tv_bot), 4),
           "n_distinctive": nd,
           "distinctive_top_down_TV": round(sum(tv_top_d) / max(1, len(tv_top_d)), 4),
           "distinctive_bottom_up_TV": round(sum(tv_bot_d) / max(1, len(tv_bot_d)), 4)}

    base = per_level["marginal (composite)"]
    fullv = per_level["all 11 variables"]
    best_full = per_level["all 11 variables"]["log_loss"]
    out = {"dataset": "OpinionQA", "n_questions": len(models), "n_test_rows": len(te),
           "tuned_shrinkage_alpha": best_alpha, "shrinkage_sweep_all11_trainval": shrink_sweep,
           "individual_by_variable_richness": per_level, "aggregate_subgroups": agg,
           "grounding_beats_composite_individual": fullv["log_loss"] < base["log_loss"],
           "best_individual_beats_composite": min(d["log_loss"] for d in per_level.values()) < base["log_loss"],
           "grounding_beats_composite_distinctive": agg["distinctive_bottom_up_TV"] < agg["distinctive_top_down_TV"]}

    print(f"EXP-040 grounded simulation vs composite — OpinionQA, {len(models)} questions, {len(te)} test answers")
    print(f"  shrinkage tuned on train-val: alpha={best_alpha} (sweep all-11 log-loss {shrink_sweep})")
    print("  Q1/Q2 INDIVIDUAL simulation quality by variable richness (does mapping MORE real vars help?):")
    for name, d in per_level.items():
        print(f"    {name:<28} vars={d['n_vars']:<2} log_loss {d['log_loss']}  accuracy {d['accuracy']}")
    print(f"  -> grounded (all 11, tuned shrink) beats composite on individual log-loss: "
          f"{out['grounding_beats_composite_individual']}  ({base['log_loss']} -> {fullv['log_loss']})")
    print("  Q3 AGGREGATE subgroup share — bottom-up grounded vs top-down composite (TV, lower better):")
    print(f"    all subgroups   top-down {agg['top_down_composite_TV']}  bottom-up {agg['bottom_up_grounded_TV']}"
          f"  (n={agg['n_subgroups']})")
    print(f"    DISTINCTIVE     top-down {agg['distinctive_top_down_TV']}  bottom-up {agg['distinctive_bottom_up_TV']}"
          f"  (n={agg['n_distinctive']})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
