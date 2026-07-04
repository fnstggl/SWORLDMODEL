"""Simulation-vs-classifier benchmark (Phases 7/8/10): does a real multi-step, multi-actor
simulation beat raw LLM + context — and the old state->classifier — on no-cheat HN slices?

Pipeline (all as-of correct; predict a post BEFORE observing its outcome, then update state):
  1. stream the 1200-post HN window in time order; for each post compute as-of author reputation +
     history depth, as-of domain reputation + depth, and merge the LLM-extracted action features.
  2. temporal split (train 840 / test 360).
  3. fit tiers on train:
       - old_classifier      : logistic over rich features (state->classifier; the thing to beat)
       - old_onestep         : AggregateWorld one-step state-transition
       - learned_gbdt         : GBDT over rich features (nonlinear classifier)
       - simulation          : HNSimulationEngine — probability from the trajectory DISTRIBUTION
       - hybrid              : gate*simulation + (1-gate)*calibrated_llm_prior
     LLM tiers (raw_llm title-only, raw_llm+context) come from the agent-extracted p_t / p_c.
  4. score every tier overall and on each slice (repeat-author, deep-author, repeat-domain,
     strong-domain, Show/Ask, AI/security topics, high/low-context, semantics- vs state-dominant).

Writes experiments/results/exp013_simulation_benchmark.json.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.simulation.engine import HNSimulationEngine
from swm.simulation.policies import PolicyParams
from swm.state.entity_history import EntityHistoryStore
from swm.state.factors import tag_topic
from swm.transition.learned_transition import LearnedTransition
from swm.transition.readout import LogisticReadout

THR = 40
RESULT = "experiments/results/exp013_simulation_benchmark.json"
_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731
_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))  # noqa: E731

RICH_FEATS = ["novelty", "controversy", "specificity", "technical_depth", "emotional_valence",
              "source_credibility", "audience_fit", "hn_native"]
TOPICS = ["ai", "security", "programming", "science", "business", "politics", "hardware", "crypto"]


def _action_feats(s, lf):
    topic = tag_topic(s["title"])
    f = {k: float(lf.get(k, 0.4)) for k in RICH_FEATS}
    f[f"topic_{topic}"] = 1.0
    f[f"cat_{lf.get('cat', 'Other')}"] = 1.0
    f["title_len"] = min(1.0, len(s["title"]) / 80)
    f["_topic"] = topic
    f["_cat"] = lf.get("cat", "Other")
    return f


def _classifier_vector(rec):
    """Flat numeric vector for the classifier baselines: rich LLM feats + as-of state + shallow."""
    f = rec["feats"]
    return [f.get(k, 0.4) for k in RICH_FEATS] + [
        f.get("title_len", 0.5),
        1.0 if f["_cat"] == "Show" else 0.0, 1.0 if f["_cat"] == "Ask" else 0.0,
        1.0 if f["_cat"] == "Launch" else 0.0, 1.0 if f["_cat"] == "Research" else 0.0,
        rec["author_rep"], math.log1p(rec["author_depth"]), rec["author_frac_hit"],
        rec["domain_rep"], math.log1p(rec["domain_depth"]),
    ] + [1.0 if f["_topic"] == t else 0.0 for t in TOPICS]


def build_records():
    stream = json.load(open("data/hn_agg_stream.json"))
    win = json.load(open("data/wm_window.json"))
    window, cut = win["window"], win["cut"]
    lf_all = json.load(open("data/wm_features.json"))
    eh = EntityHistoryStore()
    dom_hist = {}                 # domain -> list of log scores (as-of)
    global_logs = []
    recs = []
    for s in window:
        lf = lf_all.get(str(s["id"])) or lf_all.get(s["id"]) or {}
        af = eh.get(s["author"]).features(now=s["ts"])
        gmean = (sum(global_logs) / len(global_logs)) if global_logs else 1.5
        author_rep = (af["eh_mean_logscore"] - gmean) if af["eh_depth"] > 0 else 0.0
        dh = dom_hist.get(s["domain"], [])
        domain_rep = ((sum(dh) / len(dh)) - gmean) if (s["domain"] and dh) else 0.0
        recs.append({
            "id": s["id"], "title": s["title"], "score": float(s["score"]),
            "feats": _action_feats(s, lf), "lf": lf,
            "author_rep": author_rep, "author_depth": int(af["eh_depth"]),
            "author_frac_hit": af["eh_frac_hit"],
            "domain_rep": domain_rep, "domain_depth": len(dh), "domain": s["domain"],
            "p_t": lf.get("p_t"), "p_c": lf.get("p_c"),
        })
        # transition: observe AFTER recording (as-of)
        eh.observe(s["author"], s["ts"], float(s["score"]))
        if s["domain"]:
            dom_hist.setdefault(s["domain"], []).append(math.log1p(s["score"]))
        global_logs.append(math.log1p(s["score"]))
    return recs, cut


# ------------------------------------------------------------------ simulation engine fit
def _sim_ctx(rec):
    return {"exposure_mult": 1.0 + 0.25 * max(-1.0, min(2.0, rec["domain_rep"]))}


def fit_simulation(train, *, n_traj=26, seed=0):
    """Coordinate search over the few key dynamics params to minimize train log loss of the
    trajectory-derived P(hit); then a Platt readout on the simulated probability."""
    rng = random.Random(seed)
    sub = train if len(train) <= 260 else [train[i] for i in
                                           sorted(rng.sample(range(len(train)), 260))]
    y = [1 if r["score"] >= THR else 0 for r in sub]

    def loss(params):
        eng = HNSimulationEngine(params=params)
        ps = []
        for i, r in enumerate(sub):
            s = eng.simulate(r["feats"], author_rep=r["author_rep"], ctx=_sim_ctx(r),
                             n_samples=n_traj, seed=1000 + i)
            ps.append(min(1 - 1e-3, max(1e-3, s["p_hit_raw"])))
        return log_loss(y, ps)

    p = PolicyParams()
    grid = {
        "new_page_exposure": [30, 45, 60, 80],
        "frontpage_threshold": [6, 8, 11, 15],
        "frontpage_multiplier": [20, 30, 45],
        "author_rep_gain": [0.3, 0.6, 1.0],
    }
    best = p
    best_loss = loss(p)
    for _ in range(2):                          # 2 coordinate passes
        for key, vals in grid.items():
            for v in vals:
                cand = PolicyParams.from_dict({**best.to_dict(), key: float(v)})
                lv = loss(cand)
                if lv < best_loss:
                    best_loss, best = lv, cand
    eng = HNSimulationEngine(params=best)
    # Platt readout on full-train simulated probabilities
    raws, yy = [], []
    for i, r in enumerate(train):
        s = eng.simulate(r["feats"], author_rep=r["author_rep"], ctx=_sim_ctx(r),
                         n_samples=50, seed=5000 + i)
        raws.append(s["p_hit_raw"]); yy.append(1 if r["score"] >= THR else 0)
    eng.fit_readout(raws, yy)
    return eng, best, round(best_loss, 4)


# ------------------------------------------------------------------ metrics + slices
def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"n": len(y), "pos": sum(y), "log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score(y, p), 4), "ece": round(expected_calibration_error(y, p), 4),
            "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


def _slices(test):
    S = {"all": lambda r: True,
         "repeat_author(>=3)": lambda r: r["author_depth"] >= 3,
         "deep_author(>=5)": lambda r: r["author_depth"] >= 5,
         "cold_author(0)": lambda r: r["author_depth"] == 0,
         "repeat_domain(>=5)": lambda r: r["domain_depth"] >= 5,
         "strong_domain": lambda r: r["domain_rep"] > 0.5,
         "Show": lambda r: r["feats"]["_cat"] == "Show",
         "Ask": lambda r: r["feats"]["_cat"] == "Ask",
         "ai_topic": lambda r: r["feats"]["_topic"] == "ai",
         "security_topic": lambda r: r["feats"]["_topic"] == "security",
         "high_context": lambda r: r["author_depth"] >= 3 or r["domain_depth"] >= 5,
         "low_context": lambda r: r["author_depth"] == 0 and r["domain_depth"] < 2,
         "state_dominant": lambda r: r["author_depth"] >= 4,      # rich entity state present
         "semantics_dominant": lambda r: r["author_depth"] == 0 and r["domain_depth"] == 0}
    return {name: [r for r in test if fn(r)] for name, fn in S.items()}


def run():
    recs, cut = build_records()
    train, test = recs[:cut], recs[cut:]
    base = sum(1 for r in train if r["score"] >= THR) / len(train)
    print(f"window={len(recs)} train={len(train)} test={len(test)} base(train)={base:.3f}")

    # --- fit classifiers ---
    Xtr = [_classifier_vector(r) for r in train]
    ytr = [1 if r["score"] >= THR else 0 for r in train]
    old_clf = LogisticReadout(epochs=300).fit(Xtr, ytr)
    gbdt = LearnedTransition(thresholds=(THR,)).fit(Xtr, [r["score"] for r in train])
    # --- fit simulation ---
    print("fitting simulation dynamics...")
    eng, best_params, fit_loss = fit_simulation(train)
    print(f"  best params: npe={best_params.new_page_exposure} fpt={best_params.frontpage_threshold} "
          f"fpm={best_params.frontpage_multiplier} arg={best_params.author_rep_gain}  train-loss {fit_loss}")

    # --- predict test ---
    preds = {t: [] for t in ["raw_llm", "raw_llm_context", "old_classifier", "learned_gbdt",
                             "simulation", "hybrid"]}
    y = [1 if r["score"] >= THR else 0 for r in test]
    # calibrated LLM prior (Platt on p_c over train LLM preds where available)
    tr_llm = [(r["p_c"], 1 if r["score"] >= THR else 0) for r in train if r["p_c"] is not None]
    llm_platt = None
    if len(tr_llm) > 20 and len({b for _, b in tr_llm}) == 2:
        m = LogisticReadout(epochs=200).fit([[_LOGIT(min(1 - 1e-6, max(1e-6, a)))] for a, _ in tr_llm],
                                            [b for _, b in tr_llm])
        llm_platt = m
    for i, r in enumerate(test):
        preds["raw_llm"].append(r["p_t"] if r["p_t"] is not None else base)
        preds["raw_llm_context"].append(r["p_c"] if r["p_c"] is not None else base)
        preds["old_classifier"].append(old_clf.predict_proba(_classifier_vector(r)))
        preds["learned_gbdt"].append(gbdt.predict(_classifier_vector(r))["thresholds"][THR])
        sim = eng.predict(r["feats"], author_rep=r["author_rep"], ctx=_sim_ctx(r), n_samples=200, seed=7000 + i)
        preds["simulation"].append(sim["p_hit"])
        # hybrid gate: trust simulation more with entity-history depth + domain evidence
        cal_llm = (_SIG(_LOGIT(min(1 - 1e-6, max(1e-6, r["p_c"]))) * 1.0)
                   if llm_platt is None else llm_platt.predict_proba(
                       [_LOGIT(min(1 - 1e-6, max(1e-6, r["p_c"] if r["p_c"] is not None else base)))]))
        depth = r["author_depth"] + 0.5 * r["domain_depth"]
        gate = depth / (depth + 6.0)          # 0 cold -> ~1 deep
        preds["hybrid"].append(gate * sim["p_hit"] + (1 - gate) * cal_llm)

    # --- overall + slices ---
    slices = _slices(test)
    tiers = list(preds)
    out = {"base_rate_train": round(base, 4), "n_test": len(test), "params": best_params.to_dict(),
           "overall": {t: _score(y, preds[t]) for t in tiers}, "slices": {}}
    idx = {r["id"]: k for k, r in enumerate(test)}
    for name, rows in slices.items():
        if len(rows) < 12:
            continue
        js = [idx[r["id"]] for r in rows]
        ys = [y[j] for j in js]
        if sum(ys) < 3:
            continue
        out["slices"][name] = {t: _score(ys, [preds[t][j] for j in js]) for t in tiers}

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    _print(out)
    return out


def _print(out):
    tiers = ["raw_llm", "raw_llm_context", "old_classifier", "learned_gbdt", "simulation", "hybrid"]
    print(f"\n=== OVERALL (n={out['n_test']}, base {out['base_rate_train']}) — log loss ===")
    for t in tiers:
        s = out["overall"][t]
        print(f"  {t:<18} ll {s['log_loss']:.4f}  brier {s['brier']:.4f}  ece {s['ece']:.4f}  up@20 {s['uplift@20']:+.3f}")
    print("\n=== BY SLICE — log loss (lower=better); winner* ===")
    print(f"  {'slice':<22}{'n':>4} " + " ".join(f"{t[:9]:>10}" for t in tiers))
    for name, d in out["slices"].items():
        lls = {t: d[t]["log_loss"] for t in tiers}
        win = min(lls, key=lls.get)
        cells = []
        for t in tiers:
            mark = "*" if t == win else " "
            cells.append(f"{lls[t]:>9.4f}{mark}")
        print(f"  {name:<22}{d[tiers[0]]['n']:>4} " + " ".join(cells))


if __name__ == "__main__":
    run()
