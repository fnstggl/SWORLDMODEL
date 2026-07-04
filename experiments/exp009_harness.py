"""EXP-009 harness: raw LLM vs world model on identical HN items, same metrics.

Builds ONE common held-out set of real HN stories and scores up to six tiers on it:

  1. raw_llm            LLM, title only                         [agent-swarm predictions file]
  2. raw_llm_context    LLM + as-of author/domain track record  [agent-swarm predictions file]
  3. structured         logistic over as-of retrieval features  (live, no LLM)
  4. calibrated         structured + Platt layer                (live)
  5. aggregate_world    explicit state-transition model         (live)
  6. individual_world   per-author response model               (live; HN author = the "individual")

Key honesty points:
- "as-of context" for HN is the author's PAST scores + the domain's PAST scores, strictly before the
  story's timestamp (leakage-free retrieval) — so tier 2 legitimately measures whether retrieval
  helps a raw LLM. External NEWS retrieval remains blocked-on-corpus (see market report).
- All non-LLM tiers run live and are reproducible from the committed stream sample.
- LLM tiers need an agent predictor (no ANTHROPIC_API_KEY here); `prep` writes the inputs, agents
  write data/exp009_pred_*.json, `score` folds them in. If absent, those tiers are marked BLOCKED
  and the verdict is rendered on whatever tiers are present.

Usage:
  python -m experiments.exp009_harness prep   --n-llm 120
  python -m experiments.exp009_harness score  --preds "data/exp009_pred_*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from swm.eval.benchmark_matrix import to_markdown
from swm.eval.metrics import log_loss
from swm.eval.raw_llm_vs_world_model import run_benchmark, score_tier
from swm.retrieval.context import PostRecord, as_of, author_context, domain_context
from swm.state.factors import tag_topic
from swm.state.state import Action
from swm.transition.readout import LogisticReadout
from swm.worlds.aggregate_world import AggregateWorld
from swm.worlds.individual_world import IndividualWorld

STREAM = "data/hn_agg_stream.json"
LLM_INPUTS = "data/exp009_llm_inputs.json"
COMMON = "data/exp009_common.json"
RESULT = "experiments/results/exp009_raw_llm_vs_world_model.json"
THR = 40


def _records(stories):
    return [PostRecord(entity_id=s["author"], timestamp=s["ts"], score=float(s["score"]),
                       title=s["title"], domain=s["domain"]) for s in stories]


def _asof_features(records, s):
    """Leakage-free as-of retrieval features for one story."""
    ac = author_context(records, s["author"], s["ts"])
    dc = domain_context(records, s["domain"], s["ts"])
    t = s["title"].lower()
    return {
        "author_n_past": ac["n_past"] or 0,
        "author_median_past": ac["median_past"] if ac["median_past"] is not None else 0.0,
        "author_max_past": ac["max_past"] if ac["max_past"] is not None else 0.0,
        "author_frac_ge10": ac["frac_ge10"] if ac["frac_ge10"] is not None else 0.0,
        "domain_n": dc["n_domain"] or 0,
        "domain_mean_logscore": dc["domain_mean_logscore"] if dc["domain_mean_logscore"] is not None else 0.0,
        "title_len": min(1.0, len(s["title"]) / 80),
        "is_show": 1.0 if t.startswith("show hn") else 0.0,
        "is_ask": 1.0 if t.startswith("ask hn") else 0.0,
        "is_text": 1.0 if s["is_text"] else 0.0,
    }


_FEAT_ORDER = ["author_n_past", "author_median_past", "author_max_past", "author_frac_ge10",
               "domain_n", "domain_mean_logscore", "title_len", "is_show", "is_ask", "is_text"]


def _action(s):
    dt = datetime.fromtimestamp(s["ts"], tz=timezone.utc)
    t = s["title"].lower()
    return Action(action_id=f"{s['author']}-{s['id']}", actor_id=s["author"],
                  content_features={"title_len": min(1.0, len(s["title"]) / 80),
                                    "is_show": 1.0 if t.startswith("show hn") else 0.0,
                                    "is_ask": 1.0 if t.startswith("ask hn") else 0.0,
                                    "is_text": 1.0 if s["is_text"] else 0.0,
                                    "topic": tag_topic(s["title"])},
                  timing={"hour": dt.hour, "weekday": dt.weekday(), "ts": s["ts"]},
                  meta={"domain": s["domain"], "title": s["title"]})


def _platt(p, ab):
    a, b = ab
    z = a * math.log(max(1e-6, p) / max(1e-6, 1 - p)) + b
    return 1 / (1 + math.exp(-z))


def _fit_platt(y, p):
    """1-D logistic on the logit of p (Platt scaling)."""
    m = LogisticReadout(epochs=200).fit([[math.log(max(1e-6, q) / max(1e-6, 1 - q))] for q in p], y)
    return m


def prep(n_llm: int):
    stories = json.loads(Path(STREAM).read_text())
    n = len(stories)
    cut = int(0.7 * n)
    train, test = stories[:cut], stories[cut:]
    records_all = _records(stories)

    # common LLM subset: the most RECENT n_llm test stories (most likely post-cutoff, least memorized)
    common = test[-n_llm:] if n_llm < len(test) else test
    Path(COMMON).write_text(json.dumps(common))

    # write LLM inputs: title + as-of context (author/domain track record, leakage-free)
    inputs = []
    for s in common:
        f = _asof_features(records_all, s)
        inputs.append({
            "id": s["id"], "title": s["title"],
            "context": {
                "author_past_submissions": int(f["author_n_past"]),
                "author_median_past_score": f["author_median_past"],
                "author_max_past_score": f["author_max_past"],
                "author_frac_past_ge10": round(f["author_frac_ge10"], 3),
                "domain": s["domain"] or "(text post)",
                "domain_past_posts": int(f["domain_n"]),
                "domain_mean_log_score": round(f["domain_mean_logscore"], 3),
            }})
    Path("data").mkdir(exist_ok=True)
    Path(LLM_INPUTS).write_text(json.dumps(inputs, indent=1))
    print(f"{n} stories; train {len(train)} test {len(test)}; common LLM subset {len(common)}")
    print(f"wrote {LLM_INPUTS} (title + as-of context) and {COMMON}")
    print("Agents: predict p_ge_40_title_only and p_ge_40_with_context per id -> data/exp009_pred_*.json")


def _live_tiers(stories, common):
    """Compute the non-LLM tiers on the common subset, as-of correct."""
    n = len(stories)
    cut = int(0.7 * n)
    train = stories[:cut]
    records_all = _records(stories)
    common_ids = {s["id"] for s in common}
    y = [1 if s["score"] >= THR else 0 for s in common]

    # --- structured: logistic on as-of retrieval features. Fit on the FIRST 80% of train only, so
    #     the Platt layer below is fit on data genuinely held out from struct's fit window. ---
    fit_slice = train[:int(0.8 * len(train))]
    val = train[int(0.8 * len(train)):]
    Xtr = [[_asof_features(records_all, s)[k] for k in _FEAT_ORDER] for s in fit_slice]
    ytr = [1 if s["score"] >= THR else 0 for s in fit_slice]
    struct = LogisticReadout(epochs=300).fit(Xtr, ytr)
    struct_p = [struct.predict_proba([_asof_features(records_all, s)[k] for k in _FEAT_ORDER])
                for s in common]

    # --- calibrated: Platt layer fit on the held-out last 20% of TRAIN (unseen by struct) ---
    val_p = [struct.predict_proba([_asof_features(records_all, s)[k] for k in _FEAT_ORDER]) for s in val]
    val_y = [1 if s["score"] >= THR else 0 for s in val]
    platt = _fit_platt(val_y, val_p) if len(set(val_y)) == 2 else None
    cal_p = ([platt.predict_proba([math.log(max(1e-6, q) / max(1e-6, 1 - q))]) for q in struct_p]
             if platt else struct_p)

    # --- aggregate_world: state-transition, as-of predict carrying state through the whole stream ---
    aw = AggregateWorld(domain="hn", target_threshold=THR)
    samples = [(_action(s), float(s["score"])) for s in stories]
    tr = aw.transition
    from swm.state.population import PopulationState
    from swm.transition.transition_head import OutcomeHead
    pop = PopulationState(timestamp=samples[0][0].timing["ts"])
    Xa, ya = [], []
    for action, mag in samples[:cut]:
        Xa.append(tr.feature_vector(pop, action)); ya.append(mag); tr.transition(pop, action, mag)
    tr.head = OutcomeHead(thresholds=aw.thresholds).fit(Xa, ya)
    agg_p_by_id = {}
    for action, mag in samples[cut:]:
        p = tr.predict(pop, action)["thresholds"].get(THR, 0.0)
        sid = int(action.action_id.split("-")[-1])
        if sid in common_ids:
            agg_p_by_id[sid] = p
        tr.transition(pop, action, mag)
    agg_p = [agg_p_by_id.get(s["id"], sum(ya_ >= THR for ya_ in ya) / len(ya)) for s in common]

    # --- individual_world: per-author response model (HN author = the individual) ---
    iw_samples = []
    for s in stories:
        f = _asof_features(records_all, s)
        iw_samples.append((s["author"], {"title_len": f["title_len"], "is_show": f["is_show"],
                                         "is_ask": f["is_ask"], "is_text": f["is_text"]},
                           1 if s["score"] >= THR else 0))
    seg = sum(1 for s in train if s["score"] >= THR) / len(train)
    from swm.transition.individual_transition import IndividualTransition
    im = IndividualTransition(message_feature_names=["title_len", "is_show", "is_ask", "is_text"],
                              segment_rate=seg,
                              sources=frozenset({"segment", "person", "message"}))
    im.fit_stream(iw_samples[:cut], segment_rate=seg)
    ind_p_by_id = {}
    for (author, mf, o), s in zip(iw_samples[cut:], stories[cut:]):
        p = im.predict(author, mf)["p_mean"]
        if s["id"] in common_ids:
            ind_p_by_id[s["id"]] = p
        im.transition(author, o)
    ind_p = [ind_p_by_id.get(s["id"], seg) for s in common]

    return y, {"structured": struct_p, "calibrated": cal_p, "aggregate_world": agg_p,
               "individual_world": ind_p}


def score(pred_globs):
    stories = json.loads(Path(STREAM).read_text())
    common = json.loads(Path(COMMON).read_text())
    common_order = [s["id"] for s in common]
    y, live = _live_tiers(stories, common)

    # LLM predictions
    raw, rawc = {}, {}
    for g in pred_globs or []:
        for fp in glob.glob(g):
            for p in json.loads(Path(fp).read_text()):
                if "p_ge_40_title_only" in p:
                    raw[p["id"]] = min(0.99, max(0.01, p["p_ge_40_title_only"]))
                if "p_ge_40_with_context" in p:
                    rawc[p["id"]] = min(0.99, max(0.01, p["p_ge_40_with_context"]))
    raw_p = [raw[i] for i in common_order] if all(i in raw for i in common_order) else None
    rawc_p = [rawc[i] for i in common_order] if all(i in rawc for i in common_order) else None

    tiers = {"raw_llm": raw_p, "raw_llm_context": rawc_p, **live}
    res = run_benchmark(y, tiers, target=f"HN P(score>={THR})")
    d = res.to_dict()
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(d, indent=1))

    print(f"\nEXP-009  n={res.n}  base rate={res.base_rate:.3f}  target=P(score>={THR})\n")
    print(f"  {'tier':<20}{'log_loss':>9}{'brier':>8}{'ece':>7}{'uplift@20':>11}")
    for name in ["raw_llm", "raw_llm_context", "structured", "calibrated", "aggregate_world",
                 "individual_world"]:
        t = res.tiers.get(name, {})
        if "log_loss" in t:
            print(f"  {name:<20}{t['log_loss']:>9.4f}{t['brier']:>8.4f}{t['ece']:>7.4f}{t['uplift@20']:>11.4f}")
        else:
            print(f"  {name:<20}  {t.get('status','?')}")
    print(f"\n  VERDICT: {res.verdict}")
    print(f"  wrote {RESULT}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--n-llm", type=int, default=120)
    s = sub.add_parser("score"); s.add_argument("--preds", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.n_llm)
    else:
        score(a.preds)


if __name__ == "__main__":
    main()
