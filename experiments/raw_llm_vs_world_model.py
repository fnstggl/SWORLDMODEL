"""EXP-007: is the system genuinely more than raw LLM + context?

Same no-cheat temporal HN split (train = hn_train2 / March 2026; test = round 4 / May 2026,
114 posts with committed LLM+context forecasts + real outcomes). Five predictors compared head to
head on the identical test posts:

  1. raw LLM, TITLE ONLY            (blind swarm forecast from title+timing; data/exp7_titleonly_*)
  2. raw LLM + retrieved context    (the committed round4 forecasts, made WITH author priors+domain)
  3. non-LLM structured statistical (logistic over content + prior + entity factors)
  4. current calibrated system      (#2 + the fitted Platt calibration layer)
  5. state-transition model         (#3 + transitioned context: domain_reputation, topic_salience)

Metrics: log loss, Brier, ECE, uplift@k. Plus: decision lift over raw-LLM+context (#2), factor
ablation, and (Part B) transitions-vs-static on the repeat-author sequences. Reports honestly; if
#5 does not beat #2, it says so.

  python -m experiments.raw_llm_vs_world_model packets   # writes title-only packets for the swarm
  python -m experiments.raw_llm_vs_world_model score --titleonly "data/exp7_titleonly_*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from swm.eval.metrics import (brier_score, expected_calibration_error, log_loss, uplift_at_k)
from swm.state.factors import build_hn_registry, tag_topic
from swm.state.state import Action, Posterior, WorldState
from swm.transition.readout import LogisticReadout

TRAIN, TEST = "data/hn_train2", "data/hn_r4"
THRESHOLDS = [10, 40]
CONTENT = ["title_len", "is_show", "is_ask", "is_text", "hour_sin", "hour_cos", "is_weekend"]
ENTITY = ["author_quality", "author_ceiling", "author_standing", "author_volume", "author_recency"]
CONTEXT = ["domain_reputation", "topic_salience"]


def _content(x: dict) -> dict:
    t = x["title"].lower()
    return {"title_len": min(1.0, len(x["title"]) / 80), "is_show": 1.0 if t.startswith("show hn") else 0.0,
            "is_ask": 1.0 if t.startswith("ask hn") else 0.0, "is_text": 1.0 if x["is_text_post"] else 0.0,
            "topic": tag_topic(x["title"])}


def _seed(world: WorldState, x: dict) -> None:
    """Seed an author's latent state from retrieved pre-as_of priors on first sight (as-of clean)."""
    a = x["author"]
    if a in world.entity_states:
        return
    e = world.entity(a)
    med, mx = x.get("author_median_past"), x.get("author_max_past")
    frac, n = x.get("author_frac_ge10_past"), x.get("author_n_past") or 0
    if med is not None:
        e.stable_traits["quality"] = Posterior(math.log1p(med), 2)
    if mx is not None:
        e.stable_traits["ceiling"] = Posterior(math.log1p(mx), 2)
    if frac is not None:
        e.relationship_stance["standing"] = Posterior(frac, 2)
    e.history_features["n_posts"] = n


def build_rows():
    """Process train + test posts in global time order through the registry; return per-post factor
    dicts, scores, and split membership. Entity state is seeded from priors and TRANSITIONS forward."""
    reg = build_hn_registry()
    tr = [(x, 0) for x in json.load(open(f"{TRAIN}_inputs.json"))]
    te = [(x, 1) for x in json.load(open(f"{TEST}_inputs.json"))]
    tro = {o["id"]: o for o in json.load(open(f"{TRAIN}_outcomes.json"))}
    teo = {o["id"]: o for o in json.load(open(f"{TEST}_outcomes.json"))}
    posts = []
    for x, split in tr + te:
        o = (tro if split == 0 else teo).get(x["id"])
        if o:
            posts.append((x, split, o["score"]))
    posts.sort(key=lambda p: p[0].get("date", "") + f"{p[0].get('hour_utc',0):02d}")
    world = WorldState(timestamp=0)
    rows, scores, splits, ids = [], [], [], []
    for x, split, score in posts:
        _seed(world, x)
        cf = _content(x)
        action = Action(action_id=str(x["id"]), actor_id=x["author"], content_features=cf,
                        timing={"hour": x.get("hour_utc", 12), "weekday": x.get("weekday", 2)},
                        meta={"domain": x.get("domain", ""), "title": x["title"]})
        e = world.entity(x["author"])
        rows.append({f.name: f.extract(e, action, world.context_state) for f in reg.active()})
        scores.append(score); splits.append(split); ids.append(x["id"])
        reg.apply_update(e, world.context_state, action, score)   # transition with actual outcome
    return reg, rows, scores, splits, ids


def _fit_predict(rows, scores, splits, names, thr):
    tr = [i for i in range(len(rows)) if splits[i] == 0]
    te = [i for i in range(len(rows)) if splits[i] == 1]
    Xtr = [[rows[i][n] for n in names] for i in tr]; ytr = [1 if scores[i] >= thr else 0 for i in tr]
    Xte = [[rows[i][n] for n in names] for i in te]
    if len(set(ytr)) < 2:
        base = sum(ytr) / len(ytr); return {ids: base for ids in te}, te
    m = LogisticReadout(seed=thr).fit(Xtr, ytr)
    return {i: m.predict_proba(Xte[k]) for k, i in enumerate(te)}, te


def _metrics(y, p):
    return (log_loss(y, p), brier_score(y, p), expected_calibration_error(y, p), uplift_at_k(y, p, 0.2))


def score(titleonly_glob: str):
    reg, rows, scores, splits, ids = build_rows()
    idpos = {ids[i]: i for i in range(len(ids))}
    te = [i for i in range(len(ids)) if splits[i] == 1]

    # raw LLM predictions
    llm_ctx = {p["id"]: p for p in json.load(open(f"{TEST}_predictions.json".replace(TEST, "data/round4")))} \
        if Path("data/round4_predictions.json").exists() else {}
    llm_ctx = {p["id"]: p for p in json.load(open("data/round4_predictions.json"))}
    titleonly = {}
    for f in glob.glob(titleonly_glob):
        for p in json.load(open(f)):
            titleonly[p["id"]] = p
    cal = json.loads(Path("data/calibration.json").read_text()) if Path("data/calibration.json").exists() else {}

    def platt(p, thr):
        a, b = cal.get(str(thr), [1.0, 0.0])
        z = math.log(min(1 - 1e-6, max(1e-6, p)) / (1 - min(1 - 1e-6, max(1e-6, p))))
        return 1 / (1 + math.exp(-(a * z + b)))

    print(f"EXP-007  test = round 4 (n={len(te)} posts), train = March.  Metrics per threshold.\n")
    lift_ref = {}
    for thr in THRESHOLDS:
        y = [1 if scores[i] >= thr else 0 for i in te]
        # statistical (#3) and state (#5)
        stat_p, _ = _fit_predict(rows, scores, splits, CONTENT + ENTITY, thr)
        state_p, _ = _fit_predict(rows, scores, splits, CONTENT + ENTITY + CONTEXT, thr)
        preds = {
            "1 raw LLM title-only": [min(.999, max(.001, titleonly.get(ids[i], {}).get(f"p_ge_{thr}", 0.13))) for i in te],
            "2 raw LLM + context":  [min(.999, max(.001, llm_ctx.get(ids[i], {}).get(f"p_ge_{thr}", 0.13))) for i in te],
            "3 statistical (no LLM)": [stat_p[i] for i in te],
            "4 calibrated (=#2+Platt)": [platt(llm_ctx.get(ids[i], {}).get(f"p_ge_{thr}", 0.13), thr) for i in te],
            "5 state-transition":   [state_p[i] for i in te],
        }
        print(f"-- P(score >= {thr}) | base rate {sum(y)/len(y):.3f} --")
        print(f"   {'method':<26}{'logloss':>9}{'brier':>8}{'ece':>7}{'uplift@20':>11}")
        for name, p in preds.items():
            ll, br, ece, up = _metrics(y, p)
            print(f"   {name:<26}{ll:>9.4f}{br:>8.4f}{ece:>7.4f}{up:>11.4f}")
        # decision lift vs raw LLM+context (#2): uplift@20 delta
        if thr == 40:
            lift_ref = {name: _metrics(y, p)[3] for name, p in preds.items()}
        print()

    print("== decision lift vs raw LLM+context (#2), uplift@20 at >=40 ==")
    ref = lift_ref.get("2 raw LLM + context", 0.0)
    for name, u in lift_ref.items():
        print(f"   {name:<26} uplift@20 {u:+.4f}   (vs #2: {u-ref:+.4f})")

    print("\n== factor ablation (KEEP only if removing worsens held-out) ==")
    from swm.state.ablation import run_ablation
    for r in run_ablation(reg, rows, scores, thr_key=40)[:8]:
        print(f"   {r.factor:<20} d_logloss {r.delta_logloss:+.4f}  -> {r.verdict}")


def packets():
    te = json.load(open(f"{TEST}_inputs.json"))
    pk = [{"id": x["id"], "title": x["title"], "hour_utc": x.get("hour_utc"),
           "weekday": x.get("weekday")} for x in te]
    k = 4
    for j in range(k):
        Path(f"data/exp7_titleonly_batch{j}.json").write_text(json.dumps(pk[j::k], indent=1))
    print(f"wrote {k} title-only batch files (title + timing ONLY; no author, no domain)")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("packets")
    s = sub.add_parser("score"); s.add_argument("--titleonly", required=True)
    a = ap.parse_args()
    if a.cmd == "packets":
        packets()
    else:
        score(a.titleonly)


if __name__ == "__main__":
    main()
