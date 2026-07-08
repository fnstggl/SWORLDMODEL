"""EXP-076: the corpus-scale calibration harvest — fit elasticities across EVERY dataset, persist them.

Architecture item #1 at scale. For each dataset, fit calibrated weights and register the learned elasticities
into `learned_priors.json` under a semantic outcome-class, so the compiler serves every future question's
variables pre-calibrated from real data. Multi-domain:
  - GSS  demographics → conservative_opinion   (15 items, signed to the conservative pole)
  - OpinionQA demographics → opinion (per big question)
  - CMV  person×message features → persuasion   (if the joined cache exists)
  - FOMC macro → rate_hike
  - Upworthy headline lexical features → headline_engagement
  - StackExchange features → question_answered  (if the fetched cache exists)

Run: python -m experiments.exp076_corpus_harvest
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from experiments.datasets_gss import load as load_gss
from experiments.exp073_event_backtest import ATTRS_FULL, _item_rows
from swm.eval.harvest import harvest_source, onehot
from swm.variables.llm_prior import ITEM_POLE
from swm.variables.prior_registry import PriorRegistry, semantic_key

REGISTRY_PATH = "swm/variables/learned_priors.json"
RESULT = "experiments/results/exp076_corpus_harvest.json"
OQA = "experiments/results/exp028_oqa/oqa_parsed.json"
UPW = "experiments/results/exp054_upworthy/upworthy_parsed.json"
FOMC = "experiments/results/exp071/fomc_macro.json"
EXTRA = "experiments/results/harvest_extra"
UPW_FEATS = ["log_words", "has_question", "has_exclaim", "has_number", "caps_frac", "has_you", "curiosity",
             "emotion", "ends_period", "commas"]


def _gss(reg):
    rows = load_gss()
    done = 0
    for item, pole in ITEM_POLE.items():
        irows = _item_rows(rows, item)
        if len(irows) < 300:
            continue
        for r in irows:
            r["ys"] = r["y"] if pole > 0 else 1 - r["y"]
        X, names = onehot(irows, ATTRS_FULL, lambda r: r["demo"])
        if harvest_source(reg, X, [r["ys"] for r in irows], names, "conservative_opinion",
                          source=f"gss:{item}", seed=hash(item) & 255):
            done += 1
    return {"source": "GSS", "outcome_class": "conservative_opinion", "fits": done}


def _oqa(reg, top=8):
    data = json.loads(Path(OQA).read_text())
    byq = defaultdict(list)
    for r in data:
        if r.get("n_opt") == 2 and int(r["answer_idx"]) in (0, 1):
            byq[r["qid"]].append(r)
    big = sorted(byq.items(), key=lambda kv: -len(kv[1]))[:top]
    attrs = ["party", "ideology", "religion", "attendance", "race", "age", "education", "income", "sex", "marital"]
    fits = 0
    for qid, rs in big:
        for r in rs:
            r["y"] = int(r["answer_idx"])
        X, names = onehot(rs, attrs, lambda r: r["demo"])
        if harvest_source(reg, X, [r["y"] for r in rs], names, f"oqa:{qid}", source=f"oqa:{qid}"):
            fits += 1
    return {"source": "OpinionQA", "outcome_class": "oqa:<per-question>", "fits": fits}


def _cmv(reg):
    path = f"{EXTRA}/cmv.json"
    if not os.path.exists(path):
        return {"source": "CMV", "fits": 0, "note": "cache absent (fetch pending)"}
    rows = json.loads(Path(path).read_text())
    names = sorted({k for r in rows for k in r["features"]})
    X = [[float(r["features"].get(n, 0.0)) for n in names] for r in rows]
    cw = harvest_source(reg, X, [int(r["y"]) for r in rows], names, "persuasion", source="cmv")
    return {"source": "CMV", "outcome_class": "persuasion", "fits": 1 if cw else 0}


def _fomc(reg):
    data = json.loads(Path(FOMC).read_text())
    rows = []
    for i, d in enumerate(data):
        if d.get("move_fwd3") is None:
            continue
        rate3 = data[max(0, i - 3)]["rate"]
        rows.append(([d["inflation"] / 10.0, d["unemp"] / 10.0, d["rate"] / 10.0,
                      max(-1.0, min(1.0, d["rate"] - rate3))], 1 if d["move_fwd3"] > 0.1 else 0))
    names = ["inflation", "unemployment", "rate_level", "recent_move"]
    cw = harvest_source(reg, [x for x, _ in rows], [y for _, y in rows], names, "rate_hike",
                        source="fomc", epochs=120)
    return {"source": "FOMC", "outcome_class": "rate_hike", "fits": 1 if cw else 0}


def _upworthy(reg):
    from experiments.exp054_interventional import _features
    data = json.loads(Path(UPW).read_text())
    X, y = [], []
    for t in data:
        arms = t["arms"]
        if len(arms) < 2:
            continue
        ctrs = [a["ctr"] for a in arms]
        med = sorted(ctrs)[len(ctrs) // 2]
        for a in arms:
            X.append(_features(a["headline"]))
            y.append(1 if a["ctr"] > med else 0)
    cw = harvest_source(reg, X, y, UPW_FEATS, "headline_engagement", source="upworthy", cap=6000)
    return {"source": "Upworthy", "outcome_class": "headline_engagement", "fits": 1 if cw else 0}


def _isnum(v):
    try:
        float(v)
        return not isinstance(v, bool)
    except (TypeError, ValueError):
        return False


def _extra_generic(reg, name, outcome_class):
    """Harvest a fetched cache with the {features:{...}, y} shape. Uses only NUMERIC feature keys (drops
    categorical strings like country), so mixed caches harvest cleanly."""
    path = f"{EXTRA}/{name}.json"
    if not os.path.exists(path):
        return {"source": name, "fits": 0, "note": "cache absent"}
    rows = json.loads(Path(path).read_text())
    keys = [k for k in sorted({k for r in rows for k in r["features"]})
            if all(_isnum(r["features"].get(k, 0)) for r in rows[:50])]
    X = [[float(r["features"].get(k, 0.0)) for k in keys] for r in rows]
    cw = harvest_source(reg, X, [int(r["y"]) for r in rows], keys, outcome_class, source=name, cap=6000)
    return {"source": name, "outcome_class": outcome_class, "fits": 1 if cw else 0}


def run() -> dict:
    reg = PriorRegistry.load(REGISTRY_PATH)
    summ = [_gss(reg), _oqa(reg), _cmv(reg), _fomc(reg), _upworthy(reg),
            _extra_generic(reg, "stackexchange", "question_answered"),
            _extra_generic(reg, "telco_churn", "customer_churn"),
            _extra_generic(reg, "globalopinions", "opinion_consensus")]
    reg.save(REGISTRY_PATH)

    flag = {"conservative_opinion": ["party=republican", "ideology=liberal", "attendance=high"],
            "persuasion": ["arg_addresses_crux", "arg_evidence", "op_skepticism"],
            "rate_hike": ["inflation", "unemployment"],
            "headline_engagement": ["curiosity", "has_number", "emotion"],
            "customer_churn": ["tenure", "monthly_charges", "contract"]}
    learned = {}
    for oc, vs in flag.items():
        for v in vs:
            rec = reg.records.get(semantic_key(v, oc))
            if rec:
                learned[f"{v} -> {oc}"] = {"elasticity": round(rec.mean, 3), "sd": round(rec.sd, 3), "n": rec.n}
    ocs = sorted({k.split("|")[1] for k in reg.records})
    res = {"sources": summ, "total_priors": len(reg.records), "n_outcome_classes": len(ocs),
           "outcome_classes": ocs, "flagship_elasticities": learned, "registry_path": REGISTRY_PATH}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-076  corpus calibration harvest — elasticities across every dataset")
    for s in summ:
        print(f"  {s['source']:14s} fits={s.get('fits')}  {s.get('note', s.get('outcome_class',''))}")
    print(f"  -> {len(reg.records)} learned priors across {len(ocs)} outcome-classes, committed to {REGISTRY_PATH}")
    print("  flagship learned elasticities (sign should match domain knowledge):")
    for k, v in learned.items():
        print(f"    {k:44s} {v['elasticity']:+.3f} ± {v['sd']:.3f} (n={v['n']})")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
