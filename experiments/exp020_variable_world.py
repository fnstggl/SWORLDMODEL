"""EXP-020: the mapped-variable world model, backtested on real data (the core-architecture proof).

Runs `VariableWorld` — every prediction flows entity+action+context -> infer ALL behavioral variables
(known from data/platform/heuristics; inferred vars optional) -> VariableMap -> calibrated readout ->
P(response). Compares, no-cheat temporal split:

  A) VariableWorld (data+platform+heuristic variables, NO LLM) vs the plain entity-state model
     — does routing everything through the full variable map cost or keep accuracy?
  B) VariableWorld + LLM-inferred latent variables (from an agent swarm, on a sample) vs without
     — do the INFERRED dispositional/relational/incentive variables earn their place?

Datasets: GitHub issue-response (entity=repo, platform=github) and Enron email-reply (entity=recipient,
platform=email). Writes experiments/results/exp020_variable_world.json.

Usage:
  python -m experiments.exp020_variable_world github
  python -m experiments.exp020_variable_world enron
  python -m experiments.exp020_variable_world llm-arm    # after agent inferences are in data/vw_infer_*.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

from swm.eval.metrics import log_loss
from swm.state.state import Action
from swm.worlds.variable_world import VariableWorld
from swm.transition.response_model import ResponseConfig, ResponseModel

RESULT = "experiments/results/exp020_variable_world.json"


def _github_instances():
    recs = json.loads(Path("data/gh_issues.json").read_text())["records"]
    insts = []
    for r in recs:
        t = r["title"]
        a = Action(action_id=str(r.get("ts")), actor_id="opener", channel="github",
                   content_features={"clarity": 0.6 if len(t) > 20 else 0.3,
                                     "effort_cost": min(1.0, r["body_len"] / 2000)},
                   timing={"ts": r["ts"]}, meta={"text": t})
        insts.append((r["repo"], a, None, r["responded"]))
    return insts, "github"


def _enron_instances():
    from experiments.datasets_enron import load_samples
    msgs = json.loads(Path("data/enron_messages.json").read_text())
    msgs = [m for m in msgs if 9e8 < m["ts"] < 1.1e9]
    msgs.sort(key=lambda m: m["ts"])
    msgs = msgs[-16000:]
    from collections import defaultdict
    reply_idx = defaultdict(list)
    for m in msgs:
        for rcp in m["to"]:
            reply_idx[(m["from"], rcp, m["nsubj"])].append(m["ts"])
    insts = []
    W = 14 * 86400
    for m in msgs:
        rcp = m["to"][0]
        cand = reply_idx.get((rcp, m["from"], m["nsubj"]), [])
        replied = 1 if any(m["ts"] < ct <= m["ts"] + W for ct in cand) else 0
        a = Action(action_id=str(m["ts"]), actor_id=m["from"], channel="email",
                   content_features={"effort_cost": min(1.0, m["body_len"] / 2000)},
                   timing={"ts": m["ts"]}, meta={"text": m["subj"]})
        insts.append((rcp, a, None, replied))
    return insts, "email"


def _plain_response(insts, mfn_fn, cut):
    """The entity-state response model baseline (best config) over the same stream."""
    samples = [(e, None, mfn_fn(a), o) for e, a, _, o in insts]
    mfn = list(mfn_fn(insts[0][1]).keys())
    m = ResponseModel(mfn, ResponseConfig(use_recency=True, readout="logistic")).fit_stream(
        samples[:cut], global_rate=(sum(s[3] for s in samples[:cut]) + 1) / (cut + 2))
    preds, y = [], []
    for e, seg, mf, o in samples[cut:]:
        preds.append(min(1 - 1e-6, max(1e-6, m.predict(e, seg, mf)))); y.append(int(o))
        m.observe(e, seg, o)
    return log_loss(y, preds)


def _mf(a):
    t = (a.meta.get("text", "") or "").lower()
    return {"len": min(1.0, len(t) / 80), "is_q": 1.0 if "?" in t else 0.0,
            "effort": a.content_features.get("effort_cost", 0.3)}


def run(which):
    insts, platform = _github_instances() if which == "github" else _enron_instances()
    n = len(insts); cut = int(0.7 * n)
    vw = VariableWorld(platform=platform)
    res, preds, y = vw.backtest(insts)
    plain_ll = _plain_response(insts, _mf, cut)
    print(f"=== {which} (n={n}, base {res['base_rate']}) ===")
    print(f"  plain entity-state model   log loss {plain_ll:.4f}")
    print(f"  VariableWorld (mapped vars) log loss {res['log_loss']:.4f}  ece {res['ece']}  up@20 {res['uplift@20']}")
    print(f"  Δ (variable-world - plain) {res['log_loss'] - plain_ll:+.4f}  "
          f"({'variable map keeps/gains accuracy' if res['log_loss'] <= plain_ll + 0.003 else 'variable map costs accuracy'})")
    out = json.loads(Path(RESULT).read_text()) if Path(RESULT).exists() else {}
    out[which] = {"n": n, "base_rate": res["base_rate"], "plain_log_loss": round(plain_ll, 4),
                  "variable_world": res}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")


def llm_arm():
    """Do LLM-INFERRED variables earn their place? Reuse the EXP-019 semantic features (extracted by
    an agent swarm) as llm-provenance VariableMap variables, and compare VariableWorld WITH vs WITHOUT
    them on the same 1,600 GitHub issues — overall and by repo-history depth."""
    sub = json.loads(Path("data/gh_sem_common.json").read_text())
    sem = {}
    for fp in glob.glob("data/gh_sem_out_*.json"):
        for r in json.loads(Path(fp).read_text()):
            sem[r["id"]] = r
    # map semantic features -> schema variables (llm provenance)
    def infer_for(i):
        s = sem.get(i)
        if not s:
            return None
        return {"clarity": {"value": s.get("clarity", 0.5), "confidence": 0.7, "evidence": "LLM"},
                "ask_directness": {"value": s.get("actionable", 0.5), "confidence": 0.7, "evidence": "LLM"},
                "effort_cost": {"value": s.get("effort_to_answer", 0.5), "confidence": 0.7, "evidence": "LLM"},
                "mood_valence": {"value": 2 * s.get("sentiment", 0.5) - 1, "confidence": 0.6, "evidence": "LLM"},
                "stakes": {"value": s.get("specificity", 0.5), "confidence": 0.4, "evidence": "LLM"}}
    insts_base, insts_llm = [], []
    for i, s in enumerate(sub):
        a = Action(action_id=str(s["ts"]), actor_id="opener", channel="github",
                   content_features={"effort_cost": min(1.0, s["body_len"] / 2000)},
                   timing={"ts": s["ts"]}, meta={"text": s["title"]})
        insts_base.append((s["repo"], a, None, s["responded"]))
        insts_llm.append((s["repo"], a, None, s["responded"], {"llm_inference": infer_for(i)}))

    def bt(insts):
        return VariableWorld(platform="github").backtest(insts)
    rb, pb, y = bt(insts_base)
    rl, pl, _ = bt(insts_llm)
    print(f"VariableWorld on 1600 GitHub issues (base {rb['base_rate']}):")
    print(f"  data+heuristic vars only   log loss {rb['log_loss']}  ece {rb['ece']}")
    print(f"  + LLM-inferred variables   log loss {rl['log_loss']}  ece {rl['ece']}  "
          f"Δ {rb['log_loss'] - rl['log_loss']:+.4f}")
    out = json.loads(Path(RESULT).read_text()) if Path(RESULT).exists() else {}
    out["llm_arm"] = {"n_test": rb["n_test"], "base": rb["base_rate"],
                      "no_llm": rb, "with_llm": rl,
                      "delta_log_loss": round(rb["log_loss"] - rl["log_loss"], 4)}
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "github"
    if arg == "llm-arm":
        llm_arm()
    else:
        run(arg)
