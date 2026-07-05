"""EXP-019: does richer CONTENT representation (LLM semantic features -> world model) help the
cold/content-dominated regime? (the highest-leverage lever from exp018)

On 1,600 GitHub issues, compare the response model with SHALLOW message features vs SHALLOW + LLM
semantic features (clarity, actionable, reproducible, specificity, sentiment, effort, feature/bug),
overall and on the COLD-repo slice (where we lose to the LLM and content should matter most). The
LLM only EXTRACTS features; the world model is still the predictor. As-of temporal split.

Run:  python -m experiments.exp019_content_richness
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.transition.stacked_response import StackedResponseModel

SEM = ["clarity", "actionable", "reproducible", "specificity", "sentiment", "effort_to_answer",
       "is_feature_request", "is_bug_report"]
SHALLOW = ["title_len", "body_len_log", "n_labels", "is_bug", "is_member"]
RESULT = "experiments/results/exp019_content_richness.json"


def _build(with_sem):
    sub = json.loads(Path("data/gh_sem_common.json").read_text())
    sem = {}
    for fp in glob.glob("data/gh_sem_out_*.json"):
        for r in json.loads(Path(fp).read_text()):
            sem[r["id"]] = r
    samples, depths_meta = [], []
    for i, s in enumerate(sub):
        t = s["title"].lower()
        mf = {"title_len": min(1.0, len(s["title"]) / 80), "body_len_log": math.log1p(s["body_len"]) / 10,
              "n_labels": min(1.0, s["n_labels"] / 5),
              "is_bug": 1.0 if any(k in t for k in ("bug", "error", "crash", "fail")) else 0.0,
              "is_member": 1.0 if s["author_assoc"] in ("MEMBER", "OWNER", "COLLABORATOR") else 0.0}
        if with_sem and i in sem:
            for k in SEM:
                mf[k] = float(sem[i].get(k, 0.5))
        samples.append((s["repo"], s["repo"].split("/")[0], mf, s["responded"]))
    mfn = SHALLOW + (SEM if with_sem else [])
    return samples, mfn, sem


def _run(with_sem):
    samples, mfn, sem = _build(with_sem)
    n = len(samples); cut = int(0.7 * n)
    m = StackedResponseModel(message_feature_names=mfn).fit_stream(
        samples[:cut], global_rate=(sum(o for *_, o in samples[:cut]) + 1) / (cut + 2))
    preds, y, depth = [], [], []
    for eid, seg, mf, o in samples[cut:]:
        e = m._ent.get(eid); depth.append(e.n if e else 0)
        preds.append(min(1 - 1e-6, max(1e-6, m.predict(eid, seg, mf)))); y.append(int(o))
        m.observe(eid, seg, o)
    cov = sum(1 for i in range(len(samples)) if i in sem)
    return preds, y, depth, cov


def _score(y, p):
    return {"n": len(y), "log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4)}


def run():
    sh_p, y, depth, _ = _run(False)
    se_p, _, _, cov = _run(True)
    print(f"semantic-feature coverage: {cov}/{len(y) + int(0.7*len(y)/0.3)} (approx)")
    out = {"overall": {"shallow": _score(y, sh_p), "shallow+semantic": _score(y, se_p)}}
    print(f"\nContent richness on GitHub issue-response (stacked combiner):")
    print(f"  overall  shallow {out['overall']['shallow']['log_loss']}  "
          f"+semantic {out['overall']['shallow+semantic']['log_loss']}  "
          f"Δ {out['overall']['shallow']['log_loss'] - out['overall']['shallow+semantic']['log_loss']:+.4f}")
    # slices by repo depth
    out["by_depth"] = {}
    for lo, hi, nm in [(0, 0, "cold(0)"), (1, 4, "repeat(1-4)"), (5, 10**9, "deep(5+)")]:
        idx = [i for i, d in enumerate(depth) if lo <= d <= hi]
        if len(idx) < 20 or sum(y[i] for i in idx) < 5:
            continue
        ys = [y[i] for i in idx]
        sh = log_loss(ys, [sh_p[i] for i in idx]); se = log_loss(ys, [se_p[i] for i in idx])
        out["by_depth"][nm] = {"n": len(idx), "shallow": round(sh, 4), "semantic": round(se, 4),
                               "delta": round(sh - se, 4)}
        print(f"  {nm:<12} n={len(idx):<5} shallow {sh:.4f}  +semantic {se:.4f}  Δ {sh-se:+.4f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
