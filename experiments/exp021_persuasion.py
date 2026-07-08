"""EXP-021: does LLM-inferred VARIABLE MAPPING predict PERSUASION? (the on-thesis test)

r/ChangeMyView: will an argument change the OP's view (earn a delta)? The outcome is driven by the
latent variables — the OP's openness/skepticism/entrenchment and the argument's crux-fit/evidence/
clarity/respect/expertise — and the OP is one-off (little entity state), so LLM INFERENCE of those
variables is the only way to estimate them. This is exactly where the VariableMap architecture should
pay off if the thesis holds.

Arms (no-cheat temporal split by challenge timestamp):
  1. base rate
  2. surface logistic       — argument/OP surface features only (no LLM)
  3. VariableWorld, no LLM   — heuristic message-fit + OP history + platform (the mapped state, no inference)
  4. VariableWorld + LLM vars — + the 8 agent-inferred latent variables mapped into the VariableMap

Writes experiments/results/exp021_persuasion.json.
Run: python -m experiments.exp021_persuasion
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.state.state import Action
from swm.transition.readout import LogisticReadout
from swm.worlds.variable_world import VariableWorld

RESULT = "experiments/results/exp021_persuasion.json"

# map the agent-inferred fields -> VariableMap schema variables
def _map_inference(inf):
    if not inf:
        return None
    c = float(inf.get("confidence", 0.5))
    def V(x, conf=c):
        return {"value": float(x), "confidence": conf, "evidence": "LLM persuasion inference"}
    resp = float(inf.get("arg_respectfulness", 0.5))
    return {
        "openness_to_outreach": V(inf.get("op_openness", 0.5)),
        "skepticism": V(inf.get("op_skepticism", 0.5)),
        "prior_stance": V(inf.get("op_entrenchment", 0.5)),          # entrenchment resists change
        "goal_alignment": V(inf.get("arg_addresses_crux", 0.5)),     # targets their crux
        "stakes": V(inf.get("arg_evidence", 0.5)),                   # evidence/substance
        "clarity": V(inf.get("arg_clarity", 0.5)),
        "trust_in_source": V(resp),
        "pushiness": V(1.0 - resp),
        "expertise": V(inf.get("arg_expertise", 0.5)),
    }


def _surface(op_text, arg_text):
    aw = arg_text.split()
    return {"arg_logwords": math.log1p(len(aw)), "arg_has_link": 1.0 if "http" in arg_text else 0.0,
            "arg_quotes": min(1.0, arg_text.count("&gt;") / 3.0), "arg_q": min(1.0, arg_text.count("?") / 3.0),
            "op_logwords": math.log1p(len(op_text.split())),
            "arg_i": min(1.0, (arg_text.lower().count(" i ") + arg_text.lower().count("i think")) / 5.0)}


def run():
    sub = json.loads(Path("data/cmv_common.json").read_text())
    inf = {}
    for fp in glob.glob("data/cmv_infer_*.json"):
        for r in json.loads(Path(fp).read_text()):
            inf[r["id"]] = r
    cov = sum(1 for s in sub if s["id"] in inf)
    n = len(sub); cut = int(0.7 * n)
    y = [s["success"] for s in sub[cut:]]
    base = sum(s["success"] for s in sub[:cut]) / cut

    # arm 2: surface logistic
    sf_names = list(_surface("", "a").keys())
    Xtr = [[_surface(s["op_text"], s["arg_text"])[k] for k in sf_names] for s in sub[:cut]]
    ytr = [s["success"] for s in sub[:cut]]
    surf = LogisticReadout(epochs=300).fit(Xtr, ytr)
    surf_p = [min(1 - 1e-6, max(1e-6, surf.predict_proba([_surface(s["op_text"], s["arg_text"])[k] for k in sf_names])))
              for s in sub[cut:]]

    # arms 3 & 4: VariableWorld (no LLM / + LLM), same stream
    def instances(with_llm):
        out = []
        for s in sub:
            a = Action(action_id=str(s["ts"]), actor_id=s["challenger"], channel="cmv",
                       timing={"ts": s["ts"]}, meta={"text": s["arg_text"]})
            extra = {}
            if with_llm:
                extra["llm_inference"] = _map_inference(inf.get(s["id"]))
            out.append((s["op_id"], a, None, s["success"], extra))
        return out

    r_no, p_no, _ = VariableWorld(platform="cmv").backtest(instances(False))
    r_llm, p_llm, _ = VariableWorld(platform="cmv").backtest(instances(True))

    def sc(p):
        p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
        return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
                "ece": round(expected_calibration_error(y, p), 4), "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}

    tiers = {
        "base_rate": sc([base] * len(y)),
        "surface_logistic": sc(surf_p),
        "variable_world_no_llm": r_no,
        "variable_world_+llm_inferred": r_llm,
    }
    print(f"CMV persuasion (delta prediction): n={n} test={len(y)} base rate {sum(y)/len(y):.3f}, "
          f"LLM coverage {cov}/{n}")
    for k, v in tiers.items():
        print(f"  {k:<32} log loss {v['log_loss']}  brier {v['brier']}  ece {v['ece']}  up@20 {v['uplift@20']}")
    d = round(tiers["variable_world_no_llm"]["log_loss"] - tiers["variable_world_+llm_inferred"]["log_loss"], 4)
    ds = round(tiers["surface_logistic"]["log_loss"] - tiers["variable_world_+llm_inferred"]["log_loss"], 4)
    print(f"  Δ (LLM-inferred variables vs no-LLM map): {d:+.4f}   vs surface logistic: {ds:+.4f}")
    out = {"n": n, "n_test": len(y), "base_rate": round(sum(y) / len(y), 4), "llm_coverage": cov,
           "tiers": tiers, "llm_vs_nollm": d, "llm_vs_surface": ds}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
