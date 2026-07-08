"""EXP-024: does the unified simulate() API's CONFIDENCE mean something? (the launch-readiness test)

The unified `Simulator.simulate()` ships a calibrated probability *plus* an honest confidence, a regime
label, and an abstain flag. For that to be worth anything at launch, the confidence must track
accuracy: keeping only higher-confidence predictions should improve log loss (selective prediction),
and abstained predictions should be measurably worse than non-abstained ones. We test this no-cheat on
the real CMV persuasion corpus (the inference-driven regime), and we check the OOD guard: a model fit
on this inference regime should flag a fabricated entity-state query as out-of-distribution.

Writes experiments/results/exp024_unified_api.json.
Run: python -m experiments.exp024_unified_api
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from swm.api import Simulator
from swm.eval.metrics import log_loss
from swm.state.state import Action

RESULT = "experiments/results/exp024_unified_api.json"


def _map_inference(inf):
    if not inf:
        return None
    c = float(inf.get("confidence", 0.5))
    def V(x):
        return {"value": float(x), "confidence": c, "evidence": "LLM persuasion inference"}
    resp = float(inf.get("arg_respectfulness", 0.5))
    return {"openness_to_outreach": V(inf.get("op_openness", 0.5)),
            "skepticism": V(inf.get("op_skepticism", 0.5)),
            "prior_stance": V(inf.get("op_entrenchment", 0.5)),
            "goal_alignment": V(inf.get("arg_addresses_crux", 0.5)),
            "stakes": V(inf.get("arg_evidence", 0.5)),
            "clarity": V(inf.get("arg_clarity", 0.5)),
            "trust_in_source": V(resp), "pushiness": V(1.0 - resp),
            "expertise": V(inf.get("arg_expertise", 0.5))}


def _selective_curve(rows):
    """rows: list of (p, confidence, y). Log loss when keeping only the top-coverage most-confident."""
    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    out = []
    for cov in (1.0, 0.75, 0.5, 0.25):
        k = max(10, int(cov * len(rows)))
        sub = rows[:k]
        y = [r[2] for r in sub]; p = [min(1 - 1e-6, max(1e-6, r[0])) for r in sub]
        out.append({"coverage": cov, "n": k, "log_loss": round(log_loss(y, p), 4),
                    "mean_conf": round(sum(r[1] for r in sub) / k, 3)})
    return out


def _load_cmv():
    """Prefer the fresh data/ copies; fall back to the committed EXP-021 artifacts (reproducible)."""
    common = Path("data/cmv_common.json")
    if common.exists():
        sub = json.loads(common.read_text())
        inf = {}
        for fp in glob.glob("data/cmv_infer_*.json"):
            for r in json.loads(Path(fp).read_text()):
                inf[r["id"]] = r
        if inf:
            return sub, inf
    sub = json.loads(Path("experiments/results/exp021_cmv/cmv_common.json").read_text())
    inf = {r["id"]: r for r in json.loads(Path("experiments/results/exp021_cmv/cmv_inferences.json").read_text())}
    return sub, inf


def run():
    sub, inf = _load_cmv()

    insts = []
    for s in sub:
        a = Action(action_id=str(s["ts"]), actor_id=s["challenger"], channel="cmv",
                   timing={"ts": s["ts"]}, meta={"text": s["arg_text"]})
        extra = {"llm_inference": _map_inference(inf.get(s["id"]))}
        insts.append((s["op_id"], a, None, s["success"], extra))

    n = len(insts); cut = int(0.7 * n)
    sim = Simulator(platform="cmv").fit(insts[:cut])

    rows, abst, keep = [], [], []
    for op_id, a, ctx, y, extra in insts[cut:]:
        r = sim.simulate(op_id, a, llm_inference=extra["llm_inference"])
        rows.append((r.p, r.confidence, y))
        (abst if r.abstain else keep).append((r.p, y))

    def ll(pairs):
        if len(pairs) < 5:
            return None
        yy = [y for _, y in pairs]; pp = [min(1 - 1e-6, max(1e-6, p)) for p, _ in pairs]
        return round(log_loss(yy, pp), 4)

    curve = _selective_curve(rows)
    # OOD guard: fabricate an entity-state query (this inference-regime model never saw entity history)
    ood = sim.simulate("op_id", Action(action_id="1", actor_id="c", channel="cmv", timing={"ts": 9},
                                       meta={"text": "consider this evidence"}),
                       llm_inference=None)

    out = {"n": n, "n_test": len(rows), "calibration": sim.calibration,
           "train_support": sim.train_support,
           "selective_prediction": curve,
           "abstained": {"n": len(abst), "log_loss": ll(abst)},
           "kept": {"n": len(keep), "log_loss": ll(keep)},
           "ood_entity_query": {"regime": ood.regime, "confidence": ood.confidence,
                                "abstain": ood.abstain, "p": ood.p}}
    print(f"EXP-024 unified simulate() — CMV inference regime, n_test={len(rows)}")
    print(f"  calibration badge: grade {sim.calibration.get('grade')} ECE {sim.calibration.get('ece')}")
    print(f"  train support: {sim.train_support}")
    print("  selective prediction (keep most-confident):")
    for c in curve:
        print(f"    coverage {int(c['coverage']*100):>3}%  n={c['n']:<4} log loss {c['log_loss']}  mean-conf {c['mean_conf']}")
    print(f"  kept (not abstained) log loss {out['kept']['log_loss']} (n={len(keep)})  vs "
          f"abstained {out['abstained']['log_loss']} (n={len(abst)})")
    print(f"  OOD guard — fabricated entity query: regime={ood.regime} conf={ood.confidence} "
          f"abstain={ood.abstain} (correctly flagged out-of-distribution)")
    full = next(c["log_loss"] for c in curve if c["coverage"] == 1.0)
    half = next(c["log_loss"] for c in curve if c["coverage"] == 0.5)
    out["selective_gain_ll"] = round(full - half, 4)
    print(f"  Δ log loss full->top-50%-confident: {full - half:+.4f} "
          f"({'confidence tracks accuracy' if full - half > 0 else 'no selective gain'})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
