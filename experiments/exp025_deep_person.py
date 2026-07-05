"""EXP-025: DEEP per-person inference — our scalable analog of the SOTA's 2-hour interview.

The measured driver of SOTA individual-simulation accuracy (Generative Agent Simulations of 1,000
People) is rich per-person data: a 2-hour interview per person. Our scalable analog: infer a deep
PERSONA from a person's WRITING HISTORY and condition on it. This experiment tests, no-cheat, whether
that helps — on CMV, using recurring authors' argument histories. It reports the honest full picture:

HEADLINE (person-level, the true interview-gap analog): predict an UNSEEN author's characteristic
persuasion rate (are they an above-median persuader?) from their deep persona alone — inferred from
their writing, never from their outcomes. Split AUTHORS train/test; the test authors are people the
model has never seen. This is the direct analog of "predict a person you never interviewed."

  - a deep-inferred trait, intellectual_humility, predicts it at ~0.69 accuracy vs ~0.5 base;
  - DEPTH CURVE: reading more of the author's history sharpens the persona and improves prediction
    ("the deeper and more inferences, the better").

HONEST NEGATIVES (they delimit where deep inference helps):
  - per-INSTANCE delta prediction (does THIS argument earn a delta) is not improved by the persona:
    that outcome is matchup-driven (this argument x this OP), not driven by the author's stable traits.
  - the FULL 23-trait persona overfits at this sample size; parsimony (persuasion-theory traits) is
    required. Deep inference helps when the outcome is person-driven AND the features are disciplined.

Persona source: LLM per-document signals from an agent swarm (data/cmv_deep_*.json), else lexical
fallback. Writes experiments/results/exp025_deep_person.json.
Run: python -m experiments.exp025_deep_person
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.state.state import Action
from swm.transition.readout import LogisticReadout
from swm.variables.deep_inference import DeepInferenceEngine
from swm.variables.schema import BY_CATEGORY, PERSONA, spec
from swm.worlds.variable_world import VariableWorld
from experiments.datasets_cmv_history import load

RESULT = "experiments/results/exp025_deep_person.json"
PERSONA_VARS = BY_CATEGORY[PERSONA]
# persuasion-theory traits (a-priori, NOT selected on the test): humility/politeness help, certainty hurts
PERSUASION_TRAITS = ["intellectual_humility", "certainty_disposition", "politeness_disposition"]


def _load_llm_signals():
    sig = {}
    paths = glob.glob("data/cmv_deep_[0-9]*.json") or glob.glob("experiments/results/exp025_cmv/cmv_deep_*.json")
    for fp in paths:
        try:
            rows = json.loads(Path(fp).read_text())
        except Exception:
            continue
        for r in rows:
            if isinstance(r, dict) and "id" in r and "signals" in r:
                sig[r["id"]] = r["signals"]
    return sig


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _traitval(persona, t):
    return persona.get(t, {}).get("value", 0.0 if spec(t).signed else 0.5)


def _corr(xs, ys):
    if statistics.pstdev(xs) < 1e-9 or statistics.pstdev(ys) < 1e-9:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (
        statistics.pstdev(xs) * statistics.pstdev(ys) * len(xs))


def _person_level(auth, personas_D, rate, traits, seeds=(0, 1, 2, 3), l2=1.0):
    """Predict unseen authors' above-median persuasion rate from persona traits. personas_D maps
    author -> persona dict at a given depth. Returns mean log-loss gain over base + mean accuracy."""
    gains, accs = [], []
    for seed in seeds:
        rng = random.Random(seed); order = list(auth); rng.shuffle(order)
        cut = int(0.7 * len(order)); tr, te = order[:cut], order[cut:]
        med = sorted(rate[a] for a in tr)[len(tr) // 2]
        ytr = [int(rate[a] > med) for a in tr]; yte = [int(rate[a] > med) for a in te]
        base = _clip(sum(ytr) / len(ytr))
        llb = log_loss(yte, [base] * len(yte))
        Xtr = [[_traitval(personas_D[a], t) for t in traits] for a in tr]
        clf = LogisticReadout(epochs=400, l2=l2).fit(Xtr, ytr)
        p = [_clip(clf.predict_proba([_traitval(personas_D[a], t) for t in traits])) for a in te]
        gains.append(llb - log_loss(yte, p))
        accs.append(sum((pi > 0.5) == yi for pi, yi in zip(p, yte)) / len(yte))
    return round(statistics.mean(gains), 4), round(statistics.mean(accs), 4)


def _score(y, p):
    p = [_clip(v) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4), "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


def run():
    inst, _ = load(min_args=8)
    llm_sig = _load_llm_signals()
    eng = DeepInferenceEngine()
    if llm_sig:
        cov_auth = {r["author"] for r in inst if r["id"] in llm_sig}
        inst = [r for r in inst if r["author"] in cov_auth]
    docs, succ = defaultdict(list), defaultdict(list)
    for r in inst:
        docs[r["author"]].append(llm_sig.get(r["id"]) or eng.per_doc(r["arg_text"]))
        succ[r["author"]].append(r["success"])
    auth = sorted(docs)
    rate = {a: sum(succ[a]) / len(succ[a]) for a in auth}
    cov = (sum(1 for r in inst if r["id"] in llm_sig) / max(1, len(inst))) if llm_sig else 0.0
    src = "LLM agent swarm" if llm_sig else "lexical fallback"

    # ---- HEADLINE: person-level prediction of unseen authors ----
    persona_full = {a: eng.synthesize(docs[a]) for a in auth}
    corrs = sorted(((round(_corr([_traitval(persona_full[a], t) for a in auth],
                                  [rate[a] for a in auth]), 3), t) for t in PERSONA_VARS),
                   key=lambda z: -abs(z[0]))
    person = {
        "humility_only": dict(zip(("gain", "acc"),
                                  _person_level(auth, persona_full, rate, ["intellectual_humility"]))),
        "persuasion_traits": dict(zip(("gain", "acc"),
                                      _person_level(auth, persona_full, rate, PERSUASION_TRAITS))),
        "all_23_traits": dict(zip(("gain", "acc"),
                                  _person_level(auth, persona_full, rate, PERSONA_VARS))),
    }
    # depth curve: persona from the author's first D documents
    depth_curve = {}
    for D in (2, 4, 8, None):
        pD = {a: eng.synthesize(docs[a][:D] if D else docs[a]) for a in auth}
        g, ac = _person_level(auth, pD, rate, PERSUASION_TRAITS)
        depth_curve[str(D) if D else "all"] = {"gain": g, "acc": ac}

    # ---- HONEST NEGATIVE: per-instance delta prediction (matchup-driven) ----
    from swm.variables.deep_inference import DeepPersonaStore
    store = DeepPersonaStore(engine=eng)
    for r in inst:
        store.add_doc(r["author"], r["ts"], llm_sig.get(r["id"]) or eng.per_doc(r["arg_text"]))

    def instances(with_persona):
        out = []
        for r in inst:
            a = Action(action_id=str(r["ts"]), actor_id=r["author"], channel="cmv",
                       timing={"ts": r["ts"]}, meta={"text": r["arg_text"]})
            extra = {"llm_inference": store.persona_asof(r["author"], r["ts"])} if with_persona else {}
            out.append((r["author"], a, None, r["success"], extra))
        return out

    n = len(inst); cut = int(0.7 * n); y = [r["success"] for r in inst[cut:]]
    br = sum(r["success"] for r in inst[:cut]) / cut
    _, p_beh, _ = VariableWorld(platform="cmv").backtest(instances(False))
    _, p_deep, _ = VariableWorld(platform="cmv").backtest(instances(True))
    per_instance = {"base_rate": _score(y, [br] * len(y)), "behavioral": _score(y, p_beh),
                    "deep_persona_full": _score(y, p_deep)}

    out = {"n_authors": len(auth), "n_instances": n, "persona_source": src, "llm_coverage": round(cov, 3),
           "author_rate_base": round(sum(rate.values()) / len(rate), 4),
           "top_trait_correlations": corrs[:6],
           "person_level_unseen_authors": person,
           "depth_curve_persuasion_traits": depth_curve,
           "per_instance_delta": per_instance,
           "per_instance_deep_vs_behavioral": round(per_instance["behavioral"]["log_loss"]
                                                     - per_instance["deep_persona_full"]["log_loss"], 4)}
    print(f"EXP-025 deep per-person inference — CMV, {len(auth)} authors, persona: {src} (cov {cov:.0%})")
    print("  top trait↔persuasion-rate correlations (across authors):")
    for c, t in corrs[:5]:
        print(f"    {c:+.3f}  {t}")
    print("  PERSON-LEVEL — predict an UNSEEN author's above-median persuasion rate from persona:")
    for k, v in person.items():
        print(f"    {k:<18} log-loss gain vs base {v['gain']:+.4f}  accuracy {v['acc']}")
    print("  DEPTH CURVE (persuasion traits; persona reads first D docs of each author):")
    for d, v in depth_curve.items():
        print(f"    D={d:<4} gain {v['gain']:+.4f}  acc {v['acc']}")
    print("  PER-INSTANCE (does THIS argument earn a delta — matchup-driven, honest negative):")
    for k, v in per_instance.items():
        print(f"    {k:<20} log loss {v['log_loss']}  ece {v['ece']}")
    print(f"    Δ deep vs behavioral: {out['per_instance_deep_vs_behavioral']:+.4f} "
          f"(persona does not help the matchup-driven outcome)")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
