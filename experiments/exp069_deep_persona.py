"""EXP-069: deep per-person inference — the interview-gap lever, measured.

SOTA individual simulation (Park et al., "Generative Agent Simulations of 1,000 People") reaches ~85%
normalized accuracy by conditioning an agent on a 2-HOUR INTERVIEW per person. We can't interview everyone;
our scalable analog is DEEP MULTI-PASS INFERENCE over a person's writing history (`swm/variables/
deep_inference.py`). This experiment measures, on REAL data, the two questions that decide whether that
lever is worth pulling:

  1. DOES DEPTH HELP? On 160 CMV authors with 8-25 documents each (real writing histories, per-document
     persona signals from an agent swarm), we predict a HELD-OUT document's persona facets from the persona
     inferred from the person's OTHER (prior, as-of) documents. If a person has a stable, learnable persona,
     this beats the population baseline; if DEPTH is the lever, the error falls as we condition on more of
     their history — the scalable analog of "more interview -> more accuracy".

  2. DOES IT PAY OFF DOWNSTREAM? On the 125 documents that carry a real persuasion outcome, does the
     challenger's DEEP persona predict whether their argument succeeds, better than a shallow (1-doc) or
     no-persona baseline?

Leakage-free by construction: a held-out action is predicted only from documents strictly before it.
Run: python -m experiments.exp069_deep_persona
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from swm.transition.readout import LogisticReadout
from swm.eval.metrics import brier_score, log_loss
from swm.variables.deep_inference import DeepInferenceEngine, DeepPersonaStore

META = "experiments/results/exp025_cmv/cmv_deep_meta.json"
SIG = "experiments/results/exp025_cmv/cmv_deep_signals.json"
COMMON = "experiments/results/exp021_cmv/cmv_common.json"
RESULT = "experiments/results/exp069_deep_persona.json"

FACETS = ["analytical_style", "epistemic_rigor", "intellectual_humility", "politeness_disposition",
          "certainty_disposition", "combativeness", "trait_openness", "verbosity"]


def _load():
    meta = json.load(open(META))
    sig = {r["id"]: r["signals"] for r in json.load(open(SIG))}
    common = {r["id"]: r for r in json.load(open(COMMON))}
    store = DeepPersonaStore(engine=DeepInferenceEngine())
    docs_by_auth = defaultdict(list)
    for m in meta:
        s = sig.get(m["id"], {})
        store.add_doc(m["author"], m["ts"], s)
        docs_by_auth[m["author"]].append((m["ts"], m["id"], s))
    for a in docs_by_auth:
        docs_by_auth[a].sort(key=lambda d: d[0])
    # population mean per facet
    pop = {}
    for f in FACETS:
        vals = [s[f]["value"] for s in sig.values() if f in s and isinstance(s[f], dict)]
        pop[f] = sum(vals) / len(vals) if vals else 0.5
    return store, docs_by_auth, common, pop, sig


def _facet(s, f):
    return s[f]["value"] if f in s and isinstance(s[f], dict) else None


def run():
    store, docs_by_auth, common, pop, sig = _load()

    # ---- 1. DOES DEPTH HELP? predict a held-out doc's facets from the as-of persona ----
    # (a) overall: persona (all prior) vs population vs confidence-blend
    e_persona = e_pop = e_blend = k = 0.0
    # (b) depth curve: for held-out docs deep in a corpus, cap the persona at j prior docs
    curve = {j: [0.0, 0] for j in (1, 2, 4, 8, 16)}
    for auth, docs in docs_by_auth.items():
        for idx, (ts, did, s) in enumerate(docs):
            if idx == 0:
                continue                                    # need prior history
            persona = store.persona_asof(auth, ts)          # all prior docs, as-of
            for f in FACETS:
                truth = _facet(s, f)
                if truth is None:
                    continue
                pv = persona.get(f, {}).get("value", pop[f])
                conf = persona.get(f, {}).get("confidence", 0.0)
                blend = conf * pv + (1 - conf) * pop[f]
                e_persona += abs(pv - truth); e_pop += abs(pop[f] - truth); e_blend += abs(blend - truth)
                k += 1
            if idx >= 8:                                     # depth curve on deep-corpus held-out docs
                for j in curve:
                    pj = store.persona_asof(auth, ts, max_docs=j)
                    err = n = 0.0
                    for f in FACETS:
                        truth = _facet(s, f)
                        if truth is None:
                            continue
                        err += abs(pj.get(f, {}).get("value", pop[f]) - truth); n += 1
                    if n:
                        curve[j][0] += err / n; curve[j][1] += 1
    depth_curve = {f"depth<={j}": round(v[0] / v[1], 4) for j, v in curve.items() if v[1]}
    part1 = {"n_predictions": int(k),
             "MAE_population_baseline": round(e_pop / k, 4),
             "MAE_deep_persona": round(e_persona / k, 4),
             "MAE_confidence_blend": round(e_blend / k, 4),
             "persona_beats_population_by": round((e_pop - e_persona) / k, 4),
             "depth_curve_MAE": depth_curve}

    # ---- 2. DOES IT PAY OFF DOWNSTREAM? challenger deep persona -> argument success ----
    labeled = [(did, common[did]) for did in common
               if did in {d[1] for docs in docs_by_auth.values() for d in docs}]
    # find each labeled doc's author + ts
    id_to = {}
    for auth, docs in docs_by_auth.items():
        for ts, did, s in docs:
            id_to[did] = (auth, ts, s)
    rows = []
    for did, c in labeled:
        if did not in id_to:
            continue
        auth, ts, s = id_to[did]
        rows.append({"auth": auth, "ts": ts, "y": int(c["success"]),
                     "deep": store.persona_asof(auth, ts),                 # persona from all prior docs
                     "shallow": store.persona_asof(auth, ts, max_docs=1)})  # persona from 1 prior doc
    rows.sort(key=lambda r: r["ts"])
    cut = int(0.6 * len(rows))
    tr, te = rows[:cut], rows[cut:]
    yte = [r["y"] for r in te]
    base = sum(r["y"] for r in tr) / len(tr) if tr else 0.5

    def feats(persona):
        return [persona.get(f, {}).get("value", pop[f]) for f in FACETS]

    def arm(keyname):
        X = [feats(r[keyname]) for r in tr]
        y = [r["y"] for r in tr]
        m = LogisticReadout(l2=0.5).fit(X, y)
        p = [m.predict_proba(feats(r[keyname])) for r in te]
        return {"log_loss": round(log_loss(yte, p), 4), "brier": round(brier_score(yte, p), 4)}

    part2 = {"n_labeled": len(rows), "n_test": len(te), "base_rate": round(sum(r["y"] for r in rows) / len(rows), 3),
             "base_rate_logloss": round(log_loss(yte, [base] * len(te)), 4),
             "shallow_1doc": arm("shallow"), "deep_full_history": arm("deep")}
    part2["deep_beats_base_logloss"] = round(part2["base_rate_logloss"] - part2["deep_full_history"]["log_loss"], 4)
    part2["deep_beats_shallow_logloss"] = round(part2["shallow_1doc"]["log_loss"] - part2["deep_full_history"]["log_loss"], 4)

    out = {"PART1_does_depth_help": part1, "PART2_downstream_payoff": part2}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-069  deep per-person inference — the interview-gap lever, measured on real CMV histories")
    print("  1. DOES DEPTH HELP? (predict a held-out doc's persona from prior docs)")
    print(f"     population baseline MAE = {part1['MAE_population_baseline']}")
    print(f"     deep persona        MAE = {part1['MAE_deep_persona']}  "
          f"(beats population by {part1['persona_beats_population_by']:+})")
    print(f"     confidence-blend    MAE = {part1['MAE_confidence_blend']}")
    print(f"     DEPTH CURVE (MAE vs # prior docs): {part1['depth_curve_MAE']}")
    print("  2. DOWNSTREAM PAYOFF (challenger deep persona -> argument success):")
    print(f"     n_labeled={part2['n_labeled']} test={part2['n_test']} base-rate={part2['base_rate']}")
    print(f"     base logloss={part2['base_rate_logloss']}  shallow(1doc)={part2['shallow_1doc']['log_loss']}  "
          f"deep={part2['deep_full_history']['log_loss']}")
    print(f"     deep beats base by {part2['deep_beats_base_logloss']:+}, beats shallow by "
          f"{part2['deep_beats_shallow_logloss']:+} log-loss")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
