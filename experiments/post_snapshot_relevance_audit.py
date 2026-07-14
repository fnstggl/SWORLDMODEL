"""Generate the outcome-free Phase relevance adjudication artifact."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.activation_corpus_200 import PHASE_FLAGS, QUESTIONS
from swm.world_model_v2.causal_relevance import adjudicate_question


OUT = Path("experiments/results/post_snapshot_benchmark/phase_relevance_audit.json")
FLAG_PHASE = {
    "p4": "phase4_actor_policy", "p6": "phase6_registry", "p7": "phase7_nonlinear",
    "p9pop": "phase9_populations", "p9net": "phase9_networks",
    "p10": "phase10_institutions", "p11": "phase11_recompilation",
}


def build():
    rows = []
    for qid, question, as_of, horizon, domain, family, labels in QUESTIONS:
        judged = adjudicate_question(question)
        runtime_labels = sorted(flag for flag, phase in FLAG_PHASE.items()
                                if judged[phase]["required"])
        disagreements = []
        for flag, phase in FLAG_PHASE.items():
            authored = flag in labels
            runtime = judged[phase]["required"]
            if authored != runtime:
                disagreements.append({"phase": phase, "authored": authored, "adjudicated": runtime,
                                      "review": ("requires manual semantic review; labels were not changed"
                                                 if qid != "sport_6" else
                                                 "authored p11 label conflicts with stated structural-change semantics")})
        rows.append({
            "qid": qid, "question": question, "domain": domain, "family": family,
            "as_of": as_of, "horizon": horizon, "authored_labels": sorted(labels),
            "adjudicated_labels": runtime_labels, "judgments": judged,
            "disagreements": disagreements,
        })
    metrics = {}
    for flag in PHASE_FLAGS:
        phase = FLAG_PHASE[flag]
        positive = [r for r in rows if flag in r["authored_labels"]]
        negative = [r for r in rows if flag not in r["authored_labels"]]
        tp = sum(r["judgments"][phase]["required"] for r in positive)
        fp = sum(r["judgments"][phase]["required"] for r in negative)
        metrics[phase] = {
            "n_authored_relevant": len(positive), "n_authored_irrelevant": len(negative),
            "recall_against_preserved_authored_labels": round(tp / len(positive), 6),
            "false_activation_against_preserved_authored_labels": round(fp / len(negative), 6),
            "recall_gate_ge_0_95": tp / len(positive) >= 0.95,
            "false_activation_gate_le_0_10": fp / len(negative) <= 0.10,
        }
    artifact = {
        "schema_version": "1.0", "outcome_data_accessed": False,
        "adjudication_basis": "question wording only; no qid, split, source, outcome, or compiler dependency can activate a phase",
        "authored_labels_mutated": False,
        "known_label_review": {
            "qid": "sport_6", "phase": "phase11_recompilation", "authored": True,
            "adjudicated": False,
            "reason": "An ordinary podium question contains no new actor, rule, coalition, regime, or structural-break cue."
        },
        "metrics": metrics, "all_recall_and_false_activation_gates_pass": all(
            m["recall_gate_ge_0_95"] and m["false_activation_gate_le_0_10"] for m in metrics.values()),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    tmp.replace(OUT)
    return artifact


if __name__ == "__main__":
    result = build()
    print(json.dumps(result["metrics"], indent=2))
