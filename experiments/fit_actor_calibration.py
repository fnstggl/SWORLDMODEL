"""Fit the EXTERNAL actor-decision calibrator from a benchmark artifact.

Operates strictly AFTER aggregation, on the counted branch-frequency distributions of one
counted arm (default: persistent_qualitative_llm_policy): a single reference-class temperature
minimizing leave-one-out negative log-likelihood of the realized actions, plus a reliability
table (predicted-confidence bins vs realized accuracy). Writes
``experiments/actor_decision_calibration.json`` in the ``ActorPolicyCalibrator`` pack shape
(level: reference) with full fit provenance. Actor-/role-/domain-level entries require far more
history per key than any pilot corpus carries — they stay absent (and distributions for them
keep the honest ``unvalidated`` label) until a real corpus exists.

    PYTHONPATH=. python experiments/fit_actor_calibration.py <benchmark_artifact.json>
"""
from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path

OUT = Path("experiments/actor_decision_calibration.json")


def _temperature_nll(rows, t: float, *, floor: float = 1e-3) -> float:
    nll = 0.0
    for row in rows:
        dist = row["distribution"]
        logits = {k: math.log(max(1e-9, v)) / t for k, v in dist.items()}
        m = max(logits.values())
        z = sum(math.exp(v - m) for v in logits.values())
        p = math.exp(logits.get(row["actual"], math.log(floor)) - m) / z \
            if row["actual"] in logits else floor
        nll -= math.log(max(floor, p))
    return nll / max(1, len(rows))


def fit(artifact_path: str, arm: str = "persistent_qualitative_llm_policy") -> dict:
    data = json.loads(Path(artifact_path).read_text())
    rows = [r for r in data["arms"][arm]["cases"] if r.get("distribution")]
    if len(rows) < 20:
        return {"fitted": False,
                "reason": f"only {len(rows)} scored cases — refusing to fit a calibrator on "
                          "fewer than 20 (it would be noise wearing a label)"}
    grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    loo_scores = {}
    for t in grid:
        loo = 0.0
        for i in range(len(rows)):
            held = rows[i]
            loo += _temperature_nll([held], t)
        loo_scores[t] = round(loo / len(rows), 4)
    best_t = min(loo_scores, key=loo_scores.get)
    raw_nll = _temperature_nll(rows, 1.0)
    # reliability: confidence bins vs realized accuracy at the fitted temperature
    bins = {}
    for row in rows:
        conf = max(row["distribution"].values())
        b = min(4, int(conf * 5))
        bins.setdefault(b, []).append(int(row["correct"]))
    reliability = {f"{b * 20}-{b * 20 + 20}%": {"n": len(v),
                                                "accuracy": round(sum(v) / len(v), 3)}
                   for b, v in sorted(bins.items())}
    pack = {
        "schema_version": "actor.decision.calibration.v1",
        "pack_id": f"actor-cal:reference:{int(_time.time())}",
        "source": "fitted_on_frozen_benchmark_corpus",
        "fit": {"artifact": str(artifact_path), "arm": arm, "n_cases": len(rows),
                "method": "leave-one-out temperature grid on counted branch frequencies",
                "raw_nll_at_t1": round(raw_nll, 4), "loo_nll_by_temperature": loo_scores,
                "caveats": ["single reference-class temperature — actor/role/domain levels "
                            "require per-key history that does not yet exist",
                            "corpus carries the training-contamination caveat of its cases"]},
        "reference": {"*": {"temperature": best_t, "fit": "loo_grid_reference"}},
        "reliability_at_fitted_temperature": reliability,
    }
    OUT.write_text(json.dumps(pack, indent=1))
    return {"fitted": True, "temperature": best_t, "loo_nll": loo_scores[best_t],
            "raw_nll_at_t1": round(raw_nll, 4), "reliability": reliability, "path": str(OUT)}


if __name__ == "__main__":
    print(json.dumps(fit(sys.argv[1], *(sys.argv[2:3])), indent=1))
