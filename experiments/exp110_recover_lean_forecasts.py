"""EXP-110: recover the five lean BTF-3 probabilities from EXISTING checkpoints under the
forecast-availability contract — no reruns, no new LLM calls, no leakage surface.

Each stored `simulation_result` already contains everything the layered recovery needs: the
per-model raw distributions (weighted terminal shares incl. unresolved_mechanism mass), the
evidence-updated posterior means recorded by the phase-3 manifest, and the grounded prior
provenance. This harness re-derives the labeled headline per model with the SAME
`forecast_recovery` code the runtime now runs natively, aggregates per-model estimates exactly
like the ensemble assembly, and writes the recovered rows next to the original checkpoints
(originals untouched). The original as_of/evidence/seeds are untouched by construction — this
reads artifacts, it cannot see resolutions (scoring joins later in exp109 as always).

Also applies to the FF checkpoints when present (the FF arm is never rerun for this).

Run: python -m experiments.exp110_recover_lean_forecasts
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from swm.world_model_v2.forecast_recovery import recover_forecast

LEAN_DIR = Path("experiments/results/exp108_checkpoints")
FF_DIR = Path("experiments/results/exp107_checkpoints")
OUT = Path("experiments/results/exp110_recovered_forecasts.json")

_P3 = re.compile(r"prior ([0-9.]+)→post ([0-9.]+)")


def _posterior_mean(model_prov: dict):
    """The phase-3 posterior mean as recorded by the manifest ('N eff obs; prior a→post b')."""
    ph3 = ((model_prov.get("active_component_manifest") or {}).get("phase3_posterior") or {})
    m = _P3.search(str(ph3.get("reason") or ""))
    executed = bool(ph3.get("executed"))
    return (float(m.group(2)) if (m and executed) else None,
            float(m.group(1)) if m else None, executed)


def recover_checkpoint(path: Path) -> dict:
    d = json.loads(path.read_text())
    r = d["simulation_result"]
    prov = r.get("provenance") or {}
    pm = prov.get("per_model_provenance") or {}
    options = None
    per_model = {}
    for mid, mp in pm.items():
        dist = None
        # per-model distributions ride the structural block when present; fall back to the
        # ensemble headline for single-model runs
        sd = r.get("structural_disagreement") or {}
        dist = sd.get(mid) or {}
        post_mean, prior_mean, executed = _posterior_mean(mp)
        rec = recover_forecast(distribution=dist, options=options, unresolved_mass=None,
                               posterior_mean=post_mean, posterior_n_eff=1 if post_mean else 0,
                               prior_mean=prior_mean if post_mean is None else None,
                               prior_source_class="reference_class" if prior_mean else "")
        if rec is not None:
            per_model[mid] = {"probability": rec.probability, "source": rec.probability_source,
                              "grade": rec.grounding_grade,
                              "conditional_on_resolved":
                                  rec.probability_conditional_on_resolved,
                              "unresolved_mass": rec.unresolved_mass,
                              "interval": rec.uncertainty_interval,
                              "weight_sensitive": rec.weight_sensitive}
    row = {"qid": d["metrics"]["qid"], "question": d["metrics"]["question"][:110],
           "original_status": d["metrics"]["status"],
           "original_p_raw": d["metrics"].get("p_raw"),
           "per_model": per_model}
    if per_model:
        ps = [v["probability"] for v in per_model.values() if v["probability"] is not None]
        if ps:
            row["recovered_probability"] = round(sum(ps) / len(ps), 4)
            row["recovered_interval"] = [round(min(ps), 4), round(max(ps), 4)]
            row["weight_sensitive"] = (min(ps) < 0.5 < max(ps)) \
                or any(v["weight_sensitive"] for v in per_model.values())
            srcs = sorted({v["source"] for v in per_model.values() if v["source"]})
            row["probability_source"] = srcs[0] if len(srcs) == 1 else "mixed:" + "+".join(srcs)
            order = ["grounded", "partially_grounded", "exploratory", "ungrounded"]
            grades = [v["grade"] for v in per_model.values() if v["grade"]]
            row["grounding_grade"] = max(grades, key=lambda g: order.index(g)
                                         if g in order else 2) if grades else ""
            row["aggregation"] = "equal_weight_mixture_of_per_model_recovered_probabilities " \
                                 "(same rule as the runtime's ensemble recovery)"
    return row


def run() -> dict:
    out = {"experiment": "EXP-110 forecast recovery from existing checkpoints "
                         "(no reruns, no new calls, original as_of/evidence untouched)",
           "lean": [], "full_fidelity": []}
    for arm, d in (("lean", LEAN_DIR), ("full_fidelity", FF_DIR)):
        for p in sorted(d.glob("*.json")):
            try:
                out[arm].append(recover_checkpoint(p))
            except Exception as e:  # noqa: BLE001 — a bad checkpoint is reported, not fatal
                out[arm].append({"qid": p.stem, "error": f"{type(e).__name__}: {e}"[:200]})
    OUT.write_text(json.dumps(out, indent=1, default=str))
    for arm in ("lean", "full_fidelity"):
        for row in out[arm]:
            print(f"{arm:14s} {row.get('qid', '')[:8]} {row.get('original_status', ''):16s} "
                  f"p={row.get('recovered_probability')} "
                  f"src={row.get('probability_source', '')} "
                  f"grade={row.get('grounding_grade', '')} "
                  f"interval={row.get('recovered_interval')}")
    return out


if __name__ == "__main__":
    run()
