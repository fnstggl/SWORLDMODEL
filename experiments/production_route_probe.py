"""Live production-route probe — proves the final default path end to end.

Runs the PUBLIC `simulate_world` (no special wiring) and records what actually executed:
requested/actual consequence mode, scenario schema id/source, cascade manifest, run
classification, the epistemic contract, and the hard invariants. Offline mode (no key)
exercises the minimal-schema recovery and the honest degradation labels through the same
route.

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/production_route_probe.py live
    PYTHONPATH=. python experiments/production_route_probe.py offline
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

RESULTS = Path("experiments/results")

QUESTION = ("Will Halvorsen Marine's works council and management sign the revised "
            "shift-pattern agreement before the March deadline?")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    llm = None
    if mode == "live":
        from swm.api.deepseek_backend import deepseek_chat_fn
        llm = deepseek_chat_fn(temperature=0.9, max_tokens=4000)
    from swm.world_model_v2.unified_runtime import simulate_world
    t0 = _time.monotonic()
    res = simulate_world(QUESTION, llm=llm, as_of="2026-07-01", horizon="2026-09-30",
                         compute_budget={"n_particles": 4}, seed=17)
    prov = res.provenance or {}
    crep = prov.get("consequence_report") or {}
    probe = {
        "schema_version": "production.route.probe.v2",
        "mode": mode, "question": QUESTION,
        "simulation_status": res.simulation_status,
        "distribution": getattr(res, "raw_distribution", None),
        "run_classification": prov.get("run_classification"),
        "epistemic_contract": prov.get("epistemic_contract"),
        "consequence_mode": {"requested": crep.get("requested_mode"),
                             "actual": crep.get("actual_mode")},
        "scenario_schema": {"id": crep.get("scenario_schema_id"),
                            "recovery": prov.get("scenario_schema_recovery", "")},
        "generated_manifests": prov.get("generated_manifests"),
        "joint_world": (prov.get("joint_world")
                        or (getattr(res, "plan", None) and {})),
        "invariants": {k: crep.get(k) for k in
                       ("fixed_ontology_uses", "legacy_scalar_writes",
                        "human_reactions_written_directly", "tier1_numeric_fallbacks",
                        "tier2_numeric_fallbacks")},
        "actor_policy_report": {k: (prov.get("actor_policy_report") or {}).get(k)
                                for k in ("requested_actor_policy_mode",
                                          "actual_actor_policy_mode",
                                          "actors_routed_qualitatively",
                                          "tier1_numeric_fallbacks",
                                          "tier2_numeric_fallbacks")},
        "wall_s": round(_time.monotonic() - t0, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"production_route_probe_{mode}.json"
    path.write_text(json.dumps(probe, indent=1, default=str))
    print(json.dumps({k: probe[k] for k in ("simulation_status", "consequence_mode",
                                            "invariants", "wall_s")}, indent=1))
    print("run_class:", (probe["run_classification"] or {}).get("run_class"))
    print("actor_simulation:", (probe["epistemic_contract"] or {}).get("actor_simulation"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
