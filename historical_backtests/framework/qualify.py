"""Full-system execution proof + qualification gates (per-row, machine-verifiable).

A row QUALIFIES only when the canonical production facade demonstrably ran the complete call
graph: every current production phase carries a PhaseExecutionRecord, no causally-relevant phase
is blocked, the terminal came from simulated world states through the first-passage readout,
actor decisions actually executed where the plan declared strategic actors, and the particle
floor completed. Anything else is preserved as a visible failure — never counted as a forecast.
"""
from __future__ import annotations

import time


def current_phase_contract() -> list:
    """The CURRENT production phase list (never a frozen historical number)."""
    from swm.world_model_v2.unified_runtime import _PHASES
    return list(_PHASES)


def extract_proof(res, *, entrypoint: str, runtime_commit: str) -> dict:
    prov = getattr(res, "provenance", None) or {}
    recs = prov.get("phase_execution_records") or {}
    census = prov.get("operator_delta_census") or {}
    evt = prov.get("event_time") or {}
    manifest = prov.get("active_component_manifest") or {}
    actor_census = census.get("production_actor_policy") or {}
    n_actor_deltas = int(actor_census.get("n_deltas") or 0)
    return {
        "entrypoint": entrypoint,
        "runtime_fingerprint": prov.get("runtime"),
        "implementation_commit": runtime_commit,
        "simulation_status": getattr(res, "simulation_status", None),
        "phase_execution_records": {p: {k: r.get(k) for k in
                                        ("relevant", "execution_status", "n_state_deltas",
                                         "no_op_reason", "terminal_influence",
                                         "implementation_version")}
                                    for p, r in recs.items()},
        "n_phases_recorded": len(recs),
        "operator_delta_census": {op: {"n_deltas": c.get("n_deltas"),
                                       "event_types": c.get("event_types")}
                                  for op, c in census.items()},
        "n_actor_action_deltas": n_actor_deltas,
        "actors_with_views": len(actor_census.get("fields_written") or []) // 2,
        "n_particles": evt.get("n_particles"),
        "event_time_readout": {k: evt.get(k) for k in
                               ("p_event_by_deadline", "p_censored", "occurrence_resolves",
                                "n_particles")} if evt else None,
        "terminal_source": ("simulated_world_states/first_passage_readout" if evt
                            else "simulated_world_states/terminal_projection"
                            if getattr(res, "raw_distribution", None) else "missing"),
        "fully_integrated": prov.get("fully_integrated"),
        "integration_failures": prov.get("phase_integration_failures") or [],
        "fallbacks_used": list(getattr(res, "fallbacks_used", []) or [])[:6],
        "manifest_executed": {p: bool(m.get("executed")) for p, m in manifest.items()},
        "proof_at": time.time(),
    }


def qualify(proof: dict, *, min_particles: int = 200) -> tuple:
    """(qualified: bool, reasons: list) — every gate explicit."""
    reasons = []
    contract = current_phase_contract()
    if proof.get("simulation_status") not in ("completed", "completed_with_degradation"):
        reasons.append(f"simulation_status={proof.get('simulation_status')}")
    recs = proof.get("phase_execution_records") or {}
    missing = [p for p in contract if p not in recs]
    if missing:
        reasons.append(f"missing_phase_records:{missing}")
    for p, r in recs.items():
        if r.get("relevant") and str(r.get("execution_status", "")).startswith(
                ("blocked", "execution_failed")):
            reasons.append(f"relevant_phase_blocked:{p}:{r.get('execution_status')}")
    if proof.get("terminal_source") == "missing":
        reasons.append("no_terminal_from_simulated_worlds")
    evt = proof.get("event_time_readout")
    if not evt:
        reasons.append("no_first_passage_readout(binary deadline questions must route event-time)")
    np_ = proof.get("n_particles")
    if not isinstance(np_, int) or np_ < min_particles:
        reasons.append(f"particle_floor:{np_}<{min_particles}")
    p4 = recs.get("phase4_actor_policy") or {}
    if p4.get("relevant") and p4.get("execution_status") == "causally_active" \
            and proof.get("n_actor_action_deltas", 0) <= 0:
        reasons.append("actor_decisions_expected_but_no_actor_action_deltas")
    if p4.get("relevant") and str(p4.get("execution_status", "")).startswith("blocked"):
        reasons.append("actor_policy_blocked")
    return (len(reasons) == 0), reasons
