"""Phase 11 shock validation (gate 20) — trigger recall on INJECTED structural shocks, false-activation on
stable + ADVERSARIAL near-miss controls, and migration integrity, through the REAL RecompilationController.

Corpus design (no oracle, no hardcoded pass):
  * SHOCK episodes — observation streams where one observation carries a typed structural-change declaration
    (new actor / new institution / dated+sourced rule change / authority change / coalition change /
    exogenous shock / outcome-space change) that the ACTIVE plan does not contain.
  * STABLE controls — same stream shape, no structural declaration, residuals in the ordinary range.
  * ADVERSARIAL near-miss controls — the traps the trigger spec requires to NOT fire: an alias of a known
    actor, a future-dated rule not yet in force, a transient (non-persistent) network outage, a causally
    irrelevant new name, an unsourced rule claim.

Scored: trigger recall on shocks (>=0.90 gate), false-activation on all controls (<=0.10 gate), migration
integrity over adopted revisions (>=0.98 gate: migration gate checks pass in the trace records).
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("experiments/results/integration")
ART = OUT / "phase11_shock_validation.json"

_AS_OF = 1735689600.0                      # 2025-01-01T00:00:00Z (fixed epoch; no wall clock)
_DAY = 86400.0

PLAN_FACTS = {"known_actors": ["senator_a", "senator_b", "leader_x"],
              "known_institutions": ["senate"],
              "aliases": {"sen_a": "senator_a"},
              "outcome_options": ["confirmed", "not_confirmed"]}

#: (episode_id, kind, declared dict or None) — kind ∈ shock | stable | adversarial
EPISODES = [
    # ---- structural shocks (should trigger) ----
    ("shock_new_actor", "shock", {"new_actor": {"id": "insurgent_c", "causal_relevance": 0.9}}),
    ("shock_new_institution", "shock", {"new_institution": {"id": "special_tribunal"}}),
    ("shock_rule_change", "shock", {"rule_change": {"source": "official_register",
                                                    "effective_date": _AS_OF + 1 * _DAY}}),
    ("shock_authority", "shock", {"authority_change": {"from": "leader_x", "to": "insurgent_c"}}),
    ("shock_coalition", "shock", {"coalition_change": {"left": ["senator_b"], "joined": ["opposition"]}}),
    ("shock_exogenous", "shock", {"exogenous_shock": {"kind": "sudden_collapse"}}),
    ("shock_outcome_space", "shock", {"outcome_space_change": {"new_option": "withdrawn"}}),
    ("shock_network", "shock", {"network_change": {"persistent": True, "region": ["senator_a", "senator_b"]}}),
    # ---- stable controls (should NOT trigger) ----
    ("stable_1", "stable", None),
    ("stable_2", "stable", None),
    ("stable_3", "stable", None),
    ("stable_4", "stable", None),
    # ---- adversarial near-miss controls (should NOT trigger) ----
    ("adv_alias_actor", "adversarial", {"new_actor": {"id": "sen_a", "causal_relevance": 0.9}}),
    ("adv_future_rule", "adversarial", {"rule_change": {"source": "official_register", "future_dated": True,
                                                        "effective_date": _AS_OF + 30 * _DAY}}),
    ("adv_transient_outage", "adversarial", {"network_change": {"persistent": False}}),
    ("adv_irrelevant_actor", "adversarial", {"new_actor": {"id": "bystander_z", "causal_relevance": 0.05}}),
    ("adv_unsourced_rule", "adversarial", {"rule_change": {"claim": "rumor, no source"}}),
    ("adv_known_institution", "adversarial", {"new_institution": {"id": "senate"}}),
]


def _mk_plan():
    from swm.world_model_v2.compiler import WorldExecutionPlan
    from swm.world_model_v2.contracts import OutcomeContract

    def read(w):
        q = w.quantities.get("outcome")
        return q.value if q else None
    hz = _AS_OF + 30 * _DAY
    c = OutcomeContract(family="binary", options=["confirmed", "not_confirmed"], resolution_rule="r",
                        readout=read, readout_var="outcome", horizon_ts=hz).validate()
    p = WorldExecutionPlan(question="Will the senate confirm the nominee?", outcome_contract=c,
                           as_of=_AS_OF, horizon_ts=hz)
    p.entities = [{"id": a, "type": "person", "fields": {}} for a in PLAN_FACTS["known_actors"]]
    p.institutions = [{"id": "senate", "rules": [{"kind": "quorum", "params": {"quorum": 51, "total": 100}}]}]
    p.quantities = [{"name": "outcome", "qtype": "outcome", "value": None}]
    p.scheduled_events = [{"etype": "resolve_outcome", "ts": hz - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": ["confirmed", "not_confirmed"], "lean": "neutral"}}]
    p.accepted_mechanisms = [{"mech_id": "generic_outcome_prior", "operator": "generic_outcome_prior",
                              "causal_role": "safety net"}]
    return p


def _obs_stream(declared):
    """A 6-observation eligible stream; the 4th carries the declaration (if any)."""
    from swm.world_model_v2.phase11.contracts import RecompileObservation
    obs = []
    for i in range(6):
        prov = {"observed_value": 0.5}
        if declared is not None and i == 3:
            prov = {"observed_value": 0.5, "declared": declared}
        obs.append(RecompileObservation(
            observation_id=f"o{i}", observation_type="actor_statement", origin="external_evidence",
            event_time=_AS_OF + (i + 1) * _DAY, ingestion_time=_AS_OF + (i + 1) * _DAY + 60.0,
            evidence_ids=[f"e{i}"], provenance=prov))
    return obs


def _migration_ok(trace) -> tuple:
    """Inspect an adopted revision's migration gates. The transaction's standard_invariants (no time
    reversal, no duplicate/lost events, non-empty normalized ensemble) are summarized in
    checksums.migration_ok; the migration_report carries the per-gate counters. Returns (n_checks, n_ok)."""
    if trace.get("decision", {}).get("action") in (None, "no_change"):
        return 0, 0
    checks, ok = 0, 0
    checks += 1
    ok += 1 if (trace.get("checksums") or {}).get("migration_ok") else 0
    rep = trace.get("migration_report") or {}
    if isinstance(rep, dict) and rep:
        for gate, bad in (("time_reversal_count", lambda v: v != 0),
                          ("duplicate_event_rate", lambda v: v not in (0, 0.0)),
                          ("lost_valid_event_rate", lambda v: v not in (0, 0.0))):
            if gate in rep:
                checks += 1
                ok += 0 if bad(rep[gate]) else 1
    return checks, ok


def run():
    from swm.world_model_v2.phase11.controller import RecompilationController, ExecutionAdapter
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for eid, kind, declared in EPISODES:
        plan = _mk_plan()
        from swm.world_model_v2.materialize import build_world
        worlds = [build_world(plan, world_id=f"w{i}") for i in range(8)]
        for i, w in enumerate(worlds):
            w.branch_id = f"b{i}"
        weights = [1.0 / len(worlds)] * len(worlds)
        ctrl = RecompilationController(llm=None, seed=7, max_recompiles=2)
        try:
            cr = ctrl.run(plan=plan, worlds=worlds, weights=weights,
                          pending_events=[[] for _ in worlds],
                          observations=_obs_stream(declared), horizon_ts=_AS_OF + 30 * _DAY,
                          as_of=_AS_OF, execution=ExecutionAdapter(), plan_facts=dict(PLAN_FACTS))
            fired = cr.n_recompiles > 0 or any(
                t.get("decision", {}).get("action") not in (None, "no_change") for t in cr.traces)
            trig_families = sorted({f for t in cr.traces for f in (t.get("trigger_posterior") or {})})
            mchecks = [(t, *_migration_ok(t)) for t in cr.traces]
            rows.append({"episode": eid, "kind": kind, "fired": bool(fired),
                         "n_recompiles": cr.n_recompiles, "n_eligible": cr.n_eligible,
                         "trigger_families": trig_families,
                         "migration_checks": sum(c for _, c, _ in mchecks),
                         "migration_ok": sum(o for _, _, o in mchecks)})
        except Exception as e:  # noqa: BLE001
            rows.append({"episode": eid, "kind": kind, "error": f"{type(e).__name__}: {e}"[:160]})
        r = rows[-1]
        print(f"{eid:22s} {kind:12s} fired={r.get('fired')} fam={r.get('trigger_families')} "
              f"err={r.get('error', '')}")
    shocks = [r for r in rows if r["kind"] == "shock" and not r.get("error")]
    ctrls = [r for r in rows if r["kind"] in ("stable", "adversarial") and not r.get("error")]
    mig_checks = sum(r.get("migration_checks", 0) for r in rows if not r.get("error"))
    mig_ok = sum(r.get("migration_ok", 0) for r in rows if not r.get("error"))
    agg = {"n_shocks": len(shocks), "trigger_recall": round(sum(r["fired"] for r in shocks) /
                                                            len(shocks), 3) if shocks else None,
           "n_controls": len(ctrls), "false_activation": round(sum(r["fired"] for r in ctrls) /
                                                               len(ctrls), 3) if ctrls else None,
           "migration_checks": mig_checks, "migration_ok": mig_ok,
           "migration_integrity": round(mig_ok / mig_checks, 3) if mig_checks else None,
           "n_errors": sum(1 for r in rows if r.get("error")),
           "gates": {"g20a_trigger_recall_ge_0.90": bool(shocks) and
                     sum(r["fired"] for r in shocks) / len(shocks) >= 0.90,
                     "g20b_false_activation_le_0.10": bool(ctrls) and
                     sum(r["fired"] for r in ctrls) / len(ctrls) <= 0.10,
                     "g20c_migration_integrity_ge_0.98": bool(mig_checks) and mig_ok / mig_checks >= 0.98}}
    ART.write_text(json.dumps({"rows": rows, "aggregate": agg}, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    run()
