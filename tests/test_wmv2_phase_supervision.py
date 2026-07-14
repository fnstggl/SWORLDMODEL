"""Mandatory phase supervision tests: every phase gets a record, no silent disappearance, explicit no-ops,
blocked relevant phases surface as integration failures, manifest derived from records."""
from __future__ import annotations
from types import SimpleNamespace

from swm.world_model_v2 import phase_supervision as PS
from swm.world_model_v2.compiler import WorldExecutionPlan
from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.state import parse_time

AS_OF = parse_time("2025-01-01")
HZ = parse_time("2025-02-01")


def _plan(processes=(), institutions=None, deps=None, question="q"):
    def read(w):
        q = w.quantities.get("outcome")
        return q.value if q else None
    c = OutcomeContract(family="binary", options=["yes", "no"], resolution_rule="r",
                        readout=read, readout_var="outcome", horizon_ts=HZ).validate()
    p = WorldExecutionPlan(question=question, outcome_contract=c, as_of=AS_OF, horizon_ts=HZ)
    p.mechanism_choices = [{"process": pr} for pr in processes]
    p.institutions = institutions or []
    p.scheduled_events = [{"etype": "resolve_outcome", "ts": HZ - 1.0, "participants": [],
                           "payload": {"outcome_var": "outcome", "family": "binary",
                                       "options": ["yes", "no"], "lean": "neutral"}}]
    p.provenance = {"causal_dependencies": deps or {}}
    return p


def _res(census=None):
    return SimpleNamespace(provenance={"operator_delta_census": census or {}},
                           support_grade="empirically_supported", limitations=[])


def test_every_phase_gets_a_record_every_run():
    recs = PS.assess(_plan(), has_as_of=True, has_bundle=True)
    assert set(recs) == set(PS.PHASES)                       # no phase can silently disappear
    for r in recs.values():
        assert r.relevance_assessed and r.execution_status in PS.STATUSES


def test_irrelevant_phase_is_explicit_no_op():
    recs = PS.assess(_plan(), has_as_of=True, has_bundle=True)
    r = recs["phase10_institutions"]
    assert not r.relevant and r.execution_status == "no_op_causally_irrelevant" and r.no_op_reason


def test_relevant_phase_without_declared_state_is_blocked_missing_state():
    p = _plan(deps={"institutional_decision_process": True},
              question="Will the senate confirm the nominee?")  # relevant but NO institution declared
    recs = PS.assess(p, has_as_of=True, has_bundle=True)
    assert recs["phase10_institutions"].execution_status == "blocked_missing_state"


def test_relevant_executed_phase_is_causally_active_with_delta_evidence():
    p = _plan(deps={"institutional_decision_process": True},
              question="Will the senate confirm the nominee?",
              institutions=[{"id": "senate", "rules": [{"kind": "quorum", "params": {"quorum": 51}}]}])
    p.entities = [{"id": "senator", "type": "person", "fields": {}}]
    p.accepted_mechanisms = [
        {"operator": "institutional_decision", "mech_id": "x", "parameter_source": "declared rule"},
        {"operator": "production_actor_policy", "mech_id": "actor", "parameter_source": "broad prior"},
    ]
    recs = PS.assess(p, has_as_of=True, has_bundle=True)
    out = PS.finalize(recs, p, _res({
        "institutional_decision": {"n_deltas": 30, "fields_written": ["quantities[outcome]"],
                                   "event_types": ["institutional_decision"]},
        "production_actor_policy": {"n_deltas": 30, "fields_written": ["senator.current_action"],
                                    "event_types": ["actor_action"]},
        # p6 is co-required whenever an institutional/social phase is (a behavioral mechanism carries it)
        "structural_process_prior": {"n_deltas": 30, "fields_written": ["quantities[mech]"],
                                     "event_types": ["structural_process_prior"]}}),
        phase_meta={k: {"executed": True} for k in ("phase1_compiler", "phase2_evidence",
                                                    "phase3_posterior", "phase8_persistence",
                                                    "phase11_recompilation")})
    r = out["records"]["phase10_institutions"]
    assert r.execution_status == "causally_active" and r.n_state_deltas == 30
    assert r.terminal_influence == "direct_resolution"
    assert out["fully_integrated"]


def test_relevant_phase_with_no_deltas_is_blocked_and_fails_integration():
    p = _plan(deps={"institutional_decision_process": True},
              question="Will the senate confirm the nominee?",
              institutions=[{"id": "senate", "rules": []}])
    recs = PS.assess(p, has_as_of=True, has_bundle=True)
    out = PS.finalize(recs, p, _res({}), phase_meta={})
    r = out["records"]["phase10_institutions"]
    assert r.execution_status == "blocked_no_mechanism"
    assert not out["fully_integrated"]
    assert any(f["phase"] == "phase10_institutions" for f in out["integration_failures"])
    assert r.support_implication == "lowers_support:integration_failure"


def test_manifest_is_derived_from_records():
    recs = PS.assess(_plan(), has_as_of=False, has_bundle=False)
    out = PS.finalize(recs, _plan(), _res({}), phase_meta={})
    m = out["manifest"]
    assert set(m) == set(PS.PHASES)
    assert m["phase2_evidence"]["status"] in ("no_op_causally_irrelevant", "blocked_invalid_contract")
    for ph in PS.PHASES:
        assert m[ph]["status"] == out["records"][ph].execution_status


def test_p6_fallback_makes_required_process_executable():
    """A required social process no registry family answers must still execute (structural prior)."""
    from swm.world_model_v2.activation_synthesis import synthesize_activation
    from swm.world_model_v2.materialize import run_from_plan
    p = _plan(processes=["union_bargaining_pressure"],
              deps={"strategic_actor_decisions": True},
              question="Will the union reach a contract before the strike deadline?")
    p.quantities = [{"name": "outcome", "qtype": "outcome", "value": None}]
    p.accepted_mechanisms = [{"mech_id": "generic_outcome_prior", "operator": "generic_outcome_prior",
                              "causal_role": "safety net"}]
    synthesize_activation(p)
    assert any(e["etype"] == "structural_process_prior" for e in p.scheduled_events)
    assert any(f.get("family") == "structural_process_prior" for f in p.fallbacks_used)
    res, branches = run_from_plan(p, llm=None, seed=3)
    ops = {}
    for b in branches:
        for d in b.log:
            ops[d.operator] = ops.get(d.operator, 0) + 1
    assert ops.get("structural_process_prior", 0) > 0        # the process EXECUTES, never silently omitted


def test_unchecked_compiler_dependency_does_not_drive_relevance():
    from swm.world_model_v2.activation_synthesis import phase_requirements
    p = _plan(processes=["household_purchase_accumulation"],   # no population keyword anywhere
              deps={"aggregate_population_behavior": True})
    p.populations = [{"id": "buyers", "segments": [{"id": "s", "weight": 1.0, "differs_on": ["budget"]}]}]
    assert not phase_requirements(p)["phase9_populations"]["required"]


def test_question_level_population_semantics_drive_relevance():
    from swm.world_model_v2.activation_synthesis import phase_requirements
    p = _plan(processes=["household_purchase_accumulation"],
              question="Will household adoption exceed half of eligible households?")
    assert phase_requirements(p)["phase9_populations"]["required"]
