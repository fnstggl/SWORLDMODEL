"""Pins the rollout-viability invariant (root fix for the silent-empty-rollout class): after plan surgery
and operator instantiation, at least one outcome-capable (event, operator) pair must exist — otherwise the
plan is repaired in place (recorded), never rolled out unanswerable."""
from types import SimpleNamespace

from swm.world_model_v2.fallback import GenericOutcomeOperator
from swm.world_model_v2.materialize import ensure_outcome_pathway


def _plan(events, horizon_ts=2_000_000.0):
    return SimpleNamespace(scheduled_events=list(events), horizon_ts=horizon_ts,
                           outcome_contract=SimpleNamespace(readout_var="outcome", family="binary",
                                                            options=["Yes", "No"]),
                           provenance={"outcome_lean": "neutral"}, posterior_rate_particles=None)


def _resolver_event():
    return {"etype": "resolve_outcome", "ts": 1_999_999.0, "participants": [],
            "payload": {"outcome_var": "outcome", "family": "binary", "options": ["Yes", "No"],
                        "lean": "neutral"}}


def test_healthy_plan_untouched():
    plan = _plan([_resolver_event()])
    ops = [GenericOutcomeOperator()]
    rep = ensure_outcome_pathway(plan, ops)
    assert rep["repaired"] is False and rep["outcome_capable_events"] == ["resolve_outcome"]
    assert len(plan.scheduled_events) == 1 and len(ops) == 1


def test_dropped_writer_is_reinstantiated_not_resolver_added():
    # the EXP-105 class: an absorbing institutional vote exists but its operator was silently rejected
    plan = _plan([{"etype": "institutional_decision", "ts": 1_500_000.0, "participants": [],
                   "payload": {"institution_id": "board", "absorbing": True, "outcome_var": "outcome"}}])
    ops = []                                                   # writer dropped at instantiation
    rep = ensure_outcome_pathway(plan, ops, rejections=[{"mech_id": "x", "reason": "TypeError"}])
    assert rep["repaired"] is True
    assert any(r.startswith("reinstantiated_dropped_writer:institutional_decision") for r in rep["repairs"])
    assert any(getattr(o, "name", "") == "institutional_decision" for o in ops)
    assert all(e["etype"] != "resolve_outcome" for e in plan.scheduled_events)   # no resolver bolted on
    assert rep["operator_rejections"]                          # rejections surfaced, not discarded


def test_no_outcome_event_at_all_readds_canonical_resolver():
    # worst case: surgery left NOTHING that can write the outcome
    plan = _plan([{"etype": "actor_action", "ts": 1_000.0, "participants": [], "payload": {}}])
    ops = []
    rep = ensure_outcome_pathway(plan, ops)
    assert rep["repaired"] is True and "readded_canonical_resolve_outcome" in rep["repairs"]
    ev = [e for e in plan.scheduled_events if e["etype"] == "resolve_outcome"]
    assert len(ev) == 1 and ev[0]["payload"]["outcome_var"] == "outcome"
    assert any(getattr(o, "name", "") == "generic_outcome_prior" for o in ops)


def test_ornamental_vote_does_not_count_as_capable():
    # an institutional_decision event with neither outcome_var nor absorbing cannot answer the question
    plan = _plan([{"etype": "institutional_decision", "ts": 1_000.0, "participants": [], "payload": {}}])
    ops = []
    rep = ensure_outcome_pathway(plan, ops)
    assert rep["repaired"] is True and "readded_canonical_resolve_outcome" in rep["repairs"]
