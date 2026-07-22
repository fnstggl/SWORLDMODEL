"""D16 — dimensional outcome mechanisms. Universal machinery only.

Locks: the terminal is produced as the EXACT variable in the EXACT units; a numeric terminal is
never collapsed to a boolean; a threshold is compared in the output's dimension; the dependency
chain is unbroken; the wrong-dimension mechanism (votes for a rate, count for a rate) is rejected."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.outcome_mechanism import (
    DIM_COUNT, DIM_EVENT, DIM_RATE, DIM_RATIO, MechanismInput, OutcomeMechanismSpec, Transition,
    dimensions_compatible, infer_dimension, validate_outcome_mechanism)
from swm.world_model_v2.lean_v2.resolution_spec import ResolutionSpec


# ============================================================ dimension inference
def test_infer_dimension():
    assert infer_dimension("tankers/day") == DIM_RATE
    assert infer_dimension("tankers per day") == DIM_RATE
    assert infer_dimension("votes") == DIM_COUNT
    assert infer_dimension("seats") == DIM_COUNT
    assert infer_dimension("%") == DIM_RATIO
    assert infer_dimension("basis points") == DIM_RATIO
    assert infer_dimension("by the deadline") == DIM_EVENT
    assert not dimensions_compatible("votes", "tankers/day")
    assert dimensions_compatible("tankers/day", "ships per day")


# ============================================================ 53 — votes mechanism for a vote
def test_53_votes_mechanism_matches_a_votes_terminal():
    spec = OutcomeMechanismSpec(
        output_variable="yes_votes", output_unit="votes", comparator=">=", threshold=5,
        threshold_unit="votes", aggregation="sum",
        inputs=[MechanismInput("member_vote", "votes", "actor_decision")],
        transitions=[Transition("sum", ["member_vote"], "yes_votes", "votes")])
    rs = ResolutionSpec(measured_variable="yes votes", unit="votes", comparator=">=", threshold=5)
    ok, diag = validate_outcome_mechanism(spec, rs)
    assert ok, diag


# ============================================================ 54 — rate mechanism for a rate
def test_54_rate_mechanism_matches_a_rate_terminal():
    # daily tanker observations (a daily rate) averaged over the window → tankers/day
    spec = OutcomeMechanismSpec(
        output_variable="avg_daily_transits", output_unit="tankers/day", comparator="<=",
        threshold=50, threshold_unit="tankers/day", aggregation="average", window="daily",
        inputs=[MechanismInput("daily_transits", "tankers/day", "observation")],
        transitions=[Transition("average", ["daily_transits"], "avg_daily_transits", "tankers/day")])
    rs = ResolutionSpec(measured_variable="tanker transits", unit="tankers/day", comparator="<=",
                        threshold=50)
    ok, diag = validate_outcome_mechanism(spec, rs)
    assert ok, diag


# ============================================================ 55 — no boolean collapse
def test_55_numeric_terminal_cannot_collapse_to_boolean():
    spec = OutcomeMechanismSpec(
        output_variable="closed", output_unit="by the deadline", comparator="==", threshold=1,
        transitions=[Transition("last", [], "closed", "event")])
    rs = ResolutionSpec(measured_variable="tanker transits", unit="tankers/day", comparator="<=",
                        threshold=50)
    ok, diag = validate_outcome_mechanism(spec, rs)
    assert not ok
    assert any("boolean" in x or "wrong quantity" in x for x in diag)


# ============================================================ 56 — wrong dimension rejected
def test_56_wrong_dimension_mechanism_rejected():
    # a votes mechanism cannot score a tankers/day terminal
    spec = OutcomeMechanismSpec(output_variable="yes_votes", output_unit="votes", comparator=">=",
                                threshold=5, threshold_unit="votes",
                                transitions=[Transition("sum", [], "yes_votes", "votes")])
    rs = ResolutionSpec(measured_variable="tanker transits", unit="tankers/day")
    ok, diag = validate_outcome_mechanism(spec, rs)
    assert not ok and any("wrong quantity" in x or "!=" in x for x in diag)


# ============================================================ 57 — threshold dimension must match
def test_57_threshold_must_be_in_output_dimension():
    # threshold in a bare count against a per-day rate output
    spec = OutcomeMechanismSpec(
        output_variable="avg_daily", output_unit="tankers/day", threshold=50,
        threshold_unit="tankers", aggregation="average",
        inputs=[MechanismInput("daily", "tankers/day")],
        transitions=[Transition("average", ["daily"], "avg_daily", "tankers/day")])
    rs = ResolutionSpec(unit="tankers/day")
    ok, diag = validate_outcome_mechanism(spec, rs)
    assert not ok and any("threshold unit" in x for x in diag)


# ============================================================ 58 — unbroken dependency chain
def test_58_broken_dependency_chain_rejected():
    # the output variable is produced by no transition and is not a declared input
    spec = OutcomeMechanismSpec(
        output_variable="mystery_total", output_unit="votes",
        inputs=[MechanismInput("member_vote", "votes", "actor_decision")],
        transitions=[Transition("sum", ["member_vote"], "yes_votes", "votes")])
    ok, diag = validate_outcome_mechanism(spec, ResolutionSpec(unit="votes"))
    assert not ok and any("dependency chain is broken" in x for x in diag)


def test_58b_undefined_upstream_variable_rejected():
    spec = OutcomeMechanismSpec(
        output_variable="yes_votes", output_unit="votes",
        inputs=[MechanismInput("member_vote", "votes", "actor_decision")],
        transitions=[Transition("sum", ["member_vote", "ghost_input"], "yes_votes", "votes")])
    ok, diag = validate_outcome_mechanism(spec, ResolutionSpec(unit="votes"))
    assert not ok and any("undefined variable" in x for x in diag)


# ============================================================ hybrid boundary is typed
def test_hybrid_inputs_are_typed_by_source():
    spec = OutcomeMechanismSpec(
        output_variable="yes_votes", output_unit="votes",
        inputs=[MechanismInput("member_vote", "votes", "actor_decision"),
                MechanismInput("quorum_present", "count", "observation")],
        transitions=[Transition("sum", ["member_vote"], "yes_votes", "votes")])
    sources = {i.name: i.source for i in spec.inputs}
    assert sources["member_vote"] == "actor_decision"       # actors decide behavioral inputs
    assert sources["quorum_present"] == "observation"       # deterministic code reads observations
