"""Pins the deadline-aware outside-view prior (§8-9): the base rate is conditioned on the event's stage and
the remaining time; occurrence questions default low, recurrences-due stay high — grounded, not hardcoded
pessimism. Plumbing pinned with a stub LLM (live behavior verified separately in the branch report)."""
from types import SimpleNamespace

from swm.world_model_v2.phase3_priors import (build_outcome_rate_prior, estimate_reference_base_rate,
                                              _OCCURRENCE_STAGES)


def _stub(payload):
    import json
    return lambda _prompt: json.dumps(payload)


def _plan(question, as_of_ts, horizon_ts, lean="neutral"):
    return SimpleNamespace(question=question, as_of=as_of_ts, horizon_ts=horizon_ts,
                           provenance={"outcome_lean": lean, "as_of": "2026-05-07"})


def test_estimator_threads_horizon_and_returns_stage_status_quo():
    est = estimate_reference_base_rate(
        "Will X be dissolved by June?", horizon_days=49,
        llm=_stub({"reference_class": "parliament dissolutions", "is_recurrence": False,
                   "stage": "mere_proposal_or_speculation", "base_rate": 0.05,
                   "status_quo": "the body continues its normal session",
                   "n_examples": 20, "transport_risk": "moderate", "evidence_quality": "model_memory"}))
    assert est["stage"] == "mere_proposal_or_speculation" and est["base_rate"] == 0.05
    assert est["deadline_conditioned"] is True and est["horizon_days"] == 49
    assert est["status_quo"]


def test_occurrence_stage_yields_low_grounded_prior():
    # a specific one-off event, short deadline, early stage -> low grounded prior mean (the EXP-107 fix)
    plan = _plan("Will the Knesset be dissolved by June 30?", 1_746_000_000.0, 1_750_000_000.0)
    spec = build_outcome_rate_prior(plan, llm=_stub(
        {"reference_class": "Knesset dissolutions", "is_recurrence": False,
         "stage": "mere_proposal_or_speculation", "base_rate": 0.05, "status_quo": "normal session continues",
         "n_examples": 20, "transport_risk": "moderate", "evidence_quality": "model_memory"}))
    assert spec.mean < 0.25
    assert spec.provenance.get("occurrence_class") is True
    assert spec.provenance.get("stage") == "mere_proposal_or_speculation"
    assert spec.provenance.get("status_quo")


def test_recurrence_due_stays_high_not_blanket_lowered():
    # a recurrence due this window keeps a HIGH grounded prior -- no blanket pessimism
    plan = _plan("Will Apple announce visionOS 27 at WWDC 2026?", 1_746_000_000.0, 1_749_000_000.0)
    spec = build_outcome_rate_prior(plan, llm=_stub(
        {"reference_class": "annual visionOS at WWDC", "is_recurrence": True,
         "stage": "recurring_due_this_window", "base_rate": 0.95, "status_quo": "would slip to a later event",
         "n_examples": 6, "transport_risk": "low", "evidence_quality": "sourced"}))
    assert spec.mean > 0.75
    assert spec.provenance.get("occurrence_class") is False   # recurring_due is not the occurrence class


def test_occurrence_stage_set_membership():
    assert "mere_proposal_or_speculation" in _OCCURRENCE_STAGES
    assert "recurring_due_this_window" not in _OCCURRENCE_STAGES
    assert "essentially_decided_awaiting_formality" not in _OCCURRENCE_STAGES
