"""D17 — readiness = structurally faithful (not merely executable). Universal machinery only.

Locks: readiness aggregates the fidelity verdicts of the resolution (D5), institution
representation (D7), evidence (D11), behavior grounding (D8), and outcome mechanism (D16); an
impossible institution is never made ready by rescaling its real threshold; the report is the
worst dimension."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.evidence_store import CanonicalFact, EvidenceStore
from swm.world_model_v2.lean_v2.readiness import assess_structural_fidelity
from swm.world_model_v2.lean_v2.representation import ensure_faithful_representation
from swm.world_model_v2.lean_v2.resolution_spec import INSTITUTION_VOTE, ResolutionSpec


def _vote_bp(n_modeled=9):
    actors = [{"id": f"m{i}", "role": "member"} for i in range(n_modeled)]
    inst = {"id": "b", "members": [a["id"] for a in actors], "decision_rule": "majority",
            "rule_params": {"option": "Raise"}}
    return ConsumerWorldBlueprint(
        actors=actors, institutions=[inst],
        terminal={"kind": "institution_vote", "institution_id": "b",
                  "decision_rule": "majority", "rule_params": {"option": "Raise"}},
        resolution={"interpretation": "Will the board raise?"})


def _store():
    s = EvidenceStore(as_of="2025-06-01")
    s.add(CanonicalFact(content="inflation elevated", date="2025-05-01", terminal_relevance=0.8))
    return s


_G = {"outcome_reference_class": {"provenance": {"denominator": 8}},
      "actor_state_reference_classes": {"m0": [{}]}}


# ============================================================ faithful world is ready
def test_faithful_world_is_ready():
    bp = _vote_bp(9)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5)
    rep = ensure_faithful_representation(bp, rs, evidence_text="the nine-member board")
    r = assess_structural_fidelity(bp, resolution_spec=rs, representation=rep,
                                   evidence_store=_store(), grounding=_G)
    assert r.verdict == "ready"
    assert r.checks["institution"]["verdict"] == "ready"


# ============================================================ 26 — never rescale to become ready
def test_26_impossible_threshold_is_not_ready_never_rescaled():
    bp = _vote_bp(9)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=12)
    rep = ensure_faithful_representation(bp, rs, evidence_text="the nine-member board")
    r = assess_structural_fidelity(bp, resolution_spec=rs, representation=rep,
                                   evidence_store=_store(), grounding=_G)
    assert r.verdict == "not_ready"
    assert rep.threshold == 12                              # the real threshold is NOT rescaled
    assert r.checks["institution"]["threshold"] == 12


# ============================================================ collapsed roster is repairable
def test_collapsed_roster_is_repairable_by_expansion():
    # 5 modeled of a 9-member board — repairable by expanding the roster, not by rescaling
    bp = _vote_bp(5)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5)
    from swm.world_model_v2.lean_v2.representation import build_representation, validate_representation
    raw = validate_representation(build_representation(bp, rs, evidence_text="nine-member board"))
    r = assess_structural_fidelity(bp, resolution_spec=rs, representation=raw,
                                   evidence_store=_store(), grounding=_G)
    assert r.verdict == "repairable"
    assert any("expand the institution roster" in x for x in r.repairs_needed)
    assert any("never rescale" in x for x in r.repairs_needed)


# ============================================================ 57 — wrong outcome dimension fails
def test_57_wrong_outcome_dimension_is_not_ready():
    bp = _vote_bp(9)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5)
    rep = ensure_faithful_representation(bp, rs, evidence_text="the nine-member board")
    r = assess_structural_fidelity(
        bp, resolution_spec=rs, representation=rep, evidence_store=_store(), grounding=_G,
        mechanism_dim={"ok": False, "dimension": "event", "required_dimension": "rate",
                       "diagnostics": ["boolean collapse of a numeric terminal"]})
    assert r.verdict == "not_ready"
    assert r.checks["outcome"]["verdict"] == "not_ready"


# ============================================================ 72 — missing evidence is repairable
def test_72_missing_evidence_is_repairable():
    bp = _vote_bp(9)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5)
    rep = ensure_faithful_representation(bp, rs, evidence_text="the nine-member board")
    empty = EvidenceStore(as_of="2025-06-01")
    r = assess_structural_fidelity(bp, resolution_spec=rs, representation=rep,
                                   evidence_store=empty, grounding=_G)
    assert r.checks["evidence"]["verdict"] == "repairable"
    assert r.verdict in ("repairable", "not_ready")


# ============================================================ 73 — ungrounded behavior repairable
def test_73_ungrounded_behavior_is_repairable():
    bp = _vote_bp(9)
    rs = ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5)
    rep = ensure_faithful_representation(bp, rs, evidence_text="the nine-member board")
    r = assess_structural_fidelity(bp, resolution_spec=rs, representation=rep,
                                   evidence_store=_store(),
                                   grounding={"actor_state_reference_classes": {},
                                              "outcome_reference_class": {}})
    assert r.checks["behavior"]["verdict"] == "repairable"
