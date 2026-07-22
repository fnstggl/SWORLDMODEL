"""D15 — conservative decision-cache context key. Universal machinery only.

Locks: every MATERIAL field participates in the cache key, so a changed proposal / stage /
decision rule / visible tally / substantive message / fact credibility / deadline is a DIFFERENT
decision context and MUST miss the cache — a false hit reusing a stale reply is impossible; while a
context with identical material reuses correctly."""
from __future__ import annotations

from swm.world_model_v2.lean_context import (DecisionContextDifference,
                                               DecisionRelevantContext)


def _ctx(**kw):
    base = dict(actor_id="gov", actor_role="Governor", day="2025-06-01",
                private_state={"stance": "hawkish"},
                observations=[{"fact_id": "f1", "content": "inflation 3.5%"}],
                feasible_actions=["Hold", "Raise"], proposal="Raise to 1.0%", stage="deliberation",
                decision_rule="majority", visible_tally={"raise": 2}, deadline="2025-06-19")
    base.update(kw)
    return DecisionRelevantContext(**base)


# ============================================================ 37 — identical material reuses
def test_37_identical_material_context_reuses():
    assert _ctx().signature() == _ctx().signature()


# ============================================================ 38 — a material change always misses
def test_38_material_change_causes_a_cache_miss():
    base = _ctx().signature()
    assert _ctx(proposal="Raise to 1.25%").signature() != base          # different proposal
    assert _ctx(stage="final_vote").signature() != base                 # different stage
    assert _ctx(decision_rule="unanimity").signature() != base          # different rule
    assert _ctx(visible_tally={"raise": 4}).signature() != base         # different tally
    assert _ctx(substantive_messages=["deputy argues for patience"]).signature() != base
    assert _ctx(deadline="2025-07-01").signature() != base              # different deadline
    assert _ctx(private_state={"stance": "dovish"}).signature() != base  # different mindset


# ============================================================ 39 — credibility is material
def test_39_fact_credibility_change_is_a_different_context():
    a = _ctx(fact_credibility={"f1": "confirmed"}).signature()
    b = _ctx(fact_credibility={"f1": "rumored"}).signature()
    assert a != b       # the same fact at a different credibility is a materially different context


# ============================================================ the diff report names the field
def test_diff_report_identifies_the_changed_material_field():
    d = DecisionContextDifference.between(_ctx(), _ctx(visible_tally={"raise": 5}))
    assert not d.equal and "visible_tally" in d.differing_components
    d2 = DecisionContextDifference.between(_ctx(), _ctx(proposal="something else"))
    assert "proposal" in d2.differing_components


# ============================================================ final-decision context not stripped
def test_final_decision_context_retains_the_process_fields():
    c = _ctx(substantive_messages=["m1"], commitments=["will support if inflation persists"])
    d = c.as_dict()
    for field in ("proposal", "stage", "decision_rule", "visible_tally", "substantive_messages",
                  "fact_credibility", "deadline"):
        assert field in d          # never stripped from the key for hit-rate
