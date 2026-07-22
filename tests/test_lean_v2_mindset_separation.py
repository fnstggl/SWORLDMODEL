"""D9 — split latent mindset from unobserved external events. Universal machinery only.

Locks: an unsupported EXTERNAL EVENT claim (a secret memo, a private threat) is never handed to
the actor as established reality — it is relabeled a simulated possibility. The actor's LATENT
mindset (beliefs/goals/preferences/risk tolerance) is kept, freely hypothesized."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.evidence_store import CanonicalFact, EvidenceStore
from swm.world_model_v2.lean_v2.states import (ActorStateHypothesis, classify_assertion,
                                               separate_mindset_from_events)


# ============================================================ classification
def test_classify_latent_vs_external_event():
    assert classify_assertion("believes inflation is too high") == "latent"
    assert classify_assertion("wants to protect the currency") == "latent"
    assert classify_assertion("fears the PM will retaliate") == "latent"          # belief-framed
    assert classify_assertion("received a secret memo from the PM") == "external_event"
    assert classify_assertion("the finance minister threatened to resign") == "external_event"
    assert classify_assertion("was handed a backchannel offer") == "external_event"


# ============================================================ 18 — unsupported events not fact
def test_18_unsupported_external_event_is_never_presented_as_known():
    h = ActorStateHypothesis(
        actor_id="gov", state_id="s1",
        claim="The governor received a secret memo from the PM demanding a hike.",
        beliefs=["believes inflation is persistent"],
        goals=["preserve independence"], stances=["risk-averse"])
    m = separate_mindset_from_events(h, evidence_store=None)
    # the invented external event is relabeled a simulated possibility, removed from beliefs
    assert any("secret memo" in a for a in h.hypothetical_assumptions)
    assert all("secret memo" not in b for b in h.beliefs)
    assert m["unsupported_external_events_relabeled"]
    # the latent mindset survives intact
    assert "believes inflation is persistent" in h.latent_beliefs
    assert h.latent_goals == ["preserve independence"]
    assert h.latent_risk_tolerance == "risk-averse"
    assert h.mindset_separated


def test_18b_evidence_supported_event_becomes_an_observation():
    store = EvidenceStore(as_of="2025-06-01")
    store.add(CanonicalFact(content="The prime minister publicly urged the bank to hike",
                            date="2025-05-01"))
    h = ActorStateHypothesis(
        actor_id="gov", state_id="s2",
        claim="The prime minister publicly urged the bank to hike rates.",
        beliefs=["believes growth is slowing"])
    separate_mindset_from_events(h, evidence_store=store)
    # a claim matched by a real fact is a supported observation, not a mere hypothetical
    assert any("prime minister" in o.lower() for o in h.evidence_supported_observations)
    assert not any("prime minister" in a.lower() for a in h.hypothetical_assumptions)


def test_18c_supporting_evidence_id_makes_it_supported():
    h = ActorStateHypothesis(
        actor_id="a", state_id="s3", claim="A rival announced a competing product.",
        supporting_evidence_ids=["e1"])
    separate_mindset_from_events(h, evidence_store=None)
    # the actor cited supporting evidence → the external event is a supported observation
    assert any("competing product" in o for o in h.evidence_supported_observations)
    assert not h.hypothetical_assumptions


def test_18d_pure_latent_state_moves_nothing_external():
    h = ActorStateHypothesis(
        actor_id="a", state_id="s4", claim="A cautious policymaker who prioritizes stability.",
        beliefs=["believes the risks are two-sided", "prefers to wait for more data"],
        goals=["avoid a policy error"], stances=["risk-averse"])
    m = separate_mindset_from_events(h, evidence_store=None)
    assert not h.hypothetical_assumptions and not m["unsupported_external_events_relabeled"]
    assert len(h.latent_beliefs) >= 2
    assert h.latent_risk_tolerance == "risk-averse"
