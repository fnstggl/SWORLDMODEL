"""D13 — actor knowledge packets. Universal machinery only.

Locks: an actor reasons from REAL fact content (never hashes/labels); the packet composes the
actor's own latent mindset (D9), the facts they can see (D11), the world that affects them (D12),
and the live institution process (D14); and it NEVER leaks another actor's private state, a secret
ballot, a future event, or a post-as_of fact."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.evidence_store import (
    CONFIRMED, ROLE_PRIVATE, SECRET, CanonicalFact, EvidenceStore)
from swm.world_model_v2.lean_v2.knowledge_packet import build_knowledge_packet
from swm.world_model_v2.lean_v2.states import ActorStateHypothesis, separate_mindset_from_events


def _store():
    s = EvidenceStore(as_of="2025-06-01")
    s.add(CanonicalFact(content="Inflation rose to 3.5% in April", date="2025-05-01",
                        credibility=CONFIRMED, terminal_relevance=0.9))
    s.add(CanonicalFact(content="Governor got a private FX briefing", date="2025-05-20",
                        visibility=ROLE_PRIVATE, actor_access=["gov"], terminal_relevance=0.5))
    s.add(CanonicalFact(content="Deputy secretly plans to dissent", date="2025-05-25",
                        visibility=SECRET, actor_access=["deputy"]))
    return s


def _gov_state(store):
    st = ActorStateHypothesis(actor_id="gov", state_id="s",
                              claim="The governor received a secret memo from the PM.",
                              beliefs=["believes inflation is persistent"],
                              goals=["preserve credibility"], stances=["risk-averse"])
    separate_mindset_from_events(st, evidence_store=store)
    return st


# ============================================================ 14 — real content, not hashes
def test_14_packet_renders_real_fact_content_not_hashes():
    s = _store()
    pkt = build_knowledge_packet({"id": "gov", "role": "Governor"}, evidence_store=s,
                                 state=_gov_state(s), shared_world={"inflation": "high"},
                                 day="2025-06-01", feasible_actions=["Hold", "Raise"])
    r = pkt.render()
    assert "Inflation rose to 3.5%" in r                     # real content
    assert not any(f.fact_id in r for f in s.facts)          # never a hash/id
    assert "Governor" in r and "inflation=high" in r


# ============================================================ 43 — no other actor's private state
def test_43_never_leaks_another_actors_secret():
    s = _store()
    pkt = build_knowledge_packet({"id": "gov", "role": "Governor"}, evidence_store=s,
                                 state=_gov_state(s), day="2025-06-01")
    r = pkt.render()
    assert "secretly plans to dissent" not in r              # the deputy's secret never leaks
    assert "FX briefing" in r                                # own role-private fact is included


# ============================================================ 44 — unsupported event stays a maybe
def test_44_invented_external_event_is_flagged_not_asserted():
    s = _store()
    pkt = build_knowledge_packet({"id": "gov", "role": "Governor"}, evidence_store=s,
                                 state=_gov_state(s), day="2025-06-01")
    r = pkt.render()
    # the invented secret memo appears ONLY under the UNSURE/possibility section, never as fact
    assert "secret memo" in r                                # it is carried...
    assert "UNSURE" in r or "simulated possibility" in r     # ...but explicitly as unverified
    assert "secret memo" not in r.split("UNSURE")[0]         # not in the established-facts region


# ============================================================ 45 — institution process is visible
def test_45_packet_carries_the_live_institution_process():
    s = _store()
    ist = {"proposal": "Raise to 1.0%", "stage": "deliberation", "deadline": "2025-06-19",
           "tally_visible": True, "tally": {"raise": 2, "hold": 1},
           "public_positions": {"deputy": "leaning hold", "gov": "undecided"},
           "messages_to": {"gov": ["The deputy argues for patience"]}}
    pkt = build_knowledge_packet({"id": "gov", "role": "Governor"}, evidence_store=s,
                                 state=_gov_state(s), institution_state=ist, day="2025-06-01",
                                 feasible_actions=["Hold", "Raise"])
    r = pkt.render()
    assert "Raise to 1.0%" in r and "deliberation" in r      # proposal + stage
    assert "raise: 2" in r                                   # visible tally
    assert "The deputy argues for patience" in r             # received message
    assert "deputy: leaning hold" in r                       # others' PUBLIC positions
    assert "gov" not in pkt.visible_positions                # not the actor's own position back


# ============================================================ future / secret-ballot guards
def test_secret_ballot_and_future_are_never_in_the_packet():
    s = _store()
    # a hidden tally is not shown; only a visible tally is
    pkt = build_knowledge_packet({"id": "gov"}, evidence_store=s, state=_gov_state(s),
                                 institution_state={"tally_visible": False,
                                                    "tally": {"raise": 2}}, day="2025-06-01")
    assert pkt.visible_tally == {}                           # a secret ballot's tally is not exposed
    # a post-as_of fact never enters (leakage guard in the store)
    s.add(CanonicalFact(content="the board hiked on June 15", date="2025-06-15"))
    pkt2 = build_knowledge_packet({"id": "gov"}, evidence_store=s, state=_gov_state(s),
                                  day="2025-06-01")
    assert "hiked on June 15" not in pkt2.render()
