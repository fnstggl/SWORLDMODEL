"""D11 — canonical fact store. Universal machinery only.

Locks: decisive facts survive typed relevance selection (never a blind char truncation); actors
receive real fact CONTENT, never hashes; who-knows-what is modeled (visibility/access); post-as_of
facts never leak; conflicting reports are represented as contradictions."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.evidence_store import (
    CONFIRMED, INSTITUTION_PRIVATE, PUBLIC, REPORTED, RUMORED, ROLE_PRIVATE, SECRET,
    CanonicalFact, EvidenceStore, build_evidence_store)


def _store():
    s = EvidenceStore(as_of="2025-06-01")
    s.add(CanonicalFact(content="Headline inflation rose to 3.5% in April", date="2025-05-01",
                        credibility=CONFIRMED, terminal_relevance=0.9,
                        source_quotes=["CPI +3.5% y/y"], numeric_values={"cpi": 3.5}, units="%"))
    s.add(CanonicalFact(content="A board member privately signalled a hike", date="2025-05-20",
                        visibility=SECRET, actor_access=["gov"], credibility=RUMORED,
                        terminal_relevance=0.5))
    return s


# ============================================================ 11 — real content, never a hash
def test_11_actor_receives_fact_content_not_a_hash():
    s = _store()
    facts = s.facts_for_actor("gov", day="2025-06-01")
    rendered = [f.render() for f in facts]
    assert any("inflation rose to 3.5%" in r.lower() for r in rendered)
    # the fact_id is a hash, but what the actor reads is the CONTENT, not the id
    assert all(f.fact_id not in f.render() for f in facts)
    assert all(f.content and not f.content.startswith("f_") for f in facts)


# ============================================================ 12 — leakage guard (post-as_of)
def test_12_post_as_of_facts_never_leak():
    s = _store()
    s.add(CanonicalFact(content="The board hiked on June 15", date="2025-06-15",
                        credibility=CONFIRMED, terminal_relevance=1.0))
    contents = [f.content for f in s.facts]
    assert not any("hiked on June 15" in c for c in contents)
    assert s.manifest()["n_dropped_leakage"] == 1
    # a fact dated exactly on as_of is also unknowable (leakage boundary is inclusive of as_of)
    s.add(CanonicalFact(content="as_of-day announcement", date="2025-06-01"))
    assert not any("as_of-day" in f.content for f in s.facts)


# ============================================================ 13 — visibility / who-knows-what
def test_13_visibility_models_who_knows_what():
    s = _store()
    # the secret signal reaches only the granted actor; the public fact reaches everyone
    assert {f.content for f in s.facts_for_actor("gov")} == {
        "Headline inflation rose to 3.5% in April", "A board member privately signalled a hike"}
    assert {f.content for f in s.facts_for_actor("outsider")} == {
        "Headline inflation rose to 3.5% in April"}
    # institution-private visibility: only members of an institution with access see it
    s.add(CanonicalFact(content="internal staff projection", visibility=INSTITUTION_PRIVATE,
                        institution_access=["board"]))
    insts = {"m1": ["board"], "m2": ["other"]}
    assert any("staff projection" in f.content
               for f in s.facts_for_actor("m1", institutions=insts))
    assert not any("staff projection" in f.content
                   for f in s.facts_for_actor("m2", institutions=insts))


# ============================================================ 14 — typed relevance, decisive survive
def test_14_decisive_facts_survive_relevance_selection():
    s = EvidenceStore(as_of="2025-06-01")
    # 30 low-relevance facts + one decisive fact; a blind truncation could drop the decisive one,
    # relevance selection keeps it at the top
    for i in range(30):
        s.add(CanonicalFact(content=f"minor detail {i}", date="2025-01-01",
                            terminal_relevance=0.05, decision_relevance={"gov": 0.05}))
    s.add(CanonicalFact(content="the decisive swing fact", date="2025-05-30",
                        terminal_relevance=0.99, decision_relevance={"gov": 0.99}))
    top = s.facts_for_actor("gov", k=5)
    assert top[0].content == "the decisive swing fact"       # survives, ranked first
    assert "the decisive swing fact" in {f.content for f in s.terminal_relevant_facts(k=3)}


# ============================================================ 15 — contradictions represented
def test_15_conflicting_reports_are_contradictions_not_both_asserted():
    s = EvidenceStore(as_of="2025-06-01")
    s.add(CanonicalFact(content="Source A: the plant will reopen", date="2025-05-01",
                        credibility=REPORTED, contradiction_group="plant_status"))
    s.add(CanonicalFact(content="Source B: the plant will stay closed", date="2025-05-02",
                        credibility=RUMORED, contradiction_group="plant_status"))
    groups = s.contradictions()
    assert "plant_status" in groups and len(groups["plant_status"]) == 2
    # both are present as competing reports, neither is silently dropped or asserted as truth
    assert {f.credibility for f in groups["plant_status"]} == {REPORTED, RUMORED}


# ============================================================ credibility ordering
def test_credibility_ranks_confirmed_above_rumor():
    s = EvidenceStore(as_of="2025-06-01")
    s.add(CanonicalFact(content="confirmed X", credibility=CONFIRMED,
                        decision_relevance={"a": 0.5}))
    s.add(CanonicalFact(content="rumored X", credibility=RUMORED,
                        decision_relevance={"a": 0.5}))
    ranked = s.facts_for_actor("a")
    assert ranked[0].credibility == CONFIRMED                # tie on relevance → credibility wins


# ============================================================ build from grounding + blueprint
def test_build_evidence_store_from_grounding():
    bp = ConsumerWorldBlueprint(
        institutions=[{"id": "board", "members": ["m1", "m2"]}],
        grounded_rates=[{"quantity": "prior hikes rate", "basis_quote": "hiked 6 of last 8",
                         "value_range": [0.6, 0.8]}],
        resolution={"interpretation": "Will the board hike?"})
    grounding = {"shared_world_conditions": {
        "infl": {"claim": "inflation is elevated", "affects_actors": ["m1", "m2"],
                 "table": {"provenance": {"denominator": 4, "cases": [
                     {"included": True, "basis_quote": "CPI 3.5%", "date": "2025-05-01"}]}}}},
        "outcome_reference_class": {"quantity": "board hikes", "provenance": {"denominator": 8,
            "cases": [{"included": True, "description": "hiked in 2023",
                       "basis_quote": "raised 25bp", "date": "2023-07-01"}]}}}
    store = build_evidence_store(bp, grounding, as_of="2025-06-01")
    contents = [f.content for f in store.facts]
    assert any("inflation is elevated" in c for c in contents)      # shared condition → fact
    assert any("hiked in 2023" in c for c in contents)              # verified reference case → fact
    assert any("prior hikes rate" in c for c in contents)           # grounded rate → fact
    assert store.get(store.facts[0].fact_id) is not None
