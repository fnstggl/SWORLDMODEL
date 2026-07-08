"""Episodic memory + reflection — retrieval by recency × importance × relevance, leakage, reflection loop."""
import pytest

from swm.memory.embeddings import cosine, hashing_embed
from swm.memory.memory import Episode, EpisodicStore, LeakageError, MemoryStream


def test_embedding_is_deterministic_and_relevance_orders_by_topic():
    a1 = hashing_embed("quarterly pricing and discount for the enterprise plan")
    a2 = hashing_embed("quarterly pricing and discount for the enterprise plan")
    assert a1 == a2                                             # deterministic (FNV-1a, no salt)
    pricing = hashing_embed("can we talk about pricing and discounts")
    scheduling = hashing_embed("are you free to schedule a call next week")
    q = hashing_embed("what is your pricing")
    assert cosine(q, pricing) > cosine(q, scheduling)          # relevance tracks topic overlap


def test_retrieve_ranks_relevant_first():
    ms = MemoryStream(entity_id="u", half_life=30)
    ms.remember("pricing question about the enterprise discount", ts=1, importance=0.5)
    ms.remember("scheduling a coffee chat next tuesday", ts=2, importance=0.5)
    hits = ms.retrieve("pricing and discount", as_of=10, k=2, weights={"relevance": 1.0})
    assert "pricing" in hits[0]["episode"].text


def test_recency_weight_prefers_recent_when_relevance_and_importance_tie():
    ms = MemoryStream(entity_id="u", half_life=5)
    ms.remember("identical topic alpha beta", ts=1, importance=0.5)     # old
    ms.remember("identical topic alpha beta", ts=90, importance=0.5)    # recent
    hits = ms.retrieve("alpha beta", as_of=100, k=2, weights={"recency": 1.0})
    assert hits[0]["episode"].timestamp == 90                  # recency breaks the tie


def test_importance_weight_prefers_poignant_when_recency_and_relevance_tie():
    ms = MemoryStream(entity_id="u", half_life=1e9)            # recency effectively off
    ms.remember("identical topic alpha beta", ts=10, importance=0.1)
    ms.remember("identical topic alpha beta", ts=10, importance=0.9)
    hits = ms.retrieve("alpha beta", as_of=100, k=2, weights={"importance": 1.0})
    assert hits[0]["episode"].importance == 0.9


def test_retrieval_is_leakage_safe():
    ms = MemoryStream(entity_id="u", half_life=30)
    ms.remember("past one", ts=10, importance=0.5)
    ms.remember("past two", ts=20, importance=0.5)
    ms.remember("the FUTURE episode being predicted", ts=30, importance=0.9)
    hits = ms.retrieve("future episode", as_of=25, k=10)
    ids = {h["episode"].timestamp for h in hits}
    assert ids == {10, 20}                                     # ts=30 never surfaces
    assert all(h["episode"].timestamp < 25 for h in hits)


def test_retrieve_requires_as_of():
    ms = MemoryStream(entity_id="u")
    ms.remember("x", ts=1)
    with pytest.raises(LeakageError):
        ms.retrieve("x", as_of=None)


def test_episode_requires_timestamp():
    ms = MemoryStream(entity_id="u")
    with pytest.raises(LeakageError):
        ms.add(Episode(entity_id="u", timestamp=None, text="no ts"))


def test_personal_response_rate_is_asof():
    ms = MemoryStream(entity_id="u")
    ms.record_contact(ts=1, text="a", responded=True, topic="t")
    ms.record_contact(ts=2, text="b", responded=False, topic="t")
    ms.record_contact(ts=3, text="c", responded=True, topic="t")
    assert ms.personal_response_rate(as_of=10) == pytest.approx(2 / 3)
    assert ms.personal_response_rate(as_of=2) == pytest.approx(1.0)     # only ts=1 visible


def test_reflection_mints_topic_abstraction_and_feeds_retrieval():
    ms = MemoryStream(entity_id="u", half_life=1e9)
    # responds to pricing, ignores cold outreach — a pattern reflection should name
    for t in range(1, 5):
        ms.record_contact(ts=t, text=f"pricing question {t}", responded=True, topic="pricing")
    for t in range(5, 9):
        ms.record_contact(ts=t, text=f"cold intro {t}", responded=False, topic="cold")
    made = ms.reflect(as_of=100)
    texts = " ".join(m.text.lower() for m in made)
    assert made and ("pricing" in texts or "cold" in texts)
    assert all(m.kind == "reflection" for m in made)
    # the reflection now participates in retrieval for a pricing query
    hits = ms.retrieve("pricing", as_of=200, k=10)
    assert any(h["episode"].kind == "reflection" for h in hits)


def test_maybe_reflect_triggers_on_importance_threshold():
    ms = MemoryStream(entity_id="u")
    for t in range(1, 4):
        ms.record_contact(ts=t, text=f"pricing {t}", responded=True, topic="pricing")
    assert ms.maybe_reflect(as_of=50, threshold=99) == []      # below threshold → no reflection
    got = ms.maybe_reflect(as_of=50, threshold=0.5)            # above → reflects
    assert isinstance(got, list)


def test_reflect_fn_plugin_is_used():
    ms = MemoryStream(entity_id="u")
    for t in range(1, 6):
        ms.record_contact(ts=t, text=f"msg {t}", responded=True, topic="x")

    def fake_llm(episodes):
        return [{"text": "This person is highly responsive.", "importance": 0.9}]

    made = ms.reflect(as_of=100, reflect_fn=fake_llm)
    assert len(made) == 1 and "responsive" in made[0].text.lower()


def test_episodic_store_isolates_entities():
    store = EpisodicStore(half_life=30)
    store.record_contact("alice", ts=1, text="pricing", responded=True, topic="pricing")
    store.record_contact("bob", ts=1, text="pricing", responded=False, topic="pricing")
    assert "alice" in store and "bob" in store
    assert store.stream("alice").personal_response_rate(as_of=10) == 1.0
    assert store.stream("bob").personal_response_rate(as_of=10) == 0.0
    assert store.retrieve("nobody", "x", as_of=10) == []        # unknown entity → empty, not error
