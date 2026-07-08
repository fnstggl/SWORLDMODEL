"""Retrieval-augmented response_fn — situation-conditioned recall on top of the global persona.

Verifies the four honest properties: (1) cold start / no history → base prediction UNTOUCHED; (2) a
history where similar messages did better than the person's average → p moves UP; worse → DOWN; (3) when
similar-history matches the personal base rate → NO move (the signal isolates the situation, not the
person); (4) leakage-safe (only past episodes inform the prediction).
"""
from swm.memory.memory import EpisodicStore
from swm.memory.retrieval_response import retrieval_augmented_response_fn


def _base_fn(p=0.5):
    """A stand-in global-persona response model: returns a fixed base probability."""
    def fn(variables, state, message):
        return {"p": p, "drivers": {}}
    return fn


def test_cold_start_returns_base_unchanged():
    store = EpisodicStore(half_life=30)
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0)
    out = fn({}, {}, {"text": "pricing question", "_entity_id": "u", "_as_of": 100})
    assert out["p"] == 0.5                                      # no history → exactly the base
    assert out["memory"]["evidence_weight"] == 0.0


def test_missing_framework_keys_passes_through():
    store = EpisodicStore()
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0)
    out = fn({}, {}, {"text": "pricing question"})             # no _entity_id / _as_of
    assert out == {"p": 0.5, "drivers": {}}


def _seed(store, eid="u"):
    """Person responds to pricing, ignores scheduling; overall base rate = 0.5."""
    for t in range(1, 6):
        store.record_contact(eid, ts=t, text=f"pricing discount enterprise {t}", responded=True,
                             topic="pricing")
    for t in range(6, 11):
        store.record_contact(eid, ts=t, text=f"schedule a call next week {t}", responded=False,
                             topic="scheduling")


def test_similar_history_moves_prediction_up_for_favored_topic():
    store = EpisodicStore(half_life=1e9)
    _seed(store)
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0, k=10)
    out = fn({}, {}, {"text": "another pricing and discount question", "topic": "pricing",
                      "_entity_id": "u", "_as_of": 100})
    assert out["p"] > 0.5                                       # similar (pricing) history did better
    assert out["memory"]["observed"] > out["memory"]["personal_base"]


def test_similar_history_moves_prediction_down_for_disfavored_topic():
    store = EpisodicStore(half_life=1e9)
    _seed(store)
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0, k=10)
    out = fn({}, {}, {"text": "can we schedule a call next week", "topic": "scheduling",
                      "_entity_id": "u", "_as_of": 100})
    assert out["p"] < 0.5                                       # scheduling history did worse


def test_signal_is_zero_when_topic_matches_personal_average():
    """If similar-history response rate == the person's overall rate, there is nothing situation-specific
    to add, so the prediction must not move."""
    store = EpisodicStore(half_life=1e9)
    # every contact is the same topic and responded rate 0.5 == personal base -> observed == base
    for t in range(1, 11):
        store.record_contact("u", ts=t, text=f"uniform topic zeta {t}", responded=(t % 2 == 0),
                             topic="zeta")
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0, k=10)
    out = fn({}, {}, {"text": "uniform topic zeta again", "topic": "zeta",
                      "_entity_id": "u", "_as_of": 100})
    assert abs(out["p"] - 0.5) < 1e-6


def test_retrieval_augmentation_is_leakage_safe():
    store = EpisodicStore(half_life=1e9)
    # only ignored-scheduling BEFORE t=6; the favorable pricing history is all in the FUTURE (>=6)
    for t in range(1, 6):
        store.record_contact("u", ts=t, text=f"schedule call {t}", responded=False, topic="scheduling")
    for t in range(6, 11):
        store.record_contact("u", ts=t, text=f"pricing discount {t}", responded=True, topic="pricing")
    fn = retrieval_augmented_response_fn(_base_fn(0.5), store, beta=2.0, k=10)
    # as_of=6 → the model can only see the (unfavorable) pre-6 history, none of the future pricing wins
    out = fn({}, {}, {"text": "pricing discount question", "topic": "pricing",
                      "_entity_id": "u", "_as_of": 6})
    assert out["memory"]["personal_base"] == 0.0               # only the 5 ignored contacts are visible
