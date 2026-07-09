"""Recipient conditioning from history: ingest a corpus -> deep_inference persona -> recipient vars + base.

The product rule (ingest all recipient data) and the anti-compression mechanism (per-recipient base rate).
Validated as an estimator: an open/humble history yields higher persuadability than an entrenched one, and
the as-of store never leaks future documents.
"""
from swm.decision.recipient_history import HistoryStore, persona_to_recipient, recipient_from_history
from swm.variables.deep_inference import DeepInferenceEngine

OPEN = ["I think I might be wrong, that's a fair point, I appreciate your view, perhaps I should reconsider",
        "you make a good point and I could be mistaken, maybe, I'm open to changing my mind"]
ENTRENCHED = ["This is obviously true, everyone knows it, I will never change my mind, absolutely certain",
              "definitely correct, no one can argue, the fact is undeniable, it has always been this way"]


def test_recipient_from_history_returns_vars_and_base():
    rv, base = recipient_from_history(OPEN * 3)
    for k in ("openness_to_outreach", "skepticism", "op_openness", "op_entrenchment"):
        assert k in rv
    assert 0.0 <= base <= 1.0


def test_open_history_is_more_persuadable_than_entrenched():
    _, base_open = recipient_from_history(OPEN * 3)
    _, base_entrenched = recipient_from_history(ENTRENCHED * 3)
    assert base_open > base_entrenched                    # per-recipient base rate reflects disposition
    rv_open, _ = recipient_from_history(OPEN * 3)
    rv_ent, _ = recipient_from_history(ENTRENCHED * 3)
    assert rv_open["op_openness"] > rv_ent["op_openness"]


def test_history_store_is_asof_leakage_safe():
    store = HistoryStore(engine=DeepInferenceEngine())
    store.ingest("e", OPEN[0], ts=1)
    store.ingest("e", OPEN[1], ts=2)
    store.ingest("e", ENTRENCHED[0], ts=10)
    assert store.depth("e", now=5) == 2                   # only docs strictly before now=5
    assert store.depth("e", now=100) == 3
    rv, base = store.recipient("e", now=5)                # persona from pre-now docs only
    assert 0.0 <= base <= 1.0


def test_more_history_raises_confidence_moves_off_prior():
    # a single doc barely moves off the population prior; a deep, consistent history moves fully
    _, base_thin = recipient_from_history(OPEN[:1])
    _, base_deep = recipient_from_history(OPEN * 5)
    assert abs(base_deep - 0.5) >= abs(base_thin - 0.5)   # deeper evidence -> further from the 0.5 prior
