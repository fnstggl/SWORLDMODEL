"""Tests for the correlation-aware, partially-pooled readout (EXP-041, the estimator Part 2)."""
from swm.variables.pooled_readout import PooledLogisticReadout, encode, onehot_vocab


def _rows(qid, n, sep=True):
    # party perfectly separates the answer when sep=True; noise otherwise
    out = []
    for i in range(n):
        party = "democrat" if i % 2 == 0 else "republican"
        ans = (0 if party == "democrat" else 1) if sep else (i % 2)
        out.append({"qid": qid, "answer_idx": ans,
                    "demo": {"party": party, "religion": "none"}})
    return out


def test_onehot_and_encode():
    rows = _rows("q", 4)
    vocab = onehot_vocab(rows, ["party"])
    assert len(vocab) == 2
    x = encode({"party": "democrat"}, ["party"], vocab)
    assert sum(x) == 1.0


def test_learns_informative_variable():
    m = PooledLogisticReadout(attrs=["party"], tau=0.0).fit(_rows("q", 60), min_q=12)
    assert m.predict("q", {"party": "democrat"}) < 0.5      # democrats -> answer 0
    assert m.predict("q", {"party": "republican"}) > 0.5    # republicans -> answer 1


def test_partial_pooling_shrinks_data_poor_toward_marginal():
    # a tiny question (n below min_q) has no fitted model -> returns the marginal, never overfits
    m = PooledLogisticReadout(attrs=["party"], tau=20.0).fit(_rows("small", 8), min_q=12)
    p = m.predict("small", {"party": "democrat"})
    assert abs(p - 0.5) < 0.2                               # ~marginal (balanced), not a confident 0/1


def test_tau_controls_pooling_strength():
    rows = _rows("q", 40)
    no_pool = PooledLogisticReadout(attrs=["party"], tau=0.0).fit(rows)
    heavy = PooledLogisticReadout(attrs=["party"], tau=1000.0).fit(rows)
    # heavy pooling pulls predictions toward the marginal (0.5 here) vs the confident unpooled model
    assert abs(heavy.predict("q", {"party": "democrat"}) - 0.5) < \
           abs(no_pool.predict("q", {"party": "democrat"}) - 0.5)


def test_unknown_question_returns_half():
    m = PooledLogisticReadout(attrs=["party"]).fit(_rows("q", 40))
    assert m.predict("does_not_exist", {"party": "democrat"}) == 0.5


def test_proba_normalized():
    m = PooledLogisticReadout(attrs=["party"], tau=0.0).fit(_rows("q", 40))
    d = m.proba("q", {"party": "democrat"})
    assert abs(sum(d) - 1.0) < 1e-9 and all(0 <= v <= 1 for v in d)
