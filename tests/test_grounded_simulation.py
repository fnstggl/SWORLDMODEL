"""Tests for EXP-040 grounded-variable simulation (the north-star thesis test)."""
from experiments.exp040_grounded_simulation import _QModel, _split, _tv


def _rows(qid="Q", n_opt=2):
    # a question where party perfectly separates the answer -> a real, learnable variable
    out = []
    for i in range(40):
        party = "democrat" if i % 2 == 0 else "republican"
        ans = 0 if party == "democrat" else 1
        out.append({"uid": f"u{i}", "qid": qid, "n_opt": n_opt,
                    "demo": {"party": party, "ideology": "x", "religion": "x", "attendance": "x",
                             "age": "x", "education": "x", "income": "x", "race": "x", "sex": "x",
                             "region": "x", "marital": "x"},
                    "answer_idx": ans})
    return out


def test_grounding_beats_marginal_when_variable_is_informative():
    m = _QModel(_rows(), 2)
    dem = {"party": "democrat"}
    marg = m.predict(dem, [])                      # composite: the marginal (≈0.5/0.5)
    grounded = m.predict(dem, ["party"], alpha=1.0)  # grounded on the informative variable
    assert abs(marg[0] - 0.5) < 0.1                # marginal is near 50/50
    assert grounded[0] > marg[0]                   # knowing they are a democrat raises P(answer 0)
    assert grounded[0] > 0.7                       # and does so decisively


def test_uninformative_variable_does_little_harm_under_shrinkage():
    m = _QModel(_rows(), 2)
    dem = {"party": "democrat", "religion": "x"}
    only_party = m.predict(dem, ["party"], alpha=4.0)
    plus_noise = m.predict(dem, ["party", "religion"], alpha=4.0)   # religion is constant -> no signal
    assert abs(only_party[0] - plus_noise[0]) < 0.05               # shrinkage keeps the noise variable inert


def test_predict_returns_normalized_distribution():
    m = _QModel(_rows(), 2)
    p = m.predict({"party": "republican"}, ["party"])
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(0.0 <= x <= 1.0 for x in p)


def test_split_is_disjoint_and_salt_changes_partition():
    recs = [{"uid": f"u{i}"} for i in range(200)]
    tr, te = _split(recs, 0.3)
    tr_ids = {r["uid"] for r in tr}; te_ids = {r["uid"] for r in te}
    assert tr_ids.isdisjoint(te_ids)
    assert len(tr) + len(te) == 200
    tr2, _ = _split(recs, 0.3, salt=1)
    assert {r["uid"] for r in tr2} != tr_ids       # a different salt yields a different partition


def test_tv_distance():
    assert _tv([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert _tv([0.5, 0.5], [0.5, 0.5]) == 0.0
