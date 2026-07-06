"""Tests for the interventional harness (EXP-054) — feature extraction + off-policy scoring pieces."""
from experiments.exp054_interventional import _RidgeGD, _features, _split


def test_features_shape_and_signals():
    f = _features("Why This Amazing Thing Will Shock You?")
    assert len(f) == 10
    assert f[1] == 1.0                                    # has question mark
    assert f[6] > 0 and f[7] > 0                          # curiosity + emotional words fire


def test_features_distinguish_headlines():
    plain = _features("A city council meeting was held.")
    clicky = _features("You Won't Believe What Happened Next! 7 Shocking Reasons Why")
    assert clicky[3] == 1.0 and plain[3] == 0.0          # number present only in the clicky one
    assert clicky[7] >= plain[7]                          # more emotional words


def test_ridge_learns_linear_signal():
    X = [[float(i)] for i in range(40)]
    y = [0.02 + 0.001 * i for i in range(40)]             # ctr increases with the feature
    m = _RidgeGD(l2=0.01, epochs=600).fit(X, y)
    assert m.predict([39.0]) > m.predict([0.0])           # recovers the positive relationship


def test_split_is_deterministic_and_disjoint():
    tests = [{"test_id": f"t{i}", "arms": []} for i in range(200)]
    tr, te = _split(tests)
    ids_tr = {t["test_id"] for t in tr}; ids_te = {t["test_id"] for t in te}
    assert ids_tr.isdisjoint(ids_te)
    tr2, _ = _split(tests)
    assert {t["test_id"] for t in tr2} == ids_tr          # deterministic (crc32)
