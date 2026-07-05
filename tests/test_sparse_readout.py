"""Tests for learned feature discipline — L1 sparsity and correlation-screened readouts."""
import random

from swm.transition.sparse_readout import ScreenedLogisticReadout, SparseLogisticReadout, _soft_threshold


def test_soft_threshold_produces_zeros():
    assert _soft_threshold(0.3, 0.5) == 0.0        # inside the threshold -> exactly zero
    assert abs(_soft_threshold(0.8, 0.5) - 0.3) < 1e-9
    assert abs(_soft_threshold(-0.8, 0.5) + 0.3) < 1e-9


def _noisy_data(n=400, d=8, informative=(0, 1), seed=0):
    """Only a couple of the d features carry signal; the rest are pure noise."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        row = [rng.gauss(0, 1) for _ in range(d)]
        z = sum(row[j] for j in informative)
        X.append(row); y.append(int(z + rng.gauss(0, 0.5) > 0))
    return X, y


def test_l1_drives_noise_features_to_zero():
    X, y = _noisy_data()
    m = SparseLogisticReadout(l1=0.05, l2=0.01, epochs=300).fit(X, y)
    assert m.sparsity() > 0.0                       # some coefficients are exactly zero
    kept = {n for n, _ in m.nonzero([f"f{j}" for j in range(len(X[0]))])}
    assert "f0" in kept or "f1" in kept             # keeps at least one informative feature


def test_screened_readout_selects_informative_features():
    X, y = _noisy_data(informative=(2, 5))
    m = ScreenedLogisticReadout(k=2).fit(X, y)
    sel = set(m.selected([f"f{j}" for j in range(len(X[0]))]))
    assert {"f2", "f5"} & sel                        # recovers a truly informative feature
    assert len(m.keep_) == 2


def test_screened_readout_tunes_k_without_leakage():
    X, y = _noisy_data(informative=(0,))
    m = ScreenedLogisticReadout(k=None, k_grid=(1, 2, 3)).fit(X, y)
    assert 1 <= len(m.keep_) <= 3
    # predictions are valid probabilities
    p = m.predict_proba(X[0])
    assert 0.0 <= p <= 1.0


def test_screened_beats_dense_on_mostly_noise():
    """Filtering to informative features should generalize better than using all noisy ones."""
    from swm.transition.readout import LogisticReadout
    from swm.eval.metrics import log_loss
    Xtr, ytr = _noisy_data(n=300, d=20, informative=(0, 1), seed=1)
    Xte, yte = _noisy_data(n=300, d=20, informative=(0, 1), seed=2)
    dense = LogisticReadout(l2=1.0, epochs=300).fit(Xtr, ytr)
    scr = ScreenedLogisticReadout(k=None, k_grid=(1, 2, 3, 5)).fit(Xtr, ytr)
    ll_dense = log_loss(yte, [min(1 - 1e-9, max(1e-9, dense.predict_proba(x))) for x in Xte])
    ll_scr = log_loss(yte, [min(1 - 1e-9, max(1e-9, scr.predict_proba(x))) for x in Xte])
    assert ll_scr <= ll_dense + 1e-6                 # screening is no worse, usually better
