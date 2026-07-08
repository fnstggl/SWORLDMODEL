"""Tests for the latent-factor readout (EXP-048) — decorrelate instead of shrink."""
from swm.variables.latent_factor_readout import LatentFactorReadout, _top_factors


def _rows(qid, n):
    out = []
    for i in range(n):
        # party and ideology are PERFECTLY correlated here (a redundant pair) -> one latent axis
        left = i % 2 == 0
        out.append({"qid": qid, "answer_idx": 0 if left else 1,
                    "demo": {"party": "democrat" if left else "republican",
                             "ideology": "liberal" if left else "conservative",
                             "religion": "none" if left else "protestant"}})
    return out


def test_top_factors_are_orthonormal():
    cov = [[2.0, 0.8, 0.0], [0.8, 2.0, 0.0], [0.0, 0.0, 1.0]]
    fs = _top_factors(cov, 2)
    assert abs(sum(a * a for a in fs[0]) - 1.0) < 1e-3            # unit norm
    assert abs(sum(a * b for a, b in zip(fs[0], fs[1]))) < 1e-3   # orthogonal


def test_collapses_redundant_variables_and_predicts():
    m = LatentFactorReadout(attrs=["party", "ideology", "religion"], k=2, tau=0.0).fit(_rows("q", 80))
    # the redundant party/ideology pair loads on one factor; the readout still separates the two groups
    assert m.predict("q", {"party": "democrat", "ideology": "liberal", "religion": "none"}) < 0.5
    assert m.predict("q", {"party": "republican", "ideology": "conservative", "religion": "protestant"}) > 0.5


def test_unknown_question_returns_half():
    m = LatentFactorReadout(attrs=["party"], k=2).fit(_rows("q", 40))
    assert m.predict("nope", {"party": "democrat"}) == 0.5


def test_data_poor_question_falls_back_to_marginal():
    m = LatentFactorReadout(attrs=["party"], k=2, tau=20.0).fit(_rows("small", 8), min_q=12)
    p = m.predict("small", {"party": "democrat"})
    assert abs(p - 0.5) < 0.25                                    # ~marginal, not overfit
