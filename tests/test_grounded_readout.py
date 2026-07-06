"""Tests for the unified GroundedReadout (EXP-050) — factors + LLM prior + reliability weighting."""
from swm.variables.grounded_readout import GroundedReadout, RELIABILITY


def _rows(qid="cappun", n=80, noisy=False):
    out = []
    for i in range(n):
        rep = i % 2 == 0
        demo = {"party": "republican" if rep else "democrat",
                "ideology": "conservative" if rep else "liberal"}
        if noisy:
            demo["noisy"] = "a" if (i // 3) % 2 == 0 else "b"     # unrelated to the answer
        out.append({"qid": qid, "answer_idx": 1 if rep else 0, "demo": demo})
    return out


def test_prior_grounds_prediction_direction():
    m = GroundedReadout(attrs=["party", "ideology"], k=2, use_factors=False).fit(_rows())
    assert m.predict("cappun", {"party": "republican", "ideology": "conservative"}) > \
           m.predict("cappun", {"party": "democrat", "ideology": "liberal"})


def test_reliability_weight_lookup():
    assert RELIABILITY["data"] == 1.0
    assert RELIABILITY["llm"] < 1.0 and RELIABILITY["heuristic"] < RELIABILITY["llm"]


def test_reliability_attenuates_inferred_variable():
    # a low-reliability (llm) variable gets a smaller feature -> smaller effect than if trusted fully
    prov = {"party": "data", "ideology": "data", "noisy": "llm"}
    m = GroundedReadout(attrs=["party", "ideology", "noisy"], provenance=prov, use_factors=False,
                        use_reliability=True).fit(_rows(noisy=True))
    j = m.vocab.get(("noisy", "a"))
    assert m.rel[j] == RELIABILITY["llm"]                        # the inferred var's features are down-weighted


def test_factor_prior_projection_shapes():
    m = GroundedReadout(attrs=["party", "ideology"], k=2, use_factors=True, use_prior=True).fit(_rows())
    pc = m._prior_coef("cappun")
    assert len(pc) == 2                                          # prior projected into the 2-factor space


def test_fit_auto_selects_and_predicts():
    m = GroundedReadout(attrs=["party", "ideology"], k=2).fit_auto(_rows(n=120))
    assert set(m.chosen) == {"use_factors", "k", "use_prior"}
    p = m.predict("cappun", {"party": "republican", "ideology": "conservative"})
    assert 0.0 <= p <= 1.0


def test_unseen_question_returns_half():
    m = GroundedReadout(attrs=["party"]).fit(_rows())
    assert m.predict("nonexistent", {"party": "republican"}) == 0.5
