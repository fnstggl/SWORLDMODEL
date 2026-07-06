"""Tests for the LLM-informed prior estimator (EXP-049)."""
from swm.variables.llm_prior import LLMPriorReadout, prior_features, prior_value


def test_prior_direction_matches_world_knowledge():
    # conservatives favor the death penalty (cappun "1" pole is conservative) -> positive push
    assert prior_value("cappun", "ideology", "conservative") > 0
    assert prior_value("cappun", "ideology", "liberal") < 0
    # the secular favor legal marijuana (grass "1" pole is liberal) -> secular pushes toward YES
    assert prior_value("grass", "relig", "none") > 0
    assert prior_value("grass", "relig", "protestant") < 0


def test_item_attr_amplification():
    # race matters more on the racial-spending item than the pure axis
    assert abs(prior_value("natrace", "race", "black")) > abs(prior_value("natheal", "race", "black"))


def test_prior_features_length():
    attrs = ["party", "ideology", "relig"]
    f = prior_features("cappun", {"party": "republican", "ideology": "conservative", "relig": "none"}, attrs)
    assert len(f) == 3


def test_prior_only_predicts_without_data():
    # zero-shot: a conservative should be predicted more likely to favor the death penalty than a liberal
    m = LLMPriorReadout(attrs=["party", "ideology", "relig", "attendance"], prior_only=True).fit([])
    con = m.predict("cappun", {"party": "republican", "ideology": "conservative", "relig": "protestant",
                               "attendance": "high"})
    lib = m.predict("cappun", {"party": "democrat", "ideology": "liberal", "relig": "none",
                               "attendance": "low"})
    assert con > lib


def test_data_updates_the_prior():
    attrs = ["party"]
    # a dataset where the prior direction is RIGHT; fitting should keep/So sharpen it, not flip it
    rows = []
    for i in range(60):
        rep = i % 2 == 0
        rows.append({"item": "cappun", "answer": 1 if rep else 0,
                     "demo": {"party": "republican" if rep else "democrat"}})
    m = LLMPriorReadout(attrs=attrs, l2=1.0).fit(rows)
    assert m.predict("cappun", {"party": "republican"}) > m.predict("cappun", {"party": "democrat"})
