"""Tests for the state-grounding layer (offline, mock grounders) — triage → measure → feed the model."""
from swm.api.model_spec import ModelSpec, SpecVar
from swm.api.state_grounding import (DataGrounder, GroundedValue, RetrievalGrounder, StateGrounder,
                                     ground_features)


def _readout_spec():
    # two readout variables: one HIGH-leverage (big weight + spread) and one negligible (tiny weight)
    return ModelSpec(mechanism="calibrated_readout",
                     variables=[SpecVar("inflation", value=0.5, est_sd=0.3, lo=0.0, hi=1.0,
                                        weight=4.0, weight_sd=0.5, center=0.5),
                                SpecVar("noise", value=0.5, est_sd=0.3, lo=0.0, hi=1.0,
                                        weight=0.001, weight_sd=0.1, center=0.5)],
                     outcome={"event": {"op": ">", "value": 0.5}}, extra={"intercept": 0.0})


def test_data_grounder_returns_value_or_none():
    g = DataGrounder(lambda var, as_of: (0.83, 0.02) if var == "inflation" else None, name="fred")
    gv = g.ground("inflation")
    assert isinstance(gv, GroundedValue) and gv.value == 0.83 and gv.sd == 0.02 and gv.source == "fred"
    assert g.ground("unemployment") is None                       # unmeasurable -> None (stays at prior)


def test_retrieval_grounder_uses_extractor():
    retr = type("R", (), {"retrieve": lambda self, q, as_of=None: ["doc about a 0.9 value"]})()
    g = RetrievalGrounder(retr, lambda var, q, ev: {"value": 0.9, "sd": 0.15}, name="rag")
    gv = g.ground("sentiment", question="how positive?")
    assert gv.value == 0.9 and gv.sd == 0.15 and gv.source == "rag"
    g2 = RetrievalGrounder(retr, lambda var, q, ev: None)
    assert g2.ground("sentiment") is None


def test_ground_spec_grounds_only_high_leverage_and_records_provenance():
    # a source that CAN measure both variables, but triage should only invest in the high-leverage one
    fetch = lambda var, as_of: ({"inflation": (0.9, 0.02), "noise": (0.1, 0.02)}).get(var)
    sg = StateGrounder(grounders={"inflation": DataGrounder(fetch, name="fred"),
                                  "noise": DataGrounder(fetch, name="fred")}, keep_frac=0.9)
    grounded, report = sg.ground_spec(_readout_spec(), question="up?")
    inf = grounded.var("inflation")
    assert inf.value == 0.9 and inf.est_sd == 0.02             # high-leverage var replaced with the measurement
    assert grounded.var("noise").value == 0.5                 # negligible var left at its prior (not grounded)
    rec = {r["var"]: r for r in report}
    assert rec["inflation"]["grounded"] is True and rec["inflation"]["source"] == "fred"
    assert rec["noise"]["grounded"] is False and rec["noise"]["high_leverage"] is False
    # the original spec is untouched (ground_spec returns a copy)
    assert _readout_spec().var("inflation").value == 0.5


def test_ground_all_bypasses_triage():
    fetch = lambda var, as_of: ({"inflation": (0.9, 0.02), "noise": (0.1, 0.02)}).get(var)
    sg = StateGrounder(grounders={"inflation": DataGrounder(fetch), "noise": DataGrounder(fetch)},
                       ground_all=True)
    grounded, report = sg.ground_spec(_readout_spec())
    assert grounded.var("inflation").value == 0.9 and grounded.var("noise").value == 0.1   # both grounded


def test_ungroundable_high_leverage_var_stays_at_prior():
    sg = StateGrounder(grounders={}, default=DataGrounder(lambda var, as_of: None))
    grounded, report = sg.ground_spec(_readout_spec())
    assert grounded.var("inflation").value == 0.5             # nothing measured -> honest prior, no fake precision
    assert all(r["grounded"] is False for r in report)


def test_default_grounder_tried_when_no_specific_one():
    sg = StateGrounder(grounders={}, default=DataGrounder(lambda var, as_of: (0.7, 0.05), name="def"),
                       ground_all=True)
    grounded, _ = sg.ground_spec(_readout_spec())
    assert grounded.var("inflation").value == 0.7 and grounded.var("noise").value == 0.7


def test_grounder_list_tried_in_order():
    miss = DataGrounder(lambda var, as_of: None, name="a")
    hit = DataGrounder(lambda var, as_of: (0.6, 0.1), name="b")
    sg = StateGrounder(grounders={"inflation": [miss, hit]}, ground_all=True)
    grounded, report = sg.ground_spec(_readout_spec())
    assert grounded.var("inflation").value == 0.6
    assert next(r for r in report if r["var"] == "inflation")["source"] == "b"   # fell through to the 2nd


def test_ground_features_falls_back_to_guess():
    fetch = lambda var, as_of: (0.8, 0.02) if var == "a" else None
    sg = StateGrounder(grounders={"a": DataGrounder(fetch)}, ground_all=True)
    vec = ground_features(["a", "b"], sg, guess={"a": 0.1, "b": 0.5})
    assert vec == [0.8, 0.5]                                   # a measured, b falls back to the guess
