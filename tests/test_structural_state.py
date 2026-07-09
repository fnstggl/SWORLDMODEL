"""Tests for the structural-state layer — detection, fragility, and the structural×event coupling (no network)."""
import json

from swm.api.latent_forecast import _social_driver, latent_forecast
from swm.api.structural_state import StructuralGrounder, detect_iso3


def test_detect_iso3():
    assert detect_iso3("Will there be a coup in Sudan?")[0] == "SDN"
    assert detect_iso3("Will Ukraine join NATO?")[0] == "UKR"
    assert detect_iso3("Will Bitcoin hit 100k?")[0] is None


def test_fragility_orders_states():
    frag = StructuralGrounder._fragility
    robust = frag({"electoral_democracy": 0.9, "rule_of_law": 0.9, "corruption": 0.1},
                  {"inflation_pct": 2.0, "unemployment_pct": 4.0})
    fragile = frag({"electoral_democracy": 0.18, "rule_of_law": 0.01, "corruption": 0.97},
                   {"inflation_pct": 40.0, "unemployment_pct": 20.0})
    assert 0.0 <= robust < 0.3 and 0.7 < fragile <= 1.0      # institutions + economy both order the state
    assert frag({}, {}) == 0.5                               # no data -> neutral


def test_social_driver_scaled_by_fragility():
    social = {"driver": {"escalation_magnitude": 0.8, "escalation_trend": 0.4, "goldstein": -2.0,
                         "violence_rate": 0.1}}

    class S:
        raw = {"conflict_pushes": 1}
    robust = _social_driver(S(), social, fragility=0.1)["strength"]
    fragile = _social_driver(S(), social, fragility=0.9)["strength"]
    assert fragile > robust                                  # a fragile state amplifies the same shock
    assert _social_driver(S(), social)["strength"] > 0       # fragility optional (backward compatible)


def test_structural_block_and_coupling_flow_through_forecast():
    struct = {"iso3": "VEN", "name": "venezuela", "vdem": {}, "econ": {}, "fragility": 0.71,
              "block": "STRUCTURAL STATE of Venezuela: fragility 0.71\n"}
    soc = {"country": "VE", "name": "venezuela", "state": {}, "block": "MEASURED SOCIAL STATE: conflictual\n",
           "driver": {"escalation_magnitude": 0.8, "escalation_trend": 0.4, "goldstein": -2.0,
                      "violence_rate": 0.1}}

    class SG:
        def ground_structural(self, q, a):
            return struct

    class SoG:
        def ground_social(self, q, a):
            return soc

    seen = {}

    def llm(p):
        seen["p"] = p
        return json.dumps({"base_rate": 0.3, "kind": "event", "conflict_pushes": 1, "drivers": []})

    p, spec = latent_forecast("Will Venezuela see regime change?", 1780000000, 1795000000, llm, n=1500,
                              social_grounder=SoG(), structural_grounder=SG())
    assert "STRUCTURAL STATE" in seen["p"] and "MEASURED SOCIAL" in seen["p"]   # both state layers in the prompt
    assert spec.raw["_structural"] == {"iso3": "VEN", "fragility": 0.71}
    # a fragile state pushes the escalation forecast higher than a robust one would
    struct2 = {**struct, "fragility": 0.1, "block": "STRUCTURAL STATE: fragility 0.1\n"}

    class SG2:
        def ground_structural(self, q, a):
            return struct2

    p_robust, _ = latent_forecast("Will Venezuela see regime change?", 1780000000, 1795000000, llm, n=1500,
                                  social_grounder=SoG(), structural_grounder=SG2())
    assert p > p_robust
