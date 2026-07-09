"""Tests for GDELT social-state grounding — country detection + driver derivation (no network, no LLM)."""
import json

from swm.api.gdelt_social import GdeltSocialGrounder, detect_country
from swm.api.latent_forecast import _social_driver, latent_forecast


def test_country_detection_boundaries_and_longest_match():
    assert detect_country("Will there be a ceasefire in Ukraine?")[0] == "UP"   # trailing punctuation
    assert detect_country("Will Israel strike Iran.")[0] in ("IS", "IR")        # a country is found
    assert detect_country("Ukraine?")[0] == "UP"
    assert detect_country("Will Bitcoin hit 100k?")[0] is None                  # no country -> no grounding
    assert detect_country("")[0] is None


def test_social_driver_sign_follows_llm_polarity():
    social = {"driver": {"escalation_magnitude": 0.8, "escalation_trend": 0.4, "goldstein": -2.0,
                         "violence_rate": 0.1}}

    class S:                                                  # a spec stub carrying the LLM's polarity read
        raw = {"conflict_pushes": 1}
    d = _social_driver(S(), social)
    assert d["direction"] == 1.0 and d["grounded"] is True and d["strength"] > 0
    S.raw = {"conflict_pushes": -1}                           # a ceasefire/peace question: conflict pushes NO
    assert _social_driver(S(), social)["direction"] == -1.0


def test_social_driver_falls_back_to_goldstein_when_polarity_missing():
    social = {"driver": {"escalation_magnitude": 0.6, "escalation_trend": 0.2, "goldstein": -3.0,
                         "violence_rate": 0.1}}

    class S:
        raw = {}                                             # no conflict_pushes -> conflictual state => escalation
    assert _social_driver(S(), social)["direction"] == 1.0


def test_social_driver_none_when_calm():
    social = {"driver": {"escalation_magnitude": 0.0, "escalation_trend": 0.0, "goldstein": 5.0,
                         "violence_rate": 0.0}}

    class S:
        raw = {"conflict_pushes": 1}
    assert _social_driver(S(), social) is None               # nothing measured -> no spurious driver


def test_grounder_returns_none_without_country_or_asof():
    g = GdeltSocialGrounder(window_days=8)
    assert g.ground_social("Will Bitcoin hit 100k?", 1737331200) is None
    assert g.ground_social("Will Russia escalate?", None) is None


def test_latent_forecast_injects_grounded_social_driver(monkeypatch):
    # stub the grounder so the test is network-free; verify the block reaches the prompt and the driver the sim
    block = "MEASURED SOCIAL STATE of Ukraine as of 2025-01-20 (GDELT ...): conflictual, rising\n"
    social = {"country": "UP", "name": "ukraine", "state": {}, "block": block,
              "driver": {"escalation_magnitude": 0.8, "escalation_trend": 0.5, "goldstein": -2.0,
                         "violence_rate": 0.1}}

    class StubGrounder:
        def ground_social(self, q, as_of):
            return social

    seen = {}

    def mock_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({"base_rate": 0.3, "kind": "event", "conflict_pushes": 1,
                           "drivers": [{"factor": "war", "direction": 1, "strength": 0.4, "grounded": True}]})

    p, spec = latent_forecast("Will Russia escalate the war in Ukraine?", 1737331200, 1750000000,
                              mock_llm, n=1500, social_grounder=StubGrounder())
    assert "MEASURED SOCIAL STATE" in seen["prompt"] and "conflict_pushes" in seen["prompt"]
    assert len(spec.drivers) == 2                             # LLM driver + grounded social driver
    assert spec.raw["_country"] == "UP"
    # with the escalation driver, the forecast sits above the same sim without it
    p0, _ = latent_forecast("Will Russia escalate the war in Ukraine?", 1737331200, 1750000000, mock_llm, n=1500)
    assert p > p0
