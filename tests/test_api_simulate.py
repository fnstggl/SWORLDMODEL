"""Tests for the unified simulate() API — regime routing, confidence, abstention, OOD guard."""
import random

from swm.api import Prediction, Simulator
from swm.state.state import Action


def _stream(n=800, seed=0):
    rng = random.Random(seed)
    theta = {f"e{i}": 0.12 + 0.75 * rng.random() for i in range(30)}
    insts = []
    for t in range(n):
        e = f"e{rng.randrange(30)}"
        a = Action(action_id=str(t), actor_id="s", channel="email", timing={"ts": t},
                   meta={"text": "hi, quick question?"})
        insts.append((e, a, None, int(rng.random() < theta[e])))
    return insts


def test_fit_produces_calibration_badge_and_support():
    sim = Simulator(platform="email").fit(_stream())
    assert sim.calibration["grade"] in {"A", "B", "C", "F"}
    assert sim.calibration["ece"] is not None
    # this stream is an entity-history regime, so most training mass is entity_state
    assert sim.train_support["entity_state"] > 0.5
    assert abs(sum(sim.train_support.values()) - 1.0) < 1e-6


def test_entity_state_query_is_confident_and_not_abstained():
    sim = Simulator(platform="email").fit(_stream())
    a = Action(action_id="x", actor_id="s", channel="email", timing={"ts": 2000},
               meta={"text": "hey, got a sec?"})
    r = sim.simulate("e0", a)
    assert isinstance(r, Prediction)
    assert r.regime == "entity_state"
    assert not r.abstain
    assert r.confidence > 0.3
    assert r.calibration["grade"] in {"A", "B", "C", "F"}   # badge rides on every prediction


def test_out_of_distribution_query_abstains_and_shrinks():
    """A model fit on the entity-state regime must flag an inference-only query it never trained on."""
    sim = Simulator(platform="email").fit(_stream())
    a = Action(action_id="x", actor_id="s", channel="email", timing={"ts": 2000},
               meta={"text": "hey"})
    r = sim.simulate("NEVER_SEEN", a,
                     llm_inference={"openness_to_outreach": {"value": 0.95, "confidence": 0.8},
                                    "skepticism": {"value": 0.05, "confidence": 0.8}})
    assert r.abstain                                        # inference_driven unsupported by this model
    # shrunk toward the base rate rather than an overconfident extreme
    assert abs(r.p - sim.base_rate) < abs(0.95 - sim.base_rate)


def test_cold_start_returns_prior_and_abstains():
    sim = Simulator(platform="email").fit(_stream())
    a = Action(action_id="y", actor_id="s", channel="generic", timing={"ts": 2000}, meta={"text": ""})
    r = sim.simulate("GHOST", a)
    assert r.abstain
    assert r.regime in {"cold_start", "message_only"}
    assert 0.0 < r.p < 1.0


def test_prediction_is_serializable_and_auditable():
    sim = Simulator(platform="email").fit(_stream())
    a = Action(action_id="x", actor_id="s", channel="email", timing={"ts": 2000},
               meta={"text": "quick q?"})
    d = sim.simulate("e1", a).as_dict()
    for k in ("p", "confidence", "regime", "abstain", "reason", "calibration", "provenance", "drivers"):
        assert k in d
