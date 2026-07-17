"""General social-message model: universal levers + per-recipient situational levers + LLM message encoder.

The fix for the Thiel-overfit set. The universal levers are the physics of any inbound ask; recipient-
specific qualities (contrarian for Thiel, traction for Cuban) are LLM-generated situational levers; and the
message encoder is a tightly-prompted LLM (with the lexical encoder as fallback), tested here with a mock.
"""
import json

from swm.decision.compositional_search import encode_text_to_strategy
from swm.decision.llm_moves import llm_message_encoder
from swm.decision.situational_levers import generate_levers, levers_summary
from swm.decision.strategy_scorer import MESSAGE_VARS, Lever, scorer_from_recipient


# --- the general set is recipient-agnostic and includes responder incentive ----------------------

def test_general_set_is_universal_and_has_responder_incentive():
    for v in ("responder_incentive", "credibility_proof", "relevance_fit", "warmth", "low_effort_ask"):
        assert v in MESSAGE_VARS
    for v in ("contrarian_pitch", "secret_density"):
        assert v not in MESSAGE_VARS                 # Thiel-specific vars are no longer universal


def test_responder_incentive_helps():
    sc = scorer_from_recipient({"status": 0.8, "attention_availability": 0.4}, 0.2)
    neutral = {v: 0.3 for v in MESSAGE_VARS}
    incent = {**neutral, "responder_incentive": 0.9}
    assert sc.mean(incent) > sc.mean(neutral)


# --- situational levers (LLM-generated) -----------------------------------------------------------

def _lever_mock(prompt):
    return json.dumps([
        {"name": "contrarian thesis", "description": "makes a specific non-consensus claim",
         "elasticity": 1.8, "confidence": 0.7},
        {"name": "personalization", "description": "should be ignored (dupes a universal lever)",
         "elasticity": 1.0, "confidence": 0.5},
    ])


def test_generate_levers_parses_and_dedupes_universal():
    levers = generate_levers(_lever_mock, "Peter Thiel", {"skepticism": 0.9})
    names = [lv.name for lv in levers]
    assert "contrarian_thesis" in names               # snake_cased
    assert "personalization" not in names             # dropped: duplicates a universal lever
    # caricature guard shrinks the mean toward zero by evidence confidence:
    # 1.8 × (0.4 + 0.6·0.7) = 1.476 — persona-derived levers are weak evidence about inbox behavior
    assert abs(levers[0].elasticity_mean - 1.8 * (0.4 + 0.6 * 0.7)) < 1e-9


def test_generate_levers_offline_returns_none():
    assert generate_levers(None, "Anyone") == []


def test_lever_enters_the_optimizable_space_and_scores():
    levers = generate_levers(_lever_mock, "Peter Thiel", {"skepticism": 0.9})
    sc = scorer_from_recipient({"skepticism": 0.9}, 0.2, levers=levers)
    assert "contrarian_thesis" in sc.optimizable_vars()
    base = {v: 0.3 for v in MESSAGE_VARS}
    assert sc.mean({**base, "contrarian_thesis": 1.0}) > sc.mean({**base, "contrarian_thesis": 0.0})


# --- LLM message encoder (mock), with lexical fallback --------------------------------------------

def test_llm_message_encoder_scores_general_vars():
    def enc_mock(prompt):
        # the LLM reads meaning: imperative ask is high directness even with no "?"
        return json.dumps({v: 0.5 for v in MESSAGE_VARS} | {"ask_directness": 0.9, "credibility_proof": 0.8})
    encode = llm_message_encoder(enc_mock)
    out = encode("Reply yes and I'll send the deck. We hit 10k ARR.")
    assert out["ask_directness"] == 0.9 and out["credibility_proof"] == 0.8
    assert set(MESSAGE_VARS).issubset(out.keys())


def test_llm_encoder_falls_back_to_lexical_on_error():
    def broken(prompt):
        raise RuntimeError("no backend")
    encode = llm_message_encoder(broken)
    out = encode("I'm a Wharton student. Reply yes.")
    # fell back to lexical, which still catches the broadened credential lexicon + imperative ask
    assert out["credential_signaling"] > 0.3 and out["ask_directness"] > 0.3


def test_llm_encoder_scores_situational_levers_too():
    lv = Lever("contrarian_thesis", 1.8, 0.7, "makes a specific non-consensus claim")
    def enc_mock(prompt):
        return json.dumps({v: 0.4 for v in MESSAGE_VARS} | {"contrarian_thesis": 0.95})
    out = llm_message_encoder(enc_mock, levers=[lv])("Everyone is wrong about inference.")
    assert out["contrarian_thesis"] == 0.95
