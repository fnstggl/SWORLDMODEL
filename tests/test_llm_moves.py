"""Live-LLM seams (proposer / judge / rewriter) — tested hermetically with a MOCK chat_fn (no network).

Proves the plumbing: the strategy vector becomes writing instructions, the LLM writes candidate MOVES,
the LLM judges sentences, the LLM rewrites a flagged line, and the whole pipeline runs through the LLM
path and still produces a critic-clean email. The live model is exercised only in experiments/exp084.
"""
import json

from swm.decision.llm_moves import (SenderBrief, llm_proposer, llm_rewriter, llm_sentence_judge,
                                     spec_to_instructions, _extract_list)
from swm.decision.message_pipeline import RecipientState, optimize_message
from swm.decision.strategy_scorer import MESSAGE_VARS


def mock_chat(prompt: str) -> str:
    """Deterministic stand-in for a live LLM, keyed on what the prompt is asking for."""
    if "Rewrite this ONE line" in prompt:
        return "The cost of running models is rising faster than the cost of training them."
    if "COHERENT" in prompt and "ANNOYING" in prompt:
        n = prompt.count("\n") + 20
        return json.dumps([{"coherent": True, "annoying": False, "reason": ""} for _ in range(n)])
    # proposer
    return json.dumps([
        "Peter, running models now costs more than training them.",
        "I'm 17 and building infrastructure for inference.",
        "Do you think that's wrong?",
    ])


# --- strategy -> instructions ---------------------------------------------------------------------

def test_spec_to_instructions_reflects_the_strategy():
    rules = spec_to_instructions({**{v: 0.3 for v in MESSAGE_VARS}, "credential_signaling": 0.0,
                                  "credibility_proof": 0.9, "responder_incentive": 0.9, "pushiness": 0.0})
    text = " ".join(rules).lower()
    assert "do not mention" in text and ("schools" in text or "titles" in text)  # sign-flip -> writing rule
    assert "proof" in text or "traction" in text                                 # general lever
    assert "gets out of engaging" in text or "personally" in text                # responder_incentive
    assert "asap" in text or "no urgency" in text


def test_spec_to_instructions_surfaces_situational_levers():
    from swm.decision.strategy_scorer import Lever
    lv = Lever("contrarian_thesis", 1.8, 0.7, "make one specific non-consensus claim")
    rules = spec_to_instructions({**{v: 0.3 for v in MESSAGE_VARS}, "contrarian_thesis": 0.9}, levers=[lv])
    assert any("non-consensus" in r for r in rules)


# --- proposer -------------------------------------------------------------------------------------

def test_llm_proposer_returns_candidate_strings():
    propose = llm_proposer(mock_chat, recipient_notes="Peter is a contrarian.",
                           sender=SenderBrief(sender="Beckett", thesis="inference is the bottleneck"))
    opts = propose("thesis", {v: 0.5 for v in MESSAGE_VARS}, {"prefix": ""})
    assert isinstance(opts, list) and all(isinstance(o, str) for o in opts)
    assert any("inference" in o.lower() or "models" in o.lower() for o in opts)


def test_proposer_allows_empty_for_optional_slots():
    propose = llm_proposer(mock_chat)
    assert "" in propose("hook", {v: 0.5 for v in MESSAGE_VARS}, {})


# --- judge + rewriter -----------------------------------------------------------------------------

def test_llm_sentence_judge_shape():
    judge = llm_sentence_judge(mock_chat)
    out = judge(["A concrete sentence.", "Another one."])
    assert len(out) == 2 and all("coherent" in v and "annoying" in v for v in out)


def test_llm_rewriter_returns_a_line():
    rw = llm_rewriter(mock_chat, sender=SenderBrief(sender="Beckett"))
    out = rw("most of the AI stack rents margin it should own", ["vague metaphor"], {v: 0.5 for v in MESSAGE_VARS})
    assert isinstance(out, str) and out and "\n" not in out


# --- tolerant parsing -----------------------------------------------------------------------------

def test_extract_list_handles_json_and_lines():
    assert _extract_list('["a", "b", "c"]') == ["a", "b", "c"]
    assert _extract_list('1. first option\n2. second option') == ["first option", "second option"]


# --- pipeline through the LLM path ----------------------------------------------------------------

def test_pipeline_uses_llm_path_end_to_end():
    rs = RecipientState(vars={"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9,
                              "openness_to_outreach": 0.9, "attention_availability": 0.4,
                              "platform_response_norm": 0.3, "relationship_strength": 0.0},
                        base_mean=0.2, base_n_effective=6.0, label="Peter Thiel")
    res = optimize_message(rs, chat_fn=mock_chat, sender_brief=SenderBrief(sender="Beckett"),
                           recipient_notes="contrarian", n_mc=400, beam=2, seed=0)
    assert res.email.text                                    # produced an email via the LLM proposer
    assert res.email.critique.source == "llm"                # judged by the LLM critic
    assert res.email.critique.quality >= 0.5
