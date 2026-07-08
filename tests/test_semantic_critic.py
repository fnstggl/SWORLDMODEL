"""The semantic critic: catch AI slop the variable model is blind to, and repair it at the gate.

Two failure modes the lexical strategy scorer cannot see: incoherent/embellished sentences that don't
parse when read, and annoying tryhard lines. The critic must flag both, pass clean concrete writing, and
the polish gate must rewrite flagged lines out of a constructed email.
"""
from swm.decision.compositional_search import (ConstructedEmail, construct_email, encode_text_to_strategy,
                                                polish_email)
from swm.decision.message_optimizer import optimize_strategy
from swm.decision.semantic_critic import SemanticCritic, strip_em_dashes
from swm.decision.strategy_scorer import scorer_from_recipient

SKEPTIC = {"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9, "openness_to_outreach": 0.9,
           "attention_availability": 0.4, "platform_response_norm": 0.3, "relationship_strength": 0.0}

SLOP = "The secret I'm betting on: most of the AI stack rents margin it should own, and inference is where that flips."
ANNOYING = "Is that thesis obviously wrong to you? One line back and I'll leave you alone."
CLEAN = ("Peter, I read your essay on secrets. I'm 17 and building in AI instead of going to college. "
         "I build software that cuts the cost of running large models by about 40%. Do you think that's wrong?")


def test_critic_flags_incoherent_embellishment():
    c = SemanticCritic().critique(SLOP)
    assert c.coherence < 0.4                       # "rents margin it should own / that flips" doesn't parse
    assert c.flags()
    reasons = " ".join(r for f in c.flags() for r in f["reasons"])
    assert "vague referent" in reasons or "metaphor" in reasons


def test_critic_flags_annoying_lines():
    c = SemanticCritic().critique(ANNOYING)
    assert c.naturalness < 0.4
    assert any("leave you alone" in r or "one line" in r or "obviously wrong" in r
               for f in c.flags() for r in f["reasons"])


def test_critic_passes_clean_concrete_writing():
    c = SemanticCritic().critique(CLEAN)
    assert c.quality >= 0.7 and not c.flags()


# --- em-dash bias (all messages) ------------------------------------------------------------------

def test_strip_em_dashes_removes_all_dashes():
    assert "—" not in strip_em_dashes("Peter — the cost is rising — fast.")
    assert "–" not in strip_em_dashes("inference, not training – most disagree.")
    assert strip_em_dashes("— Beckett") == "Beckett"                 # sign-off dash dropped
    assert strip_em_dashes("no dashes here.") == "no dashes here."    # untouched


def test_critic_flags_em_dashes():
    c = SemanticCritic().critique("This is a fine sentence but it has a dash — right here.")
    assert c.naturalness < 1.0
    assert any("dash" in r for f in c.flags() for r in f["reasons"])


def test_constructed_email_has_no_em_dashes():
    scorer = scorer_from_recipient(SKEPTIC, 0.2)
    spec = optimize_strategy(scorer)
    email = construct_email(scorer, spec.strategy)
    assert "—" not in email.text and "–" not in email.text


def test_critic_quality_is_min_of_axes():
    c = SemanticCritic().critique(SLOP + " " + ANNOYING)
    assert c.quality == min(c.coherence, c.naturalness)


def test_llm_judge_path_is_used_when_provided():
    def judge(sentences):
        return [{"coherent": False, "annoying": True, "reason": "llm says slop"} for _ in sentences]
    c = SemanticCritic(judge_fn=judge).critique("Any two sentences. Here is another.")
    assert c.source == "llm" and c.coherence == 0.0 and c.naturalness == 0.0


# --- construction avoids slop, and the gate repairs it -------------------------------------------

def test_beam_search_avoids_slop_and_annoying_moves():
    scorer = scorer_from_recipient(SKEPTIC, 0.2)
    spec = optimize_strategy(scorer)
    email = construct_email(scorer, spec.strategy, critic=SemanticCritic())
    assert "rents margin" not in email.text          # the incoherent thesis is pruned
    assert "leave you alone" not in email.text        # the annoying ask is pruned
    assert email.critique.quality >= 0.6


def test_polish_gate_repairs_a_sloppy_email():
    scorer = scorer_from_recipient(SKEPTIC, 0.2)
    spec = optimize_strategy(scorer)
    # force a sloppy assembly (the slop thesis + the annoying ask)
    sloppy_slots = {"opener": "Peter —", "hook": "I'm 17 and building in AI instead of going to college.",
                    "thesis": SLOP, "ask": ANNOYING, "close": ""}
    text = " ".join(v for v in sloppy_slots.values() if v)
    dirty = ConstructedEmail(text=text, strategy=encode_text_to_strategy(text), score=0.0,
                             mean=0.0, lower_bound=0.0, slots=sloppy_slots)
    before = SemanticCritic().critique(text)
    polished = polish_email(dirty, scorer, spec.strategy, critic=SemanticCritic())
    assert "rents margin" not in polished.text and "leave you alone" not in polished.text
    assert polished.critique.quality > before.quality
    assert len(polished.critique.flags()) < len(before.flags())
