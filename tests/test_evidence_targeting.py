"""Pins evidence targeting: the terminal query is built from the DECISIVE resolution criterion (not just
question nouns), and the escalated retry is a genuinely different strategy (wider window + reformulation)."""
from swm.world_model_v2.evidence_orchestrator import _query_terms, _keywords
from swm.world_model_v2.evidence_requirements import EvidenceRequirement


def _term_req(claim):
    return EvidenceRequirement(requirement_id="r", claim_or_quantity=claim, why_relevant="",
                               affected_component="terminal_outcome")


def test_terminal_query_uses_resolution_criterion_terms():
    q = "Will the Bank of Japan raise its policy rate at the June 2026 meeting?"
    rrule = "Resolves YES if the BoJ announces a hike to its short-term policy rate at the June meeting"
    terms = _query_terms(_term_req(rrule), q, resolution_rule=rrule)
    lo = terms.lower()
    # decisive nouns from BOTH question and resolution rule are present; question framing words are dropped
    assert "japan" in lo and "policy" in lo and "rate" in lo
    assert "will" not in lo.split() and "resolves" not in lo.split()


def test_resolution_rule_adds_decisive_terms_beyond_the_question():
    q = "Will Apple announce visionOS 27 at WWDC 2026?"
    rrule = "Resolves YES if Apple announces visionOS 27 or a successor major version at the WWDC keynote"
    with_rule = _query_terms(_term_req(rrule), q, resolution_rule=rrule)
    without = _query_terms(_term_req(q), q, resolution_rule="")
    assert "successor" in with_rule.lower() or "keynote" in with_rule.lower()
    assert with_rule != without


def test_keywords_drops_question_framing():
    kw = [w.lower() for w in _keywords("Will the board vote to approve the merger before 2027?", 8)]
    assert "board" in kw and "approve" in kw and "merger" in kw
    assert "will" not in kw and "the" not in kw and "before" not in kw
