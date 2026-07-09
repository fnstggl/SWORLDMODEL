"""Tests for the dossier-assembly layer (Pillar 1: evidence from user context + history + web)."""
from swm.variables.dossier import (Dossier, DossierAssembler, needs_user_context, context_questions,
                                    infer_variables)


def test_assemble_priority_user_first():
    a = DossierAssembler(search_fn=lambda q, ao: ["web fact 1", "web fact 2"])
    d = a.assemble("Alex", user_context="we met at work; skeptical of hype",
                   message_history=["hey", "not sure about that"], question="will they invest")
    assert d.tags[0] == "user" and "web" in d.tags and "history" in d.tags
    assert d.passages[0].startswith("we met")           # user context ranked first


def test_strength_and_needs_user_context():
    thin = DossierAssembler().assemble("Stranger")       # no sources at all
    assert thin.strength == 0.0 and needs_user_context(thin)
    rich = DossierAssembler().assemble("Friend", user_context=["a", "b", "c", "d"])
    assert rich.strength > 0.35 and not needs_user_context(rich)


def test_context_questions_are_specific():
    qs = context_questions("Jordan", "a fundraising ask")
    assert len(qs) >= 3 and any("relationship" in q or "know" in q for q in qs)
    assert any("fundraising ask" in q for q in qs)       # topic-specific stance question


def test_infer_variables_uses_extractor_and_reports_strength():
    class FakeExtractor:
        def extract(self, variable, question, evidence):
            return {"value": 0.7 if evidence else 0.5, "sd": 0.1}
    d = Dossier("X", passages=["knows the topic well"], tags=["user"])
    out = infer_variables(d, ["openness", "topic_stance"], FakeExtractor(), question="q")
    assert set(out) == {"openness", "topic_stance"}
    assert out["openness"]["value"] == 0.7 and out["openness"]["evidence_strength"] == round(d.strength, 3)
