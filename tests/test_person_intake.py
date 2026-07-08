"""Tests for person-intake: the ask-the-user flow wired into the front door (deterministic, no network)."""
from swm.api.person_intake import PersonIntake
from swm.api.anchored_extractor import AnchoredExtractor
from swm.api.retrieval_grounding import CalibratedExtractor
from swm.api.world_model import WorldModel


def _identify_llm(is_person, entity="Jordan", variables=("openness", "relationship")):
    def fn(prompt):
        if "depend mainly on ONE SPECIFIC individual" in prompt:
            return ({"is_person": True, "entity": entity, "variables": list(variables)} if is_person
                    else {"is_person": False})
        return {"value": 0.5, "ci95": 0.2, "confidence": 0.6}     # any grounding/reference call
    return fn


def test_non_person_question_proceeds():
    pi = PersonIntake(llm=_identify_llm(False))
    assert pi.preflight("Will the Fed cut rates in March?")["mode"] == "proceed"


def test_person_question_thin_evidence_asks():
    pi = PersonIntake(llm=_identify_llm(True), search_fn=None)   # no footprint, no user context
    pf = pi.preflight("Will Jordan take a call if I email?")
    assert pf["mode"] == "ask" and pf["person"] == "Jordan" and len(pf["questions"]) >= 3


def test_person_question_with_user_context_proceeds_and_grounds():
    pi = PersonIntake(llm=_identify_llm(True), search_fn=None,
                      extractor=AnchoredExtractor(CalibratedExtractor(_identify_llm(True))))
    pf = pi.preflight("Will Jordan take a call?", user_context="met twice, warm, curious, said let's go deeper")
    assert pf["mode"] == "proceed" and pf["person"] == "Jordan"
    assert "Jordan" in pf["enriched_context"] and pf["inferred_person_variables"]     # variables inferred from dossier


def test_worldmodel_short_circuits_on_ask():
    # a WorldModel with person_intake returns the ask result instead of running the compiler
    class BoomCompiler:
        def compile(self, *a, **k):
            raise AssertionError("compiler must not run when we should ask the user")
    wm = WorldModel(compiler=BoomCompiler(), validate=False, person_intake=PersonIntake(llm=_identify_llm(True)))
    out = wm.simulate("Will my old manager reply to my note?")
    assert out["mode"] == "needs_user_context" and out["forecast"] is None and out["questions"]


def test_worldmodel_without_intake_unchanged():
    # no person_intake -> the preflight is skipped entirely (existing behavior)
    class Spec:
        mechanism = "x"; variables = []; equations = []; outcome = "o"; horizon = 1; rationale = "r"
    class Compiled:
        spec = Spec()
    class C:
        def compile(self, *a, **k): return Compiled()
    out = WorldModel(compiler=C(), validate=False, n=1).simulate("anything")
    assert out.get("person") is None and "mode" not in out
