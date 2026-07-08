"""Tests for the candidate-and-select compiler loop: validity gating, critic ranking, agreement, drop-in."""
from swm.api.compiler import StructuralCompiler, cached_compile_fn
from swm.api.selecting_compiler import (SelectingCompiler, agreement, score_candidate, validity_score)
from swm.api.model_spec import parse_spec
from swm.api.world_model import WorldModel

GOOD = {"mechanism": "generic_scm",
        "variables": [{"name": "x", "value": 0.4, "est_sd": 0.02, "volatility": 0.02}],
        "equations": {"x": "0.2*(0.5 - x)"},               # equilibrium 0.5 -> inside [0,1] (clean)
        "outcome": {"variable": "x", "event": {"op": ">", "value": 0.5}}, "horizon": 8, "rationale": "clean"}
BAD = {"mechanism": "generic_scm",
       "variables": [{"name": "x", "value": 0.4, "est_sd": 0.02, "volatility": 0.02}],
       "equations": {"x": "0.2*(1.5 - x)"},                # equilibrium 1.5 -> OUTSIDE [0,1] (pins to bound)
       "outcome": {"variable": "x", "event": {"op": ">", "value": 0.5}}, "horizon": 8, "rationale": "buggy"}


def test_validity_gates_a_degenerate_candidate():
    good_score, gd = score_candidate(parse_spec(GOOD), "q")
    bad_score, bd = score_candidate(parse_spec(BAD), "q")
    assert good_score > bad_score                          # the mis-structured spec scores below the clean one
    assert gd["validity"] == 1.0 and bd["validity"] < 1.0
    assert any("equilibrium" in i["code"] or "bound" in i["code"] for i in bd["issues"])


def test_selector_picks_the_valid_candidate():
    cache = {"good": GOOD, "bad": BAD}
    sc = SelectingCompiler(StructuralCompiler(cached_compile_fn(cache)), keys=["bad", "good"])
    selected = sc.compile("Will x exceed 0.5?")
    assert selected.spec.equations["x"] == GOOD["equations"]["x"]     # picked the clean one despite bad first
    assert selected.verification["n_candidates"] == 2
    assert selected.verification["selected_detail"]["validity"] == 1.0
    # runs like any CompiledModel
    assert 0.0 <= selected.run(n=1000)["p_event"] <= 1.0


def test_critic_ranks_among_valid_candidates():
    # two CLEAN specs (a committee and a generic_scm); a critic that prefers 'committee' should tip selection
    committee = {"mechanism": "committee", "outcome": {"event": {"op": ">", "value": 0.5}},
                 "extra": {"agents": [{"id": f"a{i}", "position": p} for i, p in
                                      enumerate([0.72, 0.6, 0.55, 0.45, 0.38])], "rounds": 3},
                 "rationale": "a vote"}
    cache = {"scm": GOOD, "comm": committee}
    critic = lambda prompt: {"score": 0.95 if "committee" in prompt else 0.15, "critique": "mechanism fit"}
    sc = SelectingCompiler(StructuralCompiler(cached_compile_fn(cache)), keys=["scm", "comm"], critic_fn=critic)
    selected = sc.compile("Will the board approve?")
    assert selected.spec.mechanism == "committee"          # critic tipped it despite both being valid


def test_agreement_flags_mechanism_disagreement():
    from swm.api.compiler import CompiledModel
    scm = CompiledModel(parse_spec(GOOD))
    committee = CompiledModel(parse_spec({"mechanism": "committee",
                                          "outcome": {"event": {"op": ">", "value": 0.5}},
                                          "extra": {"agents": [{"id": "a", "position": 0.7}], "rounds": 2}}))
    agr = agreement([scm, committee], n=800)
    assert agr["mechanism_agreement"] == 0.5               # split vote -> low structural confidence
    assert set(agr["mechanism_vote"]) == {"generic_scm", "committee"}


def test_selecting_compiler_is_worldmodel_dropin():
    cache = {"good": GOOD, "bad": BAD}
    sc = SelectingCompiler(StructuralCompiler(cached_compile_fn(cache)), keys=["good", "bad"])
    wm = WorldModel(compiler=sc, validate=False)           # SelectingCompiler already validates each candidate
    out = wm.simulate("Will x exceed 0.5?")
    assert out["mechanism"] == "generic_scm" and out["forecast"]["p_event"] is not None
