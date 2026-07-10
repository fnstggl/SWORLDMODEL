"""The agent engine's constitution, enforced as tests — offline (stub LLM + fixture retrieval).

Each test pins one anti-regression clause from swm/engine/__init__.py: native answer types, loud
abstention on starved grounding, resolved-by-evidence short-circuit, LLM-reasoned decisions (no ODE
constants anywhere), scenario-specific individual predictions with no base-rate fallback, real dated
rounds, and grade-or-abstain calibration.
"""
from __future__ import annotations

import json

from swm.engine.agents import Persona, decide, slice_private_facts
from swm.engine.calibrate import GradeRegistry, fit_shrink, shrink_distribution
from swm.engine.casting import CastingDirector
from swm.engine.front_door import AgentWorldModel
from swm.engine.grounding import SceneDossier, SceneGrounder
from swm.engine.retrieval import Passage
from swm.engine.society import SocietyRollout, _dates


# ---------------------------------------------------------------- stubs
def _passages(n=8):
    return [Passage(f"Candidate A leads candidate B in poll {i} of the District 9 primary",
                    "google_news:Stub Daily", "2026-07-01") for i in range(n)]


class ScriptedLLM:
    """Answers by prompt marker — a deterministic, offline 'model'."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if "professional SUPERFORECASTER" in prompt:       # observer-panel forecaster
            return json.dumps({"base_rate": 0.4, "p": 0.6, "why": "evidence leans yes"})
        if "Classify how this question's outcome is GENERATED" in prompt:
            proc = any(w in prompt.lower() for w in ("bitcoin", "price", "index", "s&p"))
            return json.dumps({"kind": "process" if proc else "people", "why": "stub"})
        if "targeted web-search queries" in prompt or '"checklist"' in prompt:
            return json.dumps({"checklist": ["who is running", "current polls", "election date"],
                               "queries": ["district 9 primary candidates", "district 9 primary polls"]})
        if "grounding a forecasting question" in prompt:
            return json.dumps({"facts": [
                {"fact": "who is running", "detail": "A and B", "source": "google_news", "date": "2026-07-01"},
                {"fact": "current polls", "detail": "A 52, B 41", "source": "google_news", "date": "2026-07-01"},
                {"fact": "election date", "detail": "2026-08-15", "source": "google_news"}],
                "actors": ["progressive voters", "moderate voters"], "missing": [], "resolved": None})
        if "CASTING DIRECTOR" in prompt:
            return json.dumps({"process": "collective_choice",
                               "answer_space": {"type": "named_options", "options": ["A", "B"]},
                               "actors": [{"name": "progressive voters", "kind": "segment", "weight": 0.5,
                                           "role": "left flank", "n_variants": 2},
                                          {"name": "moderate voters", "kind": "segment", "weight": 0.5,
                                           "role": "center", "n_variants": 2}],
                               "resolve_by": "2026-08-15", "horizon_days": 37, "cadence_days": 14,
                               "interaction": "voters see published polls", "rationale": "a primary"})
        if '"variants"' in prompt:
            return json.dumps({"variants": [{"sketch": "engaged 34yo renter, reads local news"},
                                            {"sketch": "low-attention 61yo homeowner"}]})
        if '"probs"' in prompt:
            if '"engage"' in prompt:
                p = {"engage": 0.6, "ignore": 0.4} if "Silence" in prompt else {"engage": 0.3, "ignore": 0.7}
            elif '"yes"' in prompt:                        # binary event framing
                p = {"yes": 0.6, "no": 0.4}
            else:
                p = {"A": 0.7, "B": 0.3} if "progressive" in prompt else {"A": 0.45, "B": 0.55}
            return json.dumps({"probs": p, "statement": "leaning", "why": "fits my values"})
        if '"states"' in prompt:
            return json.dumps({"states": [{"state": "busy travel week", "weight": 0.5},
                                          {"state": "at desk, curious", "weight": 0.5}]})
        if '"decision"' in prompt:
            return json.dumps({"decision": "respond" if "at desk" in prompt else "no_response",
                               "why": "depends on my week"})
        if '"dossier"' in prompt:
            return json.dumps({"dossier": "A well-documented public investor.", "identified": True})
        if '"candidates"' in prompt:
            return json.dumps({"candidates": [{"text": "Silence, engineered.", "angle": "feature"},
                                              {"text": "Hear what you've been missing.", "angle": "emotion"}]})
        return "{}"


# ---------------------------------------------------------------- grounding: loud abstention
def test_grounding_abstains_loudly_when_starved():
    g = SceneGrounder(ScriptedLLM(), search_fn=lambda qs, k: [])          # retrieval returns NOTHING
    d = g.ground("who wins the district 9 primary?")
    assert d.abstain and "STARVED" in d.abstain_reason
    assert d.missing == d.checklist                       # nothing pretended to be grounded


def test_grounding_grounds_from_passages_with_citations():
    g = SceneGrounder(ScriptedLLM(), search_fn=lambda qs, k: _passages())
    d = g.ground("who wins the district 9 primary?")
    assert not d.abstain and len(d.facts) == 3
    assert all(f.get("source") for f in d.facts)          # every fact cited
    assert "current polls" in d.brief()


def test_resolved_by_evidence_short_circuits():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            if "grounding a forecasting question" in prompt:
                return json.dumps({"facts": [{"fact": "result", "detail": "A won", "source": "g"}],
                                   "actors": [], "missing": [],
                                   "resolved": {"answer": "A", "evidence": "A wins primary", "source": "g"}})
            return super().__call__(prompt)
    wm = AgentWorldModel(llm=LLM(), search_fn=lambda qs, k: _passages())
    res = wm.simulate("who wins the district 9 primary?")
    assert res["mechanism"] == "resolved_by_evidence"
    # 0.97, not 1.0 — a projection isn't a certification; a wrong 1.0 is a log-loss catastrophe
    assert res["distribution"] == {"A": 0.97} and "ALREADY RESOLVED" in res["headline"]


# ---------------------------------------------------------------- casting: native answer space
def test_casting_yields_named_options_and_normalized_weights():
    c = CastingDirector(ScriptedLLM()).cast("who wins?", "scene", today="2026-07-09")
    assert c.answer_space["options"] == ["A", "B"]        # NAMED options, not an abstract logit
    assert abs(sum(a.weight for a in c.actors) - 1.0) < 1e-9
    assert c.resolve_by == "2026-08-15"                   # real calendar time


# ---------------------------------------------------------------- society: reasoning, dated rounds
def test_society_native_distribution_and_dated_rounds():
    c = CastingDirector(ScriptedLLM()).cast("who wins?", "scene", today="2026-07-09")
    d = SceneGrounder(ScriptedLLM(), search_fn=lambda qs, k: _passages()).ground("who wins?")
    res = SocietyRollout(ScriptedLLM(), branches=2, max_rounds=2).run("who wins?", c, d,
                                                                      today="2026-07-09")
    assert set(res.distribution) == {"A", "B"}
    assert abs(sum(res.distribution.values()) - 1.0) < 1e-6
    assert len(res.rounds) >= 1 and all(r.startswith("2026-") for r in res.rounds)   # real dates
    assert res.audit and all("why" in a for a in res.audit)         # decisions carry REASONS
    # the aggregate is the reasoned mixture: progressives 0.7/0.3, moderates 0.45/0.55 at equal weight
    assert abs(res.distribution["A"] - 0.575) < 0.02


def test_dates_map_horizon_to_real_calendar():
    ds = _dates(30, 7, 3, today="2026-07-09")
    assert ds[-1] == "2026-08-08"                          # 30 days forward is 30 real days


def test_decide_drops_bad_parse_instead_of_inventing():
    p = Persona("x", "segment", 1.0, "d")
    assert decide(lambda _: "not json", p, "q", ["A", "B"]) is None


def test_private_slices_differ_across_personas():
    facts = [{"fact": f"f{i}", "detail": ""} for i in range(6)]
    assert slice_private_facts(facts, 0) != slice_private_facts(facts, 1)


# ---------------------------------------------------------------- individual: never the base rate
def test_individual_abstains_when_person_cannot_be_grounded():
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: [])
    import swm.engine.individual as ind
    old = ind.multi_search
    ind.multi_search = lambda qs, k=6: []                  # person retrieval starved
    try:
        res = wm.simulate("will he reply?", recipient="Unfindable Person", message="hi")
    finally:
        ind.multi_search = old
    assert res["abstain"] and "base rate" in res["abstain_reason"].lower()
    assert res["distribution"] == {}                       # NO number was emitted


def test_individual_mixture_is_scenario_specific():
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: _passages())
    import swm.engine.individual as ind
    old = ind.multi_search
    ind.multi_search = lambda qs, k=6: _passages(4)
    try:
        res = wm.simulate("will he reply?", recipient="Known Person", message="specific pitch")
    finally:
        ind.multi_search = old
    assert not res["abstain"]
    p = res["distribution"]["responds"]
    assert 0.2 < p < 0.8                                   # mixture of respond-state and busy-state
    assert len(res["detail"]["per_state"]) == 2            # the p comes FROM the latent states


# ---------------------------------------------------------------- the outcome flywheel (the moat)
def test_flywheel_log_resolve_refit_closes_the_loop(tmp_path):
    from swm.engine.calibrate import GradeRegistry
    from swm.engine.flywheel import FlywheelLog
    fw = FlywheelLog(path=str(tmp_path / "log.jsonl"))
    # 1. LOG: an overconfident stream (says 0.9/0.1, right only ~70% of the time)
    rids = []
    for i in range(20):
        rids.append(fw.log(question=f"will thing {i} happen?", question_class="society:event",
                           domain="deliberation", mechanism="panel", p=0.9 if i % 2 == 0 else 0.1,
                           as_of="2026-01-01", resolve_by="2026-02-01", ts=1_700_000_000 + i))
    assert len(fw.load()) == 20 and all(r.status == "open" for r in fw.load())
    # idempotent: re-logging the same forecast doesn't duplicate
    fw.log(question="will thing 0 happen?", question_class="society:event", domain="deliberation",
           mechanism="panel", p=0.9, as_of="2026-01-01", ts=1_700_000_000)
    assert len(fw.load()) == 20
    # 2. RESOLVE: outcomes land (70% agree with the forecast side)
    for i, rid in enumerate(rids):
        said_yes = i % 2 == 0
        correct = i % 10 < 7
        fw.record_outcome(rid, 1.0 if (said_yes == correct) else 0.0)
    assert all(r.status == "resolved" for r in fw.load())
    # 3. REFIT: the registry's temperature updates from the PROPRIETARY stream (T>1: it learned to temper)
    reg = GradeRegistry(path=str(tmp_path / "grades.json"))
    rep = fw.refit(reg, min_n=10)
    assert rep["classes"]["society:event"]["n"] == 20
    assert rep["classes"]["society:event"]["temperature"] > 1.0
    assert reg.temperature_for("society:event") > 1.0      # the live engine now reads the refit T


def test_flywheel_auto_resolve_from_evidence(tmp_path):
    from swm.engine.flywheel import FlywheelLog
    fw = FlywheelLog(path=str(tmp_path / "log.jsonl"))
    rid = fw.log(question="Will candidate A win the election?", question_class="society:event",
                 domain="deliberation", mechanism="panel", p=0.8, resolve_by="2020-01-01",
                 ts=1_500_000_000)
    out = fw.auto_resolve(
        lambda p: json.dumps({"resolved": True, "outcome": "yes", "evidence": "A won [news]"}),
        search_fn=lambda qs, k=6: _passages(4))
    assert out == {"checked": 1, "resolved": 1}
    r = {x.rid: x for x in fw.load()}[rid]
    assert r.status == "resolved" and r.outcome == 1.0 and r.resolution_source.startswith("auto:")


def test_front_door_logs_forecasts_to_flywheel(tmp_path):
    from swm.engine.flywheel import FlywheelLog
    fw = FlywheelLog(path=str(tmp_path / "log.jsonl"))
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: _passages(), branches=2, flywheel=fw)
    wm.simulate("who wins the district 9 primary?")
    recs = fw.load()
    assert len(recs) == 1 and recs[0].p is not None        # the emitted forecast is in the stream
    assert recs[0].status == "open" and recs[0].resolve_by == "2026-08-15"   # real resolution date captured


def test_partitioned_brief_keeps_standing_common_but_slices_differ():
    from swm.engine.grounding import SceneDossier
    d = SceneDossier(question="q", standing="A favored (poll +14)",
                     facts=[{"fact": f"f{i}", "detail": f"d{i}"} for i in range(8)])
    b0, b1 = d.partitioned_brief(0), d.partitioned_brief(1)
    assert "A favored" in b0 and "A favored" in b1          # deciding signal is COMMON knowledge
    assert b0 != b1                                          # but the peripheral evidence slices differ


# ---------------------------------------------------------------- calibration: grade-or-abstain
def test_ungraded_class_ships_flagged_not_confident(tmp_path):
    reg = GradeRegistry(path=str(tmp_path / "grades.json"))
    cal = reg.calibration_for("society:collective_choice")
    assert cal["grade"] == "ungraded" and cal["abstain_confident"]


def test_graded_class_carries_fitted_shrink(tmp_path):
    reg = GradeRegistry(path=str(tmp_path / "grades.json"))
    reg.record("society:collective_choice",
               backtest_report={"n": 40, "brier": 0.18, "skill": {"base_rate": 0.2}},
               preds=[0.9, 0.8, 0.7, 0.2], outcomes=[1, 1, 0, 0])
    cal = GradeRegistry(path=str(tmp_path / "grades.json")).calibration_for("society:collective_choice")
    assert cal["grade"] == "A" and not cal["abstain_confident"] and 0.4 <= cal["shrink"] <= 1.0


def test_log_linear_pool_downweights_wishy_washy_and_never_hits_0_1():
    from swm.engine.calibrate import pool_distribution
    # the key property: two decisive YES-leaners + one uninformative 0.5 → the pool should be MORE decisive
    # than the linear mean (0.70), because a 0.5 carries no information in log-odds space (fixes the
    # 'a few wishy-washy 0.5s drag a real signal to the middle' failure).
    pooled = pool_distribution([{"yes": 0.8, "no": 0.2}, {"yes": 0.8, "no": 0.2}, {"yes": 0.5, "no": 0.5}])
    assert pooled["yes"] > 0.70
    # a lone certain persona cannot force 1.0 (finite-sample smoothing via min_p)
    certain = pool_distribution([{"yes": 1.0, "no": 0.0}] * 3)
    assert certain["yes"] < 0.99 and certain["no"] > 0.0
    # a real dissenter widens it back toward the middle
    split = pool_distribution([{"yes": 0.9, "no": 0.1}, {"yes": 0.1, "no": 0.9}])
    assert 0.3 < split["yes"] < 0.7


def test_temperature_recalibration_out_of_sample():
    from swm.engine.calibrate import apply_temperature, crossfit_temperature, fit_temperature
    # an OVERCONFIDENT forecaster (says 0.95 but is right only ~60%) → T>1 should be chosen (tempering)
    preds = [0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05] * 3
    ys = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1] * 3
    assert fit_temperature(preds, ys) > 1.0
    assert apply_temperature(0.95, 2.0) < 0.95              # tempering pulls toward 0.5
    cf = crossfit_temperature(preds, ys)
    assert cf["temperature"] > 1.0 and "logloss_after" in cf


def test_shrink_tempers_toward_ignorance():
    d = shrink_distribution({"A": 0.9, "B": 0.1}, 0.5)
    assert 0.5 < d["A"] < 0.9 and abs(sum(d.values()) - 1) < 1e-9
    assert fit_shrink([0.99, 0.99], [0, 0]) < 1.0          # overconfident+wrong => shrink chosen


# ---------------------------------------------------------------- diffusion / virality class
def test_diffusion_cascade_native_output():
    from swm.engine.diffusion import DiffusionSimulator
    from swm.engine.grounding import SceneDossier
    import json as _j
    def llm(prompt):
        if '"archetypes"' in prompt:
            return _j.dumps({"archetypes": [
                {"name": "influencers", "share": 0.1, "reach_mult": 8.0, "sketch": "big accounts"},
                {"name": "fans", "share": 0.5, "reach_mult": 1.0, "sketch": "engaged fans"},
                {"name": "lurkers", "share": 0.4, "reach_mult": 0.4, "sketch": "rarely post"}]})
        if '"amplify"' in prompt:
            eager = "influencers" in prompt or "fans" in prompt or "engaged" in prompt or "big accounts" in prompt
            early = "EARLY" in prompt
            return _j.dumps({"amplify": eager and early, "sentiment": 0.5, "why": "fits my feed"})
        return "{}"
    d = SceneDossier(question="q", facts=[{"fact": "context", "detail": "x"}])
    df = DiffusionSimulator(llm, n_nodes=150, n_worlds=60, reps_per_archetype=4).simulate("BIG NEWS", d)
    assert set(df.reach) == {"p10", "p50", "p90"} and df.reach["p90"] >= df.reach["p10"]
    assert df.narrative_leaders and df.narrative_leaders[0]["archetype"] in ("influencers", "fans")
    assert df.inflection_round is not None
    # late decay was learned from sampled decisions, not assumed: p_late < p_early for eager archetypes
    eager = next(a for a in df.archetypes if a["name"] == "fans")
    assert eager["p_late"] < eager["p_early"]


def test_front_door_routes_viral_questions_to_diffusion():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            import json as _j
            if '"archetypes"' in prompt:
                return _j.dumps({"archetypes": [
                    {"name": "core", "share": 0.6, "reach_mult": 2.0, "sketch": "core audience"},
                    {"name": "casual", "share": 0.4, "reach_mult": 0.5, "sketch": "casuals"}]})
            if '"amplify"' in prompt:
                return _j.dumps({"amplify": "EARLY" in prompt, "sentiment": 0.2, "why": "novel"})
            return super().__call__(prompt)
    wm = AgentWorldModel(llm=LLM(), search_fn=lambda qs, k: _passages())
    res = wm.simulate("Will this announcement go viral?", message="We are launching X")
    assert res["mechanism"] == "grounded_agents:diffusion"
    assert "reach>0.2" in res["distribution"] and res["detail"]["narrative_leaders"]
    assert "hypothesis" in res["headline"]                 # ungraded class ships flagged


def test_grounding_extracts_relations_graph():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            if "grounding a forecasting question" in prompt:
                return json.dumps({"facts": [{"fact": "f", "detail": "d", "source": "s"}],
                                   "actors": [], "missing": [],
                                   "relations": [{"a": "AOC", "rel": "endorsed", "b": "Candidate A"}],
                                   "standing": {"favored": "Candidate A", "margin": "poll +10",
                                                "basis": "poll", "confidence": 0.8}})
            return super().__call__(prompt)
    g = SceneGrounder(LLM(), search_fn=lambda qs, k: _passages())
    d = g.ground("who wins?")
    assert d.relations == [{"a": "AOC", "rel": "endorsed", "b": "Candidate A"}]
    assert "AOC —endorsed→ Candidate A" in d.brief()       # the actor graph reaches every agent


# ---------------------------------------------------------------- the decisive ablation
def test_ablation_scoring_and_head_to_head():
    from swm.eval.ablation import score_arms
    # FULL is a better forecaster than EVIDENCE on this synthetic set (sharper AND right)
    rows = []
    for i in range(24):
        y = i % 2
        rows.append({"full": 0.85 if y else 0.15, "raw": 0.5, "evidence": 0.65 if y else 0.35,
                     "base_rate": 0.5, "parametric": 0.55 if y else 0.45, "outcome": y})
    sb = score_arms(rows)
    assert sb["arms"]["full"]["brier"] < sb["arms"]["evidence"]["brier"]        # sim beats single call
    assert sb["arms"]["evidence"]["brier"] < sb["arms"]["raw"]["brier"]          # evidence beats no-evidence
    assert sb["arms"]["full"]["direction"] == 1.0
    h = sb["head_to_head_full_vs_evidence"]
    assert h["full_better"] and h["full_minus_evidence"] < 0 and h["n_both"] == 24
    # arms that abstain are scored only on what they answered
    rows2 = [{"full": None, "evidence": 0.6, "base_rate": 0.5, "raw": 0.5, "parametric": 0.5, "outcome": 1}]
    s2 = score_arms(rows2)
    assert s2["arms"]["full"]["n"] == 0 and s2["arms"]["full"]["n_abstain"] == 1


# ---------------------------------------------------------------- front door: one mechanism
def test_front_door_society_output_is_native_and_flagged():
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: _passages(), branches=2)
    res = wm.simulate("who wins the district 9 primary?")
    assert res["mechanism"] == "grounded_agents:collective_choice"
    assert set(res["distribution"]) == {"A", "B"}          # the answer names the candidates
    assert res["calibration"]["grade"] in ("ungraded", "A", "B", "C", "F")
    assert "hypothesis" in res["headline"] or "%" in res["headline"]
    assert res["grounding"]["detail"]                      # provenance rides along


# ---------------------------------------------------------------- paradigm router (people → agents)
def test_router_sends_people_questions_to_agents():
    from swm.engine.router import ParadigmRouter
    # a classifier that WRONGLY says "process" must be overridden by the lexical people fast-path
    r = ParadigmRouter(llm=lambda p: '{"kind":"process","why":"wrong"}')
    for q in ["Who will win the NY-10 primary?", "Will Thiel reply to this email?",
              "Will the Senate confirm the nominee?", "Best headline to maximize AirPods sales?"]:
        assert r.route(q) == "agents", q


def test_router_sends_nonhuman_process_to_parametric():
    from swm.engine.router import ParadigmRouter
    r = ParadigmRouter(llm=lambda p: '{"kind":"process","why":"market price"}')
    assert r.route("Will Bitcoin be above $80,000 by year end?") == "parametric"
    # no-LLM lexical fallback still diverts an unmistakable price process, defaults people→agents otherwise
    r2 = ParadigmRouter(llm=None)
    assert r2.route("Will the S&P 500 index close above 7000?") == "parametric"
    assert r2.route("Will voters approve the referendum?") == "agents"


def test_binary_kind_separates_contests_from_deliberation():
    from swm.engine.router import ParadigmRouter
    r = ParadigmRouter(llm=None)
    assert r.binary_kind("NY Knicks vs SA Spurs, NBA final game 3") == "contest"
    assert r.binary_kind("Will Barcelona win the El Clasico?") == "contest"
    assert r.binary_kind("Will Apple release the M5 Mac mini at WWDC?") == "announcement"
    # election words must NOT be misread as contests
    assert r.binary_kind("Will the incumbent win the mayoral race?") == "deliberation"
    assert r.binary_kind("Will X beat Y in the Senate election?") == "deliberation"
    assert r.binary_kind("Will the Senate confirm the nominee?") == "deliberation"


def test_front_door_routes_contest_to_parametric():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            if "STRUCTURAL-MODEL COMPILER" in prompt or "mechanism" in prompt.lower() and "JSON" in prompt:
                return "{}"                                # parametric compile fails → base-rate fallback
            return super().__call__(prompt)
    wm = AgentWorldModel(llm=LLM(), search_fn=lambda qs, k: _passages(), route_contests=True)
    res = wm.simulate("Will Barcelona win the El Clasico?", binary=True)
    assert res["mechanism"].startswith("parametric:")      # routed OUT of the deliberation panel
    assert 0.0 < res["distribution"]["yes"] < 1.0


def test_per_domain_temperature_registry(tmp_path):
    from swm.engine.calibrate import GradeRegistry
    reg = GradeRegistry(path=str(tmp_path / "g.json"))
    reg.record("society:event", backtest_report={"n": 20, "skill_vs": {"crowd": 0.1}, "rmse": 0.4},
               preds=[0.6] * 20, outcomes=[1, 0] * 10, temperature=1.5)
    reg.record_domain_temperatures("society:event", {"tech": 3.0, "election": 1.2})
    reg2 = GradeRegistry(path=str(tmp_path / "g.json"))
    assert reg2.temperature_for("society:event", domain="tech") == 3.0       # per-domain
    assert reg2.temperature_for("society:event", domain="election") == 1.2
    assert reg2.temperature_for("society:event", domain="other") == 1.5      # falls back to class T


def test_multi_family_panel_runs_every_family():
    from swm.engine.observer_panel import LENSES, ObserverPanel
    from swm.engine.grounding import SceneDossier
    import json as _j
    calls = {"a": 0, "b": 0}
    def mk(fam):
        def fn(prompt):
            calls[fam] += 1
            return _j.dumps({"base_rate": 0.4, "p": 0.6, "why": "x"})
        return fn
    d = SceneDossier(question="Will X?", facts=[{"fact": "f", "detail": "d"}], standing="X leads",
                     standing_struct={"favored": "yes", "confidence": 0.7})
    pf = ObserverPanel(None, reps_per_lens=1, model_llms={"a": mk("a"), "b": mk("b")}).forecast("Will X?", d)
    assert calls["a"] == len(LENSES) and calls["b"] == len(LENSES)            # both families forecasted
    assert pf.families == 2 and pf.n_forecasters == 2 * len(LENSES)


def test_router_defaults_to_agents_on_failure_or_ambiguity():
    from swm.engine.router import ParadigmRouter
    r = ParadigmRouter(llm=lambda p: "not json")           # classifier failure → fail toward agents
    assert r.route("Will the coup succeed in Fooland?") == "agents"


def test_hybrid_routes_process_to_parametric_and_people_to_agents():
    # a stub parametric engine records if it was called
    class Parametric:
        called = False
        def simulate(self, q, as_of=""):
            Parametric.called = True
            return {"mechanism": "diffusion", "forecast": {"p_event": 0.4}}
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: _passages(),
                         parametric=Parametric(), branches=1)
    proc = wm.simulate("Will Bitcoin be above $80,000 by year end?")
    assert proc["engine"] == "parametric_mechanism" and Parametric.called
    Parametric.called = False
    ppl = wm.simulate("Who will win the district 9 primary?")     # people → agents, parametric untouched
    assert not Parametric.called and ppl["mechanism"].startswith(("grounded_agents", "resolved"))


# ---------------------------------------------------------------- leak-free backtest path
def test_evidence_injection_bypasses_live_retrieval():
    # a search_fn that would EXPLODE if called proves the frozen-evidence path never touches the web
    def boom(_qs, _k):
        raise AssertionError("live retrieval called during a leak-free backtest")
    g = SceneGrounder(ScriptedLLM(), search_fn=boom)
    d = g.ground("who wins?", evidence=["Candidate A leads B in the poll, 52-41",
                                        "A holds a fundraising edge per the latest filing",
                                        "the election date is 2026-08-15"])
    assert not d.abstain and d.facts                       # grounded purely from the injected evidence


def test_injected_evidence_is_gated_on_content_not_passage_count():
    # 2 injected passages (below the live-retrieval min_passages=3 floor) must NOT auto-abstain —
    # the floor is a dead-search signal; injected evidence is judged on coverage (content) only.
    g = SceneGrounder(ScriptedLLM(), search_fn=lambda qs, k: [])
    d = g.ground("who wins?", evidence=["Poll: A 52, B 41", "A leads fundraising"])
    assert not d.abstain and d.facts                       # content sufficed; count did not gate it


def test_grading_p_yes_and_domain_filter():
    from swm.eval.grade_agent_engine import is_domain, p_yes
    assert p_yes({"distribution": {"yes": 0.7, "no": 0.3}}) == 0.7
    assert p_yes({"distribution": {"no": 0.8, "yes": 0.2}}) == 0.2
    assert p_yes({"distribution": {}}) is None
    assert is_domain("Will Mikie Sherrill win the New Jersey Governor Election?")
    assert not is_domain("Will the Pirates win the 2025 World Series?")     # off-domain sports


def test_grade_pooled_math_and_registry(tmp_path):
    from swm.engine.calibrate import GradeRegistry
    from swm.eval.grade_agent_engine import grade_pooled
    # a discriminating forecaster: high on the YES cases, low on the NO cases → should beat the class rate
    rounds = [{"preds": [0.8, 0.75, 0.2, 0.15, 0.1, 0.25], "outcomes": [1, 1, 0, 0, 0, 0],
               "items": [{"q": f"q{i}"} for i in range(6)], "n_domain": 6, "n_abstained": 1}]
    reg = GradeRegistry(path=str(tmp_path / "g.json"))
    rep = grade_pooled(rounds, registry=reg, question_class="society:event")
    assert rep.n_scored == 6 and rep.n_abstained == 1
    assert rep.backtest["skill_vs"]["class_rate"] > 0        # discrimination beat the base-rate guess
    assert rep.grade_entry["grade"] in ("A", "B", "C", "F", "ungraded")
    assert GradeRegistry(path=str(tmp_path / "g.json")).grades["society:event"]["n"] == 6


def test_asof_google_news_drops_post_cutoff_items(monkeypatch):
    # Google's before:/after: is not a hard guarantee, so the code MUST drop any item dated after as_of.
    import swm.engine.retrieval as R
    rss = """<rss><channel>
      <item><title>Campaign heats up</title><pubDate>Mon, 20 Oct 2025 10:00:00 GMT</pubDate>
        <source>Pre</source></item>
      <item><title>WINNER DECLARED — leaks the outcome</title><pubDate>Wed, 05 Nov 2025 03:00:00 GMT</pubDate>
        <source>Post</source></item>
    </channel></rss>"""
    monkeypatch.setattr(R, "_get", lambda url, timeout=15: rss)
    import time
    as_of = time.mktime(time.strptime("2025-10-25", "%Y-%m-%d"))
    out = R.asof_google_news("the race", as_of)
    texts = [p.text for p in out]
    assert any("Campaign" in t for t in texts)             # pre-cutoff kept
    assert not any("WINNER" in t for t in texts)           # post-cutoff outcome DROPPED (no leak)


def test_observer_panel_base_rate_anchored_and_diverse_lenses():
    from swm.engine.observer_panel import LENSES, ObserverPanel
    from swm.engine.grounding import SceneDossier
    import json as _j
    # a stub forecaster: skeptic lens fades to base rate, insider adjusts up on a strong standing
    def llm(prompt):
        if "contrarian skeptic" in prompt:
            return _j.dumps({"base_rate": 0.05, "p": 0.08, "why": "fringe, base rate low"})
        if "domain insider" in prompt:
            return _j.dumps({"base_rate": 0.5, "p": 0.85, "why": "leads the only poll big"})
        return _j.dumps({"base_rate": 0.4, "p": 0.5, "why": "neutral"})
    d = SceneDossier(question="Will X win?", facts=[{"fact": "poll", "detail": "X +15"}],
                     standing="X leads the only poll 55-40")
    pf = ObserverPanel(llm, reps_per_lens=1).forecast("Will X win?", d)
    assert pf.n_forecasters == len(LENSES)                 # every lens forecasted (diverse ensemble)
    assert 0.0 < pf.p_event < 1.0
    assert any(a["lens"] == "skeptic" and a["p"] < 0.2 for a in pf.audit)   # skeptic anchored low
    assert any(a["lens"] == "insider" and a["p"] > 0.7 for a in pf.audit)   # insider adjusted up


def test_front_door_binary_uses_observer_panel():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            if "professional SUPERFORECASTER" in prompt:
                import json as _j
                return _j.dumps({"base_rate": 0.3, "p": 0.7, "why": "evidence favors yes"})
            return super().__call__(prompt)
    wm = AgentWorldModel(llm=LLM(), search_fn=lambda qs, k: _passages(), event_engine="panel")
    res = wm.simulate("Will the incumbent win the mayoral race?", binary=True)
    assert res["mechanism"] == "grounded_agents:observer_panel"
    assert res["distribution"] and 0.0 < res["distribution"]["yes"] < 1.0


def test_grade_vs_crowd_scores_against_the_market(tmp_path):
    from swm.engine.calibrate import GradeRegistry
    from swm.eval.grade_vs_crowd import grade_vs_crowd

    class Item:  # a minimal BacktestItem stand-in
        def __init__(self, q, y, crowd, cat="politics"):
            self.question, self.outcome, self.crowd_prob = q, y, crowd
            self.category, self.cutoff_clean, self.as_of = cat, True, 1_700_000_000

    items = [Item("Will candidate A win the mayoral election?", 1, 0.55),
             Item("Will the Senate confirm the nominee?", 0, 0.5),
             Item("Will voters approve the referendum?", 1, 0.6),
             Item("Will Bitcoin be above $80,000?", 1, 0.5)]  # process → filtered out (not people)
    wm = AgentWorldModel(llm=ScriptedLLM(), search_fn=lambda qs, k: _passages(), branches=1)
    reg = GradeRegistry(path=str(tmp_path / "g.json"))
    # inject a stub as-of search factory so no live network is needed
    rep = grade_vs_crowd(wm, items, registry=reg,
                         search_fn_factory=lambda ts: (lambda qs, k=6: _passages()), verbose=False)
    assert rep.n_domain == 3                                # the BTC (process) item was routed out
    assert rep.n_scored >= 1
    assert "skill_vs_crowd" in rep.scoreboard["overall"]    # scored against the market, not just 0.5
    assert "side_correct" in rep.direction                  # direction-accuracy metric emitted
    assert reg.grades["society:event"]["n"] == rep.n_scored


def test_front_door_artifact_mode_returns_ranked_real_texts():
    class LLM(ScriptedLLM):
        def __call__(self, prompt):
            if "CASTING DIRECTOR" in prompt:
                return json.dumps({"process": "artifact_optimization",
                                   "answer_space": {"type": "artifacts", "options": []},
                                   "actors": [{"name": "premium buyers", "kind": "segment", "weight": 1.0,
                                               "role": "audience", "n_variants": 2}],
                                   "resolve_by": "", "horizon_days": 7, "cadence_days": 7,
                                   "interaction": "", "rationale": ""})
            return super().__call__(prompt)
    wm = AgentWorldModel(llm=LLM(), search_fn=lambda qs, k: _passages())
    res = wm.simulate("what is the best landing page headline for the X100 headphones?")
    assert res["ranked_artifacts"] and res["ranked_artifacts"][0]["text"]   # ACTUAL texts, ranked
    assert res["answer_space"]["type"] == "artifacts"
    assert "p_engage" in res["ranked_artifacts"][0]


# ---------------------------------------------------------------- PART A/B: guardrails + tiered harness
def test_human_questions_never_route_to_logistic_compiler():
    """AUDIT_PART_A §4 guardrail: the banned pattern (logistic-over-invented-variables, swm/api/compiler.py)
    is reachable only via the parametric path. Pin that core HUMAN-COGNITION questions route to AGENTS and
    classify as DELIBERATION — so they hit the panel/society, never _parametric_binary or route→parametric."""
    from swm.engine.router import ParadigmRouter
    r = ParadigmRouter(llm=None)                                   # lexical, people-biased (production default)
    human = [
        "Will Zohran Mamdani win the 2025 NYC mayoral election?",
        "Will the Senate confirm the nominee before recess?",
        "Will Graham Platner be the Democratic nominee for Senate?",
        "Will Pierre Poilievre become Prime Minister of Canada?",
        "Will Congress pass the budget reconciliation bill?",
        "Will Peter Thiel reply to this cold email?",
        "Will customers adopt the new subscription tier?",
    ]
    for q in human:
        assert r.route(q) == "agents", f"{q!r} routed away from agents"
        assert r.binary_kind(q) == "deliberation", f"{q!r} misclassified as {r.binary_kind(q)}"


def test_counting_llm_meters_calls_and_tokens():
    from swm.eval.instrument import CountingLLM, Meter, evidence_hash
    m = Meter()
    llm = CountingLLM(lambda p: "x" * 40, m)
    llm("a" * 80); llm("b" * 80)
    assert m.calls == 2 and m.tokens_in == 40 and m.tokens_out == 20    # 80//4=20 in each, 40//4=10 out each
    assert m.cost_usd() >= 0.0
    assert evidence_hash("same") == evidence_hash("same") != evidence_hash("other")


def test_stratify_sample_covers_every_stratum():
    from swm.eval.tiered_ablation import stratify_sample
    items = [{"g": g, "i": i} for g in ("a", "b", "c") for i in range(10)]
    picked = stratify_sample(items, lambda it: it["g"], 0.2, seed=1)
    got = {items[i]["g"] for i in picked}
    assert got == {"a", "b", "c"}                                   # each stratum represented
    assert 3 <= len(picked) <= 12                                   # ~20% of 30, at least 1 per stratum


def test_grounded_ensemble_pools_and_filters_none():
    """B3 pools N grounded direct forecasts log-linearly; identical inputs pool to themselves; bad parses
    drop rather than poison the pool."""
    from swm.eval.tiered_ablation import _grounded_ensemble

    class Stub:
        def __init__(self): self.n = 0
        def __call__(self, prompt):
            self.n += 1
            return "" if self.n == 2 else json.dumps({"p": 0.7})    # one unparseable call drops out
    p = _grounded_ensemble(Stub(), "q", "2025-01-01", _DossierStub(), n=4)
    assert abs(p - 0.7) < 1e-6                                      # pool of 0.7's is 0.7, None ignored


class _DossierStub:
    abstain = False
    facts = []
    def brief(self): return "EVIDENCE: A leads B 52-41."


def test_report_marginals_computes_paired_stats():
    """Synthetic runs where 'full' is uniformly better than 'base_rate' → marginal must show hi_better with a
    permutation p and a bootstrap CI; an all-abstain arm yields insufficient-n, not a crash."""
    from swm.eval.tiered_ablation import report_marginals
    runs = []
    for i in range(20):
        y = i % 2
        runs.append({
            "outcome": y,
            "base_rate": {"p": 0.5, "spend": {"calls": 0, "cost_usd": 0.0, "seconds": 0.0}},
            "grounded_1shot": {"p": 0.6 if y else 0.4, "spend": {"calls": 1, "cost_usd": 0.001, "seconds": 1.0}},
            "full": {"p": 0.9 if y else 0.1, "spend": {"calls": 10, "cost_usd": 0.01, "seconds": 8.0}},
            "grounded_ens": {"p": None, "spend": {"calls": 0, "cost_usd": 0.0, "seconds": 0.0}},
        })
    rep = report_marginals(runs)
    assert rep["arms"]["full"]["brier"] < rep["arms"]["base_rate"]["brier"]
    whole = [m for m in rep["marginals"] if m["hi"] == "full" and m["lo"] == "grounded_1shot"][0]
    assert whole["hi_better"] and 0.0 <= whole["p_perm"] <= 1.0 and len(whole["ci95"]) == 2
    ens = [m for m in rep["marginals"] if m["hi"] == "grounded_ens"][0]
    assert ens.get("insufficient")                                 # all-None arm → not scored, not a crash
    assert rep["spend"]["full"]["mean_calls"] == 10.0


# ---------------------------------------------------------------- PART I: TRIBE adapter is quarantined
def test_experimental_is_quarantined_from_engine():
    """The production engine must NEVER import the experimental (unvalidated, non-commercial) modules."""
    import pathlib
    import re
    engine_dir = pathlib.Path(__file__).resolve().parent.parent / "swm" / "engine"
    for f in engine_dir.glob("*.py"):
        src = f.read_text()
        assert not re.search(r"import\s+swm\.experimental|from\s+swm\.experimental", src), \
            f"{f.name} imports the quarantined swm.experimental package"


def test_tribe_adapter_refuses_by_default_and_blocks_commercial_use():
    from swm.experimental.tribe_adapter import TribeAdapter, TribeUnavailable
    ad = TribeAdapter()                                            # disabled by default
    assert not ad.available()
    try:
        ad.features_for("Silence, engineered.")
        assert False, "disabled adapter must refuse"
    except TribeUnavailable:
        pass
    try:
        TribeAdapter(commercial_use=True)                          # license forbids commercial use
        assert False, "commercial_use=True must be rejected"
    except TribeUnavailable:
        pass


# ---------------------------------------------------------------- PART E: dataset registry is well-formed
def test_dataset_registry_is_valid_and_honest():
    """The registry must parse, cover the six capabilities, and every entry must carry the fields a
    reviewer needs to judge fitness (license, labels, randomized/causal, access, priority)."""
    import pathlib
    reg = json.loads((pathlib.Path(__file__).resolve().parent.parent / "data"
                      / "dataset_registry.json").read_text())
    ds = reg["datasets"]
    assert len(ds) >= 10
    required = {"id", "name", "capability", "source", "license", "labels", "randomized", "causal",
                "access_status", "priority", "supports"}
    ids = set()
    for d in ds:
        assert required <= set(d), f"{d.get('id')} missing {required - set(d)}"
        assert d["priority"] in {"P0", "P1", "P2", "reject"}
        ids.add(d["id"])
    # the three P0 build-now datasets are present
    assert {"upworthy", "criteo_uplift", "higgs_twitter"} <= ids
    # every capability bucket maps to real dataset ids
    for cap, refs in reg["product_capability_to_dataset"].items():
        for r in refs:
            assert r in ids, f"capability {cap} references unknown dataset {r}"


# ---------------------------------------------------------------- PART C: forward-locked multi-arm ledger
def _fake_pred(ev_hash, commit="abc123", model="deepseek-chat", as_of="2025-06-08"):
    return {
        "_meta": {"evidence_hash": ev_hash, "commit": commit, "model": model, "as_of": as_of, "abstain": False},
        "base_rate": {"p": 0.1, "spend": {"calls": 0, "cost_usd": 0.0, "seconds": 0.0}},
        "grounded_1shot": {"p": 0.3, "spend": {"calls": 1, "cost_usd": 0.001, "seconds": 1.0}},
        "full": {"p": 0.8, "spend": {"calls": 10, "cost_usd": 0.01, "seconds": 8.0}},
    }


def test_forward_ledger_versions_on_config_change_and_is_append_only(tmp_path):
    from swm.engine.forward_ledger import ForwardLedger
    led = ForwardLedger(path=str(tmp_path / "fwd.jsonl"))
    p = _fake_pred("ev1")
    v1 = led.lock_from_prediction(p, question="Will X win?", question_class="society:event",
                                  domain="deliberation", config={"panel_reps": 2})
    # re-locking the SAME config is idempotent (no new version)
    v1b = led.lock_from_prediction(p, question="Will X win?", question_class="society:event",
                                   domain="deliberation", config={"panel_reps": 2})
    assert v1 == v1b and len(led.load()) == 1
    # a changed model (or commit/evidence/config) MUST write a NEW version, never overwrite
    p2 = _fake_pred("ev1", model="deepseek-v9")
    v2 = led.lock_from_prediction(p2, question="Will X win?", question_class="society:event",
                                  domain="deliberation", config={"panel_reps": 2})
    assert v2 != v1 and len(led.load()) == 2
    # a changed config likewise
    v3 = led.lock_from_prediction(p, question="Will X win?", question_class="society:event",
                                  domain="deliberation", config={"panel_reps": 4})
    assert v3 not in (v1, v2) and len(led.load()) == 3


def test_forward_ledger_resolve_and_score_and_eligibility(tmp_path):
    from swm.engine.forward_ledger import ForwardLedger
    led = ForwardLedger(path=str(tmp_path / "fwd.jsonl"))
    qids = []
    for i in range(10):
        y = i % 2
        p = _fake_pred(f"ev{i}", as_of="2025-06-08")
        p["full"]["p"] = 0.85 if y else 0.15                       # 'full' is well-aligned
        p["grounded_1shot"]["p"] = 0.6 if y else 0.4               # weaker
        led.lock_from_prediction(p, question=f"Q{i}?", question_class="society:event",
                                 domain="deliberation", config={"panel_reps": 2})
        from swm.engine.forward_ledger import _qid
        qids.append((_qid(f"Q{i}?", "2025-06-08"), y))
    assert len(led.open_rows()) == 10
    for qid, y in qids:
        led.resolve(qid, float(y), source="test")
    assert len(led.open_rows()) == 0
    sc = led.score(min_n=4)
    assert sc["n_resolved"] == 10
    # per-class best architecture identified; 'full' should beat base_rate on Brier
    best = sc["per_class_best_architecture"]["society:event"]
    assert best["arm_brier"]["full"] < best["arm_brier"]["base_rate"]
    # resolved rows are eligible for calibration until flagged reported
    assert len(led.refit_eligible()) == 10


# ---------------------------------------------------------------- behavior-model adapter (quarantined)
def test_behavior_adapter_deepseek_backend_runs_offline_and_stubs_refuse():
    from swm.experimental.behavior_models import (BehaviorModelAdapter, BehaviorRequest,
                                                  DeepSeekBehaviorBackend, osim_backend, BackendUnavailable)

    class Stub:
        def __call__(self, prompt):
            return json.dumps({"action": "respond", "p": 0.7, "why": "curious"})
    ad = BehaviorModelAdapter(enabled=True, backends={
        "deepseek": DeepSeekBehaviorBackend(llm=Stub()),
        "osim": osim_backend(runner=None),        # no GPU/weights → must refuse (abstain), never fabricate
    })
    req = BehaviorRequest(dossier="a busy investor", scenario="cold email arrives",
                          stimulus="Quick question about your fund", allowed_actions=["respond", "no_response"])
    r = ad.decide("deepseek", req)
    assert r.action == "respond" and 0.0 <= r.p <= 1.0 and r.backend == "deepseek" and not r.abstain
    # the behavior-trained stub has no runner here → honest abstention, not a made-up choice
    r2 = ad.decide("osim", req)
    assert r2.abstain and "runner" in r2.abstain_reason.lower()


def test_behavior_adapter_disabled_by_default():
    from swm.experimental.behavior_models import BehaviorModelAdapter, BehaviorRequest, BackendUnavailable
    ad = BehaviorModelAdapter()                    # disabled
    assert not ad.available()
    try:
        ad.decide("deepseek", BehaviorRequest(dossier="x", scenario="y"))
        assert False, "disabled adapter must refuse"
    except BackendUnavailable:
        pass


# ---------------------------------------------------------------- labeled behavior-eval harnesses (offline)
def test_behavior_eval_num_parse_and_wasserstein():
    from swm.eval.behavior_eval import _num, _wasserstein1
    assert _num("[33]") == 33.0 and _num("I would open 40 boxes") == 40.0 and _num("nope") is None
    assert _wasserstein1([1, 2, 3], [1, 2, 3]) == 0.0
    assert _wasserstein1([10, 10, 10], [0, 0, 0]) > 5    # far distributions → large distance


def test_upworthy_ranking_scorer():
    from swm.eval.response_datasets import score_headline_ranking
    tests = [
        {"variants": [{"headline": "A", "ctr": 0.10}, {"headline": "B", "ctr": 0.05}], "winner_headline": "A"},
        {"variants": [{"headline": "C", "ctr": 0.02}, {"headline": "D", "ctr": 0.08}], "winner_headline": "D"},
    ]
    perfect = score_headline_ranking(tests, lambda hs: sorted(hs, key={"A": 0, "B": 1, "C": 1, "D": 0}.get))
    assert perfect["precision_at_1"] == 1.0 and perfect["pairwise_accuracy"] == 1.0
    worst = score_headline_ranking(tests, lambda hs: list(reversed(
        sorted(hs, key={"A": 0, "B": 1, "C": 1, "D": 0}.get))))
    assert worst["precision_at_1"] == 0.0


def test_enron_time_forward_split_is_leak_free():
    from swm.eval.response_datasets import time_forward_split
    recs = [{"date_ts": t} for t in [5, 1, 4, 2, 3]]
    tr, te = time_forward_split(recs, test_frac=0.4)
    assert max(r["date_ts"] for r in tr) < min(r["date_ts"] for r in te)   # no future in train
