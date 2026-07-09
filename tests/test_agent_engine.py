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
    assert res["distribution"] == {"A": 1.0} and "ALREADY RESOLVED" in res["headline"]


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
