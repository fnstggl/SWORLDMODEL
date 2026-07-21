"""Phase 10 — production institutional world modeling: acceptance tests (Part 28).

Covers the five planes: evidence + as-of versioning, structural (authority/stage/threshold), execution
(WorldState → StateDelta, invalid-action blocking, terminal effect), selection by causal need, and the
adversarial institutional distinctions (quorum vs threshold, majority-of-all vs present, advisory vs
decision authority, formal vs informal). Referenced as `test_ref` by the Phase-10 families/templates.
"""
import random

import pytest

from swm.world_model_v2.events import Event
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.state import Entity, SimulationClock, WorldState, parse_time
from swm.world_model_v2.institutions_v2 import families as F
from swm.world_model_v2.institutions_v2.authority import AuthorityGraph, InformationBoundary
from swm.world_model_v2.institutions_v2.decisions import (ThresholdSpec, apply_veto_and_override,
                                                          evaluate_decision)
from swm.world_model_v2.institutions_v2.evidence import (active_rules, leakage_audit,
                                                         validate_template_rules)
from swm.world_model_v2.institutions_v2.operators import InstitutionOperator, InstitutionRuntime
from swm.world_model_v2.institutions_v2.procedure import Matter, ResourceQueue, StageEngine
from swm.world_model_v2.institutions_v2.record import AuthorityEdge, InstitutionInstance, Stage
from swm.world_model_v2.institutions_v2.store import load_store
from swm.world_model_v2.institutions_v2.compile import select_institution

T0 = parse_time("2021-06-01T00:00:00Z")
ELIG100 = [f"s{i}" for i in range(100)]


# ---------------- decisions: quorum / thresholds / veto (adversarial) ----------------
def test_quorum_is_strict_majority_not_half():
    spec = ThresholdSpec("simple_majority", 0.5, base="present", quorum_fraction=0.5)
    r50 = evaluate_decision(spec, {f"s{i}": "yes" for i in range(50)}, eligible=ELIG100)
    r51 = evaluate_decision(spec, {f"s{i}": "yes" for i in range(51)}, eligible=ELIG100)
    assert not r50.quorum_met and r51.quorum_met      # 51 of 100, never 50


def test_majority_of_all_vs_majority_of_present():
    votes = {**{f"s{i}": "yes" for i in range(40)}, **{f"s{40 + i}": "no" for i in range(30)}}  # 30 absent
    present = evaluate_decision(ThresholdSpec("simple_majority", 0.5, base="present"), votes, eligible=ELIG100)
    allm = evaluate_decision(ThresholdSpec("absolute_majority", 0.5, base="all_members"), votes, eligible=ELIG100)
    assert present.passed and not allm.passed         # 40>30 present, but 40 < 51 of all members


def test_abstention_is_present_absence_is_not():
    # 51 vote (26 yes, 25 abstain) → quorum met (abstain counts present); yes 26 > no 0 present-base
    votes = {**{f"s{i}": "yes" for i in range(26)}, **{f"s{26 + i}": "abstain" for i in range(25)}}
    r = evaluate_decision(ThresholdSpec("simple_majority", 0.5, base="present"), votes, eligible=ELIG100)
    assert r.quorum_met and r.abstain == 25 and r.passed


def test_supermajority_and_veto_override():
    ov = ThresholdSpec("supermajority", 2 / 3, base="present")
    passed = evaluate_decision(ThresholdSpec("simple_majority", 0.5, base="present"),
                               {f"s{i}": ("yes" if i < 55 else "no") for i in range(100)}, eligible=ELIG100)
    res = apply_veto_and_override(passed, vetoed=True, override_spec=ov,
                                  override_votes={f"s{i}": ("yes" if i < 60 else "no") for i in range(100)},
                                  eligible=ELIG100)
    assert res.vetoed and not res.overridden          # 60/100 < 2/3 → veto sustained
    passed2 = evaluate_decision(ThresholdSpec("simple_majority", 0.5, base="present"),
                                {f"s{i}": "yes" for i in range(55)}, eligible=ELIG100)
    res2 = apply_veto_and_override(passed2, vetoed=True, override_spec=ov,
                                   override_votes={f"s{i}": ("yes" if i < 70 else "no") for i in range(100)},
                                   eligible=ELIG100)
    assert res2.overridden                             # 70/100 ≥ 2/3 → overridden


def test_recusal_changes_the_base():
    # 3 of 5 directors, one recused → eligible base is 4; 2 yes of 4 present is not a majority
    votes = {"d1": "yes", "d2": "yes", "d3": "no", "d4": "no"}
    r = evaluate_decision(ThresholdSpec("simple_majority", 0.5, base="present"),
                          votes, eligible=["d1", "d2", "d3", "d4", "d5"], recused={"d5"})
    assert r.eligible == 4 and not r.passed            # 2 yes vs 2 no → not a majority of present


# ---------------- authority + information ----------------
def test_unauthorized_action_is_identified():
    ag = AuthorityGraph(edges=[AuthorityEdge("senator", "final_decision", subject_matter=["senate_vote"]),
                               AuthorityEdge("lobbyist", "advise")])
    inst = InstitutionInstance("s", "t", "1", "2021-01-01",
                               actor_bindings={"sen": "senator", "lob": "lobbyist"})
    ok, _ = ag.authorize(inst, {"actor": "sen", "type": "vote", "subject": "senate_vote",
                                "required_authority": "final_decision"})
    bad, reason = ag.authorize(inst, {"actor": "lob", "type": "vote", "subject": "senate_vote",
                                      "required_authority": "final_decision"})
    assert ok and not bad and "advis" in reason.lower() or "lacks" in reason.lower()


def test_information_boundary_blocks_sealed():
    ib = InformationBoundary(rights={"sealed": ["judge"]})
    obs = {"public_fact": {"info_class": "public", "v": 1}, "sealed_filing": {"info_class": "sealed", "v": 2}}
    party_view = ib.filter_observations("party", obs)
    judge_view = ib.filter_observations("judge", obs)
    assert "sealed_filing" not in party_view and "sealed_filing" in judge_view


# ---------------- stage engine + queues ----------------
def test_stage_engine_transitions_and_acyclic():
    stages = [Stage("a", permitted_actions=["x"], next_stages={"passed": "b"}),
              Stage("b", permitted_actions=["y"], next_stages={"passed": "c"}),
              Stage("c", terminal=True)]
    eng = StageEngine.from_stages(stages)
    m = Matter("m1", "bill", stage="a")
    nxt, term = eng.advance(m, "passed")
    assert nxt == "b" and not term
    assert eng.validate_acyclic() == []


def test_queue_capacity_delays_completion():
    q = ResourceQueue("q", capacity_per_period=1.0, discipline="fifo")
    for i in range(5):
        q.enqueue(f"m{i}", period=i)
    res = F.queue_capacity_service(q, periods=10, target_matter="m3")
    assert res["completed_at_period"] == 3            # served one per period, m3 is 4th → period 3
    assert res["estimated_wait_periods"] == 3


# ---------------- as-of versioning + leakage ----------------
def test_as_of_versioning_and_leakage():
    s = load_store(reload=True)
    tpl = s.templates["us_congress_legislative"]
    assert {r.rule_id for r in active_rules(tpl, "1800-01-01")} >= {"passage", "veto_override"}
    la = leakage_audit(tpl, "2021-01-01", outcome_events=[{"id": "future", "date": "2050-01-01"}])
    assert la["clean"] and "future" in la["future_outcomes_excluded"]
    assert validate_template_rules(tpl) == {}          # all rules pass deterministic validation


# ---------------- families ----------------
def test_hierarchical_chain_stops_on_reject():
    r = F.hierarchical_approval(["mgr", "finance", "exec"], {"mgr": "approve", "finance": "reject"})
    assert r["outcome"] == "rejected" and r["stopped_at"] == "finance"
    ok = F.hierarchical_approval(["mgr", "finance"], {"mgr": "approve", "finance": "approve"})
    assert ok["outcome"] == "approved"


def test_legislative_bicameral_veto_override():
    specs = {"house": ThresholdSpec("simple_majority", 0.5, base="present"),
             "senate": ThresholdSpec("simple_majority", 0.5, base="present")}
    votes = {"house": {f"h{i}": ("yes" if i < 60 else "no") for i in range(100)},
             "senate": {f"s{i}": ("yes" if i < 55 else "no") for i in range(100)}}
    elig = {"house": [f"h{i}" for i in range(100)], "senate": ELIG100}
    r = F.legislative_process(chamber_specs=specs, chamber_votes=votes, chamber_eligible=elig, vetoed=False)
    assert r["outcome"] == "enacted"
    ov = {"house": ThresholdSpec("supermajority", 2 / 3, base="present"),
          "senate": ThresholdSpec("supermajority", 2 / 3, base="present")}
    r2 = F.legislative_process(chamber_specs=specs, chamber_votes=votes, chamber_eligible=elig, vetoed=True,
                               override_spec=ov, override_votes=votes)
    assert r2["outcome"] == "vetoed_sustained"         # 55-60% < 2/3 → override fails


def test_court_appeal_reverses():
    r = F.adjudicative_court(decision="granted", appealed=True, appellate_decision="reverse")
    assert r["outcome"] == "denied" and r["final"]


# ---------------- execution through WorldState → StateDelta ----------------
def _world(actors):
    w = WorldState(world_id="p10", branch_id="root", clock=SimulationClock(now=T0, as_of=T0),
                   network=RelationGraph(), information=InformationLedger())
    for a in actors:
        w.entities[a] = Entity(identity=a)
    return w


def _runtime(store):
    tpl = store.templates["us_congress_legislative"]
    inst = InstitutionInstance("s1", tpl.template_id, tpl.version, "2021-06-01", current_stage="floor_first",
                               actor_bindings={"chair": "senator", "lob": "representative"})
    rt = InstitutionRuntime(template=tpl, instance=inst, as_of="2021-06-01")
    rt.thresholds["passage"] = ThresholdSpec("simple_majority", 0.5, base="present")
    return rt


def _fire(w, rt, action, **kw):
    op = InstitutionOperator()
    ev = Event(ts=w.clock.now, etype="institutional_action",
               payload={"institution": rt, "action": action, **kw})
    w.clock.advance_to(ev.ts)
    return op.run(w, ev, random.Random(0))[0]


def test_invalid_action_blocked_mutates_nothing():
    s = load_store(reload=True)
    w, rt = _world(["chair", "lob"]), _runtime(s)
    d = _fire(w, rt, {"actor": "lob", "type": "vote", "subject": "senate_vote",
                      "required_authority": "final_decision"})
    assert "blocked_invalid_action" in d.reason_codes and not d.changes   # blocked → no mutation


def test_execution_emits_statedelta_and_moves_terminal():
    s = load_store(reload=True)
    w, rt = _world(["chair"]), _runtime(s)
    d = _fire(w, rt, {"actor": "chair", "type": "vote", "subject": "senate_vote",
                      "required_authority": "final_decision"},
              decision={"decision_id": "passage", "votes": {f"s{i}": ("yes" if i < 55 else "no") for i in range(100)},
                        "eligible": ELIG100}, outcome_var="enacted")
    assert d.changes and "enacted" in w.quantities and w.quantities["enacted"].value == "passed"
    # counterfactual: raise the threshold → same votes now FAIL (institution materially affects terminal)
    w2, rt2 = _world(["chair"]), _runtime(s)
    rt2.thresholds["passage"] = ThresholdSpec("supermajority", 2 / 3, base="present")
    d2 = _fire(w2, rt2, {"actor": "chair", "type": "vote", "subject": "senate_vote",
                         "required_authority": "final_decision"},
               decision={"decision_id": "passage", "votes": {f"s{i}": ("yes" if i < 55 else "no") for i in range(100)},
                         "eligible": ELIG100}, outcome_var="enacted")
    assert w2.quantities["enacted"].value == "failed"


# ---------------- compiler selection by causal need + gates ----------------
def test_selection_by_causal_need_and_adversarial():
    s = load_store(reload=True)
    leg = select_institution(s, "evaluate_quorum_and_threshold", {"jurisdiction": "US-federal"},
                             as_of="2020-01-01", jurisdiction="US-federal")
    assert leg.template_id == "us_congress_legislative" and leg.tier == 1
    board = select_institution(s, "issue_formal_decision", {"jurisdiction": "US-DE"},
                               as_of="2020-01-01", jurisdiction="US-DE")
    assert board.template_id == "delaware_board_default"
    # a legislature must NOT be selected for a corporate approval chain
    chain = select_institution(s, "process_required_approval_chain", {"jurisdiction": "US-DE"},
                               as_of="2020-01-01", jurisdiction="US-DE")
    assert chain.template_id != "us_congress_legislative"


def test_production_gate_requires_evidence_and_replay():
    s = load_store(reload=True)
    assert s.templates["us_congress_legislative"].status == "production_eligible"
    # a template with no verified evidence cannot reach evidence_encoded
    from swm.world_model_v2.institutions_v2.record import InstitutionTemplate
    bare = InstitutionTemplate("bare", "collective_vote_body", "1.0.0", "Bare", "nowhere")
    assert s.template_blockers(bare, "evidence_encoded")


def test_formal_vs_informal_scotus_rule_of_four():
    s = load_store(reload=True)
    cert = s.templates["scotus_certiorari"]
    assert any(not p["formal"] for p in cert.informal_practice)   # rule of four is a CUSTOM, not formal law


# ================= Phase 10 CONTINUATION: competing-model execution, extraction, 2nd category, prediction ===
import os                                                          # noqa: E402
import json as _json                                              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- competing rule-model EXECUTION + live Phase-3 posterior (priorities #1, #5) ----------------
def test_competing_rule_models_execute_separately_and_diverge():
    from swm.world_model_v2.institutions_v2.particles import RuleHypothesis, execute_competing_models
    # veto-override interpretation dispute: 2/3 of PRESENT vs 2/3 of ALL MEMBERS, 66 yes of 96 present (100 total)
    votes = {**{f"s{i}": "yes" for i in range(66)}, **{f"s{66 + i}": "no" for i in range(30)}}
    h_present = RuleHypothesis("two_thirds_present", weight=0.6,
                               threshold=ThresholdSpec("supermajority", 2 / 3, base="present"))
    h_all = RuleHypothesis("two_thirds_all_members", weight=0.4,
                           threshold=ThresholdSpec("supermajority", 2 / 3, base="all_members"))
    out = execute_competing_models([h_present, h_all], votes, base_eligible=ELIG100)
    # 66/96 present ≥ 2/3 → pass ; 66/100 all-members < 2/3 → fail : the interpretation is OUTCOME-DETERMINING
    assert out["divergence"]["disagree"] and out["divergence"]["interpretation_matters"]
    assert len(out["particles"]) == 2                    # incompatible rules kept SEPARATE (never averaged)
    assert 0.0 < out["p_pass_weighted"] < 1.0            # weighted terminal distribution, not collapsed
    assert abs(out["p_pass_weighted"] - 0.6) < 1e-6      # only the 2/3-present model (w=0.6) passes


def test_hypothesis_posterior_is_real_phase3_and_normalised():
    from swm.world_model_v2.institutions_v2.particles import institutional_hypothesis_posterior
    w, res = institutional_hypothesis_posterior(
        ["interp_a", "interp_b"], [1.0, 1.0],
        [{"counts": {"interp_a": 7, "interp_b": 2}, "reliability": 1.0, "source": "authorities"}])
    assert abs(sum(w.values()) - 1.0) < 1e-6             # a Dirichlet posterior over the hypothesis simplex
    assert w["interp_a"] > w["interp_b"]                 # more supporting sources → more posterior weight
    assert res.diagnostics["posterior_family"] == "dirichlet_conjugate"


# ---------------- automatic evidence-backed rule extraction (priority #4), offline-deterministic ----------------
def test_extraction_grounding_rejects_ungrounded_span():
    from swm.world_model_v2.institutions_v2.extract import _grounded
    src = "A Majority of each House shall constitute a Quorum to do Business."
    assert _grounded("a majority of each house shall constitute a quorum to do business", src)
    assert not _grounded("two thirds of the members present shall be required to expel", src)   # fabricated


def test_deterministic_extractor_grounds_and_validates_quorum():
    from swm.world_model_v2.institutions_v2.extract import extract_rules
    src = ("Section 5: a Majority of each House shall constitute a Quorum to do Business. If after such "
           "Reconsideration two thirds of that House shall agree to pass the Bill, it shall become a Law.")
    saved = os.environ.pop("DEEPSEEK_API_KEY", None)     # force the offline deterministic path
    try:
        res = extract_rules(src, source_id="art1_test")
    finally:
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved
    assert res["source_tag"] == "deterministic_fallback"
    kinds = {r.kind for r in res["rules"]}
    assert "quorum" in kinds                             # grounded + deterministically validated
    assert all(r.evidence_id == "art1_test" for r in res["rules"])   # every accepted rule is source-tagged
    assert all(not r.verified for r in res["rules"])     # auto-extracted rules are NOT marked verified


# ---------------- second REAL institution category (court) with a non-voting dimension (priority #3) ---------
def test_second_institution_category_is_court_and_replayed():
    s = load_store(reload=True)
    assert "scotus_merits" in s.templates
    tpl = s.templates["scotus_merits"]
    assert tpl.family_id == "adjudicative_court"
    assert tpl.status == "cross_institution_tested"      # real SCDB replay promoted it (2nd category)
    # the replay recorded BOTH a decision dimension AND a non-voting (timing/deadline) dimension
    hr = [v for v in tpl.validation if v.get("kind") == "historical_replay"]
    assert hr and "timing_within_deadline" in hr[0]


def test_court_replay_timing_helpers_offline():
    from experiments.wmv2_phase10_court_replay import _ymd, _days
    assert _ymd("6/28/2019") == (2019, 6, 28)
    assert _days((2019, 1, 1), (2019, 1, 11)) == 10
    assert _days((2020, 6, 15), (2020, 6, 30)) >= 0      # within the June-30 term deadline


def test_third_category_direct_democracy_double_majority():
    # the direct_democracy family's CONJUNCTIVE double majority is genuinely distinct from a single vote:
    # a measure can win the People yet FAIL on the sub-units (Swiss Ständemehr)
    people = {f"v{i}": ("yes" if i < 55 else "no") for i in range(100)}
    cantons = {**{f"c{i}": "no" for i in range(14)}, **{f"c{i}": "yes" for i in range(14, 26)}}
    r = F.direct_democracy(people_spec=ThresholdSpec("simple_majority", 0.5, base="present"),
                           people_votes=people, people_eligible=[f"v{i}" for i in range(100)],
                           requires_double_majority=True, subunit_results=cantons)
    assert r["people_majority"] and not r["subunit_majority"] and r["outcome"] == "rejected"
    # an OPTIONAL referendum (single majority) with the same popular vote passes
    r2 = F.direct_democracy(people_spec=ThresholdSpec("simple_majority", 0.5, base="present"),
                            people_votes=people, people_eligible=[f"v{i}" for i in range(100)])
    assert r2["outcome"] == "accepted"


def test_third_category_replayed_and_promoted():
    s = load_store(reload=True)
    assert "swiss_federal_referendum" in s.templates
    tpl = s.templates["swiss_federal_referendum"]
    assert tpl.family_id == "direct_democracy" and tpl.status == "cross_institution_tested"
    # THREE institution categories now carry a real, passing historical replay — TWO of them non-Congress
    replayed = {t.family_id for t in s.templates.values()
                if any(v.get("kind") == "historical_replay" and v.get("passed") for v in t.validation)}
    assert {"legislative_process", "adjudicative_court", "direct_democracy"} <= replayed


# ---------------- predictive path vs procedural reconstruction: metric separation (priorities #2, #6) --------
def test_predict_scoring_and_institution_specs_offline():
    from experiments.wmv2_phase10_predict import _brier, _logloss, _spec_for, _naive_partyline_pass
    assert _brier([1.0, 0.0], [True, False]) < 1e-9      # perfect probabilistic forecast → 0
    assert abs(_brier([0.5, 0.5], [True, False]) - 0.25) < 1e-9
    assert _logloss([0.9], [True]) < _logloss([0.1], [True])          # confident-correct beats confident-wrong
    assert _spec_for("cloture_legislation_3_5")[0].fraction == 0.6    # institution rule by matter type
    assert _spec_for("treaty_2_3")[0].fraction > 0.66
    assert _naive_partyline_pass("majority", 51)                      # 51-seat majority clears simple majority
    assert not _naive_partyline_pass("cloture_legislation_3_5", 51)   # but NOT a 3/5 legislation cloture


def test_committed_phase10_artifacts_separate_reconstruction_from_prediction():
    """Regression guard on the committed machine-readable artifacts: the honest metric separation and the
    court replay's real scale must persist (no silent drift toward inflated 'prediction' numbers)."""
    court = _json.load(open(os.path.join(_ROOT, "experiments/results/phase10/wmv2_phase10_court_replay.json")))["replay"]
    assert court["decision_dimension"]["majority_rule_reconstructs_decision"] > 0.95
    assert court["timing_dimension_non_voting"]["n_timed"] > 100      # real cases, non-voting dimension
    pred = _json.load(open(os.path.join(_ROOT, "experiments/results/phase10/wmv2_phase10_predict.json")))["predictive_path"]
    assert pred["design"]["out_of_sample"] and pred["design"]["leakage_safe"]
    # forward prediction is HONESTLY weaker than procedural reconstruction, but M2 beats the base-rate Brier
    assert pred["M2_phase3_matter_type_propensity"]["brier"] < pred["baselines"]["predict_base_rate_brier"]
    assert pred["statedelta_wiring_proof"]["routed_through_operator"]  # the chain runs through the real engine
