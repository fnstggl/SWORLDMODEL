"""Regression suite from the failed Thiel output (the message that passed every old gate and was
still a bad cold email). Each test encodes one diagnosed failure so it can never silently return:

  FAILED OUTPUT: "Peter, treating data center power as a static budget ignores that dynamic
  scheduling against grid forecasts cut GPU-hours by 84% in our simulated replay of public
  production traces. Which assumption in that claim is wrong?"

Diagnoses: (1) no identity — starts mid-conversation; (2) extraordinary claim reads implausible;
(3) the ask demands unpaid diligence; (4) adversarial debate-bait framing from a stranger;
(5) no explicit next step; (6) additive scorer let maxed levers buy back failed gates; (7) 'any
reply' objective counts an irritated correction as success; (8) caricature situational levers
('challenge him') maxed by the optimizer.
"""
from __future__ import annotations

from swm.decision.compositional_search import encode_text_to_strategy
from swm.decision.llm_moves import SenderBrief
from swm.decision.mc_evaluation import mc_evaluate_funnel
from swm.decision.outreach_contract import plain_baseline_draft, validate
from swm.decision.response_funnel import FunnelScorer, funnel_scorer_from_recipient
from swm.decision.situational_levers import generate_levers

FAILED_OUTPUT = ("Peter, treating data center power as a static budget ignores that dynamic "
                 "scheduling against grid forecasts cut GPU-hours by 84% in our simulated replay "
                 "of public production traces. Which assumption in that claim is wrong?")

BRIEF = SenderBrief(
    sender="Beckett",
    thesis="AI infrastructure has a planning problem disguised as a power problem: schedulers "
           "optimize the next placement, no system chooses the fleet's best trajectory",
    ask="permission to send the one-page technical memo",
    facts=["17 years old, starting Princeton in the fall",
           "building Aurelius (runaurelius.com), AI infrastructure",
           "+724% SLA-safe goodput per dollar vs the production scheduler in simulated replay of "
           "~1.5M requests of public production traces",
           "-84% GPU-hours in the same replay",
           "working with a small batch of infrastructure operators in read-only shadow mode"])

THIEL = {"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9, "openness_to_outreach": 0.9,
         "attention_availability": 0.4, "platform_response_norm": 0.3, "relationship_strength": 0.0}


# ---------------------------------------------------------------- contract (content, not style)
def test_contract_rejects_the_failed_output():
    v = validate(FAILED_OUTPUT, BRIEF)
    assert not v.ok
    assert any("identity" in m for m in v.missing), v.missing
    assert any("next_step" in m for m in v.missing) or \
        any("diligence" in f for f in v.flags), (v.missing, v.flags)


def test_contract_flags_diligence_bait_and_accepts_permission_ask():
    v = validate(FAILED_OUTPUT, BRIEF)
    assert any("diligence_bait" in f for f in v.flags)
    good = ("Peter, I'm Beckett, 17, building Aurelius, an AI-infrastructure scheduler. "
            "Schedulers optimize the next placement; nothing plans the fleet's trajectory. "
            "In replays of public production traces it beat a production-style scheduler while "
            "cutting GPU-hours 84%. May I send you the one-page memo? Beckett")
    v2 = validate(good, BRIEF)
    assert v2.ok, (v2.missing, v2.flags)
    assert not any("diligence" in f for f in v2.flags)


def test_contract_flags_unanchored_extraordinary_claim():
    bare = ("Peter, I'm Beckett, building Aurelius. We cut GPU-hours by 84%. "
            "May I send you the one-page memo? Beckett")
    v = validate(bare, BRIEF)
    assert any("unanchored_claim" in f for f in v.flags), v.flags
    anchored = ("Peter, I'm Beckett, building Aurelius. In replays of public production traces "
                "we cut GPU-hours by 84% against the production scheduler. "
                "May I send you the one-page memo? Beckett")
    v2 = validate(anchored, BRIEF)
    assert not any("unanchored_claim" in f for f in v2.flags), v2.flags


def test_plain_baseline_draft_satisfies_contract():
    text = plain_baseline_draft(BRIEF, "Peter Thiel")
    v = validate(text, BRIEF)
    assert v.ok, (text, v.missing)


# ---------------------------------------------------------------- funnel (conjunctive, valenced)
def test_funnel_is_conjunctive_one_failed_gate_kills():
    """A maxed 'impressive' vector with zero identity must not beat a balanced vector — the additive
    model's failure mode. Under the funnel, the failed understand-gate multiplies through."""
    sc = FunnelScorer(recipient=THIEL, base_responsiveness=0.3, n_weight_samples=80, seed=0)
    impressive_no_identity = {v: 1.0 for v in sc.optimizable_vars()}
    impressive_no_identity.update({"identity_legibility": 0.0, "cognitive_effort": 0.0,
                                   "adversarial_framing": 0.0, "pushiness": 0.0,
                                   "convenience_selling": 0.0, "credential_signaling": 0.0})
    balanced = dict(impressive_no_identity)
    balanced["identity_legibility"] = 1.0
    assert sc.mean(balanced) > sc.mean(impressive_no_identity) * 1.5


def test_funnel_penalizes_adversarial_framing_for_strangers():
    sc = FunnelScorer(recipient=THIEL, base_responsiveness=0.3, n_weight_samples=80, seed=1)
    base = {v: 0.7 for v in sc.optimizable_vars()}
    base.update({"pushiness": 0.0, "convenience_selling": 0.0, "credential_signaling": 0.0,
                 "adversarial_framing": 0.0, "cognitive_effort": 0.2})
    bait = dict(base)
    bait["adversarial_framing"] = 1.0
    bait["cognitive_effort"] = 0.9
    assert sc.mean(base) > sc.mean(bait)
    # and the VALENCE: debate bait raises the negative-reply probability
    d_base, d_bait = sc.score_dist(base), sc.score_dist(bait)
    assert d_bait.mean_neg > d_base.mean_neg


def test_funnel_rewards_believability_over_bare_magnitude():
    sc = FunnelScorer(recipient=THIEL, base_responsiveness=0.3, n_weight_samples=80, seed=2)
    anchored = {v: 0.7 for v in sc.optimizable_vars()}
    anchored.update({"pushiness": 0.0, "adversarial_framing": 0.0, "convenience_selling": 0.0,
                     "credential_signaling": 0.0, "claim_believability": 0.9})
    bare = dict(anchored)
    bare["claim_believability"] = 0.15
    assert sc.mean(anchored) > sc.mean(bare)


def test_funnel_ranks_plain_baseline_above_failed_output():
    """THE regression: under the corrected objective (lexical encoding, deterministic), the plain
    human-register draft must beat the failed 'which assumption is wrong?' output."""
    plain = plain_baseline_draft(BRIEF, "Peter Thiel")
    e_failed = encode_text_to_strategy(FAILED_OUTPUT)
    e_plain = encode_text_to_strategy(plain)
    r_failed = mc_evaluate_funnel(THIEL, 0.2, e_failed, n_samples=200, seed=3)
    r_plain = mc_evaluate_funnel(THIEL, 0.2, e_plain, n_samples=200, seed=3)
    assert r_plain.objective > r_failed.objective, (r_plain.objective, r_failed.objective)
    # the stage trace must DIAGNOSE the failure: the failed output loses on understand and/or easy
    assert r_failed.stage_trace["understand"] < r_plain.stage_trace["understand"]
    assert r_failed.stage_trace["easy"] < r_plain.stage_trace["easy"]


def test_valenced_objective_penalizes_negative_replies():
    sc = FunnelScorer(recipient=THIEL, base_responsiveness=0.3, n_weight_samples=60, seed=4)
    strat = {v: 0.6 for v in sc.optimizable_vars()}
    strat.update({"adversarial_framing": 1.0, "pushiness": 0.8})
    d = sc.score_dist(strat)
    assert d.mean - 0.25 * d.mean_neg < d.mean       # λ > 0: irritated replies subtract


# ---------------------------------------------------------------- caricature guard
def test_caricature_guard_clamps_combat_levers():
    fake_llm = lambda prompt, **kw: (
        '[{"name": "intellectual_combat_invitation", "description": "challenge him to prove the '
        'thesis wrong — he loves intellectual combat", "elasticity": 2.5, "confidence": 0.9},'
        '{"name": "energy_thesis_relevance", "description": "connect to his stated interest in '
        'energy economics", "elasticity": 1.2, "confidence": 0.6}]')
    levers = generate_levers(fake_llm, "Peter Thiel", THIEL)
    by_name = {lv.name: lv for lv in levers}
    assert by_name["intellectual_combat_invitation"].elasticity_mean <= 0.0, \
        "combat-flavored levers may only penalize, never reward"
    assert 0 < by_name["energy_thesis_relevance"].elasticity_mean < 1.2, \
        "non-caricature levers survive but shrink toward zero by confidence"


# ---------------------------------------------------------------- L1 under the funnel
def test_l1_on_funnel_wants_identity_and_next_step_not_adversarial():
    from swm.decision.message_optimizer import optimize_strategy
    sc = funnel_scorer_from_recipient(THIEL, 0.2, seed=0)
    spec = optimize_strategy(sc, q=0.2, restarts=8, seed=0)
    s = spec.strategy
    assert s["identity_legibility"] >= 0.8
    assert s["next_step_clarity"] >= 0.8
    assert s["claim_believability"] >= 0.8
    assert s["adversarial_framing"] <= 0.2
    assert s["cognitive_effort"] <= 0.2
    assert s["credential_signaling"] <= 0.2          # the sign-flip survives the new objective
