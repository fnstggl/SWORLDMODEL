"""Universal actor-cognition boundary — offline acceptance tests (no network).

Covers: typed interpretation (parse/clamp/abstain), the fitted action policy (anchor reduction +
learnability), typed action distributions (normalization + interpretation-shaping), temporal processes
(bounded, mean-reverting, workload-coupled), the structured Enron world at every C-level, and the
UNIVERSALITY acceptance: the identical schema + policy machinery runs on a NON-messaging domain
(negotiation) with zero email-specific code."""
import json
import math
import random

from swm.world_model_v2.actor_cognition import (ACTION_PACKS, FEATURE_DIMS, INTENTS, ActionPolicy,
                                                Interpretation, action_distribution, attention_transition,
                                                fit_action_policy, hidden_state_latents, interpret,
                                                relationship_modulator, relationship_strength,
                                                relationship_transition)

STUB_JSON = json.dumps({"intent": "request_action", "urgency": 0.8, "obligation": 0.7,
                        "task_ownership": 1.0, "effort_required": 0.3, "relevance_to_goals": 0.9,
                        "risk_of_inaction": 0.6, "benefit_of_action": 0.7, "relationship_salience": 0.6,
                        "needs_clarification": 0.2, "needs_delegation": 0.1, "thread_continuity": 1.0,
                        "why": "direct ask"})


def test_interpret_parses_clamps_and_meters():
    meter = {}
    it = interpret(lambda p: STUB_JSON, actor="b", channel="email", context="-", content="x", meter=meter)
    assert it.intent == "request_action" and it.urgency == 0.8 and it.thread_continuity == 1.0
    assert meter["calls"] == 1 and meter["tokens"] > 0
    # out-of-range values clamp, junk intent maps to "other"
    bad = json.dumps({"intent": "??", "urgency": 7, "obligation": -3})
    it2 = interpret(lambda p: bad, actor="b", channel="email", context="-", content="x")
    assert it2.intent == "other" and it2.urgency == 1.0 and it2.obligation == 0.0


def test_interpret_abstains_on_garbage():
    assert interpret(lambda p: "not json at all", actor="b", channel="c", context="-", content="x") is None


def test_feature_vector_contract():
    x = Interpretation().features()
    assert len(x) == len(FEATURE_DIMS) + len(INTENTS)


def test_fitted_policy_reduces_to_anchor_and_learns():
    # zero weights → exactly the metadata anchor
    pol0 = ActionPolicy(w=[0.0] * 18, w_anchor=1.0, b=0.0)
    assert abs(pol0.p_engage(Interpretation().features(), 0.2) - 0.2) < 1e-6
    # synthetic: urgency drives engagement → fitted weight positive, hi > lo
    rng = random.Random(0)
    samples = []
    for _ in range(400):
        it = Interpretation(urgency=rng.random())
        samples.append((it.features(), 0.2, 1 if rng.random() < 0.05 + 0.5 * it.urgency else 0))
    pol = fit_action_policy(samples)
    assert pol.w[0] > 0.3
    hi = pol.p_engage(Interpretation(urgency=0.95).features(), 0.2)
    lo = pol.p_engage(Interpretation(urgency=0.05).features(), 0.2)
    assert hi > lo + 0.05


def test_action_distribution_normalizes_and_shapes():
    it = Interpretation(urgency=0.9, needs_clarification=0.8, needs_delegation=0.6)
    for pack in ACTION_PACKS:
        d = action_distribution(pack, it, 0.4)
        assert set(d) == set(ACTION_PACKS[pack]) and abs(sum(d.values()) - 1.0) < 1e-9
        assert all(v >= 0 for v in d.values())
    # engagement mass is preserved (calibration is never re-decided by the split)
    d = action_distribution("messaging", it, 0.4)
    assert abs(d["reply_now"] + d["reply_later"] + d["ask_clarification"] - 0.4) < 1e-9
    # clarification need moves mass into ask_clarification; urgency moves reply mass earlier
    d0 = action_distribution("messaging", Interpretation(urgency=0.9), 0.4)
    assert d["ask_clarification"] > d0["ask_clarification"]
    dl = action_distribution("messaging", Interpretation(urgency=0.05), 0.4)
    assert d0["reply_now"] > dl["reply_now"]


def test_attention_process_bounded_and_workload_coupled():
    rng = random.Random(1)
    a = 0.9
    for _ in range(200):
        a = attention_transition(a, dt_days=0.3, workload_norm=1.0, hour=11, rng=rng)
        assert 0.05 <= a <= 1.0
    assert a < 0.75                          # high workload pulls attention down toward its target
    b = 0.1
    for _ in range(200):
        b = attention_transition(b, dt_days=0.3, workload_norm=0.0, hour=11, rng=rng)
    assert b > 0.5                           # low workload daytime pulls it back up


def test_relationship_state_and_transition():
    weak = relationship_strength(0, 0.03, 0.034)
    strong = relationship_strength(40, 0.4, 0.034)
    assert 0.0 <= weak < strong <= 1.0
    assert relationship_modulator(strong, 0.9) > 1.0 > relationship_modulator(weak, 0.1)
    s = relationship_transition(0.5, engaged=True)
    assert s > 0.5 and relationship_transition(0.5, engaged=False) < 0.5
    assert relationship_transition(1.0, engaged=True) <= 1.0


def test_hidden_state_latents_are_labeled_and_correlated():
    lat, cors = hidden_state_latents("recipient", workload_norm=0.8, hetero_sd=0.3)
    paths = {r.path for r in lat}
    assert {"recipient.responsiveness", "recipient.attention",
            "recipient.obligation_sensitivity"} <= paths
    assert all(r.method in ("prior", "dataset") for r in lat)      # labeled, never unsupported precision
    assert cors and cors[0].strength == 0.3


def _fm_ex():
    from swm.world_model_v2.reference.enron import FittedMechanisms, MessageExample
    fm = FittedMechanisms(global_rate=0.034, hazard=[0.01, 0.008, 0.006, 0.004, 0.003],
                          workload_mult=[1.1, 1.0, 0.9], hour_mult={h: 1.0 for h in range(4)},
                          weekday_mult={d: 1.0 for d in range(6)}, terciles=(3, 10),
                          check_rate_per_day=5.0)
    ex = MessageExample(msg_id="m1", sender="a@x.com", recipient="b@x.com", subject="Re: budget",
                        body="can you send the Q3 numbers today?", sent_ts=1e9, replied=True,
                        delay_days=0.2, feats={"pair_n": 10, "pair_rate": 0.3, "rcpt_n": 100,
                                               "rcpt_rate": 0.1, "inbox_7d": 20, "hour": 10,
                                               "weekday": 2, "thread": 1})
    return fm, ex


def test_structured_world_runs_at_every_level_with_typed_actions():
    from swm.world_model_v2.reference.enron import interpret_message, v2_predict_actor
    fm, ex = _fm_ex()
    interp = interpret_message(ex, lambda p: STUB_JSON)
    pol = ActionPolicy(w=[0.0] * len(interp.features()), w_anchor=1.0, b=0.0)
    seen_actions = set()
    for lvl in (2, 3, 4, 5, 6):
        o = v2_predict_actor(ex, fm, interp, pol, level=lvl, n_particles=16, seed=3)
        assert 0.0 <= o["p_by"][7.0] <= 1.0
        assert o["p_by"][14.0] >= o["p_by"][1.0]             # cumulative monotone
        assert o["n_deltas"] > 0 and not o["abstained"]
        assert o["interpretation"]["intent"] == "request_action"
        seen_actions |= set(o["terminal_actions"])
    assert seen_actions <= {"reply", "reply_later", "delegate", "ignore"}
    # abstention path: no interpretation → metadata anchor only, flagged
    o = v2_predict_actor(ex, fm, None, pol, level=6, n_particles=8, seed=1)
    assert o["abstained"] and o["interpretation"] is None


def test_structured_world_content_actually_moves_the_prediction():
    from swm.world_model_v2.reference.enron import interpret_message, v2_predict_actor
    fm, ex = _fm_ex()
    interp = interpret_message(ex, lambda p: STUB_JSON)
    # a policy with real feature weights: urgent-obligated messages engage far more
    pol = fit_action_policy([(Interpretation(urgency=u, obligation=u).features(), 0.1,
                              1 if u > 0.5 else 0) for u in [i / 40 for i in range(40)] * 8])
    hi = v2_predict_actor(ex, fm, interp, pol, level=6, n_particles=64, seed=5)
    low_interp = Interpretation(urgency=0.02, obligation=0.02)
    lo = v2_predict_actor(ex, fm, low_interp, pol, level=6, n_particles=64, seed=5)
    assert hi["p_engage"] > lo["p_engage"] + 0.1
    assert hi["p_by"][7.0] > lo["p_by"][7.0]


def test_universality_negotiation_domain_uses_identical_machinery():
    """The acceptance test PART 0 demands: the SAME interpretation schema + fitted policy + action pack
    machinery handles a negotiation actor with no messaging code involved."""
    offer = json.dumps({"intent": "transactional", "urgency": 0.6, "obligation": 0.4,
                        "task_ownership": 1.0, "effort_required": 0.5, "relevance_to_goals": 0.95,
                        "risk_of_inaction": 0.7, "benefit_of_action": 0.55, "relationship_salience": 0.8,
                        "needs_clarification": 0.5, "needs_delegation": 0.0, "thread_continuity": 0.6,
                        "why": "credible offer near reservation"})
    it = interpret(lambda p: offer, actor="seller (SME owner)", channel="acquisition negotiation",
                   context="- 3 prior rounds\n- your reservation price: unknown to buyer",
                   content="We propose $2.1M cash, 30-day close, no earnout.")
    assert it is not None and it.intent == "transactional"
    pol = ActionPolicy(w=[0.0] * len(it.features()), w_anchor=1.0, b=0.0)
    p = pol.p_engage(it.features(), 0.35)
    d = action_distribution("negotiation", it, p)
    assert set(d) == set(ACTION_PACKS["negotiation"]) and abs(sum(d.values()) - 1.0) < 1e-9
    # engagement mass equals the calibrated p; passive mass (delay+reject) is the complement
    assert abs(d["accept"] + d["counteroffer"] + d["ask_clarification"] - p) < 1e-9
    assert abs(d["delay"] + d["reject"] - (1 - p)) < 1e-9
