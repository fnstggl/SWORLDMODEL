"""The three-layer message optimizer: objective, L1 search, L2 construction, L3 evaluation, guardrail.

The optimizer must (a) express recipient-conditioned truths (the credential sign-flip), (b) find the
optimal STRATEGY in variable space, (c) CONSTRUCT an email move-by-move that beats naive drafts, (d)
evaluate it as a distribution over the recipient's hidden state, and (e) not be gameable (lower-bound
objective + saturating encoders).
"""
from swm.decision.compositional_search import construct_email, encode_text_to_strategy
from swm.decision.mc_evaluation import mc_evaluate
from swm.decision.message_optimizer import optimize_strategy
from swm.decision.message_pipeline import RecipientState, optimize_message
from swm.decision.strategy_scorer import MESSAGE_VARS, scorer_from_recipient

# a prestige-skeptic, contrarian, outreach-friendly recipient (Thiel-shaped)
SKEPTIC = {"status_orientation": 0.85, "skepticism": 0.9, "status": 0.9, "openness_to_outreach": 0.9,
           "attention_availability": 0.4, "platform_response_norm": 0.3, "relationship_strength": 0.0}
# a status-driven, consensus-minded recipient
STATUS_SEEKER = {"status_orientation": 0.15, "skepticism": 0.2, "status": 0.8, "openness_to_outreach": 0.5,
                 "attention_availability": 0.5, "platform_response_norm": 0.3, "relationship_strength": 0.0}

NEUTRAL = {v: 0.3 for v in MESSAGE_VARS}


# --- the objective: recipient-conditioned sign-flip -----------------------------------------------

def test_credential_signaling_flips_sign_by_recipient():
    skeptic = scorer_from_recipient(SKEPTIC, 0.2)
    seeker = scorer_from_recipient(STATUS_SEEKER, 0.2)
    cred = {**NEUTRAL, "credential_signaling": 0.9}
    assert skeptic.mean(cred) < skeptic.mean(NEUTRAL)      # hurts the prestige-skeptic
    assert seeker.mean(cred) > seeker.mean(NEUTRAL)        # helps the status-seeker


def test_proof_pays_off_with_a_skeptic():
    # the general form of "traction for Cuban": a skeptic rewards concrete proof over assertion
    skeptic = scorer_from_recipient(SKEPTIC, 0.2)
    proof = {**NEUTRAL, "credibility_proof": 0.9}
    assert skeptic.mean(proof) > skeptic.mean(NEUTRAL)


def test_situational_lever_moves_the_score():
    from swm.decision.strategy_scorer import Lever
    skeptic = scorer_from_recipient(SKEPTIC, 0.2, levers=[Lever("contrarian_thesis", 1.8, 0.7, "a non-consensus claim")])
    assert "contrarian_thesis" in skeptic.optimizable_vars()
    hi = {**NEUTRAL, "contrarian_thesis": 1.0}
    lo = {**NEUTRAL, "contrarian_thesis": 0.0}
    assert skeptic.mean(hi) > skeptic.mean(lo)             # the situational lever pays off


def test_lower_bound_is_pessimistic():
    s = scorer_from_recipient(SKEPTIC, 0.2)
    d = s.score_dist({**NEUTRAL, "personalization": 0.9})
    assert d.lower_bound(0.2) <= d.mean


# --- L1: strategy optimization --------------------------------------------------------------------

def test_l1_finds_the_right_corner_for_a_skeptic():
    spec = optimize_strategy(scorer_from_recipient(SKEPTIC, 0.2))
    st = spec.strategy
    assert st["pushiness"] < 0.2                 # never be pushy with high-status
    assert st["personalization"] > 0.7
    assert st["credential_signaling"] < 0.2      # the sign-flip: drop credentials for a prestige-skeptic
    assert st["credibility_proof"] > 0.7         # a skeptic rewards proof


def test_l1_recommends_credentials_for_a_status_seeker():
    spec = optimize_strategy(scorer_from_recipient(STATUS_SEEKER, 0.2))
    # same objective, opposite recommendation on the conditioned variable
    assert spec.strategy["credential_signaling"] > optimize_strategy(
        scorer_from_recipient(SKEPTIC, 0.2)).strategy["credential_signaling"]


# --- the text encoder: saturating (anti-Goodhart) -------------------------------------------------

def test_encoder_saturates_on_repeated_markers():
    one = encode_text_to_strategy("Is this wrong?")
    many = encode_text_to_strategy("Wrong??????????? wrong wrong wrong wrong wrong wrong?")
    assert many["ask_directness"] <= 1.0 and many["credibility_proof"] <= 1.0
    # spamming question marks does not keep increasing directness without bound
    assert many["ask_directness"] - one["ask_directness"] < 0.5


def test_encoder_catches_imperative_ask_and_broad_credentials():
    # the two Cuban-email misses that motivated the LLM encoder: imperative asks and "Wharton"
    e = encode_text_to_strategy("I'm a Wharton student. We hit 10k ARR. Reply yes and I'll send the deck.")
    assert e["ask_directness"] > 0.3            # "reply yes" is a real ask even without a "?"
    assert e["credential_signaling"] > 0.3      # "Wharton" is now in the lexicon
    assert e["credibility_proof"] > 0.3         # "10k ARR" is proof


def test_encoder_detects_pushiness_and_credentials():
    e = encode_text_to_strategy("Princeton admit featured in the NYT. Please respond ASAP, circling back.")
    assert e["pushiness"] > 0.4 and e["credential_signaling"] > 0.4


# --- L2: construction beats naive drafts and avoids bad moves -------------------------------------

def test_l2_constructed_email_beats_baselines_and_avoids_bad_moves():
    scorer = scorer_from_recipient(SKEPTIC, 0.2)
    spec = optimize_strategy(scorer)
    email = construct_email(scorer, spec.strategy)
    cred = encode_text_to_strategy("Dear Mr. Thiel, I'm a Princeton admit featured in the NYT. Please respond ASAP.")
    assert scorer.mean(email.strategy) > scorer.mean(cred)
    # for a prestige-skeptic the search must not pick the credential opener or the pushy ask
    assert "Princeton" not in email.text
    assert "ASAP" not in email.text and "circling back" not in email.text
    assert email.strategy["pushiness"] < 0.2


# --- L3: Monte-Carlo evaluation -------------------------------------------------------------------

def test_l3_returns_a_distribution_and_tracks_responsiveness():
    strat = optimize_strategy(scorer_from_recipient(SKEPTIC, 0.2)).strategy
    lo = mc_evaluate(SKEPTIC, 0.05, strat, base_n_effective=6, n_samples=1500, seed=1)
    hi = mc_evaluate(SKEPTIC, 0.45, strat, base_n_effective=6, n_samples=1500, seed=1)
    assert hi.p_mean > lo.p_mean                              # higher base rate -> more replies
    assert lo.interval80[0] < lo.interval80[1]                # a real interval, not a point
    assert 0.0 <= lo.p_reply <= 1.0


# --- full pipeline --------------------------------------------------------------------------------

def test_pipeline_constructed_email_beats_naive_drafts():
    rs = RecipientState(vars=SKEPTIC, base_mean=0.2, base_n_effective=6.0,
                        confidences={k: 0.5 for k in SKEPTIC}, label="Skeptic")
    res = optimize_message(rs, n_mc=1500, seed=0)
    best = res.evaluation.p_mean
    for b in res.baselines.values():
        assert best > b["mc"].p_mean                          # the constructed email wins on the evaluator
    assert res.email.strategy["credential_signaling"] < 0.2
