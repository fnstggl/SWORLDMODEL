"""Phase 3 — production posterior world-state inference (scripted, no network).

These tests prove the posterior is REAL and CAUSALLY CONSUMED, not a Bayesian-looking interface:

  * observation models turn qualitative tags into numeric likelihoods with FIXED params (Part C);
  * dependence correction collapses syndicated copies BEFORE likelihood multiplication (Part D);
  * infer_posterior produces a numeric, likelihood-updated, deterministic posterior (Parts H-K);
  * the posterior is INJECTED into the resolve_outcome event and MOVES the terminal distribution
    (the WORLD-STATE→EXECUTION bridge — the anti-scaffolding invariant);
  * an ornamental (unconsumed / unlinked) latent is rejected, never counted as production inference.
"""
import json

import pytest

from swm.world_model_v2.compiler import compile_world
from swm.world_model_v2.materialize import run_from_plan
from swm.world_model_v2.phase3_latent_spec import (ClaimTag, LatentVariableSpec, outcome_rate_spec,
                                                   structural_spec)
from swm.world_model_v2.phase3_observation import (DirectionalRateModel, StructuralDetectionModel,
                                                   collapse_by_dependence)
from swm.world_model_v2.phase3_posterior import infer_posterior

AS_OF, HORIZON = "2023-05-01", "2023-06-01"


def _llm(decomp):
    return lambda p: json.dumps(decomp)


def _binary_plan(lean="neutral", hyps=None):
    decomp = {"outcome": {"family": "binary", "options": ["yes", "no"], "resolution_rule": "yes",
                          "readout_var": "out"}, "outcome_lean": lean,
              "entities": [{"id": "a", "type": "person", "fields": {}}],
              "required_causal_processes": ["decide"], "rationale": "x"}
    if hyps:
        decomp["structural_hypotheses"] = hyps
    return compile_world("Will yes?", llm=_llm(decomp), evidence="", as_of=AS_OF, horizon=HORIZON, seed=1)


def _p_yes(result):
    dist = result.get("distribution") or {}
    return float(dist.get("yes", 0.0))


# ============================================================ Part C: observation models (numeric from tags)
def test_directional_rate_model_votes_with_direction():
    m = DirectionalRateModel()
    yes = ClaimTag(claim_id="c", outcome_direction="supports_yes", strength="strong", reliability=0.9)
    no = ClaimTag(claim_id="c", outcome_direction="supports_no", strength="strong", reliability=0.9)
    neu = ClaimTag(claim_id="c", outcome_direction="neutral")
    # a strong yes-vote is MORE likely under a high rate than a low rate
    assert m.likelihood(yes, 0.9) > m.likelihood(yes, 0.1)
    # a strong no-vote is MORE likely under a low rate
    assert m.likelihood(no, 0.1) > m.likelihood(no, 0.9)
    # neutral is uninformative (never reweights)
    assert m.likelihood(neu, 0.9) == 1.0 == m.likelihood(neu, 0.1)


def test_reliability_flattens_toward_coin_flip():
    m = DirectionalRateModel()
    strong_src = ClaimTag(claim_id="c", outcome_direction="supports_yes", strength="strong", reliability=1.0)
    weak_src = ClaimTag(claim_id="c", outcome_direction="supports_yes", strength="strong", reliability=0.0)
    # a zero-reliability source is a coin flip: likelihood ~0.5 regardless of rate (near-uninformative)
    assert abs(m.likelihood(weak_src, 0.9) - 0.5) < 1e-6
    # a reliable source discriminates strongly
    assert m.likelihood(strong_src, 0.9) > 0.7


def test_strategic_statement_is_discounted():
    m = DirectionalRateModel()
    sincere = ClaimTag(claim_id="c", outcome_direction="supports_yes", strength="strong",
                       reliability=0.9, is_strategic=False)
    strategic = ClaimTag(claim_id="c", outcome_direction="supports_yes", strength="strong",
                         reliability=0.9, is_strategic=True)
    # the strategic statement moves the likelihood LESS far from 0.5 (worth less)
    assert abs(m.likelihood(strategic, 0.9) - 0.5) < abs(m.likelihood(sincere, 0.9) - 0.5)


def test_structural_detection_supports_named_hypothesis():
    m = StructuralDetectionModel()
    tag = ClaimTag(claim_id="c", supports_hypotheses=["H1"], strength="strong", reliability=0.9)
    # a claim that supports H1 raises H1's log-likelihood above a hypothesis it is silent on
    assert m.loglik_for_hypothesis(tag, "H1") > m.loglik_for_hypothesis(tag, "H2")


# ============================================================ Part D: dependence correction
def test_dependence_collapses_syndicated_copies():
    tags = [ClaimTag(claim_id=f"c{i}", outcome_direction="supports_no", strength="moderate",
                     reliability=0.8, dependence_group="wire-42") for i in range(4)]
    collapsed = collapse_by_dependence(tags)
    assert len(collapsed) == 1                                   # 4 copies → 1 effective observation
    assert collapsed[0].n_collapsed == 4
    assert collapsed[0].outcome_direction == "supports_no"


def test_ungrouped_tags_pass_through_as_singletons():
    tags = [ClaimTag(claim_id="a", dependence_group=""), ClaimTag(claim_id="b", dependence_group="")]
    assert len(collapse_by_dependence(tags)) == 2


def test_dependence_makes_posterior_less_confident():
    """Four syndicated no-copies must move the posterior LESS than four independent no-reports."""
    plan = _binary_plan()
    grouped = [ClaimTag(claim_id=f"c{i}", outcome_direction="supports_no", strength="strong",
                        reliability=0.85, dependence_group="wire-1") for i in range(4)]
    indep = [ClaimTag(claim_id=f"c{i}", outcome_direction="supports_no", strength="strong",
                      reliability=0.85, dependence_group=f"src-{i}") for i in range(4)]
    p_grouped = infer_posterior(plan, None, grouped, seed=3)
    p_indep = infer_posterior(plan, None, indep, seed=3)
    assert p_grouped.n_effective_observations == 1
    assert p_indep.n_effective_observations == 4
    # both push the rate DOWN (supports_no), but the dependence-collapsed one moves less far from 0.5
    assert p_grouped.outcome_rate_mean > p_indep.outcome_rate_mean


# ============================================================ Parts H-K: the posterior itself
def test_no_evidence_leaves_prior_unchanged():
    plan = _binary_plan(lean="neutral")
    post = infer_posterior(plan, None, [], seed=0)
    assert post.n_effective_observations == 0
    assert abs(post.outcome_rate_mean - post.outcome_rate_prior_mean) < 0.05   # no update without evidence


def test_yes_evidence_raises_posterior_rate():
    plan = _binary_plan(lean="neutral")
    tags = [ClaimTag(claim_id="c1", outcome_direction="supports_yes", strength="strong", reliability=0.9)]
    post = infer_posterior(plan, None, tags, seed=0)
    assert post.outcome_rate_mean > post.outcome_rate_prior_mean + 0.05


def test_posterior_is_deterministic_under_seed():
    plan = _binary_plan()
    tags = [ClaimTag(claim_id="c1", outcome_direction="supports_yes", strength="moderate", reliability=0.8)]
    a = infer_posterior(plan, None, tags, seed=7)
    b = infer_posterior(plan, None, tags, seed=7)
    assert a.outcome_rate_mean == b.outcome_rate_mean
    assert a.outcome_rate_particles == b.outcome_rate_particles


def test_structural_posterior_normalizes_and_updates():
    hyps = [{"id": "H1", "describe": "s1", "prior": 0.5, "lean": "weak_yes"},
            {"id": "H2", "describe": "s2", "prior": 0.5, "lean": "weak_no"}]
    plan = _binary_plan(hyps=hyps)
    tags = [ClaimTag(claim_id="c1", supports_hypotheses=["H1"], strength="strong", reliability=0.9)]
    post = infer_posterior(plan, None, tags, seed=0)
    assert abs(sum(post.structural_posterior.values()) - 1.0) < 1e-6      # normalized
    assert post.structural_posterior["H1"] > post.structural_prior["H1"]  # evidence for H1 raised its mass


def test_extreme_likelihoods_do_not_nan_or_collapse():
    plan = _binary_plan()
    tags = [ClaimTag(claim_id=f"c{i}", outcome_direction="supports_yes", strength="strong",
                     reliability=1.0) for i in range(30)]                 # a mountain of strong evidence
    post = infer_posterior(plan, None, tags, seed=1)
    assert 0.0 <= post.outcome_rate_mean <= 1.0
    assert post.outcome_rate_mean > 0.7                                   # converged high, no NaN/collapse
    for _, w in post.outcome_rate_particles:
        assert w == w                                                    # no NaN weights


def test_all_neutral_tags_do_not_update_the_rate():
    plan = _binary_plan(lean="neutral")
    tags = [ClaimTag(claim_id=f"c{i}", outcome_direction="neutral", strength="strong") for i in range(5)]
    post = infer_posterior(plan, None, tags, seed=0)
    assert post.n_effective_observations == 5                            # they ARE observations...
    assert abs(post.outcome_rate_mean - post.outcome_rate_prior_mean) < 0.05   # ...but neutral → no rate move


def test_contradictory_evidence_stays_uncertain():
    """Equal strong yes and no evidence must keep the rate near the middle and NOT fabricate confidence."""
    plan = _binary_plan(lean="neutral")
    tags = ([ClaimTag(claim_id=f"y{i}", outcome_direction="supports_yes", strength="strong",
                      reliability=0.85, dependence_group=f"y{i}") for i in range(3)] +
            [ClaimTag(claim_id=f"n{i}", outcome_direction="supports_no", strength="strong",
                      reliability=0.85, dependence_group=f"n{i}") for i in range(3)])
    post = infer_posterior(plan, None, tags, seed=2)
    assert 0.35 < post.outcome_rate_mean < 0.65                          # contradiction → no false confidence
    assert post.outcome_rate_sd > 0.12                                   # and honest spread retained


def test_particle_filter_resamples_without_degenerating():
    """A run of strong same-direction evidence must trigger resampling (ESS collapse) yet finish with a
    healthy particle set (rejuvenation prevents impoverishment) — deterministically."""
    plan = _binary_plan(lean="neutral")
    tags = [ClaimTag(claim_id=f"c{i}", outcome_direction="supports_yes", strength="strong",
                     reliability=0.95, dependence_group=f"s{i}") for i in range(12)]
    post = infer_posterior(plan, None, tags, seed=3, n_rate_particles=200, resample_threshold=0.5)
    assert any(row["resampled"] for row in post.assimilation_ledger)     # ESS collapse DID trigger a resample
    assert post.rate_ess > 5                                             # but the filter did not degenerate
    distinct = len({round(r, 3) for r, _ in post.outcome_rate_particles})
    assert distinct > 10                                                # rejuvenation kept particle diversity


# ============================================================ the WORLD-STATE → EXECUTION bridge
def test_posterior_moves_the_terminal_distribution():
    """The anti-scaffolding proof: a posterior attached to the plan must change the terminal frequencies vs
    the prior-only path, and a yes-posterior and a no-posterior must diverge."""
    base = _binary_plan(lean="neutral")
    prior_result, _ = run_from_plan(base, seed=0, n_particles=200)
    p_prior = _p_yes(prior_result)

    yes_plan = _binary_plan(lean="neutral")
    yes_plan.posterior_rate_particles = [(0.9, 1.0)]                      # posterior concentrated at 0.9
    yes_result, yes_branches = run_from_plan(yes_plan, seed=0, n_particles=200)
    p_yes = _p_yes(yes_result)

    no_plan = _binary_plan(lean="neutral")
    no_plan.posterior_rate_particles = [(0.1, 1.0)]                       # posterior concentrated at 0.1
    no_result, _ = run_from_plan(no_plan, seed=0, n_particles=200)
    p_no = _p_yes(no_result)

    assert p_yes > p_prior > p_no                                        # the posterior CAUSALLY drives the terminal
    assert p_yes > 0.8 and p_no < 0.2                                    # and drives it to the posterior rate
    # forensic proof the resolver READ the posterior (not the prior Beta)
    srcs = {(d.uncertainty or {}).get("rate_source") for b in yes_branches for d in b.log
            if d.event_type == "resolve_outcome"}
    assert "posterior" in srcs


def test_prior_only_path_labels_rate_source_prior():
    base = _binary_plan(lean="neutral")
    _, branches = run_from_plan(base, seed=0, n_particles=50)
    srcs = {(d.uncertainty or {}).get("rate_source") for b in branches for d in b.log
            if d.event_type == "resolve_outcome"}
    assert srcs == {"prior_beta"}                                        # no posterior → honest prior label


def test_structural_posterior_weights_strata():
    """When a structural posterior is attached, particle strata are allocated by posterior mass (not prior)."""
    hyps = [{"id": "H1", "describe": "s1", "prior": 0.5, "lean": "strong_yes"},
            {"id": "H2", "describe": "s2", "prior": 0.5, "lean": "strong_no"}]
    plan = _binary_plan(hyps=hyps)
    plan.structural_posterior = {"H1": 0.9, "H2": 0.1}                    # evidence strongly favors H1
    result, _ = run_from_plan(plan, seed=0, n_particles=200)
    assert result.get("structural_source") == "phase3_evidence_posterior"
    # H1 leans strong_yes and holds 90% of the mass → the terminal should lean yes
    assert _p_yes(result) > 0.5


# ============================================================ anti-ornamental invariant (representation principle)
def test_measurable_latent_requires_evidence_model_and_consumer():
    good = outcome_rate_spec("Will yes?", ["c1", "c2"])
    assert good.measurable()                                             # has support + observation model + consumer
    ornamental = LatentVariableSpec(
        variable_id="trust", definition="how much A trusts B",
        measurable_interpretation="a vibe", support_type="bounded_continuous",
        observation_models=[], consumed_by=[])                           # no model, no consumer → ornamental
    assert not ornamental.measurable()                                  # rejected: not production inference


def test_structural_spec_only_when_competing_structures_exist():
    single = _binary_plan()
    assert structural_spec(single) is None                              # <2 hypotheses → no structural latent
    multi = _binary_plan(hyps=[{"id": "H1", "prior": 0.5, "lean": "weak_yes"},
                               {"id": "H2", "prior": 0.5, "lean": "weak_no"}])
    ss = structural_spec(multi)
    assert ss is not None and ss.measurable() and ss.support_type == "discrete_structural"


# ============================================================ representation-choice principle
def test_ornamental_scalar_representation_is_refused():
    from swm.world_model_v2.phase3_representation import (OrnamentalRepresentationError,
                                                         RepresentationCandidate, assert_not_ornamental)
    bare = RepresentationCandidate(concept="trust", kind="scalar_point")   # no evidence, no consumer
    with pytest.raises(OrnamentalRepresentationError):
        assert_not_ornamental(bare)                                     # the trust=0.7 anti-pattern is rejected
    linked = RepresentationCandidate(concept="outcome_rate", kind="continuous_probabilistic",
                                     evidence_claim_ids=["c1"], consumed_by=["resolve_outcome:rate"])
    assert_not_ornamental(linked)                                      # evidence-linked + consumed → allowed


def test_llm_proposes_kinds_never_numbers():
    from swm.world_model_v2.phase3_representation import propose_representations
    # default menu when no LLM: candidates are qualitative KINDS with no numeric parameters
    cands = propose_representations("regime stability", llm=None, evidence_claim_ids=["c1"],
                                   consumed_by=["x"])
    assert cands and all(c.kind in
                         ("scalar_point", "continuous_probabilistic", "discrete_hypothesis", "mixture",
                          "hybrid_interpretable") for c in cands)
    for c in cands:
        d = c.as_dict()
        assert not any(isinstance(v, float) for v in d.values())       # no numbers minted at proposal time


def test_scalar_baseline_loses_to_structured_representation():
    """A miniature of the representation ablation: on a genuinely bimodal truth, the mixture representation
    beats the arbitrary-scalar baseline on held-out log-loss (structure must not be scalarized away)."""
    import random as _r
    from swm.world_model_v2.phase3_representation import choose_representation
    rng = _r.Random(1)

    def episode():
        theta = 0.85 if rng.random() < 0.5 else 0.15                    # two plausible worlds
        votes = []
        for _ in range(8):
            rel = 0.6 + 0.35 * rng.random()
            d = 0.5 + 0.35 * rel
            p = theta * d + (1 - theta) * (1 - d)
            votes.append((1 if rng.random() < p else -1, rel))
        return votes, (1 if rng.random() < theta else 0)

    eps = [episode() for _ in range(900)]
    card = choose_representation("bimodal", eps[:300], eps[300:],
                                candidates=["scalar_point", "mixture", "continuous_probabilistic"])
    assert card.winner != "scalar_point"                               # the arbitrary scalar never wins here
    assert card.metrics["mixture"]["held_out_logloss"] < card.metrics["scalar_point"]["held_out_logloss"]


# ============================================================ Part B: prior construction + transport-risk
def _beta_var(spec):
    a, b = spec.alpha, spec.beta
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


def test_transport_risk_widens_reference_prior():
    from swm.world_model_v2.phase3_priors import reference_class_prior
    low = reference_class_prior("comparable Fed meetings", 30, 100, transport_risk="low")
    high = reference_class_prior("comparable Fed meetings", 30, 100, transport_risk="high")
    severe = reference_class_prior("comparable Fed meetings", 30, 100, transport_risk="severe")
    # the reference MEAN is preserved across transport levels (~0.3)
    for s in (low, high, severe):
        assert abs(s.mean - 0.3) < 0.12
    # more transport risk ⇒ fewer retained effective observations ⇒ WIDER prior (higher variance)
    assert low.retained_effective_n > high.retained_effective_n > severe.retained_effective_n
    assert _beta_var(severe) > _beta_var(high) > _beta_var(low)


def test_no_reference_data_falls_back_to_generic_lean():
    from swm.world_model_v2.phase3_priors import reference_class_prior
    spec = reference_class_prior("some class", 0, 0, transport_risk="low", lean="weak_yes")
    assert spec.source_class == "generic_weakly_informative"           # no data → NOT a reference-class prior
    assert spec.provenance.get("class") == "generic_weakly_informative"


def test_reference_prior_never_tighter_than_cap():
    from swm.world_model_v2.phase3_priors import reference_class_prior, MAX_EFFECTIVE_N
    huge = reference_class_prior("big class", 5000, 10000, transport_risk="none")
    assert huge.retained_effective_n <= MAX_EFFECTIVE_N + 1e-6         # a reference prior stays broad
    assert abs(huge.mean - 0.5) < 0.05


def test_prior_spec_changes_posterior_starting_point():
    plan = _binary_plan(lean="neutral")
    from swm.world_model_v2.phase3_priors import reference_class_prior
    low_prior = reference_class_prior("class", 20, 100, transport_risk="none")   # centered 0.2, fairly tight
    post = infer_posterior(plan, None, [], seed=0, prior_spec=low_prior)
    assert post.n_effective_observations == 0
    assert abs(post.outcome_rate_prior_mean - 0.2) < 0.05              # the prior mean flows through
    assert post.outcome_rate_mean < 0.4                               # with no evidence, posterior ~ prior
