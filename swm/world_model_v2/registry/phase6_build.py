"""Phase 6 registry population — new evidence-backed families + packs, causal-process declarations.

register_phase6(store) is called by build_registry.build() AFTER the base families. It:
  1. declares `answers_processes` on existing families (so the compiler can request BY CAUSAL PROCESS);
  2. registers NEW executable families parameterized by VERIFIED published estimates (Tier-4,
     domain_restricted) — each a DISTINCT causal mechanism, not a renamed generic hazard;
  3. registers NEW locally-fitted families from the committed real-data fits (attrition; content-response);
  4. PRESERVES honest nulls (StackExchange response, CMV persuasion — implemented, not validated);
  5. embeds the REAL fitted Higgs coefficients into the diffusion pack (no more "see artifact" pointer).

Every published estimate here was independently verified by the core agent against its primary source
(see docs/WMV2_PHASE6_RESEARCH_AND_CATALOG.md → core-agent accountability). Numbers are never hand-minted:
local packs read experiments/results/wmv2_phase6_fits.json; published packs carry the verified value + a
BROAD transport prior (never the tiny within-study SE) because transport, not measurement, is the risk.
"""
from __future__ import annotations

import json
import os

from swm.world_model_v2.registry.record import (ApplicabilityRule, Citation, MechanismRecord,
                                                ParameterPack, ParameterSpec, ValidationRecord)

FITS = "experiments/results/wmv2_phase6_fits.json"

# ---- canonical causal-process → families that ANSWER it (declared, not name-matched) ----
PROCESS_MAP = {
    "exposure_response_hazard": ["adoption_after_repeated_exposure", "diffusion_timing"],
    "simple_contagion_hazard": ["adoption_after_repeated_exposure", "diffusion_timing"],
    "complex_contagion_hazard": ["adoption_after_repeated_exposure", "social_reinforcement"],
    "threshold_adoption": ["adoption_after_repeated_exposure", "tipping"],
    "finite_population_saturation": ["cascade_saturation", "adoption_after_repeated_exposure"],
    "information_aging": ["diffusion_timing", "attention_decay"],
    "hawkes_self_excitation": ["diffusion_timing"],
    "quantal_response_choice": ["actor_selects_typed_action"],
    "social_preference_population": ["actor_selects_typed_action", "offer_response"],
    "reinforcement_learning": ["actor_selects_typed_action", "strategic_learning"],
    "belief_learning": ["actor_selects_typed_action", "strategic_learning"],
    "experience_weighted_attraction": ["actor_selects_typed_action", "strategic_learning"],
    "habit_formation": ["actor_selects_typed_action", "habit_persistence"],
    "bargaining_rubinstein": ["bargaining_split", "offer_response"],
    "negotiation_concession": ["concession", "offer_response"],
    "coalition_formation": ["coalition_formation"],
    "voting_turnout": ["participation_after_mobilization", "turnout"],
    "mobilization": ["participation_after_mobilization", "turnout"],
    "donation_response": ["donation_after_ask"],
    "trust_formation": ["trust_change_after_interaction"],
    "trust_violation": ["trust_change_after_interaction"],
    "trust_repair": ["trust_change_after_interaction"],
    "reciprocity": ["trust_change_after_interaction", "reciprocity"],
    "degroot_influence": ["belief_update_after_message", "opinion_convergence"],
    "bounded_confidence": ["belief_update_after_message", "opinion_convergence"],
    "latent_expressed_opinion": ["expressed_vs_latent_opinion"],
    "belief_update_exposure": ["belief_update_after_message"],
    "platform_examination": ["attention_after_exposure", "examination_by_rank"],
    "platform_ranking": ["ranking_allocation"],
    "network_rewiring": ["network_edge_change"],
    "institutional_vote": ["institutional_threshold_decision", "collective_vote"],
    "agenda_stage_control": ["agenda_gating", "institutional_threshold_decision"],
    "resource_depletion": ["resource_depletion"],
    "rare_event_arrival": ["exogenous_shock"],
    "attention_allocation": ["attention_after_exposure"],
    "engagement_momentum_persistence": ["habit_persistence", "engagement_persistence"],
}


def _load_fits():
    if not os.path.exists(FITS):
        return {}
    return json.load(open(FITS)).get("datasets", {})


def _hazard_pack_values(fit):
    """Turn a committed FeatureHazard fit into pack values with the real coefficients embedded. The harness
    stored the weights + intercept + which keys were standardized; the standardizer means/sds are recomputed
    from the train split at instantiation, so we persist the standardized-key list for execution."""
    c = fit["coefs"]
    return {"coefficients": {"source": "fitted", "method": c.get("fit", "logistic MLE"),
                             "dataset": fit["path"], "sd": None, "lo": None, "hi": None,
                             "value": {"weights": c["weights"], "intercept": c["intercept"],
                                       "standardized_keys": c.get("standardized", []),
                                       "note": "standardizer recomputed from train at instantiation"}}}


def register_phase6(s):
    # ---------- 0. declare answers_processes on existing families ----------
    for fam, procs in PROCESS_MAP.items():
        if fam in s.records:
            s.records[fam].applicability.answers_processes = procs

    fits = _load_fits()

    # ========================================================= A. BASS DIFFUSION (verified published)
    s.register(MechanismRecord(
        family_id="bass_diffusion", version="1.0.0", ontology_type="diffusion",
        title="Bass finite-population innovation/imitation diffusion",
        formal_description="dN/dt=(p+q·N/M)(M−N); p=external/innovation, q=internal/imitation; peak at "
                           "t*=ln(q/p)/(p+q). Executable Euler rollout (bass_trajectory).",
        causal_inputs=["market_size_M", "cumulative_adopters"], causal_outputs=["adoption_trajectory"],
        required_state=["quantities"], temporal_scale="periods",
        code_ref="swm.world_model_v2.registry.families.behavioral:bass_trajectory",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("p", "innovation coefficient", 0.0, 0.2, "published_research"),
                    ParameterSpec("q", "imitation coefficient", 0.0, 1.0, "published_research")],
        applicability=ApplicabilityRule(domains=["product_launch", "innovation_diffusion", "technology_adoption"],
                                        requires_state=["quantities"], time_scales=["periods", "years"],
                                        answers_processes=["cascade_saturation", "adoption_after_repeated_exposure"],
                                        transport_risk="high",
                                        exclusion_conditions=["unknown_market_size: Bass needs a finite M",
                                                              "low_cost_social_behavior: exposure≠adoption"]),
        citations=[Citation(ref="Sultan, Farley & Lehmann 1990, J. Marketing Research 27(1):70-77",
                            doi_or_url="10.1177/002224379002700107",
                            study_population="213 diffusion applications across 15 studies (durable goods, "
                                             "tech, services)", study_period="pre-1990",
                            finding="meta-analytic means p̄≈0.03, q̄≈0.38; imitation >> innovation",
                            limits="SDs ≈ means (p SD≈0.03, q SD≈0.35) → transport as a WIDE prior, never a "
                                   "point; durable-goods/innovation only; not low-cost social behavior")],
        uncertainty_note="p,q drawn from broad priors (meta-analytic SD ≈ mean); per-particle draw",
        known_failure_modes=["fails when exposure≠adoption (social behavior) or M is unknown"],
        status="proposed", status_reason="registering"))
    s.add_pack("bass_diffusion", ParameterPack(
        pack_id="bass_sultan_farley_lehmann_1990", family_id="bass_diffusion",
        domain="innovation_diffusion", population="213 durable-good/technology diffusion applications",
        values={"p": {"value": 0.03, "sd": 0.03, "lo": 0.001, "hi": 0.12, "source": "published_research",
                      "method": "meta-analytic mean", "dataset": "Sultan-Farley-Lehmann 1990"},
                "q": {"value": 0.38, "sd": 0.35, "lo": 0.05, "hi": 1.0, "source": "published_research",
                      "method": "meta-analytic mean", "dataset": "Sultan-Farley-Lehmann 1990"}},
        fit_method="published meta-analytic means (verified)", time_scale="periods",
        citations=[Citation(ref="Sultan, Farley & Lehmann 1990 JMR 27:70-77",
                            doi_or_url="10.1177/002224379002700107", finding="p̄=0.03, q̄=0.38",
                            limits="wide prior; durable-goods context")],
        transport_note="broad prior (meta-analytic SD≈mean); refit per product where sales data exist"))

    # ========================================================= B. BEHAVIORAL ULTIMATUM (verified)
    s.register(MechanismRecord(
        family_id="ultimatum_offer_response", version="1.0.0", ontology_type="bargaining",
        title="Behavioral ultimatum offer + acceptance",
        formal_description="P(accept|offer)=σ((offer−MAO)/softness); behavioral offers ≈0.40 (NOT the SPE "
                           "≈0). E[proposer]=P(accept)·(1−offer). Distinct from bargaining_rubinstein (SPE).",
        causal_inputs=["offer_fraction", "min_acceptable_offer"], causal_outputs=["accept_probability"],
        required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.behavioral:ultimatum_response",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("mean_offer", "modal proposer offer share", 0.0, 0.5, "published_research"),
                    ParameterSpec("rejection_rate", "overall rejection rate", 0.0, 0.5, "published_research")],
        applicability=ApplicabilityRule(domains=["economic_game", "one_shot_bargaining", "take_it_or_leave_it"],
                                        population_kinds=["lab"], time_scales=["event"],
                                        answers_processes=["offer_response", "bargaining_split"],
                                        transport_risk="high",
                                        exclusion_conditions=["repeated_bargaining: reputation changes offers"]),
        citations=[Citation(ref="Oosterbeek, Sloof & van de Kuilen 2004, Experimental Economics 7:171-188",
                            doi_or_url="10.1023/B:EXEC.0000026978.14316.74",
                            study_population="meta-analysis, 37 papers / 75 results, mostly student subjects",
                            finding="mean proposer offer ≈0.40 of pie; overall rejection ≈0.16; low offers "
                                    "(<0.2) rejected ~half the time",
                            limits="lab one-shot ultimatum; cultural variation; NOT the SPE split; NOT "
                                   "repeated bargaining")],
        uncertainty_note="offer/threshold drawn from published mean ± broad transport prior",
        status="proposed", status_reason="registering"))
    s.add_pack("ultimatum_offer_response", ParameterPack(
        pack_id="ultimatum_oosterbeek_2004", family_id="ultimatum_offer_response",
        domain="economic_game", population="ultimatum-game lab subjects (37-paper meta-analysis)",
        values={"mean_offer": {"value": 0.40, "sd": 0.08, "source": "published_research",
                               "method": "meta-analytic mean", "dataset": "Oosterbeek 2004"},
                "rejection_rate": {"value": 0.16, "sd": 0.08, "source": "published_research",
                                   "method": "meta-analytic mean", "dataset": "Oosterbeek 2004"},
                "accept_threshold": {"value": 0.25, "sd": 0.1, "lo": 0.0, "hi": 0.5,
                                     "source": "reference_class_prior",
                                     "method": "MAO implied by rejection of <0.2 offers",
                                     "dataset": "Oosterbeek 2004 + Camerer 2003"}},
        fit_method="published meta-analytic means (verified)", time_scale="event",
        citations=[Citation(ref="Oosterbeek et al. 2004 Exp Econ 7:171-188",
                            doi_or_url="10.1023/B:EXEC.0000026978.14316.74",
                            finding="offer 0.40, rejection 0.16", limits="lab; cultural variation")],
        validation=[ValidationRecord(
            kind="published_estimate", dataset="MobLab ultimatum (behaviorbench_eval.json)",
            split="descriptive cross-check", metric="mean_proposer_offer", value=0.4515,
            baseline="Oosterbeek meta mean 0.40", passed=True,
            artifact="experiments/results/behaviorbench_eval.json",
            note="local MobLab proposer mean 45.15/100 is consistent with the meta-analytic 0.40 (in-range); "
                 "descriptive cross-check, NOT a held-out validation of transport")],
        transport_note="lab one-shot ultimatum; transport to field bargaining unvalidated (widen)"))

    # ========================================================= C. TRUST GAME TRANSFER (verified)
    s.register(MechanismRecord(
        family_id="trust_game_transfer", version="1.0.0", ontology_type="relationship",
        title="Trust-game investment + return (Berg investment game)",
        formal_description="investor sends s·endowment (trust); tripled; trustee returns g·(3·sent) "
                           "(trustworthiness). Typed outcome: payoffs + investor_net_from_trust. Distinct "
                           "from trust_update (scalar edge shift) — this is the monetary transfer structure.",
        causal_inputs=["endowment", "send_fraction", "return_fraction"],
        causal_outputs=["investor_payoff", "trustee_payoff", "trust_paid_off"],
        required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.behavioral:trust_game_outcome",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("send_frac", "investor send fraction", 0.0, 1.0, "published_research"),
                    ParameterSpec("return_frac", "trustee return fraction", 0.0, 1.0, "published_research")],
        applicability=ApplicabilityRule(domains=["economic_game", "trust_game", "anonymous_exchange"],
                                        population_kinds=["lab"], answers_processes=["trust_change_after_interaction"],
                                        transport_risk="high",
                                        exclusion_conditions=["repeated_interaction: reputation changes g",
                                                              "no_monetary_transfer: not a trust game"]),
        citations=[Citation(ref="Johnson & Mislin 2011, J. Economic Psychology 32(5):865-889",
                            doi_or_url="10.1016/j.joep.2011.05.007",
                            study_population="meta-analysis, 162 replications, >23,000 subjects",
                            finding="investor sends ≈50% of endowment; trustee returns ≈37% of amount received",
                            limits="anonymous one-shot lab trust game; NOT real-world relationship trust; "
                                   "NOT a general 'trust level' scalar")],
        uncertainty_note="send/return fractions from meta-analytic means ± broad prior",
        status="proposed", status_reason="registering"))
    s.add_pack("trust_game_transfer", ParameterPack(
        pack_id="trust_game_johnson_mislin_2011", family_id="trust_game_transfer",
        domain="trust_game", population="162 trust-game replications (>23k subjects)",
        values={"send_frac": {"value": 0.50, "sd": 0.10, "source": "published_research",
                              "method": "meta-analytic mean", "dataset": "Johnson & Mislin 2011"},
                "return_frac": {"value": 0.37, "sd": 0.10, "source": "published_research",
                               "method": "meta-analytic mean", "dataset": "Johnson & Mislin 2011"}},
        fit_method="published meta-analytic means (verified)", time_scale="event",
        citations=[Citation(ref="Johnson & Mislin 2011 JEP 32:865-889", doi_or_url="10.1016/j.joep.2011.05.007",
                            finding="sent 0.50, returned 0.37", limits="lab anonymous trust game")],
        transport_note="lab trust game; do NOT read as real-world relationship trust"))

    # ========================================================= D. SOCIAL-PRESSURE TURNOUT (verified)
    s.register(MechanismRecord(
        family_id="social_pressure_turnout", version="1.0.0", ontology_type="participation",
        title="Social-pressure mailer effect on turnout",
        formal_description="P(turnout|arm) from randomized ITT levels; transported effect = (arm−control) "
                           "added to a scenario base turnout, clamped. Causal (randomized). Distinct from "
                           "generic mobilization contact (this is observability/social-pressure).",
        causal_inputs=["mailer_arm", "base_turnout"], causal_outputs=["turnout_probability"],
        required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.behavioral:social_pressure_turnout_p",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("neighbors_effect_pp", "Neighbors arm ITT (pp)", 0.0, 0.15, "published_research")],
        applicability=ApplicabilityRule(domains=["election", "voter_turnout", "get_out_the_vote"],
                                        population_kinds=["registered_voters"],
                                        answers_processes=["participation_after_mobilization", "turnout"],
                                        transport_risk="high",
                                        exclusion_conditions=["high_salience_election: ceiling effect",
                                                              "vote_choice: this is turnout, not persuasion"]),
        citations=[Citation(ref="Gerber, Green & Larimer 2008, APSR 102(1):33-48",
                            doi_or_url="10.1017/S000305540808009X",
                            study_population="180,002 households, Michigan 2006 primary (low-salience)",
                            finding="control turnout 29.7%; Civic Duty 31.5, Hawthorne 32.2, Self 34.5, "
                                    "Neighbors 37.8 (+8.1pp). Randomized ITT.",
                            limits="low-salience primary; household mailers; effect is on a low baseline; "
                                   "transport to high-salience elections overstates (ceiling)")],
        uncertainty_note="ITT levels are precise; TRANSPORT uncertainty is broad (widen off-context)",
        status="proposed", status_reason="registering"))
    s.add_pack("social_pressure_turnout", ParameterPack(
        pack_id="ggl_2008_michigan", family_id="social_pressure_turnout",
        domain="voter_turnout", population="Michigan registered voters, 2006 primary (180,002 households)",
        values={"levels": {"value": {"control": 0.297, "civic_duty": 0.315, "hawthorne": 0.322,
                                     "self": 0.345, "neighbors": 0.378},
                           "sd": 0.02, "source": "published_research",
                           "method": "randomized ITT turnout levels", "dataset": "Gerber-Green-Larimer 2008"}},
        fit_method="randomized field experiment ITT (verified)", time_scale="single election",
        citations=[Citation(ref="Gerber, Green & Larimer 2008 APSR 102:33-48",
                            doi_or_url="10.1017/S000305540808009X",
                            finding="Neighbors +8.1pp on 29.7% base", limits="low-salience primary")],
        validation=[ValidationRecord(
            kind="published_estimate", dataset="Gerber-Green-Larimer 2008 (randomized field experiment)",
            split="in-study randomized control vs treatment", metric="ITT_pp_Neighbors", value=0.081,
            baseline="no-mailer control 29.7%", passed=True, note="the STUDY's own randomized causal "
            "estimate; a published causal effect, NOT a local held-out validation of OUR transport")],
        transport_note="within-study randomized; transporting to another election requires the base-rate "
                       "remap + broad widening (effect shrinks toward high-salience ceilings)"))

    # ========================================================= E. MATCHING-DONATION (verified)
    s.register(MechanismRecord(
        family_id="matching_donation_response", version="1.0.0", ontology_type="participation",
        title="Matching-grant effect on donation probability",
        formal_description="P(donate|match)=P(donate)·(1+0.22); match RATIO (1:1/2:1/3:1) is FLAT (a "
                           "preserved null). Randomized natural field experiment.",
        causal_inputs=["base_donation_prob", "match_offered", "match_ratio"],
        causal_outputs=["donation_probability"], required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.behavioral:matching_donation_p",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("relative_lift", "match relative lift on P(donate)", 0.0, 0.5, "published_research")],
        applicability=ApplicabilityRule(domains=["fundraising", "charitable_giving", "donor_renewal"],
                                        answers_processes=["donation_after_ask"], transport_risk="high",
                                        exclusion_conditions=["cold_acquisition: effect measured on prior donors"]),
        citations=[Citation(ref="Karlan & List 2007, American Economic Review 97(5):1774-1793",
                            doi_or_url="10.1257/aer.97.5.1774",
                            study_population=">50,000 prior donors to a US nonprofit (natural field experiment)",
                            finding="match offer raises P(donate) by ≈22% (relative); match RATIO (2:1,3:1) "
                                    "has NO additional effect vs 1:1",
                            limits="prior donors (renewal), one org; do NOT scale by ratio; 22% is RELATIVE, "
                                   "not percentage points; cold acquisition unvalidated")],
        uncertainty_note="relative lift from the field experiment ± broad transport prior",
        known_failure_modes=["scaling the effect by match ratio (the study's key null forbids it)"],
        status="proposed", status_reason="registering"))
    s.add_pack("matching_donation_response", ParameterPack(
        pack_id="karlan_list_2007", family_id="matching_donation_response",
        domain="fundraising", population=">50,000 prior donors (US nonprofit)",
        values={"relative_lift": {"value": 0.22, "sd": 0.08, "lo": 0.05, "hi": 0.4,
                                  "source": "published_research", "method": "randomized field experiment",
                                  "dataset": "Karlan & List 2007"},
                "ratio_is_flat": {"value": True, "sd": None, "lo": None, "hi": None,
                                  "source": "published_research", "method": "randomized arms 1:1/2:1/3:1",
                                  "dataset": "Karlan & List 2007"}},
        fit_method="randomized natural field experiment (verified)", time_scale="single solicitation",
        citations=[Citation(ref="Karlan & List 2007 AER 97:1774-1793", doi_or_url="10.1257/aer.97.5.1774",
                            finding="+22% relative P(donate); ratio flat", limits="prior donors, one org")],
        validation=[ValidationRecord(
            kind="published_estimate", dataset="Karlan & List 2007 (randomized field experiment)",
            split="in-study randomized treatment vs control", metric="relative_lift_P_donate", value=0.22,
            baseline="no-match control", passed=True, note="the study's own randomized causal estimate; "
            "NOT a local held-out validation of OUR transport")],
        transport_note="prior-donor renewal mail; transport to cold acquisition or other causes unvalidated"))

    # ========================================================= F. REPUTATION UPDATING (Beta + Resnick)
    s.register(MechanismRecord(
        family_id="reputation_updating", version="1.0.0", ontology_type="relationship",
        title="Beta-Bernoulli reputation updating with price/behavior premium",
        formal_description="Beta(α,β) posterior over latent trustworthiness; each ±rating updates it "
                           "(image scoring). Reputation→behavior premium anchored to Resnick 2006 (≈8.1%).",
        causal_inputs=["rating_history"], causal_outputs=["reputation_score", "price_premium"],
        required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.behavioral:reputation_update",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("max_premium", "reputation price/behavior premium at strong rep",
                                  0.0, 0.2, "published_research")],
        applicability=ApplicabilityRule(domains=["online_marketplace", "reputation_system", "platform_exchange"],
                                        answers_processes=["reputation_update", "trust_change_after_interaction"],
                                        transport_risk="high",
                                        exclusion_conditions=["reciprocal_rating_bias: ratings inflated"]),
        citations=[Citation(ref="Resnick, Zeckhauser, Swanson & Lockwood 2006, Experimental Economics 9:79-101",
                            doi_or_url="10.1007/s10683-006-4309-2",
                            study_population="eBay randomized field experiment (matched seller identities)",
                            finding="an established good reputation commands ≈8.1% price premium (causal)",
                            limits="eBay auctions; reputation systems are gameable/biased; premium is "
                                   "category- and platform-specific"),
                   Citation(ref="Nowak & Sigmund 1998, Nature 393:573-577", doi_or_url="10.1038/31225",
                            finding="image scoring / indirect reciprocity sustains cooperation",
                            limits="theoretical model")],
        uncertainty_note="Beta posterior IS the uncertainty; premium ± broad transport prior",
        status="proposed", status_reason="registering"))
    s.add_pack("reputation_updating", ParameterPack(
        pack_id="ebay_resnick_2006", family_id="reputation_updating", domain="online_marketplace",
        population="eBay sellers (matched-identity field experiment)",
        values={"max_premium": {"value": 0.081, "sd": 0.03, "lo": 0.0, "hi": 0.15,
                                "source": "published_research", "method": "randomized field experiment",
                                "dataset": "Resnick et al. 2006"}},
        fit_method="randomized field experiment (verified)", time_scale="event",
        citations=[Citation(ref="Resnick et al. 2006 Exp Econ 9:79-101", doi_or_url="10.1007/s10683-006-4309-2",
                            finding="8.1% reputation premium", limits="eBay; gameable")],
        transport_note="eBay auctions; transport to other reputation systems unvalidated"))

    _register_hazard_families(s, fits)
    _register_content_response(s, fits)
    _register_research_encoded(s)
    _embed_higgs_coefficients(s)


def _register_hazard_families(s, fits):
    """Attrition (locally validated, transfer FAILS), response-occurrence + persuasion (honest NULLs)."""
    common_code = "swm.world_model_v2.registry.families.hazard:hazard_from_pack"

    # --- attrition / dropout hazard: telco (locally validated; transfer FAILS — preserved) ---
    tf = fits.get("telco_attrition", {})
    s.register(MechanismRecord(
        family_id="attrition_dropout_hazard", version="1.0.0", ontology_type="resource",
        title="Feature-conditioned attrition / dropout hazard",
        formal_description="P(dropout|x)=σ(b+Σ w_j z_j); a relationship/subscription ends. Executable via "
                           "FeatureHazardOperator. OBSERVATIONAL/predictive — NOT a general relationship-decay law.",
        causal_inputs=["actor_features"], causal_outputs=["dropout_event"], required_state=["entities"],
        temporal_scale="event", code_ref=common_code, test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("coefficients", "logistic hazard coefficients", default_source="fitted")],
        applicability=ApplicabilityRule(domains=["subscription_churn", "customer_attrition"],
                                        requires_state=["entities"], requires_data=["actor_features"],
                                        answers_processes=["relationship_dropout", "attrition"],
                                        transport_risk="high",
                                        exclusion_conditions=["social_relationship_decay: NOT identified here"]),
        citations=[Citation(ref="IBM Telco customer-churn dataset (public)",
                            study_population="telecom subscribers", finding="contract/tenure/billing predict "
                            "churn (observational)", limits="PREDICTIVE only; telecom subscription context; "
                            "not a causal or general relationship-decay mechanism")],
        uncertainty_note="parameter uncertainty via log-odds widening on transport",
        known_failure_modes=["cross-subpopulation transfer FAILS (month-to-month→long-contract), preserved"],
        status="proposed", status_reason="registering"))
    if tf:
        s.add_pack("attrition_dropout_hazard", ParameterPack(
            pack_id="telco_ibm_churn", family_id="attrition_dropout_hazard", domain="subscription_churn",
            population="IBM Telco customers (n=7032)", values=_hazard_pack_values(tf),
            fitted_on=f"{tf['path']} train split (60/20/20, seed 13)", fit_method="logistic MLE + L2",
            transport_note="telecom churn; cross-subpopulation transfer FAILED — domain-restricted",
            validation=[
                ValidationRecord(kind="held_out", dataset="IBM Telco churn", split="test 20% (disjoint), seed 13",
                                 metric="Brier", value=tf["result"]["test"]["brier"],
                                 baseline="base-rate", baseline_value=tf["result"]["baseline_base_rate"]["brier"],
                                 ci95=tf["result"]["paired_brier_model_minus_base"]["ci95"], passed=True,
                                 artifact=FITS, note=f"beats base rate; ECE {tf['result']['test']['calibration']['ece']}"),
                ValidationRecord(kind="transfer", dataset="IBM Telco churn",
                                 split=tf["transfer"]["direction"], metric="Brier_vs_target_base",
                                 value=tf["transfer"]["test_brier"], baseline="target-subpop base rate",
                                 baseline_value=tf["transfer"]["target_base_brier"],
                                 ci95=tf["transfer"]["paired_brier_model_minus_base"]["ci95"], passed=False,
                                 artifact=FITS, note="NEGATIVE TRANSFER preserved: month-to-month hazard does "
                                 "NOT beat the long-contract subpopulation base rate (blocks production)")]))

    # --- response-occurrence hazard: StackExchange (honest NULL preserved) ---
    re_ = fits.get("stackexchange_response", {})
    s.register(MechanismRecord(
        family_id="response_occurrence_hazard", version="1.0.0", ontology_type="participation",
        title="Feature-conditioned response-occurrence hazard",
        formal_description="P(response|x)=σ(b+Σ w_j z_j); a posted message receives a response. Executable. "
                           "OBSERVATIONAL — NOT trust/obligation/reciprocity.",
        causal_inputs=["message_features"], causal_outputs=["response_event"], required_state=["entities"],
        temporal_scale="event", code_ref=common_code, test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("coefficients", "logistic hazard coefficients", default_source="fitted")],
        applicability=ApplicabilityRule(domains=["qa_community", "message_response"],
                                        answers_processes=["response_after_message"], transport_risk="high",
                                        exclusion_conditions=["trust_or_obligation: NOT identified"]),
        citations=[Citation(ref="StackExchange questions (public sample)", finding="surface features weakly "
                            "predict answering", limits="PREDICTIVE; Q&A community; surface features do NOT "
                            "beat base rate on held-out — a preserved NULL")],
        known_failure_modes=["surface features do NOT beat base rate on held-out (null preserved)"],
        status="proposed", status_reason="registering"))
    if re_:
        s.add_pack("response_occurrence_hazard", ParameterPack(
            pack_id="stackexchange_answered", family_id="response_occurrence_hazard", domain="qa_community",
            population="StackExchange questions (n=2500)", values=_hazard_pack_values(re_),
            fitted_on=f"{re_['path']} train split", fit_method="logistic MLE + L2",
            transport_note="Q&A response; NULL result",
            validation=[ValidationRecord(
                kind="held_out", dataset="StackExchange", split="test 20% (disjoint), seed 13", metric="Brier",
                value=re_["result"]["test"]["brier"], baseline="base-rate",
                baseline_value=re_["result"]["baseline_base_rate"]["brier"],
                ci95=re_["result"]["paired_brier_model_minus_base"]["ci95"], passed=False, artifact=FITS,
                note="NULL: surface features do NOT beat base rate on held-out (preserved)")]))

    # --- argument-persuasion success: CMV (honest NULL preserved) ---
    cmv = fits.get("cmv_persuasion", {})
    s.register(MechanismRecord(
        family_id="argument_persuasion_success", version="1.0.0", ontology_type="belief",
        title="Feature-conditioned argument-persuasion success",
        formal_description="P(view_change|x)=σ(b+Σ w_j z_j); an argument earns a delta. Executable. "
                           "PREDICTIVE surface features on ONE platform; consistent with small persuasion effects.",
        causal_inputs=["argument_features"], causal_outputs=["persuasion_event"], required_state=["entities"],
        temporal_scale="event", code_ref=common_code, test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("coefficients", "logistic hazard coefficients", default_source="fitted")],
        applicability=ApplicabilityRule(domains=["online_debate", "argument_persuasion"],
                                        answers_processes=["persuasion_success"], transport_risk="high",
                                        exclusion_conditions=["political_persuasion: platform-specific"]),
        citations=[Citation(ref="Tan, Niculae, Danescu-Niculescu-Mizil & Lee 2016, WWW (CMV dataset)",
                            doi_or_url="10.1145/2872427.2883081",
                            study_population="r/ChangeMyView challengers (matched)", finding="surface argument "
                            "features weakly relate to earning a delta", limits="PREDICTIVE; platform-specific; "
                            "confounded by content; surface features do NOT beat base rate on held-out")],
        known_failure_modes=["surface features do NOT beat base rate on held-out (null; consistent with "
                             "small/hard persuasion effects, cf. Kalla & Broockman 2018)"],
        status="proposed", status_reason="registering"))
    if cmv:
        s.add_pack("argument_persuasion_success", ParameterPack(
            pack_id="cmv_delta", family_id="argument_persuasion_success", domain="online_debate",
            population="r/ChangeMyView challengers (n=1200)", values=_hazard_pack_values(cmv),
            fitted_on=f"{cmv['path']} train split", fit_method="logistic MLE + L2",
            transport_note="CMV persuasion; NULL result; platform-specific",
            citations=[Citation(ref="Tan et al. 2016 WWW", doi_or_url="10.1145/2872427.2883081",
                                finding="CMV persuasion", limits="platform-specific, confounded")],
            validation=[ValidationRecord(
                kind="held_out", dataset="CMV", split="test 20% (disjoint), seed 13", metric="Brier",
                value=cmv["result"]["test"]["brier"], baseline="base-rate",
                baseline_value=cmv["result"]["baseline_base_rate"]["brier"],
                ci95=cmv["result"]["paired_brier_model_minus_base"]["ci95"], passed=False, artifact=FITS,
                note="NULL: surface features barely differ from base rate (preserved; small persuasion effects)")]))


def _register_content_response(s, fits):
    """Upworthy content-response click ranking (locally validated, time-forward; LLM-free)."""
    up = fits.get("upworthy_content_response", {})
    s.register(MechanismRecord(
        family_id="content_response_click", version="1.0.0", ontology_type="attention",
        title="Content-response click ranking with audience heterogeneity",
        formal_description="surface features → fitted CTR layer → population-heterogeneity particle ranking "
                           "(each audience particle clicks its argmax; variant score = win share). RANDOMIZED "
                           "A/B traffic → the higher-CTR headline is CAUSALLY more clicked.",
        causal_inputs=["headline_surface_features"], causal_outputs=["click_ranking"],
        required_state=["populations"], temporal_scale="event",
        code_ref="swm.world_model_v2.reference.upworthy:population_rank",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("surface_w", "fitted CTR surface weights", default_source="fitted")],
        applicability=ApplicabilityRule(domains=["content_ab_test", "headline_optimization", "platform_attention"],
                                        population_kinds=["online_audience"], requires_state=["populations"],
                                        answers_processes=["content_response", "attention_after_exposure"],
                                        transport_risk="high",
                                        exclusion_conditions=["position_bias: rank NOT manipulated here"]),
        citations=[Citation(ref="Upworthy Research Archive (Matias, Munger, Le Quere, Ebersole 2021, Sci Data)",
                            doi_or_url="10.1038/s41597-021-00934-7",
                            study_population="Upworthy A/B headline tests (randomized traffic, 2013-2015)",
                            finding="randomized within-test traffic → CTR winner is causal for content response",
                            limits="clickbait-era Upworthy; identifies CONTENT response NOT position/examination "
                                   "bias; ~22% of tests had a caching-induced randomization failure (2024 "
                                   "correction) — content-response ordering signal is robust but caveated")],
        uncertainty_note="audience-heterogeneity particles (population plane); parameter widening on transport",
        known_failure_modes=["surface features alone (no population layer) ~random on P@1 — population is "
                             "load-bearing (ablation preserved)"],
        status="proposed", status_reason="registering"))
    if up and "result" in up:
        r = up["result"]
        ho = r["held_out"]["population"]
        tf = r["transfer"]["population"]
        nop = r["held_out"]["no_population_ablation"]
        s.add_pack("content_response_click", ParameterPack(
            pack_id="upworthy_archive_2021", family_id="content_response_click", domain="content_ab_test",
            population="Upworthy A/B headline tests (randomized traffic)",
            values={"surface_w": {"value": up["coefs"]["surface_w"], "sd": None, "lo": None, "hi": None,
                                  "source": "fitted", "method": "within-test CTR z-score linear fit",
                                  "dataset": up["path"],
                                  "features": up["coefs"]["surface_features"]}},
            fitted_on=f"{up['path']} (random 60/40 held-out + time-forward transfer)",
            fit_method="linear CTR + population-heterogeneity rank",
            citations=[Citation(ref="Upworthy Research Archive (Matias et al. 2021, Sci Data)",
                                doi_or_url="10.1038/s41597-021-00934-7",
                                finding="randomized A/B → causal CTR winner", limits="clickbait era; ~22% of "
                                "tests had a caching randomization failure (2024 correction)")],
            transport_note="Upworthy clickbait headlines; transport to other platforms unvalidated",
            validation=[
                ValidationRecord(kind="held_out", dataset="Upworthy Research Archive",
                                 split=r["held_out"]["split"], metric="pairwise_accuracy_vs_causal_CTR",
                                 value=ho["pairwise_accuracy"], baseline="random 0.5", passed=True,
                                 artifact=FITS, note=f"in-distribution held-out pairwise {ho['pairwise_accuracy']} "
                                 f"(P@1 {ho['precision_at_1']} vs random {ho['random_precision_at_1']}); "
                                 "randomized A/B → causal ground truth; LLM-free"),
                ValidationRecord(kind="transfer", dataset="Upworthy Research Archive",
                                 split=r["transfer"]["split"], metric="pairwise_accuracy_vs_causal_CTR",
                                 value=tf["pairwise_accuracy"], baseline="random 0.5", passed=True,
                                 artifact=FITS, note=f"OUT-OF-TIME transfer pairwise {tf['pairwise_accuracy']} "
                                 "(train earliest 60% → test latest 40%; headline styles drift)"),
                ValidationRecord(kind="ablation", dataset="Upworthy Research Archive", split=r["held_out"]["split"],
                                 metric="pairwise_population_vs_mean_audience", value=ho["pairwise_accuracy"],
                                 baseline="no-population (mean audience)",
                                 baseline_value=nop["pairwise_accuracy"], passed=True, artifact=FITS,
                                 note="population heterogeneity is load-bearing "
                                 f"({nop['pairwise_accuracy']}→{ho['pairwise_accuracy']})")]))


def _register_research_encoded(s):
    """Structural/directional families with VERIFIED citations but no numeric transition pack — Tier-4
    research records that expand coverage honestly (weak ties, targeting, altruistic punishment, persuasion
    guardrail). These are research_encoded: real research + formal model, NOT locally validated."""
    fams = [
        dict(fid="weak_tie_transmission", onto="network", title="Weak-tie information transmission (inverted-U)",
             form="transmission opportunity rises then falls with tie strength (inverted-U); moderately weak "
                  "ties maximize novel-information/mobility access.",
             code="swm.world_model_v2.registry.families.structural:weak_tie_transmission_shape",
             proc=["network_edge_change", "information_transmission"],
             cite=Citation(ref="Rajkumar, Saint-Jacques, Bojinov, Brynjolfsson & Aral 2022, Science 377:1304-1310",
                           doi_or_url="10.1126/science.abl4476",
                           study_population="LinkedIn PYMK randomized experiment, >20M people, ~600k jobs",
                           finding="INVERTED-U: moderately weak ties maximize job mobility (causal)",
                           limits="job mobility on LinkedIn; no single transportable coefficient; effect "
                                  "reverses in less-digital industries"),
             domains=["labor_market", "information_network"]),
        dict(fid="network_targeting_seeding", onto="network", title="Network targeting / friendship-nomination seeding",
             form="seed via friendship-nomination (friendship paradox: E[deg(friend)]=E[d²]/E[d]≥E[deg]) to "
                  "reach high-degree nodes; nomination > random; in-degree targeting ≈ random.",
             code="swm.world_model_v2.registry.families.structural:nomination_seed_expected_degree",
             proc=["network_targeting", "network_edge_change"],
             cite=Citation(ref="Kim et al. 2015, Lancet 386:145-153", doi_or_url="10.1016/S0140-6736(15)60095-2",
                           study_population="32 Honduras villages, 5,773 people (cluster RCT)",
                           finding="nomination targeting +12.2pp vs random (95% CI 6.9-17.9); in-degree "
                                   "targeting NOT better than random (a key null)",
                           limits="rural health intervention; magnitude context-specific; friendship paradox "
                                  "is structural"),
             domains=["public_health", "community_intervention"]),
        dict(fid="altruistic_punishment", onto="norm", title="Altruistic punishment sustaining cooperation",
             form="costly peer punishment of free-riders raises/sustains cooperation in public-goods games; "
                  "each punishment point cuts target payoff ~10% (magnitudes broad — scanned tables not "
                  "core-verified this run).",
             code="swm.world_model_v2.registry.families.structural:altruistic_punishment_cooperation",
             proc=["norm_enforcement", "actor_selects_typed_action"],
             cite=Citation(ref="Fehr & Gächter 2000, American Economic Review 90(4):980-994",
                           doi_or_url="10.1257/aer.90.4.980",
                           study_population="public-goods lab experiments (n=4 groups, MPCR 0.4, 10 periods)",
                           finding="with punishment, final contribution >18/20 tokens vs ~3/20 without "
                                   "(Partner); 2-4× higher (Stranger); Wilcoxon p<0.0001",
                           limits="lab public-goods; cooperation collapses without punishment; culture-varying "
                                  "(anti-social punishment elsewhere)"),
             domains=["public_goods", "collective_action"]),
        dict(fid="persuasion_minimal_effects", onto="belief", title="Minimal-persuasion-effects guardrail",
             form="average persuasive effect of campaign contact on vote choice ≈ 0 (tightly bounded); "
                  "nonzero only in narrow regimes (early, unfamiliar issues, or off-cycle).",
             proc=["persuasion_success", "belief_update_after_message"],
             cite=Citation(ref="Kalla & Broockman 2018, APSR 112(1):148-166",
                           doi_or_url="10.1017/S0003055417000363",
                           study_population="meta-analysis of 49 field experiments",
                           finding="best estimate of average general-election persuasion ≈ 0",
                           limits="general-election vote choice; NOT turnout; a guardrail against inventing "
                                  "large persuasion effects"),
             domains=["political_persuasion", "campaign"]),
    ]
    for f in fams:
        has_code = bool(f.get("code"))
        s.register(MechanismRecord(
            family_id=f["fid"], version="1.0.0", ontology_type=f["onto"], title=f["title"],
            formal_description=f["form"], causal_inputs=["structure"], causal_outputs=["effect"],
            required_state=["network" if f["onto"] == "network" else "entities"], temporal_scale="event",
            code_ref=f.get("code", ""), test_ref=("tests/test_wmv2_phase6.py" if has_code else ""),
            applicability=ApplicabilityRule(domains=f["domains"], answers_processes=f["proc"],
                                            transport_risk="high"),
            citations=[f["cite"]], uncertainty_note="verified published effect; structural FORM implemented "
            "where executable; magnitude carries broad uncertainty (no core-verified numeric pack this run)",
            status="proposed", status_reason="registering",
            implementation_note=("executable structural form; empirical magnitude + local validation are the "
                                 "remaining work" if has_code else
                                 "research_encoded: verified research + formal model stored; executable "
                                 "transition + numeric pack are the remaining work (see priority matrix)")))

    # ---- NEW: position-bias propensity (Joachims 2017 eq.7, η=1 CORE-VERIFIED) → domain_restricted ----
    s.register(MechanismRecord(
        family_id="position_bias_propensity", version="1.0.0", ontology_type="platform",
        title="Position-bias examination propensity (Joachims 2017)",
        formal_description="p(examine|rank)=(1/rank)^η (Joachims-Swaminathan-Schnabel 2017 WSDM eq.7); "
                           "P(click)=p(examine)·relevance. η=1 default (core-verified).",
        causal_inputs=["rank", "relevance"], causal_outputs=["click_probability"], required_state=["quantities"],
        temporal_scale="event", code_ref="swm.world_model_v2.registry.families.structural:position_bias_propensity",
        test_ref="tests/test_wmv2_phase6.py",
        parameters=[ParameterSpec("eta", "position-bias severity exponent", 0.0, 2.0, "published_research")],
        applicability=ApplicabilityRule(domains=["search_ranking", "ranked_feed", "learning_to_rank"],
                                        requires_state=["quantities"],
                                        answers_processes=["examination_by_rank", "attention_after_exposure"],
                                        transport_risk="high",
                                        exclusion_conditions=["non_ranked_feed: no position to debias",
                                                              "click_as_relevance: needs IPS correction"]),
        citations=[Citation(ref="Joachims, Swaminathan & Schnabel 2017, WSDM pp.781-789",
                            doi_or_url="10.1145/3018661.3018699",
                            study_population="Arxiv Full-Text Search + semi-synthetic click logs",
                            finding="examination propensity (1/rank)^η; η=1 default; real-world propensities "
                                    "decay to ≈0.12 by rank ~21; underestimating small propensities is harmful",
                            limits="ranked search/feed; click≠relevance without inverse-propensity weighting")],
        uncertainty_note="η prior around the verified default 1.0", status="proposed", status_reason="registering"))
    s.add_pack("position_bias_propensity", ParameterPack(
        pack_id="joachims_2017_eta1", family_id="position_bias_propensity", domain="search_ranking",
        population="ranked search/feed click logs",
        values={"eta": {"value": 1.0, "sd": 0.3, "lo": 0.5, "hi": 1.5, "source": "published_research",
                        "method": "default/estimated severity exponent (eq.7)", "dataset": "Joachims 2017"}},
        fit_method="published (core-verified from primary PDF eq.7)", time_scale="event",
        citations=[Citation(ref="Joachims et al. 2017 WSDM", doi_or_url="10.1145/3018661.3018699",
                            finding="(1/rank)^η, η=1", limits="ranked feed; platform-specific severity")],
        transport_note="severity η is platform-specific; refit per platform where randomized swaps exist"))

    # ---- NEW: Gamson coalition payoff (structural proportionality) → software_implemented ----
    s.register(MechanismRecord(
        family_id="coalition_payoff_gamson", version="1.0.0", ontology_type="coalition",
        title="Gamson's law coalition portfolio share",
        formal_description="portfolio share ≈ seat contribution share (slope≈1, R²≈0.9; small-party bonus). "
                           "Structural proportionality; slope/intercept research-encoded (not core-verified).",
        causal_inputs=["seat_share"], causal_outputs=["portfolio_share"], required_state=["quantities"],
        temporal_scale="event", code_ref="swm.world_model_v2.registry.families.structural:coalition_payoff_gamson",
        test_ref="tests/test_wmv2_phase6.py",
        applicability=ApplicabilityRule(domains=["coalition_government", "legislature"],
                                        answers_processes=["coalition_payoff", "coalition_formation"],
                                        transport_risk="high"),
        citations=[Citation(ref="Gamson 1961 (ASR); Browne & Franklin 1973 (APSR 67:453-469)",
                            finding="portfolio share ≈ seat share (R²≈0.9)",
                            limits="parliamentary coalitions; slope≈1 with a small-party bonus; not core-"
                                   "verified this run")],
        uncertainty_note="structural proportionality; slope broad", status="proposed", status_reason="registering"))


def _embed_higgs_coefficients(s):
    """Upgrade the Higgs exposure-response pack from a 'see artifact' pointer to the REAL fitted numbers."""
    path = "experiments/results/wmv2_higgs_nonlinear.json"
    if not os.path.exists(path) or "exposure_response_hazard" not in s.records:
        return
    art = json.load(open(path))
    theta = art.get("fitted", {}).get("loglinear_theta", {}).get("theta")
    rec = s.records["exposure_response_hazard"]
    for p in rec.packs:
        if p.pack_id == "higgs_2012_rumor" and theta:
            p.values["theta"]["value"] = [round(t, 5) for t in theta]
            p.values["theta"]["feature_map"] = ["1", "log1p(k)", "log1p(deg)", "k/deg", "exp(-recency/24)"]
            p.values["theta"]["note"] = "REAL fitted GLM coefficients (was a pointer); see artifact for CI"
