"""Build the committed machine-readable mechanism registry (Phase 6 deliverable 14/15).

Registers every mechanism family with: executable code_ref (checked to resolve), formal description,
citations WITH transport limits, parameter specs, applicability rules, and REAL validation records
pointing to committed result artifacts. Statuses are earned by the promotion gates — a family only
reaches production_eligible with a passed held-out/transfer record; families whose validation is a
recorded FAILURE (Hawkes) are quarantined with the failure preserved.

Run: PYTHONPATH=. python -m swm.world_model_v2.registry.build_registry
Writes swm/world_model_v2/registry/data/{registry,packs}.json (integrity-hashed).

HONESTY CONTRACT: no family is promoted past its evidence. The ~40-family target is met on the CODE
plane (executable transitions + tests) but the docs and this script mark exactly which families are
empirically validated vs implemented-only vs experimental. See docs/WMV2_MECHANISM_REGISTRY.md.
"""
from __future__ import annotations

from swm.world_model_v2.registry.record import (ApplicabilityRule, Citation, MechanismRecord,
                                                ParameterPack, ParameterSpec, ValidationRecord)
from swm.world_model_v2.registry.store import RegistryStore


def _p(name, desc, lo=None, hi=None, src="reference_class_prior"):
    return ParameterSpec(name=name, description=desc, lo=lo, hi=hi, default_source=src)


def build() -> RegistryStore:
    s = RegistryStore()

    # ============================================================ DIFFUSION (validated on Higgs)
    s.register(MechanismRecord(
        family_id="exposure_response_hazard", version="1.0.0", ontology_type="diffusion",
        title="Nonlinear log-linear exposure-response activation hazard",
        formal_description="λ_i(t)=exp(θ·x), x=[1,log1p(k_i),log1p(deg_i),k_i/deg_i,exp(-recency/24)]; "
                           "window activation P=1-exp(-∫λ dt), integrated event-by-event over the cohort "
                           "subgraph with in-sample exposure growth (rollout).",
        causal_inputs=["active_followee_count", "out_degree", "exposure_recency"],
        causal_outputs=["activation_event"], required_state=["network", "entities"],
        temporal_scale="hours", code_ref="swm.world_model_v2.registry.families.diffusion:contagion_window_predict",
        test_ref="tests/test_diffusion_families.py",
        parameters=[_p("theta", "hazard GLM coefficients", src="fitted")],
        applicability=ApplicabilityRule(domains=["social_media_diffusion", "information_diffusion"],
                                        requires_state=["network"], requires_data=["activity_log"],
                                        time_scales=["hours", "days"], population_kinds=["online_social"],
                                        transport_risk="high",
                                        exclusion_conditions=["no_follower_graph: needs a network"]),
        citations=[Citation(ref="Romero, Meeder & Kleinberg 2011 (WWW)", finding="exposure-response is "
                            "concave and hashtag-dependent; not linear complex contagion universally",
                            study_population="Twitter hashtags 2009", limits="Twitter only; hashtag "
                            "adoption, not general behavior")],
        uncertainty_note="per-particle lognormal hazard-scale draw (parameter uncertainty); frailty sd "
                         "fitted by profile likelihood",
        known_failure_modes=["rollout adds nothing when in-sample subgraph is sparse (honest scope)"],
        status="proposed", status_reason="registering"))
    s.add_pack("exposure_response_hazard", ParameterPack(
        pack_id="higgs_2012_rumor", family_id="exposure_response_hazard",
        domain="social_media_diffusion", population="Twitter users, 2012 Higgs-boson rumor window",
        values={"theta": {"source": "fitted", "method": "Bernoulli-hazard GLM, cloglog link",
                          "dataset": "SNAP higgs-twitter, time-forward train cohort", "sd": None,
                          "lo": None, "hi": None, "value": "see wmv2_higgs_nonlinear.json"}},
        fitted_on="SNAP higgs-twitter train cohort (t0+24h, seed 13)", fit_method="Bernoulli-hazard GLM",
        time_scale="24h window",
        transport_note="fitted per-cascade; transporting to another cascade needs refit + 2.5x sd widening",
        validation=[
            ValidationRecord(kind="held_out", dataset="SNAP higgs-twitter",
                             split="time-forward test cohort t0+48h, n=4000, seed 17",
                             metric="Brier_paired_vs_fitted_logistic", value=-0.000192,
                             ci95=[-0.000525, 0.000145], baseline="fitted exposure logistic (H1)",
                             passed=True, artifact="experiments/results/wmv2_higgs_nonlinear.json",
                             note="closes the gap to the fitted logistic (prior LINEAR world was "
                                  "significantly WORSE at +0.00234); CI straddles 0 = statistically tied"),
            ValidationRecord(kind="ablation", dataset="SNAP higgs-twitter", split="test n=4000",
                             metric="Brier_paired_nonlinear_vs_linear", value=-0.00253,
                             ci95=[-0.003411, -0.001694], baseline="linear q·k hazard (M1)", passed=True,
                             artifact="experiments/results/wmv2_higgs_nonlinear.json",
                             note="nonlinearity is the load-bearing improvement, CI excludes 0")]))

    for fam, form, cite in [
        ("simple_contagion_hazard", "λ=q·k (independent per-exposure transmission)",
         Citation(ref="classic SI/threshold literature", finding="linear-in-exposure baseline",
                  limits="known too simple for social reinforcement — kept as comparator")),
        ("complex_contagion_hazard", "λ=exp(θ0)·k^α/(c^α+k^α) (Hill; α>1 = social reinforcement, saturating)",
         Citation(ref="Centola & Macy 2007 (AJS)", finding="complex contagion needs multiple exposures",
                  study_population="online health-behavior experiment", limits="experimental; α,c "
                  "must be fit per setting")),
    ]:
        s.register(MechanismRecord(
            family_id=fam, version="1.0.0", ontology_type="diffusion", title=form.split("(")[0].strip(),
            formal_description=form, causal_inputs=["active_followee_count"],
            causal_outputs=["activation_event"], required_state=["network"], temporal_scale="hours",
            code_ref="swm.world_model_v2.registry.families.diffusion:contagion_window_predict",
            test_ref="tests/test_diffusion_families.py",
            applicability=ApplicabilityRule(domains=["social_media_diffusion"], requires_state=["network"],
                                            requires_data=["activity_log"], transport_risk="high"),
            citations=[cite], status="proposed", status_reason="registering"))
        s.add_pack(fam, ParameterPack(
            pack_id=f"higgs_2012_{fam}", family_id=fam, domain="social_media_diffusion",
            population="Twitter 2012 Higgs window",
            values={"param": {"source": "fitted", "sd": None, "lo": None, "hi": None,
                              "value": "see wmv2_higgs_nonlinear.json", "method": "grid+profile"}},
            fitted_on="SNAP higgs train", fit_method="profile likelihood",
            validation=[ValidationRecord(
                kind="ablation", dataset="SNAP higgs-twitter", split="test n=4000",
                metric="Brier", value=None, baseline="log-linear hazard",
                passed=(fam == "complex_contagion_hazard"),
                artifact="experiments/results/wmv2_higgs_nonlinear.json",
                note="Hill beat linear but log-linear won validation selection")]))

    s.register(MechanismRecord(
        family_id="hawkes_self_excitation", version="1.0.0", ontology_type="diffusion",
        title="Hawkes self-exciting point process",
        formal_description="λ(t)=μ+αω·Σ_{t_i<t} exp(-ω(t-t_i)); exponential kernel, EM/MLE fit",
        causal_inputs=["event_history"], causal_outputs=["event_intensity"], required_state=["quantities"],
        temporal_scale="minutes", code_ref="swm.world_model_v2.registry.families.diffusion:fit_hawkes",
        test_ref="tests/test_diffusion_families.py",
        citations=[Citation(ref="Hawkes 1971; Zhao et al. 2015 SEISMIC",
                            finding="retweet cascades are self-exciting",
                            limits="constant background + single exponential kernel underfit bursty streams")],
        known_failure_modes=["constant μ + single exponential kernel underfit the Higgs activity burst"],
        status="proposed", status_reason="registering"))
    s.add_pack("hawkes_self_excitation", ParameterPack(
        pack_id="higgs_2012_stream", family_id="hawkes_self_excitation", domain="social_media_diffusion",
        population="Twitter 2012 Higgs activity stream",
        values={"mu": {"source": "fitted", "sd": None, "lo": None, "hi": None, "value": "see artifact"},
                "alpha": {"source": "fitted", "sd": None, "lo": None, "hi": None, "value": "see artifact"}},
        fitted_on="Higgs stream 0-48h", fit_method="EM/MLE exponential kernel",
        validation=[ValidationRecord(
            kind="held_out", dataset="SNAP higgs activity stream", split="held-out 24-72h counts",
            metric="MAE_per_bin", value=1098.9, baseline="Poisson base rate", baseline_value=973.0,
            passed=False, artifact="experiments/results/wmv2_higgs_nonlinear.json",
            note="FAILED: Hawkes MAE 1098.9 > Poisson 973.0 — the fit underpredicted the burst "
                 "(preserved negative result)")]))
    s.set_status("hawkes_self_excitation", "quarantined",
                 reason="held-out count forecast FAILED to beat Poisson on the Higgs stream; preserved")

    # ============================================================ CHOICE (validated on BehaviorBench)
    s.register(MechanismRecord(
        family_id="quantal_response_choice", version="1.0.0", ontology_type="decision",
        title="Quantal response (logit) choice over expected utilities",
        formal_description="p_a ∝ exp(λ·u_a); λ precision fitted on payoff scale; λ=0 uniform, λ→∞ best reply",
        causal_inputs=["action_utilities"], causal_outputs=["action_distribution"],
        required_state=["entities"], temporal_scale="event",
        code_ref="swm.world_model_v2.policy:logit_choice", test_ref="tests/test_policy_families.py",
        parameters=[_p("lambda", "logit precision on payoff scale", lo=0.0, src="fitted")],
        applicability=ApplicabilityRule(domains=["negotiation", "economic_game", "decision"],
                                        transport_risk="medium"),
        citations=[Citation(ref="McKelvey & Palfrey 1995 (GEB)", finding="logit QRE fits lab game data",
                            study_population="lab games", limits="λ is payoff-scale-dependent; refit per game")],
        status="proposed", status_reason="registering"))
    s.register(MechanismRecord(
        family_id="social_preference_population", version="1.0.0", ontology_type="decision",
        title="Fehr-Schmidt inequity-aversion population mixture",
        formal_description="u=x_own-α·max(x_other-x_own,0)-β·max(x_own-x_other,0); discrete (α,β) type "
                           "mixture, weights fitted jointly across games (partial pooling); drives every "
                           "two-player game through one utility model; cross-game interaction via shared "
                           "responder-threshold / banker-return beliefs.",
        causal_inputs=["payoff_structure", "partner_beliefs"], causal_outputs=["action_distribution"],
        required_state=["entities", "populations"], temporal_scale="event",
        code_ref="swm.world_model_v2.registry.families.choice:fit_game_policy",
        test_ref="tests/test_policy_families.py",
        parameters=[_p("type_weights", "FS (α,β) mixture weights", src="fitted"),
                    _p("lambda_per_game", "QRE precision", src="fitted")],
        applicability=ApplicabilityRule(domains=["economic_game", "negotiation"], population_kinds=["lab"],
                                        transport_risk="high"),
        citations=[Citation(ref="Fehr & Schmidt 1999 (QJE 114:817)", finding="inequity aversion "
                            "calibrated type distribution", study_population="1990s lab ultimatum, students",
                            limits="NOT a universal law; comonotonic α,β assumption; lab stakes")],
        known_failure_modes=["public_goods badly misfit (W1 0.292, KS 0.458) — FS→PG mapping too crude",
                             "does not beat per-game fitted histogram in-distribution"],
        uncertainty_note="per-particle type draw from the fitted mixture",
        status="proposed", status_reason="registering"))
    s.add_pack("social_preference_population", ParameterPack(
        pack_id="behaviorbench_moblab", family_id="social_preference_population", domain="economic_game",
        population="MobLab economic-game players (~200/game)",
        values={"type_weights": {"source": "fitted", "sd": None, "lo": None, "hi": None,
                                 "value": "see wmv2_behaviorbench_policy.json", "method": "EG/W1 joint"}},
        fitted_on="BehaviorBench train (50/50 split seed 13)", fit_method="exponentiated-gradient on W1",
        transport_note="lab economic games; transport to field behavior unvalidated",
        validation=[
            ValidationRecord(kind="held_out", dataset="BehaviorBench moblab", split="test 50%, 7 games",
                             metric="mean_W1_norm", value=0.0988, baseline="train histogram (A1)",
                             baseline_value=0.0376, passed=False,
                             artifact="experiments/results/wmv2_behaviorbench_policy.json",
                             note="beats raw LLM (0.185) + elicitation (0.123) but LOSES to per-game "
                                  "histogram (0.038) and hand-crafted V2 (0.058) in-distribution"),
            ValidationRecord(kind="transfer", dataset="BehaviorBench moblab",
                             split="leave-one-game-out, per game",
                             metric="W1_norm_vs_pooled_transfer_baseline", value=None, passed=True,
                             artifact="experiments/results/wmv2_behaviorbench_policy.json",
                             note="LOGO beats pooled-histogram transfer on dictator/proposer/investor "
                                  "(CI excludes 0); FAILS on public_goods (+0.328)")]))

    for fam, title, form, cite in [
        ("reinforcement_learning", "Q-learning value update", "Q←Q+α(r−Q)",
         Citation(ref="Watkins 1989; Erev & Roth 1998 (AER)", finding="reinforcement learning tracks "
                  "human game learning", study_population="repeated lab games", limits="α must be fit")),
        ("belief_learning", "Fictitious-play belief learning", "beliefs←normalized action counts",
         Citation(ref="Fudenberg & Levine 1998", finding="belief learning in games",
                  limits="assumes stationary opponent")),
        ("experience_weighted_attraction", "EWA (Camerer-Ho)", "A←(φNA+[1 or δ]·payoff)/N'; N'=ρN+1",
         Citation(ref="Camerer & Ho 1999 (Econometrica 67:827)", finding="EWA nests reinforcement + "
                  "belief learning", study_population="lab games", limits="4 free params; identification "
                  "needs many rounds")),
        ("habit_formation", "Habit-stock accumulation", "H←(1−γ)H+γ·1[acted]; boosts repeat probability",
         Citation(ref="Wood & Neal 2007 (Psych Review)", finding="habit as automaticity from repetition",
                  limits="γ context-dependent")),
    ]:
        s.register(MechanismRecord(
            family_id=fam, version="1.0.0", ontology_type="learning", title=title,
            formal_description=form, causal_inputs=["outcome", "prior_latent"],
            causal_outputs=["updated_latent"], required_state=["entities"], temporal_scale="event",
            code_ref={"reinforcement_learning": "swm.world_model_v2.registry.families.learning:reinforcement_update",
                      "belief_learning": "swm.world_model_v2.registry.families.learning:belief_update_counts",
                      "experience_weighted_attraction": "swm.world_model_v2.registry.families.learning:EWAState",
                      "habit_formation": "swm.world_model_v2.registry.families.learning:habit_update"}[fam],
            test_ref="tests/test_policy_families.py", citations=[cite],
            applicability=ApplicabilityRule(domains=["repeated_game", "decision"], transport_risk="high"),
            status="proposed", status_reason="registering"))

    # ============================================================ SOCIAL / INFLUENCE (implemented, candidate)
    for fam, title, form, ref, code in [
        ("trust_formation", "Asymmetric trust gain", "trust←trust+gain·(1−trust) on cooperation",
         "Slovic 1993 (Risk Analysis)", "trust_update"),
        ("trust_violation", "Steep trust loss", "trust←trust−loss·trust on defection (loss>gain)",
         "Slovic 1993 asymmetry principle", "trust_update"),
        ("trust_repair", "Partial trust repair", "trust←trust+repair·(1−trust) after remediation",
         "Kim et al. 2004 (JAP)", "trust_update"),
        ("reciprocity", "Direct reciprocity edge update", "edge←edge+rate·(other_kindness−baseline)",
         "Fehr & Gächter 2000 (AER)", "reciprocity_update"),
        ("degroot_influence", "DeGroot naive consensus", "o_i←Σ_j w_ij o_j (row-stochastic)",
         "DeGroot 1974 (JASA); Chandrasekhar et al. 2020", "degroot_step"),
        ("bounded_confidence", "Hegselmann-Krause opinion dynamics", "average opinions within ε only",
         "Hegselmann & Krause 2002 (JASSS)", "bounded_confidence_step"),
        ("threshold_adoption", "Granovetter threshold cascade", "adopt iff active-neighbor frac ≥ threshold",
         "Granovetter 1978 (AJS)", "threshold_adopt"),
        ("latent_expressed_opinion", "Preference falsification", "expressed=latent unless pressure>conviction",
         "Kuran 1995 (Private Truths)", "expressed_opinion"),
    ]:
        s.register(MechanismRecord(
            family_id=fam, version="1.0.0",
            ontology_type=("relationship" if "trust" in fam or fam == "reciprocity" else "influence"),
            title=title, formal_description=form, causal_inputs=["interaction_outcome"],
            causal_outputs=["updated_edge_or_opinion"],
            required_state=(["network"] if "trust" in fam or fam == "reciprocity" else ["entities", "network"]),
            temporal_scale="event",
            code_ref=f"swm.world_model_v2.registry.families.social:{code}",
            test_ref="tests/test_social_families.py",
            citations=[Citation(ref=ref, finding=title, limits="candidate structural family — parameters "
                                "and applicability must be validated per setting; NOT a universal law")],
            applicability=ApplicabilityRule(
                domains=(["negotiation", "coalition", "organizational"] if "trust" in fam or fam == "reciprocity"
                         else ["election", "social_media_diffusion", "protest"]), transport_risk="high"),
            status="proposed", status_reason="registering"))

    # ============================================================ INSTITUTIONAL / MEASUREMENT / EXOGENOUS
    #  (executable operators already in transitions.py — registered as families that mirror them)
    for fam, otype, title, code, tref in [
        ("institutional_vote", "institutional", "Deterministic vote execution",
         "swm.world_model_v2.transitions:InstitutionalVoteOperator", "tests/test_world_model_v2.py"),
        ("belief_update_exposure", "belief", "Bounded Bayesian-ish belief shift on exposure",
         "swm.world_model_v2.transitions:BeliefUpdateOperator", "tests/test_world_model_v2.py"),
        ("relationship_update_bounded", "relationship", "Bounded typed edge strength/trust shift",
         "swm.world_model_v2.transitions:RelationshipUpdateOperator", "tests/test_world_model_v2.py"),
        ("resource_depletion", "resource", "Conservation-checked resource consumption",
         "swm.world_model_v2.transitions:ResourceUpdateOperator", "tests/test_world_model_v2.py"),
        ("attention_dynamics", "attention", "Mean-reverting attention over elapsed time",
         "swm.world_model_v2.transitions:BackgroundDynamicsOperator", "tests/test_world_model_v2.py"),
        ("rare_event_arrival", "exogenous", "Poisson rare-event arrival by deadline",
         "swm.world_model_v2.transitions:RareEventArrivalOperator", "tests/test_wmv2_tier_a_fixes.py"),
        ("memory_decay", "memory", "Exponential exposure-salience decay",
         "swm.world_model_v2.information:InformationLedger", "tests/test_world_model_v2.py"),
        ("gaussian_measurement", "measurement", "Gaussian measurement with bias/missingness/delay",
         "swm.world_model_v2.observation:GaussianMeasurement", "tests/test_world_model_v2.py"),
        ("bernoulli_detection", "observation", "Bernoulli detection observation model",
         "swm.world_model_v2.observation:BernoulliDetection", "tests/test_world_model_v2.py"),
        ("attention_allocation", "attention", "Interpretation-driven attention allocation",
         "swm.world_model_v2.actor_cognition:attention_transition", "tests/test_actor_cognition.py"),
        ("information_interpretation", "interpretation", "Structured typed reading of an incoming item",
         "swm.world_model_v2.actor_cognition:interpret", "tests/test_actor_cognition.py"),
        ("relationship_strength_inference", "relationship", "Interaction-history edge strength",
         "swm.world_model_v2.actor_cognition:relationship_strength", "tests/test_actor_cognition.py"),
        ("observation_exposure", "observation", "Actor-specific exposure / information visibility",
         "swm.world_model_v2.transitions:observable_view", "tests/test_world_model_v2.py"),
        ("finite_population_saturation", "diffusion", "Finite susceptible-population cascade saturation",
         "swm.world_model_v2.registry.families.diffusion:closed_form_window_p", "tests/test_diffusion_families.py"),
        ("susceptibility_frailty", "diffusion", "Lognormal susceptibility heterogeneity (fitted sd)",
         "swm.world_model_v2.registry.families.diffusion:fit_frailty_sigma", "tests/test_diffusion_families.py"),
        ("information_aging", "diffusion", "Age-weighted exposure decay (fitted half-life)",
         "swm.world_model_v2.registry.families.diffusion:fit_aging_tau", "tests/test_diffusion_families.py"),
        ("engagement_momentum_persistence", "memory", "Momentum/burstiness persistence state",
         "swm.world_model_v2.reference.omnibehavior:momentum_state", "experiments/wmv2_persistence_power.py"),
        ("hierarchical_rate_shrinkage", "measurement", "Empirical-Bayes partial-pooling rate posterior",
         "swm.world_model_v2.inference_layer:hierarchical_rates", "tests/test_inference_layer.py"),
        ("norm_compliance", "norm", "Obligation/norm-driven action utility shift",
         "swm.world_model_v2.actor_cognition:relationship_modulator", "tests/test_actor_cognition.py"),
    ]:
        s.register(MechanismRecord(
            family_id=fam, version="1.0.0", ontology_type=otype, title=title, formal_description=title,
            causal_inputs=["state"], causal_outputs=["state_delta"], required_state=["entities"],
            temporal_scale="event", code_ref=code, test_ref=tref,
            citations=[Citation(ref="see module docstring", finding=title, limits="parameters are labeled "
                                "priors / fitted where data exists; see record")],
            applicability=ApplicabilityRule(domains=["*"], transport_risk="high"),
            status="proposed", status_reason="registering"))

    # ============================================================ BARGAINING / COALITION / PARTICIPATION / PLATFORM
    for fam, otype, title, form, ref, code in [
        ("bargaining_rubinstein", "bargaining", "Rubinstein alternating-offers split",
         "x*=(1−δ_b)/(1−δ_a δ_b)", "Rubinstein 1982 (Econometrica 50:97)",
         "swm.world_model_v2.registry.families.interaction:rubinstein_split"),
        ("negotiation_concession", "bargaining", "Time-dependent concession tactic",
         "offer decays initial→reservation as deadline nears (β tough/conceder)",
         "Faratin, Sierra & Jennings 1998 (RAS)",
         "swm.world_model_v2.registry.families.interaction:concession_offer"),
        ("coalition_formation", "coalition", "Minimal-winning coalition + Banzhaf power",
         "greedy MWC by support; Banzhaf pivotality index", "Banzhaf 1965; Riker 1962",
         "swm.world_model_v2.registry.families.interaction:banzhaf_power"),
        ("voting_turnout", "participation", "Calculus-of-voting turnout",
         "p=σ(logit(base)−k_c·cost+k_b·benefit+k_d·duty+k_m·mobilized)",
         "Riker & Ordeshook 1968 (APSR 62:25)",
         "swm.world_model_v2.registry.families.interaction:turnout_probability"),
        ("mobilization", "participation", "Mobilization boost to participation",
         "mobilized contact raises turnout logit", "Gerber & Green 2000 (APSR 94:653)",
         "swm.world_model_v2.registry.families.interaction:turnout_probability"),
        ("donation_response", "participation", "Bounded donation response to an ask",
         "fraction of capacity ↑ in ask/affinity/history, saturating",
         "reference-class fundraising form",
         "swm.world_model_v2.registry.families.interaction:donation_amount"),
        ("platform_examination", "platform", "Position-bias examination (cascade model)",
         "P(examine|rank)=γ^rank", "Craswell et al. 2008 (WSDM)",
         "swm.world_model_v2.registry.families.interaction:position_examination"),
        ("platform_ranking", "platform", "Platform ranking / allocation by score",
         "order items by score desc → rank", "learning-to-rank / feed-allocation practice",
         "swm.world_model_v2.registry.families.interaction:rank_by_score"),
        ("network_rewiring", "network", "Edge co-evolution (homophily + tie-age decay)",
         "P(edge)=σ(w·(homophily−.5)+log age_decay)", "Snijders SAOM 2010",
         "swm.world_model_v2.registry.families.interaction:rewire_probability"),
        ("agenda_stage_control", "institutional", "Agenda / stage-transition control",
         "procedure rule gates actions by stage variable", "institutional procedure (see institutions.py)",
         "swm.world_model_v2.institutions:Rule"),
    ]:
        s.register(MechanismRecord(
            family_id=fam, version="1.0.0", ontology_type=otype, title=title, formal_description=form,
            causal_inputs=["actor_state"], causal_outputs=["action_or_edge"],
            required_state=(["institutions"] if otype == "institutional" else ["entities"]),
            temporal_scale="event", code_ref=code, test_ref="tests/test_interaction_families.py",
            citations=[Citation(ref=ref, finding=title, limits="candidate structural family; parameters "
                                "labeled priors, must be fit/validated per setting")],
            applicability=ApplicabilityRule(
                domains={"bargaining": ["negotiation"], "coalition": ["coalition", "legislation"],
                         "participation": ["election", "fundraising", "protest"],
                         "platform": ["social_media_diffusion", "product_launch"],
                         "network": ["social_media_diffusion"],
                         "institutional": ["legislation", "organizational"]}.get(otype, ["*"]),
                transport_risk="high"),
            status="proposed", status_reason="registering"))

    # persistence: adequately-powered held-out win + person-disjoint transfer (OmniBehavior n=7074)
    s.add_pack("engagement_momentum_persistence", ParameterPack(
        pack_id="omnibehavior_kuaishou", family_id="engagement_momentum_persistence",
        domain="platform_engagement", population="Kuaishou short-video/e-commerce users, 90-day traces",
        values={"momentum_lift": {"source": "fitted", "value": 6.777, "sd": None, "lo": None, "hi": None,
                                  "method": "train p_hot/p_cold ratio"}},
        fitted_on="OmniBehavior train (per-user 70% chronological prefix)",
        fit_method="momentum ratio + hierarchical user-rate shrinkage",
        transport_note="short-video engagement; transport to deliberation domains unvalidated",
        validation=[
            ValidationRecord(kind="held_out", dataset="OmniBehavior (Kuaishou)",
                             split="per-user time-forward test, n=7074 passive-exposure events",
                             metric="Brier_paired_vs_memoryless_userrate", value=-0.006496,
                             ci95=[-0.009211, -0.003483], baseline="hierarchical user-rate (no persistence)",
                             passed=True, artifact="experiments/results/wmv2_persistence_power.json",
                             note="adequately powered (0.993 at empirical paired sd); CI excludes 0 — "
                                  "REVERSES the prior n=48 null"),
            ValidationRecord(kind="transfer", dataset="OmniBehavior (Kuaishou)",
                             split="person-disjoint (14 users never in train), n=216",
                             metric="Brier_paired_vs_userrate", value=-0.027127,
                             ci95=[-0.032386, -0.02117], passed=True,
                             artifact="experiments/results/wmv2_persistence_power.json",
                             note="persistence transfers to held-out PEOPLE")]))

    # ---- Phase 6: register evidence-backed families/packs + causal-process declarations ----
    from swm.world_model_v2.registry.phase6_build import register_phase6
    register_phase6(s)

    # ---- promotion pass: promote families with real passed held-out/transfer records ----
    _promote(s)
    _promote_phase6(s)
    return s


def _promote_phase6(s: RegistryStore):
    """Phase-6 lifecycle placement (enforced gates decide the ceiling; these only REQUEST a target):
      * verified published-estimate families → domain_restricted (Tier-4; NOT locally validated);
      * structural/directional research records → research_encoded;
      * telco attrition stays locally_validated (passed held-out; FAILED transfer blocks production);
      * Upworthy content-response → production_eligible (passed in-distribution held-out + out-of-time
        transfer + citation); the nulls (StackExchange, CMV) stay implemented (their held-out FAILED)."""
    # attrition_dropout_hazard is deliberately NOT here: it genuinely earned locally_validated (passed
    # held-out) — its FAILED cross-subpopulation transfer blocks PRODUCTION but not local validation.
    domain_restricted = ["bass_diffusion", "ultimatum_offer_response", "trust_game_transfer",
                         "social_pressure_turnout", "matching_donation_response", "reputation_updating"]
    research_encoded = ["weak_tie_transmission", "network_targeting_seeding", "altruistic_punishment",
                        "persuasion_minimal_effects"]
    for fam in domain_restricted:
        if fam in s.records:
            try:
                s.set_status(fam, "domain_restricted",
                             reason="verified published estimate / local fit valid in declared domain only")
            except Exception:
                pass                                          # gate refused — keep earned status
    for fam in research_encoded:
        if fam in s.records:
            try:
                s.set_status(fam, "research_encoded",
                             reason="verified research + formal model; executable numeric pack is remaining work")
            except Exception:
                pass


def _promote(s: RegistryStore):
    """Apply the enforced lifecycle gates. Only families with executable code + tests + a PASSED
    held-out/transfer record reach production_eligible; the rest settle at implemented (executable+tested)
    or stay proposed. Hawkes stays quarantined (failure preserved)."""
    for fam in list(s.records):
        rec = s.records[fam]
        if rec.status in ("quarantined", "rejected"):
            continue
        # implemented: executable + test_ref
        if rec.executable() and rec.test_ref:
            try:
                s.set_status(fam, "implemented", reason="executable transition + tests present")
            except Exception:
                pass
        # locally_validated: + held-out/PPC record (any outcome recorded)
        if any(v.kind in ("held_out", "posterior_predictive") and v.passed is not None
               for v in rec.validation + [v for p in rec.packs for v in p.validation]):
            try:
                s.set_status(fam, "locally_validated", reason="held-out validation recorded")
            except Exception:
                pass
        # production_eligible: + a PASSED held-out AND a citation (gate enforces the rest)
        try:
            s.set_status(fam, "production_eligible",
                         reason="passed held-out validation + evidence-backed")
        except Exception:
            pass                                            # gate refused — stays at its earned status


if __name__ == "__main__":
    store = build()
    paths = store.save()
    summ = store.summary()
    print("registry built:", paths)
    print("families:", summ["n_families"], "packs:", summ["n_packs"])
    print("by_status:", summ["by_status"])
    print("no executable transition:", summ["empty_entries"])
    print("no validation history:", len(summ["families_without_validation"]))
