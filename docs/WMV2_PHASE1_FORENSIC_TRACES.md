# WMv2 Phase 1 — Forensic Traces (B15)

*One deep, end-to-end trace per domain category. Every intermediate structure is shown so a reviewer can confirm the forecast was produced by typed mechanisms / broad priors (never an LLM-minted number) and that no coherent question was refused. Machine-readable companion: `experiments/results/wmv2_phase1_forensic_traces.json`.*

Model: DeepSeek V3 · 16 domains · 11 calls · ~$0.0267 · 183.7s.

## messaging

**Q:** Will my manager reply to the budget-approval email I sent this morning by end of week?  · as-of 2023-05-01 → horizon 2023-05-08

- **outcome**: `response_occurrence` over `['reply', 'no_reply']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 2 entities, 1 institutions, 0 populations, 4 latents ['manager', 'sender']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'poisson_arrival', 'generic_outcome_prior', 'generic_outcome_prior']; rejected ['observation_exposure']; experimental ['email_reading_attention_allocation']
- **tiers**: {'manager reads email': 7, 'manager decides to reply': 7, 'manager sends reply': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'default_attention', 'lean': 'neutral', 'prior': 0.5}, {'id': 'delayed_attention', 'lean': 'weak_no', 'prior': 0.3}, {'id': 'delegation', 'lean': 'weak_no', 'prior': 0.2}]
- **fidelity**: explicit ['manager', 'sender', 'company', 'manager.attention_to_email', 'manager.decision_style', 'manager.email_reading_habit', 'sender.relationship_with_manager']; marginalized-with-uncertainty []; 66 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 66 StateDeltas, readout `terminal_states`; structural posterior {'default_attention': 0.5, 'delayed_attention': 0.303, 'delegation': 0.197}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'no_reply': 0.5303, 'reply': 0.4697}** (p=0.4697)
- **limitations**: ["omitted (negligible-sensitivity): other employees' emails to manager", 'omitted (negligible-sensitivity): content of budget email', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `2d03cd221578`

## negotiation

**Q:** Will the buyer accept our counteroffer of $4.2M for the office building?  · as-of 2023-04-01 → horizon 2023-05-01

- **outcome**: `binary` over `['accept', 'reject']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 2 entities, 1 institutions, 1 populations, 6 latents ['buyer', 'seller']
- **mechanisms**: accepted ['agent_decision', 'generic_outcome_prior']; rejected ['bargaining_rubinstein', 'belief_update_exposure', 'negotiation_concession']; experimental ['property_valuation_model']
- **tiers**: {'buyer_reservation_price_formation': 7, 'seller_reservation_price_formation': 7, 'bargaining_process': 7, 'outside_option_evolution': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'buyer_has_lower_reservation', 'lean': 'strong_no', 'prior': 0.3}, {'id': 'buyer_has_higher_reservation', 'lean': 'strong_yes', 'prior': 0.3}, {'id': 'bargaining_equilibrium', 'lean': 'neutral', 'prior': 0.4}]
- **fidelity**: explicit ['buyer', 'seller', 'commercial_real_estate_market', 'real_estate_transaction_framework', 'buyer.reservation_price', 'seller.reservation_price', 'buyer.outside_options', 'buyer.patience', 'seller.patience', 'market_price_trend']; marginalized-with-uncertainty []; 76 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 76 StateDeltas, readout `terminal_states`; structural posterior {'buyer_has_lower_reservation': 0.3026, 'buyer_has_higher_reservation': 0.3026, 'bargaining_equilibrium': 0.3947}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'reject': 0.5, 'accept': 0.5}** (p=0.5)
- **limitations**: ['omitted (negligible-sensitivity): financing_approval', 'omitted (negligible-sensitivity): regulatory_approvals', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `f35562fb6cdd`

## organizational_decision

**Q:** Will the engineering VP approve the request to hire two more backend engineers this quarter?  · as-of 2023-07-01 → horizon 2023-09-30

- **outcome**: `binary` over `['approve', 'reject']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 4 entities, 2 institutions, 2 populations, 6 latents ['engineering_vp', 'ceo', 'cto', 'hr_head']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'relationship_update', 'resource_update', 'background_dynamics', 'generic_outcome_prior', 'generic_outcome_prior']; rejected []; experimental ['organizational_decision_making']
- **tiers**: {"VP's decision process considering budget, priorities, and advice": 7, 'Budget review and allocation': 7, 'Potential turnover events affecting workload': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'budget_constraint_driven', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'strategic_priority_driven', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'risk_averse_vp', 'lean': 'weak_no', 'prior': 0.2}]
- **fidelity**: explicit ['engineering_vp', 'ceo', 'cto', 'engineering_team', 'company_leadership', 'company_hiring_policy', 'quarterly_budget_review', 'engineering_vp.risk_tolerance', 'engineering_vp.priority_engineering_projects', 'ceo.strategic_priorities', 'cto.technical_strategy', 'engineering_team.backend_engineers.morale', 'engineering_team.backend_engineers.turnover_risk']; marginalized-with-uncertainty ['hr_head']; 76 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 237 StateDeltas, readout `terminal_states`; structural posterior {'budget_constraint_driven': 0.3947, 'strategic_priority_driven': 0.3947, 'risk_averse_vp': 0.2105}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'approve': 0.6053, 'reject': 0.3947}** (p=0.6053)
- **limitations**: ['omitted (negligible-sensitivity): company stock price', 'omitted (negligible-sensitivity): external market conditions for engineers', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `94f55d735f6b`

## election

**Q:** Will the incumbent mayor win re-election in the upcoming city vote?  · as-of 2023-09-01 → horizon 2023-11-07

- **outcome**: `binary` over `['incumbent_wins', 'incumbent_loses']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 1 institutions, 1 populations, 11 latents ['incumbent_mayor', 'challenger_1', 'challenger_2']
- **mechanisms**: accepted ['generic_outcome_prior']; rejected []; experimental []
- **tiers**: {'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'incumbent_advantage', 'lean': 'weak_yes', 'prior': 0.5}, {'id': 'anti_incumbent_wave', 'lean': 'weak_no', 'prior': 0.3}, {'id': 'competitive_three_way', 'lean': 'weak_yes', 'prior': 0.2}]
- **fidelity**: explicit ['incumbent_mayor', 'challenger_1', 'challenger_2', 'city_electorate', 'city_election_commission', 'incumbent_mayor.scandal_severity', 'city_electorate.partisan_lean', 'city_electorate.issue_priorities', 'challenger_1.quality', 'challenger_2.quality', 'campaign_effectiveness_incumbent', 'campaign_effectiveness_challenger_1', 'campaign_effectiveness_challenger_2', 'voter_turnout_mobilization_effect']; marginalized-with-uncertainty ['external_economic_shock', 'external_natural_disaster']; 80 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 80 StateDeltas, readout `terminal_states`; structural posterior {'incumbent_advantage': 0.5, 'anti_incumbent_wave': 0.3, 'competitive_three_way': 0.2}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'incumbent_wins': 0.55, 'incumbent_loses': 0.45}** (p=0.55)
- **limitations**: ['degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `8b91ea74c55d`

## legislation

**Q:** Will the infrastructure bill pass the Senate before the recess?  · as-of 2023-06-01 → horizon 2023-08-01

- **outcome**: `binary` over `['pass', 'fail']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 1 institutions, 3 populations, 4 latents ['Senate_Majority_Leader', 'Senate_Minority_Leader', 'President']
- **mechanisms**: accepted ['poisson_arrival', 'generic_outcome_prior']; rejected ['institutional_vote', 'whipcount_binomial', 'agenda_stage_control', 'bargaining_rubinstein', 'coalition_formation', 'belief_update_exposure', 'quantal_response_choice']; experimental ['filibuster_decision', 'amendment_process']
- **tiers**: {'whip_count_evolution': 7, 'undecided_lean_formation': 7, 'filibuster_decision': 7, 'cloture_vote': 7, 'final_passage_vote': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'leadership_control', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'bipartisan_compromise', 'lean': 'weak_yes', 'prior': 0.3}, {'id': 'filibuster_block', 'lean': 'weak_no', 'prior': 0.3}]
- **fidelity**: explicit ['Senate_Majority_Leader', 'Senate_Minority_Leader', 'President', 'Senate_Democrats', 'Senate_Republicans', 'Senate_Independents', 'Senate', 'Senate_Democrats.segments.Dem_moderates.swing_lean', 'Senate_Republicans.segments.GOP_moderates.bipartisan_tendency', 'Senate.undecided_count', 'Senate.filibuster_probability']; marginalized-with-uncertainty []; 66 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 66 StateDeltas, readout `terminal_states`; structural posterior {'leadership_control': 0.3939, 'bipartisan_compromise': 0.303, 'filibuster_block': 0.303}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'pass': 0.5303, 'fail': 0.4697}** (p=0.5303)
- **limitations**: ['omitted (negligible-sensitivity): public_opinion_polls', 'omitted (negligible-sensitivity): campaign_finance_contributions', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `1bf93c8a7066`

## acquisition

**Q:** Will the proposed merger between the two regional banks be completed?  · as-of 2023-05-01 → horizon 2023-12-31

- **outcome**: `binary` over `['completed', 'not_completed']`, readout `merger_completed`; lean `neutral`
- **world**: 6 entities, 6 institutions, 3 populations, 3 latents ['regional_bank_A', 'regional_bank_B', 'federal_reserve', 'doj_antitrust', 'fdic', 'state_banking_regulator']
- **mechanisms**: accepted ['generic_outcome_prior']; rejected []; experimental []
- **tiers**: {'outcome_resolution': 6}; fallbacks [('outcome_resolution', 6)]
- **structural hypotheses**: —
- **fidelity**: explicit ['regional_bank_A', 'regional_bank_B', 'federal_reserve', 'doj_antitrust', 'fdic', 'state_banking_regulator', 'shareholders_bank_A', 'shareholders_bank_B', 'regulators', 'regional_bank_A.fields.board_approval', 'regional_bank_B.fields.board_approval', 'shareholders_bank_A.segments.institutional_shareholders_A.differs_on.voting_preference']; marginalized-with-uncertainty []; 45 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 45 StateDeltas, readout `terminal_states`; structural posterior None
- **RESULT**: status `completed_with_degradation`, grade `exploratory`, rec `not_requested` → **{'not_completed': 0.5111, 'completed': 0.4889}** (p=0.4889)
- **limitations**: ['degraded: support grade exploratory; 1 fallback mechanism(s) used']
- plan_hash `5d5ecb87d008`

## product_launch

**Q:** Will the new smartphone model launch on the announced date?  · as-of 2023-08-01 → horizon 2023-10-01

- **outcome**: `binary` over `['yes', 'no']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 4 entities, 3 institutions, 2 populations, 4 latents ['manufacturer_company', 'supplier_chipset', 'regulatory_body', 'retail_partner']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'resource_update', 'poisson_arrival', 'generic_outcome_prior']; rejected ['institutional_vote', 'agenda_stage_control', 'bernoulli_detection', 'rare_event_arrival']; experimental ['production_throughput', 'supply_chain_delay_propagation']
- **tiers**: {'production_completion': 7, 'regulatory_certification': 7, 'supply': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'baseline_on_time', 'lean': 'strong_yes', 'prior': 0.5}, {'id': 'supply_chain_disruption', 'lean': 'strong_no', 'prior': 0.3}, {'id': 'regulatory_hold', 'lean': 'strong_no', 'prior': 0.1}, {'id': 'strategic_delay', 'lean': 'weak_no', 'prior': 0.1}]
- **fidelity**: explicit ['manufacturer_company', 'supplier_chipset', 'regulatory_body', 'retail_partner', 'manufacturing_workforce', 'manufacturer_company.production_status', 'manufacturer_company.supply_chain_health', 'manufacturer_company.regulatory_approval_status', 'manufacturing_workforce.labor_dispute_risk']; marginalized-with-uncertainty ['consumer_base']; 74 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 74 StateDeltas, readout `terminal_states`; structural posterior {'baseline_on_time': 0.5, 'supply_chain_disruption': 0.2973, 'regulatory_hold': 0.0946, 'strategic_delay': 0.1081}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'yes': 0.5, 'no': 0.5}** (p=0.5)
- **limitations**: ['degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `e13a524bcec5`

## social_media_diffusion

**Q:** Will the campaign hashtag trend nationally within 24 hours of launch?  · as-of 2023-05-01 → horizon 2023-05-02

- **outcome**: `binary` over `['trend', 'not_trend']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 1 institutions, 1 populations, 5 latents ['campaign_team', 'platform', 'hashtag']
- **mechanisms**: accepted ['poisson_arrival', 'generic_outcome_prior', 'generic_outcome_prior']; rejected ['simple_contagion_hazard', 'complex_contagion_hazard', 'engagement_momentum_persistence', 'finite_population_saturation', 'platform_ranking', 'platform_examination', 'attention_dynamics', 'memory_decay', 'information_aging', 'exposure_response_hazard', 'mobilization', 'rare_event_arrival']; experimental ['trending_threshold_activation']
- **tiers**: {'initial_posting_and_amplification': 7, 'user_exposure_and_engagement': 7, 'viral_cascade_dynamics': 7, 'platform_trending_algorithm_activation': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'organic_grassroots', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'coordinated_astroturf', 'lean': 'strong_yes', 'prior': 0.3}, {'id': 'suppression_or_counter', 'lean': 'strong_no', 'prior': 0.3}]
- **fidelity**: explicit ['campaign_team', 'platform', 'hashtag', 'social_media_users', 'platform.trending_algorithm.threshold', 'campaign_team.coordination_level', 'social_media_users.supporters.engagement_rate', 'social_media_users.opponents.engagement_rate', 'social_media_users.neutral.engagement_rate']; marginalized-with-uncertainty []; 71 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 71 StateDeltas, readout `terminal_states`; structural posterior {'organic_grassroots': 0.3944, 'coordinated_astroturf': 0.2958, 'suppression_or_counter': 0.3099}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'trend': 0.5493, 'not_trend': 0.4507}** (p=0.5493)
- **limitations**: ['omitted (negligible-sensitivity): detailed_hashtag_content', 'omitted (negligible-sensitivity): cross_platform_spillover', 'omitted (negligible-sensitivity): long_term_memory_effects']
- plan_hash `f6aea04347e2`

## protest

**Q:** Will the planned climate march draw more than 10,000 participants?  · as-of 2023-09-01 → horizon 2023-09-20

- **outcome**: `binary` over `['yes', 'no']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 5 entities, 3 institutions, 2 populations, 5 latents ['march_organizer', 'city_government', 'police_department', 'media_outlets', 'counter_protest_groups']
- **mechanisms**: accepted ['poisson_arrival', 'generic_outcome_prior']; rejected ['mobilization', 'exposure_response_hazard', 'belief_update_exposure', 'attention_dynamics', 'memory_decay', 'simple_contagion_hazard', 'threshold_adoption', 'finite_population_saturation', 'rare_event_arrival']; experimental ['weather_impact_on_turnout']
- **tiers**: {'organizer_mobilization': 7, 'media_attention_spread': 7, 'participant_decision_to_attend': 7, 'weather_impact': 7, 'counter_protest_mobilization': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'high_mobilization', 'lean': 'strong_yes', 'prior': 0.3}, {'id': 'low_mobilization', 'lean': 'strong_no', 'prior': 0.5}, {'id': 'counter_protest_effect', 'lean': 'weak_yes', 'prior': 0.2}]
- **fidelity**: explicit ['march_organizer', 'city_government', 'police_department', 'media_outlets', 'counter_protest_groups', 'potential_participants', 'potential_participants.core_activists.mobilization_likelihood', 'potential_participants.sympathizers.mobilization_likelihood', 'potential_participants.general_public.awareness', 'weather_forecast_rain_probability', 'city_government.permit_status']; marginalized-with-uncertainty ['counter_protesters']; 71 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 71 StateDeltas, readout `terminal_states`; structural posterior {'high_mobilization': 0.2958, 'low_mobilization': 0.507, 'counter_protest_effect': 0.1972}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'no': 0.5775, 'yes': 0.4225}** (p=0.4225)
- **limitations**: ['omitted (negligible-sensitivity): national_politics', 'omitted (negligible-sensitivity): celebrity_endorsements', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `8a8f3746168a`

## strike

**Q:** Will the transit strike be resolved before Monday's commute?  · as-of 2023-07-14 → horizon 2023-07-17

- **outcome**: `binary` over `['resolved', 'not_resolved']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 4 entities, 2 institutions, 2 populations, 6 latents ['transit_authority', 'union_leadership', 'mayor', 'mediator']
- **mechanisms**: accepted ['poisson_arrival', 'generic_outcome_prior', 'generic_outcome_prior']; rejected ['bargaining_rubinstein', 'belief_update_exposure', 'relationship_update_bounded', 'institutional_vote']; experimental ['strike_negotiation_dynamics']
- **tiers**: {'bargaining between transit authority and union': 7, 'mediation facilitation': 7, 'public pressure influence': 7, 'deadline-driven decision making': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'rapid_resolution', 'lean': 'weak_yes', 'prior': 0.3}, {'id': 'deadlock_extension', 'lean': 'strong_no', 'prior': 0.5}, {'id': 'mayoral_intervention', 'lean': 'weak_yes', 'prior': 0.2}]
- **fidelity**: explicit ['transit_authority', 'union_leadership', 'mayor', 'mediator', 'striking_workers', 'commuters', 'mediation_board', 'union_leadership.hardline_stance', 'transit_authority.financial_reserves', 'mayor.intervention_willingness', 'mediator.effectiveness', 'striking_workers.hardliners.wage_demand', 'striking_workers.moderates.wage_demand']; marginalized-with-uncertainty []; 76 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 76 StateDeltas, readout `terminal_states`; structural posterior {'rapid_resolution': 0.3026, 'deadlock_extension': 0.5, 'mayoral_intervention': 0.1974}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'not_resolved': 0.6053, 'resolved': 0.3947}** (p=0.3947)
- **limitations**: ['omitted (negligible-sensitivity): individual commuter mode choice details', 'omitted (negligible-sensitivity): detailed financial markets impact', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `21dd43e979dd`

## court_ruling

**Q:** Will the appeals court uphold the lower court's ruling in the antitrust case?  · as-of 2023-05-01 → horizon 2023-10-01

- **outcome**: `categorical` over `['affirm', 'reverse', 'remand']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 4 entities, 2 institutions, 2 populations, 5 latents ['appeals_court', 'lower_court', 'plaintiff', 'defendant']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'background_dynamics', 'generic_outcome_prior', 'generic_outcome_prior']; rejected ['institutional_vote', 'rare_event_arrival']; experimental ['appellate_decision_making']
- **tiers**: {'panel composition and ideology': 7, 'legal argument evaluation': 7, 'precedent interpretation': 7, 'deliberation and voting': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'ideological_decision', 'lean': 'neutral', 'prior': 0.4}, {'id': 'legal_merit', 'lean': 'neutral', 'prior': 0.4}, {'id': 'external_pressure', 'lean': 'neutral', 'prior': 0.2}]
- **fidelity**: explicit ['appeals_court', 'lower_court', 'plaintiff', 'defendant', 'judges_pool', 'legal_community', 'appeals_court.ideological_lean', 'lower_court.legal_error', 'plaintiff.legal_quality', 'defendant.legal_quality', 'public_opinion_pressure']; marginalized-with-uncertainty []; 71 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 226 StateDeltas, readout `terminal_states`; structural posterior {'ideological_decision': 0.3944, 'legal_merit': 0.3944, 'external_pressure': 0.2113}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'reverse': 0.4507, 'affirm': 0.2958, 'remand': 0.2535}** (p=0.2958)
- **limitations**: ['omitted (negligible-sensitivity): detailed economic impact of ruling', 'omitted (negligible-sensitivity): media coverage specifics', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `556a7dbad6fc`

## fundraising

**Q:** Will the startup close its Series B round within the quarter?  · as-of 2023-06-01 → horizon 2023-09-30

- **outcome**: `binary` over `['closed', 'not_closed']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 2 institutions, 1 populations, 8 latents ['startup', 'lead_investor', 'syndicate_members']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'relationship_update', 'resource_update', 'background_dynamics', 'generic_outcome_prior', 'poisson_arrival', 'generic_outcome_prior']; rejected ['rare_event_arrival', 'quantal_response_choice', 'donation_response', 'negotiation_concession', 'trust_formation', 'trust_violation', 'trust_repair', 'bargaining_rubinstein', 'belief_learning', 'belief_update_exposure', 'observation_exposure', 'information_interpretation', 'information_aging', 'memory_decay', 'attention_allocation', 'attention_dynamics', 'engagement_momentum_persistence', 'mobilization', 'network_rewiring', 'reciprocity', 'relationship_strength_inference', 'relationship_update_bounded', 'resource_depletion', 'social_preference_population', 'norm_compliance', 'habit_formation', 'reinforcement_learning', 'experience_weighted_attraction', 'hierarchical_rate_shrinkage', 'susceptibility_frailty', 'finite_population_saturation', 'complex_contagion_hazard', 'simple_contagion_hazard', 'threshold_adoption', 'bounded_confidence', 'degroot_influence', 'coalition_formation', 'institutional_vote', 'voting_turnout', 'poll_error_aggregation', 'whipcount_binomial', 'agenda_stage_control', 'bernoulli_detection', 'gaussian_measurement', 'platform_examination', 'platform_ranking', 'exposure_response_hazard']; experimental ['investment_committee_decision', 'term_sheet_negotiation']
- **tiers**: {'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'strong_lead_investor', 'lean': 'weak_yes', 'prior': 0.3}, {'id': 'valuation_gap', 'lean': 'weak_no', 'prior': 0.3}, {'id': 'market_downturn', 'lean': 'strong_no', 'prior': 0.2}, {'id': 'existing_backers_support', 'lean': 'weak_yes', 'prior': 0.2}]
- **fidelity**: explicit ['startup', 'lead_investor', 'syndicate_members', 'potential_investors', 'board_of_directors', 'market_conditions', 'startup.valuation_ask', 'startup.revenue_run_rate', 'startup.burn_rate', 'startup.cash_on_hand', 'lead_investor.due_diligence_status', 'lead_investor.decision_timeline', 'market_sentiment_index', 'interest_rate']; marginalized-with-uncertainty []; 80 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 561 StateDeltas, readout `terminal_states`; structural posterior {'strong_lead_investor': 0.3, 'valuation_gap': 0.3, 'market_downturn': 0.2, 'existing_backers_support': 0.2}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'not_closed': 0.55, 'closed': 0.45}** (p=0.45)
- **limitations**: ['degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `76ce112ac5ce`

## coalition

**Q:** Will the three parties form a governing coalition after the election?  · as-of 2023-09-01 → horizon 2023-11-01

- **outcome**: `binary` over `['coalition_formed', 'no_coalition']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 2 institutions, 1 populations, 10 latents ['party_A_leader', 'party_B_leader', 'party_C_leader']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'relationship_update', 'resource_update', 'background_dynamics', 'generic_outcome_prior', 'poisson_arrival', 'generic_outcome_prior']; rejected ['institutional_vote', 'poll_error_aggregation', 'whipcount_binomial', 'agenda_stage_control', 'attention_allocation', 'attention_dynamics', 'bargaining_rubinstein', 'belief_learning', 'belief_update_exposure', 'bernoulli_detection', 'bounded_confidence', 'coalition_formation', 'complex_contagion_hazard', 'degroot_influence', 'donation_response', 'engagement_momentum_persistence', 'experience_weighted_attraction', 'exposure_response_hazard', 'finite_population_saturation', 'gaussian_measurement', 'habit_formation', 'hierarchical_rate_shrinkage', 'information_aging', 'information_interpretation', 'latent_expressed_opinion', 'memory_decay', 'mobilization', 'negotiation_concession', 'network_rewiring', 'norm_compliance', 'observation_exposure', 'platform_examination', 'platform_ranking', 'quantal_response_choice', 'rare_event_arrival', 'reciprocity', 'reinforcement_learning', 'relationship_strength_inference', 'relationship_update_bounded', 'resource_depletion', 'simple_contagion_hazard', 'social_preference_population', 'susceptibility_frailty', 'threshold_adoption', 'trust_formation', 'trust_repair', 'trust_violation', 'voting_turnout']; experimental []
- **tiers**: {'election_outcome_determination': 7, 'post_election_negotiation': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'ideological_proximity', 'lean': 'neutral', 'prior': 0.5}, {'id': 'office_seeking', 'lean': 'weak_yes', 'prior': 0.3}, {'id': 'electoral_punishment', 'lean': 'weak_no', 'prior': 0.2}]
- **fidelity**: explicit ['party_A_leader', 'party_B_leader', 'party_C_leader', 'electorate', 'parliament', 'electoral_commission', 'party_A_leader.ideology', 'party_B_leader.ideology', 'party_C_leader.ideology', 'party_A_leader.risk_aversion', 'party_B_leader.risk_aversion', 'party_C_leader.risk_aversion', 'party_A_leader.office_motivation', 'party_B_leader.office_motivation', 'party_C_leader.office_motivation', 'electorate.undecided_lean']; marginalized-with-uncertainty []; 80 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 242 StateDeltas, readout `terminal_states`; structural posterior {'ideological_proximity': 0.5, 'office_seeking': 0.3, 'electoral_punishment': 0.2}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'no_coalition': 0.525, 'coalition_formed': 0.475}** (p=0.475)
- **limitations**: ['degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `e8b9ed6ca3b3`

## market

**Q:** Will the stock rise more than 5% the day after the earnings call?  · as-of 2023-07-25 → horizon 2023-07-27

- **outcome**: `binary` over `['rise_more_than_5%', 'rise_5%_or_less_or_fall']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 3 entities, 2 institutions, 3 populations, 4 latents ['company_issuing_stock', 'ceo', 'cfo']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'background_dynamics', 'generic_outcome_prior', 'resource_update', 'generic_outcome_prior']; rejected ['attention_allocation', 'attention_dynamics', 'information_interpretation', 'observation_exposure', 'quantal_response_choice']; experimental ['price_discovery', 'earnings_surprise_calculation']
- **tiers**: {'earnings_surprise_determination': 7, 'investor_attention_allocation': 7, 'belief_update_from_earnings': 7, 'trading_decision': 7, 'price_impact_of_trades': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'efficient_market', 'lean': 'neutral', 'prior': 0.3}, {'id': 'behavioral_overreaction', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'information_cascade', 'lean': 'weak_yes', 'prior': 0.3}]
- **fidelity**: explicit ['company_issuing_stock', 'ceo', 'cfo', 'investors', 'analysts', 'media', 'company_issuing_stock.earnings_actual_eps', 'company_issuing_stock.guidance', 'investors.sentiment', 'analysts.reaction']; marginalized-with-uncertainty ['stock_exchange', 'sec']; 66 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 132 StateDeltas, readout `terminal_states`; structural posterior {'efficient_market': 0.303, 'behavioral_overreaction': 0.3939, 'information_cascade': 0.303}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'rise_more_than_5%': 0.5455, 'rise_5%_or_less_or_fall': 0.4545}** (p=0.5455)
- **limitations**: ['omitted (negligible-sensitivity): macroeconomic_indicators', 'omitted (negligible-sensitivity): political_events', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `fd005b7d21e0`

## reputation_crisis

**Q:** Will the company's CEO resign within a month of the scandal breaking?  · as-of 2023-05-01 → horizon 2023-06-01

- **outcome**: `binary` over `['resign', 'not_resign']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 2 entities, 2 institutions, 5 populations, 10 latents ['ceo', 'company']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'relationship_update', 'resource_update', 'background_dynamics', 'generic_outcome_prior', 'poisson_arrival', 'generic_outcome_prior']; rejected ['institutional_vote', 'rare_event_arrival', 'norm_compliance', 'quantal_response_choice', 'trust_violation', 'trust_repair', 'mobilization', 'attention_dynamics', 'memory_decay']; experimental ['board_deliberation_dynamics', 'reputation_damage_function']
- **tiers**: {'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'board_driven', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'ceo_voluntary', 'lean': 'weak_yes', 'prior': 0.3}, {'id': 'no_resignation', 'lean': 'weak_no', 'prior': 0.3}]
- **fidelity**: explicit ['ceo', 'company', 'board_members', 'employees', 'shareholders', 'media_outlets', 'regulators', 'board', 'company_governance', 'ceo.personal_reputation_cost', 'ceo.legal_risk', 'ceo.severance_package', 'board_members.independence', 'board_members.alignment_with_ceo', 'company.scandal_severity', 'company.governance_quality', 'media_outlets.coverage_intensity', 'shareholders.activism_likelihood', 'regulators.investigation_scope']; marginalized-with-uncertainty []; 80 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 323 StateDeltas, readout `terminal_states`; structural posterior {'board_driven': 0.4, 'ceo_voluntary': 0.3, 'no_resignation': 0.3}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'resign': 0.5375, 'not_resign': 0.4625}** (p=0.5375)
- **limitations**: ['degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `b6697fe85191`

## best_action

**Q:** Should we send the discount offer now or wait until the customer's renewal date to maximize retention?  · as-of 2023-06-01 → horizon 2023-07-01

- **outcome**: `binary` over `['renew', 'not_renew']`, readout `outcome` (repaired→canonical); lean `neutral`
- **world**: 2 entities, 1 institutions, 1 populations, 3 latents ['customer', 'company']
- **mechanisms**: accepted ['agent_decision', 'belief_update', 'resource_update', 'background_dynamics', 'generic_outcome_prior', 'generic_outcome_prior']; rejected ['exposure_response_hazard', 'memory_decay', 'quantal_response_choice']; experimental ['discount_effect_on_renewal']
- **tiers**: {'customer_decision_to_renew': 7, 'discount_offer_impact_on_decision': 7, 'timing_effect_on_attention_and_salience': 7, 'outcome_resolution': 7}; fallbacks [('outcome_resolution', 7)]
- **structural hypotheses**: [{'id': 'timing_irrelevant', 'lean': 'neutral', 'prior': 0.2}, {'id': 'now_better', 'lean': 'weak_yes', 'prior': 0.4}, {'id': 'renewal_better', 'lean': 'weak_yes', 'prior': 0.4}]
- **fidelity**: explicit ['customer', 'company', 'customer_segments', 'customer.discount_sensitivity', 'customer.attention_to_offer', 'customer.current_satisfaction']; marginalized-with-uncertainty []; 61 particles
- **provenance statuses** (no `observed` fabrication): {}
- **rollout**: 272 StateDeltas, readout `terminal_states`; structural posterior {'timing_irrelevant': 0.1967, 'now_better': 0.3934, 'renewal_better': 0.4098}
- **RESULT**: status `completed_with_degradation`, grade `highly_speculative`, rec `not_requested` → **{'renew': 0.541, 'not_renew': 0.459}** (p=0.541)
- **limitations**: ['omitted (negligible-sensitivity): competitor offers', 'omitted (negligible-sensitivity): customer demographics', 'degraded: support grade highly_speculative; 1 fallback mechanism(s) used']
- plan_hash `d04d912f30d2`

---

Across all traces: every question produced a forecast (no forecast abstention); every fallback names its tier; no entity field was stamped `observed`; the terminal distribution is over the declared option space; and the only numbers the LLM supplied were qualitative leans, not probabilities.