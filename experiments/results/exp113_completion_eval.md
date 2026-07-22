# EXP-113 — Lean V2 simulation-completion evaluation

The same 5 frozen BTF-3 questions as EXP-112, rerun through the canonical lean_v2 profile after the simulation-completion fix. Prior and simulation are fully separate; the headline is the mass-based recovery blend; no combiner was calibrated and no BTF-3 outcome was trained on.

## Forecast decomposition (the required table)

| Question | Prior (n) | Sim-conditional | Resolved mass | Headline | Source | Final−Prior | Outcome | Brier (headline) | Brier (sim) |
|---|---|---|---|---|---|---|---|---|---|
| Banxico | 0.8333 (2) | 0.0988 | 1.0 | 0.0988 | mass_weighted:completed_rollouts+grounded_prior | -0.7345 | 1 | 0.8122 | 0.8122 |
| BoJ | 0.875 (3) | 0.0364 | 1.0 | 0.0364 | mass_weighted:completed_rollouts+grounded_prior | -0.8386 | 1 | 0.9285 | 0.9285 |
| visionOS | 0.8333 (2) | 1.0 | 0.75 | 0.9583 | mass_weighted:partial_rollouts+grounded_prior | 0.125 | 1 | 0.0017 | 0.0 |
| Wale | 0.1667 (2) | 0.1302 | 1.0 | 0.1302 | mass_weighted:completed_rollouts+grounded_prior | -0.0365 | 1 | 0.7566 | 0.7566 |
| Hormuz | 0.5 (2) | 1.0 | 0.6667 | 0.8333 | mass_weighted:partial_rollouts+grounded_prior | 0.3333 | 0 | 0.6944 | 1.0 |

- mean Brier (headline): 0.6387 (exp112: 0.0743, Lean V1: 0.3369, full-fidelity: 0.4403)
- mean Brier (simulation-only forecast): 0.6995
- simulation moved the forecast toward the outcome (vs prior): 1/5
- mean resolved simulation mass: 0.8833
- completion acceptance all-ok: 3/5 | resolved≥80% target met: 3/5 | unknown-state mass zero: 5/5
- totals: 143 calls, 756.8s, $0.1223 | 12min/100-call guard passed on every question: True

## Per-question: exactly what simulated under the hood

### Banxico — headline 0.0988 (mass_weighted:completed_rollouts+grounded_prior), outcome 1

- readiness: **repairable** (terminal round-trip ok=True); status **completed**
- prior_forecast 0.8333 (counted n=2) | simulation_forecast 0.0988 on resolved mass 1.0 | bounds [0.0, 0.5236] (residual 0.142625)
- unresolved by cause: {}
- actor-state completeness: victoria_rodriguez_ceja: 0→3 (regenerated, r=0.0); jonathan_heath: 0→3 (regenerated, r=0.05); irene_espinosa_cantellano: 0→3 (regenerated, r=0.0); galia_borja_gomez: 0→3 (regenerated, r=0.05); omar_mejia_castelazo: 0→3 (regenerated, r=0.05)
- reversal-state search: ran=False, added=0
- world: shared conditions [{'easing_cycle_ended': 'easing_cycle_active', 'economic_contraction_and_inflation_forecast': 'contraction_with_high_inflation', 'internal_disagreement_persists': 'unanimous_recent'}, {'easing_cycle_ended': 'easing_cycle_active', 'economic_contraction_and_inflation_forecast': 'contraction_with_high_inflation', 'internal_disagreement_persists': 'split_recent'}, {'easing_cycle_ended': 'easing_cycle_active', 'economic_contraction_and_inflation_forecast': 'growth_with_low_inflation', 'internal_disagreement_persists': 'unanimous_recent'}, {'easing_cycle_ended': 'easing_cycle_ended', 'economic_contraction_and_inflation_forecast': 'contraction_with_high_inflation', 'internal_disagreement_persists': 'unanimous_recent'}, {'easing_cycle_ended': 'easing_cycle_active', 'economic_contraction_and_inflation_forecast': 'growth_with_low_inflation', 'internal_disagreement_persists': 'split_recent'}, {'easing_cycle_ended': 'easing_cycle_ended', 'economic_contraction_and_inflation_forecast': 'contraction_with_high_inflation', 'internal_disagreement_persists': 'split_recent'}]; actor states {'victoria_rodriguez_ceja': 3, 'jonathan_heath': 3, 'irene_espinosa_cantellano': 3, 'galia_borja_gomez': 3, 'omar_mejia_castelazo': 3}; waves 2; weighted nodes 1464 (merged 0); completion rounds 1
- decisions: 18 unique contexts, 8748 reuses, actor calls 25
- first simulated decisions:
    - day 2026-06-25 — galia_borja_gomez [dovish_cut_to_stimulate_growth] act: record_vote (vote cut)
    - day 2026-06-25 — irene_espinosa_cantellano [dovish_cut_to_avoid_deep_recession] act: hold_interest_rate (vote hold)
    - day 2026-06-25 — jonathan_heath [dovish_cut_to_avoid_deep_recession] act: record_vote (vote cut)
    - day 2026-06-25 — omar_mejia_castelazo [dovish_cut_to_stimulate_growth] act: record_vote (vote cut)
    - day 2026-06-25 — victoria_rodriguez_ceja [dovish_cut_to_avoid_deeper_recession] gather_information: gather_information (vote cut)
    - day 2026-06-25 — galia_borja_gomez [dovish_cut_to_stimulate_growth] act: record_vote (vote cut)
    - day 2026-06-25 — irene_espinosa_cantellano [dovish_cut_to_avoid_deep_recession] act: hold_interest_rate (vote hold)
    - day 2026-06-25 — jonathan_heath [dovish_cut_to_avoid_deep_recession] act: record_vote (vote cut)
    - day 2026-06-25 — omar_mejia_castelazo [hawkish_hold_to_restore_credibility] act: record_vote (vote hold)
    - day 2026-06-25 — victoria_rodriguez_ceja [dovish_cut_to_avoid_deeper_recession] gather_information: gather_information (vote cut)
    - day 2026-06-25 — galia_borja_gomez [dovish_cut_to_stimulate_growth] act: record_vote (vote cut)
    - day 2026-06-25 — irene_espinosa_cantellano [dovish_cut_to_avoid_deep_recession] act: hold_interest_rate (vote hold)
- acceptance: {'terminal_unknown_state_mass': 0.0, 'terminal_unknown_state_ok': True, 'terminal_missing_mechanism_mass': 0.0, 'terminal_missing_mechanism_ok': True, 'provider_failure_mass': 0.0, 'provider_failure_ok': True, 'resolved_share': 1.0, 'resolved_target_met': True, 'resolved_hard_floor_met': True, 'all_ok': True}
- Brier: headline 0.8122 | simulation-only 0.8122 | prior 0.0278 | sim moved toward outcome: False
- cost: 33 calls, 184.9s, $0.0323

### BoJ — headline 0.0364 (mass_weighted:completed_rollouts+grounded_prior), outcome 1

- readiness: **ready** (terminal round-trip ok=True); status **completed**
- prior_forecast 0.875 (counted n=3) | simulation_forecast 0.0364 on resolved mass 1.0 | bounds [0.0, 0.7153] (residual 0.63136)
- unresolved by cause: {}
- actor-state completeness: ueda_kazuo: 2→2 (generated, r=0.2); nakagawa_junko: 1→4 (regenerated, r=0.2); takata_hajime: 1→4 (regenerated, r=0.2); tamura_naoki: 1→4 (regenerated, r=0.2); other_members: 2→2 (generated, r=0.1)
- reversal-state search: ran=False, added=0
- world: shared conditions [{'boj_gradual_hiking_cycle': 'tightening_cycle_active', 'dissenting_pressure_for_hike': 'dissenting_faction_active', 'market_expectations_elevated': 'market_expects_hike'}, {'boj_gradual_hiking_cycle': 'tightening_cycle_active', 'dissenting_pressure_for_hike': 'dissenting_faction_active', 'market_expectations_elevated': 'market_uncertain'}, {'boj_gradual_hiking_cycle': 'tightening_cycle_active', 'dissenting_pressure_for_hike': 'dissenting_faction_inactive', 'market_expectations_elevated': 'market_expects_hike'}, {'boj_gradual_hiking_cycle': 'tightening_cycle_paused', 'dissenting_pressure_for_hike': 'dissenting_faction_active', 'market_expectations_elevated': 'market_expects_hike'}, {'boj_gradual_hiking_cycle': 'tightening_cycle_active', 'dissenting_pressure_for_hike': 'dissenting_faction_inactive', 'market_expectations_elevated': 'market_uncertain'}, {'boj_gradual_hiking_cycle': 'tightening_cycle_paused', 'dissenting_pressure_for_hike': 'dissenting_faction_active', 'market_expectations_elevated': 'market_uncertain'}]; actor states {'ueda_kazuo': 2, 'nakagawa_junko': 4, 'takata_hajime': 4, 'tamura_naoki': 4, 'other_members': 2}; waves 3; weighted nodes 1548 (merged 0); completion rounds 1
- decisions: 21 unique contexts, 10368 reuses, actor calls 37
- first simulated decisions:
    - day 2026-06-16 — nakagawa_junko [nakagawa_domestic_political_pressure] gather_information: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — other_members [other_swing_voters_hike] act: vote_on_rate (vote Raise to 1.0%)
    - day 2026-06-16 — takata_hajime [takata_data_dependent_hawk] wait: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — tamura_naoki [tamura_credibility_hawk] act: vote_on_rate (vote Raise to 1.0%)
    - day 2026-06-16 — ueda_kazuo [ueda_cautious_gradualist] wait: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — nakagawa_junko [nakagawa_domestic_political_pressure] gather_information: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — other_members [other_swing_voters_hold] act: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — takata_hajime [takata_data_dependent_hawk] wait: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — tamura_naoki [tamura_credibility_hawk] act: vote_on_rate (vote Raise to 1.0%)
    - day 2026-06-16 — ueda_kazuo [ueda_cautious_gradualist] wait: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — nakagawa_junko [nakagawa_domestic_political_pressure] gather_information: vote_on_rate (vote Maintain at 0.75%)
    - day 2026-06-16 — other_members [other_swing_voters_hike] act: vote_on_rate (vote Raise to 1.0%)
- acceptance: {'terminal_unknown_state_mass': 0.0, 'terminal_unknown_state_ok': True, 'terminal_missing_mechanism_mass': 0.0, 'terminal_missing_mechanism_ok': True, 'provider_failure_mass': 0.0, 'provider_failure_ok': True, 'resolved_share': 1.0, 'resolved_target_met': True, 'resolved_hard_floor_met': True, 'all_ok': True}
- Brier: headline 0.9285 | simulation-only 0.9285 | prior 0.0156 | sim moved toward outcome: False
- cost: 41 calls, 190.0s, $0.0314

### visionOS — headline 0.9583 (mass_weighted:partial_rollouts+grounded_prior), outcome 1

- readiness: **repairable** (terminal round-trip ok=True); status **partially_resolved**
- prior_forecast 0.8333 (counted n=2) | simulation_forecast 1.0 on resolved mass 0.75 | bounds [0.722, 1.0] (residual 0.278)
- unresolved by cause: {'unresolved_missing_mechanism': 0.25}
- actor-state completeness: tim_cook: 2→2 (generated, r=0.05); marketing_team: 2→2 (generated, r=0.05)
- reversal-state search: ran=False, added=0
- mechanism recovery: validated=False; failure proof: the resolution criterion carries no parseable numeric threshold — a bounded numeric mechanism is not the right bridge for this terminal
- world: shared conditions [{'apple_os_naming_convention': 'year_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_abandoned', 'wwdc_annual_os_announcement_pattern': 'annual_wwdc_os_announcement'}, {'apple_os_naming_convention': 'year_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_active', 'wwdc_annual_os_announcement_pattern': 'annual_wwdc_os_announcement'}, {'apple_os_naming_convention': 'version_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_abandoned', 'wwdc_annual_os_announcement_pattern': 'annual_wwdc_os_announcement'}, {'apple_os_naming_convention': 'year_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_abandoned', 'wwdc_annual_os_announcement_pattern': 'no_annual_announcement'}, {'apple_os_naming_convention': 'version_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_active', 'wwdc_annual_os_announcement_pattern': 'annual_wwdc_os_announcement'}, {'apple_os_naming_convention': 'year_based_naming', 'vision_pro_platform_abandonment_reports': 'platform_active', 'wwdc_annual_os_announcement_pattern': 'no_annual_announcement'}]; actor states {'tim_cook': 2, 'marketing_team': 2}; waves 2; weighted nodes 36 (merged 0); completion rounds 1
- decisions: 7 unique contexts, 48 reuses, actor calls 16
- first simulated decisions:
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
    - day 2026-06-08 — tim_cook [vision_pro_platform_continues] act: include_in_keynote (vote include_in_keynote)
    - day 2026-06-08 — tim_cook [vision_pro_platform_sunset] wait: wait
- acceptance: {'terminal_unknown_state_mass': 0.0, 'terminal_unknown_state_ok': True, 'terminal_missing_mechanism_mass': 0.25, 'terminal_missing_mechanism_ok': True, 'provider_failure_mass': 0.0, 'provider_failure_ok': True, 'resolved_share': 0.75, 'resolved_target_met': False, 'resolved_hard_floor_met': True, 'all_ok': False}
- Brier: headline 0.0017 | simulation-only 0.0 | prior 0.0278 | sim moved toward outcome: True
- cost: 21 calls, 111.8s, $0.0176

### Wale — headline 0.1302 (mass_weighted:completed_rollouts+grounded_prior), outcome 1

- readiness: **ready** (terminal round-trip ok=True); status **completed**
- prior_forecast 0.1667 (counted n=2) | simulation_forecast 0.1302 on resolved mass 1.0 | bounds [0.0, 0.8607] (residual 0.737856)
- unresolved by cause: {}
- actor-state completeness: matthew_wale: 0→3 (regenerated, r=0.2); john_agovaka: 0→3 (regenerated, r=0.2); frederick_kologeto: 0→3 (regenerated, r=0.2); jeremiah_manele: 0→3 (regenerated, r=0.2); opposition_coalition_mps: 0→3 (regenerated, r=0.2)
- reversal-state search: ran=False, added=0
- world: shared conditions [{'china_taiwan_diplomatic_rivalry': 'pro_china', 'economic_dependence_on_aid': 'high_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}, {'china_taiwan_diplomatic_rivalry': 'pro_taiwan', 'economic_dependence_on_aid': 'high_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}, {'china_taiwan_diplomatic_rivalry': 'neutral', 'economic_dependence_on_aid': 'high_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}, {'china_taiwan_diplomatic_rivalry': 'pro_china', 'economic_dependence_on_aid': 'low_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}, {'china_taiwan_diplomatic_rivalry': 'pro_taiwan', 'economic_dependence_on_aid': 'low_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}, {'china_taiwan_diplomatic_rivalry': 'neutral', 'economic_dependence_on_aid': 'low_aid_dependence', 'solomon_islands_coalition_instability': 'fluid_coalitions'}]; actor states {'matthew_wale': 3, 'john_agovaka': 3, 'frederick_kologeto': 3, 'jeremiah_manele': 3, 'opposition_coalition_mps': 3}; waves 3; weighted nodes 1470 (merged 0); completion rounds 0
- decisions: 15 unique contexts, 7290 reuses, actor calls 24
- first simulated decisions:
    - day 2026-05-21 — frederick_kologeto [self_ambition] act: cast_vote (vote Frederick Kologeto)
    - day 2026-05-21 — jeremiah_manele [accept_defeat_and_seek_exile] act: cast_vote (vote Frederick Kologeto)
    - day 2026-05-21 — john_agovaka [dark_horse_self_promotion] wait: defer (vote John Agovaka)
    - day 2026-05-21 — matthew_wale [confident_winner_with_china_backing] act: cast_vote (vote Matthew Wale)
    - day 2026-05-21 — opposition_coalition_mps [factional_split_over_wale] gather_information: gather_information (vote Frederick Kologeto)
    - day 2026-05-21 — frederick_kologeto [self_ambition] act: cast_vote (vote Frederick Kologeto)
    - day 2026-05-21 — jeremiah_manele [accept_defeat_and_seek_exile] act: cast_vote (vote Frederick Kologeto)
    - day 2026-05-21 — john_agovaka [dark_horse_self_promotion] wait: defer (vote John Agovaka)
    - day 2026-05-21 — matthew_wale [confident_winner_with_china_backing] act: cast_vote (vote Matthew Wale)
    - day 2026-05-21 — opposition_coalition_mps [pragmatic_deal_with_manele_loyalists] gather_information: defer (vote Jeremiah Manele)
    - day 2026-05-21 — frederick_kologeto [self_ambition] act: cast_vote (vote Frederick Kologeto)
    - day 2026-05-21 — jeremiah_manele [accept_defeat_and_seek_exile] act: cast_vote (vote Frederick Kologeto)
- acceptance: {'terminal_unknown_state_mass': 0.0, 'terminal_unknown_state_ok': True, 'terminal_missing_mechanism_mass': 0.0, 'terminal_missing_mechanism_ok': True, 'provider_failure_mass': 0.0, 'provider_failure_ok': True, 'resolved_share': 1.0, 'resolved_target_met': True, 'resolved_hard_floor_met': True, 'all_ok': True}
- Brier: headline 0.7566 | simulation-only 0.7566 | prior 0.6944 | sim moved toward outcome: False
- cost: 30 calls, 168.9s, $0.0257

### Hormuz — headline 0.8333 (mass_weighted:partial_rollouts+grounded_prior), outcome 0

- readiness: **repairable** (terminal round-trip ok=True); status **partially_resolved**
- prior_forecast 0.5 (counted n=2) | simulation_forecast 1.0 on resolved mass 0.6667 | bounds [0.6859, 1.0] (residual 0.3141)
- unresolved by cause: {'unresolved_missing_mechanism': 0.333336}
- actor-state completeness: us_navy: 2→2 (generated, r=0.05); iran_revolutionary_guards: 2→2 (generated, r=0.05); oil_tanker_operators: 3→3 (generated, r=0.05)
- reversal-state search: ran=False, added=0
- mechanism recovery: validated=False; failure proof: the resolution criterion carries no parseable numeric threshold — a bounded numeric mechanism is not the right bridge for this terminal
- world: shared conditions [{'global_oil_market_disruption': 'normal', 'hormuz_closure_regime': 'closed'}, {'global_oil_market_disruption': 'normal', 'hormuz_closure_regime': 'partially_open'}, {'global_oil_market_disruption': 'normal', 'hormuz_closure_regime': 'fully_open'}, {'global_oil_market_disruption': 'disrupted', 'hormuz_closure_regime': 'closed'}, {'global_oil_market_disruption': 'disrupted', 'hormuz_closure_regime': 'partially_open'}, {'global_oil_market_disruption': 'disrupted', 'hormuz_closure_regime': 'fully_open'}]; actor states {'us_navy': 2, 'iran_revolutionary_guards': 2, 'oil_tanker_operators': 3}; waves 2; weighted nodes 78 (merged 0); completion rounds 1
- decisions: 7 unique contexts, 216 reuses, actor calls 10
- first simulated decisions:
    - day 2026-06-01 — iran_revolutionary_guards [mining_and_attacks_high] act: Maintain current posture and continue closure without clearing mines
    - day 2026-06-01 — oil_tanker_operators [risk_acceptance_high] act: tanker_operators_transit (vote tanker_operators_transit)
    - day 2026-06-01 — us_navy [blockade_enforcement_high] act: Continue aggressive interdiction of all vessels attempting to reach Iranian ports.
    - day 2026-06-01 — iran_revolutionary_guards [mining_and_attacks_high] act: Maintain current posture and continue closure without clearing mines
    - day 2026-06-01 — oil_tanker_operators [risk_aversion_high] wait: wait
    - day 2026-06-01 — us_navy [blockade_enforcement_high] act: Continue aggressive interdiction of all vessels attempting to reach Iranian ports.
    - day 2026-06-01 — iran_revolutionary_guards [mining_and_attacks_high] act: Maintain current posture and continue closure without clearing mines
    - day 2026-06-01 — oil_tanker_operators [selective_transit] act: Transit lower-risk tankers immediately; hold US-linked tanker pending escort or reassessment
    - day 2026-06-01 — us_navy [blockade_enforcement_high] act: Continue aggressive interdiction of all vessels attempting to reach Iranian ports.
    - day 2026-06-01 — iran_revolutionary_guards [mining_and_attacks_low] act: Continue selective mine placement and harassment without full closure
    - day 2026-06-01 — oil_tanker_operators [risk_acceptance_high] act: tanker_operators_transit (vote tanker_operators_transit)
    - day 2026-06-01 — us_navy [blockade_enforcement_high] act: Continue aggressive interdiction of all vessels attempting to reach Iranian ports.
- acceptance: {'terminal_unknown_state_mass': 0.0, 'terminal_unknown_state_ok': True, 'terminal_missing_mechanism_mass': 0.333336, 'terminal_missing_mechanism_ok': True, 'provider_failure_mass': 0.0, 'provider_failure_ok': True, 'resolved_share': 0.6667, 'resolved_target_met': False, 'resolved_hard_floor_met': True, 'all_ok': False}
- Brier: headline 0.6944 | simulation-only 1.0 | prior 0.25 | sim moved toward outcome: False
- cost: 18 calls, 101.2s, $0.0153

