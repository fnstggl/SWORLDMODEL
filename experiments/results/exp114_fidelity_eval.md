# EXP-114 — Lean V2 real-world-fidelity five-question measurement

Same five frozen BTF-3 questions, cold-cache, sequential, through `simulate_world(..., "lean_v2")` carrying the full D1–D18 fidelity architecture. No outcome or prior in any prompt; outcomes joined only after each forecast froze.

## Measurement table

| Q | status | headline | source | Brier (D1–D18) | Brier exp112 | Brier full-fid | Brier Lean V1 | calls | wall s | struct pass |
|---|---|---|---|---|---|---|---|---|---|---|
| Banxico | completed | 0.4398 | deliberative_institution_vote | 0.3138 | 0.0625 | 0.0732 | 0.0534 | 37 | 239.1 | True |
| BoJ | completed | 0.0385 | deliberative_institution_vote | 0.9245 | 0.0156 | 0.6084 | 0.1897 | 32 | 220.5 | True |
| visionOS | completed | 0.0 | mass_weighted:completed_rollouts+grounded_prior | 1.0 | 0.0156 | 0.0276 | 0.3399 | 11 | 102.5 | True |
| Wale | under_modeled | 0.1667 | grounded_reference_prior | 0.6944 | 0.0278 | 0.7083 | 0.3192 | 4 | 83.7 | True |
| Hormuz | completed | 0.125 | mass_weighted:completed_rollouts+grounded_prior | 0.0156 | 0.25 | 0.7838 | 0.7823 | 29 | 118.6 | True |

**Mean Brier (D1–D18, 5 scored):** 0.5897

## Per-question under-the-hood (structure, then frozen forecast, then outcome)

### Banxico

- terminal_kind: institution_vote; structural_pass: True
- structural_invariants: {'faithful_roster': True, 'threshold_not_rescaled': True, 'deliberation_ran': True, 'evidence_present': True, 'packets_built': True, 'no_packet_leakage': True, 'behavior_grounded': True, 'structural_fidelity_not_broken': True}
- representation: {'real_member_count': None, 'represented_voting_power': 5, 'total_voting_power': 5, 'threshold': 5.0, 'declared_threshold': 5.0, 'n_decision_units': 5, 'candidates': [], 'faithful': True, 'verdict': 'ready'}
- deliberation: {'institution_type': 'consensus_body', 'rounds_run': 8, 'material_changes': 40, 'n_messages': 45}
- evidence: {'n_facts': 6, 'dropped_leakage': 0, 'contradiction_groups': []}; packets: {'n': 5, 'leakage_flags': {}}
- outcome dimension: {'output_dimension': None, 'required_dimension': None, 'dimension_ok': None}
- structural_fidelity: ready {'resolution': 'ready', 'institution': 'ready', 'evidence': 'ready', 'behavior': 'ready'}
- FROZEN forecast: 0.4398 (source deliberative_institution_vote, status completed)
- guard: {'wall_s': 239.1, 'n_calls': 37, 'passed_hard_stop': True}
- AFTER-FREEZE outcome: 1; Brier {'full_fidelity': 0.0732, 'lean_v1': 0.0534, 'exp112_lean_v2': 0.0625, 'exp114_fidelity': 0.3138}

### BoJ

- terminal_kind: institution_vote; structural_pass: True
- structural_invariants: {'faithful_roster': True, 'threshold_not_rescaled': True, 'deliberation_ran': True, 'evidence_present': True, 'packets_built': True, 'no_packet_leakage': True, 'behavior_grounded': True, 'structural_fidelity_not_broken': True}
- representation: {'real_member_count': None, 'represented_voting_power': 5, 'total_voting_power': 5, 'threshold': 5.0, 'declared_threshold': 5.0, 'n_decision_units': 5, 'candidates': [], 'faithful': True, 'verdict': 'ready'}
- deliberation: {'institution_type': 'consensus_body', 'rounds_run': 8, 'material_changes': 28, 'n_messages': 33}
- evidence: {'n_facts': 11, 'dropped_leakage': 0, 'contradiction_groups': []}; packets: {'n': 5, 'leakage_flags': {}}
- outcome dimension: {'output_dimension': None, 'required_dimension': None, 'dimension_ok': None}
- structural_fidelity: ready {'resolution': 'ready', 'institution': 'ready', 'evidence': 'ready', 'behavior': 'ready'}
- FROZEN forecast: 0.0385 (source deliberative_institution_vote, status completed)
- guard: {'wall_s': 220.5, 'n_calls': 32, 'passed_hard_stop': True}
- AFTER-FREEZE outcome: 1; Brier {'full_fidelity': 0.6084, 'lean_v1': 0.1897, 'exp112_lean_v2': 0.0156, 'exp114_fidelity': 0.9245}

### visionOS

- terminal_kind: unknown; structural_pass: True
- structural_invariants: {'evidence_present': True, 'packets_built': True, 'no_packet_leakage': True, 'behavior_grounded': True, 'structural_fidelity_not_broken': True}
- representation: {'real_member_count': None, 'represented_voting_power': None, 'total_voting_power': None, 'threshold': None, 'declared_threshold': None, 'n_decision_units': 0, 'candidates': None, 'faithful': None, 'verdict': None}
- deliberation: None
- evidence: {'n_facts': 5, 'dropped_leakage': 0, 'contradiction_groups': []}; packets: {'n': 2, 'leakage_flags': {}}
- outcome dimension: {'output_dimension': None, 'required_dimension': None, 'dimension_ok': None}
- structural_fidelity: ready {'resolution': 'ready', 'evidence': 'ready', 'behavior': 'ready'}
- FROZEN forecast: 0.0 (source mass_weighted:completed_rollouts+grounded_prior, status completed)
- guard: {'wall_s': 102.5, 'n_calls': 11, 'passed_hard_stop': True}
- AFTER-FREEZE outcome: 1; Brier {'full_fidelity': 0.0276, 'lean_v1': 0.3399, 'exp112_lean_v2': 0.0156, 'exp114_fidelity': 1.0}

### Wale

- terminal_kind: institution_vote; structural_pass: True
- structural_invariants: {'faithful_roster': True, 'threshold_not_rescaled': True, 'deliberation_ran': True, 'evidence_present': True, 'packets_built': True, 'no_packet_leakage': True, 'behavior_grounded': True, 'structural_fidelity_not_broken': True}
- representation: {'real_member_count': 26, 'represented_voting_power': 26, 'total_voting_power': 26, 'threshold': 26.0, 'declared_threshold': 26.0, 'n_decision_units': 5, 'candidates': ['matthew_wale', 'john_agovaka'], 'faithful': True, 'verdict': 'ready'}
- deliberation: {'institution_type': 'independent_body', 'rounds_run': 1, 'material_changes': 0, 'n_messages': 3}
- evidence: {'n_facts': 8, 'dropped_leakage': 0, 'contradiction_groups': []}; packets: {'n': 4, 'leakage_flags': {}}
- outcome dimension: {'output_dimension': None, 'required_dimension': None, 'dimension_ok': None}
- structural_fidelity: ready {'resolution': 'ready', 'institution': 'ready', 'evidence': 'ready', 'behavior': 'ready'}
- FROZEN forecast: 0.1667 (source grounded_reference_prior, status under_modeled)
- guard: {'wall_s': 83.7, 'n_calls': 4, 'passed_hard_stop': True}
- AFTER-FREEZE outcome: 1; Brier {'full_fidelity': 0.7083, 'lean_v1': 0.3192, 'exp112_lean_v2': 0.0278, 'exp114_fidelity': 0.6944}

### Hormuz

- terminal_kind: unknown; structural_pass: True
- structural_invariants: {'evidence_present': True, 'packets_built': True, 'no_packet_leakage': True, 'behavior_grounded': True, 'structural_fidelity_not_broken': True}
- representation: {'real_member_count': None, 'represented_voting_power': None, 'total_voting_power': None, 'threshold': None, 'declared_threshold': None, 'n_decision_units': 0, 'candidates': None, 'faithful': None, 'verdict': None}
- deliberation: None
- evidence: {'n_facts': 6, 'dropped_leakage': 0, 'contradiction_groups': []}; packets: {'n': 3, 'leakage_flags': {}}
- outcome dimension: {'output_dimension': None, 'required_dimension': None, 'dimension_ok': None}
- structural_fidelity: ready {'resolution': 'ready', 'evidence': 'ready', 'behavior': 'ready'}
- FROZEN forecast: 0.125 (source mass_weighted:completed_rollouts+grounded_prior, status completed)
- guard: {'wall_s': 118.6, 'n_calls': 29, 'passed_hard_stop': True}
- AFTER-FREEZE outcome: 0; Brier {'full_fidelity': 0.7838, 'lean_v1': 0.7823, 'exp112_lean_v2': 0.25, 'exp114_fidelity': 0.0156}
