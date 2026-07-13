# WMv2 Phase 12 — Forensic Traces
*Stratified end-to-end traces assembled from the frozen Phase-12 artifacts. Each shows the full chain from question to calibrated user-facing result so a reviewer can confirm the output came from the real max-capacity posterior simulation (not a disguised direct forecast). Machine-readable: `experiments/results/phase12/forensic_traces.json`.*

## `phase3acc_locked:indonesia_2024` — elections
**Q:** Will Prabowo Subianto win Indonesia's 2024 presidential election in a single round?  
as_of **2024-02-10**, horizon **39d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.3286** → calibrator **identity** → calibrated **0.3286** (provisional, eff cal n=75)
- **support grade** `highly_speculative` (expected_error=0.2968)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.85, 'ensemble_p': 0.8209, 'disagreement': 0.521, 'flag': 'large_disagreement', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:rba_nov24` — econ
**Q:** Will the Reserve Bank of Australia hold its cash rate at its November 2024 meeting?  
as_of **2024-10-28**, horizon **8d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.7375** → calibrator **identity** → calibrated **0.7375** (provisional, eff cal n=75)
- **support grade** `transfer_supported` (expected_error=0.2102)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.85, 'ensemble_p': 0.85, 'disagreement': 0.112, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:ez_recession_2024` — macro
**Q:** Will the Eurozone enter a technical recession during 2024?  
as_of **2024-01-15**, horizon **351d**, realized outcome **0**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.5077** → calibrator **identity** → calibrated **0.5077** (provisional, eff cal n=75)
- **support grade** `transfer_supported` (expected_error=0.22)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.35, 'ensemble_p': 0.3824, 'disagreement': 0.158, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:silver_40_2024` — finance
**Q:** Will silver exceed 40 US dollars per ounce in 2024?  
as_of **2024-05-01**, horizon **244d**, realized outcome **0**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.5333** → calibrator **identity** → calibrated **0.5333** (provisional, eff cal n=75)
- **support grade** `transfer_supported` (expected_error=0.2042)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.15, 'ensemble_p': 0.15, 'disagreement': 0.383, 'flag': 'large_disagreement', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:vision_pro_2024` — tech
**Q:** Will Apple release the Vision Pro headset in 2024?  
as_of **2024-01-05**, horizon **361d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.7273** → calibrator **identity** → calibrated **0.7273** (provisional, eff cal n=75)
- **support grade** `empirically_supported` (expected_error=0.1369)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.95, 'ensemble_p': 0.95, 'disagreement': 0.223, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:nk_nuke_2024` — geopolitics
**Q:** Will North Korea conduct a nuclear weapons test in 2024?  
as_of **2024-02-01**, horizon **334d**, realized outcome **0**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.7237** → calibrator **identity** → calibrated **0.7237** (provisional, eff cal n=75)
- **support grade** `empirically_supported` (expected_error=0.1084)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.15, 'ensemble_p': 0.15, 'disagreement': 0.574, 'flag': 'large_disagreement', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:worldseries_2024` — sports
**Q:** Will the Los Angeles Dodgers win the 2024 World Series?  
as_of **2024-10-22**, horizon **10d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.3333** → calibrator **identity** → calibrated **0.3333** (provisional, eff cal n=75)
- **support grade** `highly_speculative` (expected_error=0.2731)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.35, 'ensemble_p': 0.3146, 'disagreement': 0.017, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:chandrayaan3_2023` — science
**Q:** Will India's Chandrayaan-3 successfully soft-land near the Moon's south pole in 2023?  
as_of **2023-08-01**, horizon **30d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.6552** → calibrator **identity** → calibrated **0.6552** (provisional, eff cal n=75)
- **support grade** `transfer_supported` (expected_error=0.1943)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.75, 'ensemble_p': 0.7042, 'disagreement': 0.095, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row

## `phase3acc_locked:sbf_2023` — politics
**Q:** Will Sam Bankman-Fried be convicted of fraud in 2023?  
as_of **2023-10-01**, horizon **91d**, realized outcome **1**
- **active components** — ON: as_of_evidence, evidence_conditioned_compile, posterior_hidden_state, structural_hypotheses; OFF/not-wired: population_heterogeneity, multilayer_networks, executable_institutions, learned_actor_policies, persistence, nonlinear_mechanisms, dynamic_recompilation
- raw forecast **0.8** → calibrator **identity** → calibrated **0.8** (provisional, eff cal n=75)
- **support grade** `exploratory` (expected_error=0.2493)
- **uncertainty decomposition** (epistemic 0.08333, aleatoric 0.16667): {'parameter_hidden_state': 0.08333, 'evidence': 0.0, 'structural': 0.0}
- **direct-model disagreement (critic)**: {'direct_p': 0.85, 'ensemble_p': 0.85, 'disagreement': 0.05, 'flag': 'ok', 'note': 'critic annotates only; never overwrites the simulation number'}
- **limitations**: provisional calibrator (pre-Phase-11); Phases 8/9/11 not on the forecast path for this row
