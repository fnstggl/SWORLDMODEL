# WMV2 Phase 6 — Research & Catalog

Mechanism ontology is driven by the **causal processes a general social world model must represent**, not by
which datasets happen to be on disk. The research-first priority matrix
(`registry/data/priority_matrix.json`, 49 families) determines implementation priority; datasets only
parameterize/validate the families that research establishes.

## 1. Methodology & primary-source rules

Eight parallel research agents swept the ontology (choice/learning, trust/relationship, bargaining/coalition,
participation/mobilization, opinion/belief, diffusion/contagion, network, platform/institutional/resource).
Each produced verified structured records with strict anti-fabrication rules: verify every source; never
invent a DOI/coefficient/SE/n/dataset; mark unverifiable numbers `verified:false` and store a formal model +
citation + null rather than a fabricated number. Result: **93/97 sources verified, 91 verified coefficients**
(`studies.json`, `coefficients.json`).

Primary-source priority: original peer-reviewed paper → official working paper → dataset documentation →
replication → meta-analysis → authoritative methodological text. Blogs/listicles are never coefficient
sources.

## 2. Core-agent accountability (safeguard #5)

I (the orchestrating agent) do not convert an agent summary into a pack. For every pack I minted, I
**independently verified the primary source myself** (WebSearch/WebFetch) before creating it. Record:

| Cluster | I read it | Sources I independently re-verified | Revised / rejected | Decision changed by research |
|---|---|---|---|---|
| participation | ✓ | GGL 2008 (control 29.7→Neighbors 37.8, exact levels), Karlan-List 2007 (+22% relative, ratio-flat), Johnson-Mislin (via trust) | Used turnout LEVELS not the agent's "+4.9pp Self" (authoritative = 34.5%→+4.8pp) | minted `social_pressure_turnout`, `matching_donation_response` as domain_restricted (NOT production) |
| bargaining_coalition | ✓ | Oosterbeek 2004 (offer 0.40, rejection 0.16) | my search query mis-stated rejection as 40%; source confirms **0.16** — corrected | minted `ultimatum_offer_response`; behavioral offer ≠ SPE encoded as a distinct family from `bargaining_rubinstein` |
| diffusion | ✓ | Sultan-Farley-Lehmann 1990 (p=0.03, q=0.38, SD≈mean) | Higgs Hill α=0.75 (<1) ⇒ complex contagion NOT present locally — did not overclaim | minted `bass_diffusion` with WIDE prior; kept Hawkes quarantined |
| trust_relationship | ✓ | Johnson-Mislin 2011 (sent 0.50, returned 0.37), Resnick 2006 (8.1% premium) | trust_repair 2×2 (Kim 2004) NOT representable by a scalar — logged as blocker, did not pack it | minted `trust_game_transfer`, `reputation_updating` |
| choice_learning | ✓ (summary + spot-check) | did NOT re-verify EWA/RL/belief coefficients myself | EWA δ/φ/ρ verified by agent from primary PDF but not core-verified → kept as research, no domain_restricted pack | left EWA/RL/belief/habit as `implemented` (no new pack) — status discipline |
| network | ✓ (summary) | did NOT re-verify Rajkumar/Aral/Kim magnitudes myself | weak ties inverted-U + targeting are structural/directional, not a transportable coefficient | minted `weak_tie_transmission`, `network_targeting_seeding` as **research_encoded** (no numeric pack) |
| opinion | ✓ (summary) | Kalla-Broockman persuasion ≈ 0 (direction) | DeGroot/HK/Deffuant params not empirically pinned → all `structural_candidate_only` | minted `persuasion_minimal_effects` as a **research_encoded guardrail** (broad near-zero) |
| platform_institutional | ✓ (summary) | did NOT re-verify Joachims/Fehr-Gächter myself | Upworthy identifies CONTENT-response, NOT position/examination bias; ~22% caching randomization failure noted | recorded `altruistic_punishment` as research_encoded; did not re-pack `platform_examination` this run |

**Unverified-and-therefore-not-packed** (stored as research only): EWA/RL/belief/habit/norm lab coefficients;
weak-tie/targeting/homophily magnitudes; Fehr-Gächter contribution levels (scanned PDF, could not core-verify);
Gamson slope. These are candidates for future promotion once core-verified and given a numeric pack.

**Deepened after the first tranche** (structural FORM now executable, magnitude still broad): the
`weak_tie_transmission` (inverted-U), `network_targeting_seeding` (friendship-paradox E[d²]/E[d]),
`altruistic_punishment` (punishment-sustains-cooperation), and `coalition_payoff_gamson` (portfolio≈seats)
families gained executable transitions in `registry/families/structural.py` — they are now
software-implemented (executable + tested) while remaining `research_encoded` (Tier-4 selectable) because
they lack a core-verified numeric pack / local validation. `position_bias_propensity` was **core-verified**
by me from the Joachims 2017 primary PDF (eq. 7: p(examine|rank)=(1/rank)^η, η=1) → a `domain_restricted`
pack.

## 3. Verified packs minted (each core-verified)

| Family (pack) | Estimate | Source (DOI) | Status |
|---|---|---|---|
| bass_diffusion (`sultan_farley_lehmann_1990`) | p=0.03, q=0.38 (SD≈mean) | Sultan-Farley-Lehmann 1990 JMR (10.1177/002224379002700107) | domain_restricted |
| ultimatum_offer_response (`oosterbeek_2004`) | offer 0.40, rejection 0.16 | Oosterbeek et al. 2004 Exp Econ (10.1023/B:EXEC.0000026978.14316.74) | domain_restricted |
| trust_game_transfer (`johnson_mislin_2011`) | sent 0.50, returned 0.37 | Johnson & Mislin 2011 JEP (10.1016/j.joep.2011.05.007) | domain_restricted |
| social_pressure_turnout (`ggl_2008_michigan`) | control 29.7% → Neighbors 37.8% | Gerber, Green & Larimer 2008 APSR (10.1017/S000305540808009X) | domain_restricted |
| matching_donation_response (`karlan_list_2007`) | +22% relative; ratio-flat | Karlan & List 2007 AER (10.1257/aer.97.5.1774) | domain_restricted |
| reputation_updating (`ebay_resnick_2006`) | 8.1% price premium | Resnick et al. 2006 Exp Econ (10.1007/s10683-006-4309-2) | domain_restricted |

## 4. Locally-fitted packs (real held-out, this run)

| Family (pack) | Dataset | Held-out | Identifies vs predicts |
|---|---|---|---|
| content_response_click (`upworthy_archive_2021`) | Upworthy Research Archive | pairwise 0.738 (in-dist) + 0.719 (time-fwd) | CAUSAL content-response ordering (randomized A/B); NOT position/examination bias |
| attrition_dropout_hazard (`telco_ibm_churn`) | IBM Telco churn | Brier .141 vs .198 | PREDICTIVE churn; NOT general relationship decay; cross-subpop transfer FAILS |
| response_occurrence_hazard (`stackexchange_answered`) | StackExchange | NULL (≈ base rate) | PREDICTIVE response occurrence; NOT trust/obligation |
| argument_persuasion_success (`cmv_delta`) | r/ChangeMyView | NULL (≈ base rate) | PREDICTIVE argument success; NOT causal persuasion levers |

Causal-identification honesty is enforced in code: each dataset pack in `wmv2_phase6_fits.json` records
`identifies`, `causally_identified`, `forbidden`, and `missing_variables`.

## 5. Famous models treated as candidate structural families (not universal laws)

EWA, QRE, DeGroot, bounded-confidence (HK/Deffuant), SIR/Hawkes, threshold contagion, generic RL/utility —
all treated as candidate structural families whose applicability + parameters must be established per
context. The matrix marks 19 as `structural_candidate_only`; DeGroot-vs-Bayesian carries the Chandrasekhar
2020 evidence that DeGroot averaging often fits better in the field, but no transportable W matrix exists.

## 6. Remaining research gaps

Position-bias propensity form (Joachims η), altruistic-punishment cost schedule, Gamson coalition slope, and
the choice/learning lab coefficients are **verified in `studies.json`/`coefficients.json`** but await
core-verification + an executable numeric transition before promotion. Full seed-map coverage and the
100–200-pack target are not met this run (see FINAL_REPORT §4).
