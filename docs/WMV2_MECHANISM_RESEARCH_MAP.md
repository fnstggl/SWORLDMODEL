# WMv2 Mechanism Research Map

*Primary-literature basis for each mechanism family, with the exact transport limits each citation
licenses. Every family in the registry carries these as `Citation(ref, finding, study_population, limits)`.
A citation alone never licenses production use — only a PASSED held-out record does (see the registry).*

## Diffusion / contagion

| family | primary source | imports | transport limit |
|---|---|---|---|
| exposure_response_hazard | Romero, Meeder & Kleinberg 2011 (WWW) | concave, content-dependent exposure response | Twitter hashtags 2009; refit per cascade |
| complex_contagion_hazard | Centola & Macy 2007 (AJS 113:3) | multiple-exposure social reinforcement (Hill α>1) | online health-behavior experiment; α,c fit per setting |
| simple_contagion_hazard | classic SI/threshold | linear-in-exposure baseline | known too simple; comparator only |
| threshold_adoption | Granovetter 1978 (AJS 83:6) | heterogeneous-threshold cascades | threshold distribution must be estimated |
| hawkes_self_excitation | Hawkes 1971; Zhao et al. 2015 (SEISMIC, KDD) | self-exciting retweet intensity | constant μ + single exp kernel underfit bursts (QUARANTINED after held-out failure) |
| information_aging | novelty-decay literature | age-weighted exposure half-life | τ fit on a grid |
| finite_population_saturation | SIR-family finite-N | susceptible depletion | population size must be known |
| network_rewiring | Snijders SAOM 2010 | homophily + tie-age edge co-evolution | parameters labeled priors |

## Choice / strategic

| family | primary source | imports | transport limit |
|---|---|---|---|
| social_preference_population | Fehr & Schmidt 1999 (QJE 114) | inequity-aversion (α,β) type distribution | 1990s lab ultimatum, students; comonotonic assumption; lab stakes |
| quantal_response_choice | McKelvey & Palfrey 1995 (GEB 10) | logit QRE | λ payoff-scale-dependent; refit per game |
| bargaining_rubinstein | Rubinstein 1982 (Econometrica 50) | alternating-offers SPE split | complete-information, stationary discounting |
| negotiation_concession | Faratin, Sierra & Jennings 1998 (RAS) | time-dependent concession tactic | β (tough/conceder) fit per negotiator |
| coalition_formation | Banzhaf 1965; Riker 1962 | voting power + minimal-winning coalition | weighted-voting bodies |

## Learning / adaptation

| family | primary source | transport limit |
|---|---|---|
| reinforcement_learning | Watkins 1989; Erev & Roth 1998 (AER 88) | α fit; stationary environment |
| belief_learning | Fudenberg & Levine 1998 | stationary opponent |
| experience_weighted_attraction | Camerer & Ho 1999 (Econometrica 67) | 4 params; needs many rounds to identify |
| habit_formation | Wood & Neal 2007 (Psych Review 114) | γ context-dependent |

## Social / relational / opinion

| family | primary source | transport limit |
|---|---|---|
| trust_formation / violation / repair | Slovic 1993 (Risk Analysis 13); Kim et al. 2004 (JAP 89) | asymmetry rates are reference-class priors; refit per relationship |
| reciprocity | Fehr & Gächter 2000 (AER 90) | rate fit per dyad |
| degroot_influence | DeGroot 1974 (JASA 69); Chandrasekhar et al. 2020 (AER) | people are NOT naive DeGroot updaters — candidate only |
| bounded_confidence | Hegselmann & Krause 2002 (JASSS 5) | ε must be fit/varied |
| latent_expressed_opinion | Kuran 1995 (Private Truths, Public Lies) | pressure/conviction scale unobserved |

## Participation / institution / platform / measurement

| family | primary source | transport limit |
|---|---|---|
| voting_turnout / mobilization | Riker & Ordeshook 1968 (APSR 62); Gerber & Green 2000 (APSR 94) | coefficients reference-class; refit per electorate |
| donation_response | fundraising practice | reference-class form; refit per campaign |
| platform_examination | Craswell et al. 2008 (WSDM) | γ (position bias) fit per platform |
| institutional_vote / agenda_stage_control | institutional procedure (executable rules) | rules must be retrieved, not fabricated |
| gaussian_measurement / bernoulli_detection | measurement-error modeling | house effects fit from measurement-vs-truth pairs |

## The stance on "universal laws"

Per the engineering rules, DeGroot, bounded confidence, SIR, QRE, EWA, Hawkes and generic threshold models
are treated as **candidate structural families whose parameters and applicability must be validated**, NOT
human laws. This is enforced: every one is registered with `transport_risk="high"`, published coefficients
carry their study limits, and none is production-eligible without a passed local held-out record. To date
exactly two families cleared that bar (exposure_response_hazard, engagement_momentum_persistence), and one
was quarantined for failing it (hawkes_self_excitation).
