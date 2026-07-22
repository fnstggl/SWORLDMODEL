# Phase 6 Audit Notes (Part 0) — preserved facts

## Registry baseline (from build_registry.build())
- 47 named families; ALL executable (code_ref resolves) + have test_ref -> reach `implemented`
- packs: 6 total across 6 families
- by_status: implemented=43, production_eligible=2, locally_validated=1, quarantined=1
- production_eligible: engagement_momentum_persistence (OmniBehavior held-out+transfer),
  social_preference_population (BehaviorBench transfer passed)
- locally_validated: exposure_response_hazard (Higgs held-out passed; blocked from prod by NO transfer record)
- quarantined: hawkes_self_excitation (held-out count forecast FAILED vs Poisson; PRESERVED)
- 41 families have NO validation history

## Promotion gates (record.py promotion_blockers) — enforced, honest
- implemented: executable + test_ref + formal_description
- locally_validated: + >=1 pack + held_out/PPC record (passed is not None)
- transfer_validated: + transfer record
- production_eligible: + citation + PASSED held_out/transfer record

## Compiler integration flaw (compiler.py ~417-429)
- _score_production_registry(scenario) calls rank_mechanisms(load_registry(), scenario)
- BUG: applicable = selected[0] (single top-ranked scenario family) applied to EVERY required
  causal process. No PER-PROCESS matching. select_tier gets has_domain_pack=bool(applicable)
  (treats any selected family as tier-2 even without a pack). Fix: per-causal-process selection.

## Datasets actually on disk (raw NOT committed; these ARE committed parsed sets + fit artifacts)
- experiments/results/exp054_upworthy/upworthy_parsed.json  (4863 randomized A/B headline tests; CC-BY)
- experiments/results/exp077/upworthy_ab.json                (1.9MB alt Upworthy A/B)
- experiments/results/harvest_extra/telco_churn.json         (7032 rows; y=churn; base 0.266)
- experiments/results/harvest_extra/stackexchange.json       (2500 rows; y=answered; base 0.576)
- experiments/results/exp021_cmv/cmv_common.json             (1200 CMV challenger args; success label 781/419)
- experiments/results/exp072/baby_names.json                 (name -> year -> fraction; 1880+)
- experiments/results/exp028_oqa/oqa_parsed.json             (9.9MB OpinionQA)
- experiments/results/harvest_extra/globalopinions.json      (3MB cross-national opinion)
- Higgs/OmniBehavior/BehaviorBench: raw absent, but fitted coefs live in wmv2_*.json artifacts
- data hosts reachable via proxy (SNAP 200, OSF 200)

## Real fitted coefficients already committed (to embed into packs, not re-point)
- Higgs loglinear_theta (5), linear_q=0.0224, hill(theta0/alpha/c), aging_tau=48h, frailty_sigma=0
- Upworthy surface_w=[0.0312,-0.0015,-0.4205,0.0753,0.1089,0.0707], random_p1=0.336
- OmniBehavior momentum_lift=6.777
- BehaviorBench per-game W1 norms + calibration coverage

## DATASET -> MECHANISM causal-identification honesty (per user correction)
- Upworthy: RANDOMIZED traffic -> CTR winner is CAUSAL for CONTENT-RESPONSE/attention-capture; NOT
  examination/position bias (position not manipulated here). Supports content_response / click_propensity.
- StackExchange: y=answered is a RESPONSE-OCCURRENCE hazard (predictive/observational); NOT trust/obligation.
- CMV: success = argument earns delta; persuasion/argument-response; platform-specific; matched design.
- Telco churn: attrition/dropout transition (observational/predictive); NOT general relationship decay.
- Higgs: diffusion timing / nonlinear hazard / aging / cascade; preserve Hawkes failure.
- Baby names: cultural adoption / popularity diffusion; NOT persuasion.
