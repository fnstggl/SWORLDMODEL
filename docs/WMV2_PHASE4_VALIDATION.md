# WMV2 Phase 4 — Validation, Failures, and Production Verdict

## Verdict

**HOLD.** Software and end-to-end execution are implemented. Phase 4 is neither empirically validated nor production eligible.

The final frozen run used commit `b3823d4284c63d3e35eafd51ae56ca3ca03a676c`, seed `404`, all committed rows, train-only fitting, calibration-only temperature selection, and untouched test labels. It produced 21 checksum-sealed artifacts under `experiments/results/phase4/`.

| Status | Result |
|---|---:|
| Software implemented | Yes |
| Executes through shared world | Yes |
| Empirically validated | No |
| Production eligible | No |

## Datasets and evidence limits

| Dataset | Setting and population | Records | Available pre-action state | Action-set mapping | Primary split | Critical limitation |
|---|---|---:|---|---|---|---|
| CMV | Communication/persuasion; ChangeMyView original posters and challengers | 1,200 | OP/challenger text, IDs, dyad, timestamp | `award_delta` / `hold_position` | Person-disjoint | Delta is reconstructed from the eventual outcome field, not a timestamped button event; private beliefs and full thread state are absent. |
| OpinionQA | Repeated survey choice; American Trends Panel respondents over 15 waves | 23,340 | respondent UID, question ID, wave, demographics | `select_option_0` / `select_option_1` | Person-disjoint | Cache lacks question/option text and polarity; only indexed choice is valid, not support/oppose semantics. |
| Upworthy | Randomized platform exposure; weighted audience impressions | 45,243 aggregate action rows | headline text, test/arm identity, impression exposure | `click` / `ignore` | Context-disjoint | Aggregate binomial counts, no individual identity/history/device/position or archive problem flag. Test effective weight is 12,283,525 impressions. |

Committed input hashes begin `ed5048e78afe` (CMV), `1304c09d0867` (OpinionQA), and `c0be6af9e616` (Upworthy). Source/license/access, population/time, missing state, action reconstruction, network/institution information, risks, and limitations are in `dataset_manifests.json`.

These are three different observed choice settings, but they are not three rich causal actor-world reconstructions. OpinionQA in particular is semantically impoverished. That is one reason the overall empirical status remains false.

## Frozen splits and leakage controls

| Dataset | Method | Train | Calibration | Validation | Test | Identity/context separation |
|---|---|---:|---:|---:|---:|---|
| CMV | Person-disjoint | 723 | 176 | 121 | 180 | Original-poster IDs do not overlap across partitions. |
| OpinionQA | Person-disjoint | 14,096 | 3,456 | 2,277 | 3,511 | Respondent UIDs do not overlap across partitions. |
| Upworthy | Context-disjoint | 27,212 | 6,800 | 4,460 | 6,771 | Test IDs do not overlap; impression counts remain row weights. |
| CMV secondary | Time-forward | 720 | 180 | 120 | 180 | Every training timestamp precedes calibration/validation/test timestamps. |

The split code rejects post-action feature flags, label-feature flags, duplicate candidate actions, and observed actions missing from the reconstructed historical set. Test actors/contexts are excluded from fitting; test labels are not used for selection, calibration, feature construction, or promotion. Exact IDs and checksums are in `split_manifests.json`.

No row was silently dropped to make results better. Upworthy rows with zero count for an action are omitted because they represent zero observations, while all positive click/ignore counts remain as frequency weights.

## Baselines

| Arm | Status | Definition |
|---|---|---|
| B0 majority/frequency | Run | Training empirical action frequency over the row’s candidate set. |
| B1 reference class | Run | Training role-conditioned frequency, falling back to global. |
| B2 raw LLM | Unavailable | No frozen, same-row, actor-visible LLM output cache or model/API configuration was available. It was not fabricated after seeing test labels. |
| B3 LLM panel | Unavailable | No frozen same-row observer-panel cache was available. |
| B4 handcrafted | Run | Repository-style transparent domain heuristic using only allowed features. |
| B5 flat fitted | Run | Global fitted action model plus permitted pre-action features, no hierarchy. |
| B6 hierarchical, no execution | Run | Training-only partially pooled prediction plus calibration, but detached from shared-world execution. |
| B7 full | Run | Same universal fitted pack through actor view, feasibility, consequences, calibrated posterior, sampled typed action, event, delta, and reaction machinery. |
| B8 specialist ceiling | Unavailable on exact rows | Prior specialist artifacts do not provide leakage-safe predictions for these exact frozen rows and remain separate negative/comparison evidence. |

The absence of B2/B3 alone blocks the required raw-LLM improvement gate.

## Held-out action prediction

Log loss is lower-is-better. Accuracy and ECE are reported for context, but extreme imbalance makes Upworthy accuracy uninformative.

| Dataset | B0 log loss | B4 | B5 | B6 | B7 | B7 Brier | B7 ECE | B7 top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CMV | 0.623818 | 1.156356 | 0.623438 | **0.617355** | 0.617585 | 0.426216 | 0.030272 | 0.694444 |
| OpinionQA | 0.694283 | **0.693147** | 0.694283 | 0.693478 | 0.693478 | 0.500330 | 0.013029 | 0.497864 |
| Upworthy | 0.079365 | 0.086929 | 0.079365 | 0.079365 | 0.079365 | 0.030238 | 0.000284 | 0.984645 |

Interpretation:

- CMV B7 beats the handcrafted arm strongly but not B5 credibly; B6 is slightly better than B7.
- OpinionQA B7 credibly beats B5 by a tiny amount, is identical to B6, and does not beat the uniform handcrafted arm. Missing option semantics make this a weak product result.
- Upworthy B7 beats the handcrafted heuristic, but is statistically indistinguishable from B5/B6/frequency. The apparent 98.5% accuracy is simply the click/ignore imbalance.
- B7’s primary incremental value in this run is execution semantics. It does not add credible predictive value over B6.

### Paired bootstrap confidence intervals

Values are paired differences `B7 loss − baseline loss`; a fully negative interval favors B7. Aggregate action rows are resampled as trajectory/arm clusters while retaining frequency weights.

| Dataset | B7 − B4 mean [95% CI] | B7 − B5 | B7 − B6 |
|---|---|---|---|
| CMV | −0.538771 [−0.672871, −0.405220] | −0.005853 [−0.014527, 0.003544] | 0.000231 [−0.000336, 0.000797] |
| OpinionQA | 0.000330 [−0.000439, 0.001038] | −0.000805 [−0.001513, −0.000135] | 0.000000 [0.000000, 0.000000] |
| Upworthy | −0.007565 [−0.008302, −0.006735] | approximately 0 [−5.46e−9, 1.21e−8] | approximately 0 [−6.60e−9, 1.19e−8] |

No aggregate cross-domain superiority claim is made: impression-weighting would let Upworthy dominate an aggregate score, while equal-domain weighting would obscure different label semantics.

## Calibration and reliability

Temperature was selected on each calibration partition only.

| Dataset | Temperature | Calibration pre log loss | Post log loss | Test B7 ECE | Result |
|---|---:|---:|---:|---:|---|
| CMV | 0.8 | 0.640006 | 0.637820 | 0.030272 | Calibration improves held-out B7 log loss versus uncalibrated 0.623818 to 0.617585. |
| OpinionQA | 2.0 | 0.693617 | 0.693145 | 0.013029 | Calibration improves test log loss from 0.694283 to 0.693478. |
| Upworthy | 1.0 | 0.078111 | 0.078111 | 0.000284 | No change; frequency fit was already calibrated at this resolution. |

Reliability-bin weights/confidence/accuracy are preserved in `reliability_data.json`. Calibration is better or equal here, but policy credible-interval coverage and family-posterior calibration were not evaluated. The policy-uncertainty empirical gate is therefore false.

## Cold start, temporal behavior, and transfer

### Cold start

- CMV person-disjoint B7: log loss 0.617585. Versus B5 the CI crosses zero, so this is not a credible cold-start win.
- OpinionQA person-disjoint B7: 0.693478 versus B5 0.694283 with CI [−0.001513, −0.000135]. This is one small credible flat-baseline win, but it does not beat uniform B4 and has no question/option semantics.
- Upworthy has no person identifiers and cannot support a person-cold-start claim.

This is not two credible cold-start/transfer wins.

### Temporal CMV

The time-forward hierarchical policy records log loss 0.668432, Brier 0.475309, ECE 0.035507, and top-1 accuracy 0.616667 on 180 later decisions. It demonstrates temporal splitting, not sequence-model superiority.

### Cross-domain transfer

Only a semantically defensible binary active/passive mapping was attempted. Both directions fail:

| Transfer | Log loss | ECE | Top-1 | Verdict |
|---|---:|---:|---:|---|
| CMV → Upworthy | 1.166204 | 0.676953 | 0.015355 | Catastrophic negative transfer. |
| Upworthy → CMV | 2.904963 | 0.679090 | 0.305556 | Catastrophic negative transfer. |
| OpinionQA | — | — | — | Excluded: option index has no stable active/passive polarity. |

The failed transfers are preserved in `transfer_results.json` and directly block production eligibility.

## Feasibility, execution, and causal integration

| Dataset | Selected invalid action | StateDelta rate | Terminal quantity-effect rate |
|---|---:|---:|---:|
| CMV | 0.0% | 100% | 100% |
| OpinionQA | 0.0% | 100% | 100% |
| Upworthy | 0.0% | 100% | 100% |

The real-row terminal quantity is an execution counter, not an estimate of a real causal outcome. It proves the selected label cannot remain detached; it does not prove intervention validity.

Unit/adversarial tests additionally show:

- exactly zero probability on actor-known impossible actions;
- zero selection outside the perceived feasible mask;
- mistakenly perceived-feasible but actually invalid attempts become `action_blocked` events and deltas;
- resource costs and commitments alter world state;
- institution rules are rechecked against actual state;
- reactions enter the next actor’s view and the same policy operator;
- adaptation/history changes later probabilities;
- fixed inputs/seeds replay deterministically and trace corruption is detected.

Eight software integration fixtures cover messaging, negotiation, organizational approval, election participation, legislation, acquisition, platform interaction, and coalition mobilization. They create a selected action, event/delta, institution/reaction events, a later counterparty decision, and a positive or negative terminal quantity effect. They are labeled synthetic software evidence and are not counted as empirical validation.

## Required 20 ablations

All 20 slots are recorded; “recorded” does not mean scientifically identified.

| # | Ablation | Result |
|---:|---|---|
| 1 | Raw LLM | Unavailable; no frozen same-row actor-visible cache. Core-ablation gate fails. |
| 2 | Heuristic policy | B4 results reported; loses credibly on CMV/Upworthy, not OpinionQA. |
| 3 | Flat fitted | B5; B7 credibly improves only OpinionQA, not CMV/Upworthy. |
| 4 | Hierarchical policy | B6; B7 has no predictive lift, only execution effects. |
| 5 | No actor history | Ornamental: strict cold-start rows provide no fitted test-actor history. |
| 6 | No actor beliefs | Ornamental: available caches make the fitted runtime action-intercept dominated. |
| 7 | No relationship state | Ornamental: no validated relationship state beyond IDs/dyads. |
| 8 | No network state | Ornamental: no reconstructable network layers in these rows. |
| 9 | No institutional constraints | Ornamental: reconstructed observed options are all permitted. Software permission tests remain meaningful. |
| 10 | No persistent policy state | Ornamental: held-out outcomes are not consumed online. |
| 11 | No subjective reactions | Ornamental: one-step caches lack reaction labels. |
| 12 | No strategic anticipation | Ornamental: no identified opponent-response model in caches. |
| 13 | No habit/reinforcement | Ornamental on person-disjoint rows; repeated-world unit test changes behavior. |
| 14 | No family uncertainty | Ornamental: family structures are not identified by these caches. |
| 15 | No person shrinkage | Uses B5 comparison; one tiny OpinionQA gain, no second credible win. |
| 16 | No feasibility mask | Ornamental on observed rows; adversarial software test establishes the causal difference. |
| 17 | No calibration | Hurts CMV and OpinionQA, unchanged Upworthy. |
| 18 | Point world | Ornamental: committed caches do not reconstruct full posterior particles. Multi-particle software test changes probabilities. |
| 19 | No shared-world execution | Same B6 predictions, zero deltas and terminal effects. |
| 20 | Full policy | B7 predictions plus 100% action-delta and fixture terminal-effect rates. |

Because many state removals do not alter the available empirical rows, the “all core ablations run” empirical gate is false, not silently passed.

## Negative-result preservation and quarantine

Phase 4 did not overwrite old results. The quarantine artifact stores the original path and SHA-256 for existing Enron, BehaviorBench, Higgs, Omnibehavior, and Phase 3 backtest results where present.

New preserved failures:

- no valid B2/B3 output;
- CMV action reconstructed from outcome;
- OpinionQA missing semantics;
- Upworthy aggregate rather than individual behavior;
- no identified rich-state ablations or uncertainty coverage;
- no credible aggregate B7 lift over B6;
- B5 non-inferiority only, not superiority, on CMV/Upworthy;
- two catastrophic negative transfers.

The prior Enron actor-policy evidence remains quarantined for future-history leakage in its loader; the prior Omnibehavior persistence effect remains near zero. No one-step result was used to reject a universal structural family that the data cannot identify.

## Tests

| Suite | Result | Classification |
|---|---|---|
| Phase 4 focused | 39 passed | Green. |
| Directly affected V2/adjacent phases | 249 passed | Green after explicitly migrating the legacy `agent_decision` expectation. |
| Complete repository | 930 passed, 3 failed, 11 warnings | No introduced Phase 4 regression. |

Full-suite failure inventory:

1. `test_dataset_registry_is_valid_and_honest`: pre-existing missing `data/dataset_registry.json`.
2. `test_apply_toggles_ungrounds_and_limits_vars`: pre-existing backtest-toggle behavior; relevant source/tests are unchanged from base.
3. `test_predict_and_rollout_are_distinct`: environment/optional dependency, `fastapi` absent.

An allocation-sensitive legacy observer-panel test failed once in a selected-suite run because it keys ephemeral objects by `id(self)`; it passed in the final 249-test affected suite and in the complete run. Its code is unchanged from base.

## Production acceptance gates

### Architecture

| Gate | Result | Evidence/limit |
|---|---:|---|
| One universal architecture; no benchmark router | Pass | All B7 rows use the same runtime and typed contracts. |
| Actor views exclude omniscient state | Pass | Fail-closed projection and adversarial tests. |
| Scenario-specific action sets | Pass | Compiler/live capability construction; row-specific historical sets. |
| Known-impossible actions get zero mass | Pass | Mask tests and 0% selected invalid rate. |
| Institution handles actual constraints | Pass | Actual recheck and explicit blocked delta. |
| LLM does not mint probabilities | Pass | Numeric fields have no typed route and migration rejects them. |
| Parameters have provenance/uncertainty | Pass | Fitted/broad packs and trace fields. |
| Policy-family uncertainty represented | Pass (software) | Family-specific distributions and structural posterior records. |
| Posterior particles affect probabilities | Pass (software) | Multi-particle test; not validated on real full-particle caches. |
| Actions create events/deltas/reactions | Pass | Tests and eight traces. |
| History affects later behavior | Pass (software) | Adaptation/history removal test. |
| Actions affect terminal state | Pass (software) | Typed non-probability quantities; no real causal-effect claim. |
| Deterministic replay/fallbacks/no forecast abstention | Pass | Fixed-seed/checksum tests and Tier-7 typed fallback. |

### Software

| Gate | Result |
|---|---:|
| Versioned APIs, serialization/migration | Pass |
| Structured trace IDs/log fields | Pass |
| Bounded timeout/retries where applicable | Pass |
| Corruption checks and atomic/concurrent writes | Pass |
| Resumable evaluation | Pass |
| Artifact hashes, costs, latency | Pass |
| No introduced test regression | Pass |

### Empirical/production

| Gate | Result | Reason |
|---|---:|---|
| Three real held-out domains | Pass, limited | Three settings ran, but one lacks choice semantics. |
| Person-disjoint evaluation | Pass | CMV and OpinionQA. |
| Temporal/sequence-disjoint | Pass | Time-forward CMV. |
| Cold-start measured | Pass | Two person-disjoint sets; only one tiny flat-baseline win. |
| Cross-domain measured | Pass | Both measured directions fail. |
| Held-out calibration | Pass | Brier/ECE/reliability and separate calibration sets. |
| Invalid actions measured | Pass | 0%; adversarial blocked behavior also tested. |
| Policy uncertainty evaluated | **Fail** | Representation exists, but interval/family coverage is not empirically measured. |
| All core ablations | **Fail** | Raw LLM unavailable; many real-state ablations unidentified/ornamental. |
| Negative transfer/families preserved | Pass | Artifacts retain failures and old hashes. |
| Credible aggregate win over raw LLM | **Fail** | B2 unavailable. |
| Credible aggregate win over handcrafted | **Fail** | Two domain wins, one no-win, and no justified aggregate metric. |
| Non-inferiority/improvement to strongest flat/general fit | Partial, not enough | B7 is essentially B6 predictively and not credibly better than B5 in two domains. |
| Positive value from hierarchy/actor state/constraints | **Fail overall** | Tiny OpinionQA pooling gain; rich state is absent elsewhere. |
| Two transfer or cold-start wins | **Fail** | One tiny cold-start win; both transfers fail. |
| Calibration not materially worse | Pass | Better on two, equal on one. |
| Zero leakage/known-impossible selection | Pass (software evidence) | Adversarial tests and real invalid diagnostics. |
| Demonstrated deltas/later behavior | Pass (software evidence) | Tests and integration traces. |

## Exact remaining experiments

1. Freeze and run B2/B3 on the same actor-visible rows before touching test outcomes.
2. Acquire timestamped messaging decisions with reconstructed inbox/action sets and response timing.
3. Add sequential strategic/game trajectories with actions, payoffs, opponent observations, and held-out games.
4. Add institutional decisions with actual permissions, agenda stages, sanctions, and institution-disjoint splits.
5. Reconstruct real Phase 3 posterior particles and test uncertainty/interval coverage.
6. Fit and compare identifiable family structures, including EWA/RL/habit and bounded strategic response.
7. Run non-ornamental component ablations where each removed field truly changes the actor view or execution.
8. Establish at least two credible cold-start or transfer wins and a justified aggregate baseline comparison.

Until then, the correct product decision is **do not merge as production-ready and do not promote any Phase 4 policy family to production eligible**.

## 2026-07-13 empirical-completion validation

The completion protocol was committed and pushed before final evaluation in
commit `1322d2d`. Its canonical payload checksum is
`ec3a661177219a6f6a7c7642cd637044010f095e77d83cf81ee1e2d514f7d06c`.
The original 21-artifact namespace remains byte-for-byte untouched, with tree
hash `9d946bf3e77e829706ab3bd0e9cba723f1a4b4d7`; the frozen audit is recorded
separately under `phase4_completion/`.

### Data and held-out scale

| Dataset | Real decisions evaluated | Held-out decisions | Main split discipline |
|---|---:|---:|---|
| IPD longitudinal | 18,800 | 2,600 | chronological sessions/rounds; fixed and shuffled partner regimes retained |
| VoteView Senate | 283,675 | 62,175 | Congress 115 train, 117 family validation, 118 test |
| Enron repaired | 309,441 | 59,598 | time-forward split, seven-day label maturation, and temporal purge |
| **Total** | **611,916** | **124,373** | untouched test partitions after model-family selection |

### B0–B8 calibrated test log loss

Lower is better. B2/B3 are omitted from the confirmatory table because their
strict-schema coverage gate failed; conditional metrics are shown separately
and are selection-biased diagnostics.

| Dataset | B0 | B1 | B4 | B5 flat | B6 hierarchy | B7 actor policy | B8 specialist |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPD | 0.692613 | 0.692613 | 0.601129 | 0.683103 | 0.403027 | **0.403027** | 0.409799 |
| Senate | 0.826259 | 0.826259 | 0.797018 | 1.058703 | 0.998450 | **0.700671** | 1.491123 |
| Enron | 0.396829 | 0.401407 | 0.461165 | 0.415117 | 0.378601 | **0.372049** | 0.401231 |

The equal-domain B7−B6 macro log-loss delta is approximately **−0.10144**.
B7 is non-inferior at the frozen 0.01 margin in all three domains. Clustered
B7−B6 comparisons are:

| Dataset | Mean delta | Cluster-bootstrap 95% CI | Result |
|---|---:|---:|---|
| IPD | −0.000000 | [−0.000000, +0.000000] | exact practical tie; only two held-out session clusters make this CI weak |
| Senate | −0.297779 | [−0.334652, −0.261329] | superiority |
| Enron | −0.006552 | [−0.009055, −0.004037] | superiority |

B7 also beats the flat B5 log loss in every domain, with mean deltas
−0.280076 (IPD), −0.358031 (Senate), and −0.043069 (Enron). It beats the
preregistered handcrafted B4 in every domain. This does not erase the fact that
IPD B7 is exactly B6 and that no consequence component survived validation.

### DeepSeek B2/B3

The frozen design contained 288 action-stratified packets and five lenses per
packet: 1,440 request identities, with up to two retries. All request identities
completed. Of them, 1,181 produced strictly valid frozen-schema responses and
259 did not. The run preserved 2,259 corrected-run raw attempts plus 33
preflight attempts. The preflight exposed a provider-default drift toward
thinking mode; the wire request was corrected to explicitly disable thinking,
which enforces rather than changes the preregistered non-thinking intent.

| Dataset | B2 test coverage | B3 panel test coverage | Cost-matched lens coverage |
|---|---:|---:|---:|
| IPD | 85.94% (55/64) | 40.63% (26/64) | 79.69% (51/64) |
| Senate | 54.69% (35/64) | 29.69% (19/64) | 75.00% (48/64) |
| Enron | 92.19% (59/64) | 53.13% (34/64) | 89.06% (57/64) |

Conditional calibrated log losses are stored in
`llm_baseline_results.json`, but they are diagnostic only because provider
schema compliance selects the covered rows. Consequently there is **no valid
confirmatory B7-versus-B2/B3 conclusion**.

### Uncertainty, particles, sequences, and outcomes

| Dataset | 90% conformal coverage | Mean set size | Particle collapse | Mean action TV across particles |
|---|---:|---:|---:|---:|
| IPD | 89.62% | 1.194 | 0 | 0.00000 |
| Senate | 84.97% | 1.441 | 0 | 0.03920 |
| Enron | 88.16% | 0.991 | 0 | 0.01461 |

All runs used 64 unique Phase 3 particles per decision, equal weights, ESS
fraction 1.0, maximum particle weight 1/64, and no collapse. Yet every domain
misses the nominal 90% conformal target. IPD assigns zero weight to the particle
component; point-world/no-particle comparisons do not show uncertainty value.
The correct verdict is that particles were consumed and numerically stable,
but posterior uncertainty was **not empirically validated**.

Sequential scoring ran on the chronological test rows. Downstream diagnostics
include IPD payoff MAE 1.07535 and opponent-reaction log loss 0.50795, Senate
passage Brier 0.241785/log loss 0.677830 across 691 rolls, and Enron reply-delay
MAE 24.765 hours. These are observational predictions. There is no held-out
intervention demonstrating that B7 execution improved later decisions or
terminal outcomes.

### Transfer, cold start, and ablations

VoteView provides temporal transport to Congress 118; Enron provides a
time-forward test and cold-start slices. IPD fixed/shuffled regime slices were
measured, but fixed→shuffled and shuffled→fixed refits were not run. Therefore
the transfer gate fails as incomplete, and the evidence does not contain two
credible transfer/cold-start wins.

The family-selection ablation is non-ornamental: IPD selects hierarchy only,
Senate propensity particles only, and Enron a 50/50 hierarchy/propensity mix.
The consequence family receives zero weight everywhere. Point-world,
no-particle, and no-consequence comparisons do not establish value for four
rich world-state components. The historical execution inputs are identical to
prediction inputs, but execution itself has no empirical causal lift.

### Costs and tests

The corrected LLM collection made 2,259 HTTP attempts and used 1,392,703
provider-reported tokens. Summed per-request latency is approximately 5.90
million ms; it is not wall-clock elapsed time because eight requests ran
concurrently. Monetary cost is deliberately not claimed because the provider
price schedule was not frozen. Numeric timing, memory, per-domain LLM usage,
and latency are preserved in `cost_latency.json`.

Focused completion/contracts tests: 36 passed. All Phase 4 tests: 56 passed.
The complete repository run: 957 passed, 3 failed, 11 warnings in 130.50s.
The three failures are pre-existing/environmental: a missing dataset registry,
unchanged backtest-toggle behavior, and absent optional `fastapi`. An affected
V2 run also exposed one allocation-sensitive legacy `id(self)` test; relevant
code is unchanged. No introduced regression was identified.

### Production gates and recommendation

| Gate | Result |
|---|---:|
| Three real held-out settings and reconstructed action sets | Pass |
| Frozen B0–B8 numeric evaluation | Pass |
| B2/B3 confirmatory coverage | **Fail** |
| B7 non-inferior to B5 and B6 | Pass |
| B7 predictive lift over B6 in at least two domains | Pass |
| Nominal uncertainty coverage | **Fail** |
| Posterior-uncertainty incremental value | **Fail** |
| Consequence-family incremental value | **Fail** |
| Two credible transfer/cold-start wins | **Fail** |
| Four rich-state components with non-ornamental value | **Fail** |
| Real execution/sequence causal value | **Fail** |
| Production promotion | **Withheld** |

The result supports a new **draft** PR for forensic review, not a production
merge. The old PR #87 is already merged; its production status is not revised
by this addendum.

### Frozen-evidence answers

1. **Was the original frozen Phase 4 result reproduced? Yes.** All frozen artifacts and the immutable tree were checksum-verified; they were not overwritten by rerunning the old writer.
2. **Were B2 and B3 successfully frozen and evaluated? No.** They were frozen and attempted completely, but 259 strict-schema failures invalidate confirmatory evaluation.
3. **Were at least three rich-state real settings added? Yes.** IPD, Senate, and repaired Enron ran, although state richness differs by domain.
4. **Were real action sets reconstructed? Yes.** Each adapter reconstructs the feasible set without the current label/result.
5. **Were real Phase 3 posterior particles consumed? Yes.** B7 consumes 64 typed compositional-posterior particles per decision; family selection can assign them zero weight.
6. **Did posterior uncertainty improve calibration or prediction? No.** Particle propensity helps as a mean feature in two domains, but uncertainty itself has no established incremental value and conformal coverage fails.
7. **Did B7-PREDICT beat raw LLM? No confirmatory answer.** Incomplete schema coverage makes the conditional LLM comparison invalid.
8. **Did B7-PREDICT beat handcrafted policy? Yes, predictively.** B7 has lower held-out calibrated log loss than B4 in all three domains.
9. **Was B7-PREDICT non-inferior to the strongest flat fitted model? Yes.** It beats B5 log loss in all three domains under the frozen test.
10. **Did B7-PREDICT add predictive value over B6? Yes, but only in two domains.** It improves Senate and Enron and exactly ties IPD.
11. **Did execution add real sequence or downstream-outcome value? No.** Observational sequence/outcome scoring ran; causal execution value did not.
12. **Were at least two credible cold-start or transfer wins achieved? No.** Directional transfer evidence is incomplete.
13. **Did at least four rich world-state components show non-ornamental value? No.** Validation identified at most hierarchy and historical propensity; consequences received zero weight.
14. **Was policy uncertainty empirically validated? No.** All three conformal coverage gates miss.
15. **Did any major domain fail catastrophically? Yes, for the LLM baseline.** Senate B3 covers only 19/64 test packets; the numeric B7 Senate arm itself does not fail catastrophically.
16. **Which policy families were validated?** None for production. Predictively, hierarchy is selected in IPD, propensity in Senate, and their mixture in Enron.
17. **Which were quarantined?** The consequence heuristic, universal/causal interpretations, incomplete LLM comparisons, and all production policy-family claims remain quarantined.
18. **Is Phase 4 software implemented? Yes.**
19. **Does it execute end to end? Yes.**
20. **Is it empirically validated? No overall.** It has partial predictive validation only.
21. **Is it production eligible? No.**
22. **Is PR #87 ready for production merge? No/not applicable.** It is already merged; this completion belongs in a new draft PR and is not production-ready.
