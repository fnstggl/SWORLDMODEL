# WMV2 Phase 4 — Production Actor-Policy Learning

## Scope and status

This phase replaces the detached uniform/LLM decision label with one universal path:

`natural-language question → WorldExecutionPlan → posterior WorldState particles → ActorView → scenario TypedAction set → perceived/actual feasibility → fitted or broad policy-family posterior → calibrated action posterior → sampled TypedAction → action Event → StateDelta → reactions/later decisions → terminal-state quantity`

The implementation is extensive, but the evidence does not license the phrase “production ready.”

| Status | Result | Basis |
|---|---:|---|
| Software implemented | Yes | Versioned contracts, governed family records, learning/calibration/artifacts, execution, adaptation, and tests exist. |
| Executes end to end | Yes | Compiler-to-shared-world test and eight causal integration traces create events, deltas, reactions, later decisions, and terminal quantity effects. |
| Empirically validated | No | Three real held-out settings ran, but raw-LLM evidence, policy-uncertainty evaluation, and many causal ablations are unavailable; transfer failed. |
| Production eligible | No | Aggregate empirical gates, two transfer/cold-start wins, and improvement over the strongest general fitted baseline are not established. |

## Initial audit of the inherited path

The audit was performed against base commit `e067d32` from `origin/claude/world-model-v2`, before implementation.

### Adjacent phases verified in code

| Phase | Present on base | Verified implementation and Phase 4 consequence |
|---|---:|---|
| 1 | Yes, canonical | `compiler.compile_world`, `WorldExecutionPlan`, `materialize.run_from_plan`, shared `WorldState`, events, transitions, and terminal projection were retained as the production spine. |
| 2 | Yes, partly detached | Evidence claims and visibility existed, but materialized evidence did not populate the `InformationLedger` used by actor views. Phase 4 now publishes/exposes permitted evidence and schedules it at the simulation as-of boundary. |
| 3/3B | Yes, partly detached | Posterior state/particle APIs existed. `ActorPolicyRuntime.decide` now consumes multiple posterior worlds; committed real caches do not reconstruct full posterior particles. |
| 6 | Yes | `registry.record` and `RegistryStore` implement family → pack → scenario governance. Phase 4 exports all policy families as Phase 6 `MechanismRecord`/`ParameterPack` shapes and withholds production promotion. |
| 7 | Yes, integration defect found | Registered mechanisms can be configured instances, while `materialize.operators_from_plan` assumed every entry was a class and called `cls()`. It now accepts classes, factories, and deep-copied instances. |
| 9 | Yes, separate network type | `phase9_network.MultilayerNetwork` is not the base `RelationGraph`. `ActorViewBuilder` now consumes `edges_visible_to` and existence-posterior fields when available. |
| 10 | Yes, integration defect found | `AuthorityGraph` and `InformationBoundary` existed but were not canonical actor-view inputs. Phase 4 reads boundary objects from world uncertainty metadata and continues to enforce executable institution rules. |

### Inherited runtime findings

| Path | Real caller and inputs | Reads/writes before Phase 4 | Audit verdict and repair |
|---|---|---|---|
| `pipeline.simulate → compiler.compile_world → materialize.run_from_plan` | Question, compiler JSON, evidence, as-of/horizon | Materialized shared world and registered operators | Reused as the only production path. Phase 4 did not add a benchmark router. |
| `compiler.WorldExecutionPlan` | Compiler semantic output | Had entities, mechanisms, events, outcome; no first-class actor decisions | Added `actor_decisions`, scheduled decision opportunities, and migration of legacy `agent_decision` to `production_actor_policy`. |
| `transitions.AgentDecisionOperator` | Event engine, `decision_opportunity` | Chose uniformly from raw dictionaries; wrote `current_action`; emitted a decision delta | Replaced in compiled production plans. It lacked typed action metadata, calibrated learning, resources, commitments, reactions, and explicit policy uncertainty. |
| `transitions.FittedDecisionOperator` | Optional callback | Passed the omniscient full `WorldState` into fitted policy code | Invalid information boundary. Production numeric policy now receives only `ActorView` objects. |
| `transitions.observable_view` | Legacy agent prompt | Own public fields, visible ledger items, adjacent relations | Useful precedent but incomplete contract. Replaced for Phase 4 by fail-closed `ActorViewBuilder`. |
| Institution validation | Every transition/operator | Executable rules checked raw `{actor,type,target}` actions | Reused. `FeasibilityEngine` distinguishes actor-perceived and actual validity; blocked attempts become explicit events/deltas. |
| Relation/network state | Legacy view and world | Adjacent base-graph edges | Extended to Phase 9 visibility-aware edges without exposing hidden layers. |
| Evidence visibility | Phase 2 materialization versus `InformationLedger` | Claims changed belief fields, but exposure state was disconnected | Bridged. Public evidence is exposed to all materialized actors; restricted evidence only to permitted actors. |
| Action history/adaptation | Legacy `past_actions`; no production update | Selected actions were appended inconsistently; no outcome learning | Phase 4 logs every executed/blocked action and exposes a typed TD-style adaptation delta in world-local policy state. |
| Terminal outcome | `GenericOutcomeOperator` and domain mechanisms | Often independent of actor decisions; branch seed used process-randomized Python `hash` | Typed action consequences now change non-probability quantities. Generic fallback uses a stable SHA-256-derived seed across processes. |
| Parameter estimation/calibration | Reference and benchmark modules | Separate classifiers/specialists; no universal fitted actor-policy pack | Added strict trajectory ingestion, hierarchical partial pooling, calibration-only temperature fitting, and checksummed packs. Old specialists remain baselines/negative evidence. |
| Policy-family uncertainty | None in canonical decision path | One uniform/LLM policy | Added posterior weights and per-family structural distributions; probabilities are mixed after family-specific distributions, not by inventing one averaged utility. |
| Traceability | Transition log only | No actor-view hash, family posterior, consequence posterior, calibration, or artifact version | Added checksummed `DecisionTrace`, actor/simulator separation, event/delta IDs, latency, costs, warnings, support grade, and fallback tier. |

The audit found no production action-set reconstruction, hierarchical fitting, calibration, posterior-family inference, adaptation, invalid-action diagnostics, benchmark leakage controls, or reliable terminal influence in the inherited canonical path. The Enron, BehaviorBench, Higgs, and Omnibehavior modules were not promoted; their negative results are hash-preserved in the Phase 4 quarantine registry.

## Architecture and five planes

| Plane | Phase 4 implementation |
|---|---|
| Code | `phase4_policy.py`, `phase4_execution.py`, and `phase4_learning.py` provide reusable, domain-general contracts and algorithms. |
| Evidence | `TrajectoryRecord`, `DatasetManifest`, immutable split manifests, source IDs, timestamps, row-specific candidate actions, and leakage flags. |
| Posterior | Hierarchical count/logit artifact, action/role/actor intercept distributions, family weights, parameter uncertainty, consequence distributions, credible intervals, and temperature artifact. |
| World state | Scenario actors, evidence exposures, visible graph edges, institution rules, resources, commitments, history, policy state, and posterior world particles. |
| Execution | Sampled typed actions emit `actor_action` or `action_blocked`, explicit `StateDelta`, resource/commitment/quantity changes, follow-up events, reactions, and later policies. |

There is no top-level branch for CMV, OpinionQA, Upworthy, BehaviorBench, or Enron. Dataset adapters only load, map observed actions, declare row-specific candidate sets and split metadata, and score. All B7 decisions pass through `ActorPolicyRuntime`.

## Versioned APIs

### `TypedAction` (`4.0.0`)

The action contract contains: stable ID and semantic version; actor ID/role; family/name; typed target; parameters; preconditions; information, permission, authority, resource, cost, and time requirements; duration; per-actor observability; reversibility; commitments; immediate/delayed consequences; triggered mechanisms; provenance; uncertainty; compiler inclusion reason; and support status.

The ontology is extensible only when a new scenario action names an executable mechanism. Unknown semantics without a mechanism are rejected. The committed machine schema is `experiments/results/phase4/action_ontology.json`.

Families and required built-ins:

- Messaging: `reply_now`, `reply_later`, `acknowledge`, `clarify`, `delegate`, `ignore`, `follow_up`, `escalate_message`, `reveal_information`, `withhold_information`.
- Negotiation: `accept`, `reject`, `counteroffer`, `concede`, `hold_position`, `delay`, `escalate`, `reveal`, `conceal`, `exit`, `seek_mediator`.
- Participation: `support`, `oppose`, `abstain`, `volunteer`, `donate`, `persuade`, `mobilize`, `defect`, `coordinate`, `protest`, `strike`, `withdraw`.
- Platform: `ignore`, `view`, `click`, `like`, `comment`, `share`, `report`, `follow`, `unfollow`, `create_content`, `delete_content`.
- Institutional: `approve`, `reject`, `amend`, `defer`, `veto`, `refer`, `escalate`, `enforce`, `appeal`, `schedule`, `place_on_agenda`, `allocate_resource`.
- Organizational/market: `hire`, `fire`, `recommend`, `authorize`, `purchase`, `sell`, `launch`, `delay_launch`, `acquire`, `withdraw_offer`, `allocate_budget`, `request_approval`.

`migrate_typed_action` upgrades unversioned/3.x semantic payloads, normalizes legacy aliases and string targets, and deterministically creates missing IDs. It rejects future versions and any inherited `probability`, `score`, or utility-weight field. `TypedAction.from_dict/as_dict` round-trip the current schema.

### `ActorView`

`ActorViewBuilder` is a fail-closed projection. It exposes only the actor’s role, current time, observed/remembered events, perceived public actions, beliefs and beliefs-about-actors, visible relationships/network position, permitted institution rules, authority, goals/preferences/incentives, commitments/obligations, resources/workload/attention, risk beliefs, history/policy state, expected reactions, source credibility, uncertainty, and visible provenance.

It excludes private actor fields, other actors’ private beliefs, simulator posterior truth, final probabilities, resolution outcomes, future events, post-as-of evidence, invisible graph layers, and institution rules hidden by a Phase 10 boundary. Evidence must appear in `InformationLedger.visible_to(actor, at=now)`. Reaction events are explicitly projected into the responding actor’s view.

Adversarial tests inject future outcomes, private messages, hidden beliefs, inaccessible evidence, and posterior truth. They do not enter the view.

### Action-space and feasibility contracts

`ActionSpaceBuilder` uses structured compiler proposals, current institution decision rights, actor authority/resources, and visible network paths. It never routes from question keywords and never globally exposes the ontology. If evidence is too weak, it supplies typed `wait` and `abstain` actor actions with a Tier-7 broad prior; the simulation itself does not abstain.

`FeasibilityEngine` returns both perceived and actual status from: feasible, feasible with uncertainty, temporary, prohibited, outside authority, impossible, unaffordable, insufficient information, unmet precondition, binding-commitment conflict, unknown to actor, or unsupported semantics. Known perceived impossibilities are absent from the probability distribution. A mistakenly attempted but actually invalid action creates `action_blocked`, a delta, reasons, and history rather than disappearing.

### Policy families and Phase 6 governance

All 22 required families have assumptions, required actor state/evidence, mathematical form, parameter schema, uncertainty, fit method, applicability/exclusion rules, transport limits, execution behavior, diagnostics, and failure modes:

| Structural group | Implemented families |
|---|---|
| Utility/discrete choice | random utility, multinomial logit, nested discrete choice, quantal response |
| Bounded cognition | satisficing, bounded search, belief planning, limited-depth reasoning |
| Learning/history | habit, reinforcement learning, Experience-Weighted Attraction |
| Social/institutional | norm compliance, obligation, reciprocity, imitation, social proof, institutional obedience |
| Risk/strategy/time | risk-sensitive choice, loss aversion, strategic anticipation, delay/inaction hazard |
| Structural uncertainty | policy/regime mixture |

Each family produces its own distribution over the perceived feasible set. `ActorPolicyModel` averages those distributions using posterior family weights; it does not average incompatible utilities first. The posterior retains structural-particle records.

`phase6_policy_registry_records()` exports the families as governed Phase 6 `MechanismRecord` objects with broad Tier-7 `ParameterPack` records. Their lifecycle is only `implemented`; missing held-out/transfer support remains a promotion blocker. Fitted dataset packs form the second layer, and `ActorPolicyModel` bound to a current `ActorView`/action set is the scenario instance. The committed records are in `policy_family_registry.json` and `parameter_packs.json`.

### Utility, consequences, posterior, and calibration

Utility inference separates stable/situational, social, identity/norm, institutional, resource/effort/opportunity/delay, risk, future, commitment, reputation, and relationship components. Actor-specific action intercepts shrink toward role then domain/global counts. Every component records a mean, uncertainty, source, and fallback tier. A fitted pack is labeled `experimental_fitted`, not locally validated merely because fitting succeeded.

`SubjectiveConsequenceModel` receives only `ActorView`, the typed action, and a provenance-bearing pack. It returns subjective success, reaction, resource, relationship, belief, sanction/delay outcomes, and uncertainty. Simulator truth is not an input.

`ActionPosterior` contains feasible action IDs, scores, calibrated probabilities, expected utilities and consequences, family posterior/structural particles, parameter uncertainty, intervals, entropy, feasibility diagnostics, support grade, fallbacks, sensitivity, provenance, model/pack versions, and sampling. Multiple posterior worlds produce particle-specific views/feasibility/consequences before marginalization. Default simulation samples with a fixed seed.

Temperature scaling is fit only on the calibration split. Aggregate binomial records use frequency weights in fitting, metrics, calibration, and paired row-cluster bootstrap.

## Learning and artifact governance

`phase4_learning.py` implements:

- real trajectory/manifests with exact candidate actions and source/time provenance;
- person, relationship, context, institution, sequence, and time-forward splits;
- fit-time rejection of post-action features and observed actions absent from their historical set;
- hierarchical partial pooling across global, domain, institution, role, and actor levels, with stable extension points for segment/context/time effects;
- cold-start fallback to group/domain posteriors;
- calibration-only temperature selection and weighted log loss, Brier, ECE, accuracy, entropy, reliability, confusion, and invalid-action metrics;
- paired weighted bootstrap confidence intervals;
- checksummed atomic serialization, corruption detection, concurrency lock, resumable row output, and bounded timeout;
- content hashes, split checksums, code commit, seed, pack/artifact checksums, failure preservation, runtime and cost.

The present dependency-free empirical-Bayes model is intentionally modest. It is a universal parameter pack, not a claim that all requested hierarchy interactions or causal preferences are identified by the three caches.

## Shared-world execution and adaptation

`ActorPolicyRuntime.execute` rechecks actual feasibility under the full simulator, then:

1. creates `actor_action` or `action_blocked`;
2. writes `current_action` and append-only `past_actions`;
3. consumes typed resources;
4. creates commitments;
5. applies bounded quantity or target-belief deltas (never terminal probabilities);
6. emits an explicit `StateDelta` including serialized follow-ups;
7. schedules message delivery, institution submissions, reactions, and delayed effects;
8. passes reaction events into the next actor’s view and policy;
9. records trace/event/delta IDs and deterministic seed.

`apply_adaptation` is the stable Phase 4 side of longitudinal learning. Rewards/outcomes update actor-local policy values plus a provenance-bearing update log and emit an explicit adaptation delta. Removing that history changes later reinforcement-policy probabilities. It is not a substitute for Phase 8’s future cross-run persistence service.

## Migration notes

- Compiler mechanism `agent_decision` now resolves to `production_actor_policy`. The Tier-A regression assertion was updated for this deliberate migration; no other valid test expectation was weakened.
- `WorldExecutionPlan` serializes `actor_decisions` and compiler-produced decision events.
- `materialize` now accepts configured operator instances and keyed resource/belief fields.
- Phase 2 evidence observations populate actor exposure state at the simulation as-of time while preserving the original evidence time in payload/provenance.
- `StateDelta.as_dict` now includes `follow_up_events`.
- Generic outcome replay uses a stable SHA-256 branch seed rather than process-randomized `hash`.

## Reproducibility

From the repository root with Python 3.12:

```bash
PYTHONPATH=. python -m experiments.wmv2_phase4_validate
PYTHONPATH=. python -m pytest tests/test_wmv2_phase4_contracts.py tests/test_wmv2_phase4_learning.py tests/test_wmv2_phase4_e2e.py -q
PYTHONPATH=. python -m pytest tests/test_world_model_v2.py tests/test_wmv2_evidence_phase2.py tests/test_wmv2_phase3_posterior.py tests/test_wmv2_phase3b_repair.py tests/test_wmv2_phase4_contracts.py tests/test_wmv2_phase4_learning.py tests/test_wmv2_phase4_e2e.py tests/test_wmv2_phase6.py tests/test_wmv2_phase7_adversarial.py tests/test_wmv2_phase7_execution.py tests/test_wmv2_phase7_forms.py tests/test_wmv2_phase9.py tests/test_wmv2_phase10.py tests/test_wmv2_tier_a_fixes.py -q
PYTHONPATH=. python -m pytest -q
```

Exact validation commit: `b3823d4284c63d3e35eafd51ae56ca3ca03a676c`. All 21 generated JSON artifacts pass checksum verification.

## Known limitations

- No leakage-safe, frozen same-row raw-LLM or LLM-panel prediction cache was available; B2/B3 were not run.
- CMV maps a delta outcome to the OP action and does not contain a timestamped action event.
- OpinionQA lacks question and option text, so only option-index choice is identified.
- Upworthy is aggregate randomized audience exposure, not person-level history.
- The real caches do not identify rich networks, institutional sanctions, subjective reactions, full posterior world particles, or longitudinal adaptation; many corresponding ablations are therefore ornamental.
- Family weights are not empirically identifiable in these one-step caches. All families remain implemented but unpromoted rather than falsely “validated.”
- Both active/passive cross-domain transfers fail badly.
- B7 gives shared-world execution but essentially no prediction lift over B6; it is not a stronger predictive model merely because it creates deltas.
- Full repository tests have three pre-existing/environment failures: missing `data/dataset_registry.json`, a pre-existing backtest toggle assertion, and absent optional `fastapi`. None of those paths differs from the base branch.

## Anti-scaffolding answers

1. **What real scenario-specific actor policy object was constructed?** A B7 `ActorPolicyModel` bound to each dataset’s fitted pack, each held-out actor’s reconstructed `ActorView`, and that row’s exact typed candidate set, then executed by `ActorPolicyRuntime`.
2. **From which real evidence or parameter pack?** Training-only CMV, OpinionQA, or Upworthy trajectories produced checksummed hierarchical packs; each trace names its pack and source IDs.
3. **What uncertainty was represented?** Partial-pooling uncertainty, family posterior weights/structural particles, consequence uncertainty, action intervals/entropy, calibration temperature, support grade, and fallback tier. Full empirical coverage of that uncertainty was not established.
4. **What actor-visible state entered the policy?** Role, time, permitted evidence, reconstructed pre-action features as beliefs, visible relations/rules, authority, goals, resources, commitments, history, and provenance available for that row/world.
5. **Which hidden omniscient state was excluded?** Test action/label, outcome, future events, resolution result, simulator posterior truth/probability, other actors’ private beliefs, hidden evidence, and inaccessible institution/network state.
6. **How was the feasible action set constructed?** From structured compiler or historical proposals plus live actor/institution capabilities; feasibility masks were applied before probabilities. Historical adapters supplied only externally reconstructed row sets.
7. **Which policy families were considered?** The 22 governed families listed above; fitted one-step packs place weight on the identified general mixture and keep alternatives structurally represented or downweighted when state is absent.
8. **How were policy-family weights determined?** From fitted artifact weights where available, otherwise broad reference-class priors, then reduced when required actor-visible state was absent and normalized.
9. **Where did utility parameters come from?** Training-only action frequencies/features with global/domain/institution/role/actor partial pooling, or a broad Tier-7 reference prior; never from compiler/LLM numeric output.
10. **How were subjective expected consequences formed?** From typed action consequences, actor beliefs/history/relationships/rules, and pack reaction/effect priors inside `SubjectiveConsequenceModel`, without full-world truth.
11. **What final action distribution was produced?** Every trace stores a normalized calibrated posterior over the perceived feasible action IDs; examples are reproduced in the forensic document and machine artifact.
12. **What typed action was selected?** A seeded sample from that posterior; real traces include delta/hold, option 0/1, and click/ignore, while integration traces cover six ontology families and eight settings.
13. **What event did it create?** `actor_action` (or explicit `action_blocked`) plus typed delivery, institution, reaction, or delayed-effect follow-ups.
14. **What StateDelta did it cause?** Current action/history changes on every execution, plus resource, commitment, belief, or non-probability quantity changes when configured.
15. **How did another actor or institution react?** Integration traces emit `actor_reaction` and `institution_submission`; the counterparty observes the reaction event and selects/executes `acknowledge` or `ignore` through the same policy.
16. **How did the action affect later decisions?** Reaction events alter the next decision opportunity; outcome adaptation/history changes later reinforcement/habit probabilities. The real one-step caches do not measure this longitudinal effect.
17. **How did it affect a terminal outcome?** Typed quantity deltas changed scenario terminal quantities; no action assigned a terminal probability. This is software causal-integration evidence, not a real causal estimate.
18. **Which real held-out datasets validated it?** CMV (person-disjoint), OpinionQA (person-disjoint), and Upworthy (context-disjoint), plus time-forward CMV. Their limitations prevent an overall empirically-validated verdict.
19. **Which transfer evaluations passed?** None. CMV→Upworthy and Upworthy→CMV active/passive transfers failed; OpinionQA was correctly excluded.
20. **Which ablations established incremental value?** Calibration improved CMV and OpinionQA; hierarchical B7 credibly beat flat B5 only on OpinionQA; execution established delta/terminal mechanics but not predictive lift over B6. Most causal removals were ornamental on these caches.
21. **What failed?** Raw-LLM/panel comparison was unavailable, structural transfer failed, uncertainty coverage was not evaluated, many ablations lacked identified state, and B7 did not credibly beat the strongest general fitted baseline.
22. **Which policy families were quarantined or rejected?** No universal family was falsely rejected from unidentified one-step evidence. Raw-LLM numeric policy is quarantined as unavailable/non-production; the prior Enron actor evidence is quarantined for temporal leakage; prior Omnibehavior persistence lift remains a preserved near-zero result.
23. **What remains incomplete?** Leakage-safe B2/B3, richer sequential/institutional/person histories, full posterior-particle real backtests, identified family comparison, uncertainty coverage, two transfer/cold-start wins, causal outcome validation, and production monitoring/persistence.
24. **Why is this more than a classifier or LLM wrapper?** The numeric policy is fitted/calibrated without LLM probabilities, sees only actor-local state, masks infeasible actions, preserves structural uncertainty, samples a typed action, mutates the shared world through events/deltas, schedules reactions, changes later policy state, and can affect terminal quantities. The evidence still does not make it production eligible.
