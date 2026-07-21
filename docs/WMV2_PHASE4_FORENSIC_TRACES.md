# WMV2 Phase 4 — Forensic Actor-Policy Traces

## Reading this document

The checksum-sealed machine artifact `experiments/results/phase4/forensic_traces.json` contains:

- 15 real held-out traces: five each from CMV, OpinionQA, and Upworthy;
- 8 causal integration traces spanning messaging, negotiation, organizational approval, election participation, legislation, acquisition, platform interaction, and coalition mobilization;
- full candidate actions, feasibility rows, family posterior, consequence posterior, calibrated distribution, sample seed, action, events, `StateDelta`, later decision, terminal effect, latency, costs, support grade, fallback tier, and checksum.

The two groups answer different questions. Real traces are prediction evidence but their caches are one-step and state-poor. Integration traces prove shared-world causality but are deterministic software fixtures with a Tier-7 prior, not empirical validation. They are never pooled as if they were equivalent evidence.

## Real held-out trace 1 — CMV communication/persuasion

**Question/decision.** For held-out record `t1_cg7xgcc`, will original poster `theinsanity` award a delta or hold position after the challenger argument?

**Actor discovery and posterior actor state.** The row resolves the actor as role `original_poster`. The pre-action view contains log OP length `4.4886`, log argument length `4.4308`, lexical overlap `0.06875`, and normalized argument-question count `0.0`. These values enter as observed actor-view beliefs with source IDs. Goals are the broad reconstructed goal `respond_to_current_decision`; resources contain one unit of assumed attention; commitments and action history are empty because the actor is person-disjoint from training.

**Visible versus hidden.** Visible: only the OP/challenger text-derived features, IDs, dyad, timestamp, role, and source provenance available at decision time. Hidden: the observed delta label, eventual outcome, test label, future thread events, private beliefs, off-platform history, and simulator truth/probability. The offline world does not populate a separate ledger item for these numeric features, so `observed_evidence_ids` is empty even though each belief field retains its source IDs; this is a traceability limitation, not permission to expose the label.

**Relationships/institutions.** The historical record identifies an OP/challenger dyad and the CMV delta convention, but not a reconstructable network or executable moderation state. Both observed options are treated as available. No hidden relationship strength or institution rule is invented.

**Candidates and feasibility.** Two `TypedAction@4.0.0` objects are created: `award_delta` and `hold_position`. Each is included because it is in the externally reconstructed historical action set. Both are perceived and actually feasible, with no masking reason. The compiler does not receive a probability.

**Policy and parameters.** Pack `phase4:0f2a9e8f1ad71462f5bcfac2`, fit only on the CMV training partition, supplies partial-pooling action intercepts. Support remains `experimental_fitted`. Structural weights are:

- random utility `0.4660`;
- quantal response `0.3883`;
- habit `0.0583` after missing-history downweighting;
- limited-depth reasoning `0.0388`;
- risk-sensitive `0.0388`;
- obligation `0.0097`.

Those are posterior structural weights, not an LLM judgment. Family distributions are formed separately then averaged.

**Subjective consequences.** For both actions the state-poor cache supports only a broad success probability `0.5`, reaction prior `{observe: 0.5, respond: 0.1, ignore: 0.4}`, consequence uncertainty `0.5`, and the executable action-count consequence. There is no evidence-licensed relationship or belief effect. The fact that consequences are nearly action-invariant is a negative finding: the predictive distinction comes mainly from fitted action intercepts.

**Distribution and sample.** Calibrated posterior:

```text
P(award_delta)  = 0.664172
P(hold_position)= 0.335828
```

The fixed-seed sample selects `award_delta`.

**Event, delta, reaction, terminal effect.** Execution creates `actor_action`, writes `current_action`, appends `past_actions`, and changes `quantities[action_count:award_delta]` from `0` to `1`. No reaction event is scheduled because this historical candidate uses only the neutral `record_action` mechanism; the cache has no trustworthy reply sequence to reconstruct. The counter is a measurable shared-world terminal quantity effect, not a real causal estimate of persuasion.

**Limitations/support.** Person-disjoint prediction is real; the action itself is inferred from the outcome field rather than a timestamped action event. Support is `experimental_fitted`, and the trace does not license intervention advice.

## Real held-out trace 2 — OpinionQA indexed choice

**Question/decision.** For held-out row `oqa:13791`, will respondent `American_Trends_Panel_W32_2031` select indexed option 0 or 1?

**Actor-visible state.** Role is `survey_respondent`. The view includes only recorded demographics: age 30–49, weekly attendance, citizenship, college education, conservative ideology, income at least $100,000, never married, Republican, white, Northeast, Roman Catholic, and male; it also contains the source question/wave IDs and broad response goal. Actor history is absent by construction.

**Hidden state.** The selected answer, test label, outcome, question wording, option text/polarity, response time, future survey answers, private attitude, and simulator posterior are excluded. Because wording is absent from the committed cache, the implementation refuses to reinterpret option index as support/oppose.

**Candidates/feasibility.** `select_option_0` and `select_option_1` are typed scenario extensions with the executable `record_action` mechanism. Both are perceived/actually feasible.

**Policy/consequences.** Fitted pack `phase4:a1870b4f95189b7c338d0b69` has the same governed structural-family support pattern as the CMV pack. Subjective consequences remain broad (`P(success)=0.5`, uncertainty `0.5`) because option semantics are missing. This is honest non-identification.

**Distribution/sample.** `P(option_0)=0.489107`, `P(option_1)=0.510893`; the sample selects option 1. The action event writes action/history and increments `action_count:select_option_1` to 1. No empirical reaction or later choice is inferred from this row.

**Limitations/support.** The person split is real, but the near-uniform distribution and missing semantics mean this is weak product evidence. Support is `experimental_fitted`.

## Real held-out trace 3 — Upworthy platform exposure

**Question/decision.** For a representative weighted audience exposed to arm 0 of test `541ef39d88aa4b19e100005b`, does the audience impression click or ignore?

**Actor-visible state.** Role is `weighted_audience_actor`. Visible headline features are log word count `2.8332`, digit share `0.0270`, uppercase share `0.2778`, and zero normalized question/exclamation marks. The randomized headline/test/arm exposure is known. No individual identity is invented.

**Hidden state.** The row’s click/ignore count, eventual aggregate CTR, test label, device, position, individual history, archive problem flag, future exposures, and simulator probability are excluded from features.

**Candidates/feasibility.** `click` and `ignore` are typed executable actions, both perceived and actually feasible. Positive click and ignore counts remain frequency weights; the policy does not read CTR.

**Policy/consequences.** Pack `phase4:35ba2508cd46a2e06c2b681f` is fit with weighted training counts and calibrated with weighted calibration rows. Consequence state is again broad and action-invariant because individual downstream state is unavailable.

**Distribution/sample.** `P(click)=0.015071`, `P(ignore)=0.984929`. This example samples `ignore`, writes the typed event/action history, and increments `action_count:ignore`. Another stored trace samples the low-probability `click`, demonstrating posterior sampling rather than modal assignment.

**Limitations/support.** The distribution mostly reproduces a highly imbalanced reference class. It is aggregate population behavior, not a person policy; B7 is statistically indistinguishable from B5/B6. Support is `experimental_fitted`.

## Detailed causal integration trace — negotiation

This trace demonstrates the complete shared-world mechanics that the one-step real caches cannot identify.

1. **Question:** “Will the focal actor act in negotiation?”
2. **Actor discovery:** `focal`, role `negotiator`, and target `counterpart` are materialized in one shared world.
3. **Posterior actor state:** the focal actor has a broad `success=0.6` belief, one attention unit, the goal `resolve_negotiation`, empty history/commitments, a visible communication edge, and explicit authority for both candidates.
4. **Visible evidence:** ledger item `fixture:negotiation`, created before the decision and exposed only to the focal actor.
5. **Hidden evidence:** resolution outcome, simulator posterior truth, terminal probability, future reaction and delayed events, and the counterparty’s private state.
6. **Relationships/institution:** the focal actor sees the communication edge and an executable decision-right rule. The actual institution will recheck the action.
7. **Candidate actions:** `accept` and `counteroffer`, each `TypedAction@4.0.0`, target `counterpart`, cost `0.1` attention, 30-second delayed effect, institution/message/reaction mechanisms. `accept` also creates a nonbinding commitment.
8. **Feasibility:** both are perceived and actually feasible; no known-impossible mass remains.
9. **Policy structures:** Tier-7 pack `phase4:tier7-reference:4.0.0` gives random utility `0.3871`, quantal response `0.2581`, satisficing `0.2581`, limited-depth reasoning `0.0484`, and risk-sensitive `0.0484`. Missing required state downweights the last two. Support is `highly_speculative`, fallback tier 7.
10. **Subjective consequences:** both actions expect attention `−0.1`, response prior `{observe: 0.5, respond: 0.25, ignore: 0.25}`, success `0.5`, and uncertainty `0.8`. The semantic fixture supplies a positive terminal quantity delta for accept and negative for counteroffer; it does not supply a probability.
11. **Calibrated action posterior:** `P(accept)=0.499983`, `P(counteroffer)=0.500017`. The nearly uniform posterior is correct for the broad, unvalidated prior.
12. **Sample:** the fixed seed selects the slightly lower-probability `accept`, proving sampling rather than argmax.
13. **Action event:** `actor_action(focal → counterpart, accept)` with the typed action and trace ID.
14. **StateDelta:** writes `focal.current_action`, appends `focal.past_actions`, reduces attention `1.0 → 0.9`, creates `commit:negotiation`, and changes `terminal:negotiation 0 → 1`.
15. **Follow-ups:** `message_delivered`, `institution_submission`, `actor_reaction` at +10 seconds, and `delayed_action_effect` at +30 seconds are serialized on the delta.
16. **Later decision:** the clock advances to the reaction. The counterparty’s `ActorView` includes the `actor_reaction` event and excludes hidden simulator fields. The same runtime produces a 0.5/0.5 `acknowledge`/`ignore` posterior; the seeded sample selects and executes `acknowledge`, creating its own event and delta.
17. **Terminal effect:** `terminal:negotiation=1.0`. No code assigns a terminal probability.
18. **Limitation:** every numeric policy value is a broad Tier-7 prior. This trace proves causal wiring, not realistic negotiation behavior.

## Stratified causal integration traces

All eight traces use the same `ActorPolicyRuntime`, view projection, feasibility engine, family posterior, sampling, action execution, and reaction path. The table maps probabilities to action names; the machine trace retains IDs and checksums.

| Setting / family | Calibrated posterior | Sampled action | Focal delta beyond action/history | Later action | Terminal quantity |
|---|---|---|---|---|---:|
| Individual messaging / messaging | ignore 0.500017; reply now 0.499983 | ignore | attention −0.1 | acknowledge | −1 |
| Negotiation / negotiation | accept 0.499983; counteroffer 0.500017 | accept | attention −0.1; commitment | acknowledge | +1 |
| Organizational approval / institutional | approve 0.499983; defer 0.500017 | approve | attention −0.1; commitment | acknowledge | +1 |
| Election participation / participation | support 0.499983; abstain 0.500017 | abstain | attention −0.1 | ignore | −1 |
| Legislation / institutional | veto 0.500017; approve 0.499983 | approve | attention −0.1; commitment | acknowledge | +1 |
| Acquisition / organizational-market | withdraw 0.500017; acquire 0.499983 | acquire | attention −0.1; commitment | acknowledge | +1 |
| Platform interaction / platform | ignore 0.500017; click 0.499983 | ignore | attention −0.1 | ignore | −1 |
| Coalition mobilization / participation | defect 0.500017; coordinate 0.499983 | defect | attention −0.1 | acknowledge | −1 |

Every row also emits message delivery, institution submission, reaction, and delayed-effect events. That uniform mechanism coverage is intentionally an integration stress test, not a claim that all real platform or election actions generate those same events.

## Forensic conclusions

The traces show a genuine actor-policy execution boundary: actor-local state is hashed and projected; hidden fields are named; the action set is typed and masked; numeric probabilities come from a pack/calibrator; family structures are retained; a seeded sample becomes an event and explicit delta; other actors can observe and respond; later world state changes; terminal quantities can move.

They also expose the remaining weaknesses rather than hiding them:

- real one-step packs are mainly hierarchical action intercepts;
- subjective consequences are largely invariant because the caches lack identified downstream state;
- real traces do not contain reactions or longitudinal adaptation;
- integration traces are broad-prior fixtures with nearly uniform probabilities;
- terminal fixture effects validate mechanics, not causal effect sizes;
- real actor-view source IDs are attached to fields but not separately published as ledger items in the offline adapters;
- no trace establishes production-eligible intervention guidance.

The correct interpretation is: **the software models and executes an actor decision rather than merely returning a label, but the current empirical evidence is not sufficient to trust its general behavioral probabilities in production.**
