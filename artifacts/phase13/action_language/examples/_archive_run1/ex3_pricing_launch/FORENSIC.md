# Forensic trace — ex3_pricing_launch
## 1. Decision contract
```json
{
 "decision_id": "ex3",
 "decision_maker": "mara_voss",
 "authority": [
  "founder_ceo"
 ],
 "controllable_resources": {
  "launch_budget": 40000.0
 },
 "context": "Launch now vs recruit design partners privately vs delay for the pilot readout.",
 "horizon": "2025-09-16T00:00:00Z"
}
```
## 2. Stated goal & missing preferences
- goal: a committed public launch or 3+ signed design partners by mid-September, without board conflict
- missing preferences / unresolved tradeoffs: ["whether launch commitment or design partner count is more important if both cannot be achieved simultaneously", "whether budget remaining matters or not", "whether any specific board member's conflict is worse than another", "whether signed design partners must be from different industries or any industry", "tradeoff between committing launch earlier vs. signing more design partners if resources are limited"]
- goal predicates:
```json
[
 {
  "predicate_id": "public_launch_committed_by_mid_sept",
  "role": "desired_terminal",
  "record_type": "public_launch_committed",
  "field": "is_committed",
  "op": "eq",
  "value": true,
  "description": "A public launch must be committed (is_committed = true) on or before 2025-09-16T00:00:00Z",
  "by_ts": null,
  "hold_for_s": 0.0
 },
 {
  "predicate_id": "design_partners_signed_count",
  "role": "desired_terminal",
  "record_type": "design_partner_count",
  "field": "count",
  "op": "gte",
  "value": 3,
  "description": "At least 3 design partner agreements must be signed (count >= 3) by mid-September",
  "by_ts": null,
  "hold_for_s": 0.0
 },
 {
  "predicate_id": "no_board_conflict",
  "role": "forbidden",
  "record_type": "board_vote_record",
  "field": "status",
  "op": "eq",
  "value": "conflict",
  "description": "No board vote record may have status 'conflict' \u2014 any board conflict disqualifies the outcome",
  "by_ts": null,
  "hold_for_s": 0.0
 },
 {
  "predicate_id": "design_partner_agreements_signed",
  "role": "required_intermediate",
  "record_type": "design_partner_agreement",
  "field": "status",
  "op": "eq",
  "value": "signed",
  "description": "Design partner agreements must have status 'signed' to count toward the count of 3+",
  "by_ts": null,
  "hold_for_s": 0.0
 },
 {
  "predicate_id": "near_miss_launch_committed_but_no_partners",
  "role": "near_miss",
  "record_type": "public_launch_committed",
  "field": "is_committed",
  "op": "eq",
  "value": true,
  "description": "Public launch committed but fewer than 3 design partners signed \u2014 looks like partial success but fails the full goal",
  "by_ts": null,
  "hold_for_s": 0.0
 },
 {
  "predicate_id": "near_miss_partners_but_no_launch_commitment",
  "role": "near_miss",
  "record_type": "design_partner_count",
  "field": "count",
  "op": "gte",
  "value": 3,
  "description": "3+ design partners signed but no public launch committed \u2014 looks like partial success but fails the full goal",
  "by_ts": null,
  "hold_for_s": 0.0
 }
]
```
## 3. Scenario-specific action language
```json
{
 "decision_maker": "mara_voss",
 "n_controllable_objects": 0,
 "authority_sources": [],
 "channels": [
  "board_meeting",
  "direct_communication"
 ],
 "institutions": [
  "seed_investor_board"
 ],
 "resources": [
  "calder_logistics_pilot",
  "launch_budget",
  "mara_voss_discretionary_budget"
 ],
 "dimensions": [
  {
   "id": "target_actor",
   "description": "which actor to approach for design partner signing or board vote",
   "example_values": [
    "devon_reyes",
    "priya_shah",
    "external design partner"
   ],
   "open_ended": true
  },
  {
   "id": "request_type",
   "description": "what exact action to request: sign design partner agreement, approve public launch, expand",
   "example_values": [
    "design_partner_signing",
    "board_approval_vote",
    "pilot_expansion_decision"
   ],
   "open_ended": true
  },
  {
   "id": "terms",
   "description": "specific terms of design partner agreement or launch commitment (e.g., budget allocation, ",
   "example_values": [
    "fixed fee",
    "revenue share",
    "6-month pilot",
    "unlimited budget"
   ],
   "open_ended": true
  },
  {
   "id": "timing",
   "description": "when to execute the action relative to deadline and other events",
   "example_values": [
    "before next board meeting",
    "after pilot results",
    "immediately"
   ],
   "open_ended": true
  },
  {
   "id": "public_vs_private",
   "description": "whether the launch commitment is public or internal, and whether design partner signing is",
   "example_values": [
    "public launch commitment",
    "private board approval",
    "quiet design partner"
   ],
   "open_ended": true
  },
  {
   "id": "conditionality",
   "description": "what conditions or contingencies to attach to the action",
   "example_values": [
    "conditional on pilot success",
    "conditional on board vote",
    "unconditional"
   ],
   "open_ended": true
  }
 ],
 "timing_opportunities": [
  "schedule a board_meeting before mid-September to vote on public_launch_commitmen",
  "use calder_logistics_pilot results to trigger pilot_expansion_decision before de"
 ],
 "unresolved_affordances": [
  {
   "claim": "mara_voss can unilaterally commit public launch",
   "reason": "board_approval_threshold exists and board holds majority vote; CEO authority alone insufficient"
  },
  {
   "claim": "mara_voss can directly contact member_2 or member_3",
   "reason": "no channel defined to those actors except through board_meeting"
  },
  {
   "claim": "authority 'founder_ceo role in company schema'",
   "reason": "not in the declared decision contract, the sch
```
## 4-5. Every candidate generated, and why
### user_1 — Public launch announcement using discretionary budget with post-hoc board notification
- proposed by: user
- causal theory: 
  - **user_1_s1**: Announce the public launch this week using the 40k discretionary budget
    - targets ['mara_voss_discretionary_budget'] | channel direct_communication | visibility participants | timing None
    - exact content: “Announce the public launch, allocating 40k from mara_voss_discretionary_budget to fund the announcement.”
  - **user_1_s2**: Tell the board after the announcement
    - targets ['seed_investor_board'] | channel board_meeting | visibility participants | timing None
    - exact content: “Inform the seed_investor_board that the public launch has been announced, after the fact.”
### user_2 — Privately recruit three design partners with hands-on onboarding before any public launch, starting with Devon Reyes at 
- proposed by: user
- causal theory: 
  - **user_2_s1**: Recruit Devon Reyes as the first design partner via private, hands-on onboarding at Calder before any public launch
    - targets ['devon_reyes'] | channel direct_communication | visibility participants | timing None
    - exact content: “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”
### user_3 — Wait for Calder pilot readout, then decide with data
- proposed by: user
- causal theory: 
  - **user_3_s1**: Wait for the Calder pilot readout in three weeks
    - targets [] | channel direct_communication | visibility participants | timing None
    - exact content: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
  - **user_3_s2**: Decide with the pilot data in hand
    - targets ['mara_voss'] | channel board_meeting | visibility participants | timing None
    - exact content: “After receiving the pilot readout, make a decision using the pilot data.”
### do_nothing — do nothing (status quo)
- proposed by: baseline
- causal theory: the world evolves without intervention
### user_3_r1a — Wait for Calder pilot readout with proactive follow-up, then decide
- proposed by: revision (revision of ['user_3']: add_step: actor_rejection)
- causal theory: 
  - **user_3_s1**: Wait for the Calder pilot readout in three weeks
    - targets [] | channel direct_communication | visibility participants | timing None
    - exact content: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
  - **user_3_s2**: Decide with the pilot data in hand
    - targets ['mara_voss'] | channel board_meeting | visibility participants | timing None
    - exact content: “After receiving the pilot readout, make a decision using the pilot data.”
  - **user_3_r1a_s3**: Proactively request the Calder pilot readout from the responsible party
    - targets ['mara_voss'] | channel email | visibility participants | timing None
    - exact content: “Contact the Calder logistics pilot operator to explicitly request the readout, with a follow-up reminder if not received within two weeks.”
### user_3_r1b — Wait for Calder pilot readout with escalation contingency, then decide
- proposed by: revision (revision of ['user_3']: add_contingency: actor_rejection)
- causal theory: 
  - **user_3_s1**: Wait for the Calder pilot readout in three weeks
    - targets [] | channel direct_communication | visibility participants | timing None
    - exact content: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
  - **user_3_s2**: Decide with the pilot data in hand, or escalate if data is missing
    - targets ['mara_voss'] | channel board_meeting | visibility participants | timing None
    - exact content: “After the expected arrival time of the pilot readout, if data is received, make a decision using it. If no data is received by that time, escalate to the pilot operator's manager to obtain the data or a status update.”
### user_2_r1a — Privately recruit three design partners with hands-on onboarding before any public launch, starting with Devon Reyes at 
- proposed by: revision (revision of ['user_2']: add_step: missing_intermediary)
- causal theory: 
  - **user_2_s1**: Recruit Devon Reyes as the first design partner via private, hands-on onboarding at Calder before any public launch
    - targets ['devon_reyes'] | channel direct_communication | visibility participants | timing None
    - exact content: “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”
  - **user_2_r1a_s2**: Secure a formal commitment from Devon Reyes to proceed with onboarding, including confirmation of availability and access to required resources
    - targets ['devon_reyes'] | channel email | visibility participants | timing None
    - exact content: “Send a follow-up request to Devon Reyes asking for explicit confirmation of willingness to participate, a signed non-disclosure agreement, and a scheduled time for the first onboarding session at Calder, before any hands-on work begins.”
### user_2_r1b — Privately recruit three design partners with hands-on onboarding before any public launch, starting with Devon Reyes at 
- proposed by: revision (revision of ['user_2']: replace_step: actor_rejection)
- causal theory: 
  - **user_2_s1**: Recruit Devon Reyes as the first design partner via a low-commitment exploratory conversation at Calder before any public launch
    - targets ['devon_reyes'] | channel in-person | visibility participants | timing None
    - exact content: “Invite Devon Reyes to an informal, no-obligation meeting at Calder to discuss the design partner role, benefits, and expectations, without requiring any agreement or onboarding commitment at this stage.”
## 6. Feasibility rejections (typed, exact)
```json
[
 {
  "candidate_id": "user_1",
  "gates": [
   {
    "code": "infeasible_all_hypotheses",
    "detail": "[{\"code\": \"target_missing\", \"detail\": \"step user_1_s1: target 'mara_voss_discretionary_budget' does not exist\", \"in_n_worlds\": 3}]"
   }
  ]
 }
]
```
## 7. Compiled direct effects (kernel ops per surviving step)
```json
{
 "user_1": [
  {
   "step": "user_1_s1",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "Announce the public launch, allocating 40k from mara_voss_discretionary_budget to fund the announcement.",
     "structured_fields": {
      "action_name": "Announce the public launch this week using the 40k discretio",
      "content": "Announce the public launch, allocating 40k from mara_voss_discretionary_budget to fund the announcement.",
      "target": "mara_voss_discretionary_budget"
     },
     "direct_targets": [
      "mara_voss_discretionary_budget"
     ],
     "intended_visibility": "participants"
    }
   ]
  },
  {
   "step": "user_1_s2",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "Inform the seed_investor_board that the public launch has been announced, after the fact.",
     "structured_fields": {
      "action_name": "Tell the board after the announcement",
      "content": "Inform the seed_investor_board that the public launch has been announced, after the fact.",
      "target": "seed_investor_board"
     },
     "direct_targets": [
      "seed_investor_board"
     ],
     "intended_visibility": "participants"
    }
   ]
  }
 ],
 "user_2": [
  {
   "step": "user_2_s1",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.",
     "structured_fields": {
      "action_name": "Recruit Devon Reyes as the first design partner via private,",
      "content": "Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.",
      "target": "devon_reyes"
     },
     "direct_targets": [
      "devon_reyes"
     ],
     "intended_visibility": "participants"
    }
   ]
  }
 ],
 "user_3": [
  {
   "step": "user_3_s1",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "Wait for the readout of the Calder logistics pilot, expected in three weeks from now.",
     "structured_fields": {
      "action_name": "Wait for the Calder pilot readout in three weeks",
      "content": "Wait for the readout of the Calder logistics pilot, expected in three weeks from now.",
      "target": ""
     },
     "direct_targets": [],
     "intended_visibility": "participants"
    }
   ]
  },
  {
   "step": "user_3_s2",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "After receiving the pilot readout, make a decision using the pilot data.",
     "structured_fields": {
      "action_name": "Decide with the pilot data in hand",
      "content": "After receiving the pilot readout, make a decision using the pilot data.",
      "target": "mara_voss"
     },
     "direct_targets": [
      "mara_voss"
     ],
     "intended_visibility": "participants"
    }
   ]
  }
 ],
 "user_3_r1a": [
  {
   "step": "user_3_s1",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "Wait for the readout of the Calder logistics pilot, expected in three weeks from now.",
     "structured_fields": {
      "action_name": "Wait for the Calder pilot readout in three weeks",
      "content": "Wait for the readout of the Calder logistics pilot, expected in three weeks from now.",
      "target": ""
     },
     "direct_targets": [],
     "intended_visibility": "participants"
    }
   ]
  },
  {
   "step": "user_3_s2",
   "ops": [
    {
     "op": "emit_semantic_event",
     "semantic_type_id": "unmodeled_actor_action",
     "exact_content": "After receiving the pilot readout, make a decision using the pilot data.",
     "structured_fields": {
      "action_name": "Decide with the pilot data in hand",
      "content": "After receiving the pilot read
```
## 8-11. Per-world causal record (events → observations → actor choices → state)
### user_2 — particle 0
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [participants]: “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”

**Who observed what:**
- devon_reyes ← (mara_voss): “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”

**Actor invocations & choices (from the delta log):**
- generated_actor_invocation: {"decision_summary": "I will not sign immediately. I need to understand how this design partnership fits my timeline and whether it strengthens my position or makes me look reactive."}

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_2_s1"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### user_2 — particle 1
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [participants]: “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”
- t=1752801860.0: `unmodeled_actor_action` by devon_reyes → ['mara_voss'] [private]: “I need to clarify what Calder expects from me as a design partner and how this recruitment relates to my current trajectory before signing anything.”
- t=1752803720.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [private]: “I will meet with Devon one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment with his trajectory, not a detour.”
- t=1752805580.0: `unmodeled_actor_action` by devon_reyes → ['mara_voss'] [private]: “I will meet with Mara one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment, not a detour, before signing any agreement.”
- t=1752807440.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [private]: “I will meet with Devon one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment with his trajectory, not a detour, before signing any agreement.”
- t=1752809300.0: `unmodeled_actor_action` by devon_reyes → ['mara_voss'] [private]: “I will proceed with the one-on-one meeting Mara proposed, but I will push for concrete details on Calder's stability and the exit terms of the agreement before signing anything.”
- t=1752811160.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [private]: “I will hold the one-on-one meeting now, but I will control the framing: Calder's stability is a mutual concern, not a precondition Devon can leverage. I'll offer a joint review of exit terms after the pilot, not before.”
- t=1752813020.0: `unmodeled_actor_action` by devon_reyes → ['mara_voss'] [private]: “I will not accept a post-pilot review of exit terms; I need concrete exit terms before signing anything, and I will state this directly in the meeting.”
- t=1752814880.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [private]: “I will meet Devon now and concede on the principle of pre-defined exit terms, but frame them as a mutual performance-based safety valve rather than a one-sided escape hatch.”
- t=1752816740.0: `unmodeled_actor_action` by devon_reyes → ['mara_voss'] [private]: “I will accept Mara's concession on pre-defined exit terms but insist the safety valve language be tied to specific, measurable performance triggers, not left to mutual interpretation after the pilot.”
- t=1752818600.0: `unmodeled_actor_action` by mara_voss → ['devon_reyes'] [private]: “I will concede on measurable triggers but tie them tightly to Calder's pilot deliverables, framing this as a shared accountability structure rather than a concession.”

**Who observed what:**
- mara_voss ← (devon_reyes): “I need to clarify what Calder expects from me as a design partner and how this recruitment relates to my current trajectory before signing anything.”
- mara_voss ← (devon_reyes): “I will meet with Mara one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment, not a detour, before signing any agreement.”
- mara_voss ← (devon_reyes): “I will proceed with the one-on-one meeting Mara proposed, but I will push for concrete details on Calder's stability and the exit terms of the agreement before signing anything.”
- mara_voss ← (devon_reyes): “I will not accept a post-pilot review of exit terms; I need concrete exit terms before signing anything, and I will state this directly in the meeting.”
- mara_voss ← (devon_reyes): “I will accept Mara's concession on pre-defined exit terms but insist the safety valve language be tied to specific, measurable performance triggers, not left to mutual interpretation after the pilot.”
- devon_reyes ← (mara_voss): “Request that Devon Reyes sign a design partner agreement for hands-on onboarding at Calder, with the understanding that this is a private recruitment before any public launch.”
- devon_reyes ← (mara_voss): “I will meet with Devon one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment with his trajectory, not a detour.”
- devon_reyes ← (mara_voss): “I will meet with Devon one-on-one to clarify Calder's expectations and frame the recruitment as a strategic alignment with his trajectory, not a detour, before signing any agreement.”
- devon_reyes ← (mara_voss): “I will hold the one-on-one meeting now, but I will control the framing: Calder's stability is a mutual concern, not a precondition Devon can leverage. I'll offer a joint review of exit terms after the”
- devon_reyes ← (mara_voss): “I will meet Devon now and concede on the principle of pre-defined exit terms, but frame them as a mutual performance-based safety valve rather than a one-sided escape hatch.”

**Actor invocations & choices (from the delta log):**
- generated_actor_invocation: {"executed_action": "request_clarification"}
- generated_actor_invocation: {"executed_action": "negotiate_design_partner_agreements"}
- generated_actor_invocation: {"executed_action": "request_clarification"}
- generated_actor_invocation: {"executed_action": "negotiate_design_partner_agreements"}
- generated_actor_invocation: {"executed_action": "request_clarification"}
- generated_actor_invocation: {"executed_action": "negotiate_design_partner_agreements"}
- generated_actor_invocation: {"executed_action": "request_clarification"}
- generated_actor_invocation: {"executed_action": "negotiate_design_partner_agreements"}
- generated_actor_invocation: {"executed_action": "accept_concession_with_caveat"}
- generated_actor_invocation: {"executed_action": "negotiate_design_partner_agreements"}

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_2_s1"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### user_3 — particle 0
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → [] [participants]: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “After receiving the pilot readout, make a decision using the pilot data.”

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_3_s1", "user_3_s2"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### user_3 — particle 1
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → [] [participants]: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “After receiving the pilot readout, make a decision using the pilot data.”

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_3_s1", "user_3_s2"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### do_nothing — particle 0
**Semantic events (exact content):**

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": [], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### do_nothing — particle 1
**Semantic events (exact content):**

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": [], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### user_3_r1a — particle 0
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → [] [participants]: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “After receiving the pilot readout, make a decision using the pilot data.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “Contact the Calder logistics pilot operator to explicitly request the readout, with a follow-up reminder if not received within two weeks.”

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_3_s1", "user_3_s2", "user_3_r1a_s3"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

### user_3_r1a — particle 1
**Semantic events (exact content):**
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → [] [participants]: “Wait for the readout of the Calder logistics pilot, expected in three weeks from now.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “After receiving the pilot readout, make a decision using the pilot data.”
- t=1752800000.0: `unmodeled_actor_action` by mara_voss → ['mara_voss'] [participants]: “Contact the Calder logistics pilot operator to explicitly request the readout, with a follow-up reminder if not received within two weeks.”

**Who observed what:**

**Actor invocations & choices (from the delta log):**

**Resulting records (world state):**

**Plan execution here:** {"completed": ["user_3_s1", "user_3_s2", "user_3_r1a_s3"], "failed": [], "lapsed": [], "halted": false, "condition_checks": {}}
**Goal row:** success=False, forbidden=False, predicates={"public_launch_committed_by_mid_sept": false, "design_partners_signed_count": false, "no_board_conflict": false, "design_partner_agreements_signed": false, "near_miss_launch_committed_but_no_partners": false, "near_miss_partners_but_no_lau

## 12-13. Where each strategy first succeeded/failed (diagnosis)
```json
{
 "user_2": {
  "candidate_id": "user_2",
  "n_particles": 3,
  "n_success": 0,
  "step_stats": {
   "user_2_s1": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   }
  },
  "earliest_breaks": [
   {
    "kind": "missing_intermediary",
    "detail": "other actors reacted but the required conditions never materialized",
    "in_n_worlds": 2
   },
   {
    "kind": "actor_rejection",
    "detail": "no other actor produced any responsive event",
    "in_n_worlds": 1
   }
  ],
  "hypothesis_dependence": {
   "H0": {
    "n": 3,
    "success": 0
   }
  },
  "reaction_summary": {
   "devon_reyes": {
    "reacted_in": 2
   }
  },
  "truncations": [],
  "narrative": "In most simulated worlds, Devon Reyes reacted to the recruitment attempt, but the required conditions for onboarding never materialized, suggesting a missing intermediary step prevented progress. In one world, no responsive event occurred at all, indicating outright rejection or failure to engage. The core issue is that the plan lacked a necessary intermediate condition or resource to move from re"
 },
 "user_3": {
  "candidate_id": "user_3",
  "n_particles": 3,
  "n_success": 0,
  "step_stats": {
   "user_3_s1": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   },
   "user_3_s2": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   }
  },
  "earliest_breaks": [
   {
    "kind": "actor_rejection",
    "detail": "no other actor produced any responsive event",
    "in_n_worlds": 3
   }
  ],
  "hypothesis_dependence": {
   "H0": {
    "n": 3,
    "success": 0
   }
  },
  "reaction_summary": {},
  "truncations": [],
  "narrative": "The plan failed because the Calder pilot readout never arrived; no actor produced any responsive event, meaning the data that was supposed to trigger the decision never materialized, so the decision step could not be meaningfully executed despite both steps being technically completed."
 },
 "user_3_r1a": {
  "candidate_id": "user_3_r1a",
  "n_particles": 3,
  "n_success": 0,
  "step_stats": {
   "user_3_s1": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   },
   "user_3_s2": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   },
   "user_3_r1a_s3": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   }
  },
  "earliest_breaks": [
   {
    "kind": "actor_rejection",
    "detail": "no other actor produced any responsive event",
    "in_n_worlds": 3
   }
  ],
  "hypothesis_dependence": {
   "H0": {
    "n": 3,
    "success": 0
   }
  },
  "reaction_summary": {},
  "truncations": [],
  "narrative": "The plan failed because the proactive request for the Calder pilot readout was never made by the responsible party, as no actor produced any responsive event in any simulated world. This means the initial step of waiting passively for three weeks was followed by a decision step that had no data to act on, because the follow-up request\u2014which was supposed to trigger the readout\u2014was never executed. T"
 },
 "user_3_r1b": {
  "candidate_id": "user_3_r1b",
  "n_particles": 3,
  "n_success": 0,
  "step_stats": {
   "user_3_s1": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   },
   "user_3_s2": {
    "completed": 3,
    "failed": 0,
    "lapsed": 0
   }
  },
  "earliest_breaks": [
   {
    "kind": "actor_rejection",
    "detail": "no other actor produced any responsive event",
    "in_n_worlds": 3
   }
  ],
  "hypothesis_dependence": {
   "H0": {
    "n": 3,
    "success": 0
   }
  },
  "reaction_summary": {},
  "truncations": [],
  "narrative": "The plan's first step, waiting for the Calder pilot readout, never 
```
## 14-15. Revisions and their fate
```json
[
 {
  "parent": "user_3",
  "child": "user_3_r1a",
  "op": "add_step",
  "addressed": "actor_rejection"
 },
 {
  "parent": "user_3",
  "child": "user_3_r1b",
  "op": "add_contingency",
  "addressed": "actor_rejection"
 },
 {
  "parent": "user_2",
  "child": "user_2_r1a",
  "op": "add_step",
  "addressed": "missing_intermediary"
 },
 {
  "parent": "user_2",
  "child": "user_2_r1b",
  "op": "replace_step",
  "addressed": "actor_rejection"
 }
]
```
Revision children appear in §4 with ancestry; a revision that worsened forbidden-state frequency is listed in §6 with code revision_worsened_forbidden.
## 16. Matched comparison between finalists
```json
{
 "user_2": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {
   "design_partner_count": {
    "n": 1,
    "mean": 2.0,
    "min": 2.0,
    "max": 2.0,
    "median": 2.0,
    "direction": "higher_better",
    "unit": "partners"
   }
  }
 },
 "user_3": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {}
 },
 "do_nothing": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {}
 },
 "user_3_r1a": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {}
 },
 "user_3_r1b": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {}
 },
 "user_2_r1a": {
  "n_particles": 3,
  "success_count": 0,
  "forbidden_count": 0,
  "near_miss_count": 0,
  "predicate_counts": {
   "public_launch_committed_by_mid_sept": 0,
   "design_partners_signed_count": 0,
   "no_board_conflict": 0,
   "design_partner_agreements_signed": 0,
   "near_miss_launch_committed_but_no_partners": 0,
   "near_miss_partners_but_no_launch_commitment": 0
  },
  "by_hypothesis": {
   "H0": {
    "n": 3,
    "success": 0,
    "forbidden": 0
   }
  },
  "quantities": {
 
```
## 17. Final verdict
- recommendation_kind: **pareto** | recommended: **None**
- distinguishable finalists: False
- Pareto set: ['user_2', 'user_3', 'do_nothing', 'user_3_r1a', 'user_3_r1b', 'user_2_r1a', 'user_2_r1b']
- adjudicator synthesis (blind): {"goal_generator": 1, "adversarial_omission_critic": 1, "feasibility_authority_critic": 1, "mechanism_critic": 7, "domain_reality_critic": 1, "goal_gaming_critic": 1, "implementation_critic": 3, "direct_effect_compiler": 9, "final_adjudicator": 1}
- support claim: best-supported among the considered feasible actions under the stated goal, constraints, world hypotheses, and simulation support
## 18. Assumptions that could reverse the result
- none recorded

## 19. Cost, coverage, approximation limits
- particles/arm: 3 | simulated arms: 7
- LLM calls: planner/critic roles 25 + actor-simulation calls 156
- latency_s: 884.459
- stop reason: round 1: no revision materially changed the trajectory distribution
- unresolved semantics: [{"candidate_id": "user_1", "unresolved": [{"step": "user_1_s1", "reason": "all proposed effects rejected: not_a_kernel_op:; not_a_kernel_op:; not_a_kernel_op:"}, {"step": "user_1_s2", "reason": "all proposed effects rejected: not_a_kernel_op:"}]}, {"candidate_id": "user_2", "unresolved": [{"step": "user_2_s1", "reason": "all proposed effects rejected: not_a_kernel_op:; not_a_kernel_op:"}]}, {"can
- forensic truncation: per-arm worlds dumped = 2 of 3

## Raw traces
- every planner/critic/adjudicator LLM call: `role_trace.jsonl`
- every actor-simulation LLM call (what each invoked actor was SHOWN and ANSWERED, verbatim): `actor_trace.jsonl`
- complete per-world dumps: `forensic_worlds.jsonl`
