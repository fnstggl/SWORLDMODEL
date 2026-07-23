# Forensic Reconstruction — Strait of Hormuz Tanker Threshold

- **qid:** `017e64ef-7354-56c4-8a4d-e27121bc639a`
- **name:** Hormuz
- **question:** "Will daily oil tanker transits through the Strait of Hormuz reach 50 or more on any day between April 29 and June 1, 2026?"
- **as_of:** 2026-04-30 · **horizon:** 2026-06-01 · **model:** deepseek-v4-flash · **profile:** lean_v2 · **seed:** 0
- **Outcome (ground truth): 0 / NO.** Headline forecast **0.8333** (YES). **Brier 0.6944.** `sim_moved_toward_outcome=false`, `correct_side=false`.
- **Terminal kind:** `state_predicate`. 18 external LLM calls. 72 weighted trajectory nodes (78 executed in metrics; 6 shared-world roots + 12 actor-variant combos × 6 = 72 emitted). Blueprint compiled FRESH.

This report is a read-only reconstruction from the committed trace only. No code was run against the model; all quotes are copied verbatim from `llm_calls.jsonl`, the checkpoint, and the manifests.

> **One-line thesis:** the executable terminal is a **boolean OR of three de-escalation/transit events** (`__terminal_yes__`), not a numeric daily-transit count. Merely *lifting the blockade* — or a single tanker sailing — trips YES, even though the question requires **≥50 tankers on one day**. The numeric bridge (actions → daily count → compare to 50) was flagged missing, attempted by mechanism-recovery, and **failed to build**, yet the run proceeded on the boolean terminal.

---

## A. Question inputs

**Exact question (from checkpoint `metrics.question`):**
> "Will daily oil tanker transits through the Strait of Hormuz reach 50 or more on any day between April 29 and June 1, 2026?"

**Exact resolution criterion (verbatim EVIDENCE lines shown to every stage, `llm_calls` call 0/1/2):**
> "This question resolves YES if, on any calendar day (measured in UTC) on or after April 29, 2026 and on or before June 1, 2026, at least 50 commercial oil tankers (vessels classified as crude oil tankers or oil/chemical tankers by their AIS transponder type) co[…]"
>
> "\"Normal commercial oil tanker traffic\" is operationalized as at least 50 oil tanker transits per day, which represents roughly half the estimated pre-conflict daily oil tanker traffic through the strait (pre-conflict total vessel traffic was 125–140 ships/day […]"
>
> "The question resolves NO if no single day in the specified window meets this threshold."
>
> "If Reuters, Lloyd's List, or Bloomberg report that daily oil tanker transits through the Strait of Hormuz have reached or exceeded 50 on any day in the resolution window, the question resolves YES."

**The precise numeric bar is unambiguous: ≥ 50 oil-tanker transits on a single UTC calendar day.** It is stated four ways in the evidence (resolution rule, operationalization, source rule, NO rule).

**Evidence supplied — does the background contain dated verbatim transit COUNTS?** Yes, but all pre-window / sub-threshold:
- "As of April 27, 2026, only about **7 ships** transited the strait in the prior 24 hours, none of which were oil tankers carrying cargo for the global market" (Reuters). *(A dated count — 3 days before as_of, ~0 oil tankers.)*
- "Pre-conflict, between **125 and 140 ships** typically crossed in and out of the strait daily" (Reuters). *(Total vessels, not oil-tanker-only; undated baseline.)*
- "commercial crossings have fallen by approximately **94%** from peacetime levels."
- EIA STEO: Middle East oil-production shut-ins "reached **9.1 million b/d** in April 2026, up from 7.5 million b/d in March."

There is **no in-window (Apr 29–Jun 1) daily-transit count anywhere in the evidence** — there cannot be, since as_of is 2026-04-30. The only tanker-specific in-strait count is ~7 ships/day with zero oil tankers, i.e. ~50× below the bar.

**Prior + exact historical cases (`forecast_decomposition.grounded_prior`, `reference_class` call 1 `outcome_reference_class`):** grounded prior **p = 0.5, n = 2**, `hierarchy_level = broad_human_decision_class`, numerator 1 / denominator 2, interval [0.0887, 0.9113]. The two counted cases:
1. `outcome=true` — "Pre-conflict daily oil tanker traffic was roughly 50-70 tankers per day (half of 125-140 total vessels)." date 2025-06-15, basis "Pre-conflict, between 125 and 140 ships typically crossed…". *(Coded YES.)*
2. `outcome=false` — "Since the conflict began on February 28, commercial crossings have fallen by approximately 94% from peacetime levels." date 2026-03-01. *(Coded NO.)*

So the prior is literally "1 peacetime YES, 1 wartime NO → 0.5". `fallback_reason`: "all levels sparse (total 2 usable case(s)); rate is wide by construction."

**What was NOT supplied:** any numeric baseline *time series* of daily oil-tanker transits; any in-window observation; any per-regime transit rate (blockade-on vs blockade-off → count). The model had the **threshold** (50) and one **current sub-threshold datapoint** (~7 ships, 0 tankers) but no distribution of counts to interpolate.

---

## B. World compilation (`structural_generation`, call 0)

**Prompt (call 0):** "You are compiling ONE coherent causal world… Compile the MINIMAL terminal-relevant world… The terminal block must name the mechanical rule that decides YES vs NO and the actions that write it." Full schema requested `terminal.kind ∈ {institution_vote,event_occurs,state_predicate}` with `yes_when/no_when/written_by_action_ids`. Reply began with `{`, parsed cleanly, no retry (`retried` absent/false).

**Parsed blueprint (verbatim from reply):**

- `resolution.interpretation`: "YES if any day from 2026-04-29 to 2026-06-01 has ≥50 oil tanker transits through Strait of Hormuz per Reuters, Lloyd's List, or Bloomberg; else NO." `yes_means`: "≥50 tanker transits on at least one day in window". *(Interpretation layer is CORRECT and numeric.)*
- `causal_thesis`: "…Resumption of normal traffic requires either a diplomatic agreement to lift the blockade and clear mines, or a unilateral US decision to allow tankers through… The threshold of 50 tankers is unlikely without a major de-escalation." *(Note: the qualitative thesis leans NO.)*

**Actors included (4):**
| id | discretion | 1 action template | variants |
|---|---|---|---|
| `us_navy` (aliases: US Central Command) | decisive | `us_lift_blockade` | us_hawkish, us_pragmatic |
| `iran_revolutionary_guards` (IRGC Navy) | decisive | `iran_clear_mines` | irgc_retaliatory, irgc_negotiating |
| `oil_tanker_operators` (Shipping companies, Tanker owners) | decisive | `tanker_operators_transit` | operators_risk_averse, operators_opportunistic |
| `reuters_lloyds_bloomberg` (Media) | **ceremonial** | — | reporters_accurate |

**Institutions:** none. **Mechanisms:** one — `ais_tracking`, kind `physical`, `deterministic_rule`: "Count of oil tankers crossing geographic boundaries per day, reported by Reuters/Lloyd's/Bloomberg", `writes_terminal:true`. *(This is the one place a real count lives — but no action ever feeds it a number; see F/J.)*

**Action templates (3) — EXACT effects as authored by the blueprint:**
- `us_lift_blockade` → `set_state {key: blockade_status, value: lifted}` · `writes_terminal:false` · validation "" · emits `blockade_lift`.
- `iran_clear_mines` → `set_state {key: mine_status, value: cleared}` · `writes_terminal:false` · emits `mine_clearance`.
- `tanker_operators_transit` → `set_state {key: daily_transit_count, value: "≥50"}` · **`writes_terminal:true`** · validation **"blockade_status='lifted' AND mine_status='cleared'"** · emits `tanker_transit_surge`.

**EXACT terminal rule (verbatim):**
```json
"terminal": {
  "kind": "state_predicate", "decision_rule": "threshold",
  "rule_params": {"option": "", "threshold": "50"},
  "yes_when": "daily_transit_count >= 50 on any day in window",
  "no_when":  "daily_transit_count < 50 on all days in window",
  "written_by_action_ids": ["tanker_operators_transit"],
  "evaluation_day": "2026-06-01"
}
```

**CRITICAL — does the blueprint terminal encode the numeric bar?** *Textually yes, mechanically no.* The `yes_when` string reads like a numeric comparison (`daily_transit_count >= 50`), and `rule_params.threshold="50"`. But the **only writer** is `tanker_operators_transit`, whose effect **hard-codes** `daily_transit_count = "≥50"` — a string literal, not a computed integer. There is no action anywhere that writes a *number*; the AIS `ais_tracking` mechanism (which could) is never invoked by any action template. So the predicate `daily_transit_count >= 50` can only ever be compared against the literal `"≥50"` that the transit action stamps in. **The comparison is vacuous: the writer writes the answer, not the input.**

**The missing bridge, stated now (developed in J):** there is no path `actor decisions (blockade effectiveness / escort / insurance / route availability) → integer daily_transit_count → compare(count, 50)`. The blueprint jumps straight from "an operator decides to sail" to "count = ≥50."

**Omitted / low-sensitivity (blueprint's own lists):** excluded — global oil demand fluctuations, non-oil vessel traffic, LNG specifics. `reversal_capable_omissions`: "Diplomatic negotiations between US and Iran." No insurer/war-risk actor, no escort-capacity actor, no partial-count actor — the very quantities that would set a *number* are absent.

---

## C. State construction and weighting

**One `state_generation` call (call 2).** Prompt: "Propose the genuinely DIFFERENT private realities each decisive actor could be in, as of 2026-04-30… ABSOLUTELY NO probabilities." Actors passed: us_navy, iran_revolutionary_guards, oil_tanker_operators (the ceremonial media actor is excluded from state generation). `state_generation_numeric_rejections = []` (no numbers had to be stripped). Reply parsed to 7 states (`actor_states.jsonl`, 7 rows; none `eliminated`, none `is_unknown`):

| actor | states (2–3) | `action_if_state` (verbatim, abridged) | `distinguishing_observations` |
|---|---|---|---|
| us_navy | `blockade_enforcement_high` | "Continue aggressive interdiction… use of force against non-compliant tankers." | "No tanker successfully runs blockade" |
| | `blockade_enforcement_low` | "Reduce patrols, allow some tankers to transit…" | "Tanker traffic increases gradually" |
| iran_revolutionary_guards | `mining_and_attacks_high` | "Continue mining and attacking any tanker…" | "Tanker crews refuse to transit" |
| | `mining_and_attacks_low` | "Allow some tankers to pass…" | "Some tankers transit without incident" |
| oil_tanker_operators | `risk_acceptance_high` | "Send tankers through strait in **large numbers**, possibly in convoys…" | "Multiple tankers transit daily" |
| | `risk_aversion_high` | "Keep tankers at anchor… reroute around Africa." | "Tanker traffic remains **below 10 per day**" |
| | `selective_transit` | "Send a **limited number** of tankers… **but not enough to reach 50 per day**." | "Some tankers transit, but not enough to reach 50 per day" |

**Realism / framing:** the states themselves are well-drawn and *carry magnitude* — `risk_acceptance_high` = "large numbers", `selective_transit` = **explicitly "not enough to reach 50 per day"**, `risk_aversion_high` = "below 10 per day". This is exactly the information a numeric terminal would need. **It is discarded downstream**: all three operator states map onto the *same* binary action `tanker_operators_transit`, which stamps `≥50` regardless of whether the state said "large numbers" or "not enough to reach 50."

**Recovery (`state_recovery_manifest`):** `ok:true`, 0 recovery calls, 0 empty sets; `reversal_search.ran=false` ("every actor already holds a reversal-capable state"). Residual `r=0.05` per actor, bounded per uncovered option (`us_lift_blockade`, `iran_clear_mines`, `tanker_operators_transit` each "no counted cases").

**Weights (`weight_provenance.state_posteriors`):** ungrounded — every actor gets a **flat split** in every shared-condition combo: us_navy 0.5/0.5, IRGC 0.5/0.5, operators 0.3333/0.3333/0.3333, plus `unknown:0.2` per combo. Provenance: `counted_complement`, "complement of the counted matched rates (1 − 0.000) shared among N unmatched modeled state(s) — grounded, not a label." I.e. **zero reference cases matched any actor state**, so weights defaulted to uniform. `weight_sensitive=false`, `dependence_sensitive=false`.

**Bias assessment:** the *weighting* is not itself escalation-biased (it is uniform). The bias enters via **action collapse** (C→F): two of three operator states describe sub-50 traffic, yet all three drive the same YES-writing action. Combined with the terminal OR (B/F), the structure is **mechanically biased toward YES/disruption** independent of the weights.

---

## D. World generation

**Distinct shared-condition worlds (`shared_worlds.jsonl`, `weight_provenance.grounding`):** 2 conditions —
- `hormuz_closure_regime` (states: closed / partially_open / fully_open), counted rate 0.8333, n=2 (both cases `outcome=true`: Feb-28 closure; Apr-27 "~7 ships").
- `global_oil_market_disruption` (normal / disrupted / crisis), rate 0.75, n=1 (EIA STEO).

**Combo formation:** `state_posteriors` enumerates **6 shared-condition combos** actually used — `{global_oil ∈ (normal,disrupted)} × {hormuz ∈ (closed,partially_open,fully_open)}` = 6 (the `crisis` oil-state and per-condition full cross were pruned to the 6 that carry weight; `readiness` check "shared_conditions_exist" notes "6 combo(s)").

**Total nodes:** 6 shared combos × (us_navy 2 × IRGC 2 × operators 3 = **12** actor-variant combos) = **72 weighted trajectory nodes** (`world_trajectories.jsonl` = 72 lines; each node weight **0.013889 = 1/72**). Metrics `weighted_nodes_executed=78` (72 leaves + 6 shared-world roots). `branches_merged=0`.

**Reused decision contexts (`decision_manifest.decisions`):** `unique_decision_contexts=7`, `stores=7`, `hits/reuses=216`, `misses=0`. The 7 templates = one per (actor × state): us_navy×{high,low}, IRGC×{high,low}, operators×{risk_acceptance,risk_aversion,selective}. Largest reuse 36 (the four 2-state actor cohorts reuse 36× each; the three operator cohorts 24× each). `model_fingerprint: deepseek-v4-flash` on all 7 templates.

**Grouped by identical structure (all 72 nodes collapse to 12 distinct terminal outcomes, 0.0833 = 1/12 each):**

| us_navy | IRGC | operators | terminal | mass |
|---|---|---|---|---|
| blockade_high | mining_high | risk_acceptance | **YES** | 0.0833 |
| blockade_high | mining_high | risk_aversion | not_bound | 0.0833 |
| blockade_high | mining_high | selective | not_bound | 0.0833 |
| blockade_high | mining_low | risk_acceptance | **YES** | 0.0833 |
| blockade_high | mining_low | risk_aversion | not_bound | 0.0833 |
| blockade_high | mining_low | selective | not_bound | 0.0833 |
| blockade_**low** | mining_high | risk_acceptance | **YES** | 0.0833 |
| blockade_**low** | mining_high | risk_aversion | **YES** | 0.0833 |
| blockade_**low** | mining_high | selective | **YES** | 0.0833 |
| blockade_**low** | mining_low | risk_acceptance | **YES** | 0.0833 |
| blockade_**low** | mining_low | risk_aversion | **YES** | 0.0833 |
| blockade_**low** | mining_low | selective | **YES** | 0.0833 |

YES = 8/12 = 0.6667 (48 nodes). not_bound = 4/12 = 0.3333 (24 nodes). The pattern is the fingerprint of the bug: **every `blockade_low` world is YES regardless of operators or Iran** (lifting the blockade alone trips the terminal), and under `blockade_high` YES requires the operator to be `risk_acceptance` (who chooses the template transit).

---

## E. Actor decisions (all 7 unique contexts + 3 deliberations)

All decisions dated **2026-06-01**, `trigger=deadline`, `NEW OBSERVATIONS: (none)` (actors act blind to each other — no cross-actor visibility, so the operator never "sees" whether the blockade lifted). Feasible menu = the actor's single template plus "novel act / wait." Quotes below are verbatim from `llm_calls.jsonl`.

**Call 3 — operators / `risk_acceptance_high`:** chosen `tanker_operators_transit`, `act`. Summary: *"I am sending the tankers through the strait now. The US Navy can protect them, Iran will not attack neutral vessels, and insurance costs are manageable."* → **template hit, writes terminal.**

**Call 4 — us_navy / `blockade_enforcement_high`:** screened out `us_lift_blockade` ("would be seen as capitulation"); chose novel *"Continue aggressive interdiction of all vessels…"* → compiled at call 11 (novel, non-terminal).

**Call 5 — us_navy / `blockade_enforcement_low`:** chosen **`us_lift_blockade`**, `act`. Summary: *"I choose to lift the blockade to de-escalate tensions, reduce costs, and respond to domestic economic pressures…"* → **template hit; this is the action that trips terminal-YES for all 4 `blockade_low` operator combos.**

**Call 6 — operators / `selective_transit` (first pass):** chosen `tanker_operators_transit` but target *"all tankers in queue with low US linkage"*, summary *"Proceed with transit for tankers assessed as lower risk… while holding higher-risk vessels."* `missing_decisive_fact`: "Current threat level from Iran." → triggered deliberation (call 12).

**Call 7 — IRGC / `mining_and_attacks_low`:** screened out `iran_clear_mines` ("would remove our leverage"); chose novel *"Continue selective mine placement and harassment without full closure"* → deliberation (call 14) + compile (call 15).

**Call 8 — operators / `risk_aversion_high`:** **`wait`**. Screened out `tanker_operators_transit`: *"Transiting now would be reckless. My private belief is that Iran will attack any tanker… Sending vessels would guarantee loss."* Summary *"I will not send tankers through the strait."* → **no terminal write → not_bound (under blockade_high).**

**Call 9 — IRGC / `mining_and_attacks_high`:** screened out `iran_clear_mines`; chose novel *"Maintain current posture and continue closure without clearing mines"* → compile (call 10).

**Deliberations (SAME moment, no new info):**
- **Call 12 — operators/selective:** `changed:false`. New chosen_action text: *"Transit lower-risk tankers immediately; hold US-linked tanker pending escort or reassessment."* Reflection explicitly reasons sub-threshold ("Proceeding with lower-risk tankers… maintains delivery schedules"). This mutated a *template* choice into a *novel* action string → compiled at calls 13 & 16 (non-terminal) → so `selective_transit` under blockade_high writes **no** terminal → not_bound.
- **Call 14 — IRGC/mining_low:** `changed:false`, keeps "selective mine placement… without full closure." Note the actor's own words: *"stay below the threshold"* — it is reasoning about a threshold the mechanics never encode.
- **Call 17 — us_navy/blockade_low:** `changed:false`, keeps `us_lift_blockade`.

**Realism:** the role-play is coherent and, crucially, several actors reason in explicitly **sub-50 / "below the threshold"** terms. That nuance is authentic but **cannot reach the terminal**, because the terminal only reads the boolean OR.

---

## F. Consequences and interaction (incl. the 5 `consequence_compile` calls)

**Precompiled template hits: 191. Novel requests: 132.** 4 distinct novel actions compiled; 1 rejection (`consequences.rejections`): operators' "Transit lower-risk tankers…" once "treated as no-op" ("no template and novel compile unavailable").

**The FIVE `consequence_compile` calls (verbatim replies). Prompt each time: "Compile its MECHANICAL consequence structure only… Reply ONLY JSON."**

| call | actor | action | compiled key(s) → value | writes_terminal |
|---|---|---|---|---|
| 10 | IRGC | "Maintain current posture and continue closure…" | `closure_status=active`, `mine_clearance=none` | false |
| 11 | us_navy | "Continue aggressive interdiction…" | `iran_oil_imports=blocked` | false |
| 13 | operators | "Transit lower-risk tankers… hold US-linked" | `lower_risk_tanker_status=transiting`, `us_linked_tanker_status=held` | false |
| 15 | IRGC | "Continue selective mine placement/harassment…" | `mine_harassment_level=selective` | false |
| 16 | operators | "Transit lower-risk tankers… hold US-linked" | `lower_risk_tanker_status=transiting`, `us_linked_tanker_status=held` | false |

**CRITICAL — what world-state key did each action write, number vs boolean/flag?**
- **Not one action ever wrote a numeric `daily_transit_count`.** The novel operator action (calls 13/16), which semantically means "a *few* lower-risk tankers sail," writes only qualitative flags (`lower_risk_tanker_status=transiting`) and is explicitly **non-terminal** → correctly does NOT trip YES, but also never contributes a count. All five compiled consequences are qualitative flags.
- The **terminal-writing** effects live only in the 3 precompiled templates, and after canonicalization all three write the **same boolean canonical key** (see below). `tanker_operators_transit`'s value is the *string* `"≥50"` — a hard-coded verdict, semantically "traffic is at/above 50," asserted rather than measured.

**Terminal canonicalization — the mechanical crux (`provenance.lean_v2.terminal_canonicalization`, verbatim):**
```json
"needed": true, "canonical_key": "__terminal_yes__",
"rewritten": [
  {"action_id": "us_lift_blockade",         "old_key": "blockade_status",     "new_key": "__terminal_yes__"},
  {"action_id": "iran_clear_mines",         "old_key": "mine_status",         "new_key": "__terminal_yes__"},
  {"action_id": "tanker_operators_transit", "old_key": "daily_transit_count", "new_key": "__terminal_yes__"}
]
```
So the compiled templates (`provenance.lean_v2.consequences.templates`) end as:
- `us_lift_blockade`: `set_state {key: __terminal_yes__, value: "lifted"}`, **`writes_terminal:true`**
- `iran_clear_mines`: `set_state {key: __terminal_yes__, value: "cleared"}`, **`writes_terminal:true`**
- `tanker_operators_transit`: `set_state {key: __terminal_yes__, value: "≥50"}`, `writes_terminal:true`

**This collapses three semantically distinct events — "US lifts blockade," "Iran clears mines," "tankers transit ≥50" — into a single boolean `__terminal_yes__`.** The terminal now fires YES if **ANY** of them writes the key. The blueprint's `validation` on `tanker_operators_transit` ("blockade_status='lifted' AND mine_status='cleared'") is defeated too: its precondition keys were themselves rewritten to `__terminal_yes__`, so the guard no longer gates anything.

**Interaction / observers / later actors:** actors decide blind (no cross-actor messages consumed; `NEW OBSERVATIONS: (none)` in every prompt), so there is no cascade — the operator does not react to a lifted blockade, nor Iran to US posture. Each node's terminal is a pure function of `{did us_lift_blockade fire?} OR {did iran_clear_mines fire?} OR {did tanker_operators_transit fire?}`. No reconsideration reopened a decision (`completion_rounds[0].reopened_decisions=0`).

---

## G. Deadlines and completion

Single decision epoch: all actors triggered at the **2026-06-01** deadline (`readiness.repairs_applied` scheduled a `schedule_mandatory_trigger` for each of us_navy, iran_revolutionary_guards, oil_tanker_operators on 2026-06-01 — i.e. a *forced* terminal decision, not voluntary). `waiting_allowed_before_deadline=true`. No reopenings, no splits: `completion_audit.engine_completion_rounds.round 1` = `splits:0, reopened_decisions:0, re_evaluated:24` (only the 24 not_bound nodes were re-checked, none resolved). `proven_unavoidable:true`. The only non-act choice was operators/`risk_aversion` → `wait` (call 8), a genuine modeled option, not invented. No invented choices detected; the 4 novel actions are faithful expansions of the actors' stated intents.

---

## H. Terminal outcome — materially distinct groups

Two groups (`resolution_report`, `world_trajectories`):

**Group 1 — YES, mass 0.6667 (48 nodes).** What predicate evaluated YES and WHY: the terminal reads canonical key `__terminal_yes__`; YES iff any terminal-writer set it. Two sub-mechanisms:
1. **`blockade_low` worlds (all 4 operator/IRGC combos under it, mass 0.3333):** `us_navy` chose `us_lift_blockade` (call 5/17) → writes `__terminal_yes__="lifted"` → YES. **This is the wrongful YES par excellence:** lifting a blockade is a *precondition* for traffic recovery, not evidence that ≥50 tankers crossed on a day. Under e.g. `blockade_low + mining_high + risk_aversion`, Iran still mines the strait AND operators refuse to sail ("below 10 per day"), yet the node resolves YES purely because the US relaxed enforcement.
2. **`blockade_high + risk_acceptance` worlds (mass 0.3333):** operators chose the template `tanker_operators_transit` (call 3) → writes `__terminal_yes__="≥50"` → YES. Here the value literally *asserts* "≥50" regardless of how many vessels the state implied.

**Group 2 — missing_mechanism (`state_predicate_not_mechanically_bound`), mass 0.3333 (24 nodes).** These are exactly `blockade_high` × `{risk_aversion, selective_transit}`: the US keeps the blockade (no `us_lift_blockade`), Iran does not clear mines (no `iran_clear_mines`), and operators either `wait` (call 8) or take the *novel* non-terminal "transit lower-risk" action (calls 12→13/16) → **no action ever writes `__terminal_yes__`** → the predicate is never mechanically bound → unresolved, cause `state_predicate_not_mechanically_bound`. `completion_audit` lists all 24 by node id, each weight 0.013889.

**Mass conservation:** YES 0.6667 + missing_mechanism 0.3333 = 1.0 (`completion_audit.total_mass=1.000002`, rounding). Resolved mass = YES = 0.6667. **Sim-only conditional** = 0.6667/0.6667 = **1.0** (`resolved_conditional_distribution: {YES:1.0, NO:0.0}`). `honest_bounds: min_supported_yes_share 0.6667, max_possible_yes_share 1.0`. **Note there is zero resolved NO mass** — the structure has no path that resolves NO; the "true NO" can only appear as unresolved-missing-mechanism, never as a modeled NO.

---

## I. Forecast decomposition (`forecast_decomposition.json`)

- **Grounded prior:** p = 0.5, n = 2, interval [0.0887, 0.9113], `broad_human_decision_class` (1 peacetime YES / 1 wartime NO).
- **Simulation conditional:** p = **1.0**, resolved_mass 0.6667, interval [1.0,1.0], `yes_mass 0.6667 / no_mass 0.0`, `weight_sensitive:false`, `dependence_sensitive:false`.
- **Unresolved:** `unresolved_missing_mechanism = 0.333336`. `evidence_coverage = 0.3333`. `residual_bound = 0.3141`; `simulation_probability_bounds = [0.6859, 1.0]`.
- **Headline formula (mass-weighted; resolved mass takes sim, unresolved mass takes prior):**
  **0.8333 = 0.6667 × 1.0 + 0.3333 × 0.5** = 0.6667 + 0.16665. `headline_source: "mass_weighted:partial_rollouts+grounded_prior"`.
- **Combiner:** `combined:null`, `combiner_available:false`, `method:"combiner_unavailable_range_only"`, `disagreement:0.5`, combined range [0.5, 1.0]. No fixed blend applied to the top-line probability, but the reported headline (0.8333) is the mass-weighted number above.
- **Prior vs simulation contribution to the headline:** simulation supplies 0.6667×1.0 = **0.6667**; prior supplies 0.3333×0.5 = **0.16665**. The simulation dominates and pins the top-line to the wrong side. (`evaluation`: prior-only Brier 0.25 vs lean_v2 Brier 0.6944 — the simulation actively *degraded* an already-uncertain-but-correct-side prior. `exp112_lean_v2` = 0.5 → Brier 0.25.)

---

## J. Root-cause diagnosis

**The user's audit question:** why did an action such as "some tankers transit" (or a blockade/disruption flag) become a YES DESPITE the question requiring ≥ ~50 daily transits, and what is the exact missing numerical bridge?

**(1) What terminal predicate did the blueprint actually use?**
- *Nominal (text):* `yes_when = "daily_transit_count >= 50 on any day in window"`, `threshold="50"` — a numeric comparison.
- *Effective (executable):* a **boolean event-occurrence** predicate. After `terminal_canonicalization`, YES ⟺ any of `{us_lift_blockade, iran_clear_mines, tanker_operators_transit}` sets `__terminal_yes__`. `preflight` confirms this: `terminal_writer_exists → "writers=['tanker_operators_transit','us_lift_blockade','iran_clear_mines']"`. There is **no numeric count and no comparison** in the running mechanism — the "50" survives only as decoration.

**(2) Which action wrote the terminal-YES key and what did it mean vs the numeric requirement?**
- On **0.3333** of mass, **`us_lift_blockade`** wrote `__terminal_yes__="lifted"`. Semantics: *"the US relaxed blockade enforcement"* — a necessary precondition for traffic to *begin* recovering, decades away from "≥50 tankers crossed in one day." This is the clearest wrongful YES.
- On the other **0.3333**, **`tanker_operators_transit`** wrote `__terminal_yes__="≥50"`. Semantics: *"operators sent vessels"* — the string `"≥50"` is a **hard-coded assertion**, not a measured count. Even the `selective_transit` state, whose own text says *"not enough to reach 50 per day,"* would (had it kept the template) have stamped `"≥50"`.
- `iran_clear_mines` is the third latent trip-wire (never fired here, since both IRGC states refused to clear mines).

**(3) Why did the mechanism-recovery ladder fail to build the numeric bridge?** (`mechanism_recovery_manifest.json`, verbatim)
- Attempt 1 `reuse_existing_mechanisms` → "**0 verbatim numeric(s) reused**" (no counted transit numbers existed to reuse — the evidence's only in-strait count is "~7 ships, none oil tankers").
- Attempt 2 `deterministic_threshold_parse` → "**no numeric threshold in resolution text**", `diagnosis.threshold: null`.
- `validated: false`; **failure_proof:** *"the resolution criterion carries no parseable numeric threshold — a bounded numeric mechanism is not the right bridge for this terminal."*
- The diagnosis correctly names the target variable: `"daily_transit_count >= 50 on any day in window"`, `required_input_type: "numeric series in the question's units"`, `evidence_needed: ["dated observations of: daily_transit_count >= 50 on any day in window", "regime conditions under which each observation held"]`.

**Both failure legs are real, and note the important nuance:** the threshold **50 IS present** in the evidence prose ("at least 50 oil tanker transits per day"), but the *deterministic parser* returned `threshold:null` — it could not extract the number from the sentence, so the recovery ladder concluded "no parseable numeric threshold" and abandoned the numeric bridge. Simultaneously, the evidence contained **no in-window / regime-conditioned transit counts** to ground a count model even if the threshold had parsed (Attempt-1's "0 verbatim numerics"). So it is a **conjunction**: (a) threshold-parse failure on text that *did* contain the bar, AND (b) genuinely absent count series. The `readiness_manifest` had already flagged the gap — check `terminal_units: ok=false, "numeric predicates need a bounded mechanism in question units"`, `repair: mechanism_recovery` — but the repair failed and the run fell back to the boolean canonical-key terminal (`readiness.terminal_writer: ok=true "canonical-key writer or bounded mechanism"`).

**(4) The EXACT missing bridge.** No stage ever built:
```
actor decisions  →  integer daily_transit_count  →  compare(count, 50)  →  YES/NO
 (blockade effectiveness, mine density, escort capacity,
  war-risk insurance availability, route/spot-rate economics)
```
The AIS `ais_tracking` mechanism (`deterministic_rule: "Count of oil tankers crossing… per day"`) was the natural home for this integer, but **no action feeds it**, and it is never evaluated. The rich magnitude signal that *did* exist in the actor states ("large numbers" / "below 10 per day" / "not enough to reach 50") was flattened into the binary `tanker_operators_transit` template, which stamps `"≥50"` for any nonzero transit.

**First wrong step & propagation.**
1. **First wrong step — blueprint compilation (call 0):** the `tanker_operators_transit` template's effect hard-codes `daily_transit_count="≥50"` instead of writing a state-dependent *number*, and gives `us_lift_blockade`/`iran_clear_mines` terminal-adjacent keys. The predicate *looks* numeric but its only inputs are literals.
2. **Amplified at canonicalization (F):** collapsing all three writers' keys into one boolean `__terminal_yes__` turned the numeric-looking predicate into an OR of de-escalation events, and promoted `us_lift_blockade`/`iran_clear_mines` to full terminal writers (blueprint had them `writes_terminal:false`).
3. **Mechanism-recovery (J-3) failed to repair it** (threshold-parse null + no count evidence), so the boolean terminal shipped.
4. **Propagation:** YES on **0.6667** of mass — 0.3333 from blockade-lift alone, 0.3333 from the asserted `"≥50"` transit. The remaining 0.3333 escaped only as *unresolved* (never as a modeled NO). Sim-only conditional 1.0 → headline pulled to 0.8333 → Brier 0.6944, wrong side.

**Would a correct numeric mechanism reverse the forecast toward the true NO? Yes.** With a proper bridge, (a) `us_lift_blockade` and `iran_clear_mines` would set *preconditions*, not the terminal; (b) `tanker_operators_transit` would add a *state-scaled count* (`risk_acceptance`→ tens, `selective`→ single digits/"not enough to reach 50", `risk_aversion`→ ~0), gated by blockade+mine status; (c) the terminal would compare the day's summed count to 50. Given the anchor evidence (window opens at ~7 ships/day with **zero** oil tankers, 94% below peacetime, blockade+mines persist in the majority of modeled worlds, and both operator states other than `risk_acceptance` explicitly stay sub-50), the resolved distribution would move heavily to **NO**, aligning with outcome 0. At minimum, the two `blockade_low + (risk_aversion|selective)` worlds that are currently YES purely on the lift-flag (mass 0.1667) would flip, and the `"≥50"`-asserting worlds would be re-scored against a real count. The prior-only variant already achieved the correct side (exp112 p=0.5, Brier 0.25); a faithful numeric mechanism would push below 0.5.

---

### Determinability notes
- Why each `blockade_low` world is YES is established directly from `terminal_canonicalization` + `preflight` (three canonical-key writers) and confirmed by the per-combo terminal table (D/H) — not inferred.
- Actor states were generated once (call 2) and *do* carry magnitude; the loss of that magnitude is a mapping fact (one binary action per operator), fully traceable.
- The `crisis` oil-state and any unused shared combos were pruned before weighting (`state_posteriors` enumerates the 6 used combos); this is stated in the trace, not assumed.
