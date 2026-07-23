# Forensic reconstruction — Bank of Japan June 2026 rate decision
**qid** `7279494c-a775-5a57-a5f2-ac22252fb286` · read-only audit of one completed Lean V2 run
Outcome (truth) = **1 (BOJ raised)**. Headline forecast = **0.0364**. Final sim Brier = **0.9285** (confidently wrong).

All numbers below trace to the committed trace files. Where a fact cannot be recovered from the trace I say so explicitly. Directory abbreviations: `CK` = `experiments/results/exp113_checkpoints/7279494c-…json`; `CO` = `…-completion/`; `CA` = `experiments/results/exp113_cache/boj/`.

---

## A. Question inputs

**Exact question** (`CO/llm_calls.jsonl` call 0 prompt): "Will the Bank of Japan raise its short-term policy interest rate at the June 15–16, 2026 Monetary Policy Meeting?"

**Resolution criterion** (cached `blueprint_response`, `CA/3b52….json`): resolves **Yes** if the official *Statement on Monetary Policy* published after the June 15–16 2026 meeting announces a guideline for the uncollateralized overnight call rate **above 0.75%** (e.g. 1.0%); **No** if maintained at ≤0.75% or the meeting does not occur. `resolution_day` 2026-06-16.

- **as_of**: 2026-05-14 · **horizon**: 2026-06-16 (`forecast_decomposition.json` horizon_days = 0).

**Evidence supplied** (verbatim, call 0 EVIDENCE block):
- Rate maintained at ~0.75% as of 2026-05-13; confirmed at the April 27–28 2026 meeting, "the Policy Board voted 6–3 to keep the rate unchanged."
- "Three dissenting members — Nakagawa Junko, Takata Hajime, and Tamura Naoki — proposed raising the rate to 1.0%, but their proposals were defeated." "This 6–3 split represents the widest division under Governor Ueda's tenure."
- "Reuters reported the BOJ has 'locked in' signals for a June rate hike, while nearly two-thirds of Reuters-polled economists expect the rate to reach 1.0% by end-June." Oxford Economics' Shigeto Nagai argues the BOJ will not have sufficient data by June. "Polymarket implied probability for a 25 basis point hike stands at approximately 63%."

**Grounded prior** (`CA/85df….json` outcome_reference_class; `forecast_decomposition.json`): **p = 0.875, n = 3**, hierarchy `broad_human_decision_class`, quantity "BOJ raises short-term policy interest rate at June 15-16, 2026 meeting", numerator/denominator 3/3, interval [0.6317, 1.0]. **Exact historical cases**: (case_0) March 18–19 2025 raise 0.5→0.75%; (case_1) Jan 23–24 2025 raise 0.25→0.5%; (case_2) July 30–31 2024 raise 0–0.1→0.25%. All `outcome: true` — a 3-for-3 recent-hike base rate that already points strongly YES.

**What was NOT supplied**: no post-cutoff data (May CPI, wage prints, GDP, actual June result); no per-member vote intentions beyond the April 6–3 record; no roster of the *specific* other five board members (deputy governors and members are never named — see B). The prior's specificity is only `broad_human_decision_class` (n=3), which the combiner treated as low-reliability.

---

## B. World compilation (blueprint / repair / grounding were persistent-cache HITS)

`CO/cache_manifest.json` records `hits_persistent: 4`, and events `blueprint_response` (key 3b5225d3) `persistent_hit`, `blueprint_repair_response` (c36725ef) `persistent_hit`, `reference_class_grounding` (85dfe3dc) `persistent_hit`, plus consequence_templates. Consistent with the anchor: `blueprint.from_cache=true`, `repair.attempted=true`, and **0 structural_generation / 0 reference_class_grounding external calls** this run.

> **Flag — prompts are not in-trace.** Because these were persistent-cache hits, the ORIGINATING PROMPTS that produced the blueprint, the repair, and the grounding are **absent from `CO/llm_calls.jsonl`** (which contains only the 4 state_generation + 37 actor_decision calls). Only the cached RESPONSE values in `CA/*.json` are available. Everything in this section is reconstructed from those responses and the parsed blueprint in `CK` provenance.

### Cached `blueprint_response` (`CA/3b5225d3….json`, quoted)
Key fields (the file's `value` is a JSON string; quoted verbatim):
- `causal_thesis`: *"The BOJ Policy Board votes on a proposal to raise the uncollateralized overnight call rate from 0.75% to 1.0%. **A simple majority of the 9 members is required.** The outcome hinges on whether the three April dissenters (Nakagawa, Takata, Tamura) are joined by at least two more members, or whether Governor Ueda and the majority maintain the current rate due to insufficient data."*
- **Actors modeled (5):** `ueda_kazuo` (Governor; discretion "decisive"; variants `cautious_gradualist`, `hawkish_leader`); `nakagawa_junko`, `takata_hajime`, `tamura_naoki` (each one variant `consistent_hawk`, support "well_supported", evidence = the April 1.0% dissent); and the **BLOC** `other_members` — *name "Other Policy Board Members (5 remaining)"*, aliases ["Remaining members"], two variants `dovish_majority` (evidence "April vote was 6-3 to hold") and `swing_voters` (evidence "Nearly two-thirds of Reuters-polled economists expect rate to reach 1.0%").
- **Institution** `boj_policy_board`: members `[ueda_kazuo, nakagawa_junko, takata_hajime, tamura_naoki, other_members]`, `decision_rule: "majority"`, procedure stage Vote 2026-06-16 rule *"Simple majority of 9 members"*.
- **Mechanism** `boj_vote_mechanism`: `deterministic_rule: "If votes for hike >= 5, rate is raised; else rate stays at 0.75%."`, `writes_terminal: true`.
- **Action template** `vote_on_rate`: actor_ids = all 5, effect `record_vote` options `["Raise to 1.0%", "Maintain at 0.75%"]`, emits `rate_decision_announcement`, `writes_terminal: true`.
- **Terminal** (verbatim): `{"kind":"institution_vote","institution_id":"boj_policy_board","decision_rule":"majority","rule_params":{"option":"Raise to 1.0%","threshold":"5"},"yes_when":"At least 5 votes for 'Raise to 1.0%'.","no_when":"Fewer than 5 votes for 'Raise to 1.0%'.","evaluation_day":"2026-06-16"}`.
- `grounded_rates`: Polymarket 0.63; economists 0.0–0.67. `outside_risks`: unexpected data could shift votes (`could_reverse:true`).

### Cached `blueprint_repair_response` (`CA/c36725ef….json`, quoted)
Structurally identical to the blueprint (same 5 actors, same bloc, same terminal `threshold:"5"`, same `deterministic_rule "votes for hike >= 5"`). The repair only (a) appended `"Policy Board membership"` to every actor's `authority`, and (b) added a one-sided-confirmation footer:
> `"one_sided_confirmed": true, "one_sided_reason": "The decision trigger references actor 'boj_policy_board' which is an institution, not an actor … so we set one_sided_confirmed to true because the board is the only entity that can announce the decision."`

The repair **did not touch the 9→5 roster collapse or the mis-scaled threshold.** It preserved `threshold:"5"` against a 5-unit board.

### Grounding (`CA/85dfe3dc….json`) — decision rights & obligations
- Shared conditions: `boj_gradual_hiking_cycle`, `dissenting_pressure_for_hike` (affects the 3 named hawks), `market_expectations_elevated`. Each is a single-case (n=1) counted class with rate 0.75, interval [0.3387, 1.0].
- Per-actor reference classes: the 3 hawks share key `5e11a22e51d9e743` "this member dissents on a rate hold" rate 0.75 (evidence = April dissent); Ueda key `444c7999d2b8e017` "this governor votes with the majority" rate 0.75.
- `institutional_obligations.boj_policy_board`: `deadline_day 2026-06-16`, `required_participants: ["governor_ueda_kazuo","deputy_governors","other_board_members"]`, allowed terminal actions include vote_to_raise/hold/cut, abstain, recuse, absent; `quorum: "majority of voting members"`, `delegation_allowed: false`.

### Incorrect / missing / over-compressed representation
1. **The 9→5 collapse (central defect).** The board has **9 voting members**; the blueprint's own `causal_thesis` and procedure say "majority of the 9." Yet only **5 decision units** are modeled: 4 named individuals + one bloc `other_members` labeled "5 remaining" (the anchor and the state text even call it "six remaining" in places — internally inconsistent; the blueprint name says 5, the state_generation call 0 reply says "The remaining six members"). Four (or five) real seats are compressed into a single actor casting a single vote.
2. **Mis-scaled terminal threshold.** `rule_params.threshold="5"` was authored for 9 seats but applied to 5 units. Per the anchor, the terminal law translates an absolute threshold ≥ modeled-member-count into a **majority of modeled units** ⇒ YES needs **≥3 of 5** (>50% of 5). Effective bar 60% vs the true 55.6% (5/9).
3. **Deputy governors / named members omitted.** The two deputy governors and the specific other members are never individuated; no aliases, no per-seat evidence.
4. **Bloc under-specification.** `other_members` gets only 2 states (hold / hike) and is treated as one correlated unit — see D/J for how this destroys the vote arithmetic.

---

## C. State construction and weighting

**Source calls.** State hypotheses came from **4 state_generation calls** (`CO/llm_calls.jsonl` calls 0–3, tier "strong"). Call 0 is the joint generator for all 5 actors; calls 1/2/3 are targeted regenerations for the three hawks (recovery — see below). `CO/state_recovery_manifest.json` (`total_recovery_calls: 3`): nakagawa/takata/tamura each `initial_state_count 1 → targeted_regeneration "generated 3 state(s)" → final 4`; ueda and other_members `final_source "generated"` unchanged. `reversal_search.ran=false` ("every actor already holds a reversal-capable state").

**Final states** (`CO/actor_states.jsonl`, 16 rows): ueda 2, nakagawa 4, takata 4, tamura 4, other_members 2.

**Residual bounds** (`state_recovery_manifest`): principals (ueda, nakagawa, takata, tamura) `residual_r 0.2`, provenance "counted out-of-set frequency 1/1 (capped)"; `other_members` `residual_r 0.1`, provenance "uncovered feasible option(s) ['maintain at 0.75%','raise to 1.0%'] with no counted cases — bounded per option". Joint residual bound = **0.63136** (`CK …engine_primary.joint_residual_bound`).

**Weights** (`CO/weight_provenance.json → state_posteriors`; identical across all 6 shared-condition combos — shared conditions carry NO numeric weight, only `aligned_condition` labels, so the run is `dependence_sensitive:false`). Effective terminal vote (R=Raise, M=Maintain) determined uniquely (see D/E):

| Actor | State | Weight | Evidence basis | Effective vote |
|---|---|---|---|---|
| ueda_kazuo | ueda_cautious_gradualist | **0.375** | April 6-3 hold; Nagai "insufficient data" | **M** |
| ueda_kazuo | ueda_forced_hike_by_dissent | **0.625** | Reuters "locked in"; Polymarket 0.63; 3 dissents | **M** (see G — dropped) |
| nakagawa | nakagawa_domestic_political_pressure | **0.75** | invented (regen call 1) | **M** |
| nakagawa | nakagawa_hawkish_conviction | 0.0833 | April 1.0% dissent | R |
| nakagawa | nakagawa_global_risk_aversion | 0.0833 | invented (regen) | R |
| nakagawa | nakagawa_internal_board_dynamics | 0.0833 | invented (regen) | R |
| takata | takata_strategic_compromise | **0.75** | invented (regen call 2) | **M** |
| takata | takata_data_dependent_hawk | 0.0833 | April dissent + Nagai | M |
| takata | takata_inflation_overshoot_fear | 0.0833 | April dissent | R |
| takata | takata_global_risk_aversion | 0.0833 | invented (regen) | M |
| tamura | tamura_global_risk_aversion | **0.75** | invented (regen call 3) | **M** |
| tamura | tamura_structural_hawk | 0.0833 | April dissent | R |
| tamura | tamura_credibility_hawk | 0.0833 | invented (regen) | R |
| tamura | tamura_strategic_compromise | 0.0833 | invented (regen) | M |
| other_members | other_swing_voters_hold | **0.5** | April 6-3 hold | M |
| other_members | other_swing_voters_hike | **0.5** | Reuters/economists | R |

**The weighting inversion (critical).** `weight_provenance.state_posteriors[nakagawa_junko]` provenance shows the counted reference class `5e11a22e51d9e743` (rate 0.75, "this member **dissents on a rate hold**" — i.e. votes FOR a hike) was **matched to `nakagawa_domestic_political_pressure`**, a *Maintain* state, and its complement (1−0.75=0.25) "shared among 3 unmatched modeled state(s)" (0.0833 each). The identical pattern holds for takata (0.75 on `strategic_compromise`, a hold) and tamura (0.75 on `global_risk_aversion`, a hold). So a reference class whose semantics is "probability this member votes to hike = 0.75" had its 0.75 mass pinned to a **hold-voting** state for each of the three known hawks. This inverts the three April dissenters into ~75–83% "maintain."

**Per-unit weighted P(effective Raise)** (weights × effective vote): ueda **0.000**, nakagawa **0.250**, takata **0.083**, tamura **0.167**, other_members **0.500**.

**Assessment.** States are *linguistically* rich but the recovery step over-generated: to satisfy the "1/1 out-of-set residual" completeness rule the engine regenerated 3 extra states per hawk, and the generator (calls 1–3) invented dovish reversal stories ("domestic political pressure," "global risk aversion," "strategic compromise") for members whose only actual evidence is a hawkish dissent. The weighter then handed the dominant 0.75 mass to one of those invented dovish states. Net: **disagreement-biased and wrongly-weighted** — the model manufactured doubt about the three members who, in reality, all voted to raise. `other_members`, a 4–5 person bloc, is a single actor with a symmetric 0.5/0.5 coin flip (residual 0.1/option).

---

## D. World generation

**Shared-condition worlds** (`CO/shared_worlds.jsonl`): 3 binary conditions → 8 combos, of which **6** survive as node prefixes `w0_sw0 … w0_sw5` (each 256 nodes; 2 low-weight combos pruned/coalesced). Because state posteriors are identical across all 6 combos, the shared worlds only replicate structure — they do not move probability (`dependence_range [0.0364, 0.0364]`, `weight_sensitive:false`).

**Actor-state cross product**: ueda 2 × nakagawa 4 × takata 4 × tamura 4 × other 2 = **256 distinct variant tuples** (confirmed: `world_trajectories.jsonl` has exactly 256 unique tuples). × 6 shared worlds = **1536 weighted nodes** (`CO/world_trajectories.jsonl`; `CK metrics.weighted_nodes_executed = 1548` — 12 more, attributable to deliberation bookkeeping; the resolved node set is 1536). Total mass **0.999746** (≈1.0; tiny truncation).

**Merging / reuse**: `CK …coalescer.merges = 0`, `branches_merged = 0`, `truncated_mass 0.0` — no node coalescing. Decision computation was heavily reused: **21 unique decision contexts**, `decision_reuses = 10368` (`CK`), `hits 10368 / misses 0` (`decision_manifest`). `largest_context_reuse`: six contexts reused 768× each (the always-present ueda/other/nakagawa-dominant states), then 384× tiers.

**Independence treatment**: nodes are the independent product of per-actor state weights; the independent-approx recomputation of P(≥3 of 5) from the six marginals reproduces **0.0364 exactly**, confirming the engine treated the five units as independent (no genuine cross-actor correlation entered the weights).

**How the bloc distorts the institution** (grouped, not dumped): raise-count distribution over the 1536 nodes (effective votes):

| # units voting Raise | weight | nodes | terminal |
|---|---|---|---|
| 0 | 0.28650 | 72 | NO |
| 1 | 0.46524 | 384 | NO |
| 2 | 0.21165 | 624 | NO |
| 3 | 0.03464 | 384 | YES |
| 4 | 0.00172 | 72 | YES |
| 5 | 0 | 0 | — |

**Five raises is unreachable** because Ueda's unit never contributes a Raise (C/G), so the maximum is 4 (nakagawa+takata+tamura+other). Grouping four real members into one `other_members` unit means the entire dovish/swing majority casts **one** vote worth **one** of the five units, instead of 4–5 of 9 seats — see J for the quantified reversal.

---

## E. Actor decisions — all 21 unique contexts

Source: `CO/llm_calls.jsonl` actor_decision rows (calls 4–40), `CO/decision_manifest.json → decisions.templates` (21) and `decision_trace`, `CO/actor_decisions.jsonl`. Of 37 actor_decision calls: **21 unique initial decisions** (calls 4–19 and 31–35) + **16 deliberations** ("the SAME person continuing…", calls 20–30, 36–40). Every decision fired on `trigger: "deadline"`, day 2026-06-16; shared conditions in play as listed; visible info = the sealed as-of evidence + `Institutional context` (decision_rule majority, votes_recorded often "omitted_at_terminal_closure"); feasible actions = `vote_on_rate` with options `Raise to 1.0% / Maintain at 0.75%` (initial) or `cast_vote` with `vote:Maintain at 0.75%, vote:Raise to 1.0%, abstain, recuse, be_absent` (forced-closure variant).

The 21 templates (`decision_manifest.templates`) map context → (actor, cohort/state, response_hash). Committed parsed vote per (actor,state) — reconciled across `actor_decisions.jsonl`, `decision_trace`, and the unique zero-mismatch fit to `world_trajectories` terminals:

| # | call_id(s) | actor | state (variant) | act_or_wait | parsed vote / committed | intended effect (from reply) |
|---|---|---|---|---|---|---|
|1|4|nakagawa|global_risk_aversion|act|**Raise 1.0%**|"Signal BOJ's resolve to control inflation, support yen"|
|2|12|nakagawa|hawkish_conviction|act|**Raise 1.0%**|preempt overheating|
|3|13|nakagawa|internal_board_dynamics|act|**Raise 1.0%**| (regen state; votes hike) |
|4|6,34|nakagawa|domestic_political_pressure|gather_information→|**Maintain 0.75%**|avoid fiscal/bond crisis; seek future-hike commitment|
|5|5|other_members|swing_voters_hold|act|**Maintain 0.75%**|follow Ueda's lead|
|6|18|other_members|swing_voters_hike|act|**Raise 1.0%**|"providing the majority needed to pass"|
|7|7|ueda|cautious_gradualist|act(→wait at closure)|**Maintain 0.75%**|wait for wage data|
|8|8|ueda|forced_hike_by_dissent|act|"Raise 1.0%" **→ DROPPED to Maintain**|"response to accumulating evidence and board consensus"|
|9|9,14,17,32,33|takata|data_dependent_hawk|wait|**Maintain 0.75%**|"vote for hike only if May data strong; else hold"|
|10|16|takata|inflation_overshoot_fear|act|**Raise 1.0%**|anchor expectations|
|11|—|takata|global_risk_aversion|act|**Maintain 0.75%**|avoid financial accident|
|12|—|takata|strategic_compromise|gather_information|**Maintain 0.75%**|trade rate for hawkish guidance|
|13|10,11|tamura|structural_hawk / credibility_hawk|act|**Raise 1.0%**|"secure a 5-4 or 6-3 majority for a hike"|
|14|15,19|tamura|global_risk_aversion / strategic_compromise|act|**Maintain 0.75%**|prevent crisis / preserve unity|
|15|31|ueda|forced_hike_by_dissent (**forced cast**)|act|`chosen:"cast_vote"`, `vote_option:"vote:Raise to 1.0%"` **→ not counted**|"use my vote to build a majority"|
|16|35|ueda|cautious_gradualist (**forced cast**)|act|`vote_option:"vote:Maintain at 0.75%"`|status quo|

(Deliberation calls 20–30, 36–40 returned `vote_option:""` with act_or_wait wait/gather_information — they did **not** change the eventual committed vote; e.g. call 23/38 ueda "wait", call 30/40 ueda "gather_information", then forced-cast 31/35.)

**Reuse & mass**: the six highest-mass contexts (ueda_cautious, ueda_forced, other_hold, other_hike, nakagawa_domestic, nakagawa-branch) each receive 768 reuses; because ueda and other_members appear in every node, their two states each blanket ~half the 1536 nodes.

**Realism note**: the *individual* replies are plausible and internally coherent (each member reasons from its assigned private reality). The failure is upstream — which realities were generated and how they were weighted — not the per-call reasoning. The clean initial hawkish votes (calls 4,8,10,11,12,13,16,18) show the model *can* produce Raise votes; the weighting and the forced-cast plumbing suppressed them.

---

## F. Consequences and interaction

- **Distinct actions**: exactly one action type, `vote_on_rate` (record_vote), plus the closure variant `cast_vote`. Every path emits the single event `rate_decision_announcement` (observers "public").
- **Consequences / world-state changes**: `record_vote` writes the terminal directly (`writes_terminal:true`). `CK …mechanism_used:false` and `mechanism_recovery_manifest.json pre_run.why:"terminal pathway already complete"` — no intermediate mechanism transformed state; votes are tallied directly into the terminal.
- **Messages / observers / later actors affected**: none functionally. `Institutional context.votes_recorded` is `[]` at initial stage and `"omitted_at_terminal_closure"` at closure — i.e. **members never observe each other's votes**. There is no information passing from one member's decision into another's prompt.
- **Reconsideration**: deliberation rounds exist (16 calls) but only let an actor defer (wait/gather); no actor revised a vote in response to another's.

**Verdict**: this is **five isolated, simultaneous votes**, not genuine interaction. The blueprint's causal thesis ("dissenters joined by ≥2 more," "swing members shift with engagement") is not mechanized — the swing dynamic that actually drives BOJ outcomes is absent.

---

## G. Deadlines and completion

- **Deadline**: `institutional_obligations.deadline_day 2026-06-16`; every decision trigger fired with `trigger:"deadline"`, `SITUATION NOW: the decision deadline has arrived — act now`.
- **Completion audit** (`CO/completion_audit_manifest.json`): `resolved_mass 1.0`, `unresolved_mass_by_cause {}`, one engine round: `reopened_decisions 2688, re_evaluated 1536`, policy `"deadline_forced_completion:reopen_then_eval"`, `max_rounds 2`. Acceptance all-OK (terminal_unknown_state_mass 0, provider_failure 0, resolved_share 1.0).
- **Waits/gathers**: multiple actors chose wait/gather in deliberation; at the hard deadline they were force-closed. `CK engine_primary.waves = 3`.

**Forced vs voluntary final votes.** Members who deferred were driven to a terminal vote by the closure step. Two of these forced casts are visible as LLM calls **31 (Ueda, forced_hike) and 35 (Ueda, cautious)** — the ONLY two `cast_vote` calls, both Ueda; the prompt reads *"the boj_policy_board decision deadline has arrived — you must now cast one of the allowed terminal actions"* with `Your prior decision: gather_information`. The named hawks' committed votes came from their initial/deliberation `vote_on_rate` (act/wait) with clean option strings and were counted normally.

**Mass whose final vote was forced/invented.** Ueda appears in 100% of nodes; his 0.625-weight `forced_hike_by_dissent` state produced, at closure, `vote_option:"vote:Raise to 1.0%"` (call 31). The terminal law counts only the exact string `"Raise to 1.0%"` (rule_params.option); the `"vote:"`-prefixed string **fails the match and is scored as non-Raise**. Empirically this is certain: the unique zero-mismatch fit to all 1536 terminals assigns `ueda_forced_hike_by_dissent → Maintain`, and a concrete pivotal node (Ueda_forced + exactly 2 other Raises) resolves **NO** (would be YES = 3 raises if Ueda's Raise counted). So **≈0.625 of Ueda's weight — i.e. ~62% of every node's Ueda contribution, ≈0.60 of total mass (`ueda_forced_hike` nodes sum to 0.625 weight) — had a hawkish Raise silently converted to Maintain by the forced-closure/option-string mismatch.** The proximate cause (malformed `"vote:"` prefix) is an inference from the call-31 reply; the *effect* (Ueda never contributes a Raise) is proven from the terminal data.

**other_members' vote was NOT forced/invented**: both other-member votes came from clean `act` decisions (calls 5, 18), option strings `"Maintain at 0.75%"` / `"Raise to 1.0%"`, both counted correctly (effective M / R).

---

## H. Terminal outcome

Terminal law: `institution_vote`, majority; **YES iff ≥3 of the 5 modeled units vote "Raise to 1.0%"** (mis-scaled threshold-5 translated to majority-of-5). Materially distinct terminal groups by raise-count (from D):

- **NO groups** (0/1/2 raises): 0.28650 + 0.46524 + 0.21165 = **0.96339** (1080 nodes).
- **YES groups** (3/4 raises): 0.03464 + 0.00172 = **0.03636** (456 nodes). (5 raises impossible.)

**Sum**: YES 0.03636 + NO 0.96339 = 0.99975 ≈ 1.0 → renormalized headline **YES = 0.0364**. Matches `CK forecast_decomposition.simulation_conditional.provenance` (`yes_mass 0.0364, no_mass 0.9636`) and `CO/world_trajectories.jsonl`. **Mass conserved** (total 0.999746; residual is truncation, `truncated_mass 0.0`).

The YES mass sits almost entirely in the "3 raises" band, which requires `other_members=hike` (0.5) plus two of the three low-probability hawks simultaneously — a rare joint event.

---

## I. Forecast decomposition (`CO/forecast_decomposition.json`)

- **Grounded prior**: p = **0.875** (n=3, broad_human_decision_class, 3/3, interval [0.6317, 1.0]).
- **Simulation conditional**: p = **0.0364**, resolved_mass **1.0**, interval [0.0, 0.2278], `weight_sensitive:false`, `dependence_sensitive:false`, yes_mass 0.0364 / no_mass 0.9636.
- **Unresolved**: none (`unresolved_mass_by_cause {}`, unknown_state_mass 0.0). Residual bound **0.63136**; simulation_probability_bounds [0.0, 0.7153].
- **Combiner**: `combined: null`, `method:"combiner_unavailable_range_only"`, `combiner_available:false`, `fixed_blend_used:false`. Prior and sim reported separately; **combined_interval [0.0364, 0.875]**. `disagreement 0.8386` (prior_sim_divergence).
- **Headline**: **0.0364**, `headline_source "mass_weighted:completed_rollouts+grounded_prior"`. Exact formula this run: with 100% resolved simulation mass and no fitted combiner, the mass-weighted headline collapses to the **simulation conditional (0.0364)**; the grounded prior contributes ≈**0** to the headline despite pointing to 0.875. Brier: sim 0.9285 vs prior 0.0156 (`CK metrics`) — the prior alone would have been nearly perfect; the simulation overrode it.

---

## J. Root-cause diagnosis — auditing the 9→5 collapse

**The user's question: how did modeling nine members as five units (incl. `other_members`) change the threshold and the forecast, and could correcting it reverse the forecast toward the true YES?**

### 1. Threshold arithmetic
- **Reality**: 9 members, simple majority = **5 of 9 = 55.6%**.
- **Model**: 5 units, `rule_params.threshold:"5"` authored for 9 but applied to 5; terminal law (anchor) translates thr ≥ modeled-count into majority-of-modeled ⇒ **YES needs ≥3 of 5 = 60%**. The absolute bar (5 votes) was silently softened to 3, but the *denominator* shrank from 9 to 5. The bar in seat-terms: a Raise must win 3 of {Ueda, Nakagawa, Takata, Tamura, bloc}.

### 2. Vote distribution across the 21 contexts / per unit
Weighted P(effective Raise): Ueda **0.000**, Nakagawa **0.250**, Takata **0.083**, Tamura **0.167**, bloc **0.500**. Of the 21 unique contexts, Raise was the committed vote in exactly these: nakagawa {hawkish, global_risk, internal} (3), takata {inflation_overshoot} (1), tamura {structural, credibility} (2), other {hike} (1), ueda {forced_hike initial call 8 — later dropped} — i.e. **7 of 21 contexts vote Raise**, but they carry small weight (each 0.0833) except the bloc (0.5). The bloc's Raise counts as **one** of five units.

### 3. Why the mass concentrated at NO 0.964
Three compounding errors, in order of first occurrence:

**(1st wrong assumption — the roster/threshold collapse, in the CACHED blueprint.)** Four/five real seats became one `other_members` unit. Consequence: the dovish-or-swing majority can supply at most **1** of the 5 votes needed, and the three known hawks + governor must supply the other 2–3. This is the FIRST error and it is baked into the cached blueprint (`members:[…, other_members]`, `threshold:"5"`); the repair pass saw it and preserved it.

**(2nd — state over-generation + weight inversion.)** Residual-recovery regenerated 3 invented dovish states per hawk (calls 1–3), and the weighter pinned the counted "dissents-for-hike rate = 0.75" onto a *Maintain* state for each (§C). This dropped the three April dissenters from ~certain-Raise to P(Raise) 0.08–0.25. Trace: `weight_provenance.state_posteriors[nakagawa].weights = {domestic_political_pressure:0.75(M), hawkish_conviction:0.0833(R), …}`.

**(3rd — the governor's Raise dropped at forced closure.)** Ueda's 0.625 hawkish state cast `"vote:Raise to 1.0%"` (call 31) which the terminal did not count, fixing Ueda's P(Raise)=0 (§G).

Together: to reach 3 raises you need the bloc coin-flip (0.5) AND two of three suppressed hawks AND get nothing from Ueda — a joint probability of 0.0364.

### 4. Mass each error moved (counterfactuals on the actual node weights)
| Scenario | YES mass |
|---|---|
| As-run (baseline) | **0.0364** |
| Fix only Ueda's dropped Raise (forced_hike → Raise) | **0.169** |
| Model bloc as **5 seats**, 9-seat board, majority ≥5 of 9 (baseline votes) | **0.500** |
| Bloc-as-5 **and** Ueda fixed | **0.500** |
| Lower bar to ≥2 of 5 (baseline votes) | 0.248 |
| ≥2 of 5 and Ueda fixed | 0.539 |

- Fixing the governor plumbing alone lifts YES 0.036→**0.169** (~4.6×): Brier 0.93→0.69.
- **Weighting the bloc as the 5 real seats it represents is decisive**: when `other_members=hike` (weight 0.5) that alone is 5 of 9 = majority, so YES = P(bloc hikes) = **0.500**, independent of everyone else. Brier 0.93→**0.25**. This single correction moves the forecast from confidently-wrong to a coin flip on the right side.
- If, additionally, the hawk weight-inversion were corrected (restoring the three dissenters to ~0.75 Raise as their reference class intends), the bloc-as-5 model would exceed 0.5 and approach the prior's 0.875, i.e. the forecast **reverses to YES** and matches the outcome.

### 5. Verdict
The **first wrong assumption is the 9→5 roster collapse with the bloc modeled as a single vote** (cached blueprint, un-repaired). It set a structure in which the true swing majority is one out of five units and five-Raise unanimity is unreachable. It propagated through (a) the residual-driven over-generation of dovish states, (b) the reference-class weight inversion that neutered the three real dissenters, and (c) the forced-closure option-string mismatch that neutered the governor. The collapse plus these two amplifiers pushed **~0.93 of the mass onto NO** against a 3-for-3 prior and a 63% market. **Correcting the roster (or weighting the bloc as 5 votes) reverses the forecast toward the true YES** (0.036 → 0.50, and → ~0.875 if the hawk weights are also un-inverted).

Trace anchors: blueprint `members` & `threshold:"5"` (`CA/3b52…`, `CA/c367…`); `weight_provenance.state_posteriors` (0.75 on hold-states); call 31 forced `"vote:Raise to 1.0%"`; `world_trajectories.jsonl` terminal masses YES 0.03636 / NO 0.96339; `forecast_decomposition.json` headline 0.0364, disagreement 0.8386.
