# Forensic reconstruction — Matthew Wale / Solomon Islands PM

**qid:** `741b4bed-7502-5cd2-9cbe-949fbc70f857`
**Question:** "Will Matthew Wale be elected as the next Prime Minister of Solomon Islands following the May 2026 no-confidence vote?"
**as_of:** 2026-05-07 · **horizon/eval_day:** 2026-05-21 · **model:** deepseek-v4-flash · **seed:** 0
**Headline forecast:** 0.1302 · **Outcome:** 1 (Wale WON — upset) · **Brier ≈ 0.757 (confidently wrong)**

READ-ONLY audit. No code was modified, nothing was re-run, no external service called. Every number below traces to a committed trace file. Where a value is not determinable from the trace, that is stated.

A note on provenance up front (per instructions): the **blueprint, blueprint-repair, reference-class grounding, and consequence templates were persistent-cache HITS** from an earlier run (`compile_cache: hits_persistent=4, misses=1`, `blueprint.from_cache=true`). Their *response* values are committed in `experiments/results/exp113_cache/wale/*.json` and are quoted below, but **the exact PROMPTS that produced them are NOT present in this run's `llm_calls.jsonl`** (which contains only the 30 in-run calls: 6 `state_generation` + 24 `actor_decision`). Blueprint/grounding prompts are therefore reconstructed from their cached responses only.

---

## A. Question inputs

**Exact question** (`metrics.question`): "Will Matthew Wale be elected as the next Prime Minister of Solomon Islands following the May 2026 no-confidence vote?"

**Resolution criterion** (cached `blueprint_response.resolution`, echoed verbatim in state-gen prompts):
- YES: "Matthew Wale is formally elected or appointed as Prime Minister of the Solomon Islands and sworn in by the Governor-General on or after May 7, 2026."
- NO: "Any other individual is elected and sworn in as Prime Minister following the no-confidence vote; **or** No new Prime Minister has been sworn in by August 1, 2026."
- `resolution_day` in blueprint = "2026-05-21"; obligations `deadline_day` = "2026-05-21"; grounding `deadline_day` = "2026-08-01".

**as_of** 2026-05-07; **horizon** 2026-05-21 (`horizon_days` in decomposition = 0 — as_of and eval essentially collapsed).

**Evidence supplied** (the EVIDENCE block quoted identically into all 6 state-gen prompts): resolution rules; the sourcing rule (Solomon Islands Govt / ABC / RNZ); and the anchor fact "On May 7, 2026, Solomon Islands Prime Minister Jeremiah Manele was ousted via a motion of no confidence, **losing 22-26** in a parliamentary vote." That "22-26" is the only quantitative floor fact given — it implies a ~26-seat winning bloc in a 50-seat house.

**Grounded prior** (`forecast_decomposition.grounded_prior`, `reference_class_grounding.outcome_reference_class`):
- p = **0.1667**, n = 2, interval [0.0, 0.4732], `hierarchy_level = broad_human_decision_class`, quantity "Matthew Wale is elected Prime Minister", **numerator 0 / denominator 2** → low prior by construction.
- The two historical cases are **NOT about Wale** and both resolved NO ("Wale elected" = false):
  - case_0 (2019-04-24, RNZ): "Manasseh Sogavare elected prime minister after no-confidence vote, not opposition leader." outcome=false.
  - case_1 (2014-12-09, ABC): "Manasseh Sogavare elected prime minister from coalition." outcome=false.
- Note the reference class is a generic "after a Solomon Islands leadership contest, did the *specific-named person* (here Wale) win?" — and since Sogavare (not Wale) won both counted cases, both count as 0. This is a **base-rate for a named non-incumbent, not a coalition-arithmetic estimate.**

**What was NOT supplied:** the individual identities/party affiliations of the ~50 sitting MPs; the coalition's whip arrangement; any statement that Wale's coalition had actually agreed on Wale as its candidate; polling; and (critically) the fact that the coalition that ousted Manele already commanded ≥26 votes and could seat its nominee. The 50-seat house size appears only inside the blueprint's own prose ("out of 50 total parliamentary seats"), not as a structured parameter the engine used.

---

## B. World compilation (cache-reconstructed)

**Source & caveat.** `provenance.lean_v2.blueprint.hash = "725f8b172ab01adf+repair"`, `from_cache=true`, `n_actors=6`, `n_action_templates=1`. The parsed blueprint lives in the checkpoint; the raw model text is `exp113_cache/wale/28fa62a3….json` (`kind:"blueprint_response"`) and the repaired text is `cc4acc07….json` (`kind:"blueprint_repair_response"`). **Their prompts are not in-trace.**

**Cached `blueprint_response` — key quotes** (verbatim from the cache value):
- `causal_thesis`: *"Matthew Wale, as opposition leader, seeks to convert his coalition's parliamentary majority into his own election as Prime Minister, but fluid coalition politics and rival candidates like John Agovaka may produce a compromise candidate, determining the outcome."*
- `alternative_causal_reading`: *"John Agovaka, not Matthew Wale, is the most likely PM candidate, so the opposition may coalesce around him." … evidence_quote: "The Lowy Institute identified Agovaka as the 'most likely PM candidate from the opposition'."*
- Institution: *"members: [matthew_wale, john_agovaka, frederick_kologeto, jeremiah_manele, opposition_coalition_mps], decision_rule: majority, rule_params:{threshold:26}, procedure … 'Secret ballot among all 50 MPs; candidate with majority (26+) elected.'"*
- Terminal: *"kind: institution_vote … rule_params:{option:'Matthew Wale', threshold:26}, yes_when:'Matthew Wale receives at least 26 votes.'"*

**Cached `blueprint_repair_response` — the decisive repair quote:**
> `one_sided_confirmed: true`, `one_sided_reason`: *"The decision trigger references 'parliament' as an actor, but 'parliament' is an institution, not an actor; the trigger is corrected to use the institution's members, but since the world genuinely permits only one outcome path (the institution votes), no fake path is invented."*

The repair fixed a trigger-typing bug. It did **not** touch the two structural defects that decided the forecast: (i) the 50-seat body is represented by **5 "members"**, and (ii) `threshold:26` is left as-is against a 5-member roster. `terminal_canonicalization: {needed:false, why:"vote terminals read the tally directly"}` — so at run time the mis-scaled 26-of-50 threshold is applied as a **majority of the 5 modelled members → YES needs ≥3 of 5 to vote "Matthew Wale."** (Confirmed empirically in §H: a ≥3-of-5 rule reproduces all 1458 terminals with zero mismatch.)

### Included actors (6)
| id | role in model | modelled states | decisive? |
|---|---|---|---|
| matthew_wale | Leader of Opposition (the candidate) | 3 | yes |
| john_agovaka | "most likely PM candidate" (rival) | 3 | yes |
| frederick_kologeto | leader, People's First Party; tabled the motion | 3 | yes |
| jeremiah_manele | **ousted** PM, now an MP | 3 | yes |
| opposition_coalition_mps | **"26-28 MPs from six parties" collapsed into ONE actor** | 3 | yes |
| governor_general | ceremonial (swears in) | 1 | no (not a parliament member) |

**Parliament institution** = 5 members (the four named individuals **+ the single coalition bloc**). `decision_rule:"majority"`, `rule_params:{threshold:26}`.

### Omitted / over-compressed representation (the core compilation faults)
1. **Roster collapse.** The real 50-seat National Parliament is modelled as **5 voting units**. The ~26–28-MP majority coalition that *did the ousting* is a **single actor of equal weight** to lone individuals — including the man it just deposed. The other ~44 individual MPs are not modelled at all; the grounding file even *knew* real names (`institutional_obligations.required_participants = [matthew_wale, frederick_kologeto, john_maneniaru, rick_hou, peter_kenilorea_jr, jeremiah_manele]`) but these were discarded in favour of the 5-actor blueprint roster.
2. **Threshold not rescaled.** `threshold:26` was carried into a 5-member body. The engine reinterprets it as "majority" (≥3 of 5). So the coalition-of-26 fact and the number 26 both survive in text but are severed from the mechanism.
3. **Manele given a full, permanent anti-Wale vote.** The ousted PM sits in the 5-member electorate as an equal, and his one "well_supported" state is `manele_opposition` (*"I will oppose the coalition that ousted me … Block Wale or Agovaka … relationships:{Matthew Wale:'enemy'}"*). In a 5-vote house that is a structural 1/5 permanent NO.
4. **Aliases / rights:** every named actor holds `authority:["Vote in PM election","Negotiate coalition support"]` and `discretion:"decisive"`; the coalition bloc holds only `["Vote in PM election"]`. Aliases are the surnames (Wale, Agovaka, Kologeto, Manele) and "Coalition MPs". GG is `discretion:"ceremonial"`.
5. **Shared conditions (grounding):** `solomon_islands_coalition_instability` (rate 0.8333, n=2 — both Sogavare cases), `china_taiwan_diplomatic_rivalry` (0.75, n=1), `economic_dependence_on_aid` (0.75, n=1). See §C/§D for how the instability condition was pinned to "fluid" in 100% of mass.
6. **Mechanisms / deadlines / templates:** one mechanism `pm_election_mechanism` ("candidate receiving at least 26 votes … becomes PM", `writes_terminal:true`); one action template `vote_for_pm` (options `["Matthew Wale","John Agovaka","Frederick Kologeto","Jeremiah Manele","Other candidate","Abstain"]`, `writes_terminal:true`); temporal anchors 2026-05-07 (ouster) and 2026-05-11 (PM vote "expected"); obligations `deadline_day 2026-05-21`, `abstention_allowed:false` (yet Abstain remained in the option set and was used — see §E). Consequence template `vote_for_pm` cached in `91ce04e4….json` (`source:"blueprint_precompiled"`).

The blueprint's own text is self-aware of the problem — `unresolved_assumptions:[{"assumption":"All 50 MPs vote in the PM election.","reversal_capable":true}]` and `grounded_rates:[{quantity:"Opposition coalition size", value_range:[0.52,0.56], basis:"26-28 MPs (out of 50 total)"}]` — but none of that 52–56% majority fact reaches the terminal, which sees only 5 equal votes.

---

## C. State construction and weighting

**Recovery ladder** (`state_recovery_manifest`): the cached blueprint carried private-state *variants* but the compiled slice arrived with **`empty_sets_detected: 5`** (all five decisive actors had `initial_state_count: 0`). Attempt 1 `deterministic_alias_parse_repair → nothing_to_repair`; attempt 2 `targeted_regeneration → generated 3 state(s)` for each — this is exactly the **6 `state_generation` LLM calls** in `llm_calls.jsonl` (call 0 = batch proposal for all actors; calls 1–5 = one per actor regeneration). `total_recovery_calls: 5`, `final_source: "regenerated"` for all, `residual_r: 0.2` each, `under_modeled: false`, `reversal_search.ran: false` ("every actor already holds a reversal-capable state"). So the states actually used were **freshly generated in this run**, not the cached blueprint variants.

**EXACT state-generation prompts** (shared skeleton, per `llm_calls.jsonl` calls 1–5):
> "ONE actor in a causal simulation is missing their private-state hypotheses. Generate 2-3 genuinely DIFFERENT plausible private realities for EXACTLY this actor … You describe WHICH realities are possible; you NEVER say how probable they are. Actor: `<id>` — `<role>` at National Parliament of Solomon Islands. Decision faced: The question resolves Yes if Matthew Wale is elected and sworn in … Institutional incentives: majority. Known relationships: {}. Shared world conditions in play: china_taiwan_diplomatic_rivalry, economic_dependence_on_aid, solomon_islands_coalition_instability. Actor-local evidence (as-of-sealed): `<resolution + ouster 22-26 fact>`."

(Call 0's batch prompt is the same idea for all five actors at once: *"Propose the genuinely DIFFERENT private realities each decisive actor could be in, as of 2026-05-07 …"*.) Note `Known relationships: {}` — the state generator was given **no relationship graph**, so each actor was imagined in isolation with no signal that four of the five were coalition partners who had just co-operated to oust Manele.

### The 15 generated states, their weights, and how each finally VOTES

Weights are from `weight_provenance.state_posteriors` (identical across all 6 shared-condition combos — the shared conditions do not enter the vote). Marginal masses confirmed against `world_trajectories.jsonl`. "Votes" column is the final terminal vote (see §E for how it was reached; `act`=freely cast, `forced`=deferred/gathered at the mandatory terminal then written from `action_if_state`).

**matthew_wale** (residual/unknown 0.0):
| state | wt | claim (abridged) | final vote |
|---|---|---|---|
| confident_winner_with_china_backing | 0.375 | secret China backing will sway undecided MPs | **Matthew Wale** (act) |
| fearful_of_coalition_collapse | 0.375 | coalition fragile; rivals plotting defection | *no self-vote* — gather_information → forced |
| resigned_to_loss_planning_exit | 0.25 | "does not have enough votes … preparing to concede" | **Abstain** (act, call 29: "I'll keep my decision to abstain") |

**john_agovaka** (unknown 0.2):
| state | wt | claim | vote |
|---|---|---|---|
| wale_strong_clean_sweep | 0.333 | "Wale has locked in 28+ votes … clear reform mandate" | **Matthew Wale** (act) |
| dark_horse_self_promotion | 0.333 | deadlock opens a path for himself | **John Agovaka** (defer → forced) |
| wale_weak_compromise | 0.333 | Wale weak; may abstain to avoid a failing bloc | **Abstain** ("maintain_abstention") |

**frederick_kologeto** (residual 0.0):
| state | wt | claim | vote |
|---|---|---|---|
| wale_as_compromise | 0.375 | Wale is the only unifier; endorse him | **Matthew Wale** (act) |
| self_ambition | 0.25 | can become PM himself | **Frederick Kologeto** (act) |
| wale_as_threat | 0.375 | Wale would purge his allies; sabotage | **Abstain** (act) |

**jeremiah_manele** (residual 0.0):
| state | wt | claim | vote |
|---|---|---|---|
| backroom_deal_for_comeback | 0.75 | broker secret deal to return as PM | **Jeremiah Manele** (act) |
| accept_defeat_and_seek_exile | 0.125 | quit politics, seek a post abroad | **Frederick Kologeto** (act) |
| spoil_wale_through_constitutional_challenge | 0.125 | injunction / boycott to force fresh election | **Jeremiah Manele** (gather → forced) |

**opposition_coalition_mps** (unknown 0.2):
| state | wt | claim | vote |
|---|---|---|---|
| wale_as_unifying_figure | 0.333 | "only leader who can hold the six-party coalition together" | **Matthew Wale** (act) |
| factional_split_over_wale | 0.333 | a faction secretly backs a dark horse (Kologeto) | **Frederick Kologeto** (gather → forced) |
| pragmatic_deal_with_manele_loyalists | 0.333 | power-share that returns Manele/ally | **Jeremiah Manele** (defer → forced) |

**Reference classes / residual.** Manele's `backroom_deal` weight 0.75 is anchored to the counted class `788f6b12…` ("Jeremiah Manele is re-elected PM", rate 0.75, n=1 — its only in-window case is his 2024 election; the 2026 ouster case was excluded as post-as_of "leakage"). Wale's and Kologeto's own-election classes (`fc37…`, `16bc…`) both counted **0/1 → 0.25**, which is why `confident`/`wale_as_compromise` each get only 0.375 and the complement is spread over the rival states. `actor_residual_bounds = 0.2` for every actor; `joint_residual_bound = 0.737856`. `dependence_sensitive:false`, `weight_sensitive:false` (`state_posteriors.provenance` = counted_reference_class + counted_complement; "grounded, not a label").

### Realism assessment — implausible candidate/MP behaviour (this is the crux)
- **The candidate abstains from his own election in 62.5% of his own mass.** Matthew Wale votes for Matthew Wale in only the `confident` state (wt 0.375). In `fearful` (0.375) his forced action is to negotiate/delay (no self-vote) and in `resigned` (0.25) he explicitly **Abstains** (call 29, `chosen_action:"abstain"`). A sitting candidate who has forced a no-confidence vote does not, in reality, decline to vote for himself in 62.5% of futures. This is a state-generation artefact of imagining Wale "in isolation" against a low own-election base rate.
- **The coalition-of-26 backs Wale in only 1 of 3 states.** The bloc that supplies the majority votes Wale (`wale_as_unifying`) only 1/3 of the time; the other two states have the *majority coalition* voting **Kologeto** or **Manele** — i.e. the coalition installs the very man it ousted. Given the premise "the coalition had the numbers for Wale," two of the bloc's three states are coalition-arithmetic contradictions.
- **Manele (ousted) is a full anti-Wale elector.** His dominant state (0.75) votes himself back in; he never votes Wale. Giving the deposed incumbent a 1/5 vote with 0% chance of backing the winner is a structural drag.
- **Framing bias.** Every actor's evidence header repeats "losing 22-26" and the shared "coalition instability" condition (base-rated on two *Sogavare-won-instead* cases). The states are duplicate-free and internally coherent, but the whole state space is tilted toward "coalition fractures / compromise candidate," matching the cached `causal_thesis` and `alternative_causal_reading` (Agovaka). No state encodes the most likely real world: "the 26-MP coalition whips its members and seats its agreed leader."

---

## D. World generation

**Shared-condition worlds** (`provenance.lean_v2.shared_condition_worlds`): 6 worlds = china_taiwan{pro_china, pro_taiwan, neutral} × aid{high, low}, **all pinned to `solomon_islands_coalition_instability = fluid_coalitions`** (the `stable_coalitions` state carries 0 mass). Weights: the three high-aid worlds 0.25 each (=0.75), the three low-aid worlds 0.0833 each (=0.25). These conditions **do not feed the terminal** (the ≥3-of-5 rule reproduces all outcomes without them — §H), so they act only as a 6× replication factor and a framing wrapper that hard-codes "coalitions are fluid" everywhere.

**Combos.** Each of the 5 decisive actors has 3 states → **3⁵ = 243 distinct outcome combos**. 243 × 6 shared worlds = **1458 weighted nodes** (`world_trajectories.jsonl` = 1458 rows; `metrics.weighted_nodes_executed = 1470` counts a handful of scaffold/root nodes). Total mass = 1.0002 ≈ 1.0. `metrics.branches_merged = 0` (coalescer merged nothing — the 243 combos are all structurally distinct).

**Reuse / independence.** `engine_primary.decisions`: `unique_decision_contexts = 15`, `stores = 15`, `hits = 7290`, `reuses = 7290`, `misses = 0`; the 10 `largest_context_reuse` entries are each **n_reuses = 486** (a single actor-state decision reused across all 486 nodes that share it — 243 combos/3 × 6 worlds). Actors are drawn **independently** (`dependence_sensitive:false`); there is no correlation structure linking, say, "coalition unified" with "Wale confident." Real coalition whipping is exactly such a correlation, and its absence is a modelling loss.

**How the roster collapse distorts the 50-seat body.** In reality one bloc of ~26 controls the floor; here that bloc is 1 of 5 equal votes, drawn independently, backing Wale only 1/3 of the time, while the ousted PM holds another 1/5. The 50→5 compression converts a "coalition-with-the-numbers" problem into a "5 near-independent notables each ~⅓ likely to favour Wale" problem — a completely different (and far more adverse) arithmetic.

---

## E. Actor decisions — all 15 unique contexts (+ 9 deliberations)

24 `actor_decision` calls: **calls 6–20 = the 15 first-pass unique contexts** (one per state; prompt opens *"You are simulating ONE real person's complete moment of bounded cognition and decision, as of 2026-05-21…"*), **calls 21–29 = 9 deliberation re-asks** (prompt opens *"You are the SAME person continuing the SAME moment of thought — no new outside information exists…"*) for the states that did not cast a firm vote on the first pass. `behavioral_replicates_per_decision_context = 1`. All triggers are `mandatory_terminal`; `trigger x chosen` across the 200 deduped node-decisions = **cast_vote 134, defer 40, gather_information 26**. Model fingerprint `deepseek-v4-flash`, `prompt_version lean_v2.prompts.v1`, all `validation_record.ok=true`.

The prompt supplies each persona its own `Your private beliefs:` line (the state), its authority ("Negotiate coalition support, Vote in PM election"), the vote option set, and the 2026-05-21 deadline. Feasible actions = `vote:{Matthew Wale, John Agovaka, Frederick Kologeto, Jeremiah Manele, Other candidate, Abstain}` plus wait/gather. Visible info: no cross-actor tallies (secret ballot framing) — each decides blind to others.

| call | actor | state (belief cue) | chosen / act_or_wait | vote | reused mass | change on delib? |
|---|---|---|---|---|---|---|
| 6 | Kologeto | wale_as_threat ("Wale…will exclude him") | cast_vote/act | **Abstain** | 486 (wt 0.375) | — |
| 7 | Kologeto | wale_as_compromise ("Wale…acceptable") | cast_vote/act | **Matthew Wale** | 486 (0.375) | — |
| 8 | Manele | accept_defeat ("irreversible loss") | cast_vote/act | **Frederick Kologeto** | 486 (0.125) | — |
| 9 | Wale | fearful ("temporary alliance, not stable majority") | first-pass unsettled | → see call 28 | 486 (0.375) | **yes** |
| 10 | Kologeto | self_ambition ("legitimacy…tabling motion") | cast_vote/act | **Frederick Kologeto** | 486 (0.25) | (re-asked call 21, unchanged) |
| 11 | Wale | resigned ("coalition controlled by Kologeto") | first-pass unsettled | → call 29 | 486 (0.25) | **yes** |
| 12 | Coalition | wale_as_unifying ("Wale…necessary experience") | cast_vote/act | **Matthew Wale** | 486 (0.333) | — |
| 13 | Wale | confident ("China's leverage is decisive") | cast_vote/act | **Matthew Wale** | 486 (0.375) | — |
| 14 | Coalition | factional_split ("Wale cannot be trusted") | first-pass unsettled | → call 23 | 486 (0.333) | **yes** |
| 15 | Agovaka | wale_strong ("Wale locked in 28+ votes") | cast_vote/act | **Matthew Wale** | 486 (0.333) | — |
| 16 | Agovaka | dark_horse ("coalition fragile…Manele's") | first-pass unsettled | → call 22 | 486 (0.333) | **yes** |
| 17 | Coalition | pragmatic_deal ("ouster = renegotiating terms") | first-pass unsettled | → call 24 | 486 (0.333) | **yes** |
| 18 | Manele | backroom_deal ("fluid…personal loyalty") | first-pass unsettled | → call 25 | 486 (0.75) | (re-asked, → Manele) |
| 19 | Agovaka | wale_weak ("Wale lacks discipline") | first-pass unsettled | → call 27 | 486 (0.333) | **yes** |
| 20 | Manele | spoil ("procedurally flawed; judiciary") | first-pass unsettled | → call 26 | 486 (0.125) | (→ gather, forced Manele) |

**Deliberations (21–29), final resolution:**
| call | actor | state | final chosen | final vote |
|---|---|---|---|---|
| 21 | Kologeto | self_ambition | cast_vote/act | Frederick Kologeto |
| 22 | Agovaka | dark_horse | defer/wait → forced | John Agovaka |
| 23 | Coalition | factional_split | gather_information → forced | Frederick Kologeto |
| 24 | Coalition | pragmatic_deal | defer/gather → forced | Jeremiah Manele |
| 25 | Manele | backroom_deal | cast_vote/act | Jeremiah Manele |
| 26 | Manele | spoil | gather_information → forced | Jeremiah Manele |
| 27 | Agovaka | wale_weak | "maintain_abstention"/wait | Abstain |
| 28 | Wale | fearful | gather_information → forced | *(no Wale vote)* |
| 29 | Wale | resigned | "abstain"/act | Abstain |

**Verbatim realism flags:**
- Call 29 (Wale, resigned): *"…staying to fight risks a humiliating loss… My family and party want me out, and a graceful exit now preserves a chance for a diplomatic post later… I'll keep my decision to abstain."* → **the candidate declines to vote for himself.**
- Call 15 (Agovaka, wale_strong) accepts Wale has "locked in 28+ votes" and casts for Wale — the one state that internalises the real coalition arithmetic. It occurs 1/3 of Agovaka's mass.
- Call 12 (Coalition, wale_as_unifying) is the only bloc state that votes Wale; the other two (calls 23, 24) have the **majority coalition voting Kologeto / Manele** — implausible given the coalition premise.
- Manele (calls 18/25, 20/26) votes to reinstate himself in 87.5% of his mass; he is a permanent non-Wale elector.

**Intended effects / consequences:** every terminal action is a `record_vote` into `parliament` (per the single `vote_for_pm` consequence template) emitting `pm_election_result` to observers `["public"]`. There is no inter-actor message passing, no whip action, no negotiation effect that changes another actor's state — see §F.

---

## F. Consequences and interaction

- **Distinct actions:** exactly one action type, `vote_for_pm` → `record_vote`. No lobbying, whipping, coalition-agreement, or injunction action exists in the compiled world, even though several *states'* `action_if_state` describe exactly those moves (e.g. Wale-fearful "offering key posts to potential defectors and possibly delaying the parliamentary vote"; Manele-spoil "files an urgent injunction… boycott"). Those descriptions are **narrative only**; when such a state defers, the engine's `_force_terminal_vote` simply writes a ballot from the action text — it does not simulate the negotiation, and it never changes another actor's vote.
- **Observers / world-state changes:** the only emitted event is the public `pm_election_result`. No actor observes another's ballot before deciding (secret-ballot framing, `Known relationships:{}`), so **no later actor is causally affected by an earlier one**. Votes are effectively simultaneous and independent.
- **Coalition negotiation / whipping modelled?** **No.** This is the central §F finding: the simulation modelled **five isolated ballots**, not a coalition bargaining to a single nominee. The one mechanism that would push a coalition majority onto one candidate — a whip / pre-vote agreement — is absent.
- **Did the process complete (the "old failure")?** **Yes, this time it completed.** `completion_audit_manifest`: `resolved_mass 0.999999 / total 0.999999`, `unresolved_mass_by_cause {}`, `acceptance.all_ok:true`, `resolved_share 1.0`. The failure here is not premature stopping — it is that the completed model asks the wrong question (5 equal votes) and forces the unresolved 30% of member-votes into non-Wale ballots (§G).

---

## G. Deadlines and completion

- **Waits/gathers:** on the mandatory terminal (2026-05-21), of the 200 deduped node-decisions, **134 cast_vote (freely chosen), 40 defer, 26 gather_information (=66 forced)**. `completion_audit`: `engine_completion_rounds.rounds = []`, `policy = "deadline_forced_completion:reopen_then_eval"`, `max_rounds = 2`, `proven_unavoidable:false`. So **zero reopen rounds fired** — the deadline forcing happened inline within the wave (`metrics.waves = 3`), not as a separate reopen pass, and no decision was reopened by an audit round.
- **Which member-states were FORCED (deferred/gathered → `_force_terminal_vote` from `action_if_state`):** Wale/fearful, Agovaka/dark_horse, Coalition/factional_split, Coalition/pragmatic_deal, Manele/spoil. (Kologeto's states and the pro-Wale states were all freely cast.)
- **Quantified forced mass** (computed over `world_trajectories.jsonl`):
  - **30.0%** of all individual member-votes were forced.
  - **87.8%** of world mass sits in a node containing ≥1 forced member-vote.
  - **Forced votes that went to Wale: 0.0** — *every* forced ballot went to Manele, Kologeto, or Agovaka. Forcing therefore only ever suppressed Wale, never helped him.
  - Of the YES mass (0.1302), 0.071 sits in nodes that also contain a forced (non-Wale) member — i.e. Wale still cleared ≥3 despite a forced defector.
- **Invented votes?** The forced ballots are *deterministic reads of each state's own `action_if_state`*, not fabricated preferences — e.g. Coalition/pragmatic_deal's text already says "Vote for a compromise candidate aligned with Manele," so the forced "Jeremiah Manele" ballot is faithful to the state. They are "invented" only in the sense that a *deferring* actor is compelled to produce a terminal ballot; none contradicts its state text.

---

## H. Terminal outcome

**Rule actually applied:** YES iff **≥3 of the 5 modelled members vote "Matthew Wale."** Verified: labelling a member "pro-Wale" only in its single Wale-voting state and counting ≥3 reproduces **all 1458 terminals with 0 mismatches**:

```
wale-votes 0 → NO  ×288      wale-votes 3 → YES ×144
wale-votes 1 → NO  ×576      wale-votes 4 → YES ×18
wale-votes 2 → NO  ×432
```
(`terminal` counts: NO 1296 nodes, YES 162 nodes.)

**Materially distinct terminal groups.** Only **four** members can *ever* vote Wale (Manele never can), each in exactly one state:
- Wale→confident (0.375), Agovaka→wale_strong (0.3334), Kologeto→wale_as_compromise (0.375), Coalition→wale_unifying (0.3334).

YES = P(≥3 of these 4 coincide), actors independent. Closed-form (matches the sim to 4 dp):
```
exactly 0 pro-Wale : 0.1736
exactly 1 pro-Wale : 0.3819
exactly 2 pro-Wale : 0.3143   ← 87% of mass has ≤2 Wale votes → NO
exactly 3 pro-Wale : 0.1146   ┐
exactly 4 pro-Wale : 0.0156   ┘ → YES = 0.1302
```
**YES = 0.1302, NO = 0.8698.** Mass conservation: `world_trajectories` sum = 1.0002 (rounding), YES 0.130233, NO 0.870009; matches `simulation_conditional.provenance` `{yes_mass:0.1302, no_mass:0.8698}`, `resolved_mass 1.0`. `weight_sensitive:false`, `dependence_range:[0.1302,0.1302]`.

Structural consequence: because Manele is a permanent NO and Wale himself is pro only 37.5% of the time, **YES requires an improbable near-sweep of the three remaining ⅓–⅜-likely members.** The terminal never gave the 26-MP coalition the decisive weight it holds in reality.

---

## I. Forecast decomposition

From `forecast_decomposition.json` / `metrics`:
- **grounded_prior** p = 0.1667 (n=2, `counted_outcome_reference_class`, numerator 0/denominator 2, interval [0,0.4732]).
- **simulation_conditional** p = 0.1302 (`resolved_mass 1.0`, interval [0,0.4688], `yes_mass 0.1302 / no_mass 0.8698`, dependence-insensitive).
- **combined = null**; `method = "combiner_unavailable_range_only"`; **combined_interval [0.1302, 0.1667]**; `disagreement 0.0365`; `fixed_blend_used:false`, `combiner_available:false` ("no leakage-audited reliability combiner is fitted — prior and simulation reported separately").
- **headline_forecast = 0.1302**, `headline_source = "mass_weighted:completed_rollouts+grounded_prior"` (i.e. the simulation conditional is the headline; the prior only widens the reported range).
- residual_bound 0.737856; simulation_probability_bounds (residual-widened) [0.0, 0.8607]; unresolved_mass_by_cause {}; evidence_coverage 0.333; grounding_grade "exploratory"; confidence "low".
- **Prior vs simulation contribution:** both point the same way (0.167 vs 0.130). The prior is *not* the headline, but it corroborates the low simulation number, so nothing in the pipeline pushed back toward Wale. The true answer (1) lies **outside** the entire combined interval [0.130, 0.167] and outside the residual-widened upper bound only barely (0.86) — the model was confidently, structurally wrong.

---

## J. Root-cause diagnosis

**(1) Did the 50→5 collapse + threshold-majority translation misrepresent the coalition arithmetic? — YES, decisively.** The real contest is "does the ~26-of-50 coalition that just ousted Manele seat its own leader?" — an arithmetic the coalition controls. The model replaced it with "do ≥3 of 5 near-independent notables (one of whom is the ousted PM, one a rival PM aspirant) each land in their single pro-Wale state?" The 26-MP majority became one equal, independently-drawn vote backing Wale only ⅓ of the time; the mis-scaled `threshold:26` was silently reinterpreted as ≥3-of-5. This is the load-bearing error and it caps YES at 0.13.

**(2) How did the 5 members / the MP bloc vote across contexts?** Of the 15 member-states, **exactly 5 vote Wale** (one per member except Manele; wt-sum of pro-Wale states: Wale 0.375, Agovaka 0.333, Kologeto 0.375, Coalition 0.333). The remaining 10 states split to Kologeto (3), Manele (4), Agovaka (1 self + defers), and Abstain (3). The **coalition bloc backed Wale in only 1 of 3 states**; in the other two the majority voted **Kologeto or Manele**. Wale himself failed to vote for himself in 2 of 3 states (fearful, resigned). Manele never voted Wale.

**(3) Was the low prior (0/2) also a pessimistic frame? — YES, secondarily.** The 0/2 outcome class (both cases "Sogavare won instead") set prior 0.1667 and — more insidiously — seeded the `solomon_islands_coalition_instability` shared condition (rate 0.8333, same two Sogavare cases) that was **pinned to `fluid_coalitions` in 100% of world mass**, and drove the low own-election rates (0.25) that shrank each actor's pro-Wale weight. The framing "the motion's leader usually is *not* the one who becomes PM" permeated both the prior and the state weights. But note: the prior is not the headline; even a neutral prior would leave the simulation at 0.13.

**(4) First wrong assumption (ordered):**
1. **THE ROSTER COLLAPSE (primary).** Representing a 50-seat parliament as 5 equal, independent voters — with the ~26-MP majority as one of them and the ousted PM as another — is the first and decisive error. Everything downstream inherits it. Had the coalition bloc been the ~26-vote pivotal unit (or had the 5-member majority rule been "coalition bloc decides"), Wale's win becomes the modal outcome.
2. **State generation making the coalition/candidate defect (secondary, compounding).** Even within the 5-member frame, generating states in which the *candidate abstains* and the *majority coalition installs its ousted enemy* is behaviourally implausible and pushed YES below what even the broken roster required.
3. **The low 0/2 prior + fluid-coalition pin (tertiary framing).** Amplified 1 and 2 but was not independently decisive.

**Mass affected / could a faithful model reverse it?** Using the sim's own weights:
- **Counterfactual A** — fix only the candidate so Wale always votes for himself: YES rises **0.1302 → 0.2778** (still NO-modal, because the majority coalition still backs Wale only ⅓ of the time — the roster collapse dominates).
- **Counterfactual B** — make the coalition bloc pivotal (a faithful ≥26-of-50 model where the coalition's chosen candidate wins): YES = P(coalition in its Wale-backing state) = **0.3333** on the *current* (fluid-pinned) coalition weights; and if the coalition — which had already co-operated to oust Manele — is modelled as whipped/unified for its leader with realistic probability (say 0.6–0.8), YES climbs to **~0.6–0.8, i.e. reverses to a Wale-win forecast.**

So a faithful 50-seat / coalition-whip model **can and does reverse the forecast toward Wale's actual win**; the reversal is gated almost entirely on treating the majority coalition as the decisive, correlated voting unit rather than as one of five independent notables. Quantified: the entire 0.87 NO mass is an artefact of the roster collapse plus the coalition/candidate-defection states; ≥0.15 of it moves to YES on the candidate fix alone, and the majority of it moves on the roster fix.

---

### Evidence index (files quoted)
- `exp113_cache/wale/28fa62a3….json` (blueprint_response), `cc4acc07….json` (blueprint_repair_response), `7698c9cf….json` (reference_class_grounding), `91ce04e4….json` (consequence_templates) — cache RESPONSES; prompts not in-trace.
- `…-completion/llm_calls.jsonl` — 30 in-run calls (6 state_generation, 24 actor_decision); exact prompts/replies quoted in §C, §E.
- `…-completion/actor_states.jsonl`, `weight_provenance.json`, `state_recovery_manifest.json`, `decision_manifest.json`, `actor_decisions.jsonl`, `world_trajectories.jsonl`, `completion_audit_manifest.json`, `forecast_decomposition.json`, `shared_worlds.jsonl`.
- `exp113_checkpoints/741b4bed….json` — `metrics` + `simulation_result.provenance.lean_v2` (blueprint meta, obligations, shared_condition_worlds, engine_primary).
