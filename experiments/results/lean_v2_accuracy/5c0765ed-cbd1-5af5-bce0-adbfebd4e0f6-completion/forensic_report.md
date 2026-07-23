# Forensic reconstruction — Apple visionOS at WWDC 2026

**qid:** `5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6`
**Question:** "Will Apple announce visionOS 27 (or a successor major version to visionOS 26) at WWDC 2026?"
**Profile:** lean_v2 (exp113) · model `deepseek-v4-flash` · seed 0
**Outcome:** 1 (YES). **Headline forecast:** 0.9583. **Final Brier:** 0.0017 (the single best-scored question in the run).

This report is a READ-ONLY reconstruction from the committed trace files. Every number is traced to a file. Where a value is not determinable from the trace, that is stated.

Trace directory: `experiments/results/lean_v2_accuracy/5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6-completion/`
Checkpoint: `experiments/results/exp113_checkpoints/5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6.json`

Call inventory (`calls_by_stage`, checkpoint): 21 external calls = structural_generation 2, reference_class_grounding 1, state_generation 1, actor_decision 16, consequence_compile 1. Blueprint `from_cache=false` (`provenance.lean_v2.blueprint.hash=529780ae0314e618`).

---

## A. Question inputs

- **Exact question** (`metrics.question`): "Will Apple announce visionOS 27 (or a successor major version to visionOS 26) at WWDC 2026?"
- **as_of:** 2026-05-07 · **horizon:** 2026-06-12 (`metrics`).
- **Resolution criterion** (from the EVIDENCE block quoted verbatim in the call-0/1/2 prompts): resolves **YES** if Apple announces a new major version of visionOS — "whether named 'visionOS 27,' 'visionOS 3,' or any other name designating a major release successor to visionOS 26" — on or after May 6, 2026 and no later than June 12, 2026. A qualifying "announcement" is EITHER (1) the new major version being "mentioned during the WWDC 2026 keynote," OR (2) "an official Apple Newsroom press release … announces the new major visionOS version." A minor point update (e.g. visionOS 26.5/26.6) "does **not** qualify."
- **Evidence supplied** (verbatim, admissible facts only): the resolution rules above; "Apple has historically announced major new versions of its operating systems at WWDC each June"; "The current version of visionOS is visionOS 26 (latest release: 26.4, March 24, 2026; latest preview: 26.5 RC, May 4, 2026)"; the year-based-naming note; "reports indicate Apple has effectively abandoned the Vision Pro platform"; the MacRumors (April 29, 2026) M5 sales-failure report; "Apple has reportedly stopped work on Vision Pro, redistributed the team to other projects (notably Siri, led by former Vision Pro chief Mike Rockwell since March 2025), cancelled the 'Vision Air' project, and shifted focus toward AI-powered smart glasses"; "Apple has not officially discontinued the Vision Pro and continues to sell the M5 model, but reportedly has no plans for a new model"; "WWDC 2026 runs June 8–12, with the keynote expected on June 8"; "Other expected announcements include iOS 27 and macOS 27."
- **Grounded prior** (`forecast_decomposition.grounded_prior`): **0.8333**, source `counted_outcome_reference_class`, **n=2**, hierarchy `broad_human_decision_class`, quantity "Apple announces a new major version of visionOS at WWDC 2026", numerator/denominator **2/2**, interval [0.5268, 1.0], `fallback_reason` "all levels sparse (total 2 usable case(s)); rate is wide by construction". The two counted cases (from `reference_class_grounding` reply, `outcome_reference_class`): WWDC 2025 → visionOS 26 (2025-06-09, outcome true); WWDC 2024 → visionOS 2 (2024-06-10, outcome true). Both are `same_institution`, both strictly before as_of.
- **What was NOT supplied:** no polls or betting-market prices; no numeric probability in the evidence; no leaked keynote script; no dated series of any measurable quantity (this becomes decisive in H/J — there is "no parseable numeric threshold"). The evidence is qualitatively conflicting: strong recurrence prior vs. strong abandonment reports.

---

## B. World compilation

### B.1 Primary structural_generation (call 0, tier=strong, 7112→10274 chars)

The prompt asks for "the MINIMAL terminal-relevant world" with a strict compactness ceiling and a fixed JSON schema (actors / institutions / mechanisms / temporal_anchors / event_types / decision_triggers / action_templates / terminal / grounded_rates / …). Key parsed blueprint:

- **causal_thesis** (verbatim): "Apple's executive leadership, particularly Tim Cook and the software engineering team, must decide whether to allocate resources to a major visionOS release given the platform's abandonment reports. The decision hinges on whether Apple views WWDC 2026 as an opportunity to signal continued commitment or to quietly phase out the platform. The terminal event is the presence or absence of a major visionOS announcement during the WWDC keynote or Newsroom."
- **resolution.options:** `["Announce major visionOS version", "No major visionOS announcement"]`.
- **Actors (3):**
  - `tim_cook` (CEO, alias "Cook"), **discretion=decisive**, authority "Final approval on major product announcements". Two variants: `committed_to_vision_pro` (evidence_basis "Apple has not officially discontinued the Vision Pro and continues to sell the M5 model", well_supported) and `pragmatic_phase_out` (evidence_basis "Apple has reportedly stopped work on Vision Pro, redistributed the team to other projects", well_supported).
  - `software_engineering_vp` (Head of visionOS dev), **discretion=advisory**. Variants `maintain_momentum` / `resource_constrained`.
  - `marketing_team` (Keynote content & Newsroom), **discretion=advisory**, authority "Draft keynote script", "Newsroom press release". Variants `highlight_vision_pro` / `downplay_vision_pro` (both evidence_basis="unstated", support="speculative").
- **Institution:** `apple_executive_team`, members `[tim_cook, software_engineering_vp]`, decision_rule `single` ("Tim Cook decides after advisory input"), procedure stage 2026-06-01.
- **Mechanisms:** `wwdc_keynote_script` and `newsroom_press_release` (both institutional, `writes_terminal:false`).
- **Temporal anchors:** 2026-06-08 "WWDC 2026 keynote" (scheduled); 2026-06-12 "End of WWDC 2026" (scheduled).
- **Action templates (2):**
  - `include_in_keynote` — actor `tim_cook`, effect `set_state {key: keynote_visionos_included, value: true}`, emits `keynote_segment`, **writes_terminal:true**, validation "Decision made before June 8."
  - `publish_newsroom` — actor `marketing_team`, effect `set_state {key: newsroom_visionos_published, value: true}`, emits `newsroom_release`, **writes_terminal:true**, validation "Approved by Tim Cook."
- **Terminal rule (verbatim):** kind `state_predicate`; `yes_when: "keynote_visionos_included == true OR newsroom_visionos_published == true"`; `no_when: "Both keynote_visionos_included == false AND newsroom_visionos_published == false"`; `written_by_action_ids: [include_in_keynote, publish_newsroom]`; `evaluation_day: 2026-06-12`.
- **grounded_rates** (in-blueprint): one entry, value_range [0.0, 0.3], basis_quote = the abandonment sentence — this in-blueprint rate was **dropped** (`provenance.lean_v2.blueprint.dropped_grounded_rates=[]` shows none carried; the counted rates come only from the separate grounding call, per the "no_label_derived_weights_invariant" rule).
- **alternative_causal_reading:** exists=true, "driven by marketing team's desire to avoid negative press, leading to no announcement", `diverges_at: marketing_team` — this is what later triggers the challenger.

**Validators / repairs / slicing** (`provenance.lean_v2`): `validation_failures_final: []`, `repair.attempted: false` (blueprint accepted clean, no repair round). **Authority slicing pruned 1 actor:** `software_engineering_vp` removed with reason "no action, trigger, target or observation channel reaches a surviving terminal pathway (evidence-mention-only / non-causal)" (`n_pruned:1`, `promotable:[software_engineering_vp]`). So although 3 actors were compiled ("3 modeled actors", `n_actors:3`), only **2 actors are terminal-relevant and simulated**: `tim_cook`, `marketing_team` (`slice.kept_actors`). `n_merged:0`, `n_edges:2`.

### B.2 Terminal-writer canonicalization + round-trip (the fix for the old bug)

This is the crux of why this run did NOT discard its resolved simulation (contrast the old YES-label mapping bug).

`readiness_manifest.terminal_canonicalization` (= `provenance.lean_v2.terminal_canonicalization`): `needed: true`, `canonical_key: "__terminal_yes__"`, rewrites:
```
include_in_keynote : keynote_visionos_included   -> __terminal_yes__
publish_newsroom   : newsroom_visionos_published -> __terminal_yes__
```
Both YES-writing actions now write ONE canonical terminal key. The **old bug**: the terminal evaluator looked for a canonical YES key, but the writers set model-specific keys (`keynote_visionos_included` / `newsroom_visionos_published`); the mismatch meant even a fully-resolved YES node evaluated as unknown/NO and the whole resolved simulation was discarded. Canonicalization removes that mismatch before the sim runs.

**Round-trip proof** (`readiness.round_trip`, `ok:true`) — a synthetic YES and NO are pushed through the *live* terminal evaluator:
```
mapping: yes_label "Announce major visionOS version" ; no_label "No major visionOS announcement"
  aliases_yes: ["1","announce major visionos version","keynote_visionos_included == true or newsroom_visionos_publi","true","yes"]
  aliases_no:  ["0","both keynote_visionos_included == false and newsroom_visiono","false","no","no major visionos announcement"]
checks:
  case known_yes -> evaluator_out {resolved:true, outcome:"YES", detail:{predicate:"__terminal_yes__"}}, recovery_p 1.0, mapping_ok true
  case known_no  -> evaluator_out {resolved:true, outcome:"NO",  detail:{predicate:"__terminal_yes__"}}, recovery_p 0.0, mapping_ok true
```
i.e. a genuine YES flows to outcome=YES→p=1.0 and a genuine NO flows to outcome=NO→p=0.0 through the same code path the sim will use. `checkpoint.round_trip_ok: true`. Because the round-trip passed, the 0.75 of mass that reaches `__terminal_yes__` in the sim is scored as YES=1 instead of being thrown away — exactly the mass the old mapping bug used to lose.

**Readiness verdict** (`readiness.verdict`): `repairable`. Checks pass EXCEPT `terminal_units` (`ok:false`, "numeric predicates need a bounded mechanism in question units") → `repairs_needed:[terminal_units → mechanism_recovery]`. One repair was actually applied: `schedule_mandatory_trigger` for `marketing_team` on 2026-06-12 (forces a terminal decision at deadline). Preflight verdict `answerable`; both `yes_path_exists` and `no_path_exists` reported true at compile time (writer `include_in_keynote` reachable at 2026-06-08).

### B.3 Challenger structural_generation (call 14, 1366→849 chars)

Triggered by the primary's `alternative_causal_reading` at `marketing_team` (`challenger_triggered:true`). The prompt supplies the primary thesis, the challenged point, and the current variant names (note the variants were already renamed by state-generation to `tim_cook: vision_pro_platform_continues,vision_pro_platform_sunset; marketing_team: visionos_27_keynote_ready,visionos_27_keynote_dropped`). It requests a **minimal delta**:
> challenger_thesis: "The marketing team's desire to avoid negative press leads to no major visionOS announcement at WWDC 2026, regardless of engineering capability."

It changed only `marketing_team`'s `visionos_27_keynote_dropped` state (beliefs "announcing visionOS 27 would generate negative press…", stance "opposed to major visionOS announcement", support "plausible"), `changed_assumption` = marketing's press-aversion overrides engineering readiness, `unchanged_note: "everything else identical"`. This is a single-actor localized perturbation, not a new world.

### B.4 consequence_compile (call 13) — a novel action compiled

During the primary wave, the `pragmatic_phase_out`/`vision_pro_platform_sunset` Tim Cook, at the forced deadline, chose an action no template represented (`cast_vote`). The engine compiled its mechanical consequence only (see §F for full prompt+reply). Result `novel_cast_vote_silent_phase_out`, `writes_terminal:false` — crucially it does **not** write `__terminal_yes__`, so it cannot by itself produce YES.

---

## C. State construction and weighting

### C.1 state_generation (call 2, one call, 3770→7438 chars)

Prompt: "Propose the genuinely DIFFERENT private realities each decisive actor could be in, as of 2026-05-07 … you do NOT say how probable they are." Actors listed: **`tim_cook` and `marketing_team` only** (software_engineering_vp already pruned). Shared conditions in play named: apple_os_naming_convention, vision_pro_platform_abandonment_reports, wwdc_annual_os_announcement_pattern. No numbers permitted.

**tim_cook — 2 states** (both `reversal_capable:true`):
- `vision_pro_platform_continues`: "believes the Vision Pro platform remains strategically important … sees a major visionOS update as necessary to signal commitment." action_if_state: "Direct marketing_team to include a visionOS 27 segment in the WWDC keynote." aligned_condition `{apple_os_naming_convention: year-based naming continues}`. supporting evidence ids: naming + wwdc pattern; contradicting: abandonment.
- `vision_pro_platform_sunset`: "accepts that Vision Pro is a failed product … prefers a quiet maintenance mode or a minor update only." action_if_state: "omit visionOS 27 from keynote; issue no visionOS press release; announce only minor 26.x updates." aligned_condition `{vision_pro_platform_abandonment_reports: reports are accurate}`.

**marketing_team — 2 states** (both `reversal_capable:true`):
- `visionos_27_keynote_ready`: "prepared a polished segment … announcing visionOS 27." action: "Present visionOS 27 segment … with demo and beta." aligned_condition `{wwdc_annual_os_announcement_pattern: pattern holds}`.
- `visionos_27_keynote_dropped`: "directed to remove any visionOS 27 content … instead issue only a minor point update (visionOS 26.6) via press release." action: "Omit visionOS 27 from keynote; publish a brief press release … announcing visionOS 26.6 with bug fixes." aligned_condition `{vision_pro_platform_abandonment_reports: reports are accurate}`.

### C.2 Weights, reference classes, residuals

- **State posteriors** (`weight_provenance.state_posteriors`): for every shared-world combo, tim_cook = `{continues:0.5, sunset:0.5}` with `unknown:0.2`; marketing = the same 0.5/0.5 with unknown 0.2. Provenance `counted_complement`: "complement of the counted matched rates (1 − 0.000) shared among 2 unmatched modeled state(s) — grounded, not a label." Because no actor-state reference class produced a *matched* count, the two states split the mass evenly (0.5/0.5) — grounded as a counted complement, never a qualitative label mapped to a number (`no_label_derived_weights_invariant.ok=true`, `label_derived_weights_found:0`).
- **Actor residual bounds:** `actor_residual_bounds = {tim_cook:0.2, marketing_team:0.2}` in weight_provenance; the state-recovery/metrics per-option residual is 0.05 (`state_recovery_manifest`: "uncovered feasible option(s) … with no counted cases — bounded per option"). `software_engineering_vp` residual 0.2 (pruned). **joint_residual_bound = 0.278** → widens the sim interval to [0.722, 1.0].
- **Reference classes** (call 1): actor-state classes `apple_ceo` (cases: Cook announcing visionOS 2 at WWDC24 and visionOS 26 at WWDC25, both `same_individual`, outcome true) and `apple_vision_pro_team` (Bloomberg dev-prep cases). These are all YES-outcome cases, so no "dissent/drop" rate was counted → hence the 0.5/0.5 complement split.
- **state_recovery_manifest:** `ok:true`, 0 recovery calls, `reversal_search.ran=false` ("every actor already holds a reversal-capable state"), both actors `under_modeled:false`.

**Assessment:** States are realistic, cover both the pro-continuity and phase-out poles for each actor, and both actors hold a reversal-capable state (so the answer is not structurally pinned). No duplicates. The one soft spot: the two "hinge" states for each actor were split 0.5/0.5 purely because the historical cases were all YES (no counter-cases existed to counted-weight the sunset/dropped states down), so the model is deliberately agnostic between commitment and phase-out at the state level — appropriate given genuinely conflicting evidence. No framing bias detected; both states are grounded in verbatim evidence quotes.

---

## D. World generation

- **Shared-condition worlds** (`weight_provenance.shared_condition_worlds`, `checkpoint.shared_conditions`): the 3 binary conditions produce 2³=8 raw combos; the engine kept the **6** highest-mass and renormalized (the two smallest — version_based × platform_abandoned × no_annual, and version × active × no_annual — were truncated). Kept, with weights:
  | sw | naming | abandonment | wwdc pattern | weight |
  |----|--------|-------------|--------------|--------|
  | sw0 | year_based | abandoned | annual | 0.489115 |
  | sw1 | year_based | active | annual | 0.163038 |
  | sw2 | version_based | abandoned | annual | 0.163038 |
  | sw3 | year_based | abandoned | no_annual | 0.097847 |
  | sw4 | version_based | active | annual | 0.054346 |
  | sw5 | year_based | active | no_annual | 0.032616 |

  These follow from independent condition rates naming 0.75 (n=1), abandonment 0.75 (n=1), wwdc pattern 0.8333 (n=2), product-then-renormalize (e.g. sw0 = .75·.75·.8333/Z = 0.489).
- **Actor-variant combos:** 2 tim_cook × 2 marketing = 4 per shared-world.
- **Total weighted terminal nodes = 6 × 4 = 24** (`world_trajectories.jsonl`, 24 rows; `checkpoint.metrics` peak nodes 24). Small precisely because there are only 2 simulated actors × 2 states each and 6 shared-worlds. `branches_merged:0`.
- **Node executions:** `weighted_nodes_executed:36` across `waves:2` (primary wave + challenger wave); the challenger re-executed the affected subset but produced no distinct terminal group, so the persisted terminal set is 24.
- **Decision reuse:** `unique_decision_contexts:7`, `decision_reuses:48`, hits 48 / misses 0 (`decision_manifest.decisions`). The 48 reuses = the primary+challenger decision trace over all nodes; the single most-reused context (tim_cook sunset `cast_vote`, hash b9c7…) fired 12 times, the rest 6 each.

**Grouping by identical structure** (`world_trajectories.jsonl`): within EVERY shared-world the 4 variant combos resolve identically:
- (continues, ready) → YES
- (continues, dropped) → YES
- (sunset, ready) → YES
- (sunset, dropped) → **state_predicate_not_mechanically_bound** (missing_mechanism)

So each shared-world contributes ¾ of its mass to YES and ¼ to missing_mechanism, independent of the shared-world identity. That structural uniformity is why the aggregate is exactly YES 0.75 / missing 0.25.

---

## E. Actor decisions (all 16 actor_decision rows)

All calls are tier=strong, model deepseek-v4-flash. The decision prompt inhabits one actor ("YOU ARE …"), gives private beliefs/goals/stances/pressures, the institutional context, the situation, "NEW OBSERVATIONS (none)", and a mechanical feasible-action menu, and requires qualitative-only JSON with a `decision{chosen_action, act_or_wait, vote_option, intended_effect}` block. Mapping to the anchor's breakdown (7 primary unique + 3 deliberations + 6 challenger contexts):

### E.1 Seven PRIMARY unique contexts

| call | actor | as_of | variant / stance | trigger | feasible | chosen | vote_option |
|------|-------|-------|------------------|---------|----------|--------|-------------|
| 3 | tim_cook | 2026-06-08 | continues / pro-announce | keynote decision | `include_in_keynote` | **include_in_keynote** (act) | include_in_keynote |
| 4 | tim_cook | 2026-06-08 | sunset / anti-announce | keynote decision | `include_in_keynote` | **wait** | "" |
| 6 | marketing_team | 2026-06-12 | ready / pro-announce | deadline | `publish_newsroom` | **publish_newsroom** (act) | publish_newsroom |
| 7 | tim_cook | 2026-06-12 | sunset / anti-announce | deadline (mandatory) | `cast_vote` | **cast_vote** (act) | phase_out_vision_pro |
| 8 | marketing_team | 2026-06-12 | dropped / anti-announce | deadline | `publish_newsroom` | **wait** | "" |
| 9 | marketing_team | 2026-06-12 | dropped / anti-announce | deadline | `publish_newsroom` | **publish_newsroom** (minor 26.6) | publish_newsroom |
| 10 | marketing_team | 2026-06-12 | ready / pro-announce | deadline | `publish_newsroom` | **publish_newsroom** (act) | publish_newsroom |

Representative verbatim intents:
- Call 3 (continues): "Reaffirm Apple's commitment to visionOS, maintain developer momentum, and preserve strategic optionality for future AR products." decision_summary: "Abandoning now would damage our reputation for long-term investment…"
- Call 4 (sunset): chose **wait** — "Avoid committing to a visionOS announcement now. I will instruct the marketing team to keep the keynote focused on iOS and macOS AI…"
- Call 6 (ready): "Announce visionOS 27 to retain developer trust and maintain consistent OS narrative, while freeing keynote time…" → writes newsroom (YES).
- Call 7 (sunset, forced): **cast_vote / phase_out_vision_pro** — "Begin silent phase-out of Vision Pro, reallocate engineering resources to AI…"; this is the novel action of §F, does not write the terminal YES key.
- Call 9 (dropped, but chose to act): publishes a **minor** 26.6 release — "framing visionOS as a routine update." Mechanically this still fires `publish_newsroom` → writes `__terminal_yes__` (the mechanism cannot distinguish "minor" from "major" once the writer fires); this is why (·, dropped) still lands YES when marketing acts.

**Note on the two marketing "dropped" contexts (calls 8 vs 9):** both are the dropped state at deadline, but they resolve differently (wait vs publish) because they sit under different upstream Tim Cook variants/hashes (dropped-under-sunset → wait, hash ae42…; dropped-under-continues → publish, hash d884…). The decision cache keys on the full context, so these are two distinct contexts, not a contradiction.

### E.2 Three DELIBERATIONS (calls 5, 11, 12)

Same-moment second-pass reflection ("You are the SAME person continuing the SAME moment of thought — no new outside information exists"). All three returned `changed:false`:
- Call 5 (tim_cook continues): KEEP include_in_keynote — "the developer ecosystem is essential to Apple's long-term positioning."
- Call 11 (marketing): KEEP publish_newsroom.
- Call 12 (marketing): KEEP publish_newsroom.
`decision_manifest.escalations:[]`, `avoided_reasks:[]` — no forced re-asks. No decision flipped on reflection.

### E.3 Six CHALLENGER-fork contexts (calls 15–20)

Run against the challenger blueprint (marketing press-aversion). All `changed:false` where deliberative:
| call | actor | variant/stance | feasible | chosen |
|------|-------|----------------|----------|--------|
| 15 | tim_cook | sunset | `include_in_keynote`, `novel_cast_vote_silent_phase_out` | **novel_cast_vote_silent_phase_out** (approve_phase_out) |
| 16 | tim_cook | continues | `include_in_keynote`, `novel_…` | **include_in_keynote** |
| 17 | tim_cook | continues (reflection) | — | include_in_keynote (keep) |
| 18 | tim_cook | sunset (reflection) | — | novel_cast_vote_silent_phase_out (keep) |
| 19 | marketing_team | opposed (challenger dropped) | `publish_newsroom` | **wait** |
| 20 | marketing_team | opposed (challenger dropped) | `publish_newsroom` | **wait** |

The challenger sharpened marketing's opposition (calls 19/20 both **wait**), but it did not change the terminal group structure: the only non-YES cell remains (sunset, dropped). Realism is high — actors reason from their private state, screen options with reasons, and the phase-out CEO invents an internal-governance act rather than a keynote inclusion.

---

## F. Consequences and interaction

**Distinct terminal-writing actions and their mechanical consequences:**
- `include_in_keynote` (tim_cook): `set_state __terminal_yes__ = true`, emits `keynote_segment` (observers public). Writes YES.
- `publish_newsroom` (marketing_team): `set_state __terminal_yes__ = true`, emits `newsroom_release` (observers public). Writes YES. Fires even for a "minor 26.6" framing (call 9) — the writer is boolean.

**Novel action — consequence_compile (call 13), full prompt+reply:**
Prompt (verbatim core): "An actor in a causal simulation chose an action no precompiled template represents. Compile its MECHANICAL consequence structure only — no persuasion outcomes, no interpretation, no social reactions… Actor: tim_cook Day: 2026-06-12 Chosen action: cast_vote Intended effect (actor's words): Begin silent phase-out of Vision Pro, reallocate engineering resources to AI features in iOS/macOS, and signal to analysts that Apple is prioritizing high-margin Services and AI."
Reply (verbatim): `{"action_id":"novel_cast_vote_silent_phase_out", … "effects":[{"kind":"record_vote","params":{"institution_id":"apple_board","options":["approve_phase_out"],"stage":"decision","key":"vote_tim_cook","value":"approve_phase_out"}},{"kind":"set_state","params":{"institution_id":"apple_engineering","key":"vision_pro_phase_out","value":"silent_start"}},{"kind":"set_state","params":{"institution_id":"apple_engineering","key":"ai_ios_macos_priority","value":"high"}}], "emits_events":[{"etype":"internal_vote_cast","observers":["apple_board"]}], "writes_terminal":false, "validation":""}`

Crucially the compiled effects write `vote_tim_cook`, `vision_pro_phase_out`, `ai_ios_macos_priority` — **none of which is `__terminal_yes__`** — and `writes_terminal:false`. So a phase-out CEO removes himself from the YES path but does not write NO either.

**Genuine interaction?** Yes, structurally. The two writers belong to two different actors with independent authority: Tim Cook (keynote) and the marketing team (Newsroom). YES is an OR over the two writers, so a node reaches YES if EITHER actor acts. Marketing's independent power to publish is what carries three of four variant-combos to YES even when Cook is in sunset (sunset+ready → YES via marketing). The only cell with no writer at all is (sunset CEO who casts phase-out + dropped marketing who waits) → neither `__terminal_yes__` writer fires. Observers are "public" for both public writers; the internal vote is observed only by `apple_board`. No later actor reconsidered on the basis of another's action (deliberations all KEEP), so the interaction is via the terminal OR, not via message-passing persuasion.

---

## G. Deadlines and completion

- **Scheduled events / deadlines:** keynote 2026-06-08 (Tim Cook keynote-inclusion decision); institution `apple_executive_team` deadline 2026-06-12 (`readiness` check `institution_deadline:apple_executive_team=2026-06-12`). One **repair applied**: `schedule_mandatory_trigger` for `marketing_team` on 2026-06-12 (`readiness.repairs_applied`) so marketing is forced to a terminal decision at the horizon rather than silently never acting.
- **Waits/forced:** the `wait` choices (calls 4, 8, 19, 20) are voluntary at the keynote step, but at the 2026-06-12 deadline the trigger is `mandatory_terminal` (Tim Cook sunset) / `deadline` (marketing) — i.e. forced completion. `completion_audit.engine_completion_rounds`: 1 round, `splits:0`, `reopened_decisions:0`, `re_evaluated:6`, policy `deadline_forced_completion:reopen_then_eval`, `max_rounds:2`. The 6 re-evaluated are exactly the six (sunset, dropped) nodes across sw0–sw5.
- **No invented choices:** every chosen action is either a listed feasible action or the single novel `cast_vote` that was properly routed through consequence_compile. No option outside the mechanical menu was silently honored.

---

## H. Terminal outcome

Two materially distinct terminal groups (`world_trajectories.jsonl`, verified by summation):

**Group 1 — resolved YES, mass 0.749997 (≈0.75).** 18 of 24 nodes. Predicate evaluation: at least one of the two writers set the canonical key, i.e. `__terminal_yes__ == true`, evaluated on 2026-06-12 → outcome YES → recovery_p 1.0 (per the round-trip mapping in B.2). These are all (continues,·) and (sunset, ready) cells.

**Group 2 — missing_mechanism, mass 0.249999 (≈0.25).** 6 nodes, all `w0_swN.tim_cook-vision_pro_platform_sunset.marketin-visionos_27_keynote_dropped`, cause `state_predicate_not_mechanically_bound`. In these nodes Tim Cook (sunset) casts the internal phase-out vote (no terminal write) and marketing (dropped) **waits** and never publishes. The YES predicate has explicit writers, but the **NO** side (`no_when: both keys == false`) has **no action that positively writes "no announcement occurred"** — the absence of a write cannot be distinguished from an un-simulated pathway, so the engine refuses to score it as a resolved NO and classes it unresolved/missing-mechanism instead of forcing it to 0.

**Why the mechanism could not be bound** (`mechanism_recovery_manifest.pre_run`): two deterministic attempts — attempt 1 `reuse_existing_mechanisms` "0 verbatim numeric(s) reused"; attempt 2 `deterministic_threshold_parse` "no numeric threshold in resolution text". `validated:false`, **failure_proof (verbatim):** "the resolution criterion carries no parseable numeric threshold — a bounded numeric mechanism is not the right bridge for this terminal." `post_run:null`. `completion_audit.proven_unavoidable:true`.

**Mass conservation:** YES 0.749997 + missing_mechanism 0.249999 = 0.999996 ≈ 1.0 (rounding of the 24 node weights). unknown-state terminal mass = 0.0 (`terminal_unknown_state_ok:true`), provider-failure mass 0.0. **Sim-only conditional = YES_mass / resolved_mass = 0.75/0.75 = 1.0** (`forecast_decomposition.simulation_conditional.p=1.0`, `yes_mass 0.75, no_mass 0.0`). Of the mass that resolved, ALL of it is YES.

---

## I. Forecast decomposition

From `forecast_decomposition.json` / `metrics`:
- **Grounded prior:** 0.8333 (counted, n=2, broad_human_decision_class, 2/2, interval [0.5268,1.0]).
- **Simulation conditional (on resolved mass):** 1.0; resolved_mass 0.75; interval [1.0,1.0]; `weight_sensitive:false`, `dependence_sensitive:false`.
- **Unresolved:** 0.25, entirely `unresolved_missing_mechanism`.
- **Residual bound:** 0.278 → simulation probability bounds widened to **[0.722, 1.0]** (this is `uncertainty_interval`).
- **Combined:** `combined:null`, `method:"combiner_unavailable_range_only"`, `combined_interval:[0.8333,1.0]`, `fixed_blend_used:false`, `combiner_available:false`. Prior↔sim disagreement 0.1667. The system explicitly declines to fit a leakage-audited reliability combiner and reports prior and simulation separately.
- **Headline:** **0.9583**, `headline_source: mass_weighted:partial_rollouts+grounded_prior`.

**Exact headline formula** (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior):
```
headline = resolved_mass · sim_conditional + unresolved_mass · grounded_prior
         = 0.75 · 1.0        + 0.25 · 0.8333
         = 0.75 + 0.208325
         = 0.958325  ≈ 0.9583
```
- **Prior contribution:** 0.25 · 0.8333 = 0.2083 of the headline.
- **Simulation contribution:** 0.75 · 1.0 = 0.75 of the headline (the dominant term).

**Scoring** (`metrics.evaluation`, outcome=1): exp113 Brier **0.0017** (p 0.9583); prior Brier 0.0278 (p 0.8333); full-fidelity Brier 0.0276 (ff_p 0.834); exp112 lean_v2 0.0156 (p 0.875); lean_v1 0.3399 (p 0.417). `sim_moved_toward_outcome:true`, `correct_side:true`. The headline beat both the prior (0.8333) and full-fidelity (0.834), and the sim-only forecast (1.0) scored Brier 0.0.

---

## J. Root-cause diagnosis — why it worked and what to preserve

**(a) The terminal-writer canonicalization + round-trip is the component that saved the resolved simulation.** The old bug lost fully-resolved YES worlds because the writers set model-invented keys (`keynote_visionos_included` / `newsroom_visionos_published`) that the terminal evaluator did not recognize as the canonical YES key. This run rewrote **both** writers to the single `__terminal_yes__` key BEFORE simulating (`terminal_canonicalization`, `canonical_key:__terminal_yes__`) and then **proved** the wiring end-to-end with a live round-trip: `known_yes → outcome YES, recovery_p 1.0, mapping_ok true` and `known_no → outcome NO, recovery_p 0.0, mapping_ok true` (`round_trip.ok:true`). With the mapping proven, the 0.75 of YES mass was scored as YES=1 instead of being discarded — that alone is the difference between this run and the historical failure.

**(b) The actors coherently reached YES** because the world was built as an OR over two independently-empowered writers (Tim Cook's keynote and the marketing team's Newsroom). Even a phase-out CEO (who waits or casts an internal phase-out vote) does not block YES if the marketing team publishes; only the doubly-negative cell (sunset CEO + dropped-and-waiting marketing) fails to write. Because the two hinge states are each split 0.5/0.5 and marketing acts in three of four combos, ¾ of every shared-world's mass lands on YES. The decisions themselves are individually plausible and grounded in verbatim evidence, and none flipped on deliberation.

**(c) The headline correctly blended** the resolved simulation (0.75 mass at conditional 1.0) with the grounded prior (0.8333) for the unresolved 0.25 via `0.75·1.0 + 0.25·0.8333 = 0.9583`. This is the right move: the resolved worlds carry hard mechanical YES evidence, while the unresolved slice defers to the historical base rate rather than being forced to 0 or 1. Since the true outcome was YES and the prior itself was high (0.8333), taking the prior on the unresolved slice kept the forecast confidently high and correct.

**(d) The residual missing_mechanism 0.25 was proven unavoidable — and a numeric bridge was genuinely NOT needed.** `mechanism_recovery` tried reuse and deterministic threshold-parse, both returned nothing, and produced `failure_proof: "the resolution criterion carries no parseable numeric threshold — a bounded numeric mechanism is not the right bridge for this terminal"`; `proven_unavoidable:true`. The question is a boolean disclosure ("did Apple announce?"), so there is no numeric series to threshold. The 0.25 is not a modeling failure of quantity but a genuine gap: the world has YES-writers but no NO-writer, so a fully-inactive world cannot be positively resolved as NO. That is the one real weakness — but it is a *structural asymmetry in how absence-of-announcement is recorded*, not a missing numeric mechanism.

**Keepers (worth generalizing):**
1. **Terminal-writer canonicalization** — collapse every YES-writing action onto one canonical terminal key at compile time.
2. **The live round-trip proof** (synthetic YES→1 / NO→0 through the real evaluator) as a *gate before* simulating; without it, resolved mass can be silently discarded.
3. **Mass-weighted blend** (resolved mass keeps its simulated conditional; unresolved mass takes the grounded prior) reported alongside — never instead of — the separate prior and sim, with `no_label_derived_weights_invariant` guaranteeing every weight traces to a counted reference class.
4. **State completeness** — both actors carry a reversal-capable state (so the answer isn't structurally pinned), grounded by counted-complement rather than label-mapping.

**The one weakness to fix generically:** the NO side of a boolean `state_predicate` terminal has no positive writer, so a fully-inactive world resolves as missing_mechanism (here 0.25) rather than NO. For disclosure-type questions the right bridge is a **deadline "no-announcement recorded" writer** (a behavioral/temporal closure at the horizon), not the numeric mechanism the recovery step correctly rejected. Adding that would have resolved the last 0.25 as NO and, in this case, likely lowered the headline slightly toward the (also-high) sim-only 1.0 versus prior blend — but the run already scored Brier 0.0017, so the gap cost almost nothing here.

---

### Determinability notes
- "3 modeled actors" refers to the compiled blueprint (`n_actors:3`); only 2 (`tim_cook`, `marketing_team`) survive authority-slicing and are simulated. `software_engineering_vp` is pruned (`promotable`).
- The anchor's 16 actor_decision = 7 primary unique (calls 3,4,6,7,8,9,10) + 3 deliberations (5,11,12) + 6 challenger contexts (15–20) is confirmed by call stage/date/schema; note `decision_manifest.unique_decision_contexts=7` counts deduped *cached* contexts, not raw calls.
- Node weights sum to 0.999996 (YES) + missing (0.25) due to rounding of the 24 persisted 6-dp weights; treat as exactly 0.75 / 0.25.
