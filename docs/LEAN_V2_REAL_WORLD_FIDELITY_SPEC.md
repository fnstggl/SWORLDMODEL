# Lean V2 Real-World Fidelity — design spec (Phase A)

**Branch:** `claude/lean-v2-real-world-fidelity` (stacked on `claude/lean-v2-simulation-completion`,
base commit `107b28899ff20a4a660c3be483fad3967aa4a2a0`).

**Goal.** Turn Lean V2 from *a mechanically complete simulation of an often distorted generated
world* into *a lean, auditable simulation whose actors, evidence, institutions, interactions,
mechanisms, units and terminal rules faithfully represent the real decision process.*

**Universality is a hard requirement.** Every mechanism here is general world-model machinery.
There is **no** question-specific logic, no benchmark IDs in production code, no special case for
Banxico / BoJ / Apple / Wale / Hormuz, no outcome-dependent thresholds, no hardcoded consensus,
no arbitrary social probabilities. The five EXP-113 questions are only *witnesses* of general
failure classes; the fixes must generalize to any institutional / event / numeric-threshold
decision. The canonical entry point stays `unified_runtime.simulate_world(..., execution_profile="lean_v2")`.

**Not in scope this task:** calibrating a prior↔simulation combiner; tuning any parameter against
the known BTF-3 outcomes; switching the default profile. The task ends after: fix architecture →
freeze → rerun the same five simulation-only pastcasts once → measure honestly.

## The typed pipeline every production module must serve

```
Canonical evidence
→ Exact resolution specification (ResolutionSpec)
→ Faithful real-world representation (WorldRepresentationSpec)
→ Shared-world uncertainty (SharedConditionGraph)
→ Actor knowledge + behavioral hypotheses (ActorActionBaseline, ActorStateHypothesis)
→ Initial actor positions
→ Interaction and deliberation (institutional decision process)
→ Final actor actions
→ Mechanical consequences (OutcomeMechanismSpec)
→ Exact measured outcome
→ Weighted simulation-only probability
```

## Preserved components (fix semantics, do not delete)

Canonical `simulate_world` routing; `execution_profile="lean_v2"`; actor-state completeness + the
bounded recovery ladder; no terminal `unknown_state` worlds; mass conservation; actor-local
information boundaries; real LLM actor decisions; deterministic mechanical consequences;
run-scoped decision caching; no cached failures; single-flight calls; event-driven time; terminal
round-trip validation; separate prior/simulation reporting; unresolved-mass reporting;
missing-mechanism diagnosis; no silent prior fallback; exact prompt/reply gateway tracing.

## Defect → fix map (each row is a milestone with tests)

### D1. Vote/option formatting drops valid votes (BoJ Ueda `vote:Raise to 1.0%` → dropped)
- **Root cause:** substring/exact option matching in `engine._apply_decision` /
  `readiness.pure_terminal_outcome` never strips the `vote:` menu prefix or normalizes case/punct.
- **Module:** new `canonical_options.py` (typed `CanonicalOption {canonical_option_id, display_label,
  aliases, institution_id, terminal_semantics}` + `normalize_option(raw, options)`); wired into
  engine vote recording, terminal law, obligations menu.
- **Invariant:** every provider option string is normalized to a `canonical_option_id` before any
  vote/terminal comparison; an unnormalizable option fails validation → one targeted repair → never
  silently becomes another option.
- **Tests:** 1,2 (and 49,50,52).
- **Acceptance:** no valid vote lost to formatting; unknown option never silently maps.

### D2. Historical-state direction inversion (BoJ: "dissented for a hike" class pinned to a hold state)
- **Root cause:** `states.ActorStatePosteriorEngine.weight_actor_states` matches counted classes to
  states by lexical token overlap, so a pro-hike class binds to an anti-hike state.
- **Module:** `states.py` — states declare `expected_action_tendency` (a canonical_option_id / class);
  reference classes declare `action_option_id`; add a hard validator
  `reference_class.action_option_id must be action-compatible with state.expected_action_tendency`;
  on conflict reject the match, record it, assign no rate, never reverse the evidence.
- **Tests:** 3,4.
- **Acceptance:** a hike class cannot weight a hold state and vice-versa.

### D3. Arbitrary first/alphabetical action fallback (`sorted(allowed)[0]`, forced votes)
- **Root cause:** `engine._force_terminal_vote` and `_apply_decision` fall back to
  `sorted(allowed_opts)[0]`; the deadline path invents votes to raise resolved mass.
- **Module:** `engine.py` — delete every first/lexicographic/fixed-default path; replace the deadline
  fallback with the §12 ladder (validated normalization → one repair → explicit binding
  precommitment → procedural absence/abstention only when the real rule allows → else labeled
  actor-decision-failure mass with bounds).
- **Tests:** 5,47–52.
- **Acceptance:** no terminal decision invented to increase resolved mass; provenance recorded.

### D4. Terminal canonicalization collapses numeric predicates (Hormuz `count>=50` → boolean OR)
- **Root cause:** `readiness.canonicalize_terminal_writers` rewrites *any* `set_state` writer to the
  boolean `__terminal_yes__`.
- **Module:** `readiness.py` — typed terminal kinds `{BOOLEAN_EVENT, INSTITUTION_VOTE,
  NUMERIC_THRESHOLD, CATEGORICAL_STATE, FIRST_PASSAGE, DEADLINE_ABSENCE}`; boolean canonicalization
  runs only on `BOOLEAN_EVENT`; a transform must preserve kind/units/variable/comparator/threshold/
  window; verify identical-before-and-after.
- **Tests:** 7,8,55,56.
- **Acceptance:** numeric terminals stay numeric; boolean event canonicalization still works.

### D5. Numeric-threshold parsing gap (Hormuz "50" not extracted)
- **Root cause:** `mechanisms.diagnose_missing_mechanism` regex misses many phrasings; the blueprint
  LLM terminal can silently diverge from the frozen resolution text.
- **Module:** new `resolution_spec.py` — deterministic `ResolutionSpec` parser producing
  `{measured_variable, unit, comparator, threshold, aggregation_window, observation_window,
  resolution_deadline, yes_condition, no_condition}`; parses "at least/more than/fewer than/N or
  more/≥N/majority of N/unanimous N-of-N/at least N votes/by DATE/on any single day/cumulative".
  The frozen resolution criterion stays source of truth; if parser and blueprint disagree, readiness
  fails → repair. LLM blueprint may never overwrite the resolution.
- **Tests:** 6,10,57.
- **Acceptance:** all listed forms parse; parser/blueprint disagreement fails readiness.

### D6. No event-absence writer (visionOS: fully-inactive world can't resolve NO)
- **Root cause:** boolean terminals have YES-writers but no positive NO-writer, so event-NO worlds
  stay unresolved as `missing_mechanism`.
- **Module:** `engine.py`/`readiness.py` — at the evaluation deadline, deterministic code writes
  `event_absent` (→ terminal NO) for `BOOLEAN_EVENT`/`DEADLINE_ABSENCE` when no qualifying event was
  recorded. This is the mechanical complement, not a missing mechanism.
- **Tests:** 9,77.
- **Acceptance:** deadline event-absence resolves NO.

### D7. Institution roster collapse + threshold rescaling (BoJ 9→5 bloc; Wale 50-seat parliament = 5 candidates; `5-of-9`→`≥3-of-5`, `26-of-50`→`≥3-of-5`)
- **Root cause:** the blueprint compiler collapses large bodies to ≤N actors; `readiness`
  translates a mis-scaled absolute threshold to a majority-of-modeled instead of failing.
- **Module:** new `representation.py` — typed `WorldRepresentationSpec {real actors, represented
  decision units, institutions, membership, roles (candidate/voter/adviser/observer), authority,
  multiplicity, voting_power, seat_counts, quorum, threshold, decision_stages, resolution_units}`;
  remove any decisive-actor cap; **small bodies simulate every member**; **large groups are typed
  blocs that emit a distribution/count of member actions weighted by seat_count**, never one
  ordinary vote; candidates ≠ electorate; a representation validator reconciles `real voting power
  == represented voting power == terminal threshold` exactly or **fails readiness and repairs the
  roster** — the threshold is never rescaled to fit an omitted roster.
- **Tests:** 19–27,72–75.
- **Acceptance:** every acceptance-criteria roster bullet in §23.
- **Status: IMPLEMENTED.** `lean_v2/representation.py` (typed `WorldRepresentationSpec`,
  individual vs seat-weighted bloc `DecisionUnit`, candidate≠voter typing, `validate_representation`
  reconciling real==represented==threshold, `repair_representation` expanding the roster, no
  decisive-actor cap). The threshold-rescaling branch in `readiness.pure_terminal_outcome` is
  DELETED (now defers with `representation_incomplete:threshold_exceeds_modeled_roster`). Wired into
  the runtime via `institution_terminal.py`. Tests: `test_lean_v2_representation.py` (14),
  `test_lean_v2_institution_terminal.py` (5).

### D8. State-count drives probability + auto equal split (anti-consensus foundation)
- **Root cause:** `states`/engine give each generated variant ~1/N weight and branch actors
  near-independently, so more prose variants = more probability and consensus collapses.
- **Module:** new `action_baseline.py` — `ActorActionBaseline`: a count-based (partial-pooling)
  distribution over feasible **action classes** per actor/decision using the specificity hierarchy
  (same person/decision → role/institution → similar → broad); sparse ⇒ wide interval. `states.py`
  attaches private states *beneath* an action tendency: allocate mass to action tendencies first,
  then split *within* a tendency only for trajectory/sensitivity. Delete auto equal-split;
  duplicating/paraphrasing/splitting a state must not change the action probability. Typed
  state↔condition alignment (`condition_id, condition_state_id, direction`), validated.
- **Tests:** 28–35.
- **Acceptance:** state count cannot change probability; no auto equal split; typed alignment.
- **Status: IMPLEMENTED.** `lean_v2/action_baseline.py` (`ActorActionBaseline`,
  `partial_pool_categorical` — hierarchical Dirichlet-multinomial with the global Jeffreys prior
  separated from the cross-level shrinkage `tau`, so a single counted level reproduces its
  beta-binomial rate; disclosed uniform over *classes* when uncounted). `states.py
  weight_actor_states` now allocates mass to action tendencies first via
  `_allocate_by_action_class` (typed `action_option_id` seed, raw counts, world-conditional) and
  splits within a tendency — the `residual/len(states)` equal split is gone; the bounded
  completeness residual is preserved. Tests: `test_lean_v2_action_baseline.py` (18 incl. story-count
  invariance).

### D9. Fabricated external facts inside private states
- **Root cause:** `ActorStateHypothesis` mixes latent mindset with invented external events (secret
  memos, private threats), passed to actors as established reality.
- **Module:** `states.py` — split `ActorStateHypothesis` into `{latent_beliefs, latent_goals,
  latent_preferences, latent_risk_tolerance, known_commitments, evidence_supported_observations,
  hypothetical_assumptions, expected_action_tendency}`; unsupported external events become
  shared-world hypotheses with evidence/provenance or are labeled simulated-possibility, never known
  fact.
- **Tests:** 18.
- **Acceptance:** unsupported external events never presented as known.

### D10. Verified reference cases + separated prior/behavior layers
- **Module:** `grounding.py` — `VerifiedReferenceCase {case_id, source, source_available,
  source_quote, quote_verified, date, date_verified, actor_or_role, decision_type, observed_action,
  outcome, inclusion/exclusion_reason, as_of_valid}`; a case counts only when source is available in
  the permitted evidence system, predates as_of, quote+date+action/outcome match, inclusion holds;
  reject placeholder URLs / unverifiable quotes / LLM-invented claims / mismatched actor-action /
  vague-thematic. Keep three separate concepts: `OutcomeReferenceClass`, `ActorActionBaseline`,
  `ActorStateHypothesis` — never use outcome history as private-state evidence.
- **Tests:** 16,17,33.
- **Acceptance:** every numerical case verified; layers separated.

### D11. Evidence truncation loses decisive facts; actors get hashes not facts
- **Module:** new `evidence_store.py` — `CanonicalFact {fact_id, content, date, sources,
  source_quotes, credibility, visibility, actor_access, institution_access, contradiction_group,
  numeric_values, units, as_of_validity, terminal_relevance, decision_relevance}`; remove global
  char truncations; select facts per call by typed relevance (blueprint/grounding/state/decision
  packets). Actor prompts render real fact content, not a hash.
- **Tests:** 11–15.
- **Acceptance:** decisive facts survive; actors receive fact content.

### D12. Shared uncertainty independence + tail pruning
- **Module:** `states`/new `shared_conditions.py` — `SharedConditionGraph` (shared causes,
  conditional deps, mutually-exclusive, regimes, common info, correlation); sample shared conditions
  first, generate behavior conditional on them; if dependence unidentified, carry a small set of
  plausible structures + report sensitivity. Pruning merges equivalent worlds, preserves discarded
  tail mass and bounds its terminal effect; expand the tail when it could reverse the answer.
- **Tests:** 34,35 (+ dependence assertions).
- **Acceptance:** no independent multiplication under a common cause; tail preserved.

### D13. Actor knowledge packets (real facts, real messages, real institution state)
- **Module:** `engine.py` — `ActorKnowledgePacket` renders identity/role/authority/private mindset/
  selected shared conditions/canonical public facts/role-private facts/institution state/proposal/
  stage/deadline/actions/messages/commitments/relationships/visible positions/resources/credibility/
  contradictions. Never expose another actor's private state / future / post-as_of / secret ballots.
- **Tests:** 14,36,43,44,45.
- **Acceptance:** actors reason from the real researched facts, not labels/hashes.

### D14. Real interaction + deliberative convergence (the dominant defect)
- **Module:** new `deliberation.py` engine — typed institutional process `initial positions →
  proposal → substantive messages (`InteractionMessage`) → preliminary commitments → negotiation →
  revised positions → final proposal → final decision`. Convergence arises from leadership
  authority / norms / procedure / relationships / coalition incentives / communication costs /
  visible tallies / reference classes — **never a fixed numeric consensus bonus**, and different
  institution types never share one convergence rule. Actors reconsider only on material change;
  bounded rounds; call only changed contexts.
- **Tests:** 36–45,72.
- **Acceptance:** consensus neither forced nor suppressed; message content delivered.
- **Status: IMPLEMENTED** (in `institution_deliberation.py`, not `deliberation.py` — that name was
  already the actor-reflection module). Typed archetypes (`consensus`/`coalition`/`independent`/
  `hierarchical`) converge by DIFFERENT grounded forces; `classify_institution` derives every force
  from counts or typed structure (reference settling rate, consensus norm, leadership authority,
  coalition discipline, procedure), and an ungrounded force is 0.0 → independent baseline (never
  invented convergence). `run_institution_deliberation` is a bounded-round mean-field process with a
  material-change gate; the fixed point is set by grounded pull weights, not the step size, so it can
  never be a fixed consensus bonus. `seat_weighted_yes_prob` is an exact convolution against the REAL
  absolute threshold (blocs are Binomial). The calibrated consensus mixture
  `(1-w)·independent + w·collective_lean` lets a unanimity/supermajority body reach a high threshold
  independent voting almost never would, without over-sharpening. Composed end-to-end in
  `institution_terminal.py` and wired into the runtime as the authoritative institution-vote
  forecast. Tests: `test_lean_v2_institution_deliberation.py` (15), `test_lean_v2_institution_terminal.py` (5).

### D15. Conservative decision caching
- **Module:** `lean_context`/`engine` — `DecisionRelevantContext` includes every material field
  (actor, private-state id, canonical fact ids+credibility, selected shared-condition values,
  proposal, stage, rule, observable tally, substantive messages, commitments, relationships,
  deadline, authority, feasible actions, targets, resources, invalidation, model/prompt versions);
  any material change misses; only UUID/ordering/duplicate-wording/unrelated objects excluded.
- **Tests:** 37–39,59–64.
- **Acceptance:** false cache hit rejected; final-decision context never stripped for hit-rate.

### D16. Dimensional outcome mechanisms
- **Module:** `mechanisms.py` — `OutcomeMechanismSpec {inputs+units, transitions, output+unit,
  aggregation, window, comparator, threshold, evidence, assumptions, uncertainty}`; backward
  dependency graph terminal←variable←outputs←inputs←actions←authorities←observations (every link
  exists); hard dimensional validator (votes→count→votes-threshold; tankers/day→daily count→
  tankers/day threshold; event→before-deadline). Hybrid: actors decide behavioral inputs,
  deterministic code computes counts/tallies/traffic. Bounded ranges propagate, never a qualitative
  boolean.
- **Tests:** 53–58.
- **Acceptance:** every mechanism produces the exact required variable+unit; units match.

### D17. Readiness = structurally faithful (not merely executable)
- **Module:** `readiness.py` — extend gate with resolution/institution/evidence/behavior/outcome
  checks (§14). `ready|repairable|not_ready`; never repair an invalid institution by changing the
  real threshold.
- **Tests:** 10,26,57,72–77.

### D18. Self-contained traces + persistent-cache provenance
- **Module:** `gateway`/`traces`/`compile_cache` — per call store model/tier/exact prompt+reply/
  lengths/truncation flag/cache source+key/source run+call id/parsed/validation/repair/downstream;
  persistent cached artifacts retain original prompt+reply+source run+versions+hashes+validation;
  decision templates keep context/reply/reuse recipients/branch count/mass; full trajectory trace
  uncapped (a separate human sample may cap at 200).
- **Tests:** 65–71.

## Efficiency invariants kept (§16)

One call per genuinely distinct decision context; exact deterministic reuse; single-flight;
deterministic consequences; coalescing only when terminal-relevant state identical; targeted
repairs; no polling / no eventless ticks / no broad re-research / no actor call for pure arithmetic;
bounded rounds; no rerun of unaffected branches. Speed is never bought by deleting voters, grouping
unlike actors, dropping evidence, hiding conditions/proposal/tally, rescaling terminals, forcing
actions, collapsing numerics to booleans, or unsafe cache hits.

## Execution plan (Phase B commit order) & status

1. typed resolution + action semantics (D1,D4,D5,D6) — **direct correctness first**
2. reference-class direction + no-alphabetical-fallback (D2,D3)
3. canonical evidence + verified reference cases (D10,D11)
4. faithful institution representation (D7)
5. action-grounded state weighting (D8,D9,D12)
6. actor knowledge packets (D13)
7. substantive interaction + deliberation (D14)
8. deadline decision integrity (D3/§12)
9. dimensional outcome mechanisms (D16)
10. conservative decision caching (D15)
11. complete trace provenance (D18)
12. focused tests + docs; then Phase C structural dry-audit, Phase D freeze, §19 rerun.

Phase C: compile the five frozen rows with a **mock/deterministic backend** and validate structure
(actors, institution size, voting/seat power, roles, terminal variable/units/threshold, evidence
coverage, packets, action-mass weighting, interaction stages, outcome mechanism) **without inspecting
outcomes**. Phase D: freeze commit, clean tree. §19: cold-cache, sequential, one question at a time,
freeze/commit/push each, 15 min / 150 call hard guard, no outcome or prior-probability in any prompt.
§20–22: freeze → join outcomes for scoring only → measurement table + per-question under-the-hood
reports. §23 acceptance is the gate; do not claim fidelity solved from five questions, do not switch
default, do not auto-merge.
