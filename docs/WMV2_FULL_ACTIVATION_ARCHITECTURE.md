# WMV2 Full Activation — mandatory phase supervision architecture

## Root causes (from the merged PR #98 audit)

1. The runtime relied on the LLM compiler emitting exact phase-specific objects; a phase whose object was
   not emitted silently disappeared (no record, no failure, no support effect).
2. Relevance was lexical-only (process-token vocabularies), so recall was hostage to compiler phrasing
   (P6 recall 0; P9-network recall 0.22) and false execution was high where structure was over-declared.
3. Phase 6 had no resolution layer: per-process registry selections were computed, recorded in provenance
   and then dropped; an unanswered required process was silently omitted.
4. Phases 9/4 had emitters but (before PR #98) no causal consumers; activation accounting lived in a
   separate, weaker manifest not derived from what actually executed.

## The mandatory supervisor (`swm/world_model_v2/phase_supervision.py`)

Every normal simulation runs ONE supervisor pass over all eleven phases. Each phase gets a
`PhaseExecutionRecord` (relevance + reasons, input-state contract, selected mechanism/pack, StateDelta
census, state fields written, downstream events, terminal influence, latency, errors, support implication).
Statuses: `causally_active`, `no_op_causally_irrelevant` (the only normal no-op), `blocked_missing_state`,
`blocked_no_mechanism`, `blocked_invalid_contract`, `execution_failed`.

- The active-component manifest is **derived from these records** (`finalize()`); the delta census comes
  from the rollout's actual branch logs (`pipeline._operator_delta_census`) — there is no separate, weaker
  activation accounting to drift.
- A RELEVANT phase in any `blocked_*` status is an integration failure: it lowers `support_grade`, enters
  `provenance.phase_integration_failures`, sets `fully_integrated=false`, and fails the phase gate.
- A causal SIGNAL with no declared structure is `blocked_missing_state` on a relevant phase — never a
  quiet no-op.

## Phase lifecycle on the one canonical path

compile → frozen evidence (or injected replay capsule) → evidence-conditioned recompile → posterior →
rule normalization → **relevance gate** (semantic-first: the compiler's structured `causal_dependencies`
block; lexical process-token backstop) → **activation synthesis** (completes execution linkage from
declared/inferable structure; gates off ornamental execution) → **supervisor assess** → one event queue /
one StateDelta protocol / persistence-aware rollout → Phase 11 controller → **supervisor finalize** →
terminal from terminal world states → Phase-12-compatible result. No wrapper combines separate phase
pipelines; all phases read/modify one shared world lineage.

## Phase-specific fixes this run

- **Phase 6 (architectural, Part 3)**: required causal process → dependency-signal/ontology normalization →
  registry `select_for_process` → published-pack event for dispatchable families → **and when NOTHING
  answers, the `structural_process_prior` operator executes the process from a transparent broad Beta prior
  through the shared runtime** (labeled exploratory, registry gap recorded in `fallbacks_used`). A required
  process can no longer be silently omitted. If the compiler returns no process list, processes are derived
  from its dependency signals.
- **Phase 4 (Part 4)**: the no-feasible-action contract violation no longer crashes the simulation — it is
  a visible degenerate wait posterior (`no_feasible_action_all_particles`), flagged for support grading.
  Actor relevance keys on the strategic-actor dependency signal or outcome-polar candidate actions; the
  action-polarity consumer carries decisions into the terminal.
- **Phase 7 (Part 5)**: relevance from the `nonlinear_dynamics` dependency signal (semantic, not the
  literal word) + structural token backstop; executes the real `nonlinear_state_step` saturating chain.
- **Phase 9 populations (Part 6)**: `aggregate_population_behavior` signal; segment-weighted heterogeneity
  aggregation consumed by the bounded terminal modulation channel.
- **Phase 9 networks (Part 7)**: `networked_transmission` signal; when transmission is required and the
  compiler declared no edges, **relations are inferred from the declared causal world** (hub-influence +
  communication edges among declared entities; population exposure layer), then the multilayer percolation
  consumer executes. Layers have distinct transmissibility priors.
- **Phase 10 (Part 8)**: relevance requires the `institutional_decision_process` signal (an organization
  merely being mentioned does not activate it); executes the declared threshold/quorum rule over
  posterior-informed member votes, resolving the outcome quantity ahead of the generic safety net.
- **Phase 11 (Part 9)**: unchanged controller; validated by the adversarial shock corpus (recall 1.0,
  false 0.0, migration integrity 1.0) — natural-trigger validation inside full replay remains OPEN (the
  replay benchmark itself is blocked; see the validation doc).

## Compatibility / deprecations

- The compiler now emits a structured `causal_dependencies` block (stored in plan provenance); older plans
  without it fall back to the lexical vocabularies.
- `simulate_world(..., prebuilt_bundle=)` is the sealed-replay evidence injection point (recorded in the
  manifest; never silent).
- The legacy manifest fields remain but are overwritten by the record-derived manifest — the records are
  authoritative. No old phase-specific pipeline remains a production path.
