# Lean V2 — First-Principles Consumer Execution Path: Implementation + Live Evaluation

Base: PR #129 merge commit `5733433a3d94f081b9fb3db5205753c87e14c053` on `claude/world-model-v2`
(all 12 required base items verified present before branching). Profile:
`execution_profile="lean_v2"` — **opt-in**; Lean V1 (`lean_adaptive`) remains the default;
`full_fidelity` remains permanently available. Artifacts:
`experiments/results/exp111_lean_v2_banxico.json`, tests `tests/test_lean_v2.py` (15/15 green),
scoped adjacent suites green (lean units/integration 40+7, forecast availability 14, outcome
pathway 9).

## What was built (all §1–13 of the task, none lightweight)

1. **Three-valued answerability preflight** (`preflight.py`): answerable / unanswerable /
   uncertain over the compiled world — terminal predicate, instantiated writer, producible
   inputs, YES path AND NO path (symbolic vote-lattice reachability), accepted mechanism,
   recovery pathway, not-inevitably-unresolved. `uncertain` runs one bounded deterministic
   pathway probe; static unprovability is never treated as impossibility; a genuinely
   one-sided world is recorded, never fabricated around. Unanswerable stops BEFORE any actor
   call and returns the best defensible labeled forecast (test 1: 0 actor calls, labeled
   grounded prior served).
2. **ConsumerWorldBlueprint** (`blueprint.py`): ONE structured strong call compiles the
   coherent world (resolution, thesis, boundary, actors+aliases+authority+variants,
   institutions, mechanisms, anchors, event types, triggers, action templates, terminal
   pathway, risks, assumptions, reversal-capable omissions). Nine deterministic validators
   (schema, entity identity, authority, terminal pathway, event ordering, institution rules,
   causal directness, information boundaries, outcome executability) + at most ONE targeted
   repair call fed exactly the failures. Numbers exist only in `grounded_rates` with
   verbatim evidence quotes (validator-enforced; non-verbatim rates dropped + recorded).
3. **Terminal-causal backward slice** (`slice.py`): dependency graph backward from the
   terminal predicate; BROAD relevance (action/information/communication/authority/approval/
   resource/constraint/role/response/dynamic activation); deterministic alias merging,
   single-decision-right person/institution merging, ceremonial/mention-only pruning —
   all recorded; retained-when-uncertain recorded; dynamic promotion re-adds a pruned actor
   the moment an event genuinely reaches them (implemented in the engine's deliver step).
4. **Real shared compilation** (`compile_cache.py`): dependency-hash lookup BEFORE any
   component compiles (run layer + persistent immutable layer keyed by question deps,
   as_of, evidence hash, backend, prompt/schema/compiler versions); hit/miss/store events
   recorded; a repaired/challenger model reuses every unchanged component (test 6: second
   run recompiles nothing). Parse-before-cache: unparseable compiles are failures and never
   poison the cache. Mutable state and actor decisions are never persisted.
5. **Selective additional deliberation** (`deliberation.py`): information-vs-deliberation
   limitation classified in deterministic code. Information limitation → the wait/gather
   choice STANDS, reconsideration scheduled only for genuinely new information, the same
   known absence is never re-asked (engine ledger). Deliberation limitation → ONE bounded
   reflection call (same information boundary, first decision + the specific unresolved
   tradeoff), triggers/changes/calls recorded. Cap: 1 normal + ≤1 deliberation; the staged
   pipeline (schema repair + one strong re-ask) only for malformed/invalid responses.
6. **Parameterized consequence templates** (`consequences.py`): mechanical scaffolds
   precompiled once from the blueprint action language (vote recording, message DELIVERY,
   scheduling, stage transitions, authority transfer, windows, state writes); recipient
   interpretation/persuasion/reaction stay branch-local actor decisions; genuinely novel
   actions compile once (content-keyed, failures uncached) — tests 8/9.
7. **Exact weighted world-state coalescing** (`worlds.py`): WeightedWorldNode /
   WorldStateEquivalenceKey / WeightedBranchCoalescer. The key covers every terminal-relevant
   dynamic field INCLUDING future transition law (content-addressed randomness: draws seeded
   by state/context, never particle index; independent streams carry a tag that refuses the
   merge). Merges sum weights, preserve ancestry + source-particle ids; every merge and
   split asserts incoming == outgoing mass within 1e-9; node-cap overflow becomes DISCLOSED
   truncated mass — tests 10/11/12.
8. **Causal-wave concurrency** (`engine.py`): distinct decision contexts execute in a
   bounded worker pool (single-flight, deterministic application order); sequential ==
   concurrent proven (test 13). Causally dependent steps never parallelize.
9. **Genuinely conditional challenger** (`challenger.py`): deterministic triggers only
   (material instability near threshold; verified evidence-supported alternative; disputed
   reversal-capable assumption with a verified conflict quote; conflicting interpretations
   with majority-unresolved mass; reversal-capable omission near threshold) — imagination is
   not a trigger. **Localized fork**: the challenger shares the primary decision cache
   through an actor-visible mechanical frame hash, so every context untouched by the delta
   is a zero-call cache hit (test 15: 4 primary + 1 challenger call); full separate world
   only on structural-from-start divergence. Material disagreement → both worlds reported,
   equal-weight mixture headline with disclosure.
10. **No pointless replicates**: unresolved/under-modeled/failed-terminal results never
    auto-replicate (test 14); scoreable + genuine execution uncertainty + explicit request
    may (test 15). Nothing relaunches the world automatically.
11. **Persistent immutable compile cache**: see 4 — on by default for immutable artifacts,
    never for actor decisions/worlds.
12. **Localized challenger execution**: see 9.
13. **Stage checkpoints + bounded retries** (`checkpoints.py`): every stage checkpointed;
    idempotent stages retry once; no monitors, no relaunch loops.

Plus: **ConsumerComputeBudget** (~4× liberal caps: 20 min / 120 calls / 4 models / 40
deliberations / 30 novel compiles / 4096 nodes) enforced at the ONE gateway; exhaustion
finalizes with labels + skipped-work disclosure, never 0.5. **Real tier routing** at the
gateway (STRONG_ONLY stages pinned; deterministic work never reaches an LLM — parsing,
dates, aliases, hashing, authority, feasibility, vote counting, routing, cache lookups are
plain code). **Grounded world weighting**: variant support classes map deterministically to
broad weight RANGES (well_supported 0.45–0.75, plausible 0.15–0.45, speculative 0.02–0.25),
normalized midpoints run the world, a bounded corner sweep across the ranges sets
`weight_sensitive` honestly; no LLM ever invents a precise probability.

## The single live pastcast — Banxico unanimity (exact frozen row, sealed replay)

One foreground process; guard 10 min / 80 calls (via the production budget path); no
monitors; no relaunches. First attempt failed inside the guard (25 s, 1 call): the
3600-token completion truncated mid-JSON and was cached as-if-valid — fixed
(parse-before-cache + one compact re-emit + mandatory compactness rules), failed artifact
preserved in git history, run repeated once. Final run:

| metric | full fidelity (stored) | Lean V1 (stored) | **Lean V2 (live)** |
|---|---|---|---|
| probability | 0.7294 | 0.769 | **0.2245** |
| Brier (outcome YES) | 0.0732 ✓ | 0.0534 ✓ | **0.6014 ✗** |
| status | under_modeled | unresolved | **partially_resolved** |
| probability_source | mixed: grounded prior + partial rollouts | evidence_conditioned_prior | **partial_rollouts** |
| grounding grade / confidence | exploratory | exploratory | **partially_grounded / low** |
| unresolved mass | ~1.0 (per-model) | 1.0 | **0.2653 (disclosed)** |
| uncertainty interval | [0.35, 0.83] | [0.769, 0.769] | **[0.0002, 0.4802]** |
| weight_sensitive | true | true | **false (sweep: YES never crosses 0.5)** |
| external calls | 5,709 | 230 | **16** |
| actor calls | — | — | **14 (10 distinct + 4 deliberations)** |
| unique decision contexts | — | 40 | **10 (5 members × 2 grounded variants)** |
| decision reuses | — | 303 | **160** |
| weighted nodes executed | — | — | **33 (32 leaves)** |
| consequence template hits / compiles | — | — | **128 / 0** |
| structural models | 5 | 2 | **1 (challenger correctly not triggered)** |
| tokens (in/out) | 8.98M | 501K | **12.6K / 11.2K** |
| cost | $2.25 | $0.156 | **$0.0157** |
| wall clock | 909 min | 30.1 min | **67.5 s** |

Calls by stage: structural_generation 1 (blueprint), structural_compile 1 (targeted repair
of 1 validator failure), actor_decision 14. Preflight: answerable. Challenger: no
deterministic trigger fired (recorded). All five real board members simulated individually
with distinct grounded private-state variants; deterministic 5-0 vote counting over
actor-produced votes; zero pruned voters; zero unsafe merges; zero invented numbers.

## Why the probability diverged (>0.10 from both stored values — exact causal reason)

1. **Probability-source difference (the dominant mechanism, disclosed per row).** Both
   stored numbers are PRIOR-DOMINATED readouts: Lean V1's rollouts resolved nothing
   (`evidence_conditioned_prior` 0.769); full fidelity mixed four models serving a
   0.825-family grounded reference prior with one partial rollout (0.7294). Lean V2 is the
   only run whose worlds actually RESOLVED a majority of mass mechanically: 73.5% of
   weighted worlds produced five recorded votes, and the readout follows those simulated
   votes (`partial_rollouts` 0.2245 = the resolved-mass conditional).
2. **Dissent-persistence weighting.** The blueprint grounded dissent-leaning variants for
   three members in the May-meeting dissent evidence; the support-class law weighted them
   ~50% each. In those worlds the members voted their leaning → any split kills unanimity →
   aligned-vote worlds carry little mass. Reality: the June dissenters folded and the vote
   was unanimous. The weights were the deterministic support-class midpoints (auditable,
   sensitivity-swept — the corner sweep says NO plausible weighting inside the grounded
   ranges pushes YES above 0.48, hence weight_sensitive=false honestly).
3. **No grounded unanimity base rate reached the recovery.** The blueprint extracted zero
   verbatim-quotable historical rates from this background (0 grounded_rates; none dropped),
   so the 26.5% unresolved mass had no prior to blend (with the stored arms' 0.825 prior it
   would have been 0.7347×0.2245 + 0.2653×0.825 ≈ 0.384 — still below, so this is a
   secondary factor).
4. **Honest unresolved mass, not hidden.** Two LOW-weight variants (consensus-seeker
   Espinosa, pragmatic Heath) chose to WAIT at the meeting pending unavailable information —
   preserved as their real decision; their worlds' votes are missing → 26.5% disclosed
   unresolved mass (never renormalized away, never re-asked without new information).

None of the forbidden shortcuts caused the speedup: no voter was lost (5/5 simulated), no
private states merged (10 distinct contexts), no structural alternative suppressed (trigger
evaluation recorded), unresolved mass disclosed, weights deterministic + swept, every vote an
actor decision.

## The 15 answers

1. **1–5 minutes?** Yes — **67.5 s**, inside the 1–5 min target (and the 10–35 ideal call
   band: 16 calls).
2. **Total external calls:** 16 (blueprint 1, targeted repair 1, actor decisions 10,
   deliberations 4).
3. **Genuinely necessary:** the blueprint (the world), the repair (1 real validator
   failure), and the 10 distinct member×variant decisions — the irreducible human choices.
   The 4 deliberations were bounded internal-conflict reflections (none changed an action;
   correctly triggered per the deterministic rule; cheap).
4. **Lean V1 calls eliminated:** on this question L1 spent 230 calls — the multi-call
   structural chain (recon + critic + challenger compile + per-model evidence recompile +
   conditioning) collapsed into 2; its 40 per-particle decision contexts collapsed to 10
   cohort-level contexts; its consequence compilations became 128 template executions with
   0 compile calls; its stability probe became a recorded no-op. 230 → 16 = **93% fewer
   calls, 27× faster** (and 808× faster than full fidelity).
5. **Actors removed as duplicates/non-causal:** 0 on this question (the compiled roster was
   already exactly the five voters — the slice verified rather than pruned; pruning/merging
   is proven in tests 4–5).
6. **Distinct human decisions:** 10 (5 members × 2 grounded private-state variants).
7. **Weighted world nodes executed:** 33 (root + variant lattice; 32 leaves).
8. **Particle execution removed by exact coalescing:** on THIS topology, none — every leaf
   differs in terminal-relevant private state, so the coalescer correctly refused every
   merge (0 unsafe merges). The saving came from decision-context dedup: 160 decision
   applications served by 10 calls. Coalescing's merge/split/conservation behavior is
   proven in tests 10–12; vs Lean V1's 148-particle budget for this question, the weighted
   lattice replaced ~115 particle executions outright.
9. **Fidelity changes:** none removed — real LLM decisions per member, distinct private
   states, actor-local information, event-driven time, mechanical consequences with
   actor-simulated interpretation, weighted uncertainty with disclosed unresolved mass,
   terminal integrity (deterministic 5-0 count over actor votes), forecast availability +
   grounding labels. One behavioral artifact: two low-weight variants deferred AT the
   mandatory vote (honest unresolved mass rather than forced votes).
10. **Did the probability materially change?** Yes: 0.2245 vs 0.769 / 0.7294, wrong side of
    0.5 where both incumbents were right.
11. **Why:** see the four-part causal diagnosis above — real resolved rollouts vs
    prior-dominated readouts, ~50% dissent-persistence variant weights, no verbatim
    grounded base rate in this background, honest unresolved mass. Fully explained; no
    silent drift; every difference visible in `probability_source` and the manifests.
12. **Largest safe speedup:** the one-call World Blueprint replacing the fragmented
    compile/critic/conditioning chain (~100+ calls → 2), followed by cohort-level decision
    dedup (10 calls → 160 applications) and template consequences (0 compile calls).
13. **What still prevents sub-minute:** the two serial compile calls (blueprint 24.6 s +
    repair 18.3 s = 64% of wall). A persistent-cache warm start eliminates both on re-runs;
    a smaller blueprint completion or parallel repair would cut the cold start.
14. **Should Lean V2 replace Lean V1?** **Not yet.** The architecture hit every consumer
    target with all fidelity preservations intact, but on the one paired question its
    forecast landed materially below both incumbents and on the wrong side. Before any
    switch: (a) anchor variant-persistence weights to grounded reference classes when the
    evidence carries them (the 0.825-family unanimity base rate should have entered the
    recovery blend); (b) strengthen grounded-rate extraction (paraphrase-tolerant span
    verification, not verbatim-only); (c) decide the mandatory-participation semantics for
    institutional votes generically. Then run the full five-question paired comparison.
    Lean V1 stays the default (task instruction + this evidence).
15. **Risky optimizations correctly deferred:** numeric actors; smaller models for
    consequential decisions; fuzzy state merges; merging worlds with differing
    terminal-relevant state; LLM-invented cohort/world probabilities; dropping
    private-state hypotheses to hit call targets; removing structural criticism; hardcoded
    Banxico votes; outcome exposure; generic-prior replacement; removing full fidelity;
    cross-run actor-decision reuse (still disabled by default).

## Acceptance decision

Per the task's own rule and this evidence: **do not switch any default.** Lean V1 remains
the default; `lean_v2` ships as the opt-in consumer profile with its architecture validated
(16 calls / 67.5 s / $0.016, every safety invariant held, every deviation disclosed) and its
one-question forecast divergence fully diagnosed. `full_fidelity` remains permanently
available.
