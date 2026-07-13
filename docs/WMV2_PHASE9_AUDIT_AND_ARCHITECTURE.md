# WMv2 Phase 9 — Audit & Architecture (Production Population & Multilayer-Network Inference)

*Phase 9 infers scenario-specific POPULATIONS (compositional segment-weight posteriors + correlated traits)
and MULTILAYER NETWORKS (typed edge-existence posteriors + communities + structural graph hypotheses) through
the **completed Phase-3 posterior engine**, materializes posterior-weighted WorldState particles, and lets
typed multilayer mechanisms CONSUME them so graph + population uncertainty propagates into terminal outcomes.
It reuses Phase 3 — it does NOT build a second posterior system. The Phase-1 no-abstention contract is
preserved.*

---

## Part 0 — Audit of the current population & network path (before Phase 9)

### Real production call path traced
`compile_world` (LLM proposes `populations` + `relations`) → `materialize.build_world` (creates
`population.Population` from segment scalars + `network.RelationGraph` by adding LLM-proposed edges directly) →
`WorldState.populations` / `WorldState.network` → consumed by `relationship_update` mechanism +
`actor_cognition` (trust).

### Capability classification (file:symbol → status)

| capability | file:symbol | status |
|---|---|---|
| Population segments + weights | `population.py:Population,PopulationSegment` | **executable but TOY** — weights are independent scalars normalized post-hoc; traits sampled INDEPENDENTLY per field; no inference |
| Population particle sampling | `population.py:sample_particles` | **executable but ORPHANED** — not used on the production rollout path |
| Relation graph + typed edges | `network.py:RelationGraph,RelationEdge` | **executable but TOY** — edges added from the LLM proposal; strength a static dist; no existence/type posterior |
| Relation type registry | `network.py:register_relation` (11 types) | executable |
| Population INFERENCE (survey weighting / MRP / poststrat / nonresponse) | — | **ABSENT** |
| Compositional (simplex) weight posterior | — | **ABSENT** — weights were independent scalars |
| Correlated traits | — | **ABSENT** — `sf.sample()` per field independently |
| Edge existence / type / direction posterior | — | **ABSENT** — LLM-minted edges materialized directly |
| Communities / stochastic block model | — | **ABSENT** |
| Graph structural hypotheses (likelihood-updated) | — | **ABSENT** |
| Missing-edge / missing-node posterior | — | **ABSENT** |
| Multilayer separation (distinct causal semantics per layer) | partial (`uses` tags) | **TOY** — one edge list, no per-layer mechanisms |
| Actor-specific graph visibility | `RelationEdge.visibility` field | **field only** — not enforced in a mechanism |
| Graph/population → Phase-3 posterior | — | **ABSENT** — no population/graph latent ever entered `infer_posterior` |
| Real-data population/graph validation | — | **ABSENT** |

**Anti-patterns present before Phase 9:** LLM-minted edges; independent scalar segment weights; independent
demographic traits; a graph that is materialized but whose uncertainty never enters a posterior. Exactly the
list the Phase-9 spec forbids.

### Phase-3 path traced (the engine Phase 9 must reuse)
`phase3_latent_spec` (typed latents, anti-ornamental `.measurable()`) · `phase3_representation`
(representation choice + `assert_not_ornamental`) · `phase3_priors` (reference-class + transport inflation) ·
`phase3_observation` (typed likelihoods + dependence collapse) · `phase3_posterior` (`infer_posterior` particle
filter, ESS/resample, structural posterior, assimilation ledger) · `phase3_pipeline`
(`simulate_with_posterior`, posterior injected into execution). Phase 3 represents: continuous latent (particle
set), discrete structure (log-space structural posterior), priors (Beta/Dirichlet-adjacent), dependence groups,
posterior particles, provenance, mechanism consumers.

**Extension points identified:** Phase 3's outcome-rate posterior is a scalar; a segment-weight VECTOR and an
edge-existence GRAPH are genuinely new representations. So Phase 9 EXTENDS the Phase-3 engine generically
(same primitives, new representations), leaving the validated outcome-rate/structural path untouched.

---

## Part 1 — Acceptance criteria (written before implementation)
1. Population weights are a COMPOSITIONAL posterior (simplex, sum-to-one per particle), not independent
   scalars; traits are segment-conditional (correlated). Both are Phase-3 latents.
2. Every production edge carries a Phase-3 EXISTENCE posterior (log-odds from typed observation models), never
   an LLM-minted probability; ≥10 relation layers with distinct semantics.
3. Communities via a real SBM (membership posterior + block matrix); graph macro-structure via a
   likelihood-updated structural posterior; missing edges keep an explicit posterior.
4. Posterior population + graph particles MATERIALIZE into worlds and are CONSUMED by typed multilayer
   mechanisms producing StateDeltas; different particles → different terminals (anti-ornamental).
5. Actor-specific views (no omniscient leakage); actions gate on relations with reason codes.
6. Real-data validation on ≥1 real population dataset and ≥1 real graph dataset; synthetic posterior recovery.
7. No-abstention preserved; four-status grading; nothing manually supplied by a benchmark adapter.

---

## Part 2 — As-built architecture (this run)

### Modules delivered
| module | role | plane |
|---|---|---|
| `phase3_observation.py` (extended) | 16 typed NETWORK observation models across 14 relation layers + `EdgeObservation` + `collapse_edge_observations` (network dependence) | EVIDENCE→POSTERIOR |
| `phase3_posterior.py` (extended) | `infer_compositional_posterior` (Dirichlet-multinomial conjugate simplex) + `infer_edge_posterior` (Bernoulli/log-odds existence) — reuse ESS/resample/dependence/ledger | POSTERIOR |
| `phase3_representation.py` (extended) | Phase-9 representation kinds (compositional_simplex, bernoulli_edge, stochastic_block, …) | CODE |
| `phase9_population.py` | typed `PopulationSpec`, compositional weight posterior, segment-conditional Beta-binomial rates, poststratification, posterior-particle materialization, Phase-3 latent specs | POSTERIOR→WORLD-STATE |
| `phase9_network.py` | `MultilayerNetwork`/`NetworkEdge`, `infer_network_edges`, missing-edge posterior, EM `StochasticBlockModel` + `infer_communities`, `graph_structural_posterior`, actor-visibility views | POSTERIOR→WORLD-STATE |
| `phase9_execution.py` | `materialize_worlds` + typed multilayer mechanisms (communication/authority/trust/influence-diffusion) → `Phase9Delta` StateDeltas → terminal distribution | WORLD-STATE→EXECUTION |
| `phase9_pipeline.py` | `simulate_populations_networks`: discover→infer→materialize→execute→result (grade, decomposition, provenance) with no-abstention | all |

### The five planes — AS WIRED
1. **CODE** — the modules above (dependency-free; the SBM, Dirichlet posterior, log-odds engine hand-rolled).
2. **EVIDENCE** — typed survey count observations + typed `EdgeObservation`s carrying source reliability +
   dependence groups (crossing: `tag`→likelihood; `collapse_*` de-duplicates syndicated reports).
3. **POSTERIOR** — compositional segment-weight posterior (exact conjugate Dirichlet), per-edge existence
   posterior (log-odds), SBM community posterior, graph structural posterior — **all numeric, LLM-free**.
4. **WORLD-STATE** — `materialize_worlds` draws posterior-weighted particles (composition × sampled graph);
   different particles are different worlds.
5. **EXECUTION** — typed multilayer mechanisms consume the particles (`influence_diffusion`, `authority_gate`,
   `communication_delivery`) and emit `Phase9Delta`s; the terminal distribution carries graph + population
   uncertainty. Proof: `no_graph_consumed` terminal 0.017 vs `full_posterior_graph` 0.474 (real congress).

### Representation choices (representation-choice principle honored)
- Segment weights → **compositional_simplex** (Dirichlet), NOT independent scalars — validated on GSS
  (poststratification), and the compositional posterior is normalized per particle by construction.
- Edge existence → **bernoulli_edge** (log-odds) — the right representation for a binary relation.
- Communities → **stochastic_block** — mesoscale structure, not a `community_strength=0.6` scalar.
- Graph macro-structure → **discrete structural hypotheses** (Phase-3 structural posterior), likelihood-updated.
- Segment-conditional trait rates keep the segment↔trait CORRELATION a single marginal throws away.
`assert_not_ornamental` + `.measurable()` reject any latent that is neither evidence-linked nor causally
consumed.

### LLM contract (Part H) — as enforced
The LLM may propose segments, nodes, candidate edges, relation types, community/authority/coordination
hypotheses, and graph structural hypotheses. It may NOT mint segment weights, edge probabilities/strengths,
community memberships, or terminal probabilities — every number comes from a prior × likelihood update in
`phase3_posterior`. (These modules take typed evidence as input; the LLM-discovery front-end is the documented
integration boundary — see `WMV2_PHASE9_LIMITATIONS_AND_DEPENDENCIES.md`.)

Validation, ablations, gates, and the four-status grading are in `WMV2_PHASE9_VALIDATION.md`; the full
evidence→posterior→execution trace is in `WMV2_PHASE9_FORENSIC_TRACES.md`.

---

## Part 3 — Completion run: corrected audit + universal discovery architecture

*This section supersedes the "production" framing of the first Phase-9 run where it overclaimed. The prior run
built a real posterior + execution BACKEND but the universal automatic path was unwired.*

### Corrected audit of the first run (prepared-input findings)
The first run's `simulate_populations_networks(...)` and its "forensic trace" required the CALLER to supply the
entire model structure. Honest labels:

| element | first-run source | corrected status |
|---|---|---|
| `segments` (dem/rep) | supplied by the harness | **prepared input** — not discovered |
| `candidate_edges` | built from congress co-voting by the harness | **prepared input** |
| `structural_hypotheses` | hand-listed in the harness | **prepared input** |
| `segment_susceptibility` | hand-set in the harness | **prepared input** |
| `seeds`, `contagion` | hand-set in the harness | **prepared input** |
| the "non-scripted production" Senate trace | model structure supplied by the harness | **MISLABELED** — it was a mechanism-isolation FIXTURE, not a production trace |
| congress link-prediction AUROC 0.999 | co-voting signal defined AND predicted the edge | **RECONSTRUCTION**, not general hidden-edge prediction |

These are corrected in `WMV2_PHASE9_VALIDATION.md` and `WMV2_PHASE9_FORENSIC_TRACES.md`.

### Universal discovery architecture (this run)
`phase9_discovery.py`:
- `discover(question, plan, bundle, *, llm)` → `Phase9DiscoveryPlan`: population relevance, segmentation
  dimension + segments, representation choice, actors, institutions, relation layers, candidate edges (with an
  edge→claim map), structural hypotheses, seeds. A deterministic **plan-derived heuristic fallback** runs with
  no LLM; the LLM (when present) **augments** it with semantic proposals only — it mints no numbers.
- `construct_observations(discovery, bundle)` → typed `(survey_observations, edge_observations)` built from
  Phase-2 CLAIMS. A claim becomes an edge only via a typed observation model (`_edge_class_for_claim` maps
  predicate/class → layer + evidence class); reliability comes from source-type tables; dependence groups carry
  through so syndicated copies collapse.

`phase9_pipeline.simulate_with_populations_networks(question, *, as_of, horizon, ...)` — the **universal
production entry**: caller supplies ONLY question + as-of + horizon (+ optional user facts). Internally it
compiles → gathers Phase-2 evidence → **discovers** → constructs observations → infers all posteriors
(compositional + edge + community + structural) → materializes → executes. No caller-supplied
segments/edges/hypotheses/susceptibility/seeds. Records discovery/plan/evidence hashes. No-abstention preserved.

**Proof it works (live):** "Will the UN Security Council agree on a resolution?" → auto-discovered the P5 +
non-permanent members, permanent/non-permanent segments, communication/alliance/influence layers, realistic
alliance edges, bloc-polarization vs swing-vote hypotheses, and seeds — terminal 0.40 ± 0.06, 5 StateDeltas,
provenance hashed. Across **14 domains** (discovery eval): 14/14 completed, 100% discovery success, 100%
structure-reaches-execution, zero abstention.

### Informative absence + temporal evolution (this run)
- `phase3_posterior.infer_edge_posterior_exposure` (Part 4): a Binomial detection-under-exposure model where
  non-observations are informative (many opportunities + few records → edge unlikely; zero opportunities →
  uninformative). Calibrated under variable exposure (ECE ≤ 0.06), fixing the present-only overconfidence.
- `phase9_temporal.py` (Part 7): 8 typed transitions (trust gain/loss, alliance formation/defection,
  relationship decay via half-life hazard, edge expiration, rewiring, role change) — parameters from
  log-odds/Bayesian updates + hazards, never fixed constants; each emits a StateDelta, carries valid-time, and
  changes future action feasibility + terminals.

### Deep multilayer execution (this run)
`phase9_execution.py` now has typed mechanisms for **10+ layers**: communication, exposure, trust, influence,
authority, reporting (multi-hop escalation), alliance/coordination, conflict (blocks coordination), resource
(capacity-gated transfer), jurisdiction (scope-gated institutional action) — each emitting StateDeltas with
reason codes on blocked actions.

---

## Part 4 — Final hardening run (no architecture redesign)

No architecture was redesigned this run; only production gates were closed:
- **2nd real graph domain** (`wmv2_phase9_enron_graph.py`): Enron email communication → future-edge temporal
  prediction through the same exposure edge posterior.
- **Fitted likelihoods** (`wmv2_phase9_fit_likelihoods.py`): the `repeated_interaction` detect/false rates fit
  on real Enron with a node-disjoint split, compared against the fixed `EDGE_OBS_MODELS` table.
- **Real-outcome validation** (`wmv2_phase9_real_outcome.py`): population (GSS) + network (Enron) vs strong
  baselines with paired bootstrap CIs.
- **100-question automatic discovery** (`wmv2_phase9_discovery_eval.py`, 112 questions × 14 domains): the
  universal `simulate_with_populations_networks(question, as_of, horizon)` path — caller supplies only the
  question.

The central architectural finding this run: **graph inference is calibrated and causally consumed but does NOT
beat a simple frequency baseline on real-outcome point prediction** — so the correct production posture is to
use Phase-9 graph structure for calibrated uncertainty + execution structure, and keep Phase-2 as the
outcome-forecast default. Interfaces later phases should consume: `simulate_with_populations_networks`
(universal entry), `phase9_discovery.discover` (typed discovery plan), `infer_edge_posterior_exposure`
(informative-absence edge posterior), `phase9_temporal.evolve` (temporal transitions).
