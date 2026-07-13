# WMv2 Phase 2 — Audit & Architecture (Production Evidence and As-Of Grounding)

*Phase 2 makes the universal world model act on **contemporaneous, temporally-verified, claim-level
evidence**: the compiler emits typed evidence requirements, a retrieval orchestrator gathers real
multisource evidence as of the question date, every item is temporally verified and reduced to span-validated
claims with entity/dependence/contradiction/visibility structure, an immutable bundle is frozen, and the
compiler re-plans from it — changing actors, institutions, events, hypotheses and the terminal distribution,
not merely a qualitative lean. The Phase-1 no-abstention contract is preserved: weak/absent/conflicting
evidence degrades the support grade, it never blocks a forecast.*

---

## Part 1 — Current-system audit (performed before writing code)

### Runtime network reality (reproduced, not assumed)
- Outbound HTTPS works through the agent proxy (`HTTPS_PROXY` enabled).
- **Google News RSS is LIVE in this runtime.** A paired query
  `nurses union hospital contract ratification after:2023-08-01 before:2023-09-30` returned **HTTP 200, 8
  items, 16.7 KB, 0.82 s** with real 2023 sources and parseable `pubDate`s. The "previously observed Google
  News RSS failure" is **not reproducible here** — direct access is not blocked (diagnosis in Part 3).
- **archive.org Wayback availability API is LIVE** — used as an independent server-side temporal signal.

### What already existed (Session-1 foundation, preserved and verified through the real call path)
| component | file | state |
|---|---|---|
| Typed evidence item + bundle (v1) | `evidence.py` | as-of gate, quarantine, retrieval log, visibility filter, bundle hash, append-only persistence |
| Leakage auditor | `leakage_audit.py` | resolution-term / future-date / retrospective / duplicate / timestamp-basis checks |
| Google News RSS + as-of filter | `engine/retrieval.py` | `asof_google_news` used paired `before:`/`after:` + defensive pubDate drop, but flat passages — no trace, no raw persistence, no claim structure |
| GDELT as-of headlines | `retrieval/asof_news.py` | GDELT doc API (rate-limited; alt discovery arm) |

### Gaps this phase closed (schema-only / missing → executing)
1. No retrieval-trace record; failure indistinguishable from zero results. → `RetrievalTrace` with
   `connector_status` ∈ {ok, zero_results, http_error, timeout, network_error, parse_error, invalid_query}.
2. No raw-content persistence / content-addressing. → `RawContentStore` (write-once, sha256).
3. No paired-date **invariant**. → `PairedDateError` + `paired_dates_ok` + tests that fail on a historical
   query missing either operator.
4. Single collapsed timestamp; RSS pubDate trusted as truth. → multi-signal `TemporalVerifier`, 6 statuses,
   independent `verified_pre_asof` tier from Wayback.
5. Document-level only. → claim-level extraction, entity resolution, dependence/syndication, contradiction
   graph, actor visibility.
6. Evidence rendered into a prompt string; no causal effect. → evidence-conditioned plan diff +
   WorldState/actor-view/StateDelta materialization.
7. Credibility a coarse scalar. → structured source profile (typed components; scalar only as a documented,
   uncertainty-aware derivation).

Benchmark adapters (Enron, CMV, Upworthy, historical) supply dataset-specific context and remain separate;
the general path (`compile_world → requirements → orchestrator → connectors → bundle`) stays universal.

---

## Part 2 — Evidence architecture

```
question + as_of
  → compile_world (preliminary plan)                        swm/world_model_v2/compiler.py
  → requirements_from_plan (typed evidence requirements)    evidence_requirements.py
  → gather_evidence (orchestrator)                          evidence_orchestrator.py
       ├ GoogleNewsRSSConnector (paired after:/before:)     evidence_connectors.py
       ├ Wikipedia / WebPage / UserDoc / Dataset / Prior    evidence_connectors_more.py
       ├ RawContentStore (content-addressed) + RetrievalTrace
       ├ TemporalVerifier (6 statuses, Wayback verified)    evidence_temporal.py
       ├ extract_claims (LLM + strict span validation)      evidence_claims.py
       ├ EntityResolver (ambiguity preserved)               evidence_entities.py
       ├ cluster_dependence (exact/near-dup/syndication)    evidence_dependence.py
       ├ build_contradiction_graph                          evidence_contradictions.py
       ├ assign_visibility (9 states) + actor_view          evidence_visibility.py
       └ leakage audit (claim-level) + partition
  → EvidenceBundleV2 (frozen, content-addressed, hashed)    evidence_bundle_v2.py
  → recompile_with_evidence (plan diff)                     evidence_recompile.py
  → attach_evidence_observations + materialize_public       evidence_materialize.py
  → run_from_plan → SimulationResult                        pipeline.py / materialize.py
```
Entry: `evidence_pipeline.simulate_with_evidence(question, *, llm, as_of, horizon, …) -> (SimulationResult,
artifacts)`.

### Evidence requirement planning (A)
`EvidenceRequirement` carries the claim/quantity needed, why it is causally relevant, the affected component,
expected sensitivity + value-of-information, preferred/fallback/disallowed source types, as-of + event-time +
publication-time + geographic + jurisdiction + entity scope, structured fields, actor-visibility assumption,
whether absence is informative, cost estimate, stopping criteria, and the missing-evidence consequence.
`requirements_from_plan` derives them deterministically from the compiled plan (terminal outcome,
high-sensitivity latents, institutions, discriminating facts per structural hypothesis), prioritized by VoI.
The evidence system is subordinate to the compiler on WHAT to answer but may report a requirement
fulfilled/partial/contradictory/ambiguous/unmet; new requirements route back through compilation.

### Retrieval orchestration (B, C)
The orchestrator decomposes requirements into source-specific queries, issues bounded retrieval across
connectors, persists a trace for EVERY invocation (failures included, never silently dropped), normalizes
content, and drives the analysis pipeline. Historical discovery windows are mechanically derived from the
as-of timestamp + a lookback and ALWAYS carry both `after:` and `before:`. Six live-verified production
connector categories: Google News RSS, web page, Wikipedia (server-side revision timestamp), user documents,
local datasets, prior artifacts (machine-readable registry:
`experiments/results/wmv2_phase2_source_adapter_registry.json`).

### Temporal model (D)
Signals are separated (retrieval time, claimed/feed publication time, server-side verified time, event
time). `TemporalVerifier` classifies each item as `verified_pre_asof` / `likely_pre_asof` / `uncertain` /
`likely_post_asof` / `verified_post_asof` / `undated`. The RSS pubDate is CLAIMED (never a `verified` label
alone); an archive.org capture at/before as_of upgrades to `verified_pre_asof`. Only pre-asof statuses are
production-admissible; `uncertain` is sensitivity-only; post-asof is excluded to the leakage report. Google
`after:`/`before:` operators are treated as discovery, not proof — the leakage ablation shows before-only
lets 6% post-as-of items through and the independent filter zeroes the residual.

### Raw content & immutability (E, N)
`RawContentStore` is content-addressed and write-once (twenty syndicated copies of a feed cost one blob).
`EvidenceBundleV2` is versioned and immutable: `freeze()` computes a deterministic hash over every
downstream-affecting field (requirements, raw refs, docs, claims, entities, dependence, contradictions,
visibility, partitions); a correction produces a linked NEW version via `new_version()`, never an in-place
mutation. Bundles persist to `experiments/results/phase2_bundles/`.

### Claim / entity / dependence / contradiction / visibility (F–J)
- **Claims**: the LLM proposes typed claims (13 classes) with an EXACT supporting span; a claim whose span
  is not a verbatim substring of the source is rejected. Class is preserved so an actor_statement /
  allegation / forecast is never silently promoted to a fact.
- **Entities**: mentions cluster into ranked candidate entities; ambiguity is PRESERVED (no forced merge
  when support is split) because a wrong merge contaminates every downstream claim.
- **Dependence**: union-find over exact content hash, near-duplicate shingle Jaccard, and shared canonical
  link → dependence groups; the independent-source count is the number of groups, not raw documents.
- **Contradictions**: claim-level graph (mutual exclusion, numerical disagreement, denial-vs-allegation,
  correction, retraction) with temporal order; all plausible claims are preserved, never resolved by one
  LLM preference.
- **Visibility**: 9 states; public sources → public-from-publication, user/unknown → restricted (fail-safe).
  `actor_view` enforces that an actor only observes what its visibility permits, when it could.

### Leakage auditor (K)
Claim-level flags for post-as-of source, retrospective language in the span, and resolution-term presence,
producing included/excluded/suspicious partitions with reason codes; logically separable from extraction (the
same LLM call never both extracts a claim and certifies it leak-free).

### Compiler integration & WorldState materialization (O, P)
`recompile_with_evidence` uses the frozen bundle's INCLUDED claims to propose QUALITATIVE structural
revisions grounded in specific claim ids — new entities/institutions/rules/relations/events, hypothesis
reweighting, visibility and mechanism changes, lean and uncertainty — never minting numbers. It emits a
machine-readable `PlanDiff` (each change with supporting claim ids and a `lean_only` flag exposing the
failure mode the spec forbids) and preserves the pre-evidence plan. `attach_evidence_observations` schedules
`observe_evidence` events at the permitted observation time; `EvidenceObservationOperator` writes included
claims as `observed` facts (the only layer allowed to) and emits StateDeltas with provenance = claim id;
`materialize_public_evidence` fills the omniscient store while actor-restricted claims reach actors only
through the visibility-gated observation events. `evidence_causal_effect` runs the plan pre vs post and
reports exactly what changed — a component is causal only if it changes plan structure, actor view, action
feasibility, mechanism selection, event queue, StateDelta trace, structural weights, terminal distribution,
or support grade. Verified end-to-end on the "finance approval rule discovered" case: finance sign-off claim
→ `finance_dept` institution ADDED → approval event ADDED → hypothesis reweighted → 60 observation
StateDeltas → terminal approval 0.40 → 0.37 (`lean_only=False`).

### LLM evidence contract (L)
The LLM may generate queries, extract claims (with spans), classify claim type, propose contradictions and
entity matches, flag retrospective language, propose visibility hypotheses, and propose structural
revisions. It may NOT certify publication time, invent a citation/snapshot, merge ambiguous entities
silently, count duplicates as independent, emit a credibility scalar, or produce the terminal probability
from evidence text. Every LLM output passes span/schema/temporal/entity/dependence/unsupported-precision
checks.

---

## Part 3 — Google News RSS diagnosis and production connector

**Diagnosis.** A live paired query returns HTTP 200 with a well-formed RSS feed in ~0.8 s; the block the
task anticipated is not present in this runtime. Recorded facts: URL construction (`/rss/search?q=…&hl=…`),
status 200, `application/xml`, ~13 items on a broad query, item `<title>/<link>/<pubDate>/<source>`.

**Production connector** (`GoogleNewsRSSConnector`): builds the logical query
`<terms> after:YYYY-MM-DD before:YYYY-MM-DD` and encoded wire URL; both dates are REQUIRED
(`PairedDateError`) — the production historical arm never uses `before:` alone. It persists the raw feed by
content hash and records a full `RetrievalTrace` (logical + wire query, both dates, retrieval time, status,
redirects, headers, raw hash, parser version, item rank + pubDate + links), with bounded retries + backoff.
It distinguishes `zero_results` (HTTP 200, empty feed) from technical failure. RSS dates are discovery only:
every item is re-verified by `TemporalVerifier`. Empirical justification for the paired rule is in
`WMV2_PHASE2_VALIDATION.md` (before-only 6% post-as-of leakage → paired 0%).

Reproduce: `PYTHONPATH=. python -m pytest tests/test_wmv2_evidence_phase2.py -q` (offline unit tests + a
network-guarded live Google News RSS test).
