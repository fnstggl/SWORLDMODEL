# WMv2 Phase 2 — Current Evidence-Path Audit

*Audit performed before writing Phase-2 code (the mandated first step). Traces the current evidence/as-of
retrieval path, reproduces the Google News RSS situation in the actual runtime, and lists what exists only
in schemas vs. what actually executes.*

## Runtime network reality (reproduced, not assumed)

- **Outbound HTTPS works through the agent proxy** (`HTTPS_PROXY=http://127.0.0.1:45263`, status `enabled`).
- **Google News RSS is LIVE in this runtime.** A paired query
  `nurses union hospital contract ratification after:2023-08-01 before:2023-09-30` returned **HTTP 200, 8
  items, 16.7 KB, 0.82 s**, with real contemporaneous 2023 sources (National Nurses United, NYSNA,
  Flathead Beacon) and parseable `pubDate`s. The "previously observed Google News RSS failure" is **not
  reproducible here** — direct access is not blocked. (Diagnostic recorded in
  `WMV2_PHASE2_GOOGLE_RSS.md`.)
- **archive.org Wayback availability API is LIVE** — used as an independent server-side temporal signal
  (verified an old BBC URL to `verified_pre_asof`, earliest capture 2008).

## What already exists (Session-1 Phase 2 foundation, preserved)

| component | file | state |
|---|---|---|
| Typed evidence item + bundle | `swm/world_model_v2/evidence.py` | `EvidenceItem`/`EvidenceBundle` with an as-of gate (zero-slack default), quarantine, retrieval log, actor-visibility filter, bundle hash, append-only persistence. Real. |
| Leakage auditor | `swm/world_model_v2/leakage_audit.py` | present (extended in this phase) |
| Google News RSS + as-of filter | `swm/engine/retrieval.py` | `asof_google_news` already uses paired `before:`/`after:` + a defensive pubDate drop. Works, but returns flat `Passage`s — no raw persistence, no trace record, no claim structure, no content-addressing. |
| GDELT as-of headlines | `swm/retrieval/asof_news.py` | GDELT doc API; rate-limited; not the primary discovery arm. |
| Web/Wikipedia/Bing/keyed overlays | `swm/engine/retrieval.py` | keyless workhorses + optional SERPER/BRAVE/TAVILY overlays. |

## Gaps this phase addresses (schema-only or missing → executing)

1. **No retrieval-trace record** per invocation; failures could be confused with zero results. → new
   `RetrievalTrace` (status_code, redirects, headers, raw hash, parser version, `connector_status`
   separating `zero_results` from `http_error`/`timeout`/`network_error`/`parse_error`).
2. **No raw-content persistence / content-addressing.** → `RawContentStore` (write-once, sha256-keyed).
3. **No paired-date INVARIANT enforcement.** The old path *used* paired dates but nothing forbade
   `before:`-alone. → `PairedDateError` + `paired_dates_ok` + an invariant test that fails on a
   historical query missing either operator.
4. **Single collapsed timestamp; RSS pubDate trusted as truth.** → multi-signal `TemporalVerifier` with
   six typed statuses and an independent `verified_pre_asof` tier from Wayback.
5. **Document-level evidence only; no claim-level representation, entity resolution, dependence/syndication
   grouping, or contradiction graph.** → built in this phase (claims first; entity/dependence/contradiction
   as typed modules).
6. **Evidence rendered into a prompt string; it does not change plan structure, actor views, events or
   StateDelta.** → evidence-conditioned plan diff + WorldState/actor-view/StateDelta materialization (the
   central Phase-2 success criterion: evidence must change the *world*, not merely `outcome_lean`).
7. **Credibility is a coarse scalar prior.** → structured source profile (typed components), scalar only
   as a documented, uncertainty-aware derivation.

## Benchmark-specific / hand-curated evidence

`experiments/` benchmark adapters (Enron, CMV, Upworthy, historical benchmark) supply their own
context/passages; those are dataset-specific and are NOT the general retrieval path. The general path is
`compile_world` → evidence requirements → orchestrator → connectors → bundle. Phase 2 keeps the general
path universal; dataset adapters remain separate and unchanged.

## Scope note

This session builds the critical-path vertical slice (connector → temporal → claims → immutable bundle →
causal compiler/WorldState integration) with live retrieval and forensic traces; breadth (full entity
resolution, dependence, contradiction depth, and the large manually-annotated validation sets) is tracked
in `WMV2_PHASE2_DEPENDENCIES.md` with honest gate status in `WMV2_PHASE2_VALIDATION.md`.
