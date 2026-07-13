# WMv2 Phase 2 — Limitations & Dependencies

*What Phase 2 does NOT do, the exact interfaces Phase 3 must implement, migration notes, and reproducibility
commands. Phase 2 is graded separately on: software implemented · executes end-to-end · phase-locally
validated · production-eligible (see `WMV2_PHASE2_VALIDATION.md`).*

## Phase-local scope (what Phase 2 owns and does not)

Phase 2 **owns**: typed evidence requirements; live paired-`after:`/`before:` Google News RSS + 5 other
production connectors; raw + trace persistence; multi-signal temporal verification; span-validated
claim-level extraction; entity resolution with ambiguity preservation; dependence/syndication grouping;
contradiction graphs; actor-specific visibility; claim-level leakage auditing; immutable content-addressed
evidence bundles; evidence-conditioned recompilation with a machine-readable plan diff; WorldState / actor-
view / observation-StateDelta materialization; deterministic replay; real-data + ablation validation.

Phase 2 does **not** own (Phase 3): numeric Bayesian posterior inference over hidden state. Phase 2 performs
CONSERVATIVE, already-validated evidence-conditioned updates only — structural-hypothesis reweighting and
structural revision driven by included claims, materialized through the existing rollout. It does **not**
fabricate a Bayesian wrapper or arbitrary confidence arithmetic to claim numeric assimilation.

## The exact Phase 3 interface Phase 2 defines

Phase 3 (posterior world-state inference) consumes the Phase-2 evidence bundle. The observation/likelihood
contract Phase 3 must implement:

1. **Observation objects** — already produced. `EvidenceBundleV2.included_claims()` returns typed claims,
   each with subject/predicate/object/value/units, `claim_class`, `polarity`, `modality`,
   `temporal_validity_status`, `dependence_group`, `actor_visibility`, and provenance (source id + span).
   The contradiction graph (`contradiction_graph`) and dependence groups (`dependence_groups`) qualify how
   much independent weight a claim carries.

2. **Likelihood function to implement** (Phase 3):
   `likelihood(particle_world, claim, observation_model) -> float` — P(observing this claim | this particle's
   latent state), using the existing observation models (`swm/world_model_v2/observation.py`:
   `GaussianMeasurement`, `BernoulliDetection`, with `.likelihood(obs, latent)`). Phase 2 supplies the typed
   claim; Phase 3 maps claim → observation and evaluates the likelihood.

3. **Weighting rule to implement** (Phase 3): reweight `ParticlePosterior`
   (`swm/world_model_v2/posterior.py`) by the product of claim likelihoods, discounted by dependence group
   (one effective observation per group, NOT per document) and by contradiction (competing claims split
   mass rather than one winning). Phase 2 already computes the dependence groups and contradiction edges
   this rule needs.

4. **Structural-hypothesis posterior** (Phase 3 upgrade): Phase 1/2 materialize structural hypotheses as
   priors (Phase 2 reweights them heuristically from evidence via `recompile_with_evidence`). Phase 3 should
   replace the heuristic reweight with a proper posterior over hypotheses from the claim likelihoods.

The interface points are stable: Phase 3 implements `likelihood()` + the posterior weighting; it does not
need to change the bundle schema, the connectors, or the temporal/leakage layers.

## Known limitations (honest)

- **Retrieval breadth**: the general path uses Google News RSS as the primary discovery arm + Wikipedia +
  user/dataset/prior connectors. Official-filing / regulatory / court / legislative / election connectors are
  registered as categories but implemented as web-page retrieval against their public sites, not
  source-specific APIs; deeper structured adapters are future work.
- **`verified_pre_asof` on Google News items**: Google News RSS returns *redirect* URLs
  (`news.google.com/rss/articles/…`), which archive.org has no snapshot of, so the live Wayback audit
  returned `verified_pre_asof=0` on real retrieved news URLs even though **0 post-as-of items** were admitted.
  Temporal safety for those items therefore rests on the paired-window + claimed-date + defensive filter path
  (the leakage ablation shows before-only 6% → paired 0% → +filter 0%), not on Wayback. Resolving the Google
  redirect to the canonical article URL before Wayback lookup (to earn the `verified` tier for news) is a
  concrete future enhancement; Wikipedia's server-side revision timestamp already provides a genuine
  `verified` signal where it is the source.
- **Retrieval precision on private / generic questions**: keyword queries (the fix that lifted recall) surface
  topically-related public articles even for questions that are inherently private (a personal email) or
  entity-free (a generic "the incumbent mayor"). Those articles are contemporaneous and leak-free but not
  always directly relevant, so the evidence they contribute is noisier. Private-domain questions should prefer
  the `user_provided` connector over public news, and a relevance filter (requirement ↔ claim matching) should
  gate low-relevance claims — both are future work. The named-entity forensic set is the clean measure of
  relevant retrieval + causal effect.
- **Temporal verification depth**: JSON-LD `datePublished` and HTTP `Last-Modified` are defined as additional
  signals but not yet fetched per-article by default (`verify_online` is off in the batch runs for latency).
- **Entity resolution** is deterministic (normalized-alias + contextual scoring). It preserves ambiguity but
  does not link to external durable KB identifiers (Wikidata QIDs) — a future enrichment.
- **Private-domain retrieval**: personal-messaging / internal-org / best-action questions correctly return
  0 public documents; those depend on user-provided evidence (the `user_provided` connector), not public
  discovery.
- **Numeric posterior**: see Phase 3 above — evidence currently changes structure + heuristic hypothesis
  weights, not a calibrated numeric posterior.

## Migration notes

- Phase 2 adds NEW modules under `swm/world_model_v2/evidence_*` and does not modify the Phase-1 compiler's
  no-abstention contract; `simulate()` (Phase-1) is unchanged, `simulate_with_evidence()` is the additive
  Phase-2 entry.
- The Session-1 `evidence.py` v1 bundle + `leakage_audit.py` are preserved and reused; `EvidenceBundleV2` is
  a superset (claims/entities/dependence/contradictions/visibility) — v1 bundles still load.
- No Phase-1 or historical benchmark artifact was edited, hidden, or overwritten. `agent-engine-on-main` and
  PR #75 untouched. No PR merged.

## Reproducibility commands

```bash
# offline unit + network-guarded live connector/temporal/claim/entity/dependence/contradiction/visibility/
# bundle/causal tests
PYTHONPATH=. python -m pytest tests/test_wmv2_evidence_phase2.py -q

# live Google News RSS diagnosis (paired after:/before:)
PYTHONPATH=. python -c "from swm.world_model_v2.evidence_connectors import GoogleNewsRSSConnector as G; \
  i,t=G().search_historical('hospital nurses contract', after_date='2023-08-01', before_date='2023-09-30'); \
  print(t.connector_status, t.status_code, t.n_items)"

# held-out end-to-end validation (real DeepSeek + LIVE Google News RSS), resumable
DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_phase2_evidence_validation --limit 0

# before-only vs paired leakage ablation + pipeline ablations
PYTHONPATH=. python -m experiments.wmv2_phase2_ablations

# forensic traces (one per domain)
DEEPSEEK_API_KEY=… PYTHONPATH=. python -m experiments.wmv2_phase2_forensic_traces
```

Determinism: same connector snapshots (content-addressed raw store) + same seed → identical bundle hash.
Timestamps are injected, RNG is seeded, and every artifact records connector/parser/prompt/model versions.

## Continuation items (implemented interface, deeper validation ongoing)

The validation harnesses run against real live sources; the largest manually-audited annotation sets called
for by the spec (e.g. 500 hand-labeled claims, 200 blind-reviewed temporal items) are sampled at the scale
achievable this session and reported honestly in `WMV2_PHASE2_VALIDATION.md` — where a metric is measured on
a smaller real sample than the target N, that is stated explicitly rather than extrapolated. Source-specific
structured adapters (filings/court/legislative APIs) and the numeric Phase-3 posterior are the primary
open items.
