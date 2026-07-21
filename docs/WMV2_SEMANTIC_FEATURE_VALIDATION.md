# WMv2 Semantic-Feature Validation (Phase 5)

Code: `swm/world_model_v2/semantic_registry.py`. Tests: `tests/test_semantic_registry.py` (5, pass).

## Why this exists

The audit's RC5: LLM semantic features entered production unvalidated, and the portfolio showed they can
HURT. This registry makes the semantic channel **evidence-gated**: a feature reaches production only after
passing reliability + significant beneficial incremental-held-out-value + transport; otherwise it is
quarantined or rejected. Failures are preserved.

## Lifecycle (gate-enforced)

`proposed → operationally_defined → reliability_validated → locally_predictive →
incrementally_predictive → transport_validated → production_eligible`, plus `domain_restricted` /
`quarantined` / `rejected`. Promotion to `incrementally_predictive` requires a domain with a SIGNIFICANT
BENEFICIAL held-out delta (CI excludes 0) after controlling for nonsemantic features; to
`production_eligible`, transport validated.

## Seeded with the REAL portfolio evidence

All 11 interpretation `FEATURE_DIMS` (urgency, obligation, task_ownership, effort_required,
relevance_to_goals, risk_of_inaction, benefit_of_action, relationship_salience, needs_clarification,
needs_delegation, thread_continuity) are **QUARANTINED from production** because their measured held-out
incremental value is null-or-HARMFUL on every domain tested:

| domain | measured incremental value | verdict |
|---|---|---|
| Upworthy headline | +0.060 W1 [0.007, 0.12] after controlling for surface features | **HARMFUL** (CI excludes 0, wrong sign) |
| Enron messaging | +0.006 Brier [−0.006, 0.019] vs fitted metadata anchor | null |
| BehaviorBench games | 0.0 [−0.005, 0.005] vs structural preference model | null |

Each feature's failure records are **preserved** (`SemanticFeature.failures`), not deleted. The features
may still run in experimental arms; they are simply EXCLUDED from production selection
(`production_features(domain)` returns `[]` on every tested domain).

## Reliability machinery

`inter_run_agreement` computes an ICC-style agreement (1 − within-item/total variance) across repeated
LLM calls / prompts / models; unit-tested to be high for stable extractions and low for noisy ones. A
feature cannot reach `reliability_validated` with agreement < 0.6.

## Four-status

- **software-implemented**: YES (lifecycle, gates, reliability metric, failure preservation).
- **executes-end-to-end**: YES (`production_features(domain)` gates which features production selection may
  use; the interpretation channel is excluded).
- **empirically-validated**: YES — seeded with three real measured domains; the harmful/null channel is
  correctly quarantined. A prospective validation (extract → annotate → incremental test) for a NEW
  candidate feature is supported by the API but not yet run on fresh data.
- **production-eligible**: the REGISTRY is production-eligible as a gate; ZERO semantic features currently
  qualify for production (the honest state — the LLM's semantic readings have not earned held-out value in
  any world tested).

## The lesson, encoded

The system's stance on LLM semantics is now structural, not a hope: *the LLM proposes structure and extracts
features; those features enter production only by passing an evidence gate; on all evidence to date, the
interpretation channel fails that gate and is excluded.*
