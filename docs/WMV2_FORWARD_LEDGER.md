# WMv2 Forward Ledger (Phase 16)

Code: `swm/world_model_v2/forward_ledger_v2.py`. Store: `data/forward_ledger_v2.jsonl` (append-only).
Tests: `tests/test_forward_ledger_v2.py` (4, pass).

## Why

Historical backtesting is insufficient — a system can be tuned (even inadvertently) to a fixed historical
set. The forward ledger records real predictions on UNRESOLVED questions BEFORE the world resolves, with
the full provenance to score them fairly later. This is the contamination-proof complement to the
historical benchmark.

## The lock schema (every field the spec requires)

Each `ForwardLock` captures: question · as-of · horizon · evidence_bundle_hash · retrieval_log ·
leakage_grade · plan_hash · plan_version · mechanisms [{id, version, status}] · parameter_packs ·
state_posterior_summary · n_particles · code_commit · model_versions · calibration_version ·
raw_probability · calibrated_probability · confidence_grade · abstained · abstain_reason ·
uncertainty_decomposition · cost_usd · latency_s · locked_at.

## Invariants (enforced + tested)

1. **APPEND-ONLY.** Locks and resolutions are new JSONL lines; `resolve()` writes a resolution row keyed by
   (qid, lock_version) and NEVER edits the lock. (test: `test_resolution_is_a_new_line_not_an_edit`)
2. **NEVER UPDATE A FORECAST after seeing later evidence.** Re-forecasting writes a NEW lock with a NEW
   `lock_version`; the prior forecast is preserved verbatim. (test: `test_reforecast_creates_new_version_never_edits`)
3. **VERSIONED.** `lock_version` = sha256 over code commit + model versions + evidence bundle hash + plan
   hash + mechanism versions + calibration version — anything that would make a re-run non-comparable.
   Change any of them and a re-lock is a genuinely new, separately-scored forecast.

## Scoring

`score()` computes per-arm Brier on raw + calibrated probabilities over ONLY the resolved, non-abstained
locks; evaluation never tunes the calibrator (that lives in the calibrator's own train/val governance). It
reports n_resolved, n_open, and refuses to score below a minimum resolved count.

## Status

- **software-implemented**: YES (schema, append-only store, versioning, scoring; 4 tests).
- **executes-end-to-end**: YES (locks a real forecast produced by the system; resolves; scores).
- **empirically-validated**: the mechanism is validated by tests; the ledger is ACTIVE but forward
  performance requires calendar time to accrue resolutions — by construction it cannot be reported at build
  time. The historical benchmark (Phase 15) is the in-sample-time analogue that CAN be scored now.
- **production-eligible**: YES as infrastructure; forward performance numbers will populate as questions
  resolve.

## How to lock a forward forecast

```python
from swm.world_model_v2.forward_ledger_v2 import ForwardLedgerV2, ForwardLock
from swm.facade import forecast   # architecture="world_model_v2"

res = forecast(question, architecture="world_model_v2", llm=llm, evidence=bundle, as_of=..., horizon=...)
led = ForwardLedgerV2()
led.lock(ForwardLock(qid=..., question=question, as_of=..., horizon=...,
                     evidence_bundle_hash=bundle.bundle_hash(), plan_hash=res["run"]["plan_hash"],
                     raw_probability=res.get("p"), calibrated_probability=...,
                     abstained=res.get("abstain", False), n_particles=..., ...))
# later, when it resolves:
led.resolve(qid, lock_version, outcome=1.0)
```
