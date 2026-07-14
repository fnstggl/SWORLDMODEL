# WMV2 Temporal Replay v2 — benchmark construction & validation status

**Status: EXPLICITLY INCOMPLETE.** The mandatory benchmark (100 worlds × 4 cutoffs × 2 clean arms = 800
clean forecasts) **cannot be completed from this environment**: clean Arm A requires a *verifiable*
pre-cutoff model checkpoint, and none is accessible (the hosted deepseek-chat endpoint serves mutating
weights with no immutable hash — `experiments/replay_vault_v2/model_registry.json` records the exact
blocker and unblock path). Per the mandate, no smaller pilot is described as completion; all valid work is
committed with resumable state. Numbers live in machine-readable artifacts, not prose.

## What is built and verified

**100-world vault** (`experiments/replay_vault_v2/events.json`; builder `experiments/replay_v2/build_vault.py`):
- Exactly 100 independent event worlds sourced deterministically from the Polymarket archive (frozen
  eligibility + quota rules in code, hash-order sampling — no hand-selection; correlated contracts of one
  underlying event form ONE world/cluster).
- 11 domains, ≤15 per domain; 4 cutoffs per world at 20/45/70/90% of the archived trading lifetime;
  **100/100 worlds carry timestamp-matched archived market snapshots** (nearest tick at-or-before the exact
  cutoff — never a backdated closing price); ≥3 objective, timestamped, archived trajectory targets per
  world (price-path crossings + max-move day; sealed with resolutions); 42 multi-contract worlds
  (categorical/distributional support).
- Causal-category quotas: institutional 25 ✅, population 28 ✅, strategic-negotiation 13 ✅,
  network-diffusion 0 ❌, structural-change 3 ❌ (the source archive rarely poses explicit diffusion
  questions — recorded as a corpus gap; unblock: add a second source archive, e.g. Metaculus, under the
  same frozen rules).
- World-level splits frozen by hash order: 40 calibration / 20 validation / 40 locked test; every cutoff,
  arm, contract, trajectory label and market snapshot inherits its world's split.

**Time-capsule evidence** (`swm/replay/archive_evidence.py`): Wikipedia revision-pinned content
(`first_proven_available_at` = server-verified revision timestamp) + Wayback snapshots (capture
timestamp); raw bytes hashed (sha256), archive retrieval IDs, claimed vs proven timestamps, transformation
history; the capsule enforces `first_proven_available_at <= cutoff` at access time (a request for newer
content raises). No live Google anywhere in replay. Capsules are frozen to disk by the evidence process
before forecasting.

**Arms** (`experiments/replay_v2/forecast.py`): Arm A rows are emitted as explicit
`arm_a_blocked_external` failures (never silently skipped — they are what makes the 800-count formally
unreachable today). Arm B (causally blinded current model) and the diagnostic cutoff-prompted arm run the
full supervised runtime via the frozen-capsule injection point. Six leakage probes per clean row
(name-only, recognition, no-evidence, identity-permutation, counterfactual-evidence, temporal-fact), full
prompts+outputs recorded.

**Audit table** (`experiments/results/replay_v2/audit_rows.jsonl`): one Part-18 row per attempted
forecast — evidence byte hashes, proven timestamps, blinding flags, per-phase PhaseExecutionRecords,
StateDelta counts, terminal source, freeze hash. Infrastructure validated end-to-end on smoke rows
(explicitly non-benchmark): full runtime, 10-11 causally-active phases, zero blocked phases, archived
evidence consumed by the posterior.

**Scorer + Phase-12 governance** (`experiments/replay_v2/score.py`): freeze-hash verification before any
outcome access; leakage census; calibrators (identity/Platt/isotonic) fit ONLY on calibration worlds,
selected ONLY on validation worlds, identity kept if nothing beats it; the locked split opens ONCE
(`locked_access_log.json` records the single open; a second attempt refuses); clustered CIs; market
midpoint baseline from the archived snapshots; base-rate baselines.

## Isolation boundary (documented honestly)

Process separation: evidence construction (capsule freeze) → forecaster (reads frozen capsules + public
vault only; sealed store raises without `REPLAY_SCORER=1`) → scorer (separate process, verifies hashes
first). The forecaster's one network dependency is the LLM API; OS/container-level network whitelisting is
a **deployment requirement not satisfiable in this environment** — recorded on every audit row as
`open_internet_disabled=false_process_level_only`. Isolation gate: **PARTIAL, not claimed**.

## Gate scorecard (Part 24) — none of these are claimed passed

100 clean worlds ❌ (0 completed clean — Arm A blocked) · 4 cutoffs/world ✅ built ❌ executed ·
2 clean arms ❌ (1 available) · 800 clean forecasts ❌ · ≥10 domains ✅ (11) · ≥3 trajectory targets ✅ ·
≥50 market snapshots ✅ (100) · immutable evidence bytes ✅ (service verified) · cutoff rule ✅ (enforced +
tested) · leakage audit ✅ (built) · PhaseExecutionRecord coverage ✅ (verified on smoke) · frozen before
scoring ✅ · zero tampered ✅ (verifier in place) · forecaster resolution access: blocked by guard ✅ ·
open-internet: PARTIAL ❌ · split integrity ✅ (world-level, frozen).

## Reproduction

```
PYTHONPATH=. python experiments/replay_v2/build_vault.py --max-pages 200   # deterministic vault
PYTHONPATH=. python experiments/replay_v2/forecast.py --capsules           # freeze evidence capsules
PYTHONPATH=. python experiments/replay_v2/forecast.py                      # all worlds × cutoffs × arms
REPLAY_SCORER=1 PYTHONPATH=. python experiments/replay_v2/score.py         # governed scoring (locked once)
```

Unblock order: (1) pin an open-weights pre-cutoff checkpoint (published hash + documented cutoff) and
register it; (2) rerun forecast.py — Arm A rows regenerate; (3) complete 800 rows; (4) score. Phase 12
refit (Part 22) is NOT STARTED — its precondition (the 800-forecast corpus under a frozen runtime) does
not exist; the governed fit/select/lock pipeline is implemented and tested in score.py.
