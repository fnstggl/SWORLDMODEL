"""Validation suite for canonical behaviour-event data.

Modules:
  schema_validation — envelope + task-payload JSON-schema validation
  chronology        — future-info + target-leakage checks
  deduplication     — exact + near-duplicate detection (within + across datasets)
  leakage           — split-isolation aggregation
  distributions     — label/action/timing/context distributions + imbalance
  provenance        — lineage completeness + trace()
  licensing         — license matrix + training-permission guard
  orchestrator      — runs everything with critical-failure gating
"""
from __future__ import annotations

from .orchestrator import validate_dataset  # noqa: F401
