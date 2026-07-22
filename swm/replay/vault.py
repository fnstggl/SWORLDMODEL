"""Historical event vault — the public event records and the SEALED resolution store, kept apart.

Two files:
  * experiments/replay_vault/events.json         — PUBLIC: question, cutoffs, horizon, family, cluster.
  * experiments/replay_vault/SEALED_resolutions.json — outcomes + resolution rules + blinding mappings.

Structural sealing (process-level, honestly labeled): `sealed_resolutions()` refuses to load unless the
caller process declares itself the scorer (REPLAY_SCORER=1). The forecaster runner never sets it and never
imports the sealed loader; every frozen forecast row is content-hashed BEFORE the scorer runs, so a
post-hoc edit of forecasts after resolution access is detectable. This is not hardware isolation — the
limitation is recorded in the artifact — but it makes accidental outcome access impossible and deliberate
access auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

VAULT = Path("experiments/replay_vault")
EVENTS = VAULT / "events.json"
SEALED = VAULT / "SEALED_resolutions.json"


@dataclass
class HistoricalEvent:
    event_id: str
    question: str
    forecast_cutoffs: list                 # RFC3339/dates, strictly before resolution_time
    horizon: str
    domain: str = ""
    event_family: str = ""                 # correlated contracts share a family (cluster for CIs)
    entities: list = field(default_factory=list)
    outcome_contract: str = ""             # public resolution criterion wording (no outcome)

    @property
    def cluster(self) -> str:
        return self.event_family or self.event_id


def public_events() -> list:
    data = json.loads(EVENTS.read_text())
    return [HistoricalEvent(**e) for e in data["events"]]


def sealed_resolutions() -> dict:
    """Outcome store — scorer only. The forecaster process must never call this."""
    if os.environ.get("REPLAY_SCORER") != "1":
        raise PermissionError(
            "SEALED resolution store: access requires REPLAY_SCORER=1 — the forecaster process must never "
            "read outcomes. Run the scorer (experiments/replay_score.py) as a separate process AFTER the "
            "forecast artifacts are frozen and content-hashed.")
    return json.loads(SEALED.read_text())


def freeze_hash(obj) -> str:
    """Content hash for a frozen forecast row (stamped before any resolution access)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
