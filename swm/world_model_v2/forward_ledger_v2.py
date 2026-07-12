"""Forward ledger v2 — Phase 16. Append-only, versioned locks of forward (unresolved) forecasts.

Historical backtesting is insufficient; a forward ledger records real predictions BEFORE the world
resolves, with the full provenance needed to score them fairly later. The hard rules:
  * APPEND-ONLY — locks and resolutions are new lines; nothing is edited in place.
  * NEVER UPDATE A FORECAST after seeing later evidence — issue a NEW version instead.
  * VERSIONED — lock_version hashes everything that would make a re-run non-comparable (code commit,
    model versions, evidence bundle hash, plan hash, mechanism versions, calibration version).

Each lock captures every field the spec requires: question, as-of, evidence hash, retrieval log, plan
hash, state-posterior summary, mechanisms + versions, parameters, model versions, code commit, particle
count, raw + calibrated probability, abstention, cost, latency.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_LEDGER = "data/forward_ledger_v2.jsonl"


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _now():
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


@dataclass
class ForwardLock:
    qid: str
    question: str
    as_of: str
    horizon: str
    # evidence
    evidence_bundle_hash: str = ""
    retrieval_log: list = field(default_factory=list)
    leakage_grade: str = ""
    # plan / world
    plan_hash: str = ""
    plan_version: int = 1
    mechanisms: list = field(default_factory=list)          # [{mech_id, version, status}]
    parameter_packs: list = field(default_factory=list)
    state_posterior_summary: dict = field(default_factory=dict)
    n_particles: int = 0
    # models / code
    code_commit: str = field(default_factory=_commit)
    model_versions: dict = field(default_factory=dict)
    calibration_version: str = ""
    # prediction
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    confidence_grade: str = ""
    abstained: bool = False
    abstain_reason: str = ""
    uncertainty_decomposition: dict = field(default_factory=dict)
    # accounting
    cost_usd: float = 0.0
    latency_s: float = 0.0
    locked_at: str = field(default_factory=_now)
    # resolution (filled later, as a NEW line — never edits the lock)
    resolution: object = None
    resolved_at: str = ""

    def lock_version(self) -> str:
        payload = json.dumps({"commit": self.code_commit, "models": self.model_versions,
                              "evidence": self.evidence_bundle_hash, "plan": self.plan_hash,
                              "mechs": self.mechanisms, "cal": self.calibration_version},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_row(self) -> dict:
        d = asdict(self)
        d["lock_version"] = self.lock_version()
        d["kind"] = "lock"
        return d


class ForwardLedgerV2:
    """Append-only JSONL ledger. `lock()` writes a new lock row; `resolve()` writes a resolution row
    keyed by (qid, lock_version); `reforecast()` writes a NEW lock (never edits an old one)."""

    def __init__(self, path: str = DEFAULT_LEDGER):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, row: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def lock(self, lock: ForwardLock) -> str:
        row = lock.as_row()
        self._append(row)
        return row["lock_version"]

    def resolve(self, qid: str, lock_version: str, outcome: float, *, source="manual"):
        """Append a resolution row. Does NOT edit the lock (append-only)."""
        self._append({"kind": "resolution", "qid": qid, "lock_version": lock_version,
                      "outcome": outcome, "source": source, "resolved_at": _now()})

    def load(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def open_locks(self) -> list:
        rows = self.load()
        resolved = {(r["qid"], r["lock_version"]) for r in rows if r.get("kind") == "resolution"}
        return [r for r in rows if r.get("kind") == "lock"
                and (r["qid"], r["lock_version"]) not in resolved]

    def score(self, *, min_n=5) -> dict:
        """Score resolved locks: per-arm Brier on raw + calibrated. Only resolved rows are scored;
        evaluation never tunes calibration (that lives in the calibrator's own train/val)."""
        rows = self.load()
        res = {(r["qid"], r["lock_version"]): r["outcome"] for r in rows if r.get("kind") == "resolution"}
        locks = [r for r in rows if r.get("kind") == "lock"]
        scored = [(r, res[(r["qid"], r["lock_version"])]) for r in locks
                  if (r["qid"], r["lock_version"]) in res and not r.get("abstained")]
        if len(scored) < min_n:
            return {"n_resolved": len(scored), "note": f"need >= {min_n} resolved forecasts to score"}
        def brier(key):
            pr = [(r.get(key), y) for r, y in scored if r.get(key) is not None]
            return round(sum((p - y) ** 2 for p, y in pr) / len(pr), 5) if pr else None
        return {"n_resolved": len(scored), "brier_raw": brier("raw_probability"),
                "brier_calibrated": brier("calibrated_probability"),
                "n_open": len(self.open_locks())}
