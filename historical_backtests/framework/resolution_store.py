"""Outcome-isolated resolution store. IMPORT-GUARDED: importing this module without
REPLAY_SCORER=1 raises immediately, so no forecast-time code path can even load it. The
forecaster process must never import, open, query, or deserialize anything under
historical_backtests/resolution_vault/ — a sentinel test asserts the runner module tree holds no
reference to this module, and every access is appended to the outcome-access audit ledger.

SCIENTIFIC LIMITATION (stated per protocol): mechanical outcome isolation prevents the forecasting
CODE and MODEL from reading answers. It cannot make developers forget public historical outcomes
they have inspected. Reusable-regression results are engineering evidence
(REUSABLE_DEVELOPMENT_BACKTEST); strong claims require a newly selected ROTATING_SEALED_HOLDOUT
or the live forward vault.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

if os.environ.get("REPLAY_SCORER") != "1":
    raise ImportError(
        "resolution_store is scorer-only: set REPLAY_SCORER=1 in a dedicated scorer process. "
        "Forecast-time code must never import this module.")

VAULT_DIR = Path(__file__).resolve().parents[1] / "resolution_vault"
ACCESS_LOG = VAULT_DIR / "outcome_access_ledger.jsonl"


def _log_access(kind: str, path: Path, extra: dict = None):
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a") as f:
        f.write(json.dumps({"at": time.time(), "kind": kind, "path": str(path),
                            "pid": os.getpid(), **(extra or {})}) + "\n")


def seal_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".seal").write_text(
        json.dumps({"sha256": digest, "sealed_at": time.time(), "file": path.name}))
    return digest


def verify_seal(path: Path) -> str:
    seal = json.loads(path.with_suffix(path.suffix + ".seal").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != seal["sha256"]:
        raise RuntimeError(f"TAMPERED: {path.name} sha {digest[:16]} != sealed {seal['sha256'][:16]}")
    return digest


def write_resolutions(benchmark_id: str, rows: dict) -> str:
    """rows: {case_id: {actual_outcome, resolution_ts, resolution_source, source_ts, source_hash,
    notes}}. Sealed on write."""
    path = VAULT_DIR / f"{benchmark_id}_resolutions.json"
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark_id": benchmark_id, "n": len(rows),
                                "written_at": time.time(), "resolutions": rows}, indent=1))
    digest = seal_file(path)
    _log_access("write+seal", path, {"n": len(rows)})
    return digest


def read_resolutions(benchmark_id: str, *, purpose: str) -> dict:
    path = VAULT_DIR / f"{benchmark_id}_resolutions.json"
    verify_seal(path)
    _log_access("read", path, {"purpose": purpose})
    return json.loads(path.read_text())["resolutions"]
