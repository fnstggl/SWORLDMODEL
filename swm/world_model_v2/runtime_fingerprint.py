"""Runtime fingerprint (Part O) — the version lock the next sealed Phase-12 refit must match.

The unified + integration-completion runtime changes the forecast distribution, so any calibrator/support
model/forecast corpus fit on an earlier distribution is INVALID. This module produces a content-hashed
fingerprint of every phase's version + the runtime commit; the Phase-12 refit pipeline requires it, and any
corpus produced under a different fingerprint is marked `diagnostic_only` (software-compatibility testing only,
NOT product-performance evidence).
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def runtime_fingerprint():
    source_files = (
        "swm/world_model_v2/unified_runtime.py", "swm/world_model_v2/phase_supervision.py",
        "swm/world_model_v2/activation_synthesis.py", "swm/world_model_v2/causal_relevance.py",
        "swm/world_model_v2/phase_consumers.py", "swm/world_model_v2/fallback.py",
        "swm/world_model_v2/materialize.py", "swm/api/deepseek_backend.py",
    )
    fp = {
        "unified_runtime": _safe_version("swm.world_model_v2.unified_runtime", "RUNTIME_VERSION",
                                         "unified-2.0-post-snapshot"),
        "compiler": "compiler",
        "evidence": "phase2-1.0",
        "posterior": "phase3",
        "actor_policy": "phase4",
        "mechanism_registry": "phase6",
        "nonlinear": "phase7",
        "populations_networks": "phase9",
        "institutions": "phase10 + integration-completion rule-normalization",
        "persistence": "phase8-1.0",
        "recompilation": "phase11",
        "integration_completion": "integration-completion-1.0",
        "activation_synthesis": "activation-synthesis-2.0 (question adjudication + state-path consumers)",
        "phase_consumers": "phase-consumers-2.0 (causal-state transition; no terminal modulation)",
        "commit": _commit(),
        "source_sha256": _source_hashes(source_files),
    }
    fp["fingerprint_hash"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    return fp


def _source_hashes(paths):
    root = Path(__file__).resolve().parents[2]
    out = {}
    for rel in paths:
        path = root / rel
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return out


def _safe_version(mod, attr, default):
    try:
        m = __import__(mod, fromlist=[attr])
        return getattr(m, attr, default)
    except Exception:  # noqa: BLE001
        return default


def corpus_status(corpus_fingerprint_hash):
    """A corpus is product-eligible ONLY if its fingerprint matches the current runtime; else diagnostic_only."""
    cur = runtime_fingerprint()["fingerprint_hash"]
    if corpus_fingerprint_hash == cur:
        return "product_eligible"
    return "diagnostic_only"


if __name__ == "__main__":
    print(json.dumps(runtime_fingerprint(), indent=2))
