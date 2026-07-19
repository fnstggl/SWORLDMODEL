"""Runtime freeze manifest — written BEFORE locked forecasting; verified by the scorer."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[1]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_freeze(benchmark_id: str) -> Path:
    from historical_backtests.models.registry import load_registry
    from historical_backtests.framework.qualify import current_phase_contract
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    bdir = ROOT / "benchmark_versions" / benchmark_id
    caps = sorted((ROOT / "evidence_archives" / benchmark_id).glob("*.json"))
    snaps = sorted((ROOT / "fitted_packs" / "survival_snapshots").glob("*.json"))
    man = {"benchmark_id": benchmark_id, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                    time.gmtime()),
           "git_commit": head,
           "canonical_entrypoint": "swm.world_model_v2.unified_runtime.simulate_world",
           "production_phase_contract": current_phase_contract(),
           "model_registry_hash": load_registry()["registry_hash"],
           "question_vault_sha": _sha_file(bdir / "question_vault.json"),
           "n_evidence_capsules": len(caps),
           "evidence_capsules_sha": hashlib.sha256(
               "".join(_sha_file(c) for c in caps).encode()).hexdigest(),
           "n_pack_snapshots": len(snaps),
           "pack_snapshots_sha": hashlib.sha256(
               "".join(_sha_file(s) for s in snaps).encode()).hexdigest(),
           "seed": 0, "min_particles": 200,
           "scoring_rules": "framework/scorer.py + metrics.py at this commit",
           "harness_version": "historical-lab-1.0"}
    out = bdir / "runtime_freeze_manifest.json"
    out.write_text(json.dumps(man, indent=1))
    return out


if __name__ == "__main__":
    print(write_freeze(sys.argv[1] if len(sys.argv) > 1 else "openrouter_llama31_v1"))
