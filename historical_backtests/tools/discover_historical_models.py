"""Model discovery + scientific audit scaffolding (stage 1 of the two-stage process).

Queries OpenRouter's Models API for open-weight instruct checkpoints, joins endpoint metadata
(providers, quantizations, context), pulls the Hugging Face revision SHA where the repo is
resolvable, and writes an AUDIT CANDIDATES file. It NEVER auto-approves: OpenRouter metadata is
discovery evidence, not temporal proof. A human (or a later explicit audit step) must verify
release timestamps against primary sources and set approval_status in the registry by hand.
The eligibility rule is release-based:
  exact_checkpoint_public_release < question_open <= forecast_cutoff < resolution_time.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swm.api.openrouter_backend import get_endpoint_metadata, list_models

OUT = Path(__file__).resolve().parents[1] / "models" / "discovery_candidates.json"

_FAMILIES = ("meta-llama/", "mistralai/", "qwen/", "deepseek/", "google/gemma",
             "microsoft/phi", "nousresearch/", "01-ai/yi")


def _hf_sha(repo: str):
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}")
        d = json.load(urllib.request.urlopen(req, timeout=30))
        return d.get("sha"), d.get("createdAt") or d.get("lastModified")
    except Exception:  # noqa: BLE001
        return None, None


def main(min_params_b: float = 30.0):
    cands = []
    for m in list_models():
        slug = str(m.get("id") or "")
        if not any(slug.startswith(f) for f in _FAMILIES):
            continue
        name = str(m.get("name") or "")
        if ":free" in slug or "latest" in slug or "auto" in slug:
            continue                                         # mutable aliases: never eligible
        created = m.get("created")
        try:
            eps = get_endpoint_metadata(slug)
        except Exception:  # noqa: BLE001
            eps = []
        if not eps:
            continue
        hf_repo = (m.get("hugging_face_id") or "").strip()
        sha, hf_created = _hf_sha(hf_repo) if hf_repo else (None, None)
        cands.append({
            "openrouter_slug": slug, "name": name,
            "openrouter_created_ts": created,
            "hf_repo": hf_repo or None, "hf_revision_sha": sha, "hf_created": hf_created,
            "endpoints": [{"provider": e.get("provider_name"),
                           "quantization": e.get("quantization"),
                           "context_length": e.get("context_length"),
                           "pricing": e.get("pricing")} for e in eps],
            "audit_status": "pending_primary_source_verification",
            "approval_status": "UNAPPROVED (discovery only — verify release timestamp against "
                               "the developer's official announcement before registry entry)",
            "notes": "OpenRouter metadata is discovery evidence, not temporal proof.",
        })
        time.sleep(0.4)
    OUT.write_text(json.dumps({"discovered_at": time.time(), "n": len(cands),
                               "eligibility_rule": "release-based: checkpoint_public_release < "
                                                   "question_open <= cutoff < resolution",
                               "candidates": cands}, indent=1))
    print(f"wrote {OUT} ({len(cands)} candidates)")


if __name__ == "__main__":
    main()
