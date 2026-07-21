"""Frozen benchmark-background evidence bundle — the sealed-replay injection for the BTF-3 runs.

The BTF-3 leakage protocol drops live `phase2_evidence` retrieval (post-as_of retrieval could
leak the resolution). Pre-#127 harnesses passed the benchmark background only as compiler-
conditioning TEXT, which leaves the run without an evidence BUNDLE — so post-#127 the Phase-3
posterior sees zero observations and the §NAP no-silent-None ladder has no evidence-updated
fallback: a world that cannot mechanistically bind its outcome must then answer `unresolved`.
Production would have a bundle; the benchmark should too.

This builder uses the runtime's OWN sealed-replay injection point (`prebuilt_bundle` — "a FROZEN,
time-locked bundle ... replaces live retrieval. Recorded, never silent.") to construct an
EvidenceBundleV2 from EXACTLY the frozen benchmark row: the background and resolution-criteria
text authored as-of present_date by the benchmark's own pipeline. Nothing else enters. Claims are
verbatim sentence spans (span_verified), publication_time = as_of, public visibility,
temporal_validity_status verified_pre_asof (the benchmark authored them as-of). Leakage safety is
inherited from the row itself: the same allowlisted fields every prior arm used, no retrieval.

Used identically by BOTH comparison arms (EXP-107 full fidelity, EXP-108 lean adaptive)."""
from __future__ import annotations

import hashlib
import re

from swm.world_model_v2.evidence_bundle_v2 import EvidenceBundleV2

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def _sentences(text: str, cap: int = 24) -> list:
    parts = [s.strip() for s in _SENT.split(str(text or "").strip()) if len(s.strip()) >= 25]
    return parts[:cap]


def frozen_background_bundle(question: str, *, as_of_ts: float, background: str,
                             resolution_criteria: str = "", seed: int = 0) -> EvidenceBundleV2:
    doc_text = (f"Resolution criteria: {resolution_criteria}\n\n{background}"
                if resolution_criteria else str(background))
    doc_hash = hashlib.sha256(doc_text.encode()).hexdigest()[:16]
    doc = {"id": f"btf3_background_{doc_hash}", "source": "btf3_benchmark_background",
           "source_type": "benchmark_background",
           "url": "", "content_hash": doc_hash,
           "note": "the benchmark's own as-of background — authored at present_date by the "
                   "BTF-3 pipeline; the ONLY admissible document under the leakage protocol"}
    claims = []
    rows = _sentences(resolution_criteria, cap=6) + _sentences(background)
    for i, sent in enumerate(rows):
        cid = f"bg_{doc_hash}_{i}"
        claims.append({
            "claim_id": cid, "source_id": doc["id"], "subject": "benchmark_background",
            "predicate": "states", "object": sent[:300], "value": "", "units": "",
            "qualifiers": "", "modality": "asserted", "polarity": "affirm",
            "claim_class": "observed_fact", "supporting_span": sent[:400],
            "span_verified": True, "event_time": "", "publication_time": float(as_of_ts),
            "extraction_confidence": 0.9,
            "temporal_validity_status": "verified_pre_asof",
            "actor_visibility": "public", "entities": [], "contradiction_links": [],
            "dependence_group": "btf3_background",
            # render-compatibility keys: unified_runtime._bundle_text reads claims as dicts
            # via c.get("text")/c.get("source"); claims are never reconstructed into Claim
            # objects downstream, so the extra keys are inert everywhere else
            "text": sent[:400], "source": "btf3_benchmark_background",
            "provenance": {"builder": "btf3_frozen_bundle-1.0",
                           "leakage_protocol": "benchmark background only; no retrieval"}})
    bundle = EvidenceBundleV2(
        bundle_id=f"btf3_frozen_{doc_hash}", question=question, as_of=float(as_of_ts),
        documents=[doc], claims=claims,
        temporal_records={c["claim_id"]: {"status": "verified_pre_asof",
                                          "publication_time": float(as_of_ts)}
                          for c in claims},
        actor_visibility=[{"claim_id": c["claim_id"], "visibility": "public", "actors": [],
                           "earliest_observation_time": float(as_of_ts),
                           "method": "benchmark_background_public", "uncertainty": "",
                           "communication_path": "", "evidence": ""} for c in claims],
        included_claim_ids=[c["claim_id"] for c in claims],
        requirement_coverage={"benchmark_background": {"status": "covered",
                                                       "n_claims": len(claims)}},
        seed=seed,
        versions={"builder": "btf3_frozen_bundle-1.0"})
    bundle.freeze()
    return bundle


def render(bundle: EvidenceBundleV2, max_chars: int = 2400) -> str:
    rows = [f"- {c['supporting_span'][:220]}" for c in bundle.included_claims()]
    return "\n".join(rows)[:max_chars]
