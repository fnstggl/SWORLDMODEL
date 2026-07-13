"""Immutable, content-addressed evidence bundle — Phase 2.

The versioned artifact every downstream stage records. It carries the full evidence chain: requirements,
retrieval plan + traces, raw-content references (by hash), normalized docs, claims, entities, temporal
records, dependence groups, the contradiction graph, actor visibility, leakage flags, the
included/excluded/suspicious partitions, requirement coverage, missing evidence, connector failures, cost,
latency, and every parser/prompt/model version. The bundle hash deterministically covers every field that
can affect downstream inference; a frozen bundle is never mutated in place — a new retrieval or correction
creates a NEW version linked to the prior one.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = "evidence-bundle-v2.0"
EVIDENCE_SYSTEM_VERSION = "phase2-1.0"
from swm.world_model_v2.evidence_temporal import ADMISSIBLE_PRODUCTION


@dataclass
class EvidenceBundleV2:
    bundle_id: str
    question: str
    as_of: float
    compiler_plan_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    evidence_system_version: str = EVIDENCE_SYSTEM_VERSION
    prior_bundle_hash: str = ""                          # set when this is a new version of an earlier bundle
    version: int = 1
    # requirements + retrieval
    requirements: list = field(default_factory=list)     # [dict] typed evidence requirements
    retrieval_plan: list = field(default_factory=list)   # [dict]
    retrieval_traces: list = field(default_factory=list) # [RetrievalTrace.as_dict()]
    raw_content_refs: list = field(default_factory=list) # [content_hash]
    documents: list = field(default_factory=list)        # [dict] normalized docs {id, source, url, hash, ...}
    # analysis layers
    claims: list = field(default_factory=list)           # [Claim.as_dict()]
    entities: list = field(default_factory=list)         # [EntityResolution.as_dict()]
    temporal_records: dict = field(default_factory=dict) # claim_id/doc_id -> TemporalRecord.as_dict()
    dependence_groups: list = field(default_factory=list)
    contradiction_graph: list = field(default_factory=list)
    actor_visibility: list = field(default_factory=list) # [ClaimVisibility.as_dict()]
    leakage_flags: list = field(default_factory=list)
    # partitions + coverage
    included_claim_ids: list = field(default_factory=list)
    excluded_claim_ids: list = field(default_factory=list)
    suspicious_claim_ids: list = field(default_factory=list)
    requirement_coverage: dict = field(default_factory=dict)   # req_id -> {status, n_claims}
    missing_evidence: list = field(default_factory=list)
    evidence_uncertainty: dict = field(default_factory=dict)
    connector_failures: list = field(default_factory=list)
    # accounting
    cost_usd: float = 0.0
    latency_s: float = 0.0
    seed: int = 0
    versions: dict = field(default_factory=dict)         # parser/prompt/model versions
    frozen: bool = False
    _hash: str = ""

    # ------------------------------------------------------------------ hashing / immutability
    def compute_hash(self) -> str:
        """Deterministic over every downstream-affecting field. Excludes volatile accounting (cost/latency/
        timestamps) and the hash itself so the same evidence → same hash."""
        payload = {
            "schema": self.schema_version, "system": self.evidence_system_version,
            "question": self.question, "as_of": round(self.as_of, 3),
            "plan": self.compiler_plan_hash,
            "requirements": _stable(self.requirements),
            "raw": sorted(self.raw_content_refs),
            "documents": _stable([{k: d.get(k) for k in ("id", "source", "url", "content_hash")}
                                  for d in self.documents]),
            "claims": _stable([{k: c.get(k) for k in
                                ("claim_id", "subject", "predicate", "object", "value", "claim_class",
                                 "polarity", "supporting_span", "span_verified", "temporal_validity_status",
                                 "actor_visibility", "dependence_group")} for c in self.claims]),
            "entities": _stable([{"mention": e.get("mention"), "top": (e.get("candidates") or [{}])[0]
                                 .get("entity_id")} for e in self.entities]),
            "dependence": _stable([{"members": g.get("member_ids"), "type": g.get("dependence_type")}
                                   for g in self.dependence_groups]),
            "contradictions": _stable([{"a": e.get("claim_a"), "b": e.get("claim_b"), "t": e.get("ctype")}
                                       for e in self.contradiction_graph]),
            "visibility": _stable([{"c": v.get("claim_id"), "vis": v.get("visibility"),
                                    "actors": sorted(v.get("actors", []))} for v in self.actor_visibility]),
            "included": sorted(self.included_claim_ids), "excluded": sorted(self.excluded_claim_ids),
            "suspicious": sorted(self.suspicious_claim_ids),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def freeze(self) -> str:
        self._hash = self.compute_hash()
        self.frozen = True
        return self._hash

    def bundle_hash(self) -> str:
        return self._hash or self.compute_hash()

    def new_version(self) -> "EvidenceBundleV2":
        """Create a mutable successor linked to this (frozen) bundle. Never mutate a frozen bundle in place."""
        d = asdict(self)
        d.pop("_hash", None); d.pop("frozen", None)
        nxt = EvidenceBundleV2(**d)
        nxt.prior_bundle_hash = self.bundle_hash()
        nxt.version = self.version + 1
        nxt.frozen = False
        return nxt

    # ------------------------------------------------------------------ views
    def included_claims(self) -> list:
        s = set(self.included_claim_ids)
        return [c for c in self.claims if c.get("claim_id") in s]

    def visible_claim_ids(self, actor_id: str, at_time: float | None) -> list:
        from swm.world_model_v2.evidence_visibility import ClaimVisibility
        inc = set(self.included_claim_ids)
        out = []
        for v in self.actor_visibility:
            if v.get("claim_id") not in inc:
                continue
            cv = ClaimVisibility(**{k: v.get(k) for k in
                                    ("claim_id", "visibility", "actors", "earliest_observation_time",
                                     "method", "uncertainty", "communication_path", "evidence")})
            if cv.observable_by(actor_id, at_time):
                out.append(cv.claim_id)
        return out

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bundle_hash"] = self.bundle_hash()
        return d

    def persist(self, root: str = "experiments/results/phase2_bundles") -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        h = self.bundle_hash()
        p = Path(root) / f"{self.bundle_id}.{h}.json"
        doc = self.as_dict()
        doc["_persisted_at"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        p.write_text(json.dumps(doc, indent=1, default=str))
        return str(p)


def partition_claims(claims: list, temporal_records: dict, leakage_ids: set) -> tuple:
    """Split claim ids into (included, excluded, suspicious). A claim is included iff its temporal status is
    production-admissible, its span is verified, and it is not leakage-flagged. `uncertain` temporal or
    leakage flags → suspicious; post-as-of → excluded."""
    inc, exc, susp = [], [], []
    for c in claims:
        cid = c.get("claim_id")
        # the claim's own temporal status (set from its source doc) is authoritative; fall back to the
        # per-source temporal record keyed by the claim's source_id.
        tstatus = c.get("temporal_validity_status") or \
            (temporal_records.get(c.get("source_id")) or {}).get("status", "")
        if cid in leakage_ids:
            susp.append(cid); continue
        if tstatus in ("likely_post_asof", "verified_post_asof"):
            exc.append(cid); continue
        if tstatus in ADMISSIBLE_PRODUCTION and c.get("span_verified"):
            inc.append(cid)
        elif tstatus == "undated" and c.get("span_verified"):
            susp.append(cid)                             # undated but sourced → sensitivity only
        else:
            susp.append(cid)
    return inc, exc, susp


def _stable(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))
