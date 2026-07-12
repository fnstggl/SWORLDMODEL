"""Typed evidence & as-of grounding layer — Phase 2 (Tier B of the gap-audit plan).

Replaces the free-form `evidence: str` the compiler used to receive. Every item carries source identity,
BOTH timestamps (claimed publication + retrieval), a content hash, credibility, and actor visibility; the
bundle enforces the as-of gate at ADD time with ZERO slack by default (the audited 1-day windows were a
concrete leak vector). Items that cannot prove their publication time are quarantined — retrievable for
audit, EXCLUDED from prompts and worlds by default, never silently mixed in.

The bundle hash (sha256 over sorted item hashes + as_of) is what plans/worlds/ledger locks record: two
runs with the same bundle hash saw byte-identical evidence.

Provenance upgrade contract: `materialize` stamps compiler proposals `inferred`; ONLY this layer may
label a state field `observed`, and doing so requires the item id of the supporting evidence.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from swm.world_model_v2.state import parse_time, rfc3339

SOURCE_TYPES = ("news", "wire", "archive_snapshot", "wikipedia_revision", "dataset", "market",
                "official_filing", "social", "user_provided", "prior_world_state", "unknown")


class EvidenceGateError(ValueError):
    """A post-as-of or unprovable item tried to enter the bundle — refused, loudly."""


@dataclass
class EvidenceItem:
    item_id: str
    text: str                                # the passage/claim actually shown to models
    url: str = ""
    title: str = ""
    source: str = ""                         # outlet/author/dataset id
    source_type: str = "unknown"
    retrieved_at: float = 0.0                # when WE fetched it (unix)
    published_at: float | None = None        # claimed publication time (unix)
    published_verified: bool = False         # True only for server-side timestamps (wiki revisions,
    #                                          archive snapshots, dataset vintages) — not RSS pubDates
    last_modified: str = ""
    snapshot_ref: str = ""                   # immutable snapshot URL/id where one exists
    credibility: float = 0.5                 # source credibility estimate ∈ [0,1] (labeled, coarse)
    credibility_source: str = "reference_class_prior"
    visibility: str = "public"               # "public" or comma-joined actor ids
    entities: list = field(default_factory=list)
    leakage_flags: list = field(default_factory=list)
    quarantined: bool = False
    quarantine_reason: str = ""

    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.url}|{self.title}|{self.text}".encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["content_hash"] = self.content_hash()
        return d


@dataclass
class EvidenceBundle:
    question_id: str
    as_of: float
    items: list = field(default_factory=list)          # accepted items
    quarantine: list = field(default_factory=list)     # refused/suspect items, preserved for audit
    retrieval_log: list = field(default_factory=list)  # [{provider, query, at, n_returned, n_admitted}]
    slack_s: float = 0.0                               # as-of slack. NONZERO IS AN AUDIT FLAG.

    def add(self, item: EvidenceItem, *, allow_undated: bool = False) -> bool:
        """The as-of gate. Returns True if admitted, False if quarantined. Raises only on items claiming
        publication AFTER as_of + slack (those are hard leaks, not judgment calls)."""
        if item.published_at is not None and item.published_at > self.as_of + self.slack_s:
            item.quarantined = True
            item.quarantine_reason = (f"published_at {rfc3339(item.published_at)} > as_of "
                                      f"{rfc3339(self.as_of)} (+{self.slack_s:.0f}s slack)")
            self.quarantine.append(item)
            raise EvidenceGateError(item.quarantine_reason)
        if item.published_at is None:
            if not allow_undated:
                item.quarantined = True
                item.quarantine_reason = ("no publication timestamp — cannot prove as-of validity; "
                                          "quarantined (allow_undated=True admits user-provided context "
                                          "explicitly, flagged)")
                self.quarantine.append(item)
                return False
            item.leakage_flags.append("undated_admitted_explicitly")
        if self.slack_s > 0:
            item.leakage_flags.append(f"nonzero_asof_slack:{self.slack_s:.0f}s")
        self.items.append(item)
        return True

    def log_retrieval(self, provider: str, query: str, n_returned: int, n_admitted: int):
        self.retrieval_log.append({"provider": provider, "query": query[:200],
                                   "at": rfc3339(_time.time()), "n_returned": n_returned,
                                   "n_admitted": n_admitted})

    def bundle_hash(self) -> str:
        payload = "|".join(sorted(i.content_hash() for i in self.items)) + f"|{self.as_of}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def visible_to(self, actor_id: str) -> list:
        return [i for i in self.items
                if i.visibility == "public" or actor_id in i.visibility.split(",")]

    def render(self, *, max_chars: int = 6000, actor_id: str = "") -> str:
        """The compiler/policy prompt view: dated, sourced, hash-tagged lines. NEVER includes
        quarantined items. Order: verified-timestamp items first, then by recency."""
        pool = self.visible_to(actor_id) if actor_id else list(self.items)
        pool.sort(key=lambda i: (not i.published_verified, -(i.published_at or 0)))
        out, used = [], 0
        for i in pool:
            line = (f"- [{rfc3339(i.published_at) if i.published_at else 'UNDATED'}"
                    f"{'·verified' if i.published_verified else ''} | {i.source or i.source_type}"
                    f" | cred={i.credibility:.1f} | #{i.content_hash()[:8]}] "
                    f"{i.title + ': ' if i.title else ''}{i.text}")
            if used + len(line) > max_chars:
                break
            out.append(line)
            used += len(line)
        return "\n".join(out) or "(no admissible evidence as of the question date)"

    def as_dict(self) -> dict:
        return {"question_id": self.question_id, "as_of": rfc3339(self.as_of),
                "bundle_hash": self.bundle_hash(), "slack_s": self.slack_s,
                "n_items": len(self.items), "n_quarantined": len(self.quarantine),
                "items": [i.as_dict() for i in self.items],
                "quarantine": [i.as_dict() for i in self.quarantine],
                "retrieval_log": self.retrieval_log}

    # ---------------- persistence (append-only, audit-grade) ----------------
    def persist(self, root: str = "data/evidence_bundles") -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        path = Path(root) / f"{self.question_id}_{int(self.as_of)}.json"
        doc = self.as_dict()
        doc["_persisted_at"] = rfc3339(_time.time())
        if path.exists():                                  # append-only: version, never overwrite
            n = 1
            while (v := path.with_suffix(f".v{n}.json")).exists():
                n += 1
            path = v
        path.write_text(json.dumps(doc, indent=1, default=str))
        return str(path)


def item_from_asof_passage(passage, *, retrieved_at: float | None = None) -> EvidenceItem:
    """Adapter from the V1 retrieval stack's dated Passage objects (engine/retrieval.py) into typed
    items. RSS pubDates are CLAIMED timestamps (published_verified=False); Wikipedia revision timestamps
    are server-side (verified=True). Credibility stays a labeled coarse prior at this layer."""
    src = getattr(passage, "source", "") or ""
    ts = getattr(passage, "ts", None) or getattr(passage, "published_ts", None)
    if isinstance(ts, str):
        try:
            ts = parse_time(ts)
        except ValueError:
            ts = None
    is_wiki = "wikipedia" in src.lower()
    return EvidenceItem(
        item_id=hashlib.sha1(f"{src}|{getattr(passage, 'text', '')[:120]}".encode()).hexdigest()[:12],
        text=str(getattr(passage, "text", ""))[:800],
        url=str(getattr(passage, "url", "") or ""),
        title=str(getattr(passage, "title", "") or ""),
        source=src, source_type="wikipedia_revision" if is_wiki else "news",
        retrieved_at=retrieved_at or _time.time(),
        published_at=ts, published_verified=bool(is_wiki and ts),
        credibility=0.7 if is_wiki else 0.5,
        credibility_source="reference_class_prior (source-type tier)")
