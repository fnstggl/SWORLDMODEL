"""Phase 11 — deterministic serialization, content hashing, corruption detection.

Reuses the canonical-JSON discipline of Phase 8 (`phase8_events._canonical`: sorted keys, tight separators,
``default=str``) so every Phase 11 contract has a stable, replay-safe content hash. We do NOT invent a new
persistence stack — checkpoints/lineage below layer on top of this one canonical form, and the heavy
cross-run store remains Phase 8's ``PersistentStore``.
"""
from __future__ import annotations

import hashlib
import json
import os


def canonical(payload) -> str:
    """Deterministic JSON: sorted keys, tight separators, str fallback (matches phase8._canonical)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload, *, length: int = 16) -> str:
    """SHA-256 of the canonical form (Phase 8 uses sha256 for integrity; we keep the same primitive)."""
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:length]


def verify_hash(payload, expected: str) -> bool:
    """Corruption detection: recompute and compare (constant-length prefix compare)."""
    if not expected:
        return False
    return content_hash(payload, length=len(expected)) == expected


class CorruptionError(ValueError):
    """Raised when a serialized Phase 11 artifact fails its content-hash check."""


def plan_content_hash(plan, *, length: int = 16) -> str:
    """A STRONGER plan-identity hash than ``WorldExecutionPlan.plan_hash()``.

    The audit found the inherited ``plan_hash`` covers only question/as_of/horizon/len(entities)/mechs/grade —
    so a revision that only adds an institution rule, an event, or a structural hypothesis hashes IDENTICALLY
    to its parent. Phase 11 lineage/oscillation/identity must distinguish revisions, so we fold in version,
    the full component inventories, and the Phase-11 revision markers.
    """
    def _ids(items, key="id"):
        return sorted(str(i.get(key)) for i in (items or []) if isinstance(i, dict) and i.get(key))

    def _inst(items):
        return sorted(f"{i.get('id')}:{len(i.get('rules', []) or [])}"
                      for i in (items or []) if isinstance(i, dict))

    payload = {
        "q": getattr(plan, "question", ""),
        "version": getattr(plan, "version", 1),
        "parent_version": getattr(plan, "parent_version", 0),
        "entities": _ids(getattr(plan, "entities", [])),
        "institutions": _inst(getattr(plan, "institutions", [])),
        "relations": sorted(f"{r.get('src')}-{r.get('rel')}-{r.get('dst')}"
                            for r in (getattr(plan, "relations", []) or []) if isinstance(r, dict)),
        "hypotheses": sorted(f"{h.get('id')}:{h.get('prior')}"
                             for h in (getattr(plan, "structural_hypotheses", []) or []) if isinstance(h, dict)),
        "as_of": getattr(plan, "as_of", 0.0), "horizon": getattr(plan, "horizon_ts", 0.0),
        "p11": {k: v for k, v in (getattr(plan, "provenance", {}) or {}).items() if str(k).startswith("phase11")},
        "base": plan.plan_hash() if hasattr(plan, "plan_hash") else "",
    }
    return content_hash(payload, length=length)


def atomic_write_json(path: str, payload, *, indent: int = 1) -> str:
    """Crash-safe write (temp file + os.replace) — closes the atomicity gap the audit found in Phase 8's
    ``save_checkpoint`` (which used a bare ``write_text``). Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


class Versioned:
    """Mixin for a versioned, content-hashed, deterministically-serializable Phase 11 contract.

    A concrete contract sets ``SCHEMA`` (name) and ``SCHEMA_VERSION`` (semver) as class attributes and is a
    ``@dataclass``. ``as_record()`` emits ``{_schema, _schema_version, _content_hash, ...fields}`` where the
    hash covers the fields (not itself), so a stored record self-verifies. ``migrate`` upgrades an older
    record forward; the default raises for an incompatible major, matching Phase 8's ``MigrationError`` style.
    """
    SCHEMA = "phase11.contract"
    SCHEMA_VERSION = "1.0.0"

    def _fields(self) -> dict:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(self):
            return asdict(self)
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def content_hash(self) -> str:
        return content_hash(self._fields())

    def as_record(self) -> dict:
        f = self._fields()
        return {"_schema": self.SCHEMA, "_schema_version": self.SCHEMA_VERSION,
                "_content_hash": content_hash(f), **f}

    @classmethod
    def verify_record(cls, record: dict) -> bool:
        got = dict(record)
        h = got.pop("_content_hash", "")
        got.pop("_schema", None)
        got.pop("_schema_version", None)
        return verify_hash(got, h)

    @classmethod
    def major(cls, version: str) -> int:
        try:
            return int(str(version).split(".")[0])
        except (ValueError, IndexError):
            return 0
