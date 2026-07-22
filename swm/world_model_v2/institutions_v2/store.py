"""Phase 10 — institution registry store: families + templates, committed JSON, integrity hashes,
enforced promotion gates (Part 26). The machine-readable registry lives in institutions_v2/data/."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from swm.world_model_v2.institutions_v2.record import (InstitutionError, InstitutionFamily,
                                                       InstitutionTemplate, now_iso)

DATA_DIR = Path(__file__).resolve().parent / "data"
FAMILIES_FILE = DATA_DIR / "families.json"
TEMPLATES_FILE = DATA_DIR / "templates.json"

# lifecycle order for families (templates use their own gate set)
_FAM_ORDER = ("proposed", "structurally_implemented", "executable")
_TPL_ORDER = ("proposed", "evidence_encoded", "structurally_implemented", "executable",
              "locally_reconstructed", "historically_replayed", "cross_institution_tested",
              "production_eligible")


class InstitutionStore:
    def __init__(self):
        self.families: dict[str, InstitutionFamily] = {}
        self.templates: dict[str, InstitutionTemplate] = {}

    # ---------------- registration ----------------
    def register_family(self, fam: InstitutionFamily, *, replace=False):
        if fam.family_id in self.families and not replace:
            raise InstitutionError(f"family {fam.family_id!r} already registered")
        fam.created_at = fam.created_at or now_iso()
        self.families[fam.family_id] = fam
        return fam

    def register_template(self, tpl: InstitutionTemplate, *, replace=False):
        if tpl.template_id in self.templates and not replace:
            raise InstitutionError(f"template {tpl.template_id!r} already registered")
        if tpl.family_id not in self.families:
            raise InstitutionError(f"template {tpl.template_id!r} references unknown family {tpl.family_id!r}")
        tpl.created_at = tpl.created_at or now_iso()
        tpl.compute_hash()
        self.templates[tpl.template_id] = tpl
        return tpl

    # ---------------- promotion gates (enforced, honest) ----------------
    def family_blockers(self, fam: InstitutionFamily, target: str) -> list:
        b = []
        if target in ("quarantined", "rejected"):
            return []
        if target in ("structurally_implemented", "executable", "locally_reconstructed",
                      "historically_replayed", "cross_institution_tested", "production_eligible"):
            if not fam.stages:
                b.append("no stages defined")
            if not fam.roles:
                b.append("no roles defined")
        if target in ("executable",) or _tpl_rank(target) >= _tpl_rank("executable"):
            if not fam.executable():
                b.append(f"code_ref {fam.code_ref!r} does not resolve to a callable")
            if not fam.test_ref:
                b.append("no test_ref")
        return b

    def template_blockers(self, tpl: InstitutionTemplate, target: str) -> list:
        b = []
        if target in ("quarantined", "rejected"):
            return []
        rank = _tpl_rank(target)
        if rank >= _tpl_rank("evidence_encoded"):
            if not tpl.has_official_evidence():
                b.append("no verified OFFICIAL evidence — a template needs a verified primary source")
            if not tpl.rules:
                b.append("no rules")
        if rank >= _tpl_rank("structurally_implemented"):
            if not tpl.stages:
                b.append("no stage graph")
            if not tpl.roles:
                b.append("no roles")
        if rank >= _tpl_rank("executable"):
            fam = self.families.get(tpl.family_id)
            if not (fam and fam.executable()):
                b.append("family is not executable")
        if rank >= _tpl_rank("locally_reconstructed"):
            if not (tpl.valid_from or tpl.valid_to):
                b.append("not temporally versioned (no valid_from/valid_to)")
            if not any(v.get("kind") in ("reconstruction", "authorization", "stage", "decision")
                       for v in tpl.validation):
                b.append("no reconstruction/execution validation record")
        if rank >= _tpl_rank("historically_replayed"):
            if not any(v.get("kind") == "historical_replay" and v.get("passed") for v in tpl.validation):
                b.append("no PASSED historical-replay validation")
        if rank >= _tpl_rank("production_eligible"):
            if not any(v.get("kind") == "leakage_audit" and v.get("passed") for v in tpl.validation):
                b.append("no passed leakage audit")
        return b

    def set_family_status(self, family_id, status, *, reason):
        fam = self.families[family_id]
        blk = self.family_blockers(fam, status)
        if blk:
            raise InstitutionError(f"{family_id} → {status} blocked: " + "; ".join(blk))
        fam.status, fam.status_reason = status, reason
        return fam

    def set_template_status(self, template_id, status, *, reason):
        tpl = self.templates[template_id]
        blk = self.template_blockers(tpl, status)
        if blk:
            raise InstitutionError(f"{template_id} → {status} blocked: " + "; ".join(blk))
        tpl.status, tpl.status_reason = status, reason
        return tpl

    # ---------------- queries ----------------
    def templates_for(self, family_id=None, jurisdiction=None, as_of=None) -> list:
        out = []
        for t in self.templates.values():
            if family_id and t.family_id != family_id:
                continue
            if jurisdiction and jurisdiction.lower() not in (t.jurisdiction or "").lower():
                continue
            if as_of and not t.active_at(as_of):
                continue
            out.append(t)
        return out

    def summary(self) -> dict:
        return {"n_families": len(self.families), "n_templates": len(self.templates),
                "families_by_status": _count(self.families.values()),
                "templates_by_status": _count(self.templates.values())}

    # ---------------- persistence ----------------
    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fam = {fid: asdict(f) for fid, f in sorted(self.families.items())}
        tpl = {tid: asdict(t) for tid, t in sorted(self.templates.items())}
        for path, payload in ((FAMILIES_FILE, fam), (TEMPLATES_FILE, tpl)):
            body = json.dumps(payload, indent=1, sort_keys=True, default=str)
            doc = {"_integrity": {"sha256": hashlib.sha256(body.encode()).hexdigest(),
                                  "saved_at": now_iso(), "n": len(payload)}, "payload": payload}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
            os.replace(tmp, path)
        return {"families": str(FAMILIES_FILE), "templates": str(TEMPLATES_FILE)}


def _tpl_rank(status):
    return _TPL_ORDER.index(status) if status in _TPL_ORDER else 0


def _count(items):
    out = {}
    for x in items:
        out[x.status] = out.get(x.status, 0) + 1
    return out


_STORE = None


def load_store(*, reload=False):
    """Build the committed registry (institutions_v2.build.build_store) — the single source of truth."""
    global _STORE
    if _STORE is None or reload:
        from swm.world_model_v2.institutions_v2.build import build_store
        _STORE = build_store()
    return _STORE
