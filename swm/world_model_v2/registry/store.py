"""Registry persistence + the bridge to the lean in-process mechanism registry.

The machine-readable registry lives in swm/world_model_v2/registry/data/registry.json (committed —
deliverable 14) and packs.json (deliverable 15). Loading is deterministic; saving is atomic
(tmp+rename); every save appends a corruption-check hash. The store also mirrors each production
record into the existing lean `mechanisms.MechanismEntry` registry so the compiler's vocabulary and the
production registry can never drift apart.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from swm.world_model_v2.registry.record import (ApplicabilityRule, Citation, MechanismRecord,
                                                ParameterPack, ParameterSpec, RegistryError,
                                                ValidationRecord, now_iso)

DATA_DIR = Path(__file__).resolve().parent / "data"
REGISTRY_FILE = DATA_DIR / "registry.json"
PACKS_FILE = DATA_DIR / "packs.json"


class RegistryStore:
    """In-memory registry with committed JSON persistence and enforced lifecycle transitions."""

    def __init__(self):
        self.records: dict[str, MechanismRecord] = {}

    # ---------------- registration ----------------
    def register(self, rec: MechanismRecord, *, replace: bool = False) -> MechanismRecord:
        if rec.family_id in self.records and not replace:
            raise RegistryError(f"family {rec.family_id!r} already registered (replace=True to version-bump)")
        rec.created_at = rec.created_at or now_iso()
        rec.updated_at = now_iso()
        for p in rec.packs:
            p.enforce_uncertainty()
        self.records[rec.family_id] = rec
        self._mirror_lean(rec)
        return rec

    def get(self, family_id: str) -> MechanismRecord:
        r = self.records.get(family_id)
        if r is None:
            raise RegistryError(f"unknown family {family_id!r} (known: {sorted(self.records)[:12]}…)")
        return r

    def add_pack(self, family_id: str, pack: ParameterPack) -> ParameterPack:
        rec = self.get(family_id)
        pack.enforce_uncertainty()
        pack.created_at = pack.created_at or now_iso()
        if any(p.pack_id == pack.pack_id for p in rec.packs):
            raise RegistryError(f"pack {pack.pack_id!r} already exists on {family_id}")
        rec.packs.append(pack)
        rec.updated_at = now_iso()
        return pack

    def add_validation(self, family_id: str, vr: ValidationRecord, *, pack_id: str = ""):
        rec = self.get(family_id)
        vr.at = vr.at or now_iso()
        if pack_id:
            pack = next((p for p in rec.packs if p.pack_id == pack_id), None)
            if pack is None:
                raise RegistryError(f"no pack {pack_id!r} on {family_id}")
            pack.validation.append(vr)
        else:
            rec.validation.append(vr)
        rec.updated_at = now_iso()
        return vr

    # ---------------- lifecycle (enforced) ----------------
    def set_status(self, family_id: str, status: str, *, reason: str) -> MechanismRecord:
        rec = self.get(family_id)
        blockers = rec.promotion_blockers(status)
        if blockers:
            raise RegistryError(f"{family_id} → {status} blocked: " + "; ".join(blockers))
        if not reason.strip():
            raise RegistryError("status changes require a reason (provenance)")
        rec.status, rec.status_reason = status, reason.strip()
        rec.updated_at = now_iso()
        self._mirror_lean(rec)
        return rec

    # ---------------- queries ----------------
    def by_status(self, *statuses) -> list:
        return [r for r in self.records.values() if r.status in statuses]

    def executable_families(self) -> list:
        return [r for r in self.records.values() if r.executable()]

    def summary(self) -> dict:
        out = {"n_families": len(self.records),
               "n_packs": sum(len(r.packs) for r in self.records.values()),
               "by_status": {}, "families_without_validation": [], "empty_entries": []}
        for r in self.records.values():
            out["by_status"][r.status] = out["by_status"].get(r.status, 0) + 1
            if not r.has_validation(("held_out", "posterior_predictive", "transfer", "ablation",
                                     "synthetic", "failed_replication")):
                out["families_without_validation"].append(r.family_id)
            if not r.executable():
                out["empty_entries"].append(r.family_id)
        return out

    # ---------------- persistence (atomic, corruption-checked) ----------------
    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        recs = {fid: r.as_dict() for fid, r in sorted(self.records.items())}
        packs = {fid: [asdict(p) for p in r.packs] for fid, r in sorted(self.records.items()) if r.packs}
        for obj in recs.values():
            obj.pop("packs", None)             # packs live in their own file
        for path, payload in ((REGISTRY_FILE, recs), (PACKS_FILE, packs)):
            body = json.dumps(payload, indent=1, sort_keys=True, default=str)
            digest = hashlib.sha256(body.encode()).hexdigest()
            doc = {"_integrity": {"sha256": digest, "saved_at": now_iso(),
                                  "n_top_level": len(payload)},
                   "payload": payload}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
            os.replace(tmp, path)
        return {"registry": str(REGISTRY_FILE), "packs": str(PACKS_FILE)}

    @classmethod
    def load(cls) -> "RegistryStore":
        store = cls()
        if not REGISTRY_FILE.exists():
            return store
        recs = _read_checked(REGISTRY_FILE)
        packs = _read_checked(PACKS_FILE) if PACKS_FILE.exists() else {}
        for fid, d in recs.items():
            d = dict(d)
            d["parameters"] = [ParameterSpec(**p) for p in d.get("parameters", [])]
            d["applicability"] = ApplicabilityRule(**d.get("applicability", {}))
            d["citations"] = [Citation(**c) for c in d.get("citations", [])]
            d["validation"] = [ValidationRecord(**v) for v in d.get("validation", [])]
            d["packs"] = []
            rec = MechanismRecord(**d)
            for pd in packs.get(fid, []):
                pd = dict(pd)
                pd["citations"] = [Citation(**c) for c in pd.get("citations", [])]
                pd["validation"] = [ValidationRecord(**v) for v in pd.get("validation", [])]
                rec.packs.append(ParameterPack(**pd))
            store.records[fid] = rec
            store._mirror_lean(rec)
        return store

    # ---------------- bridge to the lean compiler vocabulary ----------------
    @staticmethod
    def _mirror_lean(rec: MechanismRecord):
        """Mirror into mechanisms._REGISTRY so compile_world() sees production families. The lean
        calibration_status maps from the production lifecycle; quarantined/rejected families are
        REMOVED from the compiler's vocabulary."""
        from swm.world_model_v2 import mechanisms as lean
        if rec.status in ("quarantined", "rejected"):
            lean._REGISTRY.pop(rec.family_id, None)
            return
        cal = {"proposed": "experimental", "implemented": "experimental",
               "locally_validated": "prior", "transfer_validated": "calibrated",
               "production_eligible": "calibrated"}[rec.status]
        lean.register_mechanism(lean.MechanismEntry(
            rec.family_id, _lean_type(rec.ontology_type), rec.title,
            required_state=tuple(rec.required_state),
            parameter_source=(rec.packs[0].fit_method if rec.packs else "see registry record"),
            temporal_scale=rec.temporal_scale, calibration_status=cal,
            domains=tuple(rec.applicability.domains), operator=rec.code_ref,
            experimental=rec.status in ("proposed", "implemented")))


def _lean_type(ontology_type: str) -> str:
    m = {"observation": "measurement", "attention": "exogenous", "memory": "belief",
         "interpretation": "belief", "decision": "decision", "learning": "decision",
         "belief": "belief", "relationship": "relationship", "norm": "relationship",
         "bargaining": "decision", "coalition": "relationship", "participation": "decision",
         "diffusion": "diffusion", "influence": "diffusion", "network": "diffusion",
         "platform": "measurement", "resource": "resource", "institutional": "institutional",
         "measurement": "measurement", "exogenous": "exogenous"}
    return m.get(ontology_type, "numerical")


def _read_checked(path: Path) -> dict:
    doc = json.loads(path.read_text())
    body = json.dumps(doc["payload"], indent=1, sort_keys=True, default=str)
    want = doc.get("_integrity", {}).get("sha256")
    got = hashlib.sha256(body.encode()).hexdigest()
    if want and want != got:
        raise RegistryError(f"{path} failed integrity check (sha256 mismatch) — refusing to load a "
                            f"corrupted registry")
    return doc["payload"]
