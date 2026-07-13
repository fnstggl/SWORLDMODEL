"""Additive Phase-6 registry extension for nonlinear structure — Phase 7, Part 13.

Phase 6 owns the mechanism-family registry (`registry/data/registry.json` + `packs.json`). Phase 7 must NOT
rewrite that schema — Phases 9 and 10 are editing the same core registry in parallel, and a schema change here
would collide. Instead, Phase 7 attaches a SIDECAR: a separate, integrity-hashed
`registry/data/nonlinear_extensions.json` whose records reference a Phase-6 `family_id` (and pack) by id and
add the nonlinear-specific fields (candidate forms, selected form, form posterior, context/history schemas,
nonlinear validation + ablation, extrapolation limits, status, failures). Reading the Phase-6 registry is
unchanged; a consumer that wants the nonlinear view joins on `family_id`.

This mirrors the Phase-6 `store.py` integrity discipline (sha256 over the payload; load refuses a corrupted
file) so the sidecar has the same tamper-evidence as the core registry.
"""
from __future__ import annotations

import hashlib
import json
import os
import time as _time
from dataclasses import dataclass, field, asdict

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "registry", "data", "nonlinear_extensions.json")

NL_STATUSES = ("proposed", "structural_candidate", "software_implemented", "locally_validated",
               "transfer_tested", "transfer_validated", "production_eligible", "domain_restricted",
               "quarantined", "rejected")


class ExtensionError(ValueError):
    pass


@dataclass
class NonlinearExtension:
    """One nonlinear extension of a Phase-6 mechanism family (Part 13 additive record)."""
    extension_id: str
    family_id: str                       # Phase-6 MechanismRecord.family_id (join key)
    causal_process: str
    base_pack_id: str = ""               # Phase-6 pack this refines (or "" for a new nonlinear pack)
    nonlinear_pack_id: str = ""
    candidate_forms: list = field(default_factory=list)     # [form_id] compared
    selected_form: str = ""
    baseline_form: str = "linear"        # the simpler form it had to beat
    form_posterior: dict = field(default_factory=dict)      # {form_id: weight} structural uncertainty
    context_conditioning: dict = field(default_factory=dict)   # ContextSchema.as_dict()
    history_requirements: dict = field(default_factory=dict)   # HistoryWindow.as_dict()
    posterior_schema: dict = field(default_factory=dict)       # which latents propagate (Phase 3)
    applicability: dict = field(default_factory=dict)
    transport: dict = field(default_factory=dict)
    extrapolation_limits: dict = field(default_factory=dict)
    nonlinear_validation: list = field(default_factory=list)   # [ValidationRecord-like dicts]
    nonlinear_ablation: dict = field(default_factory=dict)
    status: str = "proposed"
    status_reason: str = ""
    failures: list = field(default_factory=list)
    code_ref: str = "swm.world_model_v2.nonlinear.operators:NonlinearMechanismOperator"
    test_ref: str = "tests/test_wmv2_phase7_execution.py"
    citations: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.status not in NL_STATUSES:
            raise ExtensionError(f"{self.extension_id}: bad status {self.status!r}")
        if not self.created_at:
            self.created_at = _now()
        self.updated_at = _now()

    def promotion_blockers(self, target: str) -> list:
        """Enforced gates for a nonlinear extension (Part 27). Mirrors the Phase-6 discipline: real held-out
        evidence, a beaten baseline, calibration, execution, and no unresolved leakage."""
        if target in ("quarantined", "rejected", "domain_restricted"):
            return []
        order = {s: i for i, s in enumerate(
            ("proposed", "structural_candidate", "software_implemented", "locally_validated",
             "transfer_validated", "production_eligible"))}
        if target not in order:
            return [f"unknown target {target!r}"]
        b = []
        passed = [v for v in self.nonlinear_validation
                  if v.get("kind") in ("held_out", "posterior_predictive", "transfer") and v.get("passed")]
        if order[target] >= order["software_implemented"]:
            if not self.selected_form:
                b.append("no selected structural form")
            if not self.candidate_forms:
                b.append("no candidate-form comparison recorded")
        if order[target] >= order["locally_validated"]:
            if not passed:
                b.append("no PASSED held-out/posterior-predictive validation (a failed check does not count)")
            if not any(v.get("baseline") for v in passed):
                b.append("no simpler baseline recorded for the passed check — must beat/justify vs linear")
        if order[target] >= order["transfer_validated"]:
            if not any(v.get("kind") == "transfer" and v.get("passed") for v in self.nonlinear_validation):
                b.append("no PASSED transfer validation")
        if order[target] >= order["production_eligible"]:
            if not self.citations:
                b.append("no supporting research/dataset citation")
            if any(v.get("kind") == "transfer" and v.get("passed") is False for v in self.nonlinear_validation):
                b.append("an on-record FAILED transfer — keep domain_restricted until resolved")
            if not self.nonlinear_ablation:
                b.append("no ablation establishing the nonlinear component's incremental value")
        return b

    def as_dict(self):
        return asdict(self)


@dataclass
class NonlinearExtensionStore:
    extensions: dict = field(default_factory=dict)      # extension_id -> NonlinearExtension
    path: str = DEFAULT_PATH

    def register(self, ext: NonlinearExtension, *, replace: bool = False):
        if ext.extension_id in self.extensions and not replace:
            raise ExtensionError(f"extension {ext.extension_id!r} exists (replace=True to override)")
        self.extensions[ext.extension_id] = ext
        return ext

    def for_family(self, family_id: str) -> list:
        return [e for e in self.extensions.values() if e.family_id == family_id]

    def set_status(self, extension_id: str, status: str, *, reason: str = ""):
        ext = self.extensions[extension_id]
        blockers = ext.promotion_blockers(status)
        if blockers:
            raise ExtensionError(f"cannot promote {extension_id} to {status}: {blockers}")
        ext.status = status
        ext.status_reason = reason
        ext.updated_at = _now()
        return ext

    def save(self, path: str | None = None):
        p = path or self.path
        payload = {eid: self.extensions[eid].as_dict() for eid in sorted(self.extensions)}
        blob = json.dumps(payload, sort_keys=True, default=str)
        integrity = {"sha256": hashlib.sha256(blob.encode()).hexdigest(), "n": len(payload),
                     "at": _now()}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"_integrity": integrity, "payload": payload}, f, indent=1, default=str)
        os.replace(tmp, p)
        return integrity

    @classmethod
    def load(cls, path: str | None = None, *, verify: bool = True):
        p = path or DEFAULT_PATH
        with open(p) as f:
            doc = json.load(f)
        payload = doc["payload"]
        if verify:
            blob = json.dumps(payload, sort_keys=True, default=str)
            got = hashlib.sha256(blob.encode()).hexdigest()
            if got != doc["_integrity"]["sha256"]:
                raise ExtensionError(f"nonlinear_extensions.json integrity check FAILED "
                                     f"({got[:8]} != {doc['_integrity']['sha256'][:8]}) — refusing to load")
        store = cls(path=p)
        for eid, d in payload.items():
            store.extensions[eid] = NonlinearExtension(**d)
        return store


def _now():
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def verify_registry(path: str | None = None) -> dict:
    """CLI verb `nonlinear verify-registry`: recompute the integrity hash and report join-key health."""
    try:
        store = NonlinearExtensionStore.load(path, verify=True)
    except FileNotFoundError:
        return {"ok": False, "reason": "no nonlinear_extensions.json yet"}
    except ExtensionError as e:
        return {"ok": False, "reason": str(e)}
    # join-key sanity: every extension must reference a real Phase-6 family
    try:
        from swm.world_model_v2.registry.store import RegistryStore
        reg = RegistryStore.load()
        fams = set(reg.records)
    except Exception:
        fams = None
    dangling = []
    if fams is not None:
        dangling = [e.extension_id for e in store.extensions.values() if e.family_id not in fams]
    return {"ok": not dangling, "n_extensions": len(store.extensions),
            "statuses": _count(e.status for e in store.extensions.values()),
            "dangling_family_refs": dangling}


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out
