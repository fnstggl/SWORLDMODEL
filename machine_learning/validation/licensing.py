"""Licensing validation + the license matrix.

Turns the registry's per-dataset license fields + coarse license classes into (a) a matrix
for the readiness report, (b) a consistency check (registry flags vs the class's stated
permissions), and (c) a guard that no training-forbidden dataset can slip into a training
view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..registry_io import get_dataset, load_datasets, load_licenses, training_eligibility


def _tri(v) -> str:
    return {True: "yes", False: "no"}.get(v, str(v))


def license_matrix() -> list[dict]:
    rows = []
    licenses = load_licenses()
    for did, e in sorted(load_datasets().items()):
        lc = e.get("license_class")
        cls = licenses.get(lc, {})
        elig, reason = training_eligibility(did, require_approval=True)
        elig_noappr, reason_na = training_eligibility(did, require_approval=False)
        rows.append({
            "dataset": did,
            "license": e.get("license"),
            "license_class": lc,
            "role": e.get("dataset_role"),
            "commercial_use": e.get("commercial_use_allowed"),
            "derivatives": e.get("derivatives_allowed"),
            "redistribution": e.get("redistribution_allowed"),
            "training_allowed_by_license": _tri(cls.get("training_allowed")),
            "training_eligible_now": elig,
            "training_eligible_reason": reason,
            "eligible_if_approved": elig_noappr,
        })
    return rows


@dataclass
class LicenseConsistencyReport:
    issues: list = field(default_factory=list)
    n_datasets: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def check_consistency() -> LicenseConsistencyReport:
    """Warn where a registry flag contradicts its license class."""
    rep = LicenseConsistencyReport()
    licenses = load_licenses()
    for did, e in load_datasets().items():
        rep.n_datasets += 1
        cls = licenses.get(e.get("license_class"), {})
        # if class forbids commercial but registry says commercial yes -> contradiction
        cls_comm = cls.get("commercial_use")
        reg_comm = str(e.get("commercial_use_allowed")).lower()
        if cls_comm is False and reg_comm == "yes":
            rep.issues.append({"dataset": did, "issue": "registry commercial_use_allowed=yes "
                              f"but license_class {e.get('license_class')} is non-commercial"})
        if cls.get("derivatives") is False and str(e.get("derivatives_allowed")).lower() == "yes":
            rep.issues.append({"dataset": did, "issue": "registry derivatives_allowed=yes "
                              f"but license_class {e.get('license_class')} forbids derivatives"})
        # a training candidate must have a training-permitting class
        if e.get("dataset_role") in ("TRAIN_CANDIDATE", "VALIDATION_CANDIDATE") and not cls.get("training_allowed"):
            rep.issues.append({"dataset": did, "issue": "TRAIN candidate but license class "
                              f"{e.get('license_class')} does not permit training"})
    return rep


def verify_view_licenses(view_summary: dict) -> dict:
    """Ensure every dataset in a (non-eval) training view is training-permitted."""
    if view_summary.get("eval_only_view"):
        return {"ok": True, "violations": [], "note": "eval-only view: not training data"}
    violations = []
    for did in view_summary.get("datasets_included", []):
        e = get_dataset(did)
        cls = load_licenses().get(e.get("license_class"), {})
        if not cls.get("training_allowed"):
            violations.append({"dataset": did, "license_class": e.get("license_class")})
    return {"ok": not violations, "violations": violations}
