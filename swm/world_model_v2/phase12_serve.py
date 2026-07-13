"""Phase 12 — production serving: turn a raw V2 forecast into the calibrated user-facing result (Part M).

The Part-A audit found the calibration/support/critic code was defined but NOT wired: `calibrated_probability`
was `None` by default (ornamental). This module loads the FROZEN Phase-12 bundle (selected calibrator + fitted
support-grade model + provenance) and populates the real result contract so the calibrated fields actually
reach the caller. It NEVER overwrites the raw simulation number — it adds the calibrated projection, support
grade, calibration provenance and (when baselines are supplied) the critic, alongside the raw distribution.

Bundle status is PROVISIONAL: it was fit on the pre-Phase-11 distribution. `bundle["provisional"]` is True and
`compatible_with()` reports the model/version locks so a caller can refuse a stale calibrator after an upstream
change (Part S).
"""
from __future__ import annotations
import json
from pathlib import Path

from swm.world_model_v2.calibration import CALIBRATOR_REGISTRY, run_critic
from swm.world_model_v2 import phase12_support as sg

_DIR = Path("experiments/results/phase12")


def load_phase12_bundle(base_dir=None):
    d = Path(base_dir) if base_dir else _DIR
    calf = d / "calibrator_registry.json"
    supf = d / "support_grade_model.json"
    corf = d / "corpus.json"
    if not calf.exists():
        return None
    reg = json.loads(calf.read_text())
    name = reg.get("selected", "identity")
    cal_pairs = []
    if corf.exists():
        cal_pairs = [(r["raw_p"], r["outcome"]) for r in json.loads(corf.read_text())["rows"]
                     if r["split"] == "calibration"]
    calibrator = CALIBRATOR_REGISTRY[name]["fitter"](cal_pairs, fitted_on="phase12/serve")
    support_model = sg.load(supf) if supf.exists() else None
    return {"calibrator_name": name, "calibrator": calibrator, "support_model": support_model,
            "effective_calibration_n": len(cal_pairs), "provisional": True,
            "provenance": reg.get("compatibility", {}), "manifest_hash": reg.get("fit_manifest_hash")}


def compatible_with(bundle, *, phase11_present=False):
    """Part-S compatibility gate: a pre-Phase-11 bundle must not be silently reused once Phase 11 lands."""
    if bundle is None:
        return False, "no_bundle"
    if phase11_present and bundle.get("provisional"):
        return False, "provisional_calibrator_pre_phase11_needs_refit"
    return True, "ok"


def calibrated_result(result, *, bundle, support_row=None, direct_p=None, ensemble_p=None):
    """Populate the calibrated user-facing fields on a SimulationResult IN PLACE (returns it). Raw number is
    never changed. `support_row` supplies the pre-outcome support features for the fitted grade model."""
    if bundle is None or not result.has_forecast():
        return result
    raw = result.raw_probability
    if raw is not None:
        cal = bundle["calibrator"].apply(raw)
        result.calibrated_probability = round(cal, 6)
        result.calibrated_distribution = {"yes": round(cal, 6), "no": round(1 - cal, 6)}
    # empirically fitted support grade (does not use the outcome)
    if bundle.get("support_model") is not None and support_row is not None:
        grade, meta = bundle["support_model"].grade(support_row)
        result.support_grade = grade
        result.provenance["support_grade_reasons"] = meta
    # calibration provenance (Part M contract)
    result.provenance["calibration"] = {
        "calibrator_id": bundle["calibrator_name"], "provisional": bundle["provisional"],
        "effective_calibration_sample_size": bundle["effective_calibration_n"],
        "manifest_hash": bundle.get("manifest_hash"), "status": bundle["provenance"].get("status")}
    # critic (never overwrites; only annotates)
    if direct_p is not None or ensemble_p is not None:
        rep = run_critic(raw if raw is not None else 0.5, direct_p=direct_p, ensemble_p=ensemble_p)
        result.provenance["critic"] = rep.as_dict()
        if rep.flags:
            result.limitations.append(f"critic: {', '.join(rep.flags)}")
    return result
