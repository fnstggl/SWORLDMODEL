"""Phase 12 — fit/select/evaluate calibration on the governed corpus (Parts E/F/G/H).

fit on `calibration` split → select method on `validation` split (vs identity) → evaluate ONCE on `test`.
Also compares the hierarchical/conditioned families (Part G) and returns calibration uncertainty (Part H).
No test outcome is used for any fitting or selection. Writes machine-readable artifacts only.
"""
from __future__ import annotations
import json
from pathlib import Path

from swm.world_model_v2.calibration import (select_calibrator, fit_conditioned, ece, reliability_table,
                                            _brier, _logloss, bootstrap_calibration_uncertainty,
                                            IdentityCalibrator, CALIBRATOR_REGISTRY)

OUT = Path("experiments/results/phase12")


def _load():
    return json.loads((OUT / "corpus.json").read_text())["rows"]


def _pairs(rows, split):
    return [(r["raw_p"], r["outcome"]) for r in rows if r["split"] == split]


def _score(pairs):
    return {"n": len(pairs), "brier": round(_brier(pairs), 4), "log_loss": round(_logloss(pairs), 4),
            "ece": ece(pairs)}


def _conditioned_eval(rows, keyfn, min_cell, cal_split="calibration", test_split="test"):
    trip = [(r["raw_p"], r["outcome"], keyfn(r)) for r in rows if r["split"] == cal_split]
    cc = fit_conditioned(trip, min_cell=min_cell, fitted_on="phase12")
    tp = [(cc.apply(r["raw_p"], keyfn(r)), r["outcome"]) for r in rows if r["split"] == test_split]
    return _score(tp)


def main():
    rows = _load()
    cal, val, test = _pairs(rows, "calibration"), _pairs(rows, "validation"), _pairs(rows, "test")

    # Part F: select the calibrator on validation (vs identity)
    name, cal_obj, comparison = select_calibrator(cal, val, primary="logloss", fitted_on="phase12/calibration")

    # Part E result: evaluate raw vs selected-calibrated on the LOCKED test (scored once)
    raw_test = _score(test)
    cal_test = _score([(cal_obj.apply(p), y) for p, y in test])
    # every candidate on test (reported for transparency; selection already made on val)
    per_method_test = {}
    for m, spec in CALIBRATOR_REGISTRY.items():
        if len(cal) < spec["min_n"] and m != "identity":
            continue
        c = spec["fitter"](cal, fitted_on="phase12")
        per_method_test[m] = _score([(c.apply(p), y) for p, y in test])

    # Part G: hierarchical / conditioned comparison on test
    conditioned = {
        "no_calibration": raw_test,
        "global": cal_test,
        "domain_only": _conditioned_eval(rows, lambda r: r["domain"], min_cell=15),
        "horizon_only": _conditioned_eval(rows, lambda r: ("short" if (r["horizon_days"] or 0) <= 30 else "long"),
                                          min_cell=15),
        "support_conditioned": _conditioned_eval(rows, lambda r: r["evidence_quality"], min_cell=15),
        "hierarchical_domain_support": _conditioned_eval(
            rows, lambda r: f"{r['domain']}|{r['evidence_quality']}", min_cell=15),
    }

    # Part H: calibration uncertainty for each test row (using the selected method family)
    method_for_boot = name if name in CALIBRATOR_REGISTRY else "platt"
    cal_unc = []
    for r in (x for x in rows if x["split"] == "test"):
        u = bootstrap_calibration_uncertainty(cal, r["raw_p"], method=method_for_boot)
        cal_unc.append({"row_id": r["row_id"], "raw_p": r["raw_p"], "calibrated": round(cal_obj.apply(r["raw_p"]), 4),
                        "cal_ci90": u["ci90"], "cal_sd": u["sd"], "eff_cal_n": u["eff_n"]})

    # per-domain and per-horizon reliability on test (calibrated)
    def _by(keyfn):
        groups = {}
        for r in (x for x in rows if x["split"] == "test"):
            groups.setdefault(keyfn(r), []).append((cal_obj.apply(r["raw_p"]), r["outcome"]))
        return {k: _score(v) for k, v in groups.items()}

    # decision + honest verdict
    improved_ece = (cal_test["ece"] is not None and raw_test["ece"] is not None and cal_test["ece"] < raw_test["ece"])
    worsened_both = (cal_test["brier"] > raw_test["brier"] + 1e-9 and cal_test["log_loss"] > raw_test["log_loss"] + 1e-9)
    result = {
        "manifest_hash": json.loads((OUT / "corpus.json").read_text())["manifest_hash"],
        "n_calibration": len(cal), "n_validation": len(val), "n_test": len(test),
        "selected_calibrator": name, "selected_params": cal_obj.__dict__,
        "selection_comparison_on_validation": comparison,
        "test_raw": raw_test, "test_calibrated": cal_test, "per_method_on_test": per_method_test,
        "conditioned_comparison_on_test": conditioned,
        "reliability_raw_test": reliability_table(test),
        "reliability_calibrated_test": reliability_table([(cal_obj.apply(p), y) for p, y in test]),
        "per_domain_calibrated_test": _by(lambda r: r["domain"]),
        "per_horizon_calibrated_test": _by(lambda r: "short" if (r["horizon_days"] or 0) <= 30 else "long"),
        "calibration_uncertainty_test": cal_unc,
        "gate_calibration": {
            "selected_is_identity": name == "identity",
            "calibrated_improves_ece_or_identity": (improved_ece or name == "identity"),
            "calibrated_not_worse_on_both_proper_scores": (not worsened_both),
            "method_selected_on_validation_only": True},
        "honest_note": ("The selector fell back to IDENTITY — no calibrator reliably beat the raw probabilities "
                        "on validation (preserved negative)." if name == "identity" else
                        f"Selected {name}; it beat identity on validation proper scores and is evaluated once on test.")}
    (OUT / "calibration_results.json").write_text(json.dumps(result, indent=2))
    reg = {m: {k: v for k, v in spec.items() if k != "fitter"} for m, spec in CALIBRATOR_REGISTRY.items()}
    (OUT / "calibrator_registry.json").write_text(json.dumps(
        {"registry": reg, "selected": name, "selected_params": cal_obj.__dict__,
         "fit_manifest_hash": result["manifest_hash"],
         "compatibility": {"code_commit": "see run_manifest", "phase11_present": False,
                           "status": "PROVISIONAL (pre-Phase-11 distribution)"}}, indent=2))
    print("selected calibrator:", name)
    print("test raw:", json.dumps(raw_test), "| test calibrated:", json.dumps(cal_test))
    print("gate:", json.dumps(result["gate_calibration"]))
    return result


if __name__ == "__main__":
    main()
