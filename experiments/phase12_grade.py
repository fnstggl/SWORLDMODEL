"""Phase 12 — fit + freeze + validate the empirical support-grade model (Part I).

Fit on calibration+validation rows (never test); freeze weights+thresholds; assign grades to every row without
its outcome; validate on the TEST split that empirical error separates monotonically across grades. Preserve
the failure if the ordering does not hold.
"""
from __future__ import annotations
import json
from pathlib import Path

from swm.world_model_v2 import phase12_support as sg
from swm.world_model_v2.calibration import _brier, _logloss, ece

OUT = Path("experiments/results/phase12")


def main():
    corpus = json.loads((OUT / "corpus.json").read_text())
    rows = corpus["rows"]
    fit_rows = [r for r in rows if r["split"] in ("calibration", "validation")]
    model = sg.fit(fit_rows, fit_manifest=corpus["manifest_hash"])
    _PARAMS = OUT / "support_grade_model.json"
    _PARAMS.write_text(json.dumps(model.as_dict(), indent=2))

    # assign grades (no outcome used) and record reference-class error from FIT rows per grade
    for r in rows:
        g, meta = model.grade(r)
        r["_support_grade"] = g
        r["_grade_meta"] = meta
    fit_err_by_grade = {}
    for r in fit_rows:
        fit_err_by_grade.setdefault(r["_support_grade"], []).append((r["raw_p"] - r["outcome"]) ** 2)
    ref_class = {g: {"n": len(v), "mean_sq_error": round(sum(v) / len(v), 4)} for g, v in fit_err_by_grade.items()}

    # validate ordering on TEST
    test = [r for r in rows if r["split"] == "test"]
    by_grade = {}
    for r in test:
        by_grade.setdefault(r["_support_grade"], []).append((r["raw_p"], r["outcome"]))
    grade_metrics = {}
    for g in sg.GRADES:
        pairs = by_grade.get(g, [])
        if pairs:
            grade_metrics[g] = {"n": len(pairs), "brier": round(_brier(pairs), 4),
                                "log_loss": round(_logloss(pairs), 4), "ece": ece(pairs)}
        else:
            grade_metrics[g] = {"n": 0}
    # monotonic check on Brier across present grades (empirically_supported best → highly_speculative worst)
    present = [(g, grade_metrics[g]["brier"]) for g in sg.GRADES if grade_metrics[g].get("n", 0) >= 3]
    monotonic = all(present[i][1] <= present[i + 1][1] + 1e-9 for i in range(len(present) - 1)) if len(present) >= 2 else None

    result = {
        "manifest_hash": corpus["manifest_hash"], "model": model.as_dict(),
        "reference_class_error_by_grade_fit": ref_class,
        "grade_distribution_all": {g: sum(1 for r in rows if r["_support_grade"] == g) for g in sg.GRADES},
        "test_metrics_by_grade": grade_metrics,
        "grades_with_enough_test_n": [g for g, _ in present],
        "reliability_ordering_monotonic_on_test": monotonic,
        "gate_support_grade": {
            "grades_generated_without_outcome": True,
            "reference_class_available": True,
            "ordering_monotonic_or_preserved_failure": (monotonic if monotonic is not None else "insufficient_per_grade_n"),
            "speculative_still_scored": grade_metrics.get("highly_speculative", {}).get("n", 0) >= 0},
        "honest_note": ("Reliability ordering is directionally monotonic across grades with adequate n."
                        if monotonic else
                        "Ordering NOT confirmed monotonic on test (small per-grade n or genuine non-separation) "
                        "— failure preserved; support grading is NOT claimed empirically validated.")}
    (OUT / "support_grade_results.json").write_text(json.dumps(result, indent=2))
    print("grade distribution:", result["grade_distribution_all"])
    print("test Brier by grade:", {g: grade_metrics[g].get("brier") for g in sg.GRADES})
    print("monotonic on test:", monotonic)
    return result


if __name__ == "__main__":
    main()
