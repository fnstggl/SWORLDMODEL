"""Phase 12 — empirically fitted support-grade model (Part I).

Replaces the hardcoded signal→grade rule with a transparent, monotonic reliability model whose weights are
FITTED on calibration+validation data (predicting per-forecast squared error from PRE-OUTCOME support
features) and whose grade thresholds are frozen. The grade never sees the current outcome. It is validated by
checking that empirical error separates monotonically across grades on the held-out test split.

Support features (all pre-outcome):
  n_effective   dependence-collapsed evidence quantity (more → more support)
  struct_ent    structural-hypothesis disagreement entropy (more → less support)
  horizon_days  forecast horizon (longer → less support)
  ev_quality    {low,medium,high} evidence-quality bucket

The fitted model produces a scalar expected-error; frozen tercile thresholds map it to the four grades
(empirically_supported / transfer_supported / exploratory / highly_speculative).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

GRADES = ["empirically_supported", "transfer_supported", "exploratory", "highly_speculative"]
_PARAMS = Path("experiments/results/phase12/support_grade_model.json")
_EQ = {"low": 0.0, "medium": 0.5, "high": 1.0}


def features(row):
    """Standardized-ish feature vector from a corpus row (pre-outcome)."""
    n_eff = float(row.get("n_effective_observations") or 0)
    se = row.get("structural_entropy")
    se = 1.0 if se is None else float(se)
    hd = float(row.get("horizon_days") or 60)
    eq = _EQ.get(row.get("evidence_quality", "low"), 0.0)
    return [1.0, min(n_eff, 12) / 12.0, se, min(hd, 365) / 365.0, eq]


def _dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


class SupportGradeModel:
    """Loadable frozen support-grade model: expected_error = w·features; grade by frozen thresholds."""

    def __init__(self, weights, thresholds, provenance=None):
        self.weights = weights                                # predicts expected squared error
        self.thresholds = thresholds                          # 3 ascending cut points on expected error
        self.provenance = provenance or {}

    def expected_error(self, row):
        return _dot(self.weights, features(row))

    def grade(self, row):
        e = self.expected_error(row)
        t = self.thresholds
        g = ("empirically_supported" if e <= t[0] else "transfer_supported" if e <= t[1]
             else "exploratory" if e <= t[2] else "highly_speculative")
        return g, {"expected_error": round(e, 4), "thresholds": t,
                   "inputs": {"n_effective": row.get("n_effective_observations"),
                              "structural_entropy": row.get("structural_entropy"),
                              "horizon_days": row.get("horizon_days"),
                              "evidence_quality": row.get("evidence_quality")},
                   "model_version": self.provenance.get("version", "phase12-1.0")}

    def as_dict(self):
        return {"weights": self.weights, "thresholds": self.thresholds, "provenance": self.provenance}


def fit(rows_cal_val, *, ridge=1e-2, version="phase12-1.0", fit_manifest=""):
    """Least-squares fit of expected squared error on features (ridge-regularized), thresholds at terciles of
    the fitted expected error over the fit set. rows_cal_val are calibration+validation rows (never test)."""
    X = [features(r) for r in rows_cal_val]
    y = [(r["raw_p"] - r["outcome"]) ** 2 for r in rows_cal_val]
    d = len(X[0])
    # normal equations with ridge: (X^T X + ridge I) w = X^T y
    XtX = [[sum(X[k][i] * X[k][j] for k in range(len(X))) + (ridge if i == j else 0.0) for j in range(d)]
           for i in range(d)]
    Xty = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(d)]
    w = _solve(XtX, Xty)
    preds = sorted(_dot(w, x) for x in X)
    t1, t2, t3 = preds[len(preds) // 4], preds[len(preds) // 2], preds[3 * len(preds) // 4]
    return SupportGradeModel(w, [round(t1, 4), round(t2, 4), round(t3, 4)],
                             {"version": version, "n_fit": len(rows_cal_val), "ridge": ridge,
                              "fit_manifest_hash": fit_manifest,
                              "features": ["bias", "n_eff/12", "struct_entropy", "horizon/365", "ev_quality"]})


def _solve(A, b):
    """Gaussian elimination for the small normal-equation system."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [round(M[i][n] / M[i][i], 6) if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def load(path=None):
    p = Path(path) if path else _PARAMS
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return SupportGradeModel(d["weights"], d["thresholds"], d.get("provenance"))
