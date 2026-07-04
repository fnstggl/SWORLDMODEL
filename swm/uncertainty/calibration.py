"""Calibration grading (audit C.9, E.2).

Grades a model's calibration on held-out data so every prediction can carry an honest badge.
The grade is computed on the backtest, stored with the world version, and echoed on every
API response — a prediction without a grade is not allowed out the door.
"""
from __future__ import annotations

from swm.eval.metrics import expected_calibration_error


def calibration_grade(y_true: list[int], p_pred: list[float]) -> dict:
    """A (ECE<0.05), B (<0.10), C (<0.15), F otherwise — with the number, never just the letter."""
    if len(y_true) < 30:
        return {"grade": "ungraded", "ece": None,
                "note": f"only {len(y_true)} labeled outcomes; need >= 30 to grade"}
    ece = expected_calibration_error(y_true, p_pred)
    grade = "A" if ece < 0.05 else "B" if ece < 0.10 else "C" if ece < 0.15 else "F"
    return {"grade": grade, "ece": round(ece, 4), "n": len(y_true)}


IMPLEMENTED = True
