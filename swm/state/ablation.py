"""Automated factor ablation (audit's core principle; spec section 4).

Replaces "map every variable" with "map every candidate, keep only what earns its place."
For each factor: fit the outcome head WITH it and WITHOUT it (drop its column), compare held-out
log loss / Brier / ECE / uplift@k on a temporal split. Mark KEEP if removing the factor WORSENS
held-out performance; else EXPERIMENTAL/DROP. The registry's status is updated in place.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.eval.metrics import (brier_score, expected_calibration_error, log_loss, uplift_at_k)
from swm.transition.readout import LogisticReadout


@dataclass
class AblationResult:
    factor: str
    delta_logloss: float   # (without - with); positive => factor helps => KEEP
    delta_ece: float
    delta_uplift: float
    verdict: str


def _fit_eval(rows_tr, y_tr, rows_te, y_te, names, thr_key):
    """rows_*: list[dict factor->value]. Fit logistic on named columns, return (logloss, ece, uplift)."""
    Xtr = [[r[n] for n in names] for r in rows_tr]
    Xte = [[r[n] for n in names] for r in rows_te]
    yb_tr = [1 if s >= thr_key else 0 for s in y_tr]
    yb_te = [1 if s >= thr_key else 0 for s in y_te]
    if len(set(yb_tr)) < 2 or not names:
        base = sum(yb_tr) / len(yb_tr) if yb_tr else 0.5
        p = [base] * len(yb_te)
    else:
        m = LogisticReadout(seed=0).fit(Xtr, yb_tr)
        p = [m.predict_proba(x) for x in Xte]
    return (log_loss(yb_te, p), expected_calibration_error(yb_te, p),
            uplift_at_k(yb_te, p, 0.2))


def run_ablation(registry, rows, scores, *, thr_key: int = 40, split: float = 0.7,
                 keep_margin: float = 0.0) -> list[AblationResult]:
    """rows: list[dict factor->value] aligned with scores. Temporal order assumed (split by index).
    Updates registry factor statuses in place and returns per-factor results, most-useful first."""
    cut = int(split * len(rows))
    rows_tr, rows_te = rows[:cut], rows[cut:]
    y_tr, y_te = scores[:cut], scores[cut:]
    all_names = [n for n in registry.names() if n in rows[0]]
    full_ll, full_ece, full_up = _fit_eval(rows_tr, y_tr, rows_te, y_te, all_names, thr_key)

    results = []
    for name in all_names:
        without = [n for n in all_names if n != name]
        wl, we, wu = _fit_eval(rows_tr, y_tr, rows_te, y_te, without, thr_key)
        d_ll = wl - full_ll          # >0: removing hurts => factor helps
        d_ece = we - full_ece
        d_up = full_up - wu          # >0: removing lowers uplift => factor helps
        helps = d_ll > keep_margin or d_up > keep_margin
        verdict = "KEEP" if helps else ("DROP" if d_ll < -0.002 else "EXPERIMENTAL")
        registry.set_status(name, verdict)
        results.append(AblationResult(name, round(d_ll, 4), round(d_ece, 4), round(d_up, 4), verdict))
    results.sort(key=lambda r: r.delta_logloss, reverse=True)
    return results
