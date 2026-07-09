"""Grade-or-abstain — the rigorous wrapper the agent engine ships inside. Believable ≠ calibrated.

A society of LLM personas will produce a confident, well-narrated distribution that is plausibly wrong —
the exact failure the project defines itself against, one floor up. So the evaluator stays the product:

  - Every question-class carries a GRADE earned by backtesting the engine on RESOLVED history through
    `swm/eval/event_backtest.py` (the no-cheat harness: as-of guard, skill vs free baselines).
  - The grade lives in a persistent registry (models/agent_engine_grades.json) together with a fitted
    CALIBRATION MAP: a one-parameter logit shrink toward ignorance, fitted on the backtest (LLM societies
    are known-overconfident/too-centrist; the shrink is measured, not asserted).
  - An UNGRADED class never ships as a confident number: the forecast is still returned (native-typed,
    grounded, auditable) but flagged `abstain_confident=True` with grade "ungraded" — treat as hypothesis.
    "No ungraded logistics" generalized to "no ungraded simulation."

Calibration fuel: the in-repo backtests, plus ForecastBench's nightly resolved question sets
(swm/eval/forecastbench.py; CC BY-SA) — resolved Metaculus/Manifold/real-data questions with ground truth.
Prophet Arena (Kalshi-anchored live eval) is a leaderboard to enter, not a training set.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_PATH = os.environ.get("SWM_AGENT_GRADES", "models/agent_engine_grades.json")


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def shrink_distribution(dist: dict, lam: float) -> dict:
    """The fitted calibration map: shrink each option's logit toward ignorance by (1-lam) and renormalize.
    lam=1 leaves the sim untouched; lam<1 tempers a known-overconfident engine. Fitted, never asserted."""
    if not dist:
        return dist
    k = len(dist)
    uniform = 1.0 / k
    out = {o: _sig(lam * _logit(p) + (1 - lam) * _logit(uniform)) for o, p in dist.items()}
    z = sum(out.values()) or 1.0
    return {o: v / z for o, v in out.items()}


def fit_shrink(preds, outcomes, grid=(0.4, 0.55, 0.7, 0.85, 1.0)) -> float:
    """Choose the shrink that minimizes log-loss on backtest (p, y) pairs — the measured temperament."""
    def loss(lam):
        tot = 0.0
        for p, y in zip(preds, outcomes):
            q = _sig(lam * _logit(p))
            q = min(1 - 1e-6, max(1e-6, q))
            tot += -(y * math.log(q) + (1 - y) * math.log(1 - q))
        return tot / max(1, len(preds))
    return min(grid, key=loss) if preds else 1.0


def _letter(skill, n):
    if n < 15:
        return "ungraded"                      # too few resolved questions to claim anything
    if skill > 0.15:
        return "A"
    if skill > 0.05:
        return "B"
    if skill > 0.0:
        return "C"
    return "F"                                 # graded and LOSING to the free baseline — say so


@dataclass
class GradeRegistry:
    """question_class -> {grade, n, brier, skill, shrink}. The engine consults this on every forecast."""
    path: str = REGISTRY_PATH
    grades: dict = field(default_factory=dict)

    def __post_init__(self):
        p = Path(self.path)
        if p.exists():
            try:
                self.grades = json.loads(p.read_text())
            except (ValueError, OSError):
                self.grades = {}

    def calibration_for(self, question_class: str) -> dict:
        g = self.grades.get(question_class)
        if not g:
            return {"class": question_class, "grade": "ungraded", "shrink": 1.0,
                    "abstain_confident": True,
                    "note": ("no backtested grade for this question-class yet — the distribution is a "
                             "grounded HYPOTHESIS, not a calibrated forecast. Backtest the class on "
                             "resolved history (swm/eval/event_backtest.py or ForecastBench) to earn a "
                             "grade before trusting the numbers.")}
        return {"class": question_class, "grade": g["grade"], "shrink": g.get("shrink", 1.0),
                "abstain_confident": g["grade"] in ("ungraded", "F"),
                "n_backtest": g.get("n"), "brier": g.get("brier"), "skill": g.get("skill"),
                "note": ("graded on resolved history — the calibration map below is fitted, not asserted"
                         if g["grade"] not in ("ungraded", "F") else
                         "graded and NOT beating the free baseline — treat as hypothesis")}

    def record(self, question_class: str, *, backtest_report: dict, preds=None, outcomes=None) -> dict:
        """Store the grade a backtest earned (event_backtest.backtest output) + a fitted shrink."""
        skill_map = backtest_report.get("skill_vs") or backtest_report.get("skill") or {}
        skill = max(skill_map.values()) if skill_map else None   # skill vs the STRONGEST baseline beaten
        n = backtest_report.get("n", 0)
        rmse = backtest_report.get("rmse")
        brier = backtest_report.get("brier", round(rmse ** 2, 5) if isinstance(rmse, (int, float)) else None)
        entry = {"grade": _letter(skill if skill is not None else -1, n), "n": n,
                 "brier": brier, "skill": skill, "shrink": fit_shrink(preds or [], outcomes or [])}
        self.grades[question_class] = entry
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.grades, indent=1))
        return entry
