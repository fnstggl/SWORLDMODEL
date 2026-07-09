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


def pool_distribution(persona_dists, weights=None, *, temperature: float = 1.0, eps: float = 1e-3,
                      min_p: float = 0.02) -> dict:
    """Aggregate many persona forecasts into one distribution by a WEIGHTED LOG-LINEAR OPINION POOL
    (geometric mean of probabilities), the principled aggregator — NOT the naive linear mean.

    Why: a linear mean of independent LLM forecasts regresses toward the uniform (a handful of wishy-washy
    0.5s drags a real signal to the middle — our 'underconfident on clear favorites' failure). The log-linear
    pool multiplies the odds, so genuine AGREEMENT sharpens the result (5 personas all leaning YES → a
    confident YES) while a real dissenter widens it. `temperature` (>1 tempers, <1 sharpens) is the
    OUT-OF-SAMPLE-fitted recalibration knob (fit_temperature), never a guess. A per-forecast floor `min_p`
    keeps a single certain persona from forcing 0/1 (finite-sample smoothing)."""
    if not persona_dists:
        return {}
    options = list(persona_dists[0].keys())
    weights = weights if weights is not None else [1.0] * len(persona_dists)
    wsum = sum(weights) or 1.0
    logp = {o: 0.0 for o in options}
    for d, w in zip(persona_dists, weights):
        for o in options:
            p = min(1.0, max(min_p, d.get(o, 0.0)))       # smooth each persona away from 0
            logp[o] += w * math.log(max(eps, p))
    logp = {o: (v / wsum) / max(1e-6, temperature) for o, v in logp.items()}
    m = max(logp.values())
    ex = {o: math.exp(v - m) for o, v in logp.items()}
    z = sum(ex.values()) or 1.0
    return {o: v / z for o, v in ex.items()}


def fit_temperature(preds, outcomes, *, grid=None) -> float:
    """Temperature scaling in logit space, fit to MINIMIZE held-out log-loss. T>1 tempers an overconfident
    engine, T<1 sharpens an underconfident one — the data decides which. Binary preds (p for YES)."""
    if grid is None:
        grid = [0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.6, 3.5]

    def loss(T):
        tot = 0.0
        for p, y in zip(preds, outcomes):
            q = _sig(_logit(p) / T)
            q = min(1 - 1e-6, max(1e-6, q))
            tot += -(y * math.log(q) + (1 - y) * math.log(1 - q))
        return tot / max(1, len(preds))
    return min(grid, key=loss) if preds else 1.0


def crossfit_temperature(preds, outcomes, *, k: int = 5) -> dict:
    """OUT-OF-SAMPLE recalibration: fit the temperature on k-1 folds, measure improvement on the held-out
    fold, average. Reports the honest (not in-sample-optimistic) log-loss before/after and the mean T.
    This is what makes the calibration claim real — the shrink/temperature is validated on data it did not
    see, exactly like the grade itself."""
    n = len(preds)
    if n < k * 2:
        T = fit_temperature(preds, outcomes)
        return {"temperature": T, "n": n, "note": "n too small for cross-fit; in-sample T"}
    idx = list(range(n))
    folds = [idx[i::k] for i in range(k)]                 # deterministic strided folds (no RNG)
    before = after = 0.0
    Ts = []
    for f in folds:
        test = set(f)
        tr_p = [preds[i] for i in idx if i not in test]
        tr_y = [outcomes[i] for i in idx if i not in test]
        T = fit_temperature(tr_p, tr_y)
        Ts.append(T)
        for i in f:
            p = min(1 - 1e-6, max(1e-6, preds[i]))
            q = min(1 - 1e-6, max(1e-6, _sig(_logit(p) / T)))
            y = outcomes[i]
            before += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            after += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return {"temperature": round(sum(Ts) / len(Ts), 3), "n": n,
            "logloss_before": round(before / n, 4), "logloss_after": round(after / n, 4),
            "improved": after < before}


def apply_temperature(p: float, T: float) -> float:
    return _sig(_logit(p) / max(1e-6, T))


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
