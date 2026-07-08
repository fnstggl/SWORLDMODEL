"""Event backtest — score a forecaster against baselines on historical "predict the future" questions, no-cheat.

Every resolved historical event is a scored forecasting question: as-of some cutoff, predict the outcome;
grade it against what actually happened. The value of the world-model is not "did it produce a number" but
"did its number beat the baselines a skeptic would use for free" — persistence/momentum, the base rate, and
the market/poll price where one exists. This harness enforces that comparison and the no-cheat contract.

  - `Question`: an as-of cutoff, the forecaster's inputs (must all pre-date resolution), the realized
    `outcome`, and a dict of `baselines` {name: prediction} (persistence, base_rate, market, ...).
  - `backtest(questions, forecast_fn)`: runs the forecaster, scores it and every baseline on the SAME items,
    and reports SKILL = 1 − loss/loss_baseline vs each — the honest "did the simulation earn its keep".
  - A leakage guard: each Question carries `asof` and `resolved` timestamps; `assert_asof` refuses any item
    whose evidence is not strictly before resolution, so a backtest cannot silently cheat.

Works for a binary outcome (log-loss / Brier) or a continuous share (MAE / RMSE) — skill is scale-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Question:
    qid: str
    outcome: float                       # realized (0/1 for an event; a share in [0,1] for a proportion)
    baselines: dict = field(default_factory=dict)   # {"persistence": p, "base_rate": p, "market": p, ...}
    asof: str = ""                       # cutoff timestamp of the evidence used
    resolved: str = ""                   # when the outcome was known
    meta: dict = field(default_factory=dict)


def assert_asof(questions):
    """No-cheat guard: every question's evidence cutoff must strictly precede its resolution."""
    bad = [q.qid for q in questions if q.asof and q.resolved and q.asof >= q.resolved]
    if bad:
        raise ValueError(f"leakage: as-of ≥ resolved for {bad[:5]}{'...' if len(bad) > 5 else ''}")
    return True


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _logloss(y, p):
    return -(y * math.log(_clip(p)) + (1 - y) * math.log(1 - _clip(p)))


def _binary(outcomes):
    return all(o in (0, 0.0, 1, 1.0) for o in outcomes)


def _loss(y, p, binary):
    return _logloss(y, p) if binary else abs(y - p)      # log-loss for events; MAE for a continuous share


def backtest(questions, forecast_fn, *, check_asof=True) -> dict:
    """`forecast_fn(question) -> p`. Returns per-model mean loss + SKILL vs each baseline (1 − loss/loss_base;
    >0 means the forecaster beat that baseline). Also a Brier/RMSE secondary and per-baseline win-rate."""
    qs = list(questions)
    if check_asof:
        assert_asof(qs)
    outcomes = [q.outcome for q in qs]
    binary = _binary(outcomes)
    preds = [forecast_fn(q) for q in qs]
    n = len(qs)

    def mean_loss(ps):
        return sum(_loss(q.outcome, p, binary) for q, p in zip(qs, ps)) / n if n else float("nan")

    model_loss = mean_loss(preds)
    base_names = sorted({k for q in qs for k in q.baselines})
    skill, base_loss, winrate = {}, {}, {}
    for name in base_names:
        bp = [q.baselines.get(name, q.baselines.get("base_rate", 0.5)) for q in qs]
        bl = mean_loss(bp)
        base_loss[name] = round(bl, 5)
        skill[name] = round(1 - model_loss / bl, 4) if bl > 1e-12 else 0.0
        winrate[name] = round(sum(1 for q, p, b in zip(qs, preds, bp)
                                  if _loss(q.outcome, p, binary) < _loss(q.outcome, b, binary)) / n, 4)
    rmse = (sum((q.outcome - p) ** 2 for q, p in zip(qs, preds)) / n) ** 0.5 if n else float("nan")
    return {"n": n, "metric": "log_loss" if binary else "mae", "model_loss": round(model_loss, 5),
            "rmse": round(rmse, 5), "baseline_loss": base_loss, "skill_vs": skill, "winrate_vs": winrate,
            "beats_all_baselines": all(s > 0 for s in skill.values()) if skill else None}


def persistence_baseline(prev_value):
    """The default skeptic's forecast: the last observed value carries forward (a martingale/momentum floor)."""
    return prev_value
