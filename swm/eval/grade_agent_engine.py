"""Grade the agent engine on resolved ForecastBench questions — leak-free, no-cheat, then record the grade.

The honest backtest. For a question that has ALREADY resolved we still must forecast it as if we were
standing before the resolution date, or we are grading a lookup, not a forecast. Two leakage doors are
shut here:
  1. LIVE-RETRIEVAL LEAK — closed by feeding the engine the question's FROZEN as-of context (ForecastBench's
     `background` + `resolution_criteria`, written at question creation) via `evidence=`, and NEVER touching
     the live web (which today would return the answer).
  2. TRAINING-RECALL LEAK — mitigated by using rounds whose due date is AFTER the model's training cutoff
     (DeepSeek V3 ~mid-2024), so the model cannot simply remember the outcome. (Not a perfect guarantee —
     stated as a caveat in the report; the only airtight version forecasts questions that resolve in the
     FUTURE, which is the live-and-wait track.)

Flow: load a resolved round (swm/eval/forecastbench.py) → filter to the engine's domain (social/political
event questions) → for each, `wm.simulate(question, evidence=frozen, as_of=due_date, binary=True)` → p(yes)
→ score every forecaster + baselines on identical items via `event_backtest.backtest` (log-loss, Brier,
SKILL vs base rate) → `GradeRegistry.record` writes the grade + a fitted shrink for the class. Abstentions
are reported, not scored (a refused question is not a wrong forecast).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from swm.engine.calibrate import GradeRegistry
from swm.engine.retrieval import Passage
from swm.eval.event_backtest import Question, backtest
from swm.eval.forecastbench import load_round

# the engine is a SOCIAL world model — grade it where agents-interacting is the right mechanism
# (elections, appointments, institutional/geopolitical decisions), not sports/markets/awards.
_DOMAIN = re.compile(r"\b(election|primary|nominee|nomination|president|senate|governor|congress|"
                     r"parliament|prime minister|mayor|referendum|resign|elected|ballot|candidate|"
                     r"shutdown|appoint|confirm|coalition|budget|impeach|cabinet|approval rating|"
                     r"ceasefire|treaty|sanction|deploy|withdraw)\b", re.I)
_OFF_DOMAIN = re.compile(r"\b(win the .*(World Series|Super Bowl|NBA|MVP|Emmy|Oscar|Nobel|Cup|Finals|"
                         r"championship|Grand Prix)|box office|film|movie|album|rotten tomatoes)\b", re.I)


def is_domain(q: str) -> bool:
    return bool(_DOMAIN.search(q)) and not _OFF_DOMAIN.search(q)


def frozen_evidence(q: Question) -> list:
    """The as-of context the engine is allowed to see — the question's own frozen background + rules.
    Nothing dated after creation; nothing from the live web."""
    m = q.meta
    ev = []
    if m.get("background"):
        ev.append(Passage(f"Background (as of {q.asof}): {m['background']}", "forecastbench:background",
                          q.asof))
    if m.get("resolution_criteria"):
        ev.append(Passage(f"Resolution criteria: {m['resolution_criteria']}", "forecastbench:criteria",
                          q.asof))
    if m.get("freeze_value") is not None:
        ev.append(Passage(f"Value at freeze ({q.asof}): {m['freeze_value']}", "forecastbench:freeze",
                          q.asof))
    return ev


def p_yes(res: dict) -> float:
    """Map an engine result to P(the event resolves YES). binary=True gives {'yes','no'}; be tolerant."""
    d = res.get("distribution") or {}
    if not d:
        return None
    for k in d:
        if str(k).lower() in ("yes", "happens", "true", "responds", "1"):
            return float(d[k])
    if len(d) == 2:                                        # two-option: yes = 1 - the no-ish option
        for k in d:
            if str(k).lower() in ("no", "does_not_respond", "false", "0"):
                return float(1 - d[k])
    return float(max(d.values()))                         # single dominant option


@dataclass
class GradeReport:
    question_class: str
    n_scored: int = 0
    n_abstained: int = 0
    n_domain: int = 0
    backtest: dict = field(default_factory=dict)
    grade_entry: dict = field(default_factory=dict)
    items: list = field(default_factory=list)


def score_round(wm, due_date: str, *, limit=20, verbose=True) -> dict:
    """Forecast every in-domain resolved question leak-free. Returns raw {preds, outcomes, items,
    n_domain, n_abstained} — no grading (so several rounds can be POOLED into one grade)."""
    qs = [q for q in load_round(due_date) if is_domain(q.meta.get("question", ""))]
    preds, ys, items, n_abstained = [], [], [], 0
    for q in qs[:limit]:
        text = q.meta["question"]
        try:
            res = wm.simulate(text, evidence=frozen_evidence(q), as_of=due_date, binary=True)
        except Exception as e:
            res = {"abstain": True, "abstain_reason": f"{type(e).__name__}: {str(e)[:80]}"}
        if res.get("abstain") or p_yes(res) is None:
            n_abstained += 1
            items.append({"q": text[:90], "outcome": q.outcome, "p": None,
                          "abstain": res.get("abstain_reason", "no distribution")[:80]})
            if verbose:
                print(f"  ABSTAIN  y={q.outcome:.0f}  {text[:78]}")
            continue
        p = p_yes(res)
        preds.append(p)
        ys.append(q.outcome)
        items.append({"q": text[:90], "outcome": q.outcome, "p": round(p, 3), "as_of": due_date})
        if verbose:
            hit = "✓" if (p > 0.5) == (q.outcome > 0.5) else "✗"
            print(f"  {hit} p={p:.2f}  y={q.outcome:.0f}  {text[:76]}")
    return {"preds": preds, "outcomes": ys, "items": items, "n_domain": len(qs),
            "n_abstained": n_abstained}


def grade_pooled(rounds_scored: list, *, question_class="society:event",
                 registry: GradeRegistry = None) -> GradeReport:
    """Pool scored questions across rounds → one grade + fitted shrink. Beating the sample CLASS RATE
    (not just 0.5) is the bar: it takes discrimination, not 'these are longshots, guess low'."""
    registry = registry or GradeRegistry()
    preds = [p for r in rounds_scored for p in r["preds"]]
    ys = [y for r in rounds_scored for y in r["outcomes"]]
    items = [it for r in rounds_scored for it in r["items"]]
    rep = GradeReport(question_class=question_class, n_scored=len(preds), items=items,
                      n_abstained=sum(r["n_abstained"] for r in rounds_scored),
                      n_domain=sum(r["n_domain"] for r in rounds_scored))
    if not preds:
        return rep
    class_rate = sum(ys) / len(ys)
    scored = [Question(qid=f"q{i}", outcome=y, baselines={"base_rate": 0.5, "class_rate": class_rate})
              for i, y in enumerate(ys)]
    idx = {q.qid: p for q, p in zip(scored, preds)}
    rep.backtest = backtest(scored, lambda qq: idx[qq.qid], check_asof=False)
    rep.backtest["class_rate"] = round(class_rate, 3)
    conservative = {"skill_vs": {"class_rate": rep.backtest["skill_vs"].get("class_rate", 0.0)},
                    "rmse": rep.backtest["rmse"]}
    rep.grade_entry = registry.record(question_class, backtest_report={**conservative, "n": len(preds)},
                                      preds=preds, outcomes=ys)
    return rep


def grade_round(wm, due_date: str, *, question_class="society:event", limit=20,
                registry: GradeRegistry = None, verbose=True) -> GradeReport:
    """Single-round convenience wrapper (score then grade that round alone)."""
    r = score_round(wm, due_date, limit=limit, verbose=verbose)
    return grade_pooled([r], question_class=question_class, registry=registry)


