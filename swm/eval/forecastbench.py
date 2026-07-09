"""ForecastBench loader — resolved, contamination-controlled questions to grade the agent engine on.

ForecastBench (Karger et al., ICLR 2025; CC BY-SA 4.0) publishes nightly question sets and resolution
sets: real forecasting questions (Metaculus/Manifold/Infer markets + real-data series like ACLED/FRED/
Wikipedia) with due dates and later ground-truth resolutions. That is exactly the fuel grade-or-abstain
needs: run the engine AS-OF the question set's due date, score against `resolved_to`, and record the grade
per question-class in the registry (swm/engine/calibrate.py). The as-of contract is ForecastBench's own
design (questions are published before resolution), so the leakage gate holds by construction — but we
still route through event_backtest.Question so `assert_asof` re-checks it.

  question set : datasets/question_sets/YYYY-MM-DD-llm.json          {"questions": [{id, source, question,
                 resolution_criteria, background, ...}]}
  resolutions  : datasets/resolution_sets/YYYY-MM-DD_resolution_set.json  {"resolutions": [{id, source,
                 resolution_date, resolved_to, resolved}]}
"""
from __future__ import annotations

import json
import urllib.request

from swm.eval.event_backtest import Question

_RAW = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"


def _fetch(path):
    with urllib.request.urlopen(f"{_RAW}/{path}", timeout=30) as r:
        return json.loads(r.read())


def load_round(due_date: str, *, binary_only=True) -> list:
    """One graded round: questions due `due_date` (YYYY-MM-DD) joined to their resolutions.
    Returns [event_backtest.Question] with meta carrying the question text + background, ready for
    `backtest(questions, forecast_fn)` where forecast_fn runs the agent engine as-of the due date."""
    qset = _fetch(f"question_sets/{due_date}-llm.json")
    rset = _fetch(f"resolution_sets/{due_date}_resolution_set.json")

    def key(i):                                # ids are strings, or lists for combo questions
        return json.dumps(i) if isinstance(i, list) else str(i)

    resolved = {key(r.get("id")): r for r in rset.get("resolutions", [])
                if r.get("resolved") and r.get("resolved_to") is not None}
    out = []
    for q in qset.get("questions", []):
        r = resolved.get(key(q.get("id")))
        if r is None:
            continue
        y = float(r["resolved_to"])
        if binary_only and y not in (0.0, 1.0):
            continue
        out.append(Question(
            qid=key(q["id"]), outcome=y, asof=due_date,
            resolved=str(r.get("resolution_date", "")),
            baselines={"base_rate": 0.5},
            meta={"question": q.get("question", ""), "source": q.get("source", ""),
                  "background": (q.get("background") or "")[:1200],
                  "resolution_criteria": (q.get("resolution_criteria") or "")[:600],
                  "freeze_value": q.get("freeze_datetime_value")}))
    return out
