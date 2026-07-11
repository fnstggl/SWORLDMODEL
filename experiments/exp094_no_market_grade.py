"""EXP-094 (Lever 4): grade on NO-LIQUID-MARKET questions — the real product use case.

Beating a liquid market is a hard bar even for frontier LLMs (they mostly match it). But the product's
actual value is on questions WITHOUT a market — who wins before a market forms, will-X-happen where nobody
is betting. There the honest bar is the BASE RATE, and the metric that matters is DIRECTION (did we get the
side right) and skill vs the base/class rate. This grades the improved engine (all levers) on resolved
ForecastBench political/social questions with the engine's own multi-round as-of grounding, leak-free.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp094_no_market_grade
"""
from __future__ import annotations

import json
from pathlib import Path

RESULT = "experiments/results/exp094_no_market_grade.json"
ROUNDS = ["2025-06-08", "2025-08-31", "2025-10-26"]        # resolved, post-cutoff, no liquid market attached
LIMIT_PER_ROUND = 15


def run():
    from swm.engine.calibrate import GradeRegistry
    from swm.engine.front_door import agent_world_model
    from swm.eval.grade_agent_engine import grade_pooled, score_round

    wm = agent_world_model(branches=2, max_rounds=1)       # event_engine=panel + all levers by default
    registry = GradeRegistry()
    scored = []
    for due in ROUNDS:
        print(f"\n=== scoring {due} (no-market, as-of grounded, leak-free) ===")
        r = score_round(wm, due, limit=LIMIT_PER_ROUND, use_asof_search=True)
        print(f"  {due}: {r['n_domain']} in-domain, {len(r['preds'])} scored, {r['n_abstained']} abstained")
        scored.append(r)

    rep = grade_pooled(scored, question_class="deliberation:no_market", registry=registry)
    bt = rep.backtest
    print(f"\n===== NO-MARKET GRADE (n={rep.n_scored}, abstained={rep.n_abstained}) =====")
    if bt:
        print(f"  DIRECTION side-correct: {bt.get('direction', {}).get('side_correct')}  (0.53 = coin flip)")
        print(f"  class base rate: {bt.get('class_rate')}")
        print(f"  engine loss {bt['model_loss']}  vs base {bt['baseline_loss']}")
        print(f"  SKILL vs 0.5: {bt['skill_vs'].get('base_rate')}   vs class-rate: {bt['skill_vs'].get('class_rate')}")
        print(f"  GRADE: {rep.grade_entry.get('grade')}")
    out = {"rounds": ROUNDS, "backtest": bt, "grade": rep.grade_entry, "n_scored": rep.n_scored,
           "n_abstained": rep.n_abstained, "items": rep.items,
           "note": "no-market use case: resolved social questions, no crowd price; bar is base rate + direction."}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
