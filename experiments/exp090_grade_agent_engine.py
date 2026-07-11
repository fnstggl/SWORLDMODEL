"""EXP-090: grade the agent engine on resolved ForecastBench questions — leak-free, then earn a grade.

Runs the grounded-agent engine as-of the round's due date, on RESOLVED political/social questions, feeding
only the frozen as-of context (no live web) — a real no-cheat backtest, not a lookup. Scores against 0.5
and the sample class rate, records the class grade + fitted shrink into models/agent_engine_grades.json,
and prints the scoreboard. This is the machine that lets a forecast eventually ship WITHOUT the ungraded
flag — and that stamps an F if the engine can't beat the skeptic.

Round 2025-06-08 is chosen because its due date is past DeepSeek's ~mid-2024 training cutoff (limits
recall leakage) and its questions have resolved by today (gradeable now).

Run: DEEPSEEK_API_KEY=... python -m experiments.exp090_grade_agent_engine
"""
from __future__ import annotations

import json
from pathlib import Path

RESULT = "experiments/results/exp090_grade_agent_engine.json"
# three post-cutoff resolved rounds → n≈33 in-domain, with the YES cases (Sherrill/NJ, Støre/Norway,
# Gaza ceasefire, Platner/Maine) as the discriminating items the engine must catch to beat the base rate.
ROUNDS = ["2025-06-08", "2025-08-31", "2025-10-26"]
LIMIT_PER_ROUND = 15


def run():
    from swm.engine.calibrate import GradeRegistry
    from swm.engine.front_door import agent_world_model
    from swm.eval.grade_agent_engine import grade_pooled, score_round

    wm = agent_world_model(branches=2, max_rounds=1)      # 1 dated round is enough for as-of event readout
    registry = GradeRegistry()
    scored_rounds = []
    for due in ROUNDS:
        print(f"\n=== scoring round {due} (leak-free, frozen as-of context) ===")
        r = score_round(wm, due, limit=LIMIT_PER_ROUND)
        print(f"  round {due}: {r['n_domain']} in-domain, {len(r['preds'])} scored, "
              f"{r['n_abstained']} abstained")
        scored_rounds.append(r)

    rep = grade_pooled(scored_rounds, registry=registry)
    bt = rep.backtest
    print(f"\n===== POOLED GRADE (n={rep.n_scored}, {rep.n_abstained} abstained) =====")
    if bt:
        print(f"  class base rate (fraction YES): {bt.get('class_rate')}")
        print(f"  engine log-loss: {bt['model_loss']}  |  baselines: {bt['baseline_loss']}")
        print(f"  SKILL vs 0.5: {bt['skill_vs'].get('base_rate')}  |  "
              f"SKILL vs class-rate (the real bar): {bt['skill_vs'].get('class_rate')}")
        print(f"  win-rate vs class-rate: {bt['winrate_vs'].get('class_rate')}")
        print(f"  GRADE: {rep.grade_entry.get('grade')}  (brier={rep.grade_entry.get('brier')}, "
              f"fitted shrink={rep.grade_entry.get('shrink')})")

    out = {"rounds": ROUNDS, "backtest": bt, "grades": registry.grades, "items": rep.items,
           "caveats": [
               "Leak-free on the RETRIEVAL door: frozen as-of context only, no live web.",
               "Training-recall door mitigated (round due-date past DeepSeek ~mid-2024 cutoff), not "
               "airtight — the only airtight test forecasts questions resolving in the future.",
               "Event-market framing (binary yes/no); dedicated voter-behavior election grading with "
               "as-of polling is the next track.",
               "Small n — a first calibration read, not a final grade; grow n across rounds."]}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    print(f"wrote grades -> {registry.path}: {json.dumps(registry.grades)}")
    return out


if __name__ == "__main__":
    run()
