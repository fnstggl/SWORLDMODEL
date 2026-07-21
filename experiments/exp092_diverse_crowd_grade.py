"""EXP-092: grade the (improved) agent engine vs the crowd on a DIVERSE, cleaned, multi-domain set.

Iteration harness for the calibration/fidelity program. Grades on swm/eval/crowd_sets.diverse_set (elections,
sports, tech, culture, econ, ... — a general social world model, not political-only), leak-free via bounded
as-of Google News, and reports BOTH the raw skill vs crowd AND the out-of-sample (cross-fit temperature)
recalibrated skill — so we can see whether each engine tweak actually moves the honest number.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp092_diverse_crowd_grade [N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULT = "experiments/results/exp092_diverse_crowd_grade.json"


def run(limit=50, per_category=12, min_crowd=25):
    from swm.engine.calibrate import GradeRegistry
    from swm.engine.front_door import agent_world_model
    from swm.eval.crowd_sets import diverse_set, summarize
    from swm.eval.forecasting_corpus import load_corpus
    from swm.eval.grade_vs_crowd import grade_vs_crowd

    items = diverse_set(load_corpus(), per_category=per_category, min_crowd=min_crowd)[:limit]
    print("diverse set:", json.dumps(summarize(items)))
    wm = agent_world_model(branches=2, max_rounds=1)
    reg = GradeRegistry()
    rep = grade_vs_crowd(wm, items, limit=limit, registry=reg)

    sb = rep.scoreboard or {}
    ov, recal = sb.get("overall"), sb.get("recalibrated")
    unc = (sb.get("by_crowd_confidence", {}) or {}).get("uncertain(.35-.65)")
    print(f"\n===== DIVERSE CROWD GRADE (scored={rep.n_scored}, abstained={rep.n_abstained}) =====")
    if rep.direction:
        print(f"  DIRECTION: side-correct {rep.direction['side_correct']}  "
              f"agrees-with-crowd-side {rep.direction['agrees_with_crowd_side']}  (was 0.53 at the F baseline)")
    if ov:
        print(f"  Brier: model {ov['brier_model']}  vs crowd {ov['brier_crowd']}")
        print(f"  RAW    skill vs crowd {ov['skill_vs_crowd']}   skill vs base {ov['skill_vs_base']}")
    if recal:
        print(f"  RECAL  skill vs crowd {recal.get('skill_vs_crowd')}   (T={recal.get('temperature')}, "
              f"crossfit logloss {recal['crossfit'].get('logloss_before')}→{recal['crossfit'].get('logloss_after')})")
    if unc:
        print(f"  crowd-UNSURE slice n={unc['n']}: model brier {unc['brier_model']} vs crowd {unc['brier_crowd']}"
              f"  skill {unc['skill_vs_crowd']}")
    if sb.get("by_category"):
        print("  by category (skill vs crowd):")
        for c, s in sb["by_category"].items():
            if s:
                print(f"    {c:12s} n={s['n']:2d}  brier {s['brier_model']} vs {s['brier_crowd']}  skill {s['skill_vs_crowd']}")
    print(f"  GRADE: {rep.grade_entry.get('grade')}  (T={rep.grade_entry.get('temperature')})")

    out = {"summary": summarize(items), "scoreboard": sb, "grade": rep.grade_entry,
           "direction": rep.direction, "n_scored": rep.n_scored, "n_abstained": rep.n_abstained,
           "rows": rep.rows}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    return out


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
