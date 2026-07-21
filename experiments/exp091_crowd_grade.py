"""EXP-091: grade the agent engine vs the CROWD, leak-free, with keyless as-of Google News grounding.

The payoff of the as-of-retrieval fix. GDELT is rate-limited in this sandbox, but Google News RSS with a
bounded `before:<as_of> after:<window>` query + a hard pubDate<=as_of drop is a keyless as-of source that
works here and cannot leak a post-cutoff outcome (EXP: on the 2025 NJ race as-of 2025-10-20 it returns
Sep–Oct campaign coverage incl. a poll, zero outcome leakage). So the engine can finally be grounded on the
information the crowd had, and scored against the market — the bar ForecastBench / Prophet Arena use.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp091_crowd_grade
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RESULT = "experiments/results/exp091_crowd_grade.json"
_POL = re.compile(r"election|senate|governor|president|nominee|nomination|mayor|parliament|referend|"
                  r"congress|prime minister|coalition|impeach|confirm|primary", re.I)


def run(limit=20):
    from swm.engine.calibrate import GradeRegistry
    from swm.engine.front_door import agent_world_model
    from swm.eval.forecasting_corpus import load_corpus
    from swm.eval.grade_vs_crowd import grade_vs_crowd

    wm = agent_world_model(branches=2, max_rounds=1)
    items = [i for i in load_corpus() if i.cutoff_clean and _POL.search(i.question)]
    print(f"political, cutoff-clean corpus items: {len(items)}")
    reg = GradeRegistry()
    rep = grade_vs_crowd(wm, items, limit=limit, registry=reg)

    ov = rep.scoreboard.get("overall") if rep.scoreboard else None
    unc = (rep.scoreboard.get("by_crowd_confidence", {}) or {}).get("uncertain(.35-.65)") if rep.scoreboard else None
    print(f"\n===== CROWD GRADE (n_scored={rep.n_scored}, abstained={rep.n_abstained}) =====")
    if ov:
        print(f"  brier: model {ov['brier_model']}  vs crowd {ov['brier_crowd']}")
        print(f"  log-loss: model {ov['ll_model']}  vs crowd {ov['ll_crowd']}")
        print(f"  SKILL vs crowd: {ov['skill_vs_crowd']}   SKILL vs base: {ov['skill_vs_base']}")
        print(f"  (for reference, the crowd's own skill vs base: {ov.get('crowd_skill_vs_base')})")
        if unc:
            print(f"  on crowd-UNSURE items (.35-.65), n={unc['n']}: skill vs crowd {unc['skill_vs_crowd']}")
        print(f"  GRADE: {rep.grade_entry.get('grade')} (shrink {rep.grade_entry.get('shrink')})")

    out = {"n_scored": rep.n_scored, "n_abstained": rep.n_abstained, "scoreboard": rep.scoreboard,
           "grade": rep.grade_entry, "rows": rep.rows,
           "note": "leak-free via keyless as-of Google News (before:/after: window + pubDate<=as_of drop); "
                   "crowd = Manifold/Polymarket price at as-of; small n — a first real crowd read."}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
