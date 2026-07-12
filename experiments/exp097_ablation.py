"""EXP-097: the decisive ablation — does the society simulation beat same-model + same-evidence?

Runs all 5 arms (swm/eval/ablation) on resolved, leak-free DELIBERATION questions — the class the panel is
supposed to add value on (contests route to parametric in the full engine anyway, so they'd test nothing).
Leak-free: ForecastBench political rounds forecast as-of their due date, post-cutoff, bounded before/after
grounding. Prints the per-arm scoreboard and the head-to-head FULL vs EVIDENCE — the thesis test.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp097_ablation [limit_per_round]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULT = "experiments/results/exp097_ablation.json"
ROUNDS = ["2025-06-08", "2025-06-22", "2025-08-03", "2025-08-17", "2025-08-31", "2025-10-26"]


def run(limit_per_round=12):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.front_door import agent_world_model
    from swm.engine.retrieval import asof_search_fn
    from swm.engine.router import ParadigmRouter
    from swm.eval.ablation import predict_all_arms, score_arms
    from swm.eval.forecastbench import load_round
    from swm.eval.grade_agent_engine import is_domain

    wm = agent_world_model(branches=2, max_rounds=1)
    llm_raw = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                              max_tokens=200, temperature=0.3)
    router = ParadigmRouter(llm=None)
    rows, seen = [], set()
    for due in ROUNDS:
        qs = [q for q in load_round(due)
              if is_domain(q.meta.get("question", ""))
              and router.binary_kind(q.meta["question"]) == "deliberation"]
        class_rate = (sum(q.outcome for q in qs) / len(qs)) if qs else 0.5
        as_of_ts = time.mktime(time.strptime(due, "%Y-%m-%d"))
        for q in qs[:limit_per_round]:
            text = q.meta["question"]
            if text[:80] in seen:
                continue
            seen.add(text[:80])
            arms = predict_all_arms(wm, text, as_of=due, class_rate=round(class_rate, 3),
                                    search_fn=asof_search_fn(as_of_ts), llm_raw=llm_raw)
            arms["outcome"] = q.outcome
            arms["question"] = text[:90]
            rows.append(arms)
            got = {a: (round(arms[a], 2) if arms.get(a) is not None else "—")
                   for a in ("full", "raw", "evidence", "base_rate", "parametric")}
            print(f"  y={q.outcome:.0f} full={got['full']} raw={got['raw']} ev={got['evidence']} "
                  f"base={got['base_rate']} param={got['parametric']}  {text[:46]}")

    sb = score_arms(rows)
    print(f"\n===== ABLATION (n={len(rows)} deliberation questions, leak-free) =====")
    print(f"  {'arm':10s} {'n':>3s} {'brier':>7s} {'brier_cal':>9s} {'logloss':>8s} {'dir':>5s} "
          f"{'ece':>6s} {'decis':>6s} {'abst':>4s}")
    for a, s in sb["arms"].items():
        if s["n"]:
            print(f"  {a:10s} {s['n']:>3d} {s['brier']:>7.4f} {s['brier_cal']:>9.4f} {s['logloss']:>8.4f} "
                  f"{s['direction']:>5.2f} {s['ece']:>6.3f} {s['decision_value']:>6.2f} {s['n_abstain']:>4d}")
    h = sb["head_to_head_full_vs_evidence"]
    if h:
        print(f"\n  THESIS TEST — FULL vs EVIDENCE (same model, same evidence), n={h['n_both']}:")
        print(f"    Brier full {h['brier_full']} vs evidence {h['brier_evidence']}  "
              f"(full−evidence {h['full_minus_evidence']:+.4f}; full better on {h['full_wins_rows']:.0%} of rows)")
        print(f"    VERDICT: simulation {'ADDS value' if h['full_better'] else 'does NOT beat'} "
              f"the single grounded call.")

    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps({"n": len(rows), "scoreboard": sb, "rows": rows}, indent=1))
    print(f"\nwrote {RESULT}")
    return sb


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
