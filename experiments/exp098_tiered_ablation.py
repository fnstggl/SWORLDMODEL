"""EXP-098: the tiered ablation — isolate the MARGINAL value of each layer, practically.

Fixes the pilot (EXP-097): adds the call-matched grounded ENSEMBLE (B3) that separates simulation-value from
ordinary averaging, plus the real stakeholder-simulation arms (B5 independent / B6 interacting) that the
deliberation benchmark never actually ran (see docs/AUDIT_PART_A_WIRING.md). Tiering keeps it affordable:

  * Tier 1 (B0/B1/B2) on EVERY question.
  * Tier 2 (+B3) on a stratified ~20%.
  * Tier 3 (+B4..B9) on a stratified ~8% diagnostic sample.

Leak-free: ForecastBench political DELIBERATION rounds, forecast as-of the due date, bounded before/after
grounding, post-cutoff. Prints per-arm scores, the marginal-effect ladder (with paired bootstrap CI +
permutation p), and mean spend per arm.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp098_tiered_ablation [limit_per_round]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULT = "experiments/results/exp098_tiered_ablation.json"
ROUNDS = ["2025-06-08", "2025-06-22", "2025-08-03", "2025-08-17", "2025-08-31", "2025-10-26"]


def run(limit_per_round=12):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.front_door import agent_world_model
    from swm.engine.retrieval import asof_search_fn
    from swm.engine.router import ParadigmRouter
    from swm.eval.forecastbench import load_round
    from swm.eval.grade_agent_engine import is_domain
    from swm.eval.tiered_ablation import predict_arms, report_marginals, stratify_sample

    wm = agent_world_model(branches=2, max_rounds=2)
    llm_raw = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                              max_tokens=200, temperature=0.3)
    llm_hot_raw = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                                  max_tokens=200, temperature=0.7)
    router = ParadigmRouter(llm=None)

    # collect the leak-free deliberation questions across rounds
    items, seen = [], set()
    for due in ROUNDS:
        qs = [q for q in load_round(due)
              if is_domain(q.meta.get("question", ""))
              and router.binary_kind(q.meta["question"]) == "deliberation"]
        class_rate = round((sum(q.outcome for q in qs) / len(qs)) if qs else 0.5, 3)
        for q in qs[:limit_per_round]:
            text = q.meta["question"]
            if text[:80] in seen:
                continue
            seen.add(text[:80])
            items.append({"due": due, "q": q, "text": text, "class_rate": class_rate,
                          "domain": "deliberation"})

    # stratify: Tier 2 on ~20%, Tier 3 on ~8%, stratified by (round, outcome) so both are representative
    key = lambda it: (it["due"], int(it["q"].outcome))
    tier2 = stratify_sample(items, key, 0.20, seed=7)
    tier3 = stratify_sample(items, key, 0.08, seed=11)

    runs = []
    for i, it in enumerate(items):
        tier = 3 if i in tier3 else (2 if i in tier2 else 1)
        as_of_ts = time.mktime(time.strptime(it["due"], "%Y-%m-%d"))
        arms = predict_arms(wm, it["text"], as_of=it["due"], class_rate=it["class_rate"], tier=tier,
                            search_fn=asof_search_fn(as_of_ts), llm_raw=llm_raw, llm_hot_raw=llm_hot_raw)
        arms["outcome"] = it["q"].outcome
        arms["question"] = it["text"][:90]
        arms["tier"] = tier
        runs.append(arms)
        g = lambda a: (f"{arms[a]['p']:.2f}" if (a in arms and arms[a].get("p") is not None) else "—")
        print(f"  [T{tier}] y={it['q'].outcome:.0f} b0={g('base_rate')} b1={g('grounded_1shot')} "
              f"b2={g('full')} b3={g('grounded_ens')} b4={g('generic_panel')} "
              f"b5={g('indep_stake')} b6={g('interact_stake')}  {it['text'][:40]}")

    rep = report_marginals(runs)
    print(f"\n===== TIERED ABLATION (n={rep['n']} deliberation, leak-free) =====")
    print(f"  {'arm':15s} {'n':>3s} {'brier':>7s} {'brier_cal':>9s} {'dir':>5s} {'ece':>6s} "
          f"{'calls':>6s} {'cost$':>7s}")
    for a, s in rep["arms"].items():
        if s["n"]:
            sp = rep["spend"].get(a, {})
            print(f"  {a:15s} {s['n']:>3d} {s['brier']:>7.4f} {s['brier_cal']:>9.4f} {s['direction']:>5.2f} "
                  f"{s['ece']:>6.3f} {sp.get('mean_calls', 0):>6.1f} {sp.get('mean_cost_usd', 0):>7.4f}")
    print("\n  MARGINAL EFFECTS (paired Brier diff; negative ⇒ hi better; p=permutation):")
    for m in rep["marginals"]:
        if m.get("insufficient"):
            print(f"    {m['isolates']:52s} n={m['n_pairs']:>2d}  (insufficient paired n)")
        else:
            star = "  *" if (m["p_perm"] < 0.05 and m["hi_better"]) else ""
            print(f"    {m['isolates']:52s} n={m['n_pairs']:>2d}  Δ={m['mean_brier_diff']:+.4f} "
                  f"CI[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}] p={m['p_perm']:.3f} "
                  f"win%={m['hi_wins_rows']:.0%}{star}")

    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    # strip nothing — keep full per-row arm predictions + spend for the append-only record
    Path(RESULT).write_text(json.dumps({"n": rep["n"], "report": rep, "runs": runs}, indent=1, default=str))
    print(f"\nwrote {RESULT}")
    return rep


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
