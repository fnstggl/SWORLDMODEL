"""EXP-088: end-to-end — an arbitrary question auto-grounds and simulates, no hand-wiring.

The capstone. `general_world_model()` is the single front door: a user asks ANYTHING → DeepSeek compiles it into
a structural spec → the spec's high-leverage variable VALUES are auto-grounded from live evidence (the general
DeepSeek+web router — no feeds) → the calibrated Monte-Carlo runs on the measured world. This runs that whole
pipeline live on real questions and prints, for each: the compiled mechanism, which variables got grounded and
from where (with their measured value + CI), and the forecast.

Live smoke (needs DEEPSEEK_API_KEY + network; skips gracefully). The point is the end-to-end wiring and the
grounding provenance, not a fixed number.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp088_end_to_end
"""
from __future__ import annotations

import json
from pathlib import Path

RESULT = "experiments/results/exp088_end_to_end.json"

QUESTIONS = [
    "Will the US unemployment rate be above 4.5% twelve months from now?",
    "Will Bitcoin be above $80,000 by the end of next quarter?",
    "Will US CPI inflation be below 3% at the end of this year?",
]


def run() -> dict:
    from swm.api.world_model import general_world_model
    from swm.api.deepseek_backend import default_chat_fn
    if default_chat_fn() is None:
        print("EXP-088  end-to-end — SKIPPED (no DEEPSEEK_API_KEY / HF_TOKEN configured)")
        return {"skipped": "no LLM backend"}

    wm = general_world_model(n=4000)                             # compile + auto-ground + run, all live
    out = []
    for q in QUESTIONS:
        try:
            res = wm.simulate(q)
        except Exception as e:
            out.append({"question": q, "error": str(e)[:120]})
            continue
        g = res.get("grounding") or {}
        grounded_vars = [{"var": r["var"], "value": r.get("value"), "sd": r.get("sd"), "source": r.get("source")}
                         for r in (g.get("detail") or []) if r.get("grounded")]
        out.append({"question": q, "mechanism": res["mechanism"],
                    "grounded": g.get("grounded"), "n_high_leverage": g.get("n_high_leverage"),
                    "grounded_vars": grounded_vars, "headline": res["headline"]})

    res_all = {"pipeline": "question -> DeepSeek compile -> auto-ground high-leverage vars (general router) -> run",
               "questions": out}
    Path(RESULT).write_text(json.dumps(res_all, indent=1))

    print("EXP-088  end-to-end: ask anything -> compile -> AUTO-GROUND -> simulate (all live, general engine)")
    for r in out:
        if "error" in r:
            print(f"\n  Q: {r['question']}\n     (error: {r['error']})")
            continue
        print(f"\n  Q: {r['question']}")
        print(f"     mechanism: {r['mechanism']}  |  grounded {r['grounded']}/{r['n_high_leverage']} "
              f"high-leverage vars from live evidence:")
        for gv in r["grounded_vars"]:
            print(f"       - {gv['var']:28s} = {gv['value']}  ±{gv['sd']}  [{gv['source']}]")
        print(f"     {r['headline']}")
    print(f"\n  wrote {RESULT}")
    return res_all


if __name__ == "__main__":
    run()
