"""EXP-089 (Stages 0-1): the large-scale no-cheat backtest — run the world-model over thousands of resolved
forecasting questions, score it against the crowd, with the parametric-leakage meter.

Stage 0: build the leakage-proof corpus (Manifold + Polymarket resolved binary markets, crowd prob at as-of,
cutoff-clean tags). Stage 1: for each item, compile-as-of -> simulate -> P(YES), plus the direct-LLM estimate
(leakage meter), and score Brier/log-loss + SKILL vs the crowd and base rate, sliced by category, crowd
confidence, and clean/dirty. Resumable: every LLM call is cached on disk, and results checkpoint incrementally.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp089_backtest [max_items]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.backtest_harness import direct_estimate, forecast_item, score
from swm.eval.forecasting_corpus import build_corpus, load_corpus

CORPUS = "experiments/results/backtest_corpus.json"
RESULT = "experiments/results/exp089_backtest.json"
PRED = "experiments/results/exp089_predictions.json"


def get_corpus(manifold=3500, polymarket=1500):
    if not Path(CORPUS).exists():
        print("  Stage 0: pulling corpus from Manifold + Polymarket ...")
        build_corpus(manifold=manifold, polymarket=polymarket, path=CORPUS)
    return load_corpus(CORPUS)


def run(max_items=1200, n=2500) -> dict:
    corpus = get_corpus()
    clean = [it for it in corpus if it.cutoff_clean]
    print(f"  corpus: {len(corpus)} items ({len(clean)} cutoff-clean)")
    items = clean[:max_items]

    llm_c = resilient_chat_fn(system="You compile questions into runnable structural simulations. Emit ONLY "
                                     "the JSON spec.", max_tokens=1100)
    llm_d = resilient_chat_fn(system="You are a calibrated forecaster. Reply with ONLY compact JSON.",
                              max_tokens=80)

    done = {}
    if Path(PRED).exists():
        done = {r["qid"]: r for r in json.loads(Path(PRED).read_text())}
    rows = list(done.values())
    todo = [it for it in items if it.qid not in done]

    def _work(it):
        p, meta = forecast_item(it, llm_c, n=n, force_readout=True)
        if p is None:
            return None
        d = direct_estimate(it, llm_d)
        return {"qid": it.qid, "category": it.category, "cutoff_clean": it.cutoff_clean,
                "outcome": it.outcome, "p_model": p, "p_crowd": it.crowd_prob, "p_direct": d,
                "mechanism": meta.get("mechanism"), "question": it.question[:120]}

    with ThreadPoolExecutor(max_workers=10) as ex:              # HF router handles concurrent requests
        for fut in as_completed([ex.submit(_work, it) for it in todo]):
            r = fut.result()
            if r is None:
                continue
            rows.append(r)
            if len(rows) % 25 == 0:
                Path(PRED).write_text(json.dumps(rows))
                print(f"    {len(rows)} scored  (llm hf={llm_c.calls['hf']+llm_d.calls['hf']} "
                      f"ds={llm_c.calls['deepseek']+llm_d.calls['deepseek']} cache={llm_c.calls['cache']+llm_d.calls['cache']})")
    Path(PRED).write_text(json.dumps(rows))

    scored = [r for r in rows if r["p_model"] is not None]
    res = score(scored)
    Path(RESULT).write_text(json.dumps(res, indent=1))

    o, c = res["overall"], res["clean"]
    print(f"\nEXP-089  backtest: {len(scored)} questions scored ({c['n'] if c else 0} clean)")
    print(f"  OVERALL  model log-loss {o['ll_model']} vs crowd {o['ll_crowd']} (skill_vs_crowd {o['skill_vs_crowd']}, "
          f"skill_vs_base {o['skill_vs_base']}); crowd's own skill_vs_base {o['crowd_skill_vs_base']}")
    if c:
        print(f"  CLEAN    model log-loss {c['ll_model']} vs crowd {c['ll_crowd']} (skill_vs_crowd {c['skill_vs_crowd']})"
              + (f"  | leakage meter: direct-LLM skill_vs_crowd {c.get('direct_skill_vs_crowd')}"
                 if c.get('direct_skill_vs_crowd') is not None else ""))
    print("  by category (skill vs crowd):")
    for cat, a in sorted(res["by_category"].items(), key=lambda kv: -(kv[1]["n"] if kv[1] else 0)):
        if a:
            print(f"    {cat:12s} n={a['n']:4d}  model_ll {a['ll_model']:.3f}  crowd_ll {a['ll_crowd']:.3f}  "
                  f"skill_vs_crowd {a['skill_vs_crowd']}")
    unc = res["by_crowd_confidence"]["uncertain(.35-.65)"]
    if unc:
        print(f"  where crowd UNSURE (.35-.65): n={unc['n']} skill_vs_crowd {unc['skill_vs_crowd']}")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 1200)
