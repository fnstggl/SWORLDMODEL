"""PART C operator — forward-lock every ablation arm on OPEN questions, then resolve + score later.

The contamination-proof standing validation: for each forward (still-open) question, compute B0/B1/B2 always,
add B3 on a stratified ~20%, add B4-B9 on a diagnostic ~8%, and LOCK them all into the append-only, versioned
forward ledger BEFORE the world resolves. Weeks later, `resolve` fills outcomes and `score` reports per-arm
accuracy + the marginal ladder + per-class best architecture — the honest answer to "does simulation earn its
keep", on data no model could have trained on.

  lock   --round <YYYY-MM-DD>   forward-lock arms for a round's open questions (leak-free as-of that date)
  resolve                        auto-resolve due locked rows against current news (cited, conservative)
  score                          per-arm scores + marginals + per-class best architecture from resolved rows
  status                         ledger stats (locked/resolved, versions)

Usage:  python -m experiments.flywheel_forward <cmd> [args]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _led():
    from swm.engine.forward_ledger import ForwardLedger
    return ForwardLedger()


def lock(round_date, limit, tier_frac2=0.20, tier_frac3=0.08):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.front_door import agent_world_model
    from swm.engine.retrieval import asof_search_fn
    from swm.engine.router import ParadigmRouter
    from swm.eval.forecastbench import load_round
    from swm.eval.grade_agent_engine import is_domain
    from swm.eval.tiered_ablation import predict_arms, stratify_sample

    led = _led()
    wm = agent_world_model(branches=2, max_rounds=2)
    llm_raw = default_chat_fn(system="You are a careful forecaster. Reply ONLY compact JSON.",
                              max_tokens=200, temperature=0.3)
    router = ParadigmRouter(llm=None)
    qs = [q for q in load_round(round_date)
          if is_domain(q.meta.get("question", ""))
          and router.binary_kind(q.meta["question"]) == "deliberation"][:limit]
    class_rate = round((sum(q.outcome for q in qs) / len(qs)) if qs else 0.5, 3)
    items = [{"q": q, "text": q.meta["question"]} for q in qs]
    key = lambda it: round_date                                 # single round → stratify by outcome unknown; uniform
    t2 = stratify_sample(items, key, tier_frac2, seed=7)
    t3 = stratify_sample(items, key, tier_frac3, seed=11)
    as_of_ts = time.mktime(time.strptime(round_date, "%Y-%m-%d"))
    locked = 0
    for i, it in enumerate(items):
        tier = 3 if i in t3 else (2 if i in t2 else 1)
        arms = predict_arms(wm, it["text"], as_of=round_date, class_rate=class_rate, tier=tier,
                            search_fn=asof_search_fn(as_of_ts), llm_raw=llm_raw)
        led.lock_from_prediction(
            arms, question=it["text"], question_class="society:event", domain="deliberation",
            outcome_space={"type": "binary", "options": ["yes", "no"]},
            resolution_criteria="ForecastBench resolution", resolve_by="",
            config={"branches": wm.branches, "max_rounds": wm.max_rounds, "panel_reps": wm.panel_reps,
                    "tier": tier},
            selected_architecture="panel", router_explanation="deliberation→panel")
        locked += 1
        print(f"  [T{tier}] locked {sum(1 for a in arms if not a.startswith('_') and arms[a].get('p') is not None)} "
              f"arms  {it['text'][:56]}")
    print(f"locked {locked} forward rows from round {round_date} → {led.path}")


def resolve(limit):
    from swm.api.deepseek_backend import default_chat_fn
    from swm.engine.grounding import parse_json
    from swm.engine.retrieval import multi_search
    led = _led()
    llm = default_chat_fn(system="Reply ONLY compact JSON.", max_tokens=200, temperature=0.0)
    if llm is None:
        print("no LLM backend"); return
    checked = resolved = 0
    for r in led.open_rows()[:limit]:
        checked += 1
        passages = multi_search([r.question, f"{r.question} result outcome"], 6)
        if len(passages) < 2:
            continue
        ptxt = "\n".join(p.cite() for p in passages[:16])
        v = parse_json(llm(f"QUESTION: {r.question}\nCURRENT EVIDENCE:\n{ptxt}\n\nHas the outcome been decided?"
                           f' Be conservative. Return ONLY JSON: {{"resolved":<bool>,"outcome":"yes"|"no"|null}}')) or {}
        if v.get("resolved") and v.get("outcome") in ("yes", "no"):
            led.resolve(r.qid, 1.0 if v["outcome"] == "yes" else 0.0, source="auto")
            resolved += 1
    print(json.dumps({"checked": checked, "resolved": resolved}))


def score():
    led = _led()
    sc = led.score(min_n=8)
    print(json.dumps(sc, indent=1, default=str)[:4000])
    Path("experiments/results/forward_ledger_score.json").write_text(json.dumps(sc, indent=1, default=str))


def status():
    led = _led()
    rows = led.load()
    from collections import Counter
    print(f"ledger: {led.path}  |  {len(rows)} rows "
          f"({sum(1 for r in rows if r.status=='locked')} locked, "
          f"{sum(1 for r in rows if r.status=='resolved')} resolved)")
    print("versions:", dict(Counter(r.lock_version for r in rows)))
    print("by class:", dict(Counter(r.question_class for r in rows)))


def main():
    ap = argparse.ArgumentParser(description="Forward-locked multi-arm ledger operator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lock"); lk.add_argument("--round", required=True); lk.add_argument("--limit", type=int, default=20)
    rs = sub.add_parser("resolve"); rs.add_argument("--limit", type=int, default=40)
    sub.add_parser("score")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "lock": lock(a.round, a.limit)
    elif a.cmd == "resolve": resolve(a.limit)
    elif a.cmd == "score": score()
    elif a.cmd == "status": status()


if __name__ == "__main__":
    main()
