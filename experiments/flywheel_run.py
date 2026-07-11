"""Operate the outcome flywheel — the compounding calibration loop.

The moat is not code, it is the accumulating stream of (forecast → real outcome) pairs and the per-domain
calibration it grinds out. This is the operator for that loop. The durable store is the git-committed log
(`data/forecast_log.jsonl`) + the registry (`models/agent_engine_grades.json`) — so the stream survives the
ephemeral container: each scheduled run pulls, appends, resolves, refits, commits, pushes.

Two modes, both leak-free:

  A) REPLAY (instant, retrospective) — forecast ALREADY-RESOLVED questions as-of their due date (leak-free
     via bounded before/after grounding + post-cutoff selection) and record the KNOWN outcome immediately.
     No waiting; primes/grows the calibration corpus today. `forecast --round <YYYY-MM-DD>` + `resolve` on
     the same known outcomes, or `prime` to backfill from existing best-config backtest results.

  B) LIVE FORWARD (the real compounding stream, with delay) — forecast a round whose questions are still
     OPEN, log them with a resolve_by date, WAIT for the world, then `resolve` (auto against current news)
     and `refit`. This is contamination-proof by construction and proprietary.

Subcommands:
  prime            backfill the log from committed best-config backtest results (instant bootstrap)
  forecast --round give the round's questions to the engine, log each forecast (open or replay)
  resolve          auto-resolve due open records against current news (cited, conservative)
  refit            recalibrate per-class + per-domain temperatures from the RESOLVED stream → live registry
  status           log stats (open/resolved by class, calibration in force)

Usage:  python -m experiments.flywheel_run <subcommand> [args]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _fw():
    from swm.engine.flywheel import FlywheelLog
    return FlywheelLog()


def prime():
    """Bootstrap the log from committed best-config backtest rows (exp095 crowd + exp094 no-market). Same
    engine config, real outcomes — an instant calibration corpus with zero new LLM calls."""
    fw = _fw()
    n = 0
    p95 = Path("experiments/results/exp095_levers_crowd_grade.json")
    if p95.exists():
        for i, r in enumerate(json.load(p95.open()).get("rows", [])):
            rid = fw.log(question=f"[exp095 row {i}]", question_class="society:event",
                         domain=r.get("category", "other"), mechanism="observer_panel",
                         p=r["p_model"], as_of="2025", ts=1_700_000_000 + i)
            fw.record_outcome(rid, float(r["outcome"]), source="replay:exp095")
            n += 1
    p94 = Path("experiments/results/exp094_no_market_grade.json")
    if p94.exists():
        for i, it in enumerate(json.load(p94.open()).get("items", [])):
            if it.get("p") is None:
                continue
            rid = fw.log(question=it["q"][:200], question_class="deliberation:no_market",
                         domain="deliberation", mechanism="observer_panel", p=it["p"],
                         as_of=it.get("as_of", "2025"), ts=1_700_100_000 + i)
            fw.record_outcome(rid, float(it["outcome"]), source="replay:exp094")
            n += 1
    print(f"primed {n} resolved forecasts into {fw.path}")
    return refit()


def forecast(round_date, limit, live):
    """Forecast a ForecastBench round and LOG each forecast. `--live` leaves them OPEN (forward mode);
    otherwise records the known outcome immediately (replay mode)."""
    from swm.engine.front_door import agent_world_model
    from swm.engine.retrieval import asof_search_fn
    from swm.eval.forecastbench import load_round
    from swm.eval.grade_agent_engine import is_domain, p_yes
    from swm.engine.router import ParadigmRouter
    fw = _fw()
    wm = agent_world_model(branches=2, max_rounds=1, log_forecasts=True)
    wm.flywheel = fw                                        # log to our durable store
    router = ParadigmRouter(llm=None)
    qs = [q for q in load_round(round_date) if is_domain(q.meta.get("question", ""))][:limit]
    as_of_ts = time.mktime(time.strptime(round_date, "%Y-%m-%d"))
    logged = 0
    for q in qs:
        text = q.meta["question"]
        try:
            res = wm.simulate(text, as_of=round_date, binary=True, search_fn=asof_search_fn(as_of_ts))
        except Exception as e:
            print(f"  skip ({str(e)[:50]}): {text[:50]}"); continue
        if res.get("abstain"):
            continue
        logged += 1
        # find the record the engine just logged and, in REPLAY mode, attach the known outcome
        if not live:
            rec = fw.load()[-1]
            fw.record_outcome(rec.rid, float(q.outcome), source=f"replay:{round_date}")
        p = p_yes(res)
        print(f"  {'LOG' if live else 'LOG+RESOLVE'} p={p:.2f} y={q.outcome:.0f} [{router.binary_kind(text)}] {text[:52]}")
    print(f"logged {logged} forecasts from round {round_date} ({'forward/open' if live else 'replay/resolved'})")


def resolve(limit):
    from swm.api.deepseek_backend import default_chat_fn
    fw = _fw()
    llm = default_chat_fn(system="Reply ONLY compact JSON.", max_tokens=200, temperature=0.0)
    if llm is None:
        print("no LLM backend"); return
    print("auto-resolving due open records against current news…")
    print(json.dumps(fw.auto_resolve(llm, limit=limit)))


def refit():
    fw = _fw()
    rep = fw.refit(min_n=12)
    print("REFIT from the resolved stream:")
    print(json.dumps(rep, indent=1))
    return rep


def status():
    fw = _fw()
    recs = fw.load()
    from collections import Counter
    openc = Counter((r.question_class, r.domain) for r in recs if r.status == "open")
    resc = Counter((r.question_class, r.domain) for r in recs if r.status == "resolved")
    print(f"log: {fw.path}  |  {len(recs)} records  ({sum(1 for r in recs if r.status=='resolved')} resolved, "
          f"{sum(1 for r in recs if r.status=='open')} open)")
    print("resolved by (class, domain):", dict(resc))
    print("open by (class, domain):", dict(openc))
    from swm.engine.calibrate import GradeRegistry
    print("calibration in force:", json.dumps(GradeRegistry().grades, indent=1)[:800])


def main():
    ap = argparse.ArgumentParser(description="Operate the outcome flywheel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prime")
    fc = sub.add_parser("forecast"); fc.add_argument("--round", required=True); fc.add_argument("--limit", type=int, default=20); fc.add_argument("--live", action="store_true")
    rs = sub.add_parser("resolve"); rs.add_argument("--limit", type=int, default=30)
    sub.add_parser("refit")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "prime": prime()
    elif a.cmd == "forecast": forecast(a.round, a.limit, a.live)
    elif a.cmd == "resolve": resolve(a.limit)
    elif a.cmd == "refit": refit()
    elif a.cmd == "status": status()


if __name__ == "__main__":
    main()
