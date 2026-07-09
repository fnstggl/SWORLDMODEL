"""Grade the agent engine against the CROWD — the honest bar (what ForecastBench / Prophet Arena score on).

Main's `forecasting_corpus` gives resolved Manifold/Polymarket questions with the crowd probability
reconstructed at a fair as-of lead, tagged `cutoff_clean` (resolved after the LLM's training cutoff ⇒ no
memorization). Beating 0.5, or even a class base rate, is table stakes; beating the CROWD/market is the real
test the whole field is measured on. This harness:

  1. filters the corpus to PEOPLE-domain, cutoff_clean items (the agent engine's turf; the ParadigmRouter
     decides — non-human stochastic questions belong to main's parametric engine, not here);
  2. grounds each leak-free with `swm/retrieval/asof_news.asof_headlines` — GDELT headlines in a window
     ENDING at the item's as-of (the same information the crowd had, and nothing after);
  3. runs the engine `binary=True` → p_model, and scores model vs crowd vs base rate with main's
     `backtest_harness.score` (skill_vs_crowd, skill_vs_base, per-slice, and — key — the crowd-UNSURE
     slice where a real model can actually add value);
  4. records the grade + shrink, gated on **skill_vs_crowd** (the market bar).

Environment note: GDELT's DOC API is aggressively rate-limited (HTTP 429) and unreachable from some
sandboxes; `asof_headlines` then returns nothing and the engine abstains rather than grounding on live
(leaking) news — honest, but it means the news-grounded crowd grade needs GDELT reachable or a keyed as-of
search overlay (Serper/Tavily/Brave, auto-detected by `swm/engine/retrieval.py`). The harness is correct
either way; the abstention rate reports the grounding reality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.engine.calibrate import GradeRegistry
from swm.engine.retrieval import Passage
from swm.engine.router import ParadigmRouter
from swm.eval.grade_agent_engine import p_yes


def asof_evidence(question, as_of_ts, *, days_back=45, k=8) -> list:
    """Leak-free as-of news for the item, as engine Passages. Primary source is keyless Google-News-as-of
    (bounded window + a hard pubDate<=as_of drop — works where GDELT is rate-limited); GDELT is a fallback
    where it is reachable. Both are as-of by construction, so a post-cutoff outcome cannot leak in."""
    from swm.engine.retrieval import asof_google_news
    ev = asof_google_news(question, as_of_ts, days_back=days_back, k=k)
    if ev:
        return ev
    try:
        from swm.retrieval.asof_news import asof_headlines
        heads = asof_headlines(question, as_of_ts, days_back=days_back, k=k) or []
        return [Passage(h, "gdelt_asof", "") for h in heads]
    except Exception:
        return []


@dataclass
class CrowdGradeReport:
    n_domain: int = 0
    n_scored: int = 0
    n_abstained: int = 0
    scoreboard: dict = field(default_factory=dict)
    grade_entry: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)


def grade_vs_crowd(wm, items, *, limit=40, question_class="society:event",
                   registry: GradeRegistry = None, evidence_fn=asof_evidence, verbose=True) -> CrowdGradeReport:
    """`items`: main.forecasting_corpus BacktestItems. Runs people-domain, cutoff_clean items leak-free and
    scores vs the crowd. `evidence_fn(question, as_of_ts) -> [Passage]` is injectable (stub in tests)."""
    from swm.eval.backtest_harness import score
    registry = registry or GradeRegistry()
    router = ParadigmRouter(llm=None)                     # lexical, people-biased — pick the engine's turf
    pool = [it for it in items if it.cutoff_clean and router.route(it.question) == "agents"]
    rep = CrowdGradeReport(n_domain=len(pool))
    rows = []
    for it in pool[:limit]:
        ev = evidence_fn(it.question, it.as_of)
        as_of_str = __import__("time").strftime("%Y-%m-%d", __import__("time").localtime(it.as_of))
        try:
            # pass ev even when EMPTY — a backtest must NEVER fall back to live (current, leaking) retrieval;
            # no as-of evidence ⇒ the engine abstains, which is the honest leak-free outcome.
            res = wm.simulate(it.question, evidence=ev, as_of=as_of_str, binary=True)
        except Exception as e:
            res = {"abstain": True, "abstain_reason": f"{type(e).__name__}: {str(e)[:60]}"}
        p = None if res.get("abstain") else p_yes(res)
        if p is None:
            rep.n_abstained += 1
            if verbose:
                print(f"  ABSTAIN  crowd={it.crowd_prob:.2f} y={it.outcome}  {it.question[:64]}")
            continue
        rows.append({"outcome": it.outcome, "p_model": p, "p_crowd": it.crowd_prob,
                     "category": it.category, "cutoff_clean": it.cutoff_clean})
        if verbose:
            print(f"  p={p:.2f} crowd={it.crowd_prob:.2f} y={it.outcome}  {it.question[:60]}")
    rep.n_scored = len(rows)
    rep.rows = rows
    if rows:
        rep.scoreboard = score(rows)
        ov = rep.scoreboard["overall"]
        rep.grade_entry = registry.record(
            question_class,
            backtest_report={"skill_vs": {"crowd": ov.get("skill_vs_crowd") or -1}, "n": len(rows),
                             "rmse": ov["brier_model"] ** 0.5},
            preds=[r["p_model"] for r in rows], outcomes=[r["outcome"] for r in rows])
    return rep
