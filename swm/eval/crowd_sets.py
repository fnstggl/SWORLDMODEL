"""Diverse, cleaned crowd backtest sets — robust measurement across the WHOLE social world, not just politics.

The F-grade was measured on 17 noisy, mostly-political questions — too few and too weird to trust or to
diagnose. This builds larger, cleaner, multi-domain sets from main's `forecasting_corpus` (resolved
Manifold/Polymarket, crowd price at as-of, cutoff_clean), so the agent engine is graded on the variety a
GENERAL social world model must handle: elections, policy, company/product decisions, awards & culture,
sports contests, geopolitics — every outcome that people's choices/behavior generate.

Cleaning (the corpus has junk personal Manifold markets):
  - people-domain only (the ParadigmRouter picks the engine's turf; crypto/science/pure-market go elsewhere);
  - drop personal/degenerate questions ("Can I ...", "will they reset ...", first-person, too short);
  - keep a MIX of crowd-confident and crowd-UNSURE (.35-.65) items — the unsure slice is where a real model
    can actually add value over the market, and a set of only-longshots is trivially gamed by guessing low.
  - cap per category for balance; deterministic order (no RNG — reproducible backtests).
"""
from __future__ import annotations

import re

from swm.engine.router import ParadigmRouter

_JUNK = re.compile(r"\b(can i|will i|should i|my |we reset|reset .* usage|test market|resolve yes|resolve no|"
                   r"this market|will they reset|\bn/?a\b|placeholder)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine)\b", re.I)


def _clean(q: str) -> bool:
    if len(q) < 30 or _JUNK.search(q):
        return False
    if _FIRST_PERSON.search(q) and "AI" not in q:
        return False
    # needs a real subject: at least one capitalized proper noun or a clear social verb
    return bool(re.search(r"\b(win|elect|nominee|nomination|pass|approve|confirm|appoint|resign|become|"
                          r"defeat|beat|lead|advance|majority|vote|deal|merge|acquire|launch|release|"
                          r"nominate|endorse|announce|reach|sign|strike|ban|impeach|award|nominate)\b", q, re.I)
                or len(re.findall(r"\b[A-Z][a-z]+", q)) >= 2)


# non-human-process categories belong to main's parametric engine, not the agent society
_EXCLUDE_CAT = {"crypto", "science"}


def diverse_set(items, *, per_category=14, want_unsure_frac=0.45, min_crowd=25):
    """Filter+balance corpus BacktestItems → a diverse, cleaned, people-domain set. `items` from
    forecasting_corpus.load_corpus(). Deterministic. `min_crowd` is a LIQUIDITY floor — real forecasting
    questions have many bettors; personal Manifold junk ('will I go to the gym') has ~6-12, and its crowd
    price is noise. A higher floor also means a sharper crowd baseline (the honest, harder bar)."""
    router = ParadigmRouter(llm=None)
    pool = [it for it in items
            if it.cutoff_clean and it.n_crowd >= min_crowd and it.category not in _EXCLUDE_CAT
            and _clean(it.question) and router.route(it.question) == "agents"]
    # bucket by category, and within each keep a mix of unsure + confident
    by_cat = {}
    for it in pool:
        by_cat.setdefault(it.category, []).append(it)
    out = []
    for cat, its in by_cat.items():
        unsure = [i for i in its if 0.35 <= i.crowd_prob <= 0.65]
        confident = [i for i in its if not 0.35 <= i.crowd_prob <= 0.65]
        # deterministic: sort by qid so the same set rebuilds every run
        unsure.sort(key=lambda i: i.qid)
        confident.sort(key=lambda i: i.qid)
        n_unsure = min(len(unsure), int(per_category * want_unsure_frac))
        take = unsure[:n_unsure] + confident[: per_category - n_unsure]
        out.extend(take[:per_category])
    out.sort(key=lambda i: (i.category, i.qid))
    return out


def summarize(items) -> dict:
    from collections import Counter
    n = len(items) or 1
    return {"n": len(items), "by_category": dict(Counter(i.category for i in items).most_common()),
            "yes_rate": round(sum(i.outcome for i in items) / n, 3),
            "crowd_unsure_frac": round(sum(1 for i in items if 0.35 <= i.crowd_prob <= 0.65) / n, 3)}
