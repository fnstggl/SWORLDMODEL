"""The live grounding router — real backends wired end to end, for grounding the actual present.

Assembles the coverage layer with LIVE, keyless backends and a real LLM:
  - structured tier: `coinbase_source()` (real-time crypto market data, keyless).
  - matcher: `LLMResolver` over DeepSeek — LLM-INFERRED variable→series matching (no hardcoded token rules).
  - retrieval tier: `build_retrieval_grounder(web_search_fn(), llm)` — DuckDuckGo/Wikipedia evidence + a real
    DeepSeek value-extractor that returns a value + confidence, which the calibration layer turns into a
    trustworthy CI.

`live_router()` returns a `GroundingRouter` that drops into `StateGrounder(default=router)` (state layer) and
whose `ground_series` feeds `TransitionOperator.ground_gain` (rate layer) — the same object that grounded
committed fixtures now grounds the live world. Everything degrades gracefully: no LLM key ⇒ lexical matching +
no retrieval; a dead endpoint ⇒ that variable falls through, never a crash.

Production upgrades (documented, same interfaces): a keyed equities/rates feed (Twelve Data, FRED) for the
non-crypto markets/macro series, and a keyed search API (Tavily/Brave) for richer as-of web evidence.
"""
from __future__ import annotations

from swm.api.grounding_sources import GroundingRouter, LLMResolver
from swm.api.live_market import coinbase_source
from swm.api.live_search import web_search_fn
from swm.api.retrieval_grounding import build_retrieval_grounder


def json_llm(*, max_tokens=300):
    """A DeepSeek (or HF fallback) chat fn primed to reply with ONLY JSON — used for both extraction and the
    LLM resolver. None if no LLM key is configured."""
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="You are a precise data assistant. Reply with ONLY compact JSON, no prose.",
                           max_tokens=max_tokens, temperature=0.0)


def live_router(*, use_llm_matcher=True, ci_multiplier=1.0, extra_sources=None, llm=None) -> GroundingRouter:
    """A GroundingRouter over live backends. `use_llm_matcher` turns on LLM-inferred matching (recommended);
    `extra_sources` adds keyed structured sources (equities, macro) with the same interface."""
    llm = llm if llm is not None else json_llm()
    sources = [coinbase_source()] + list(extra_sources or [])
    retrieval = build_retrieval_grounder(web_search_fn(), llm, ci_multiplier=ci_multiplier) if llm else None
    resolver = LLMResolver(llm) if (use_llm_matcher and llm) else None
    return GroundingRouter(sources=sources, retrieval=retrieval, resolver=resolver)
