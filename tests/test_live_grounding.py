"""Tests for the LIVE grounding wiring — market source, LLM resolver, web retrieval, JSON parsing.
Deterministic: HTTP and the LLM are mocked; no network. A separate gated smoke (exp086) hits real endpoints."""
import re

import swm.api.live_market as lm
import swm.api.live_search as ls
from swm.api.grounding_sources import GroundingRouter, LLMResolver
from swm.api.retrieval_grounding import CalibratedExtractor, build_retrieval_grounder, parse_json_lenient


def _queried_variable(prompt):
    """What a real resolver LLM reads: the variable being matched (not the menu that lists every alias)."""
    m = re.search(r'variable:\s*"([^"]+)"', prompt)
    return (m.group(1) if m else "").lower()


def test_parse_json_lenient_handles_fences_and_prose():
    assert parse_json_lenient('{"a": 1}') == {"a": 1}
    assert parse_json_lenient('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_lenient('Sure! Here it is: {"value": 3.5, "confidence": 0.8} hope that helps') == \
        {"value": 3.5, "confidence": 0.8}
    assert parse_json_lenient("not json at all") is None
    assert parse_json_lenient({"already": "dict"}) == {"already": "dict"}


def test_coinbase_source_grounds_price_and_series(monkeypatch):
    canned = {"spot": {"data": {"amount": "61797.20"}},
              "candles": [[t, 100, 200, 150, 150 + t, 10] for t in range(30)]}
    monkeypatch.setattr(lm, "_get", lambda url, timeout=15: canned["spot"] if "spot" in url else canned["candles"])
    src = lm.coinbase_source()
    gv = src.ground_key("btc_usd")
    assert gv is not None and gv.value == 61797.20 and gv.sd > 0 and gv.source == "coinbase:btc_usd"
    key, seq = src.ground_series_key("btc_usd", window=4)
    assert key == "btc_usd" and len(seq) == 4 and seq == sorted(seq)      # oldest->newest closes


def test_coinbase_source_returns_none_on_failure(monkeypatch):
    def boom(url, timeout=15):
        raise OSError("network down")
    monkeypatch.setattr(lm, "_get", boom)
    assert lm.coinbase_source().ground_key("btc_usd") is None             # failure -> None, never raises


def test_llm_resolver_infers_source_semantically():
    # the "LLM" maps a paraphrase with no token overlap to the right series
    def llm(prompt):
        if "ether" in _queried_variable(prompt):                          # reads the VARIABLE, not the menu
            return '{"source": "coinbase", "key": "eth_usd", "confidence": 0.9}'
        return '{"source": null, "key": null, "confidence": 0.0}'
    src = lm.coinbase_source()
    res = LLMResolver(llm)
    m = res.resolve("the price of ether", None, [src])
    assert m is not None and m[0].name == "coinbase" and m[1] == "eth_usd"
    assert res.resolve("weather in paris", None, [src]) is None           # no match -> None -> retrieval


def test_llm_resolver_gates_low_confidence():
    llm = lambda p: '{"source": "coinbase", "key": "btc_usd", "confidence": 0.2}'
    # a confident-enough match resolves; the same match below min_conf is gated to None (-> retrieval)
    assert LLMResolver(llm, min_conf=0.1).resolve("bitcoin", None, [lm.coinbase_source()]) is not None
    assert LLMResolver(llm, min_conf=0.5).resolve("bitcoin", None, [lm.coinbase_source()]) is None


def test_router_uses_resolver_then_retrieval(monkeypatch):
    monkeypatch.setattr(lm, "_get", lambda url, timeout=15: ({"data": {"amount": "61797.2"}} if "spot" in url
                        else [[t, 1, 2, 1.5, 1.5, 9] for t in range(30)]))

    def match_llm(prompt):
        return ('{"source":"coinbase","key":"btc_usd","confidence":0.95}' if "bitcoin" in _queried_variable(prompt)
                else '{"source":null,"key":null,"confidence":0}')
    extract_llm = lambda p: '{"value": 4.1, "ci95": 0.2, "confidence": 0.8}'
    router = GroundingRouter(sources=[lm.coinbase_source()],
                             retrieval=build_retrieval_grounder(lambda q, a: ["ev"], extract_llm),
                             resolver=LLMResolver(match_llm))
    btc = router.ground("bitcoin price")
    assert btc.source == "coinbase:btc_usd" and btc.value == 61797.2       # resolver -> structured
    infl = router.ground("current inflation rate")
    assert infl.source == "retrieval" and infl.value == 4.1               # no structured match -> retrieval


def test_web_search_fn_combines_backends(monkeypatch):
    monkeypatch.setattr(ls, "ddg_snippets", lambda q, k=5: ["snippet: unemployment was 4.1%"])
    monkeypatch.setattr(ls, "wikipedia_extract", lambda q: "The United States has a population of 341 million.")
    passages = ls.web_search_fn()("us unemployment", as_of=None)
    assert any("4.1%" in p for p in passages) and any("Wikipedia" in p for p in passages)


def test_extractor_never_reports_zero_sd():
    ext = CalibratedExtractor(lambda p: '{"value": 5.25, "ci95": 0, "confidence": 0.9}')   # zero CI
    r = ext.extract("fed funds rate", None, ["ev"])
    assert r is not None and r["sd"] > 0                                   # floored, never fakes certainty
