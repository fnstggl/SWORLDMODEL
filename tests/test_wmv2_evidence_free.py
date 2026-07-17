"""Free evidence connectors + router — offline (every HTTP call stubbed at one point: _get).

What must hold: the router picks connectors by requirement CATEGORY (universal lexical detection,
no scenario lists); every connector enforces the as-of hygiene its source supports; every failure
mode (blocked host, bad payload, missing token) degrades to a recorded trace, never an abort; and
the orchestrator merges multi-source documents with per-call audit traces."""
import json
import time

import pytest

import swm.world_model_v2.evidence_connectors_free as F
from swm.world_model_v2.evidence_connectors_free import (CuratedRssConnector, FreeSourceRouter,
                                                         GdeltDocConnector, ReliefWebConnector,
                                                         UcdpGedConnector, WikidataFactsConnector,
                                                         WikipediaSearchConnector,
                                                         WorldBankConnector, category_for,
                                                         domains_for)
from swm.world_model_v2.evidence_requirements import EvidenceRequirement


def _req(claim="statement of intent by leader", ents=("Ukraine",), rid="r1", sources=None):
    return EvidenceRequirement(requirement_id=rid, claim_or_quantity=claim,
                               why_relevant="t", affected_component="actor.x",
                               preferred_source_types=list(sources or ["news"]),
                               entity_scope=list(ents))


class _Store:
    def put(self, data):
        return "hash0"


# ---------------------------------------------------------------- router
def test_category_and_domain_detection_are_lexical_and_universal():
    assert category_for(_req("count of troops deployed")) == "quantity"
    assert category_for(_req("scheduled election date")) == "calendar"
    assert category_for(_req("order of battle and weapons aid")) == "capability"
    assert category_for(_req("approval poll public support")) == "opinion"
    assert category_for(_req("public statement of commitment")) == "statement"
    assert "conflict" in domains_for("Will the ceasefire hold after the offensive?")
    assert "economy" in domains_for("Will inflation exceed 3% this year?")
    assert "politics" in domains_for("Will the coalition win the election vote?")


def test_router_activates_structured_sources_by_category():
    r = FreeSourceRouter()
    war_stmt = r.route(_req("public statement on ceasefire", ("Russia",)),
                       "Will the war end with a ceasefire agreement?")
    names = [c for c, _, _ in war_stmt]
    assert "gdelt" in names and "curated_rss" in names          # breadth for statements
    assert "ucdp" in names and "reliefweb" in names             # conflict domain activates both
    assert "wikidata" in names                                  # leadership/membership grounding
    econ_q = r.route(_req("gdp growth rate level", ("Ukraine",)),
                     "Will Ukraine's economy grow?")
    assert "worldbank" in [c for c, _, _ in econ_q]
    opin = r.route(_req("approval poll numbers", ("Germany",)), "Will the chancellor survive?")
    wq = [q for c, q, _ in opin if c == "wikipedia_search"]
    assert any("opinion polling" in q for q in wq)              # the structured page class
    cap = r.route(_req("military capability and weapons aid", ("Ukraine",)),
                  "Can the army hold the line?")
    wq2 = [q for c, q, _ in cap if c == "wikipedia_search"]
    assert any("order of battle" in q for q in wq2)
    # dedupe: no repeated (connector, query) pairs
    assert len(war_stmt) == len({(c, q) for c, q, _ in war_stmt})


# ---------------------------------------------------------------- connectors (stubbed _get)
def test_gdelt_parses_and_carries_paired_dates(monkeypatch):
    payload = {"articles": [{"url": "http://x/a", "title": "Ceasefire talks resume",
                             "domain": "x.com", "seendate": "20260710T080000Z",
                             "sourcecountry": "US"}]}
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return json.dumps(payload).encode(), 200, ""
    monkeypatch.setattr(F, "_get", fake_get)
    items, tr = GdeltDocConnector(store=_Store()).search_historical(
        "ukraine ceasefire", after_date="2026-06-01", before_date="2026-07-16",
        requirement_id="r1", k=5)
    assert len(items) == 1 and items[0].title.startswith("Ceasefire")
    assert "startdatetime=20260601000000" in seen["url"]
    assert "enddatetime=20260716235959" in seen["url"]          # paired-date ceiling in the API call
    assert tr.connector_status == "ok" and tr.n_items == 1


def test_blocked_host_degrades_to_recorded_failure(monkeypatch):
    monkeypatch.setattr(F, "_get", lambda url, **kw: (b"", None, "URLError: proxy denied"))
    items, tr = GdeltDocConnector(store=_Store()).search_historical(
        "x", after_date="2026-06-01", before_date="2026-07-16")
    assert items == [] and tr.connector_status == "network_error" and "denied" in tr.error


def test_wikipedia_search_strips_markup(monkeypatch):
    payload = {"pages": [{"key": "Opinion_polling_X", "title": "Opinion polling X",
                          "excerpt": "the <span class=\"m\">latest</span> polls"}]}
    monkeypatch.setattr(F, "_get", lambda url, **kw: (json.dumps(payload).encode(), 200, ""))
    items, tr = WikipediaSearchConnector(store=_Store()).search("opinion polling X")
    assert items[0].link.endswith("/wiki/Opinion_polling_X")
    assert "<span" not in items[0].description and "latest" in items[0].description


def test_wikidata_facts_filter_post_asof_statements(monkeypatch):
    calls = {}

    def fake_get(url, **kw):
        if "wbsearchentities" in url:
            return json.dumps({"search": [{"id": "Q212"}]}).encode(), 200, ""
        calls["sparql"] = url
        rows = [{"propLabel": {"value": "head of state"}, "valLabel": {"value": "Person A"},
                 "start": {"value": "2024-05-20T00:00:00Z"}},
                {"propLabel": {"value": "head of state"}, "valLabel": {"value": "Person B"},
                 "start": {"value": "2026-09-01T00:00:00Z"}}]     # begins AFTER as_of → dropped
        return json.dumps({"results": {"bindings": rows}}).encode(), 200, ""
    monkeypatch.setattr(F, "_get", fake_get)
    items, tr = WikidataFactsConnector(store=_Store()).facts(
        "Ukraine", as_of_iso="2026-07-17")
    assert len(items) == 1 and "Person A" in items[0].title
    assert tr.connector_status == "ok"


def test_worldbank_series_respects_asof_year_and_resolves_country(monkeypatch):
    def fake_get(url, **kw):
        if "/v2/country?" in url:
            return json.dumps([{}, [{"name": "Ukraine", "iso2Code": "UA"}]]).encode(), 200, ""
        assert "/country/UA/indicator/FP.CPI.TOTL.ZG" in url
        obs = [{"date": "2027", "value": 9.9}, {"date": "2025", "value": 12.3}]
        return json.dumps([{}, obs]).encode(), 200, ""
    monkeypatch.setattr(F, "_get", fake_get)
    WorldBankConnector._countries = None                        # reset the in-process cache
    items, tr = WorldBankConnector(store=_Store()).series(
        "ukraine", "inflation rate", as_of_iso="2026-07-17")
    assert len(items) == 1 and "2025" in items[0].title         # 2027 observation excluded
    assert tr.connector_status == "ok"


def test_ucdp_requires_free_token_and_degrades_without_it(monkeypatch):
    monkeypatch.delenv("UCDP_ACCESS_TOKEN", raising=False)
    items, tr = UcdpGedConnector(store=_Store()).events(
        "Ukraine", after_date="2026-06-01", before_date="2026-07-16")
    assert items == [] and tr.connector_status == "auth_required"
    monkeypatch.setenv("UCDP_ACCESS_TOKEN", "tok")
    payload = {"Result": [{"country": "Ukraine", "conflict_name": "Russia-Ukraine",
                           "date_start": "2026-07-01T00:00:00", "best": 14,
                           "where_description": "east"}]}
    monkeypatch.setattr(F, "_get", lambda url, **kw: (json.dumps(payload).encode(), 200, ""))
    items, tr = UcdpGedConnector(store=_Store()).events(
        "Ukraine", after_date="2026-06-01", before_date="2026-07-16")
    assert len(items) == 1 and "14 deaths" in items[0].title


def test_curated_rss_enforces_window_and_relevance(monkeypatch):
    inside = time.strftime("%a, %d %b %Y 10:00:00 GMT", time.gmtime(time.time() - 5 * 86400))
    outside = "Mon, 01 Jan 2018 10:00:00 GMT"
    feed = (f"<rss><channel>"
            f"<item><title>Ukraine ceasefire talks</title><link>http://u/1</link>"
            f"<pubDate>{inside}</pubDate></item>"
            f"<item><title>Ukraine ceasefire archive</title><link>http://u/2</link>"
            f"<pubDate>{outside}</pubDate></item>"
            f"<item><title>Panda born in zoo</title><link>http://u/3</link>"
            f"<pubDate>{inside}</pubDate></item>"
            f"</channel></rss>").encode()
    monkeypatch.setattr(F, "_get", lambda url, **kw: (feed, 200, ""))
    after = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))
    before = time.strftime("%Y-%m-%d", time.gmtime())
    items, tr = CuratedRssConnector(store=_Store()).search_historical(
        "ukraine ceasefire", after_date=after, before_date=before, domains=("conflict",))
    titles = [i.title for i in items]
    assert any("talks" in t for t in titles)                    # in-window + relevant survives
    assert not any("archive" in t for t in titles)              # out of the paired-date window
    assert not any("Panda" in t for t in titles)                # no keyword overlap


# ---------------------------------------------------------------- orchestrator integration
def test_orchestrator_merges_free_sources_with_traces(monkeypatch, tmp_path):
    import swm.world_model_v2.evidence_orchestrator as O
    from swm.world_model_v2.evidence_connectors import (GoogleNewsRSSConnector, RawContentStore,
                                                        RetrievalTrace)
    from swm.world_model_v2.evidence_connectors_more import WikipediaConnector

    def no_news(self, q, **kw):
        return [], RetrievalTrace(connector_id="google_news_rss", connector_version="x",
                                  requirement_id=kw.get("requirement_id", ""), logical_query=q,
                                  wire_url="", connector_status="zero_results")
    monkeypatch.setattr(GoogleNewsRSSConnector, "search_historical", no_news)
    monkeypatch.setattr(WikipediaConnector, "fetch",
                        lambda self, t, **kw: ([], RetrievalTrace(
                            connector_id="wikipedia", connector_version="x",
                            requirement_id=kw.get("requirement_id", ""), logical_query=t,
                            wire_url="", connector_status="zero_results")))

    def fake_get(url, **kw):
        if "gdeltproject" in url:
            arts = {"articles": [{"url": "http://n/1", "title": "Ceasefire signal", "domain": "n",
                                  "seendate": "20260701T000000Z"}]}
            return json.dumps(arts).encode(), 200, ""
        if "wbsearchentities" in url:
            return json.dumps({"search": [{"id": "Q212"}]}).encode(), 200, ""
        if "sparql" in url:
            rows = [{"propLabel": {"value": "member of"}, "valLabel": {"value": "United Nations"}}]
            return json.dumps({"results": {"bindings": rows}}).encode(), 200, ""
        return b"", None, "URLError: blocked"
    monkeypatch.setattr(F, "_get", fake_get)
    store = RawContentStore(root=str(tmp_path / "raw"))
    reqs = [_req("public statement on ceasefire commitment", ("Ukraine",), rid="rA")]
    bundle = O.gather_evidence("Will the war end with a ceasefire?", as_of="2026-07-16",
                               requirements=reqs, llm=None, store=store,
                               config=O.OrchestratorConfig(verify_online=False,
                                                           use_wikipedia=True))
    types = {d.get("source_type") for d in bundle.documents}
    assert "news" in types                                       # GDELT article landed as news
    assert "structured_fact" in types                            # wikidata fact landed
    cids = {t.get("connector_id") for t in bundle.retrieval_traces}
    assert {"gdelt_doc_v2", "wikidata_facts"} <= cids
    # blocked feeds (curated_rss inner fetches) degraded without aborting the bundle
    assert bundle.bundle_hash()
