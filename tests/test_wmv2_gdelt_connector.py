"""GDELT DOC connector — the free/keyless breadth layer: parse contract, paired-date invariant,
rate-limit pacing, orchestrator wiring (all offline; live behavior exercised by the evidence runs)."""
import json
import time

import pytest

from swm.world_model_v2.evidence_connectors import PairedDateError
from swm.world_model_v2.evidence_connectors_more import (GdeltDocConnector, _GDELT_MIN_INTERVAL_S,
                                                         _gdelt_last_call)

_FIXTURE = {
    "articles": [
        {"url": "https://ex.com/a", "title": "Coalition reaffirms support for Ukraine",
         "seendate": "20260713T213000Z", "domain": "aa.com.tr", "language": "English"},
        {"url": "https://ex.com/b", "title": "Allies gather in Paris to step up pressure",
         "seendate": "20260713T083000Z", "domain": "hurriyetdailynews.com", "language": "English"},
    ]
}


def test_parse_articles_produces_dated_discovered_items():
    c = GdeltDocConnector()
    items = c.parse_articles(json.dumps(_FIXTURE).encode(), "q [2026-07-10..2026-07-16]",
                             "wire", "req_x", "hash", k=10)
    assert len(items) == 2
    it = items[0]
    assert it.connector_id == "gdelt_doc" and it.requirement_id == "req_x"
    assert it.title.startswith("Coalition") and it.source_name == "aa.com.tr"
    assert it.feed_pubdate == "20260713T213000Z"
    assert isinstance(it.feed_pubdate_ts, float)               # parsed to unix ts for as-of checks
    # ranks are 1-based and ordered
    assert [i.rank for i in items] == [1, 2]


def test_paired_dates_are_required():
    c = GdeltDocConnector()
    with pytest.raises(PairedDateError):
        c.search_historical("ukraine ceasefire", after_date="", before_date="2026-07-16")
    with pytest.raises(PairedDateError):
        c.search_historical("ukraine ceasefire", after_date="2026-07-10", before_date="")


def test_datetime_window_format():
    assert GdeltDocConnector._dt("2026-07-10") == "20260710000000"
    assert GdeltDocConnector._dt("2026-07-16", end=True) == "20260716235959"


def test_rate_limit_pacing_is_module_wide(monkeypatch):
    """Two rapid calls must be spaced by the published >=5s limit (module-wide clock), and a
    plain-text advisory body must be classified http_error, never parsed as zero results."""
    c = GdeltDocConnector()
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    import swm.world_model_v2.evidence_connectors_more as M
    monkeypatch.setattr(M, "_time", time)

    class _Resp:
        status = 200

        def read(self):
            return b"Please limit requests to one every 5 seconds"
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: _Resp())
    _gdelt_last_call[0] = time.time()                          # a call just happened
    items, tr = c.search_historical("x y", after_date="2026-07-10", before_date="2026-07-16",
                                    retries=0)
    assert sleeps and sleeps[0] > 0                            # paced before hitting the API
    assert items == [] and tr.connector_status == "http_error"
    assert "limit requests" in tr.error


def test_orchestrator_config_defaults_enable_breadth_layers():
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig
    cfg = OrchestratorConfig()
    assert cfg.use_gdelt is True and cfg.use_wikipedia is True
    assert cfg.wiki_entities_per_req >= 2 and cfg.max_claim_docs >= 24
