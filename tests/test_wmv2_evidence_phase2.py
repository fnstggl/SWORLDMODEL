"""Phase 2 evidence — connector + temporal verification tests.

Offline unit tests always run (fixed RSS bytes, no network). A live Google News RSS test runs only when the
network is reachable, and is skipped otherwise — it must never silently pass by mocking a success.
"""
import time
import urllib.request

import pytest

from swm.world_model_v2.evidence_connectors import (GoogleNewsRSSConnector, PairedDateError, RawContentStore,
                                                    paired_dates_ok)
from swm.world_model_v2.evidence_temporal import ADMISSIBLE_PRODUCTION, TemporalVerifier

_FIXED_RSS = (b'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
              b'<title>test</title>'
              b'<item><title>Nurses ratify contract</title><link>https://ex.com/a</link>'
              b'<pubDate>Fri, 25 Aug 2023 07:00:00 GMT</pubDate><source url="https://nnu.org">NNU</source>'
              b'<description>&lt;p&gt;members voted yes&lt;/p&gt;</description></item>'
              b'<item><title>Second story</title><link>https://ex.com/b</link>'
              b'<pubDate>Wed, 13 Sep 2023 07:00:00 GMT</pubDate><source url="https://x.org">X</source>'
              b'<description>details</description></item></channel></rss>')


# ---------------------------------------------------------------- paired-date invariant (production rule)
def test_historical_query_requires_paired_after_and_before():
    conn = GoogleNewsRSSConnector(store=RawContentStore(root="/tmp/swm_ev_test"))
    with pytest.raises(PairedDateError):
        conn.search_historical("x", after_date="", before_date="2023-09-30")     # before-only REFUSED
    with pytest.raises(PairedDateError):
        conn.search_historical("x", after_date="2023-08-01", before_date="")      # after-only REFUSED
    with pytest.raises(PairedDateError):
        conn.search_historical("x", after_date="2023-8-1", before_date="2023-09-30")  # non-ISO REFUSED


def test_paired_dates_ok_validation():
    assert paired_dates_ok("2023-01-01", "2023-02-01")
    assert not paired_dates_ok("", "2023-02-01")
    assert not paired_dates_ok("2023-02-01", "")
    assert not paired_dates_ok("2023-1-1", "2023-02-01")               # non-ISO


# ---------------------------------------------------------------- parse + trace (offline, fixed bytes)
def test_feed_parse_builds_items_with_rank_and_pubdate():
    conn = GoogleNewsRSSConnector(store=RawContentStore(root="/tmp/swm_ev_test"))
    items = conn._parse(_FIXED_RSS, "logical", "wire", "REQ-1", "hash", k=20)
    assert len(items) == 2
    assert items[0].rank == 1 and items[1].rank == 2
    assert items[0].source_name == "NNU" and items[0].link == "https://ex.com/a"
    assert items[0].feed_pubdate_ts is not None                       # RFC-822 parsed
    assert "members voted yes" in items[0].description                # html stripped/unescaped
    assert items[0].item_hash() != items[1].item_hash()


def test_raw_content_store_is_write_once_and_content_addressed():
    store = RawContentStore(root="/tmp/swm_ev_store_test")
    h1 = store.put(b"same bytes")
    h2 = store.put(b"same bytes")
    assert h1 == h2                                                   # content-addressed → one blob
    assert store.get(h1) == b"same bytes"
    assert store.get("deadbeef" * 8) is None


# ---------------------------------------------------------------- temporal classification (offline)
def test_temporal_statuses_offline():
    as_of = time.mktime(time.strptime("2023-09-01", "%Y-%m-%d"))
    d = 86400
    v = TemporalVerifier(verify_online=False, margin_days=1.0)
    assert v.verify(as_of=as_of, claimed_ts=as_of - 10 * d).status == "likely_pre_asof"
    assert v.verify(as_of=as_of, claimed_ts=as_of + 10 * d).status == "likely_post_asof"
    assert v.verify(as_of=as_of, claimed_ts=as_of + 0.5 * d).status == "uncertain"   # grey zone
    assert v.verify(as_of=as_of, claimed_ts=None).status == "undated"
    # only pre-asof statuses are production-admissible
    assert "likely_pre_asof" in ADMISSIBLE_PRODUCTION
    assert "likely_post_asof" not in ADMISSIBLE_PRODUCTION
    assert "uncertain" not in ADMISSIBLE_PRODUCTION


def test_verified_capture_overrides_claimed():
    """A server-side capture at/before as_of yields verified_pre_asof (strongest signal)."""
    as_of = time.mktime(time.strptime("2023-09-01", "%Y-%m-%d"))
    v = TemporalVerifier(verify_online=False)
    # simulate an injected verified signal by calling _classify directly
    from swm.world_model_v2.evidence_temporal import TemporalRecord
    rec = TemporalRecord(status="undated", confidence=0.0, as_of=as_of, claimed_pubdate_ts=as_of + 5 * 86400)
    out = v._classify(rec, claimed_ts=as_of + 5 * 86400, verified_ts=as_of - 30 * 86400)
    assert out.status == "verified_pre_asof"                          # verified capture wins over claimed


# ---------------------------------------------------------------- live Google News RSS (skips w/o network)
def _network_ok():
    try:
        urllib.request.urlopen("https://news.google.com/rss", timeout=8).read(64)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_ok(), reason="no live network to Google News RSS")
def test_live_google_news_rss_paired_query():
    conn = GoogleNewsRSSConnector(store=RawContentStore(root="/tmp/swm_ev_live"))
    items, trace = conn.search_historical("hospital nurses contract", after_date="2023-08-01",
                                          before_date="2023-09-30", requirement_id="LIVE", k=6)
    assert trace.status_code == 200
    assert trace.connector_status in ("ok", "zero_results")
    assert trace.after_date == "2023-08-01" and trace.before_date == "2023-09-30"
    assert trace.raw_content_hash and conn.store.get(trace.raw_content_hash) is not None
    assert "after:2023-08-01" in trace.logical_query and "before:2023-09-30" in trace.logical_query
    for it in items:
        assert it.rank >= 1 and it.wire_url.startswith("https://news.google.com/rss/search")
