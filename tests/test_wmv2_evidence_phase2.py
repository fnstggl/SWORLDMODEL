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
    # returned items fall within the intended discovery window (feed pubDate ≤ before:)
    import time as _t
    before_ts = _t.mktime(_t.strptime("2023-09-30", "%Y-%m-%d")) + 86400
    for it in items:
        if it.feed_pubdate_ts is not None:
            assert it.feed_pubdate_ts <= before_ts + 7 * 86400          # within window (+ tz slack)


# ---------------------------------------------------------------- claim extraction (span validation)
def test_claim_extraction_rejects_unsupported_spans():
    import json
    from swm.world_model_v2.evidence_claims import extract_claims, valid_claims
    text = "The nurses union said members voted to ratify the contract. Management denied any layoffs."
    llm = lambda p: json.dumps({"claims": [
        {"subject": "nurses union", "predicate": "voted to ratify", "claim_class": "actor_statement",
         "supporting_span": "members voted to ratify the contract", "confidence": 0.8},
        {"subject": "management", "predicate": "denied", "claim_class": "denial", "polarity": "negate",
         "supporting_span": "Management denied any layoffs", "confidence": 0.7},
        {"subject": "x", "predicate": "y", "claim_class": "observed_fact",
         "supporting_span": "THIS SPAN IS NOT IN THE TEXT", "confidence": 0.9}]})
    claims = extract_claims(text, source_id="d1", llm=llm)
    assert len(claims) == 3 and len(valid_claims(claims)) == 2          # fabricated span rejected
    assert {c.claim_class for c in valid_claims(claims)} == {"actor_statement", "denial"}


# ---------------------------------------------------------------- entity resolution (ambiguity preserved)
def test_entity_resolution_preserves_ambiguity():
    from swm.world_model_v2.evidence_entities import EntityResolver
    res = {e.mention: e for e in EntityResolver().resolve(["Joe Biden", "Biden", "President Biden"])}
    assert res["Biden"].candidates and res["Biden"].top.canonical  # has candidates
    assert all(len(e.candidates) >= 1 for e in res.values())


# ---------------------------------------------------------------- dependence (syndication collapse)
def test_syndication_collapses_independent_count():
    from swm.world_model_v2.evidence_dependence import cluster_dependence, independent_count
    docs = [{"id": "d1", "text": "Reuters: the merger was approved by regulators today.",
             "url": "http://reuters.com/x", "source": "Reuters", "content_hash": "h1"},
            {"id": "d2", "text": "Reuters: the merger was approved by regulators today.",
             "url": "http://yahoo.com/x", "source": "Yahoo", "content_hash": "h1"},
            {"id": "d3", "text": "A completely different story about spring weather patterns.",
             "url": "http://w.com/y", "source": "W", "content_hash": "h3"}]
    groups = cluster_dependence(docs)
    assert independent_count(groups) == 2                              # two copies → one independent source
    assert any(g.dependence_type == "exact_duplicate" for g in groups)


# ---------------------------------------------------------------- contradiction graph
def test_contradiction_graph_numerical_and_denial():
    from swm.world_model_v2.evidence_claims import Claim
    from swm.world_model_v2.evidence_contradictions import build_contradiction_graph, has_material_contradiction
    a = Claim(claim_id="a", source_id="s1", subject="turnout", predicate="was", value="40",
              object="percent", publication_time=1.0)
    b = Claim(claim_id="b", source_id="s2", subject="turnout", predicate="was", value="55",
              object="percent", publication_time=2.0)
    edges = build_contradiction_graph([a, b])
    assert has_material_contradiction(edges)
    assert edges[0].ctype == "numerical_disagreement" and edges[0].temporal_order == "a_before_b"
    assert "b" in a.contradiction_links and "a" in b.contradiction_links   # claims cross-linked


# ---------------------------------------------------------------- actor visibility (adversarial)
def test_actor_visibility_enforcement():
    from swm.world_model_v2.evidence_visibility import ClaimVisibility, assign_visibility, actor_view
    import time as _t
    pub_ts = _t.mktime(_t.strptime("2023-08-15", "%Y-%m-%d"))
    # public news: visible to everyone, but only AFTER publication time
    pub = assign_visibility(claim_id="c1", source_type="news", publication_time=pub_ts)
    assert pub.visibility == "public"
    assert pub.observable_by("anyone", pub_ts + 10) is True
    assert pub.observable_by("anyone", pub_ts - 10) is False       # not knowable before it was published
    # private finance discussion: manager NOT in the room cannot see it
    priv = assign_visibility(claim_id="c2", source_type="user_provided", publication_time=pub_ts,
                             hint={"visibility": "private_group", "actors": ["cfo", "ceo"]})
    assert priv.observable_by("cfo", pub_ts + 10) is True
    assert priv.observable_by("line_manager", pub_ts + 10) is False
    # negotiation reservation value: other party cannot see it
    resv = ClaimVisibility(claim_id="c3", visibility="confidential", actors=["party_a"])
    assert resv.observable_by("party_a", None) is False            # confidential is never actor-observable
    assert resv.observable_by("party_b", None) is False
    # unknown visibility is NOT public by default (fail-safe)
    unk = assign_visibility(claim_id="c4", source_type="unknown", publication_time=pub_ts)
    assert unk.visibility == "uncertain" and unk.observable_by("someone", pub_ts + 10) is False
    view = actor_view([pub, priv], actor_id="cfo", at_time=pub_ts + 10)
    assert "c1" in view and "c2" in view


# ---------------------------------------------------------------- immutable bundle (hash + versioning)
def test_evidence_bundle_immutability_and_versioning():
    from swm.world_model_v2.evidence_bundle_v2 import EvidenceBundleV2
    b = EvidenceBundleV2(bundle_id="eb", question="Will X?", as_of=1.0e9,
                         claims=[{"claim_id": "s:c0", "subject": "x", "predicate": "y", "object": "",
                                  "value": "", "claim_class": "observed_fact", "polarity": "affirm",
                                  "supporting_span": "s", "span_verified": True,
                                  "temporal_validity_status": "likely_pre_asof", "actor_visibility": "public",
                                  "dependence_group": ""}], included_claim_ids=["s:c0"])
    h1 = b.freeze()
    assert b.frozen and b.bundle_hash() == h1 == b.compute_hash()   # deterministic
    nxt = b.new_version()                                           # a correction → NEW version, links prior
    assert nxt.version == 2 and nxt.prior_bundle_hash == h1 and not nxt.frozen
    nxt.claims.append({"claim_id": "s:c1", "subject": "z", "predicate": "w", "claim_class": "correction",
                       "span_verified": True, "temporal_validity_status": "likely_pre_asof"})
    assert nxt.compute_hash() != h1                                 # different evidence → different hash


@pytest.mark.skipif(not _network_ok(), reason="no live network for orchestrator")
def test_orchestrator_end_to_end_live():
    import json
    from swm.world_model_v2.evidence_requirements import EvidenceRequirement
    from swm.world_model_v2.evidence_orchestrator import gather_evidence, OrchestratorConfig
    llm = lambda p: json.dumps({"claims": [{"subject": "o", "predicate": "reported",
        "claim_class": "observed_fact", "supporting_span": p.split("TEXT:\n", 1)[-1].split(".")[0][:50],
        "entities": ["union"], "confidence": 0.7}]})
    reqs = [EvidenceRequirement(requirement_id="req_o", claim_or_quantity="nurses ratify contract",
            why_relevant="terminal", affected_component="terminal_outcome", entity_scope=["nurses union"])]
    b = gather_evidence("Will the nurses ratify the contract?", as_of="2023-09-30", requirements=reqs,
                        llm=llm, config=OrchestratorConfig(lookback_days=90, use_wikipedia=False),
                        store=RawContentStore(root="/tmp/swm_orch_live"))
    assert b.retrieval_traces and "after:" in b.retrieval_traces[0]["logical_query"]
    assert "before:" in b.retrieval_traces[0]["logical_query"]
    assert b.documents and all(d["temporal_status"] != "verified_post_asof" for d in b.documents)
    assert b.bundle_hash() == b.compute_hash()                     # deterministic after live retrieval


# ---------------------------------------------------------------- evidence changes the WORLD, not just lean
def _plan_with_hyps():
    import json
    from swm.world_model_v2.compiler import compile_world
    d = {"outcome": {"family": "binary", "options": ["approved", "not_approved"],
                     "resolution_rule": "the VP approves the hire", "readout_var": "decision"},
         "outcome_lean": "neutral", "entities": [{"id": "vp", "type": "person", "fields": {}}],
         "structural_hypotheses": [{"id": "H_direct", "lean": "weak_yes", "prior": 0.5},
                                   {"id": "H_needs_finance", "lean": "weak_no", "prior": 0.5}],
         "required_causal_processes": ["approval_decision"], "rationale": "vp decides"}
    return compile_world("Will the VP approve the two backend hires?", llm=lambda p: json.dumps(d),
                         evidence="", as_of="2023-07-01", horizon="2023-09-30")


def _finance_bundle(plan):
    from swm.world_model_v2.evidence_bundle_v2 import EvidenceBundleV2
    b = EvidenceBundleV2(bundle_id="eb_fin", question=plan.question, as_of=plan.as_of,
        claims=[{"claim_id": "d1:c0", "source_id": "d1", "subject": "finance department",
                 "predicate": "must approve", "object": "hiring above budget", "value": "required",
                 "claim_class": "official_record", "polarity": "affirm",
                 "supporting_span": "all hires above budget require finance sign-off", "span_verified": True,
                 "temporal_validity_status": "likely_pre_asof", "actor_visibility": "public",
                 "dependence_group": ""}],
        included_claim_ids=["d1:c0"],
        actor_visibility=[{"claim_id": "d1:c0", "visibility": "public", "actors": ["*"],
                           "earliest_observation_time": plan.as_of, "method": "m", "uncertainty": 0.1,
                           "communication_path": "", "evidence": ""}],
        requirement_coverage={"req_outcome": {"status": "fulfilled", "n_included": 1, "n_claims": 1}})
    b.freeze()
    return b


def _revise_llm(prompt):
    import json
    return json.dumps({
        "new_institutions": [{"id": "finance_dept", "rules": [{"kind": "approval",
            "params": {"actions": ["hire"], "approver": "finance_dept"}}], "claim_ids": ["d1:c0"]}],
        "new_events": [{"etype": "approval", "at": "2023-08-15", "participants": ["finance_dept"],
                        "claim_ids": ["d1:c0"]}],
        "hypothesis_reweight": [{"id": "H_needs_finance", "direction": "up", "claim_ids": ["d1:c0"]}],
        "outcome_lean": "weak_no", "lean_claim_ids": ["d1:c0"], "uncertainty": "widen"})


def test_evidence_changes_plan_structure_not_only_lean():
    from swm.world_model_v2.evidence_materialize import evidence_causal_effect
    plan = _plan_with_hyps()
    eff = evidence_causal_effect(plan, _finance_bundle(plan), llm=_revise_llm, horizon="2023-09-30",
                                 seed=7, n_particles=60)
    assert eff["evidence_is_causal"] and not eff["lean_only"]
    assert eff["n_institutions_post"] > eff["n_institutions_pre"]       # institution ADDED from evidence
    assert eff["n_events_post"] > eff["n_events_pre"]                   # approval + observe events ADDED
    assert eff["observation_state_deltas"] > 0                         # evidence produced StateDeltas
    assert eff["terminal_changed"]                                     # terminal distribution changed
    kinds = {e["kind"] for e in eff["plan_diff"]["entries"]}
    assert {"institution_added", "event_added", "hypothesis_reweighted"} <= kinds


def test_plan_diff_grounds_changes_in_claim_ids():
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence
    plan = _plan_with_hyps()
    revised, diff = recompile_with_evidence(plan, _finance_bundle(plan), llm=_revise_llm, horizon="2023-09-30")
    inst = [e for e in diff.entries if e.kind == "institution_added"][0]
    assert inst.supporting_claim_ids == ["d1:c0"]                      # change traces to the exact claim
    assert revised.provenance.get("evidence_conditioned") is True
    assert plan.provenance.get("evidence_conditioned") is not True     # pre-evidence plan untouched


def test_no_evidence_means_no_revision():
    from swm.world_model_v2.evidence_bundle_v2 import EvidenceBundleV2
    from swm.world_model_v2.evidence_recompile import recompile_with_evidence
    plan = _plan_with_hyps()
    empty = EvidenceBundleV2(bundle_id="eb0", question=plan.question, as_of=plan.as_of)
    empty.freeze()
    revised, diff = recompile_with_evidence(plan, empty, llm=_revise_llm, horizon="2023-09-30")
    assert diff.n_structural_changes == 0 and len(diff.entries) == 0    # no admissible claims → no change
