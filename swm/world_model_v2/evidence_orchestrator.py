"""Retrieval orchestrator — Phase 2.

Turns typed evidence requirements into an immutable evidence bundle by running the full pipeline:

  requirements → source-specific queries → connectors (Google News RSS paired after:/before:, Wikipedia,
  user docs, datasets, prior artifacts) → raw + trace persistence → temporal verification → claim
  extraction (span-validated) → entity resolution → dependence/syndication grouping → contradiction graph →
  actor visibility → leakage audit → included/excluded/suspicious partition → coverage → frozen bundle.

Every connector invocation produces a trace (failures included, never silently dropped). Retrieval windows
for the historical arm are mechanically derived from the as-of timestamp + a lookback and ALWAYS carry both
after: and before:. Metered (cost/latency). Deterministic given the same connector snapshots + seed.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass

from swm.world_model_v2.evidence_bundle_v2 import EvidenceBundleV2, partition_claims
from swm.world_model_v2.evidence_claims import extract_claims
from swm.world_model_v2.evidence_connectors import GoogleNewsRSSConnector, RawContentStore
from swm.world_model_v2.evidence_connectors_more import (LocalDatasetConnector, PriorArtifactConnector,
                                                         UserDocumentConnector, WebPageConnector,
                                                         WikipediaConnector)
from swm.world_model_v2.evidence_contradictions import build_contradiction_graph
from swm.world_model_v2.evidence_dependence import cluster_dependence, independent_count
from swm.world_model_v2.evidence_entities import EntityResolver
from swm.world_model_v2.evidence_temporal import TemporalVerifier
from swm.world_model_v2.evidence_visibility import assign_visibility
from swm.world_model_v2.state import parse_time, rfc3339

# resolution-term-ish and retrospective leakage patterns are reused from the Session-1 auditor
from swm.world_model_v2.leakage_audit import RETROSPECTIVE_PATTERNS


@dataclass
class OrchestratorConfig:
    lookback_days: int = 120
    max_items_per_query: int = 12
    verify_online: bool = False                 # Wayback verification (network); off by default for speed
    use_wikipedia: bool = True
    extract_claims: bool = True
    max_requirements_retrieved: int = 3         # cap RSS queries to the top-VoI requirements
    max_claim_docs: int = 8                     # cap LLM claim-extraction calls per question


def _window(as_of_ts: float, lookback_days: int) -> tuple:
    before = _time.strftime("%Y-%m-%d", _time.gmtime(as_of_ts))
    after = _time.strftime("%Y-%m-%d", _time.gmtime(as_of_ts - lookback_days * 86400))
    return after, before


#: question-framing + function words that hurt a Google News RSS keyword query (a full-sentence question
#: with "?" is matched as a phrase and returns zero results — Google News wants KEYWORDS, not a sentence).
_STOP = frozenset((
    "will", "would", "should", "could", "the", "a", "an", "to", "of", "in", "on", "its", "it", "by", "for",
    "and", "or", "reach", "be", "is", "are", "was", "were", "this", "that", "these", "those", "within",
    "before", "after", "next", "upcoming", "new", "make", "get", "getting", "do", "does", "did", "have",
    "has", "had", "at", "as", "with", "about", "if", "than", "then", "more", "less", "over", "under", "into",
    "out", "up", "down", "their", "his", "her", "our", "my", "your", "we", "they", "he", "she", "i", "you",
    "any", "some", "all", "no", "not", "so", "but", "there", "here", "when", "where", "who", "what", "which",
    "how", "why", "whether", "still", "just", "now", "vote", "day", "week", "month", "quarter", "year"))


def _keywords(text: str, k: int = 8) -> list:
    import re as _re
    seen, out = set(), []
    for w in _re.findall(r"[A-Za-z0-9]+", text):
        lw = w.lower()
        if lw in _STOP or len(w) < 3 or lw in seen:
            continue
        seen.add(lw); out.append(w)
        if len(out) >= k:
            break
    return out


def _query_terms(req, question: str) -> str:
    """Build a KEYWORD search query (never the full-sentence question). For the terminal outcome, keywords
    from the question; for a component requirement, the cleaned component + need. Named entities from the
    plan (entity_scope) are appended — those are the strongest search terms for named events."""
    if req.affected_component == "terminal_outcome":
        base = " ".join(_keywords(question, 8))
    else:
        need = req.claim_or_quantity.replace("value/context of", " ").replace("_", " ").replace(".", " ")
        base = " ".join(_keywords(f"{req.affected_component.replace('_', ' ').replace('.', ' ')} {need}", 6))
    ents = " ".join(str(e).replace("_", " ") for e in (req.entity_scope or [])[:3])
    return f"{base} {ents}".strip()[:160]


def gather_evidence(question: str, *, as_of: str, requirements: list, llm=None,
                    user_documents: list | None = None, dataset_path: str = "",
                    prior_bundle_path: str = "", config: OrchestratorConfig | None = None,
                    plan_hash: str = "", seed: int = 0, store: RawContentStore | None = None,
                    bundle_id: str = "") -> EvidenceBundleV2:
    """Run the full evidence pipeline for a set of requirements and return a FROZEN EvidenceBundleV2."""
    cfg = config or OrchestratorConfig()
    store = store or RawContentStore()
    as_of_ts = parse_time(as_of)
    as_of_iso = _time.strftime("%Y-%m-%d", _time.gmtime(as_of_ts))
    t0 = _time.time()
    gnews = GoogleNewsRSSConnector(store=store)
    wiki = WikipediaConnector(store=store)
    verifier = TemporalVerifier(verify_online=cfg.verify_online, margin_days=1.0)

    traces, documents, retrieval_plan, connector_failures = [], [], [], []
    after, before = _window(as_of_ts, cfg.lookback_days)

    # ---- retrieval per requirement (top-VoI requirements only, to bound retrieval + LLM cost) ----
    retrieved_reqs = sorted(requirements, key=lambda r: -r.expected_voi)[:cfg.max_requirements_retrieved]
    for req in retrieved_reqs:
        terms = _query_terms(req, question)
        retrieval_plan.append({"requirement_id": req.requirement_id, "terms": terms,
                               "after": after, "before": before,
                               "preferred": req.preferred_source_types})
        # Google News RSS — paired after:/before: ALWAYS
        items, tr = gnews.search_historical(terms, after_date=after, before_date=before,
                                            requirement_id=req.requirement_id, k=cfg.max_items_per_query)
        traces.append(tr.as_dict())
        if tr.connector_status not in ("ok", "zero_results"):
            connector_failures.append({"connector": tr.connector_id, "status": tr.connector_status,
                                       "error": tr.error, "requirement_id": req.requirement_id})
        for it in items:
            documents.append(_doc_from_item(it, "news"))
        # Wikipedia background for the top entity (server-side revision time)
        if cfg.use_wikipedia and req.entity_scope:
            witems, wtr = wiki.fetch(str(req.entity_scope[0]), requirement_id=req.requirement_id,
                                     as_of_iso=as_of_iso)
            traces.append(wtr.as_dict())
            for it in witems:
                documents.append(_doc_from_item(it, "wikipedia_revision"))

    # user documents (private by default via visibility hint)
    if user_documents:
        uitems, utr = UserDocumentConnector(store=store).fetch(user_documents)
        traces.append(utr.as_dict())
        for it in uitems:
            documents.append(_doc_from_item(it, "user_provided"))
    if dataset_path:
        ditems, dtr = LocalDatasetConnector(store=store).fetch(dataset_path, question,
                                                               requirement_id="dataset")
        traces.append(dtr.as_dict())
        for it in ditems:
            documents.append(_doc_from_item(it, "dataset"))
    if prior_bundle_path:
        pitems, ptr = PriorArtifactConnector(store=store).fetch(prior_bundle_path)
        traces.append(ptr.as_dict())
        for it in pitems:
            documents.append(_doc_from_item(it, "prior_world_state"))

    # ---- temporal verification (per document) ----
    temporal_records = {}
    for d in documents:
        rec = verifier.verify(as_of=as_of_ts, claimed_ts=d.get("published_at"), url=d.get("url", ""))
        d["temporal_status"] = rec.status
        temporal_records[d["id"]] = rec.as_dict()

    # ---- dependence / syndication (over documents) ----
    dep_groups = cluster_dependence(documents)
    doc_to_group = {}
    for g in dep_groups:
        for mid in g.member_ids:
            doc_to_group[mid] = g.group_id

    # ---- claim extraction (dedup docs, cap count, skip post-as-of) ----
    claims = []
    if cfg.extract_claims and llm is not None:
        seen_hashes, claim_docs = set(), []
        for d in sorted(documents, key=lambda x: (x.get("temporal_status") != "likely_pre_asof", x.get("rank", 99))):
            if d.get("content_hash") in seen_hashes:
                continue                                 # one claim-extraction per distinct article
            if d.get("temporal_status") in ("verified_post_asof", "likely_post_asof"):
                continue                                 # never extract claims from post-as-of content
            seen_hashes.add(d.get("content_hash"))
            claim_docs.append(d)
            if len(claim_docs) >= cfg.max_claim_docs:
                break
        for d in claim_docs:
            cs = extract_claims(d.get("text", ""), source_id=d["id"], llm=llm,
                                publication_time=d.get("published_at"))
            for c in cs:
                c.temporal_validity_status = d.get("temporal_status", "")
                c.dependence_group = doc_to_group.get(d["id"], "")
                c.provenance["requirement_id"] = d.get("requirement_id", "")
            claims.extend(cs)

    # ---- entity resolution (across all claim mentions) ----
    mentions = [m for c in claims for m in c.entities]
    entity_res = EntityResolver().resolve(mentions) if mentions else []

    # ---- contradiction graph ----
    contradictions = build_contradiction_graph(claims)

    # ---- actor visibility (per claim) ----
    src_type = {d["id"]: d.get("source_type", "news") for d in documents}
    visibilities = []
    for c in claims:
        hint = _visibility_hint(c, documents)
        cv = assign_visibility(claim_id=c.claim_id, source_type=src_type.get(c.source_id, "news"),
                               publication_time=c.publication_time, claim_class=c.claim_class, hint=hint)
        c.actor_visibility = cv.visibility
        visibilities.append(cv)

    # ---- leakage audit (claim-level: resolution terms / retrospective / post-as-of) ----
    leakage_ids, leakage_flags = _leakage_audit(claims, question, as_of_ts, temporal_records)

    # ---- partition + coverage ----
    inc, exc, susp = partition_claims([c.as_dict() for c in claims], temporal_records, set(leakage_ids))
    coverage = _coverage(requirements, claims, inc)
    missing = [r.requirement_id for r in requirements
               if coverage.get(r.requirement_id, {}).get("n_included", 0) == 0]

    bundle = EvidenceBundleV2(
        bundle_id=bundle_id or f"eb_{abs(hash(question)) & 0xFFFFFF:06x}", question=question, as_of=as_of_ts,
        compiler_plan_hash=plan_hash, requirements=[r.as_dict() for r in requirements],
        retrieval_plan=retrieval_plan, retrieval_traces=traces,
        raw_content_refs=sorted({d.get("raw_feed_hash", "") for d in documents if d.get("raw_feed_hash")}),
        documents=documents, claims=[c.as_dict() for c in claims],
        entities=[e.as_dict() for e in entity_res], temporal_records=temporal_records,
        dependence_groups=[g.as_dict() for g in dep_groups],
        contradiction_graph=[e.as_dict() for e in contradictions],
        actor_visibility=[v.as_dict() for v in visibilities], leakage_flags=leakage_flags,
        included_claim_ids=inc, excluded_claim_ids=exc, suspicious_claim_ids=susp,
        requirement_coverage=coverage, missing_evidence=missing,
        evidence_uncertainty={"n_independent_sources": independent_count(dep_groups),
                              "n_contradictions": len(contradictions),
                              "n_documents": len(documents)},
        connector_failures=connector_failures, latency_s=round(_time.time() - t0, 3), seed=seed,
        versions={"evidence_system": "phase2-1.0", "temporal": "1.0", "claims": "claims-extract-1.0"})
    bundle.freeze()
    return bundle


# --------------------------------------------------------------------------- helpers
def _doc_from_item(item, source_type: str) -> dict:
    import hashlib
    text = f"{item.title}. {item.description}".strip()
    # content_hash identifies THIS ARTICLE (title+text) for dependence/dedup; raw_feed_hash keeps the raw
    # feed the item was discovered in, for provenance. (Using the feed hash for dedup would collapse every
    # item from one feed into a single false "duplicate" group.)
    return {"id": item.item_hash(), "source": item.source_name, "source_type": source_type,
            "url": item.link, "title": item.title, "text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "raw_feed_hash": item.raw_content_hash, "published_at": item.feed_pubdate_ts,
            "connector_id": item.connector_id, "rank": item.rank,
            "requirement_id": item.requirement_id}


def _visibility_hint(claim, documents):
    for d in documents:
        if d["id"] == claim.source_id and d.get("source_type") == "user_provided":
            return {"visibility": "private_group", "actors": [], "communication_path": "user_supplied"}
    return {}


def _leakage_audit(claims, question, as_of_ts, temporal_records):
    """Claim-level leakage: post-as-of temporal status, retrospective language in the span, or the outcome
    resolution word appearing verbatim. Returns (flagged_claim_ids, [flag dicts])."""
    q_terms = set(w.lower() for w in question.split() if len(w) > 4)
    flagged, flags = [], []
    for c in claims:
        cid = c.claim_id
        reasons = []
        st = (temporal_records.get(c.source_id) or {}).get("status", "")
        if st in ("likely_post_asof", "verified_post_asof"):
            reasons.append("post_as_of_source")
        span = c.supporting_span or ""
        if any(p.search(span) for p in RETROSPECTIVE_PATTERNS):
            reasons.append("retrospective_language")
        if reasons:
            flagged.append(cid)
            flags.append({"claim_id": cid, "reason_codes": reasons, "severity": "high",
                          "span": span[:120], "exclusion": "recommend_exclude"})
    return flagged, flags


def _coverage(requirements, claims, included_ids):
    inc = set(included_ids)
    cov = {}
    for r in requirements:
        rid = r.requirement_id
        mine = [c for c in claims if (c.provenance or {}).get("requirement_id") == rid]
        n_inc = sum(1 for c in mine if c.claim_id in inc)
        cov[rid] = {"status": "fulfilled" if n_inc else ("partial" if mine else "unmet"),
                    "n_claims": len(mine), "n_included": n_inc}
    return cov
