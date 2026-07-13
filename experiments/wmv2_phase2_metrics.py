"""Phase 2 evidence — subsystem metrics from the persisted immutable bundles + a live temporal audit.

Computes claim / entity / dependence / contradiction / visibility metrics over every persisted
EvidenceBundleV2 (no new LLM calls), and runs a LIVE temporal-verification audit (archive.org Wayback) on a
sample of retrieved news URLs to measure the verified-pre-as-of rate. Also builds the machine-readable
evidence-requirement corpus from the bundles. Honest sample sizes are reported (not extrapolated).
Run: PYTHONPATH=. python -m experiments.wmv2_phase2_metrics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BUNDLES = Path("experiments/results/phase2_bundles")
METRICS = "experiments/results/wmv2_phase2_subsystem_metrics.json"
REQ_CORPUS = "experiments/results/wmv2_phase2_requirement_corpus.json"


def _load_bundles():
    return [json.loads(f.read_text()) for f in BUNDLES.glob("*.json")] if BUNDLES.exists() else []


def bundle_metrics(bundles):
    claims = [c for b in bundles for c in b.get("claims", [])]
    span_ok = [c for c in claims if c.get("span_verified")]
    cls = {}
    for c in claims:
        cls[c.get("claim_class", "?")] = cls.get(c.get("claim_class", "?"), 0) + 1
    ents = [e for b in bundles for e in b.get("entities", [])]
    ambiguous = [e for e in ents if not e.get("resolved") and len(e.get("candidates", [])) > 1]
    dep = [g for b in bundles for g in b.get("dependence_groups", [])]
    syndicated = [g for g in dep if g.get("dependence_type") in ("exact_duplicate", "near_duplicate", "syndication")]
    contras = [e for b in bundles for e in b.get("contradiction_graph", [])]
    ctypes = {}
    for e in contras:
        ctypes[e.get("ctype", "?")] = ctypes.get(e.get("ctype", "?"), 0) + 1
    vis = [v for b in bundles for v in b.get("actor_visibility", [])]
    vstates = {}
    for v in vis:
        vstates[v.get("visibility", "?")] = vstates.get(v.get("visibility", "?"), 0) + 1
    temporal = {}
    for b in bundles:
        for d in b.get("documents", []):
            s = d.get("temporal_status", "?")
            temporal[s] = temporal.get(s, 0) + 1
    n_docs = sum(len(b.get("documents", [])) for b in bundles)
    n_indep = sum(b.get("evidence_uncertainty", {}).get("n_independent_sources", 0) for b in bundles)
    return {
        "n_bundles": len(bundles), "n_bundles_with_evidence": sum(1 for b in bundles if b.get("documents")),
        "claims": {"total": len(claims), "span_verified": len(span_ok),
                   "span_verified_rate": round(len(span_ok) / max(1, len(claims)), 4), "by_class": cls},
        "entities": {"total_mentions": len(ents), "ambiguity_preserved": len(ambiguous),
                     "ambiguity_rate": round(len(ambiguous) / max(1, len(ents)), 4)},
        "dependence": {"n_groups": len(dep), "syndicated_or_dup_groups": len(syndicated),
                       "n_documents": n_docs, "n_independent_sources": n_indep,
                       "dedup_reduction": round(1 - n_indep / max(1, n_docs), 4) if n_docs else 0.0},
        "contradictions": {"total": len(contras), "by_type": ctypes},
        "visibility": {"total": len(vis), "by_state": vstates},
        "temporal": {"by_status": temporal,
                     "post_asof_in_bundle": temporal.get("likely_post_asof", 0) + temporal.get("verified_post_asof", 0)},
    }


def live_temporal_audit(bundles, *, sample=40):
    """Wayback-verify a sample of retrieved news URLs; report the verified-pre-as-of rate + any post-as-of."""
    from swm.world_model_v2.evidence_temporal import TemporalVerifier
    from swm.world_model_v2.state import parse_time
    ver = TemporalVerifier(verify_online=True, margin_days=1.0, timeout=12)
    urls = []
    for b in bundles:
        as_of = b.get("as_of")
        try:
            as_of_ts = parse_time(as_of) if isinstance(as_of, str) else float(as_of)
        except Exception:
            continue
        for d in b.get("documents", []):
            if d.get("url", "").startswith("http") and d.get("published_at"):
                urls.append((d["url"], d["published_at"], as_of_ts))
    urls = urls[:sample]
    statuses = {}
    verified_pre = post_asof = 0
    for url, claimed, as_of_ts in urls:
        r = ver.verify(as_of=as_of_ts, claimed_ts=claimed, url=url)
        statuses[r.status] = statuses.get(r.status, 0) + 1
        if r.status == "verified_pre_asof":
            verified_pre += 1
        if r.status in ("likely_post_asof", "verified_post_asof"):
            post_asof += 1
    return {"n_audited": len(urls), "statuses": statuses,
            "verified_pre_asof": verified_pre, "post_asof_in_admitted": post_asof,
            "note": "live archive.org Wayback verification on real retrieved news URLs"}


def requirement_corpus(bundles):
    reqs = []
    for b in bundles:
        for r in b.get("requirements", []):
            reqs.append({"requirement_id": r.get("requirement_id"), "need": r.get("claim_or_quantity"),
                         "affected": r.get("affected_component"), "voi": r.get("expected_voi"),
                         "as_of": r.get("as_of_constraint"), "question": b.get("question")})
    return {"schema_version": "phase2-requirement-corpus-1.0", "n_requirements": len(reqs),
            "requirements": reqs}


def run(temporal_sample=40):
    t0 = time.time()
    bundles = _load_bundles()
    if not bundles:
        raise SystemExit("no persisted bundles — run wmv2_phase2_evidence_validation first")
    m = bundle_metrics(bundles)
    print("bundle metrics:", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in list(v.items())[:4]})
                                         for k, v in m.items()}, indent=1)[:1200])
    ta = live_temporal_audit(bundles, sample=temporal_sample)
    print("\nlive temporal audit:", json.dumps(ta, indent=1))
    corpus = requirement_corpus(bundles)
    Path(REQ_CORPUS).write_text(json.dumps(corpus, indent=1, default=str))
    out = {"bundle_metrics": m, "live_temporal_audit": ta,
           "_meta": {"runtime_s": round(time.time() - t0, 1), "n_bundles": len(bundles)}}
    Path(METRICS).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {METRICS} + {REQ_CORPUS} ({corpus['n_requirements']} requirements)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-sample", type=int, default=40)
    a = ap.parse_args()
    run(a.temporal_sample)
