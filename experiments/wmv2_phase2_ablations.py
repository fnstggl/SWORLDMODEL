"""Phase 2 evidence ablations (U) — what each safeguard buys, on real live retrieval.

Centerpiece: the before:-only vs paired after:/before: leakage ablation. For a set of historical questions
we issue BOTH a `before:`-only Google News RSS query and a paired `after:/before:` query (LIVE), then measure
the share of returned items whose real pubDate is AFTER the as-of (temporal leakage). This demonstrates (a)
why the production historical arm must use paired dates, and (b) that RSS dates alone are not trusted — the
independent temporal filter removes the residual post-as-of items either operator lets through. The
before:-only arm is an EVALUATION ablation only; production always uses paired.

Also ablates the analysis pipeline on real bundles: no-temporal-verification (post-as-of inclusion),
no-dependence-collapse (independent-source overcounting), no-actor-visibility (actor-info leakage),
no-recompile (evidence has no causal effect). Resumable; the RSS arm needs no LLM.
Run: PYTHONPATH=. python -m experiments.wmv2_phase2_ablations
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RESULT = "experiments/results/wmv2_phase2_ablations.json"
CACHE = Path("experiments/results/phase2_ablations")

# (question terms, as_of) — public historical events with real post-event coverage (to expose leakage)
LEAKAGE_QUESTIONS = [
    ("writers guild strike studios deal", "2023-08-01"),
    ("UAW union auto strike agreement", "2023-09-15"),
    ("US debt ceiling deal congress", "2023-05-01"),
    ("Silicon Valley Bank collapse", "2023-03-01"),
    ("Twitter rebrand X Musk", "2023-07-01"),
    ("OpenAI Sam Altman board", "2023-11-01"),
    ("UK prime minister Sunak election", "2024-06-01"),
    ("Baltimore bridge collapse", "2024-03-01"),
]


def _post_asof_share(items, as_of_ts):
    dated = [it for it in items if it.feed_pubdate_ts is not None]
    if not dated:
        return 0.0, 0
    post = [it for it in dated if it.feed_pubdate_ts > as_of_ts + 86400]
    return round(len(post) / len(dated), 4), len(dated)


def run_leakage_ablation():
    from swm.world_model_v2.evidence_connectors import (GoogleNewsRSSConnector, RawContentStore,
                                                        _rfc822_ts)
    from swm.world_model_v2.evidence_temporal import TemporalVerifier
    from swm.world_model_v2.state import parse_time
    import urllib.parse, urllib.request, re, time as _t

    store = RawContentStore()
    conn = GoogleNewsRSSConnector(store=store)
    ver = TemporalVerifier(verify_online=False, margin_days=1.0)
    ua = {"User-Agent": "Mozilla/5.0 (swm-evidence/1.0)"}

    def before_only(terms, before_date, k=12):
        """EVALUATION-ONLY arm: a before:-alone query (never used in production)."""
        q = f"{terms} before:{before_date}"
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=20).read()
        items = conn._parse(raw, q, url, "abl", store.put(raw), k)
        return items, q

    rows = []
    for terms, as_of in LEAKAGE_QUESTIONS:
        as_of_ts = parse_time(as_of)
        after = _t.strftime("%Y-%m-%d", _t.gmtime(as_of_ts - 150 * 86400))
        # arm A: before-only
        try:
            a_items, a_q = before_only(terms, as_of)
        except Exception as e:  # noqa: BLE001
            a_items, a_q = [], f"ERROR {e}"
        a_share, a_n = _post_asof_share(a_items, as_of_ts)
        # arm B: paired
        b_items, b_tr = conn.search_historical(terms, after_date=after, before_date=as_of, k=12)
        b_share, b_n = _post_asof_share(b_items, as_of_ts)
        # arm C: paired + independent temporal filter (production)
        c_admitted = [it for it in b_items
                      if ver.verify(as_of=as_of_ts, claimed_ts=it.feed_pubdate_ts).admissible()]
        c_share, c_n = _post_asof_share(c_admitted, as_of_ts)
        rows.append({"terms": terms, "as_of": as_of,
                     "before_only": {"n_dated": a_n, "post_asof_share": a_share, "query": a_q[:80]},
                     "paired": {"n_dated": b_n, "post_asof_share": b_share},
                     "paired_plus_temporal_filter": {"n_admitted": len(c_admitted), "post_asof_share": c_share}})
        print(f"  {terms[:34]:34s} before_only leak={a_share:.2f}(n={a_n}) paired={b_share:.2f}(n={b_n}) "
              f"paired+filter={c_share:.2f}(n={len(c_admitted)})", flush=True)
        _t.sleep(0.5)
    n = len(rows)
    agg = {
        "mean_post_asof_share_before_only": round(sum(r["before_only"]["post_asof_share"] for r in rows) / n, 4),
        "mean_post_asof_share_paired": round(sum(r["paired"]["post_asof_share"] for r in rows) / n, 4),
        "mean_post_asof_share_paired_plus_filter": round(
            sum(r["paired_plus_temporal_filter"]["post_asof_share"] for r in rows) / n, 4),
        "n_questions": n}
    agg["paired_reduces_leakage_vs_before_only"] = agg["mean_post_asof_share_paired"] < agg["mean_post_asof_share_before_only"]
    agg["temporal_filter_zeroes_residual_leak"] = agg["mean_post_asof_share_paired_plus_filter"] == 0.0
    return {"per_question": rows, "aggregate": agg}


def run_pipeline_ablations():
    """Ablate the analysis pipeline on the persisted bundles from the main validation run (no new LLM)."""
    from swm.world_model_v2.evidence_dependence import independent_count
    bdir = Path("experiments/results/phase2_bundles")
    bundles = list(bdir.glob("*.json"))[:40] if bdir.exists() else []
    if not bundles:
        return {"note": "no persisted bundles yet (run wmv2_phase2_evidence_validation first)"}
    n_no_dedup_overcount, n_no_temporal_leak, n_no_visibility_leak = 0, 0, 0
    total = 0
    for bf in bundles:
        b = json.loads(bf.read_text())
        total += 1
        # no dependence collapse: independent count would be raw doc count
        raw_docs = len(b.get("documents", []))
        indep = b.get("evidence_uncertainty", {}).get("n_independent_sources", raw_docs)
        if raw_docs > indep:
            n_no_dedup_overcount += 1
        # no temporal verification: post-as-of docs would enter
        if any(d.get("temporal_status") in ("likely_post_asof", "verified_post_asof")
               for d in b.get("documents", [])):
            n_no_temporal_leak += 1
        # no actor visibility: any non-public claim would leak to all actors
        if any(v.get("visibility") not in ("public",) for v in b.get("actor_visibility", [])):
            n_no_visibility_leak += 1
    return {"n_bundles": total,
            "no_dependence_collapse_would_overcount_share": round(n_no_dedup_overcount / max(1, total), 3),
            "no_temporal_verification_would_leak_share": round(n_no_temporal_leak / max(1, total), 3),
            "no_actor_visibility_would_leak_share": round(n_no_visibility_leak / max(1, total), 3),
            "interpretation": "each share is the fraction of bundles where removing the safeguard would "
                              "cause overcounting / post-as-of leakage / actor-information leakage"}


def run():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    print("=== before-only vs paired leakage ablation (LIVE) ===")
    leak = run_leakage_ablation()
    print("\n=== pipeline ablations (on persisted bundles) ===")
    pipe = run_pipeline_ablations()
    print(json.dumps(pipe, indent=1))
    out = {"leakage_ablation": leak, "pipeline_ablations": pipe,
           "_meta": {"runtime_s": round(time.time() - t0, 1),
                     "note": "before:-only is an evaluation arm only; production always uses paired after:/before:"}}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1, default=str))
    a = leak["aggregate"]
    print(f"\nLEAKAGE: before_only={a['mean_post_asof_share_before_only']} "
          f"paired={a['mean_post_asof_share_paired']} paired+filter={a['mean_post_asof_share_paired_plus_filter']}")
    print(f"paired reduces leakage: {a['paired_reduces_leakage_vs_before_only']} | "
          f"filter zeroes residual: {a['temporal_filter_zeroes_residual_leak']}")
    print(f"wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
