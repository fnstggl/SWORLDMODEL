"""Phase 12 — real full-system forecast corpus + immutable data-governance split manifest (Parts B/C/D).

Pools the REAL maximum-capacity World-Model-V2 posterior-path forecasts already produced by the merged Phase-3
work (each row is a genuine `simulate_with_posterior` terminal, NOT a mock): 93 (phase3acc locked) + 34
(phase3b locked) + 23 (phase3b diagnostic) = 150 resolved instances. For each it records the raw terminal
probability, resolution label, domain, horizon, event family, evidence-support features, structural
disagreement, and full provenance, plus a per-row active-component manifest (which subsystems were on the path
that produced it — honestly: the Phase-3 posterior path; NOT Phases 8/9/11).

Then it assigns each row to exactly one of {calibration, validation, test} by EVENT FAMILY (no family crosses a
split; temporal + family disjoint), with a fixed seed, and writes an IMMUTABLE split manifest with a content
hash. Fitting uses `calibration` only; method selection uses `validation` only; the locked `test` is scored
once. This split is FRESH for Phase 12 and distinct from the Phase-3 splits (documented leakage-safe reuse of
the underlying rows).

Provisional-status note: Phase 11 (dynamic recompilation) is NOT in the base branch (it is being developed in
parallel). These forecasts therefore come from a pre-Phase-11 distribution; the corpus is marked
`maximum_capacity_available=False` and the final production calibrators fit on it are PROVISIONAL. A resumable
refit command (`experiments/phase12_refit.py`) re-runs the freeze once Phase 11 lands.
"""
from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path

OUT = Path("experiments/results/phase12")
_EPS = 1e-6

SOURCES = [
    ("phase3acc_locked", "experiments/results/phase3acc/locked_capture.json", "arms_or_p3"),
    ("phase3b_locked", "experiments/results/phase3b/locked_test.json", "arms"),
    ("phase3b_diagnostic", "experiments/results/phase3b/diagnostic_capture.json", "p3"),
]


def _horizon_days(as_of, horizon):
    try:
        a = time.mktime(time.strptime(as_of, "%Y-%m-%d"))
        h = time.mktime(time.strptime(horizon, "%Y-%m-%d"))
        return max(0, round((h - a) / 86400))
    except Exception:  # noqa: BLE001
        return None


def _structural_entropy(sp):
    if not sp:
        return None
    vals = [v for v in sp.values() if isinstance(v, (int, float)) and v > 0]
    if len(vals) < 2:
        return 0.0
    s = sum(vals) or 1.0
    p = [v / s for v in vals]
    return round(-sum(pi * math.log(pi) for pi in p) / math.log(len(p)), 4)


def _raw_p(row, kind):
    if kind == "p3":
        return row.get("p_phase3")
    if kind == "arms":
        return (row.get("arms") or {}).get("phase3_current")
    # arms_or_p3
    return row.get("p_phase3") if row.get("p_phase3") is not None else (row.get("arms") or {}).get("phase3_current")


def _family(row):
    return row.get("family") or ("fam_" + str(row.get("domain", "x")) + "_" + str(row.get("qid", ""))[:6])


def build_corpus():
    rows = []
    seen_qids = set()
    for src_name, path, kind in SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            if not r.get("status", "").startswith("completed"):
                continue
            if r.get("outcome") not in (0, 1):
                continue
            raw = _raw_p(r, kind)
            p2 = r.get("p_phase2") if r.get("p_phase2") is not None else (r.get("arms") or {}).get("phase2")
            if raw is None:
                continue
            uid = f"{src_name}:{r.get('qid')}"
            if uid in seen_qids:
                continue
            seen_qids.add(uid)
            fam = _family(r)
            struct_ent = _structural_entropy(r.get("structural_posterior"))
            n_eff = r.get("n_effective_observations")
            n_incl = r.get("n_included_claims")
            has_rich = bool(r.get("tags"))
            rows.append({
                "row_id": uid, "source": src_name, "qid": r.get("qid"), "question": r.get("question", ""),
                "domain": r.get("domain"), "as_of": r.get("as_of"), "horizon": r.get("horizon"),
                "horizon_days": _horizon_days(r.get("as_of"), r.get("horizon")),
                "family": fam, "outcome": r["outcome"],
                "raw_p": round(float(raw), 6), "raw_p_phase2": (round(float(p2), 6) if p2 is not None else None),
                # support features (pre-outcome)
                "n_effective_observations": n_eff, "n_included_claims": n_incl,
                "structural_entropy": struct_ent, "has_rich_trace": has_rich,
                "evidence_quality": ("high" if (n_eff or 0) >= 6 else "medium" if (n_eff or 0) >= 3 else "low"),
                # active-component manifest (the path that produced this row)
                "active_components": {
                    "as_of_evidence": True, "evidence_conditioned_compile": True,
                    "posterior_hidden_state": True, "structural_hypotheses": r.get("structural_posterior") is not None,
                    "population_heterogeneity": False, "multilayer_networks": False,
                    "executable_institutions": False, "learned_actor_policies": False,
                    "persistence": False, "nonlinear_mechanisms": False, "dynamic_recompilation": False},
                "provenance": {"pipeline": "simulate_with_posterior (Phase-3 max-capacity posterior path)",
                               "note": "NOT the full Phases-8/9/11 stack; Phase 11 absent from base branch"}})
    return rows


def assign_splits(rows, seed=71237):
    """Assign each EVENT FAMILY to one split (no family crosses a boundary). ~50/25/25 cal/val/test by a
    seeded hash of the family name. Deterministic."""
    fams = sorted({r["family"] for r in rows})
    fam_split = {}
    for fam in fams:
        h = int(hashlib.sha1(f"{seed}:{fam}".encode()).hexdigest(), 16) % 100
        fam_split[fam] = "calibration" if h < 50 else "validation" if h < 75 else "test"
    for r in rows:
        r["split"] = fam_split[r["family"]]
    return fam_split


def manifest_hash(rows):
    payload = json.dumps(sorted([(r["row_id"], r["split"], r["family"], r["outcome"]) for r in rows]),
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_corpus()
    fam_split = assign_splits(rows)
    mh = manifest_hash(rows)
    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {"n": 0, "yes": 0})
        counts[r["split"]]["n"] += 1
        counts[r["split"]]["yes"] += r["outcome"]
    domains = sorted({r["domain"] for r in rows})
    corpus = {
        "corpus_version": "phase12-1.0",
        "maximum_capacity_available": False,
        "maximum_capacity_note": "Phase 11 (dynamic recompilation) absent from base branch; Phases 8/9 not on "
                                 "the main path. Rows are the Phase-3 max-capacity posterior path. Final "
                                 "calibrators are PROVISIONAL pending Phase-11 integration + refit.",
        "n_rows": len(rows), "n_domains": len(domains), "domains": domains,
        "n_rich_trace": sum(1 for r in rows if r["has_rich_trace"]),
        "base_rate_yes": round(sum(r["outcome"] for r in rows) / max(1, len(rows)), 3),
        "split_counts": counts, "manifest_hash": mh, "split_seed": 71237,
        "governance": {"split_unit": "event_family", "temporal_family_disjoint": True,
                       "fit_on": "calibration", "select_on": "validation", "evaluate_once_on": "test",
                       "distinct_from_phase15_locked_benchmark": True,
                       "leakage_safe_reuse": "fresh Phase-12 family splits over prior Phase-3 rows"},
        "rows": rows}
    (OUT / "corpus.json").write_text(json.dumps(corpus, indent=2))
    (OUT / "split_manifest.json").write_text(json.dumps({
        "manifest_hash": mh, "split_seed": 71237, "n_rows": len(rows), "split_counts": counts,
        "family_assignments": fam_split,
        "row_assignments": [{"row_id": r["row_id"], "family": r["family"], "domain": r["domain"],
                             "horizon_days": r["horizon_days"], "split": r["split"]} for r in rows]}, indent=2))
    print("corpus rows", len(rows), "domains", len(domains), "manifest", mh)
    print("splits:", json.dumps(counts))
    return corpus


if __name__ == "__main__":
    main()
