"""Consolidate the Phase 6 primary-literature research clusters into machine-readable artifacts.

Reads experiments/results/phase6_research/cluster_*.json (produced by the research workstream — each is an
array of verified family research records) and writes:

  swm/world_model_v2/registry/data/priority_matrix.json  — the Phase 6 priority matrix (the mandated
      research-first deliverable): family, current status, formal model, sources, datasets, validation
      route, applicability, transport risk, priority, planned packs, promotion blocker, disposition.
  swm/world_model_v2/registry/data/studies.json          — the study registry: every VERIFIED primary
      source, deduplicated, with what it provides and which families it supports.
  swm/world_model_v2/registry/data/coefficients.json     — the index of VERIFIED reported coefficients
      (source-attributed numbers usable as published_research pack values; unverified ones are excluded).

Run: PYTHONPATH=. python -m experiments.wmv2_phase6_matrix
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

SRC = "experiments/results/phase6_research"
DATA = "swm/world_model_v2/registry/data"

MATRIX_COLS = ["family", "registry_id", "current_registry_status", "current_executable_status",
               "formal_model", "evidence_type", "temporal_scale", "datasets_in_literature",
               "local_refit_feasible", "best_validation_route", "expected_applicability",
               "expected_transport_risk", "priority", "planned_implementation",
               "planned_parameter_packs", "promotion_blocker", "disposition"]


def _write(path, payload):
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    digest = hashlib.sha256(body.encode()).hexdigest()
    doc = {"_integrity": {"sha256": digest, "n": len(payload) if isinstance(payload, (list, dict)) else 0},
           "payload": payload}
    with open(path, "w") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return digest


def main():
    files = sorted(glob.glob(f"{SRC}/cluster_*.json"))
    matrix, studies, coefs = [], {}, []
    clusters_seen = []
    for fp in files:
        cluster = os.path.basename(fp).replace("cluster_", "").replace(".json", "")
        clusters_seen.append(cluster)
        try:
            recs = json.load(open(fp))
        except Exception as e:
            print(f"  !! {fp}: {e}")
            continue
        if isinstance(recs, dict) and "payload" in recs:
            recs = recs["payload"]
        for r in recs:
            row = {c: r.get(c) for c in MATRIX_COLS}
            row["cluster"] = cluster
            # attach the single strongest source ref for quick scanning
            srcs = r.get("primary_sources") or []
            row["n_sources"] = len(srcs)
            row["n_verified_sources"] = sum(1 for s in srcs if s.get("verified"))
            matrix.append(row)
            for s in srcs:
                key = (s.get("doi_or_url") or s.get("title") or "").strip().lower()
                if not key:
                    continue
                st = studies.setdefault(key, {"title": s.get("title"), "authors": s.get("authors"),
                                              "year": s.get("year"), "venue": s.get("venue"),
                                              "doi_or_url": s.get("doi_or_url"),
                                              "verified": bool(s.get("verified")),
                                              "what_it_provides": s.get("what_it_provides"),
                                              "supports_families": []})
                if r.get("family") not in st["supports_families"]:
                    st["supports_families"].append(r.get("family"))
                st["verified"] = st["verified"] or bool(s.get("verified"))
            for c in (r.get("reported_coefficients") or []):
                if c.get("verified") and c.get("value_or_null") is not None:
                    coefs.append({"family": r.get("family"), "name": c.get("name"),
                                  "value": c.get("value_or_null"), "uncertainty": c.get("uncertainty_or_null"),
                                  "source_ref": c.get("source_ref"), "context": c.get("context")})

    os.makedirs(DATA, exist_ok=True)
    h1 = _write(f"{DATA}/priority_matrix.json", matrix)
    h2 = _write(f"{DATA}/studies.json", list(studies.values()))
    h3 = _write(f"{DATA}/coefficients.json", coefs)

    # summary
    from collections import Counter
    disp = Counter(r["disposition"] for r in matrix if r.get("disposition"))
    prio = Counter(r["priority"] for r in matrix if r.get("priority"))
    print(f"clusters: {clusters_seen}")
    print(f"families in matrix: {len(matrix)}")
    print(f"dispositions: {dict(disp)}")
    print(f"priorities: {dict(prio)}")
    print(f"verified studies: {sum(1 for s in studies.values() if s['verified'])}/{len(studies)}")
    print(f"verified coefficients: {len(coefs)}")
    print("high-priority families:",
          sorted(r["family"] for r in matrix if r.get("priority") == "high"))
    print(f"wrote priority_matrix.json ({h1[:8]}), studies.json ({h2[:8]}), coefficients.json ({h3[:8]})")


if __name__ == "__main__":
    main()
