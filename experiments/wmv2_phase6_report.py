"""Phase 6 consolidated report artifacts — the honest counts + validation/failure index.

Reads the committed registry and produces:
  experiments/results/wmv2_phase6_validation_summary.json — per-family status, planes (software/executes/
      validated/production), validation records by kind, packs by source.
  experiments/results/wmv2_phase6_failures.json — every PRESERVED negative result (passed=False + quarantines).

Prints the four family counts and six pack counts for docs/WMV2_PHASE6_FINAL_REPORT.md.
Run: PYTHONPATH=. python -m experiments.wmv2_phase6_report
"""
from __future__ import annotations

import json

from swm.world_model_v2.registry import load_registry

VS = "experiments/results/wmv2_phase6_validation_summary.json"
FR = "experiments/results/wmv2_phase6_failures.json"


def main():
    s = load_registry(reload=True)
    recs = s.records
    fam_rows, failures, pack_rows = [], [], []
    for fid, r in sorted(recs.items()):
        allv = list(r.validation) + [v for p in r.packs for v in p.validation]
        passed_kinds = sorted({v.kind for v in allv if v.passed})
        failed = [{"family": fid, "kind": v.kind, "dataset": v.dataset, "split": v.split,
                   "metric": v.metric, "value": v.value, "baseline": v.baseline,
                   "baseline_value": v.baseline_value, "ci95": v.ci95, "note": v.note}
                  for v in allv if v.passed is False]
        failures += failed
        if r.status == "quarantined":
            failures.append({"family": fid, "kind": "quarantine", "dataset": "", "split": "",
                             "metric": "status", "value": None, "note": r.status_reason})
        fam_rows.append({
            "family_id": fid, "status": r.status, "ontology": r.ontology_type,
            "software_implemented": bool(r.executable() and r.test_ref),
            "executes_end_to_end": bool(r.executable()),
            "empirically_validated": any(v.kind in ("held_out", "posterior_predictive", "transfer") and v.passed
                                         for v in allv),
            "production_eligible": r.status == "production_eligible",
            "n_packs": len(r.packs), "passed_validation_kinds": passed_kinds,
            "n_failed_or_null": len(failed), "answers_processes": r.applicability.answers_processes,
        })
        for p in r.packs:
            srcs = sorted({v.get("source") for v in p.values.values() if isinstance(v, dict)})
            pack_rows.append({"pack_id": p.pack_id, "family": fid, "family_status": r.status,
                              "domain": p.domain, "param_sources": srcs,
                              "n_validation": len(p.validation),
                              "passed": sorted({v.kind for v in p.validation if v.passed}),
                              "failed": sorted({v.kind for v in p.validation if v.passed is False})})

    # four family counts
    named = len(fam_rows)
    software = sum(r["software_implemented"] for r in fam_rows)
    validated = sum(r["empirically_validated"] for r in fam_rows)
    production = sum(r["production_eligible"] for r in fam_rows)
    # pack counts by disposition
    n_packs = len(pack_rows)
    local = sum(1 for p in pack_rows if p["passed"] and "held_out" in p["passed"])
    transfer = sum(1 for p in pack_rows if "transfer" in p["passed"])
    published = sum(1 for p in pack_rows if "published_research" in p["param_sources"])
    null_or_failed = sum(1 for p in pack_rows if p["failed"])

    summary = {
        "_meta": {"source": "committed registry", "n_families": named, "n_packs": n_packs},
        "family_counts": {"named": named, "software_implemented": software,
                          "empirically_validated": validated, "production_eligible": production},
        "pack_counts": {"total": n_packs, "with_local_held_out": local, "with_transfer": transfer,
                        "published_estimate": published, "with_failed_or_null": null_or_failed},
        "by_status": {st: sorted(r["family_id"] for r in fam_rows if r["status"] == st)
                      for st in sorted({r["status"] for r in fam_rows})},
        "families": fam_rows, "packs": pack_rows,
    }
    json.dump(summary, open(VS, "w"), indent=1, default=str)
    json.dump({"_meta": {"n_preserved_negatives": len(failures),
                         "note": "append-only; never deleted (Hawkes, telco transfer, BehaviorBench PG, "
                                 "StackExchange/CMV nulls preserved)"}, "failures": failures},
              open(FR, "w"), indent=1, default=str)

    print("=== FAMILY COUNTS ===")
    print(f"  named:                {named}")
    print(f"  software_implemented: {software}")
    print(f"  empirically_validated:{validated}")
    print(f"  production_eligible:  {production}")
    print("=== PACK COUNTS ===")
    print(f"  total:                {n_packs}")
    print(f"  local held-out:       {local}")
    print(f"  transfer:             {transfer}")
    print(f"  published estimate:   {published}")
    print(f"  failed/null preserved:{null_or_failed}")
    print("=== BY STATUS ===")
    for st, fams in summary["by_status"].items():
        print(f"  {st:20s} {len(fams)}")
    print(f"=== {len(failures)} preserved negative results ===")
    print(f"wrote {VS} and {FR}")


if __name__ == "__main__":
    main()
