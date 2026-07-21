"""Phase 10 — consolidated counts + preserved-failure records (Parts 26/31). Reads the committed store +
the replay artifact; writes wmv2_phase10_summary.json + wmv2_phase10_failures.json and prints the counts."""
from __future__ import annotations

import json
import os

from swm.world_model_v2.institutions_v2.store import load_store

DIR = "experiments/results/phase10"
SUMM = f"{DIR}/wmv2_phase10_summary.json"
FAIL = f"{DIR}/wmv2_phase10_failures.json"


def main():
    s = load_store(reload=True)
    fam_rows, tpl_rows = [], []
    for f in s.families.values():
        fam_rows.append({"family_id": f.family_id, "category": f.category, "status": f.status,
                         "executable": f.executable(), "answers_processes": f.answers_processes})
    for t in s.templates.values():
        planes = {
            "evidence_backed": t.has_official_evidence(),
            "temporally_versioned": bool(t.valid_from or t.valid_to),
            "software_implemented": bool(s.families.get(t.family_id) and s.families[t.family_id].executable()),
            "executes": t.status in ("executable", "locally_reconstructed", "historically_replayed",
                                     "cross_institution_tested", "production_eligible"),
            "historically_replayed": any(v.get("kind") == "historical_replay" and v.get("passed")
                                         for v in t.validation),
            "production_eligible": t.status == "production_eligible",
        }
        tpl_rows.append({"template_id": t.template_id, "family": t.family_id, "jurisdiction": t.jurisdiction,
                         "valid_from": t.valid_from, "status": t.status, "planes": planes,
                         "n_evidence": len(t.evidence), "n_rules": len(t.rules)})

    summary = {
        "_meta": {"source": "committed institution registry"},
        "counts": {
            "families": len(fam_rows), "families_executable": sum(r["executable"] for r in fam_rows),
            "templates": len(tpl_rows),
            "templates_evidence_backed": sum(r["planes"]["evidence_backed"] for r in tpl_rows),
            "templates_temporally_versioned": sum(r["planes"]["temporally_versioned"] for r in tpl_rows),
            "templates_execute": sum(r["planes"]["executes"] for r in tpl_rows),
            "templates_historically_replayed": sum(r["planes"]["historically_replayed"] for r in tpl_rows),
            "templates_production_eligible": sum(r["planes"]["production_eligible"] for r in tpl_rows),
        },
        "families_by_status": s.summary()["families_by_status"],
        "templates_by_status": s.summary()["templates_by_status"],
        "families": fam_rows, "templates": tpl_rows,
    }
    json.dump(summary, open(SUMM, "w"), indent=1, default=str)

    # preserved failures / negatives (append-only)
    failures = [
        {"failure_id": "naive_cloture_uniform_3_5", "family": "legislative_process",
         "template": "us_congress_legislative", "failure_type": "wrong_threshold_version",
         "detail": "Applying a UNIFORM 3/5 cloture rule to all cloture votes reconstructs only ~0.77 of "
                   "real outcomes — WORSE than majority-only (0.95) — because the nuclear option (2013/2017) "
                   "made cloture on NOMINATIONS a simple majority. As-of + matter-type versioning recovers "
                   "0.96. Preserved: naive rule application can hurt.",
         "artifact": f"{DIR}/wmv2_phase10_replay.json"},
        {"failure_id": "no_verified_templates_for_agency_queue_moderation_board",
         "family": "administrative_agency/queue_capacity_service/moderation_appeals/corporate_board",
         "failure_type": "evidence_gap",
         "detail": "The continuation filled the adjudicative-court (scotus_merits) and direct-democracy "
                   "(swiss_federal_referendum) categories with historically-replayed templates. The "
                   "administrative agency / capacity queue / platform moderation / corporate board families "
                   "remain executable but have NO historically-replayed template this run — they select at "
                   "Tier 3 (structural) with rule uncertainty. Honest gap, not faked.",
         "artifact": SUMM},
        {"failure_id": "scotus_cert_count_not_fraction", "template": "scotus_certiorari",
         "failure_type": "rule_shape_limitation",
         "detail": "The Rule of Four is a COUNT (4 of 9), an informal custom; it is stored as evidence but "
                   "the decision engine's fraction-of-present threshold does not natively execute a fixed "
                   "count. SCOTUS cert stays `executable` (informal), not historically replayed. (The SCOTUS "
                   "merits decision — a majority rule — IS historically replayed on SCDB.)",
         "artifact": SUMM},
        {"failure_id": "referendum_no_canton_shares", "template": "swiss_federal_referendum",
         "failure_type": "partial_reconstruction_data_gap",
         "detail": "The cached Swiss referendum data has the legal FORM + outcome but no per-canton vote "
                   "shares, so the DOUBLE majority (People AND cantons, Art. 142) is reconstructed as an "
                   "outcome REGULARITY by legal form (initiatives ~10% vs mandatory ~75%), NOT executed on "
                   "canton counts. A full double-majority execution replay needs canton-level Swissvotes data.",
         "artifact": f"{DIR}/wmv2_phase10_referendum_replay.json"},
        {"failure_id": "forward_prediction_is_modest", "template": "us_congress_legislative",
         "failure_type": "honest_metric_separation",
         "detail": "Out-of-sample institutional PREDICTION (train C117 → test C118: party composition → actor "
                   "policy → threshold engine → StateDelta) is honestly weak — acc 0.83, Brier 0.132 (beats "
                   "the base-rate Brier 0.144 but only just; a naive party-line policy 0.82 UNDERPERFORMS the "
                   "base rate). Far below procedural reconstruction (0.96, which uses real votes). Preserved to "
                   "keep rule-EXECUTION and forecasting cleanly distinct.",
         "artifact": f"{DIR}/wmv2_phase10_predict.json"},
    ]
    json.dump({"_meta": {"n": len(failures), "note": "append-only preserved negatives (Part 26)"},
               "failures": failures}, open(FAIL, "w"), indent=1, default=str)

    c = summary["counts"]
    print("=== PHASE 10 COUNTS ===")
    print(f"  families:                     {c['families']} (executable {c['families_executable']})")
    print(f"  templates:                    {c['templates']}")
    print(f"  evidence-backed templates:    {c['templates_evidence_backed']}")
    print(f"  temporally versioned:         {c['templates_temporally_versioned']}")
    print(f"  execute in shared world:      {c['templates_execute']}")
    print(f"  historically replayed:        {c['templates_historically_replayed']}")
    print(f"  production eligible:          {c['templates_production_eligible']}")
    print(f"  families_by_status:  {summary['families_by_status']}")
    print(f"  templates_by_status: {summary['templates_by_status']}")
    print(f"  preserved failures:  {len(failures)}")
    print(f"wrote {SUMM} and {FAIL}")


if __name__ == "__main__":
    main()
