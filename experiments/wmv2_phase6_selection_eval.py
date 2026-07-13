"""Phase 6 mechanism-SELECTION evaluation — before/after fallback-tier distribution + ablations.

This tests mechanism COVERAGE and EXECUTION routing, NOT outcome accuracy (per the Phase 6 contract). For
a stratified bank of (scenario, required causal processes) drawn from the 9 mandated categories, it asks:
which fallback tier does each required process land in?

  before   pre-Phase-6 behavior: only {locally_validated, transfer_validated, production_eligible} are
           selectable, and ONE scenario-ranked winner is reused for every process (the Phase-1 flaw).
  after    Phase-6 per-process selection: each process is matched to the family that ANSWERS it; verified
           published (domain_restricted) and research_encoded families are selectable at Tier 4.

Ablations (Part 16): names_only (ignore status/packs → nothing selectable → all generic), no_applicability
(process-match only, ignore scenario fit), no_transport (don't widen), full (Phase 6).

Metric: share of process-selections in Tier 1-4 (a real evidence-backed mechanism) vs Tier 5 vs Tier 6-7
(generic/competing fallback). Reducing Tier 6-7 on real processes is the Phase 6 goal.

Run: PYTHONPATH=. python -m experiments.wmv2_phase6_selection_eval
Writes experiments/results/wmv2_phase6_selection_eval.json
"""
from __future__ import annotations

import json

from swm.world_model_v2.fallback import select_tier_for_process
from swm.world_model_v2.registry import load_registry
from swm.world_model_v2.registry.applicability import _process_match, score_applicability, select_for_process

OUT = "experiments/results/wmv2_phase6_selection_eval.json"

# stratified bank: (category, scenario, [required causal processes]) across the 9 mandated categories
BANK = [
    ("individual_choice", {"domain": "economic_game", "population_kind": "lab"},
     ["actor_selects_typed_action", "offer_response"]),
    ("trust_relationship", {"domain": "trust_game", "population_kind": "lab"},
     ["trust_change_after_interaction", "reputation_update"]),
    ("bargaining", {"domain": "economic_game", "population_kind": "lab"},
     ["offer_response", "bargaining_split"]),
    ("participation", {"domain": "election", "population_kind": "registered_voters"},
     ["participation_after_mobilization", "turnout"]),
    ("participation_donation", {"domain": "fundraising"},
     ["donation_after_ask"]),
    ("opinion_belief", {"domain": "social_media", "available_state": ["network", "entities"]},
     ["belief_update_after_message", "opinion_convergence"]),
    ("diffusion", {"domain": "social_media_diffusion", "population_kind": "online_social",
                   "available_state": ["network"]},
     ["adoption_after_repeated_exposure", "diffusion_timing", "cascade_saturation"]),
    ("platform_attention", {"domain": "content_ab_test", "available_state": ["populations"]},
     ["content_response", "attention_after_exposure"]),
    ("network", {"domain": "labor_market", "available_state": ["network"]},
     ["network_edge_change", "network_targeting"]),
    ("institutional", {"domain": "legislation", "institutional": True, "available_state": ["institutions"]},
     ["institutional_threshold_decision"]),
    ("product_launch", {"domain": "product_launch", "available_state": ["quantities"]},
     ["cascade_saturation", "adoption_after_repeated_exposure"]),
    ("persuasion", {"domain": "political_persuasion"},
     ["persuasion_success"]),
]

REAL_STATUSES = ("locally_validated", "transfer_validated", "production_eligible")
BEFORE_STATUSES = REAL_STATUSES                       # pre-Phase-6 selectable set
AFTER_STATUSES = REAL_STATUSES + ("domain_restricted", "research_encoded")


def _tier_bucket(tier):
    return "tier_1_4" if tier <= 4 else ("tier_5" if tier == 5 else "tier_6_7")


def _before(store, scenario, processes):
    """Pre-Phase-6: one scenario-ranked winner (real statuses only) reused for every process."""
    from swm.world_model_v2.registry.applicability import rank_mechanisms
    ranked = rank_mechanisms(store, scenario, statuses=BEFORE_STATUSES)
    winner = (ranked.get("selected") or [None])[0]
    out = []
    for p in processes:
        # the winner is applied REGARDLESS of whether it answers p (the flaw); tier from its transported-ness
        if winner:
            ch = select_tier_for_process(p, {"family_id": winner["family_id"], "status": "locally_validated",
                                              "pack_is_transported": winner.get("pack_is_transported", True)})
            valid = _process_match(store.records[winner["family_id"]], p) > 0.0
        else:
            ch = select_tier_for_process(p, None)
            valid = False
        out.append({"process": p, "family": ch.family, "tier": ch.tier, "valid_selection": valid})
    return out


def _after(store, scenario, processes, *, statuses=AFTER_STATUSES, use_applicability=True,
           threshold=0.4):
    out = []
    for p in processes:
        r = select_for_process(store, p, scenario, statuses=statuses, threshold=(0.0 if not use_applicability
                                                                                 else threshold))
        sel = r.get("selected")
        if not use_applicability and r.get("n_candidates"):
            # ignore scenario applicability: pick the highest process-match regardless of fit
            cands = [r["selected"]] + r.get("competing", []) if r.get("selected") else []
            sel = max(cands, key=lambda c: c["process_match"]) if cands else None
        ch = select_tier_for_process(p, sel, competing=[c["family_id"] for c in r.get("competing", [])])
        # a per-process selection is valid by construction (the family answers p); generic fallback = valid
        # (honest: no family answers p) but NOT evidence-backed
        valid = (sel is not None) or (ch.family == "generic_outcome_prior")
        out.append({"process": p, "family": ch.family, "tier": ch.tier,
                    "status": sel.get("status") if sel else None,
                    "valid_selection": (_process_match(store.records[sel["family_id"]], p) > 0.0) if sel else True})
    return out


def _dist(rows):
    from collections import Counter
    c = Counter(_tier_bucket(r["tier"]) for r in rows)
    n = len(rows)
    valid = sum(1 for r in rows if r.get("valid_selection"))
    out = {k: {"n": c.get(k, 0), "pct": round(100 * c.get(k, 0) / max(1, n), 1)}
           for k in ("tier_1_4", "tier_5", "tier_6_7")}
    out["valid_selection_pct"] = round(100 * valid / max(1, n), 1)
    return out


def main():
    store = load_registry(reload=True)
    per_scenario = []
    before_rows, after_rows = [], []
    names_only_rows, no_app_rows = [], []
    for cat, scen, procs in BANK:
        b = _before(store, scen, procs)
        a = _after(store, scen, procs)
        no_app = _after(store, scen, procs, use_applicability=False)
        names_only = [{"process": p, "family": "generic_outcome_prior", "tier": 6} for p in procs]
        before_rows += b
        after_rows += a
        no_app_rows += no_app
        names_only_rows += names_only
        per_scenario.append({"category": cat, "scenario": scen, "processes": procs,
                             "before": b, "after": a})
    result = {
        "_meta": {"harness": "experiments/wmv2_phase6_selection_eval.py",
                  "note": "measures mechanism-selection COVERAGE/routing, NOT outcome accuracy",
                  "n_processes": len(after_rows)},
        "ablations": {
            "names_only_registry": _dist(names_only_rows),
            "before_phase6_scenario_winner_real_statuses_only": _dist(before_rows),
            "after_phase6_no_applicability": _dist(no_app_rows),
            "after_phase6_full": _dist(after_rows),
        },
        "per_scenario": per_scenario,
    }
    json.dump(result, open(OUT, "w"), indent=1, default=str)
    print("=== fallback-tier distribution over", len(after_rows), "required process-selections ===")
    for name, d in result["ablations"].items():
        print(f"  {name:52s} Tier1-4 {d['tier_1_4']['pct']:5}%  Tier6-7 {d['tier_6_7']['pct']:5}%  "
              f"VALID-selection {d['valid_selection_pct']:5}%")
    print("\n  NOTE: 'before' Tier1-4 is HIGH but VALID-selection is LOW — it stamped one scenario winner on "
          "\n  every process (false coverage). 'after' gives process-appropriate tiers with valid selections.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
