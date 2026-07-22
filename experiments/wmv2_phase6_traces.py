"""Phase 6 forensic traces — REAL end-to-end mechanism executions (not narrated).

For a stratified set of social questions across the mandated categories, each trace runs the ACTUAL Phase-6
path and captures every stage as machine-readable evidence:

  question → causal-process request → registry candidates (select_for_process) → applicability scores →
  rejected candidates → selected family + pack → transport assessment (assess_transport) → parameter
  posterior (pack values) → scenario instance (the bound hazard_spec) → WorldState before → triggering event
  → StateDelta (the operator is actually run) → WorldState after → terminal sensitivity (a parameter is
  varied and the terminal readout moves) → support tier + limitations.

Traces with an executable operator (behavioral / feature-hazard families) run END TO END here. Families whose
execution lives in a reference world (diffusion contagion rollout) show the selection + applicability + a
pointer to the reference-world validation. The point: a reader can DISTINGUISH an evidence-backed mechanism,
a transported one, an experimental structural family, and a generic fallback.

Run: PYTHONPATH=. python -m experiments.wmv2_phase6_traces
Writes experiments/results/wmv2_phase6_forensic_traces.json
"""
from __future__ import annotations

import json
import random

from swm.world_model_v2.events import Event
from swm.world_model_v2.fallback import select_tier_for_process
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.registry import load_registry, select_for_process
from swm.world_model_v2.registry.applicability import score_applicability
from swm.world_model_v2.registry.families.behavioral import BehavioralMechanismOperator
from swm.world_model_v2.registry.transport import assess_transport, pack_context_from_record
from swm.world_model_v2.state import Entity, SimulationClock, WorldState, parse_time

OUT = "experiments/results/wmv2_phase6_forensic_traces.json"
T0 = parse_time("2020-01-01T00:00:00Z")

# (category, question, causal_process, scenario, behavioral-exec spec | None)
CASES = [
    ("strategic_choice",
     "In a one-shot ultimatum split of $100, will the responder accept a $40 offer?",
     "offer_response", {"domain": "one_shot_bargaining", "population_kind": "lab"},
     {"mechanism": "ultimatum_offer_response", "params": {"offer_frac": 0.40, "accept_threshold": 0.25},
      "outcome_var": "accepted", "options": ["accept", "reject"], "vary": ("offer_frac", [0.05, 0.40])}),
    ("trust",
     "In an anonymous trust game, does the investor's trust pay off?",
     "trust_change_after_interaction", {"domain": "trust_game", "population_kind": "lab"},
     {"mechanism": "trust_game_transfer", "params": {"send_frac": 0.50, "return_frac": 0.37},
      "outcome_var": "trust_paid_off", "vary": ("return_frac", [0.20, 0.37])}),
    ("participation",
     "Will a 'Neighbors' social-pressure mailer raise this voter's turnout probability?",
     "participation_after_mobilization", {"domain": "election", "population_kind": "registered_voters"},
     {"mechanism": "social_pressure_turnout", "params": {"treatment": "neighbors"},
      "outcome_var": "turnout", "vary": ("treatment", ["control", "neighbors"])}),
    ("participation_donation",
     "Does announcing a matching grant raise this prior donor's probability of giving?",
     "donation_after_ask", {"domain": "fundraising"},
     {"mechanism": "matching_donation_response", "params": {"base_p": 0.20, "match_offered": True},
      "outcome_var": "donated", "vary": ("match_offered", [False, True])}),
    ("diffusion_product",
     "What fraction of the market adopts a new durable good over 4 years?",
     "cascade_saturation", {"domain": "product_launch", "available_state": ["quantities"]},
     {"mechanism": "bass_diffusion", "params": {"p": 0.03, "q": 0.38, "M": 1000.0, "steps": 48},
      "outcome_var": "majority_adopts", "vary": ("q", [0.10, 0.38])}),
    ("resource_dropout",
     "Will this month-to-month telecom subscriber churn?",
     "relationship_dropout", {"domain": "subscription_churn", "available_data": ["actor_features"]}, None),
    ("platform_attention",
     "Which of two headlines will win an A/B click test?",
     "content_response", {"domain": "content_ab_test", "available_state": ["populations"]}, None),
    ("diffusion_social",
     "Among exposed non-adopters on a follower graph, who activates in the next 24h?",
     "adoption_after_repeated_exposure", {"domain": "social_media_diffusion",
                                          "population_kind": "online_social", "available_state": ["network"]}, None),
    ("network",
     "Does friendship-nomination seeding outperform random seeding for a community intervention?",
     "network_targeting", {"domain": "public_health", "available_state": ["network"]}, None),
    ("opinion",
     "Will a single campaign contact persuade this voter to change their vote choice?",
     "persuasion_success", {"domain": "political_persuasion"}, None),
]


def _world():
    w = WorldState(world_id="trace", branch_id="root", clock=SimulationClock(now=T0, as_of=T0),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["actor"] = Entity(identity="actor")
    return w


def _run_behavioral(spec):
    """Actually execute the behavioral mechanism in a WorldState; return before/after + StateDelta + a
    terminal-sensitivity sweep over the varied parameter."""
    op = BehavioralMechanismOperator()

    def terminal_rate(params):
        hits = 0
        for seed in range(300):
            w = _world()
            w.branch_id = f"b{seed}"
            ev = Event(ts=T0, etype="behavioral_mechanism",
                       payload={"hazard_spec": {"kind": "behavioral", "mechanism": spec["mechanism"],
                                                "params": params, "outcome_var": spec["outcome_var"],
                                                "options": spec.get("options", ["True", "False"])}})
            w.clock.advance_to(ev.ts)
            op.run(w, ev, random.Random(seed))
            val = w.quantities[spec["outcome_var"]].value
            hits += int(val in (spec.get("options", ["True"])[0], "True"))
        return round(hits / 300, 3)

    # one concrete execution for the StateDelta record
    w = _world()
    ev = Event(ts=T0, etype="behavioral_mechanism",
               payload={"hazard_spec": {"kind": "behavioral", "mechanism": spec["mechanism"],
                                        "params": spec["params"], "outcome_var": spec["outcome_var"],
                                        "options": spec.get("options", ["True", "False"])}})
    before = dict(w.quantities)
    w.clock.advance_to(ev.ts)
    delta, vr = op.run(w, ev, random.Random(1))
    key, vals = spec["vary"]
    sweep = {}
    for v in vals:
        pp = dict(spec["params"])
        pp[key] = v
        sweep[str(v)] = terminal_rate(pp)
    return {
        "world_before": {"quantities": sorted(before.keys())},
        "triggering_event": {"etype": ev.etype, "payload_mechanism": spec["mechanism"]},
        "state_delta": delta.as_dict() if delta else None,
        "world_after": {spec["outcome_var"]: w.quantities[spec["outcome_var"]].value},
        "terminal_sensitivity": {"varied": key, "sweep": sweep,
                                 "moves": len(set(sweep.values())) > 1},
    }


def build_trace(store, cat, question, process, scenario, exec_spec):
    r = select_for_process(store, process, scenario)
    sel = r.get("selected")
    trace = {"category": cat, "question": question, "causal_process_request": process, "scenario": scenario}
    trace["registry_candidates"] = [c["family_id"] for c in ([sel] if sel else []) + r.get("competing", [])]
    trace["rejected_candidates"] = r.get("rejected", [])[:5]
    if not sel:
        trace["selected"] = None
        trace["support_tier"] = select_tier_for_process(process, None,
                                                        [c["family_id"] for c in r.get("competing", [])]).tier
        trace["note"] = "no evidence-backed family answers this process → generic/competing fallback"
        return trace
    rec = store.records[sel["family_id"]]
    pack = rec.packs[0] if rec.packs else None
    ch = select_tier_for_process(process, sel, [c["family_id"] for c in r.get("competing", [])])
    trace.update({
        "selected_family": sel["family_id"], "status": sel["status"],
        "applicability": {"combined": sel["combined"], "process_match": sel["process_match"],
                          "applicability": sel["applicability"], "subscores": sel["subscores"]},
        "selected_pack": pack.pack_id if pack else None,
        "parameter_posterior": (pack.values if pack else "no pack (research_encoded / structural)"),
        "transport": assess_transport(pack_context_from_record(pack), scenario).as_dict() if pack else None,
        "provenance": [{"ref": c.ref, "doi_or_url": c.doi_or_url, "limits": c.limits}
                       for c in (rec.citations or (pack.citations if pack else []))][:2],
        "support_tier": ch.tier, "support_grade": ch.support_grade,
        "competing_mechanisms": [c["family_id"] for c in r.get("competing", [])],
        "limitations": rec.applicability.exclusion_conditions + rec.known_failure_modes,
        "evidence_class": {"production_eligible": "evidence-backed (validated)",
                           "transfer_validated": "evidence-backed (transfer)",
                           "locally_validated": "evidence-backed (local held-out)",
                           "domain_restricted": "verified published estimate, transported (widened)",
                           "research_encoded": "research-encoded structural family (broad uncertainty)"}
                          .get(sel["status"], "experimental"),
    })
    if exec_spec:
        trace["execution"] = _run_behavioral(exec_spec)
    else:
        trace["execution"] = {"note": "executes in the reference world / family test "
                              f"({rec.code_ref}); selection+applicability+transport shown here"}
    return trace


def main():
    store = load_registry(reload=True)
    traces = [build_trace(store, *c) for c in CASES]
    exec_end_to_end = [t["category"] for t in traces if isinstance(t.get("execution"), dict)
                       and t["execution"].get("state_delta")]
    doc = {"_meta": {"n_traces": len(traces), "harness": "experiments/wmv2_phase6_traces.py",
                     "executed_end_to_end": exec_end_to_end}, "traces": traces}
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    print(f"=== {len(traces)} forensic traces ({len(exec_end_to_end)} executed end-to-end with StateDelta) ===")
    for t in traces:
        sel = t.get("selected_family", "—")
        tier = t.get("support_tier")
        ex = t.get("execution", {})
        moved = ex.get("terminal_sensitivity", {}).get("moves") if isinstance(ex, dict) else None
        print(f"  [{t['category']:20s}] {t['causal_process_request']:32s} -> {str(sel):26s} "
              f"Tier {tier} {'(terminal moves: '+str(moved)+')' if moved is not None else ''}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
