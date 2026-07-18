"""Generated actor-mediated world — five end-to-end demos + the matched 4-mode evaluation.

Every demo runs the REAL production funnel (`materialize.run_from_plan`) under the default
`generated_actor_mediated_world`: the scenario's semantic types/events/processes/predicates
are GENERATED live by the schema compiler (no repository code contains any of them), actions
compile to direct effects only, and every response comes from the affected actor's own
persistent invocation through the control plane.

    demoA  corporate product + rebrand      demoB  board cascade (open-ended, no menus)
    demoC  individual communication         demoD  novel cross-domain (chip supplier)
    demoE  non-human boundary (launch anomaly; physics labeled, never hallucinated)

    armA legacy scalar | armB fixed-v1 | armC generated+stateless | armD generated+persistent
    (matched settlement scenario, same seeds; C/D share ONE pre-compiled schema for fairness)

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/generated_world_demo.py demoA
    PYTHONPATH=. python experiments/generated_world_demo.py smoke     # offline wiring check
"""
from __future__ import annotations

import json
import os
import sys
import time as _time
from pathlib import Path

from swm.world_model_v2.compiler import WorldExecutionPlan
from swm.world_model_v2.contracts import OutcomeContract

T0 = 1_700_000_000.0
DAY = 86400.0
RESULTS = Path("experiments/results")
SEED = 23


# ---------------------------------------------------------------- extraction
def world_plane(world) -> dict:
    schema = getattr(world, "scenario_schema", None)
    return {
        "schema": {"id": getattr(schema, "schema_id", ""),
                   "version": getattr(schema, "version", ""),
                   "generated_record_types": sorted(getattr(schema, "record_types", dict)())
                   if schema else [],
                   "generated_event_types": sorted(getattr(schema, "semantic_event_types", {}))
                   if schema else [],
                   "extensions": list(getattr(schema, "ancestry", []))},
        "semantic_log": [{k: x.get(k) for k in ("event_id", "semantic_type_id",
                                                "source_actor_id", "direct_targets",
                                                "exact_content", "intended_visibility",
                                                "cascade_depth")}
                         for x in getattr(world, "semantic_log", [])],
        "records": [{"id": o.object_id, "type": o.object_type, "status": o.status,
                     "by": o.created_by,
                     "fields": {k: (v[:140] if isinstance(v, str) else v)
                                for k, v in o.attributes.items()}}
                    for o in getattr(world, "objects", {}).values()],
    }


def run_result(name, result, branches, wall) -> dict:
    rep = dict(result.get("consequence_report") or {})
    return {"demo": name, "final_distribution": result.get("distribution"),
            "consequence_report": rep,
            "invariants": {"human_reactions_written_directly":
                           rep.get("human_reactions_written_directly"),
                           "fixed_ontology_uses": rep.get("fixed_ontology_uses"),
                           "actual_mode": rep.get("actual_mode")},
            "actor_policy_report": {k: (result.get("actor_policy_report") or {}).get(k)
                                    for k in ("requested_actor_policy_mode",
                                              "actual_actor_policy_mode",
                                              "actors_routed_qualitatively", "fallbacks")},
            "branches": [world_plane(b.world) for b in branches],
            "wall_s": round(wall, 1)}


def _mech():
    return [{"mech_id": "production_actor_policy", "ontology_type": "decision",
             "causal_role": "actor decisions drive the process",
             "parameter_source": "actor policy mode router", "temporal_scale": "event",
             "calibration_status": "experimental", "operator": "production_actor_policy",
             "sensitivity": 1.0}]


def _contract(options, horizon):
    # placeholder readout; run_from_plan swaps in the generated frozen predicates
    return OutcomeContract(family="binary", options=list(options),
                           resolution_rule="resolved by the generated outcome predicates",
                           readout=lambda world: options[1], readout_var="",
                           horizon_ts=horizon).validate()


def _plan(question, entities, events, *, horizon_days, institutions=None, options=("True", "False"),
          n_particles=4, relations=None, evidence=""):
    horizon = T0 + horizon_days * DAY
    return WorldExecutionPlan(
        question=question, outcome_contract=_contract(options, horizon), as_of=T0,
        horizon_ts=horizon, entities=entities, relations=relations or [],
        quantities=[], scheduled_events=events, accepted_mechanisms=_mech(),
        institutions=institutions or [],
        support_grade="exploratory", compute_plan={"n_particles": n_particles},
        provenance={"demo": "generated_world", "evidence_summary": evidence})


def _person(pid, role, goals=(), capacity=0.8, extra=None):
    fields = {"roles": [role], "resources": {"capacity": capacity},
              "past_actions": []}
    if goals:
        fields["goals"] = list(goals)
    fields.update(extra or {})
    return {"id": pid, "type": "person", "fields": fields}


def _decide(ts, actor, situation):
    """Open-ended decision opening — deliberately NO candidate_actions menu."""
    return {"etype": "decision_opportunity", "ts": ts, "participants": [actor],
            "payload": {"situation": situation}}


# ---------------------------------------------------------------- the five demos
def plan_demoA():
    return _plan(
        "Will Meridian complete its rebrand to 'Haven' and publicly launch the experiences "
        "marketplace with at least one anchor partner signed within 45 days?",
        entities=[
            _person("sofia_ceo", "founder-CEO of Meridian (a stays marketplace)",
                    goals=["make Haven the brand of trusted local experiences"]),
            _person("marcus_brand_chief", "chief brand officer of Meridian"),
            _person("elena_anchor_partner", "CEO of CityWalks Tours, the intended anchor "
                                            "experiences partner"),
            _person("ravi_journalist", "consumer-tech journalist covering travel platforms"),
        ],
        events=[_decide(T0 + 2 * DAY, "sofia_ceo",
                        "the Haven rebrand assets and the experiences marketplace build are "
                        "ready; CityWalks has not yet signed; the press has heard rumors — "
                        "decide how to proceed"),
                _decide(T0 + 20 * DAY, "ravi_journalist",
                        "review what has publicly happened around Meridian this fortnight "
                        "and decide what, if anything, to publish")],
        relations=[{"src": "sofia_ceo", "rel": "communicates_with",
                    "dst": "elena_anchor_partner"},
                   {"src": "sofia_ceo", "rel": "communicates_with",
                    "dst": "marcus_brand_chief"}],
        horizon_days=45, options=("launched_with_anchor", "not_launched"),
        n_particles=4,
        evidence="Meridian is a five-year-old stays marketplace; CityWalks ran a successful "
                 "pilot last quarter; rebrand agency work is complete; two competitors "
                 "launched experience verticals last year.")


def plan_demoB():
    return _plan(
        "Will Northgate's board approve the $140M acquisition of Nimbus within three weeks?",
        entities=[
            _person("rhea_ceo", "CEO of Northgate",
                    goals=["close the Nimbus acquisition to fix the data gap"]),
            _person("ines_chair", "board chair of Northgate"),
            _person("arman_cfo", "CFO and board member",
                    extra={"stances": [{"actor": "arman_cfo",
                                        "commitment_level": "conditionally_opposed",
                                        "pathway": "institutional_procedure",
                                        "reliability": "medium", "capability": "high",
                                        "quote": "at this price we are betting the year"}]}),
            _person("dora_director", "independent director, ex-regulator"),
            _person("felix_director", "director representing the venture fund"),
        ],
        events=[_decide(T0 + 1 * DAY, "rhea_ceo",
                        "Nimbus accepted your indicative $140M offer subject to board "
                        "approval; exclusivity lapses in three weeks; the CFO has privately "
                        "signaled concerns — decide how to proceed")],
        relations=[{"src": "rhea_ceo", "rel": "reports_to", "dst": "ines_chair"},
                   {"src": "rhea_ceo", "rel": "communicates_with", "dst": "arman_cfo"},
                   {"src": "ines_chair", "rel": "communicates_with", "dst": "dora_director"},
                   {"src": "ines_chair", "rel": "communicates_with", "dst": "felix_director"}],
        institutions=[{"id": "northgate_board", "rules": [
            {"kind": "decision_right",
             "params": {"actions": ["approve"],
                        "holders": ["ines_chair", "arman_cfo", "dora_director",
                                    "felix_director"]}}]}],
        horizon_days=21, options=("approved", "not_approved"), n_particles=4,
        evidence="Northgate board: chair Ines, CFO Arman, independent director Dora, fund "
                 "director Felix; majority of the four approves acquisitions; CEO holds no "
                 "board vote.")


def plan_demoD():
    return _plan(
        "Will Corvid Fab requalify as Apex Devices' packaging supplier before the Q3 "
        "order is placed (60 days)?",
        entities=[
            _person("mei_apex_vp", "VP of supply chain at Apex Devices",
                    goals=["secure Q3 packaging capacity without a second qualification "
                           "failure"]),
            _person("stefan_corvid_ceo", "CEO of Corvid Fab",
                    goals=["win back the Apex qualification after the contamination "
                           "finding"]),
            _person("aiko_quality_head", "head of quality engineering at Apex, owns the "
                                         "qualification protocol"),
        ],
        events=[_decide(T0 + 1 * DAY, "stefan_corvid_ceo",
                        "Apex's audit found particulate contamination in your cleanroom "
                        "line and suspended qualification; the Q3 order window closes in "
                        "60 days — decide what to do"),
                _decide(T0 + 5 * DAY, "mei_apex_vp",
                        "Corvid is suspended; the only alternate packager quotes +18% and "
                        "a 9-week lead; decide how to manage Q3 supply")],
        relations=[{"src": "stefan_corvid_ceo", "rel": "communicates_with",
                    "dst": "mei_apex_vp"},
                   {"src": "aiko_quality_head", "rel": "reports_to", "dst": "mei_apex_vp"}],
        horizon_days=60, options=("requalified", "not_requalified"), n_particles=4,
        evidence="Apex suspended Corvid after a particulate excursion; requalification "
                 "needs a corrective-action report, a clean re-audit by Apex quality, and "
                 "a 30-day monitored trial run.")


def plan_demoE():
    return _plan(
        "Will Vanta Space fly the OTV-2 mission inside its 30-day launch window after the "
        "static-fire anomaly?",
        entities=[
            _person("noor_launch_director", "launch director at Vanta Space",
                    goals=["fly OTV-2 without compromising range safety"]),
            _person("kenji_chief_engineer", "chief propulsion engineer at Vanta Space"),
            _person("range_safety_officer", "federal range safety officer with veto over "
                                            "launch clearance"),
        ],
        events=[_decide(T0 + 1 * DAY, "kenji_chief_engineer",
                        "yesterday's static fire aborted at T-2s on a turbopump pressure "
                        "spike; telemetry is ambiguous between a sensor artifact and a real "
                        "hardware issue — decide what to do"),
                _decide(T0 + 3 * DAY, "noor_launch_director",
                        "the window opens in 12 days; engineering has not yet established "
                        "the anomaly's root cause — decide how to proceed")],
        relations=[{"src": "kenji_chief_engineer", "rel": "reports_to",
                    "dst": "noor_launch_director"},
                   {"src": "noor_launch_director", "rel": "communicates_with",
                    "dst": "range_safety_officer"}],
        horizon_days=30, options=("flew_in_window", "did_not_fly"), n_particles=4,
        evidence="Static fire aborted at T-2s, turbopump discharge pressure spiked 8% for "
                 "40ms; root cause NOT yet established — whether the hardware is actually "
                 "flightworthy is an unresolved physical question no simulated human can "
                 "settle by assertion; range safety approval is a hard legal gate.")


def run_demo(name: str, plan, llm) -> dict:
    from swm.world_model_v2.materialize import run_from_plan
    t0 = _time.monotonic()
    result, branches = run_from_plan(plan, llm=llm,
                                     n_particles=plan.compute_plan["n_particles"], seed=SEED)
    out = run_result(name, result, branches, _time.monotonic() - t0)
    if name == "demoE":
        schema = getattr(plan, "scenario_schema", None)
        out["unresolved_mechanisms_labeled"] = list(
            getattr(schema, "unresolved_mechanisms", []) or [])
        out["physical_constraints"] = dict(getattr(schema, "physical_constraints", {}) or {})
    return out


def run_demoC(llm) -> dict:
    """Individual communication in PURE generated mode: the reply schema is compiled for the
    question; the reply is a scenario-typed semantic event, not a fixed-catalog message."""
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    from swm.world_model_v2.scenario_schema import SchemaCompiler
    t0 = _time.monotonic()
    stimulus = ("I know we planned Saturday for months — I have to cancel again. Work "
                "summit moved onto that weekend. I'm really sorry. Can we do the week "
                "after?")
    schema = SchemaCompiler(llm).compile(
        question="How will Jordan react to their close friend cancelling the long-planned "
                 "Saturday trip again?",
        as_of=T0, horizon=T0 + 7 * DAY, entities=["Jordan", "you"],
        evidence="Third cancellation this year; Jordan planned the itinerary; the "
                 "friendship is close but Jordan has told others they feel deprioritized.")
    artifact = simulate_individual_reaction(
        person_id="Jordan", stimulus=stimulus,
        context={"relationship": "your close friend of ten years", "role": "close friend",
                 "your_role": "the friend who keeps cancelling",
                 "history": ["Jordan planned the full itinerary in March",
                             "You cancelled the February hike two days before",
                             "Jordan told a mutual friend they feel deprioritized"]},
        llm=llm, n_hypotheses=3, samples_per_hypothesis=2, seed=SEED, as_of=T0,
        scenario_schema=schema)
    return {"demo": "demoC_individual_communication",
            "stimulus": stimulus,
            "schema_generated_event_types": sorted(schema.semantic_event_types),
            "raw_distribution": artifact["raw_qualitative_simulation_distribution"],
            "samples": [{k: s.get(k) for k in ("hypothesis_id", "observable_response",
                                               "internal_reaction", "decision_summary")}
                        for s in artifact["samples"]],
            "consequence_report": artifact["consequence_report"],
            "wall_s": round(_time.monotonic() - t0, 1)}


# ---------------------------------------------------------------- matched 4-mode evaluation
def eval_plan():
    return _plan(
        "Will the two leaders reach a settlement within 60 days?",
        entities=[
            _person("leader_a", "principal of side A", goals=["prevail_or_settle_well"],
                    extra={"stances": [{"actor": "leader_a",
                                        "commitment_level": "committed_to_prevent",
                                        "pathway": "cooperative_agreement",
                                        "reliability": "high", "capability": "high",
                                        "quote": "we will not settle while our objectives "
                                                 "stand"}]}),
            _person("leader_b", "principal of side B", goals=["survive_and_secure_terms"],
                    capacity=0.6),
        ],
        events=[_decide(T0 + 15 * DAY, "leader_a",
                        "mediated settlement round 1: a framework is on the table; the "
                        "contested campaign is costly and slow"),
                _decide(T0 + 15 * DAY, "leader_b",
                        "mediated settlement round 1: a framework is on the table; the "
                        "contested campaign is costly and slow"),
                _decide(T0 + 30 * DAY, "leader_a", "mediated settlement round 2"),
                _decide(T0 + 30 * DAY, "leader_b", "mediated settlement round 2")],
        relations=[{"src": "leader_a", "rel": "communicates_with", "dst": "leader_b"}],
        horizon_days=60, options=("deal_reached", "no_deal"), n_particles=8,
        evidence="Two-party mediated settlement; a framework exists; both sides face "
                 "rising costs of the contested campaign.")


ARMS = {
    "armA": {"consequences": "legacy_scalar_pathway_consequences",
             "actors": "persistent_qualitative_llm_policy", "needs_llm": True,
             "label": "legacy scalar × persistent qualitative"},
    "armB": {"consequences": "fixed_semantic_consequence_policy_v1",
             "actors": "persistent_qualitative_llm_policy", "needs_llm": True,
             "label": "fixed-v1 catalog × persistent qualitative"},
    "armC": {"consequences": "generated_actor_mediated_world",
             "actors": "stateless_llm_policy", "needs_llm": True,
             "label": "generated actor-mediated × stateless"},
    "armD": {"consequences": "generated_actor_mediated_world",
             "actors": "persistent_qualitative_llm_policy", "needs_llm": True,
             "label": "generated actor-mediated × persistent (PRODUCTION DEFAULT)"},
}

SHARED_SCHEMA_PATH = RESULTS / "generated_eval_schema.json"


def eval_schema(llm):
    """C and D share ONE pre-compiled schema so the actor-policy axis is isolated."""
    from swm.world_model_v2.scenario_schema import ScenarioSemanticModel, SchemaCompiler
    if SHARED_SCHEMA_PATH.exists():
        return ScenarioSemanticModel.from_dict(json.loads(SHARED_SCHEMA_PATH.read_text()))
    p = eval_plan()
    schema = SchemaCompiler(llm).compile(
        question=p.question, as_of=T0, horizon=p.horizon_ts,
        entities=["leader_a", "leader_b"],
        evidence=(p.provenance or {}).get("evidence_summary", ""))
    SHARED_SCHEMA_PATH.write_text(json.dumps(schema.as_dict(), indent=1, default=str))
    return schema


def run_arm(arm: str, llm) -> dict:
    from swm.world_model_v2.materialize import run_from_plan
    cfg = ARMS[arm]
    prior_c, prior_a = os.environ.get("SWM_CONSEQUENCES"), os.environ.get("SWM_ACTOR_POLICY")
    os.environ["SWM_CONSEQUENCES"] = cfg["consequences"]
    os.environ["SWM_ACTOR_POLICY"] = cfg["actors"]
    try:
        t0 = _time.monotonic()
        plan = eval_plan()
        if cfg["consequences"] == "generated_actor_mediated_world":
            plan.scenario_schema = eval_schema(llm)
        else:
            # matched baselines need a readable readout: reuse the fixed audit bar contract
            from swm.world_model_v2.quantities import register_quantity_type
            plan.quantities = [{"name": "pathway_progress:cooperative_agreement",
                                "qtype": "pathway_progress", "value": 0.30,
                                "units": "process_state"}]
            var = "pathway_progress:cooperative_agreement"
            plan.outcome_contract.readout = (
                lambda world: "deal_reached"
                if float(getattr(world.quantities.get(var), "value", 0.0) or 0.0) >= 0.5
                else "no_deal")
        result, branches = run_from_plan(plan, llm=llm, n_particles=8, seed=SEED)
        rep = dict(result.get("consequence_report") or {})
        out = {"arm": arm, "label": cfg["label"], **cfg,
               "final_distribution": result.get("distribution"),
               "consequence_report": {k: v for k, v in rep.items()
                                      if k != "dual_run_legacy_shadow"},
               "n_semantic_events": sum(len(getattr(b.world, "semantic_log", []))
                                        for b in branches),
               "n_records": sum(len(getattr(b.world, "objects", {})) for b in branches),
               "invariants": {"human_reactions_written_directly":
                              rep.get("human_reactions_written_directly"),
                              "fixed_ontology_uses": rep.get("fixed_ontology_uses")},
               "branch_world_plane": [world_plane(b.world) for b in branches[:3]],
               "wall_s": round(_time.monotonic() - t0, 1)}
        return out
    finally:
        for k, v in (("SWM_CONSEQUENCES", prior_c), ("SWM_ACTOR_POLICY", prior_a)):
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def combine():
    demos = {}
    for n in ("demoA", "demoB", "demoC", "demoD", "demoE"):
        p = RESULTS / f"generated_{n}.json"
        if p.exists():
            demos[n] = json.loads(p.read_text())
    (RESULTS / "generated_demos.json").write_text(
        json.dumps({"schema_version": "generated.demos.v1", "seed": SEED, "demos": demos},
                   indent=1, default=str))
    arms = {}
    for a in ARMS:
        p = RESULTS / f"generated_{a}.json"
        if p.exists():
            arms[a] = json.loads(p.read_text())
    ev = {"schema_version": "generated.mode.evaluation.v1", "seed": SEED,
          "scenario": "matched settlement (same worlds/seeds; C/D share one generated "
                      "schema; arms differ only in consequence architecture × actor policy)",
          "arms": {a: {k: v for k, v in row.items() if k != "branch_world_plane"}
                   for a, row in arms.items()},
          "world_planes": {a: row.get("branch_world_plane") for a, row in arms.items()}}
    (RESULTS / "generated_mode_evaluation.json").write_text(
        json.dumps(ev, indent=1, default=str))
    print(json.dumps({"demos": sorted(demos), "arms": sorted(arms)}, indent=1))
    for a, row in sorted(arms.items()):
        print(f"{a} {row['label']}: dist={row['final_distribution']} "
              f"events={row.get('n_semantic_events')} records={row.get('n_records')} "
              f"invariants={row.get('invariants')}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if what == "smoke":
        # offline: schema-less generated default must degrade LOUDLY, never crash
        from swm.world_model_v2.materialize import run_from_plan
        result, branches = run_from_plan(plan_demoB(), llm=None, n_particles=2, seed=SEED)
        rep = result["consequence_report"]
        print(json.dumps({"dist": result.get("distribution"),
                          "requested": rep.get("requested_mode"),
                          "actual": rep.get("actual_mode"),
                          "degraded": rep.get("degraded"),
                          "schema_error": rep.get("scenario_schema_error")}, indent=1))
        return
    from swm.api.deepseek_backend import deepseek_chat_fn
    llm = deepseek_chat_fn(temperature=0.9, max_tokens=4000)   # schema JSONs are large
    if what in ("demoA", "demoB", "demoD", "demoE"):
        plan = {"demoA": plan_demoA, "demoB": plan_demoB,
                "demoD": plan_demoD, "demoE": plan_demoE}[what]()
        out = run_demo(what, plan, llm)
    elif what == "demoC":
        out = run_demoC(llm)
    elif what in ARMS:
        out = run_arm(what, llm)
    elif what == "combine":
        combine()
        return
    else:
        raise SystemExit(f"unknown target {what!r}")
    path = RESULTS / f"generated_{what}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    inv = out.get("invariants") or {}
    print(f"[{what}] wall={out['wall_s']}s dist="
          f"{out.get('final_distribution') or out.get('raw_distribution')} "
          f"invariants={inv} wrote {path}")


if __name__ == "__main__":
    main()
