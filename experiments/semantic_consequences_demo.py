"""Semantic world consequences — four end-to-end demos + the matched 4-mode evaluation.

Every demo runs the REAL production funnel (`materialize.run_from_plan`) under the semantic
default: qualitative decision → semantic consequence compilation → validated CausalActionProgram
→ typed world changes → follow-up events consumed by real operators → answer read from the
evolved structured world. Artifacts record the typed objects (with exact communication content),
process stage histories, consequence reports, and readouts.

    demo1  product launch        — a founder's launch becomes a product, a launch process, a
                                   public statement whose EXACT text later actors see
    demo2  negotiation           — two leaders exchange real messages; the process stages;
                                   derived summary bars are projections of the staged process
    demo3  institutional decision— a CEO's request enters a REAL board procedure; members
                                   decide; the vote writes a TYPED outcome on the submission
    demo4  individual reaction   — one person, one exact message; their reply is a real
                                   communication with content, counted into a distribution

    armA/B/C/D  matched evaluation on the settlement scenario (same seeds, same worlds):
        A legacy scalar + persistent qualitative     (the pre-phase production path)
        B semantic + numeric policy                  (actors off)
        C semantic + stateless qualitative
        D semantic + persistent qualitative          (the new production default)

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/semantic_consequences_demo.py demo1
    PYTHONPATH=. python experiments/semantic_consequences_demo.py smoke   # offline everything
    PYTHONPATH=. python experiments/semantic_consequences_demo.py combine
"""
from __future__ import annotations

import json
import os
import sys
import time as _time
from pathlib import Path

from swm.world_model_v2.compiler import WorldExecutionPlan
from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.materialize import run_from_plan

T0 = 1_700_000_000.0
DAY = 86400.0
RESULTS = Path("experiments/results")
SEED = 11


# ---------------------------------------------------------------- extraction
def _objects(world) -> list:
    out = []
    for o in world.objects.values():
        row = {"id": o.object_id, "type": o.object_type, "status": o.status,
               "by": o.created_by}
        keep = {k: (v[:160] if isinstance(v, str) else v) for k, v in o.attributes.items()
                if k in ("content", "matter", "outcome", "decided_by", "terms", "subject",
                         "process_type", "sender", "recipient", "channel", "publisher",
                         "requested_outcome", "tally", "parties")}
        if keep:
            row["attributes"] = keep
        if o.stage_history:
            row["stages"] = [s["to"] for s in o.stage_history]
        out.append(row)
    return out


def _communications(world) -> list:
    return [{"from": o.attributes.get("sender", o.created_by),
             "to": o.attributes.get("recipient", "public"),
             "channel": o.attributes.get("channel", "public"),
             "status": o.status,
             "content": str(o.attributes.get("content", ""))[:200]}
            for o in world.objects.values()
            if o.object_type in ("private_communication", "public_statement")]


def _actions(branch) -> list:
    acts = []
    for delta in branch.log:
        if delta.operator != "production_actor_policy":
            continue
        for ch in delta.changes:
            if str(ch.get("path", "")).endswith(".current_action") \
                    and isinstance(ch.get("after"), dict):
                acts.append({"actor": str(ch["path"]).split(".")[0],
                             "action": ch["after"].get("action_name", "")})
    return acts


def _branch_summary(branch, *, bar: str = "") -> dict:
    w = branch.world
    row = {"branch": w.branch_id, "actions": _actions(branch),
           "objects": _objects(w), "communications": _communications(w)}
    if bar and bar in w.quantities:
        row["derived_bar"] = {bar: round(float(w.quantities[bar].value), 4),
                              "method": w.quantities[bar].prov.method}
    return row


def _arm_metrics(result, branches, *, bar: str) -> dict:
    vals = sorted(round(float(b.world.quantities[bar].value), 4)
                  for b in branches if bar in b.world.quantities)
    stages = {}
    for b in branches:
        for o in b.world.objects.values():
            if o.object_type == "process":
                key = f"{o.attributes.get('process_type')}:{o.status}"
                stages[key] = stages.get(key, 0) + 1
    return {"final_distribution": result.get("distribution"),
            "terminal_bar_values": vals,
            "mean_terminal_bar": round(sum(vals) / len(vals), 4) if vals else None,
            "process_terminal_stages": stages,
            "n_objects_total": sum(len(b.world.objects) for b in branches),
            "n_communications": sum(len(_communications(b.world)) for b in branches),
            "consequence_report": {k: v for k, v in
                                   (result.get("consequence_report") or {}).items()
                                   if k != "dual_run_legacy_shadow"},
            "actor_policy_report": {k: (result.get("actor_policy_report") or {}).get(k)
                                    for k in ("requested_actor_policy_mode",
                                              "actual_actor_policy_mode",
                                              "actors_routed_qualitatively", "fallbacks")}}


# ---------------------------------------------------------------- demo plans
def _mech(name="production_actor_policy", role="actor decisions drive the process"):
    return [{"mech_id": name, "ontology_type": "decision", "causal_role": role,
             "parameter_source": "actor policy mode router", "temporal_scale": "event",
             "calibration_status": "experimental", "operator": name, "sensitivity": 1.0}]


def plan_demo1_launch() -> WorldExecutionPlan:
    horizon = T0 + 30 * DAY

    def readout(world):
        launched = any(o.object_type == "process"
                       and o.attributes.get("process_type") == "product_launch"
                       and o.status in ("announced", "available", "scaling")
                       for o in world.objects.values())
        return "launched_publicly" if launched else "not_launched"

    contract = OutcomeContract(
        family="binary", options=["launched_publicly", "not_launched"],
        resolution_rule="a product_launch process reaches announced/available in the typed world",
        readout=readout, readout_var="", horizon_ts=horizon).validate()
    entities = [
        {"id": "maya_founder", "type": "person", "fields": {
            "roles": ["founder_ceo of Lumen Labs"], "goals": ["make Glide the default notetaker"],
            "resources": {"budget": 40.0, "capacity": 0.9},
            "authority": ["launch"],
            "stances": [{"actor": "maya_founder", "commitment_level": "actively_pursuing",
                         "pathway": "operational_execution", "reliability": "high",
                         "capability": "high", "quote": "we ship when the demo sings"}]}},
        {"id": "dev_rival_ceo", "type": "person", "fields": {
            "roles": ["ceo of the incumbent NoteCorp"], "goals": ["defend market position"],
            "resources": {"budget": 200.0, "capacity": 0.7}}},
    ]
    events = [
        {"etype": "decision_opportunity", "ts": T0 + 2 * DAY, "participants": ["maya_founder"],
         "payload": {"situation": "Glide (your AI notetaking app) passed final QA yesterday; "
                                  "press embargo lifts whenever you choose; NoteCorp's annual "
                                  "keynote is in 12 days",
                     "candidate_actions": ["launch", "delay_launch", "hold_position"]}},
        {"etype": "decision_opportunity", "ts": T0 + 9 * DAY, "participants": ["dev_rival_ceo"],
         "payload": {"situation": "review this week's market developments and decide NoteCorp's "
                                  "next move before the keynote",
                     "candidate_actions": ["launch", "hold_position", "escalate", "publicize"]}},
    ]
    return WorldExecutionPlan(
        question="Will Lumen Labs launch Glide publicly within 30 days?",
        outcome_contract=contract, as_of=T0, horizon_ts=horizon, entities=entities,
        relations=[{"src": "maya_founder", "rel": "competes_with", "dst": "dev_rival_ceo"}],
        quantities=[], scheduled_events=events, accepted_mechanisms=_mech(),
        support_grade="exploratory", compute_plan={"n_particles": 6},
        provenance={"demo": "semantic_product_launch"})


def plan_demo2_negotiation() -> WorldExecutionPlan:
    horizon = T0 + 60 * DAY
    bar = "pathway_progress:cooperative_agreement"

    def readout(world):
        signed = any(o.object_type == "agreement" and o.status in ("signed", "active")
                     for o in world.objects.values())
        advanced = any(o.object_type == "process"
                       and o.attributes.get("process_type") == "negotiation"
                       and o.status in ("provisional_acceptance", "signed", "implemented")
                       for o in world.objects.values())
        return "deal_reached" if (signed or advanced) else "no_deal"

    contract = OutcomeContract(
        family="binary", options=["deal_reached", "no_deal"],
        resolution_rule="an agreement is signed OR the negotiation process reaches "
                        "provisional acceptance in the typed world",
        readout=readout, readout_var="", horizon_ts=horizon).validate()
    entities = [
        {"id": "leader_a", "type": "person", "fields": {
            "roles": ["principal of side A"], "goals": ["prevail_or_settle_well"],
            "resources": {"capacity": 0.8},
            "stances": [{"actor": "leader_a", "commitment_level": "committed_to_prevent",
                         "pathway": "cooperative_agreement", "reliability": "high",
                         "capability": "high",
                         "quote": "we will not settle while our objectives stand"}]}},
        {"id": "leader_b", "type": "person", "fields": {
            "roles": ["principal of side B"], "goals": ["survive_and_secure_terms"],
            "resources": {"capacity": 0.6},
            "stances": [{"actor": "leader_b", "commitment_level": "conditionally_opposed",
                         "pathway": "cooperative_agreement", "reliability": "medium",
                         "capability": "medium",
                         "quote": "talks are possible only with guarantees"}]}},
    ]
    events = []
    for r in range(2):
        ts = T0 + (r + 1) * 15 * DAY
        for aid in ("leader_a", "leader_b"):
            events.append({"etype": "decision_opportunity", "ts": ts, "participants": [aid],
                           "payload": {"situation": f"mediated settlement round {r + 1}: a "
                                                    "framework is on the table; the contested "
                                                    "campaign is costly and slow",
                                       "candidate_actions": ["accept", "counteroffer",
                                                             "hold_position", "escalate",
                                                             "delay", "seek_mediator"]}})
    return WorldExecutionPlan(
        question="Will the two leaders reach a settlement within 60 days?",
        outcome_contract=contract, as_of=T0, horizon_ts=horizon, entities=entities,
        relations=[{"src": "leader_a", "rel": "communicates_with", "dst": "leader_b"}],
        quantities=[{"name": bar, "qtype": "pathway_progress", "value": 0.30,
                     "units": "process_state"}],
        scheduled_events=events, accepted_mechanisms=_mech(),
        support_grade="exploratory", compute_plan={"n_particles": 6},
        provenance={"demo": "semantic_negotiation"})


def plan_demo3_institution() -> WorldExecutionPlan:
    horizon = T0 + 21 * DAY

    def readout(world):
        for o in world.objects.values():
            if o.object_type == "submission" and o.status == "decided" \
                    and o.attributes.get("outcome") in ("approve", "approved"):
                return "approved"
        return "not_approved"

    contract = OutcomeContract(
        family="binary", options=["approved", "not_approved"],
        resolution_rule="the board submission carries a typed decided/approve outcome",
        readout=readout, readout_var="", horizon_ts=horizon).validate()
    entities = [
        {"id": "rhea_ceo", "type": "person", "fields": {
            "roles": ["ceo"], "goals": ["acquire Nimbus to close the data gap"],
            "resources": {"budget": 90.0, "capacity": 0.85}}},
        {"id": "arman_cfo", "type": "person", "fields": {
            "roles": ["cfo"], "goals": ["protect the balance sheet"],
            "resources": {"capacity": 0.7},
            "stances": [{"actor": "arman_cfo", "commitment_level": "conditionally_opposed",
                         "pathway": "institutional_procedure", "reliability": "medium",
                         "capability": "high",
                         "quote": "at this price we would be betting the year"}]}},
        {"id": "ines_chair", "type": "person", "fields": {
            "roles": ["board chair"], "goals": ["long-term strategic positioning"],
            "resources": {"capacity": 0.8}}},
    ]
    events = [{"etype": "decision_opportunity", "ts": T0 + 1 * DAY,
               "participants": ["rhea_ceo"],
               "payload": {"situation": "Nimbus accepted your indicative offer at $140M, "
                                        "subject to your board's approval; the exclusivity "
                                        "window closes in three weeks",
                           "candidate_actions": ["request_approval", "defer", "withdraw"]}}]
    return WorldExecutionPlan(
        question="Will the board approve the Nimbus acquisition within three weeks?",
        outcome_contract=contract, as_of=T0, horizon_ts=horizon, entities=entities,
        relations=[{"src": "rhea_ceo", "rel": "reports_to", "dst": "ines_chair"}],
        quantities=[],
        scheduled_events=events, accepted_mechanisms=_mech(),
        institutions=[{"id": "board", "rules": [
            {"kind": "decision_right",
             "params": {"actions": ["approve"],
                        "holders": ["rhea_ceo", "arman_cfo", "ines_chair"]}}]}],
        support_grade="exploratory", compute_plan={"n_particles": 4},
        provenance={"demo": "semantic_institutional_decision"})


def run_demo(name: str, plan: WorldExecutionPlan, llm, *, bar: str = "") -> dict:
    t0 = _time.monotonic()
    result, branches = run_from_plan(plan, llm=llm,
                                     n_particles=plan.compute_plan["n_particles"], seed=SEED)
    return {"demo": name, "question": plan.question,
            "final_distribution": result.get("distribution"),
            "consequence_report": result.get("consequence_report"),
            "actor_policy_report": {k: (result.get("actor_policy_report") or {}).get(k)
                                    for k in ("requested_actor_policy_mode",
                                              "actual_actor_policy_mode",
                                              "actors_routed_qualitatively", "fallbacks")},
            "branches": [_branch_summary(b, bar=bar) for b in branches],
            "wall_s": round(_time.monotonic() - t0, 1)}


def run_demo4_individual(llm) -> dict:
    """One person, one exact message, K hypotheses × samples — replies are REAL communications."""
    from swm.world_model_v2.individual_reaction import simulate_individual_reaction
    t0 = _time.monotonic()
    artifact = simulate_individual_reaction(
        person_id="Priya",
        stimulus="I know it's Sunday — the demo broke on the client's machine and they "
                 "present at 9am. Any chance you could hop on for an hour tonight?",
        context={"relationship": "your manager of two years", "role": "senior engineer",
                 "your_role": "manager", "history": [
                     "Priya shipped the demo build on Friday and flagged the flaky installer",
                     "Priya covered the last two weekend incidents",
                     "Priya has told the team she is protecting family Sundays this quarter"]},
        llm=llm, n_hypotheses=3, samples_per_hypothesis=2, seed=SEED, as_of=T0)
    return {"demo": "demo4_individual_communication",
            "stimulus_delivered_exactly": artifact["stimulus"],
            "raw_distribution": artifact["raw_qualitative_simulation_distribution"],
            "calibrated_distribution": artifact["calibrated_distribution"],
            "calibration_status": artifact["calibration_status"],
            "samples": [{k: s.get(k) for k in ("hypothesis_id", "observable_response",
                                               "internal_reaction", "decision_summary")}
                        for s in artifact["samples"]],
            "consequence_report": artifact["consequence_report"],
            "n_excluded_numeric_fallbacks": artifact["n_excluded_numeric_fallbacks"],
            "llm_calls": artifact["llm_calls"],
            "wall_s": round(_time.monotonic() - t0, 1)}


# ---------------------------------------------------------------- matched 4-mode evaluation
ARMS = {
    "armA": {"consequences": "legacy_scalar_pathway_consequences",
             "actors": "persistent_qualitative_llm_policy", "needs_llm": True,
             "label": "legacy scalar + persistent qualitative (pre-phase production)"},
    "armB": {"consequences": "semantic_world_consequences",
             "actors": "numeric_policy", "needs_llm": False,
             "label": "semantic + numeric policy (actors off)"},
    "armC": {"consequences": "semantic_world_consequences",
             "actors": "stateless_llm_policy", "needs_llm": True,
             "label": "semantic + stateless qualitative"},
    "armD": {"consequences": "semantic_world_consequences",
             "actors": "persistent_qualitative_llm_policy", "needs_llm": True,
             "label": "semantic + persistent qualitative (new production default)"},
}
BAR = "pathway_progress:cooperative_agreement"


def run_arm(arm: str, llm) -> dict:
    cfg = ARMS[arm]
    prior_c = os.environ.get("SWM_CONSEQUENCES")
    prior_a = os.environ.get("SWM_ACTOR_POLICY")
    os.environ["SWM_CONSEQUENCES"] = cfg["consequences"]
    os.environ["SWM_ACTOR_POLICY"] = cfg["actors"]
    try:
        t0 = _time.monotonic()
        plan = plan_demo2_negotiation()          # the SAME matched scenario for every arm
        result, branches = run_from_plan(plan, llm=llm if cfg["needs_llm"] else None,
                                         n_particles=8, seed=SEED)
        out = {"arm": arm, "label": cfg["label"], **cfg,
                **_arm_metrics(result, branches, bar=BAR),
                "branches": [_branch_summary(b, bar=BAR) for b in branches],
                "wall_s": round(_time.monotonic() - t0, 1)}
        return out
    finally:
        for k, v in (("SWM_CONSEQUENCES", prior_c), ("SWM_ACTOR_POLICY", prior_a)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def combine() -> None:
    demos = {}
    for n in ("demo1", "demo2", "demo3", "demo4"):
        p = RESULTS / f"semantic_{n}.json"
        if p.exists():
            demos[n] = json.loads(p.read_text())
    (RESULTS / "semantic_demos.json").write_text(json.dumps(
        {"schema_version": "semantic.demos.v1", "seed": SEED, "demos": demos},
        indent=1, default=str))
    arms = {}
    for a in ARMS:
        p = RESULTS / f"semantic_{a}.json"
        if p.exists():
            arms[a] = json.loads(p.read_text())
    evaluation = {"schema_version": "semantic.mode.evaluation.v1", "seed": SEED,
                  "scenario": "matched settlement negotiation (same worlds, same seeds; arms "
                              "differ only in consequence mode × actor policy)",
                  "arms": {a: {k: v for k, v in row.items() if k != "branches"}
                           for a, row in arms.items()},
                  "branches": {a: row.get("branches") for a, row in arms.items()}}
    if "armA" in arms and "armD" in arms:
        pa = (arms["armA"].get("final_distribution") or {}).get("deal_reached", 0.0)
        pd = (arms["armD"].get("final_distribution") or {}).get("deal_reached", 0.0)
        evaluation["headline"] = {
            "p_deal_legacy_scalar_A": pa, "p_deal_semantic_default_D": pd,
            "binary_answer_moved": pa != pd,
            "mean_bar_A": arms["armA"].get("mean_terminal_bar"),
            "mean_bar_D": arms["armD"].get("mean_terminal_bar"),
            "typed_objects_A": arms["armA"].get("n_objects_total"),
            "typed_objects_D": arms["armD"].get("n_objects_total")}
    (RESULTS / "semantic_mode_evaluation.json").write_text(
        json.dumps(evaluation, indent=1, default=str))
    print(json.dumps({"demos_combined": sorted(demos), "arms_combined": sorted(arms),
                      "headline": evaluation.get("headline")}, indent=1))


# ---------------------------------------------------------------- backends & entry
def scripted_backend():
    """Offline smoke backend: picks the first candidate, lets the consequence compiler fall
    back to the deterministic ontology→primitive path (still fully semantic)."""
    import re

    def fn(prompt):
        if "CONSEQUENCE COMPILER" in prompt:
            return "no ops from the scripted backend"     # → loud deterministic fallback
        m = re.search(r"^- (\S+): \w+/", prompt, flags=re.M)   # first REAL menu line
        first = m.group(1) if m else "accept"
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "situation_interpretation": {"what_changed": "the round opened",
                                         "why_it_matters": "the outcome hangs on it",
                                         "perceived_opportunities": "", "perceived_threats": ""},
            "actor_state_update": {"current_private_beliefs": [], "beliefs_about_others": {},
                                   "personal_condition": "steady", "important_memories": []},
            "anticipated_reactions": [],
            "decision": {"act_or_wait": "act", "chosen_action": first, "target": "",
                         "timing": "immediate", "observability": "public",
                         "intended_effect": "advance my position"},
            "novel_action_proposal": {"present": False},
            "alternatives_considered": [], "decision_summary": f"I {first} now"})
    return fn


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    backend = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else \
        ("scripted" if what == "smoke" else "deepseek")
    if backend == "deepseek":
        from swm.api.deepseek_backend import deepseek_chat_fn
        llm = deepseek_chat_fn(temperature=0.9, max_tokens=2000)
    else:
        llm = scripted_backend()

    if what == "smoke":
        for name, fn in (("demo1", lambda: run_demo("demo1_product_launch",
                                                    plan_demo1_launch(), llm)),
                         ("demo2", lambda: run_demo("demo2_negotiation",
                                                    plan_demo2_negotiation(), llm, bar=BAR)),
                         ("demo3", lambda: run_demo("demo3_institutional_decision",
                                                    plan_demo3_institution(), llm)),
                         ("demo4", lambda: run_demo4_individual(llm)),
                         ("armB", lambda: run_arm("armB", None))):
            out = fn()
            rep = out.get("consequence_report") or {}
            print(f"[smoke {name}] dist={out.get('final_distribution') or out.get('raw_distribution')}"
                  f" ops={rep.get('direct_operations_applied')}"
                  f" objects={rep.get('objects_created')}"
                  f" legacy_writes={rep.get('legacy_scalar_writes')}")
        return

    if what in ("demo1", "demo2", "demo3", "demo4"):
        out = {"demo1": lambda: run_demo("demo1_product_launch", plan_demo1_launch(), llm),
               "demo2": lambda: run_demo("demo2_negotiation", plan_demo2_negotiation(), llm,
                                         bar=BAR),
               "demo3": lambda: run_demo("demo3_institutional_decision",
                                         plan_demo3_institution(), llm),
               "demo4": lambda: run_demo4_individual(llm)}[what]()
        path = RESULTS / f"semantic_{what}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"[{what}] wall={out['wall_s']}s wrote {path}")
        print(json.dumps(out.get("final_distribution") or out.get("raw_distribution"),
                         indent=1))
    elif what in ARMS:
        out = run_arm(what, llm)
        path = RESULTS / f"semantic_{what}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"[{what}] {out['label']} wall={out['wall_s']}s "
              f"dist={out['final_distribution']} bar_mean={out['mean_terminal_bar']}")
    elif what == "combine":
        combine()
    else:
        raise SystemExit(f"unknown target {what!r}")


if __name__ == "__main__":
    main()
