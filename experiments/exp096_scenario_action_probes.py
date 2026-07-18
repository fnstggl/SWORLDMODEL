"""EXP-096 — five live architectural probes of the scenario-generated action layer.

Architecture probes, NOT accuracy claims: each runs the real public API with a live LLM
backend end to end — scenario schema compiled from the question, action language + goal
contract generated, strategies proposed by separated roles, candidates compiled ONCE into
kernel ops, matched rollouts through the canonical runtime with LIVE qualitative actor
reactions, diagnosis, revision, blind adjudication — and saves the complete evidence:

    artifacts/phase13/action_language/probes/<probe>/
        contract.json        the decision contract
        schema.json          the compiled scenario world semantics
        result.json          full DecisionResult (scenario report inside)
        role_trace.jsonl     every role-labeled LLM call (stage, role, prompt, response)
        runtime_proof.json   operator delta census proving the canonical engine ran

Probes:
  1 founder_launch      launch now vs private recruitment vs delay + discovered strategies
  2 partnership_terms   negotiation: target, route, terms, timing
  3 nonmessage_ops      best candidate must institutionally/physically alter the scenario
  4 info_then_act       multi-step: information gathering changes the later action
  5 novel_user_action   a user action with no clean legacy verb, evaluated verbatim

Run:  PYTHONPATH=. python experiments/exp096_scenario_action_probes.py [--probe N] [--offline]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.world_model_v2.events import EventQueue
from swm.world_model_v2.generated_world import (GeneratedObservationDeliveryOperator,
                                                GeneratedSemanticEventOperator,
                                                GeneratedActorInvocationOperator,
                                                generated_report)
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13.scenario_actions.api import (discover_best_action,
                                                             evaluate_proposed_actions)
from swm.world_model_v2.phase13.scenario_actions.execution import ScenarioPlanOperator
from swm.world_model_v2.qualitative_actor import (QualitativeActorPolicyRuntime,
                                                  QualitativeConfig,
                                                  QualitativeDecisionEngine)
from swm.world_model_v2.scenario_schema import SchemaCompiler
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13",
                   "action_language", "probes")
T0 = 1_752_800_000.0          # 2025-07-18-ish anchor
DAY = 86400.0


class ProbeInitial:
    def __init__(self, schema, actors, resources):
        self.schema, self.actors, self.resources = schema, actors, resources

    def sample_particles(self, n, seed=0):
        import copy
        worlds = []
        for i in range(n):
            w = WorldState("probe", f"p{i}", SimulationClock(T0, T0),
                           network=RelationGraph(), information=InformationLedger())
            w.horizon = float(self.schema.horizon or T0 + 60 * DAY)
            for name in self.actors:
                e = Entity(name)
                e.set("roles", F(["person"], status="observed"))
                for rn, amt in (self.resources.get(name) or {}).items():
                    e.set("resources", F(float(amt), status="observed"), key=rn)
                e.set("past_actions", F([], status="observed"))
                w.entities[name] = e
            for a in self.actors[1:]:
                w.network.add(self.actors[0], "communicates_with", a)
            w.scenario_schema = copy.deepcopy(self.schema)
            worlds.append(w)
        return worlds


def live_context(schema, actors, *, llm, resources=None, n_particles=3):
    rep = generated_report()
    engine = QualitativeDecisionEngine(QualitativeConfig(
        llm=llm, llm_hypotheses=False, n_hypotheses=2, max_llm_calls=120))
    runtime = QualitativeActorPolicyRuntime(engine, mode="persistent_qualitative_llm_policy",
                                            consequence_mode="generated_actor_mediated_world")
    ops = [ScenarioPlanOperator(report=rep),
           GeneratedSemanticEventOperator(report=rep),
           GeneratedObservationDeliveryOperator(report=rep),
           GeneratedActorInvocationOperator(runtime, report=rep)]
    return {"initial": ProbeInitial(schema, actors, resources or {}),
            "queue_builder": lambda w: EventQueue(horizon_ts=float(w.horizon)),
            "operators": ops, "contract": None, "n_particles": n_particles,
            "max_events": 120}, rep


def compile_schema(llm, *, question, entities, institutions, evidence, horizon_days=60):
    # the schema JSON is large — schema compilation gets its own high-token backend; the
    # cheaper role/actor backend stays at its smaller budget
    schema_llm = default_chat_fn(max_tokens=3600, temperature=0.2) or llm
    return SchemaCompiler(schema_llm).compile(question=question, as_of=T0,
                                              horizon=T0 + horizon_days * DAY,
                                              entities=entities,
                                              institutions=institutions, evidence=evidence)


def seed_resources(schema, maker: str, amount_by_hint: dict):
    """Seed the maker's holdings under the SCHEMA'S OWN resource names (the compiler names
    resources scenario-natively; a probe seeding invented names starves every plan — run-4
    forensic). Every declared resource gets the probe's stated amount for its best hint
    match (or the single stated amount when the schema declares one resource)."""
    declared = sorted((schema.resource_definitions or {}))
    amounts = list(amount_by_hint.values())
    out = {}
    for rn in declared:
        hit = next((v for k, v in amount_by_hint.items()
                    if any(w in rn for w in str(k).lower().split("_") if len(w) > 3)), None)
        if hit is None and len(amounts) == 1:
            hit = amounts[0]
        if hit is not None:
            out[rn] = float(hit)
    return {maker: out}


def save(probe_dir, name, obj):
    os.makedirs(probe_dir, exist_ok=True)
    with open(os.path.join(probe_dir, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def runtime_proof(res, rep):
    """Operator/counter evidence that finalists ran through the canonical engine."""
    sr = res.provenance.get("scenario_report", {})
    return {"generated_report_counters": {k: v for k, v in rep.items()
                                          if not isinstance(v, list)},
            "fallback_reasons": rep.get("fallback_reasons", [])[:20],
            "simulation_coverage": sr.get("simulation_coverage"),
            "stages": sr.get("evaluations") and sorted(sr["evaluations"]) or [],
            "stop_reason": sr.get("stop_reason")}


def finish(probe_dir, res, rep, t0, contract, schema):
    save(probe_dir, "contract.json", {k: getattr(contract, k) for k in
                                      ("decision_id", "decision_maker", "authority",
                                       "controllable_resources", "context", "horizon")})
    save(probe_dir, "schema.json", schema.as_dict())
    save(probe_dir, "result.json", {"result": res.as_dict(),
                                    "wall_s": round(time.time() - t0, 1)})
    save(probe_dir, "runtime_proof.json", runtime_proof(res, rep))
    hs = res.provenance.get("human_summary", {})
    print(f"  -> {res.recommendation_kind}: {res.recommended}")
    print(f"     {str(hs.get('why', ''))[:140]}")
    print(f"     llm_calls={res.cost.get('llm_calls')} rollouts={res.cost.get('rollouts')}"
          f" wall={round(time.time() - t0)}s")


# ---------------------------------------------------------------- probe definitions
def probe_1_founder_launch(llm, offline):
    t0 = time.time()
    pd = os.path.join(ART, "probe1_founder_launch")
    question = ("Founder Mara Voss must decide how to take her scheduling copilot to market "
                "this quarter: immediate public launch, private design-partner recruitment, "
                "or delay for the enterprise pilot to conclude. What should she do?")
    world_question = ("Will Mara Voss's scheduling copilot reach a committed public launch "
                      "or at least three signed design partners by mid-September 2025?")
    schema = compile_schema(llm, question=world_question,
                            entities=["mara_voss", "devon_reyes", "priya_shah"],
                            institutions=["seed_investor_board"],
                            evidence="Mara Voss is the founder/CEO. Devon Reyes runs the "
                                     "enterprise pilot at Calder Logistics (decides pilot "
                                     "expansion). Priya Shah is the lead seed investor and "
                                     "chairs the 3-seat board that must approve any launch "
                                     "spend over 50k. Runway: 40k of discretionary launch "
                                     "budget without board approval. The pilot reports "
                                     "results in ~3 weeks.")
    ctxt, rep = live_context(schema, ["mara_voss", "devon_reyes", "priya_shah"], llm=llm,
                             resources=seed_resources(schema, "mara_voss", {"launch_budget": 40000.0}))
    contract = DecisionProblem(
        decision_id="probe1", decision_maker="mara_voss", role="founder_ceo",
        authority=["founder_ceo"], controllable_resources={"launch_budget_usd": 40000.0},
        context=question, horizon="2025-09-16T00:00:00Z")
    res = discover_best_action(question, ctxt, problem=contract, llm=llm, seed=11,
                               trace_path=os.path.join(pd, "role_trace.jsonl"),
                               budget="diagnostic")
    finish(pd, res, rep, t0, contract, schema)
    return res


def probe_2_partnership_terms(llm, offline):
    t0 = time.time()
    pd = os.path.join(ART, "probe2_partnership_terms")
    question = ("Kite Robotics' BD lead Amara Diallo wants a component-supply partnership "
                "with Halvorsen Motors before the fall production freeze. Who should she "
                "approach, through what route, with what terms, and when?")
    world_question = ("Will Kite Robotics and Halvorsen Motors sign a component-supply "
                      "partnership before the fall production freeze?")
    schema = compile_schema(llm, question=world_question,
                            entities=["amara_diallo", "erik_halvorsen", "sofia_brandt"],
                            institutions=["halvorsen_procurement_committee"],
                            evidence="Erik Halvorsen (CEO) delegates supplier onboarding to "
                                     "the procurement committee chaired by Sofia Brandt; "
                                     "the committee meets monthly; the production freeze is "
                                     "in ~8 weeks; Kite can offer a 12% volume discount or "
                                     "a co-marketing clause, not both.")
    ctxt, rep = live_context(schema, ["amara_diallo", "erik_halvorsen", "sofia_brandt"],
                             llm=llm)
    contract = DecisionProblem(
        decision_id="probe2", decision_maker="amara_diallo", role="bd_lead",
        authority=["bd_lead"], context=question, horizon="2025-09-30T00:00:00Z")
    res = discover_best_action(question, ctxt, problem=contract, llm=llm, seed=7,
                               trace_path=os.path.join(pd, "role_trace.jsonl"),
                               budget="diagnostic")
    finish(pd, res, rep, t0, contract, schema)
    return res


def probe_3_nonmessage(llm, offline):
    t0 = time.time()
    pd = os.path.join(ART, "probe3_nonmessage_ops")
    question = ("Site director Ines Okonkwo must keep the Brackenford data hall inside its "
                "grid power cap during the August heat event without breaching the "
                "municipal noise variance. What should she do?")
    world_question = ("Will the Brackenford data hall stay inside its grid power cap through "
                      "the August heat event without breaching the municipal noise variance?")
    schema = compile_schema(llm, question=world_question,
                            entities=["ines_okonkwo", "grid_operator_liaison",
                                      "municipal_inspector"],
                            institutions=["municipal_noise_board"],
                            evidence="Ines controls workload scheduling windows, the "
                                     "battery-buffer dispatch (4 MWh), and generator tests "
                                     "(noise-capped by the variance). The grid cap drops "
                                     "12% during declared heat events. The noise board can "
                                     "grant temporary exemptions with 72h notice.")
    ctxt, rep = live_context(schema, ["ines_okonkwo", "grid_operator_liaison",
                                      "municipal_inspector"], llm=llm,
                             resources=seed_resources(schema, "ines_okonkwo", {"battery": 4.0}))
    contract = DecisionProblem(
        decision_id="probe3", decision_maker="ines_okonkwo", role="site_director",
        authority=["site_director"], controllable_resources={"battery_mwh": 4.0},
        context=question, horizon="2025-08-31T00:00:00Z")
    res = discover_best_action(question, ctxt, problem=contract, llm=llm, seed=5,
                               trace_path=os.path.join(pd, "role_trace.jsonl"),
                               budget="diagnostic")
    finish(pd, res, rep, t0, contract, schema)
    return res


def probe_4_info_then_act(llm, offline):
    t0 = time.time()
    pd = os.path.join(ART, "probe4_info_then_act")
    question = ("Clinic owner Dr. Sam Whitfield suspects the new referral drop comes from "
                "either the January fee change or the rival clinic's opening. Before "
                "choosing between reversing the fee or launching a referral-partner "
                "program, what should the clinic do?")
    world_question = ("Will the clinic's monthly referrals recover to their December level "
                      "by mid-October 2025?")
    schema = compile_schema(llm, question=world_question,
                            entities=["sam_whitfield", "referring_gp_alvarez",
                                      "practice_manager_kim"],
                            institutions=[],
                            evidence="Kim can pull referral-source stats within a week. "
                                     "GP Alvarez was the top referrer and stopped in "
                                     "February. Reversing the fee costs 8k/quarter; the "
                                     "partner program costs 5k to stand up.")
    ctxt, rep = live_context(schema, ["sam_whitfield", "referring_gp_alvarez",
                                      "practice_manager_kim"], llm=llm,
                             resources=seed_resources(schema, "sam_whitfield", {"budget": 20000.0}))
    contract = DecisionProblem(
        decision_id="probe4", decision_maker="sam_whitfield", role="clinic_owner",
        authority=["clinic_owner"], controllable_resources={"budget_usd": 20000.0},
        context=question, horizon="2025-10-15T00:00:00Z",
        information_gathering_allowed=True)
    res = discover_best_action(question, ctxt, problem=contract, llm=llm, seed=3,
                               trace_path=os.path.join(pd, "role_trace.jsonl"),
                               budget="diagnostic")
    finish(pd, res, rep, t0, contract, schema)
    return res


def probe_5_novel_user_action(llm, offline):
    t0 = time.time()
    pd = os.path.join(ART, "probe5_novel_user_action")
    question = ("Landlord negotiating a disputed commercial deposit: what happens if I "
                "escrow the disputed amount with a neutral third party while filing a "
                "conditional withdrawal of my damage claim, contingent on the tenant "
                "withdrawing their complaint?")
    novel = ("Escrow the disputed 18,000 deposit with the county neutral-holder program AND "
             "simultaneously file a conditional withdrawal of my damage claim that only "
             "takes effect if the tenant withdraws their board complaint within 14 days.")
    world_question = ("Will the deposit dispute between landlord Petrov and tenant Okada be "
                      "resolved without a board-imposed penalty by October 2025?")
    schema = compile_schema(llm, question=world_question,
                            entities=["landlord_petrov", "tenant_okada"],
                            institutions=["tenancy_dispute_board"],
                            evidence="The tenancy dispute board resolves complaints; the "
                                     "county neutral-holder program accepts escrowed "
                                     "deposits; the tenant filed a complaint in June.")
    ctxt, rep = live_context(schema, ["landlord_petrov", "tenant_okada"], llm=llm,
                             resources=seed_resources(schema, "landlord_petrov", {"deposit": 18000.0}))
    contract = DecisionProblem(
        decision_id="probe5", decision_maker="landlord_petrov", role="landlord",
        authority=["landlord"], controllable_resources={"deposit_usd": 18000.0},
        context=question, horizon="2025-10-01T00:00:00Z")
    res = evaluate_proposed_actions(question, [novel], ctxt, problem=contract, llm=llm,
                                    seed=2, trace_path=os.path.join(pd, "role_trace.jsonl"))
    finish(pd, res, rep, t0, contract, schema)
    return res


PROBES = {1: probe_1_founder_launch, 2: probe_2_partnership_terms, 3: probe_3_nonmessage,
          4: probe_4_info_then_act, 5: probe_5_novel_user_action}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0, help="run one probe (0 = all)")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    llm = None
    if not args.offline:
        llm = default_chat_fn(max_tokens=1800, temperature=0.3)
        if llm is None:
            raise SystemExit("no LLM backend — live probes need DEEPSEEK_API_KEY "
                             "(use --offline for the structural smoke)")
    if args.offline:
        # structural smoke: offline schema compile is impossible (compiler needs an LLM);
        # reuse the test fixture schema to exercise the probe plumbing only
        from tests.scenario_fixtures import council_schema
        schema = council_schema()
        ctxt, rep = live_context(schema, ["rivera", "chen"], llm=None)
        contract = DecisionProblem(decision_id="offline_smoke", decision_maker="rivera",
                                   authority=["petitioner"], context="smoke")
        res = evaluate_proposed_actions("smoke", ["file the petition"], ctxt,
                                        problem=contract)
        print("offline smoke:", res.recommendation_kind,
              "| candidates:", len(res.provenance["scenario_report"]["candidates"]))
        return
    for n, fn in PROBES.items():
        if args.probe and n != args.probe:
            continue
        print(f"[probe {n}] {fn.__name__}")
        try:
            fn(llm, args.offline)
        except Exception as e:  # noqa: BLE001 — a failed probe is recorded, not hidden
            import traceback
            tb = traceback.format_exc()
            pd = os.path.join(ART, f"probe{n}_FAILED")
            save(pd, "failure.json", {"probe": n, "error": f"{type(e).__name__}: {e}",
                                      "traceback": tb})
            print(f"  !! probe {n} failed: {type(e).__name__}: {e}\n{tb[-1200:]}")


if __name__ == "__main__":
    main()
