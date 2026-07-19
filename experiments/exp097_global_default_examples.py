"""EXP-097 — five live end-to-end best-action examples through the PUBLIC production API.

The generated action layer is the global default: every example calls a phase13.api entry
point (`recommend_action`, `evaluate_actions`, `optimize_policy`, `value_of_information`) on
a generated scenario world compiled live from the question. Other actors respond through
their own persistent qualitative simulations; every actor LLM call is captured verbatim
(actor_trace.jsonl); matched worlds are dumped with their exact content
(forensic_worlds.jsonl); each example renders a step-by-step human-readable FORENSIC.md.

Domains: 1 consequential outreach · 2 negotiation/partnership (contingent policy) ·
3 pricing/launch (user-proposed comparison) · 4 institutional/coalition decision ·
5 operational information-gathering (VOI).

Run:  PYTHONPATH=. python experiments/exp097_global_default_examples.py [--example N]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.world_model_v2.phase13 import api as p13
from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13.scenario_actions.forensics import render_forensic_md

from experiments.exp096_scenario_action_probes import (DAY, T0, compile_schema, live_context,
                                                       save, seed_resources)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13",
                   "action_language", "examples")


class ActorTracingChat:
    """Verbatim capture of every actor-simulation LLM call: what the invoked actor was SHOWN
    (the prompt built from their own observable view) and what they ANSWERED."""

    def __init__(self, fn, path):
        self.fn, self.path, self.n = fn, path, 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, prompt, **kw):
        t0 = time.time()
        out = self.fn(prompt, **kw) if kw else self.fn(prompt)
        self.n += 1
        with open(self.path, "a") as f:
            f.write(json.dumps({"call": self.n, "wall_s": round(time.time() - t0, 2),
                                "prompt": str(prompt), "response": str(out)}) + "\n")
        return out


def run_example(name, *, schema_args, actors, resources_hint, contract_kwargs, goal,
                entry, entry_kwargs=None, n_particles=3):
    ex_dir = os.path.join(ART, name)
    os.makedirs(ex_dir, exist_ok=True)
    t0 = time.time()
    role_llm = default_chat_fn(max_tokens=1800, temperature=0.3)
    actor_llm = ActorTracingChat(default_chat_fn(max_tokens=900, temperature=0.4),
                                 os.path.join(ex_dir, "actor_trace.jsonl"))
    schema = compile_schema(role_llm, **schema_args)
    maker = contract_kwargs["decision_maker"]
    ctxt, rep = live_context(schema, actors, llm=actor_llm,
                             resources=seed_resources(schema, maker, resources_hint),
                             n_particles=n_particles)
    contract = DecisionProblem(**contract_kwargs)
    kwargs = dict(goal_text=goal, seed=13, llm=role_llm,
                  trace_path=os.path.join(ex_dir, "role_trace.jsonl"),
                  forensic_dir=ex_dir, **(entry_kwargs or {}))
    if entry == "recommend_action":
        res = p13.recommend_action(contract, ctxt, **kwargs)
    elif entry == "evaluate_actions":
        res = p13.evaluate_actions(contract, kwargs.pop("actions"), ctxt, **kwargs)
    elif entry == "optimize_policy":
        res = p13.optimize_policy(contract, True, ctxt, **kwargs)
    elif entry == "value_of_information":
        kwargs.pop("trace_path", None)
        kwargs.pop("forensic_dir", None)
        voi = p13.value_of_information(contract, [], ctxt, forensic_dir=ex_dir,
                                       trace_path=os.path.join(ex_dir, "role_trace.jsonl"),
                                       **kwargs)
        save(ex_dir, "voi.json", voi)
        res = p13.recommend_action(contract, ctxt, goal_text=goal, seed=13, llm=role_llm)
    else:
        raise ValueError(entry)
    # forensic dump comes from a dedicated re-run seam only when the entry didn't produce it;
    # to guarantee it we re-invoke the scenario layer directly with forensic_dir when needed
    if not os.path.exists(os.path.join(ex_dir, "forensic_worlds.jsonl")):
        from swm.world_model_v2.phase13.scenario_actions.api import evaluate_actions_generated
        sr = res.provenance["scenario_report"]
        finalists = [c["candidate_id"] for c in sr["candidates"]][:4]
        # re-run the surviving candidates only, deterministically, dumping worlds
        from swm.world_model_v2.phase13.scenario_actions.candidates import (ConcreteAction,
                                                                            PlanStep,
                                                                            ConditionSpec)
        cands = []
        for c in sr["candidates"]:
            if c["candidate_id"] == "do_nothing" or not c.get("steps"):
                continue
            steps = []
            for s in c["steps"]:
                st = PlanStep(step_id=s["step_id"], intent=s.get("intent", ""),
                              target_ids=list(s.get("target_ids") or []),
                              channel=s.get("channel", ""),
                              exact_content=s.get("exact_content", ""),
                              terms=dict(s.get("terms") or {}),
                              timing_ts=s.get("timing_ts"),
                              after_steps=list(s.get("after_steps") or []),
                              conditions=[ConditionSpec(**{k: v for k, v in cc.items()
                                                           if k in ("kind", "record_type",
                                                                    "field", "op", "value",
                                                                    "description")})
                                          for cc in (s.get("conditions") or [])],
                              visibility=s.get("visibility", "participants"),
                              resource_commitments=dict(s.get("resource_commitments") or {}))
                st.compiled_ops = [op for grp in sr["compiled_effects"].get(
                    c["candidate_id"], []) if grp["step"] == s["step_id"]
                    for op in grp["ops"]]
                st.compile_meta = {"compiler": "replay_precompiled"}
                steps.append(st)
            if steps:
                cands.append(ConcreteAction(candidate_id=c["candidate_id"],
                                            actor_id=c["actor_id"],
                                            title=c.get("title", ""), steps=steps))
        ctxt2, _ = live_context(schema, actors, llm=actor_llm,
                                resources=seed_resources(schema, maker, resources_hint),
                                n_particles=n_particles)
        evaluate_actions_generated(contract, cands[:4], ctxt2, goal_text=goal, seed=13,
                                   llm=None, forensic_dir=ex_dir)
    save(ex_dir, "contract.json", {k: contract_kwargs.get(k) for k in
                                   ("decision_id", "decision_maker", "authority",
                                    "controllable_resources", "context", "horizon")})
    save(ex_dir, "schema.json", schema.as_dict())
    save(ex_dir, "result.json", {"result": res.as_dict(),
                                 "actor_llm_calls": actor_llm.n,
                                 "wall_s": round(time.time() - t0, 1)})
    md = render_forensic_md(ex_dir, title=name)
    hs = res.provenance.get("human_summary", {})
    print(f"  -> {res.recommendation_kind}: {res.recommended} | actors called "
          f"{actor_llm.n}x | wall {round(time.time() - t0)}s | {os.path.relpath(md)}")
    return res


# ---------------------------------------------------------------- the five examples
def ex1_outreach():
    return run_example(
        "ex1_outreach",
        schema_args=dict(
            question="Will Halcyon Analytics renew its 120k contract after the botched "
                     "migration, or churn by the September renewal date?",
            entities=["nina_petrova", "marcus_webb", "success_lead_tran"],
            institutions=[],
            evidence="Nina Petrova is Halcyon's founder-CEO (vendor side); Marcus Webb is "
                     "the customer's VP Data whose team suffered a two-day outage during "
                     "the migration and has gone quiet; Tran is Nina's customer-success "
                     "lead with day-to-day access to Webb's engineers. Renewal decision "
                     "rests solely with Webb. A goodwill credit of up to 15k is within "
                     "Nina's authority. Renewal date ~6 weeks out.",
            horizon_days=45),
        actors=["nina_petrova", "marcus_webb", "success_lead_tran"],
        resources_hint={"goodwill_credit": 15000.0},
        contract_kwargs=dict(decision_id="ex1", decision_maker="nina_petrova",
                             role="founder_ceo", authority=["founder_ceo"],
                             controllable_resources={"goodwill_credit": 15000.0},
                             context="Recover the Halcyon renewal after the botched "
                                     "migration; Webb has gone quiet.",
                             horizon="2025-09-01T00:00:00Z"),
        goal="get Halcyon to renew (or at least re-engage) without burning trust or "
             "discounting reflexively",
        entry="recommend_action")


def ex2_partnership():
    return run_example(
        "ex2_partnership",
        schema_args=dict(
            question="Will Kite Robotics and Halvorsen Motors sign a component-supply "
                     "partnership before the fall production freeze?",
            entities=["amara_diallo", "erik_halvorsen", "sofia_brandt"],
            institutions=["halvorsen_procurement_committee"],
            evidence="Erik Halvorsen (CEO) delegates supplier onboarding to the "
                     "procurement committee chaired by Sofia Brandt; the committee meets "
                     "monthly and its decision is recorded; the production freeze is in "
                     "~8 weeks; Kite can offer a 12% volume discount or a co-marketing "
                     "clause, not both.",
            horizon_days=60),
        actors=["amara_diallo", "erik_halvorsen", "sofia_brandt"],
        resources_hint={},
        contract_kwargs=dict(decision_id="ex2", decision_maker="amara_diallo",
                             role="bd_lead", authority=["bd_lead"],
                             context="Secure the Halvorsen supply partnership before the "
                                     "freeze; committee route vs CEO route; discount vs "
                                     "co-marketing.",
                             horizon="2025-09-30T00:00:00Z"),
        goal="signed component-supply partnership before the production freeze, with terms "
             "Kite can honor (discount OR co-marketing, never both)",
        entry="optimize_policy")


def ex3_pricing_launch():
    return run_example(
        "ex3_pricing_launch",
        schema_args=dict(
            question="Will Mara Voss's scheduling copilot reach a committed public launch "
                     "or at least three signed design partners by mid-September 2025?",
            entities=["mara_voss", "devon_reyes", "priya_shah"],
            institutions=["seed_investor_board"],
            evidence="Mara Voss is the founder/CEO. Devon Reyes runs the enterprise pilot "
                     "at Calder Logistics (decides pilot expansion, reports in ~3 weeks). "
                     "Priya Shah chairs the 3-seat board that must approve launch spend "
                     "over 50k. Mara holds 40k discretionary launch budget.",
            horizon_days=60),
        actors=["mara_voss", "devon_reyes", "priya_shah"],
        resources_hint={"launch_budget": 40000.0},
        contract_kwargs=dict(decision_id="ex3", decision_maker="mara_voss",
                             role="founder_ceo", authority=["founder_ceo"],
                             controllable_resources={"launch_budget": 40000.0},
                             context="Launch now vs recruit design partners privately vs "
                                     "delay for the pilot readout.",
                             horizon="2025-09-16T00:00:00Z"),
        goal="a committed public launch or 3+ signed design partners by mid-September, "
             "without board conflict",
        entry="evaluate_actions",
        entry_kwargs={"actions": [
            "Announce the public launch this week using the 40k discretionary budget, and "
            "tell the board after the announcement.",
            "Privately recruit three design partners with hands-on onboarding before any "
            "public launch, starting with Devon Reyes at Calder.",
            "Wait for the Calder pilot readout in three weeks, then decide with the pilot "
            "data in hand.",
        ]})


def ex4_coalition():
    return run_example(
        "ex4_coalition",
        schema_args=dict(
            question="Will the Harborview Cooperative's board adopt the workspace-sublease "
                     "bylaw amendment at or before the October meeting?",
            entities=["june_okafor", "raul_mendes", "petra_lindqvist"],
            institutions=["harborview_coop_board"],
            evidence="The three-member board (June Okafor, Raul Mendes, Petra Lindqvist) "
                     "adopts bylaw amendments by majority vote, each member recording a "
                     "written decision. June (the proposer) supports subleasing; Raul "
                     "worries about liability; Petra is undecided and trusts Raul's "
                     "judgment on liability questions. Meetings are monthly; a special "
                     "meeting can be called with 7 days notice.",
            horizon_days=75),
        actors=["june_okafor", "raul_mendes", "petra_lindqvist"],
        resources_hint={},
        contract_kwargs=dict(decision_id="ex4", decision_maker="june_okafor",
                             role="board_member", authority=["board_member"],
                             context="Assemble a board majority for the sublease "
                                     "amendment; Raul's liability concern is the pivot.",
                             horizon="2025-10-31T00:00:00Z"),
        goal="the bylaw amendment adopted by board majority at or before the October "
             "meeting, with Raul's liability concern genuinely addressed (not steamrolled)",
        entry="recommend_action")


def ex5_info_gathering():
    return run_example(
        "ex5_info_gathering",
        schema_args=dict(
            question="Will the clinic's monthly referrals recover to their December level "
                     "by mid-October 2025?",
            entities=["sam_whitfield", "referring_gp_alvarez", "practice_manager_kim"],
            institutions=[],
            evidence="Kim can pull referral-source stats within a week; the practice "
                     "management system logs every referral with source and month, so "
                     "monthly counts are directly measurable against the "
                     "seasonally-adjusted December baseline of 60/month. GP Alvarez was "
                     "the top referrer (25/month) and stopped in February; whether he is "
                     "WILLING to resume is unknown until someone asks him, and his "
                     "answer (willing / conditional / permanently moved) is discoverable "
                     "by a direct conversation. Referral relationships decay slowly: a "
                     "GP who stays away past October is unlikely to return this year. "
                     "Sam Whitfield holds sole authority over fees and programs; any of "
                     "Sam's decisions can also be delayed or dropped (a decision not "
                     "recorded within two weeks lapses). Reversing the January fee "
                     "change costs 8k/quarter; a referral-partner program costs 5k.",
            horizon_days=90),
        actors=["sam_whitfield", "referring_gp_alvarez", "practice_manager_kim"],
        resources_hint={"budget": 20000.0},
        contract_kwargs=dict(decision_id="ex5", decision_maker="sam_whitfield",
                             role="clinic_owner", authority=["clinic_owner"],
                             controllable_resources={"budget": 20000.0},
                             context="Referrals dropped; cause unknown (fee change vs "
                                     "rival clinic); decide between reversing the fee, a "
                                     "partner program, or finding out first.",
                             horizon="2025-10-15T00:00:00Z",
                             information_gathering_allowed=True),
        goal="monthly referrals back to the December level by mid-October without "
             "spending on the wrong cause",
        entry="value_of_information")


EXAMPLES = {1: ex1_outreach, 2: ex2_partnership, 3: ex3_pricing_launch, 4: ex4_coalition,
            5: ex5_info_gathering}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, default=0)
    args = ap.parse_args()
    for n, fn in EXAMPLES.items():
        if args.example and n != args.example:
            continue
        print(f"[example {n}] {fn.__name__}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a failed example is recorded, never hidden
            import traceback
            ex_dir = os.path.join(ART, f"ex{n}_FAILED")
            save(ex_dir, "failure.json", {"example": n,
                                          "error": f"{type(e).__name__}: {e}",
                                          "traceback": traceback.format_exc()})
            print(f"  !! example {n} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
