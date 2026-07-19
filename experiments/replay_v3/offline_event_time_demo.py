"""OFFLINE production-path event-time demo — the full post-evidence pipeline with PINNED elicitations.

WHY OFFLINE: this environment's network policy blocks the LLM API and every evidence host, so the
LLM-elicitation outputs (mode candidates, stance classifications, process states, entailment
judgments) are PINNED to what the previous LIVE compile of the same question produced (PR #100's
branch trace, 2026-07-16). Everything downstream is the REAL production code path, unmodified:
mode_graph.canonical_modes reconciles the pinned passes; resolution_criteria.ground_actor_intentions
runs the real grounding (fake-llm returning the pinned JSON) — commitments, prohibitions, aggregate;
the QUARANTINED legacy periodic scheduler feeds the ablation arm; event_time.convert_to_event_time
builds the real machinery; materialize.run_from_plan → build_world → phase-4 policies → pathway
writes → hazard rounds → absorption monitor → EventTimeContract readout. The rollout layer is
LLM-free by design, so the simulation itself is production-exact.

Primary question: "When will the Russia-Ukraine conflict end?" (as-of 2026-07-16, horizon
2028-12-31). Cross-domain fixtures (bill / product launch / inflation threshold) prove the same
engine runs institutional-majority, operational-hierarchy, and world-driven-threshold worlds.

Run:  PYTHONPATH=. python experiments/replay_v3/offline_event_time_demo.py [--particles 200]
Logs: exact per-particle event traces for sampled particles + aggregate readout;
      JSON artifact → experiments/results/replay_v3/offline_event_time_demo.json
"""
import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swm.world_model_v2.event_time import convert_to_event_time
from swm.world_model_v2.legacy_ablations import (ABLATION_TOKEN,
                                                 legacy_periodic_review_ablation)
from swm.world_model_v2.materialize import run_from_plan
from swm.world_model_v2.mode_graph import (canonical_modes, declare_pathway_processes,
                                           ground_process_states, mode_pathway, progress_var)
from swm.world_model_v2.resolution_criteria import ground_actor_intentions
from swm.world_model_v2.state import parse_time

OUT = Path("experiments/results/replay_v3/offline_event_time_demo.json")


def _d(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


class Plan(SimpleNamespace):
    def plan_hash(self):
        return "offline_demo"


# ================================================================ pinned Ukraine compile (live trace)
CRITERION = {
    "subject": "Russia-Ukraine conflict",
    "predicate": "armed hostilities between Russia and Ukraine have ended",
    "deadline": None,
    "resolves_yes_iff": ("a formal peace treaty or ceasefire agreement is signed and in effect, OR "
                         "one side achieves decisive military victory and the other ceases all "
                         "organized combat, with no active hostilities for >=30 consecutive days"),
    "absorbing_event": "the conflict-ending agreement or decisive-victory cessation takes effect",
    "event_polarity": "occurrence_resolves_yes",
    "near_miss_states": [
        {"state": "temporary ceasefire that collapses within 30 days", "resolves": "no"},
        {"state": "victory declaration with continuing organized insurgency", "resolves": "no"}],
}

# the three pinned mode-elicitation passes (live compile variance: pass 1 time-indexed + a
# non-absorbing candidate; passes reconciled by canonical_modes majority vote)
_MODE_PASSES = {
    1: {"modes": [
        {"id": "ceasefire_2026", "prior": 0.20, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]},
         "describe": "bilateral ceasefire in effect"},
        {"id": "ceasefire_2027", "prior": 0.10, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]}},
        {"id": "comprehensive_peace_treaty", "prior": 0.15, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]}},
        {"id": "russian_victory", "prior": 0.30, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}},
        {"id": "ukrainian_victory", "prior": 0.20, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}},
        {"id": "frozen_conflict_stalemate", "prior": 0.05, "pathway": "unilateral_action"}]},
    2: {"modes": [
        {"id": "ceasefire", "prior": 0.30, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]},
         "describe": "ceasefire agreement signed and holding"},
        {"id": "peace_treaty", "prior": 0.15, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]}},
        {"id": "russian_victory", "prior": 0.30, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}},
        {"id": "ukrainian_victory", "prior": 0.25, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}}]},
    3: {"modes": [
        {"id": "ceasefire_agreement", "prior": 0.28, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]}},
        {"id": "peace_treaty", "prior": 0.12, "pathway": "cooperative_agreement",
         "decision_structure": {"rule": "unanimity", "approvers": ["Vladimir_Putin", "Volodymyr_Zelenskyy"]}},
        {"id": "russian_victory", "prior": 0.32, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}},
        {"id": "ukrainian_victory", "prior": 0.23, "pathway": "unilateral_action",
         "decision_structure": {"rule": "unilateral"}},
        {"id": "nato_direct_escalation", "prior": 0.05, "pathway": "unilateral_action"}]},
}

# entailment filter (live): frozen_conflict / NATO-escalation do NOT satisfy the criterion
_ENTAILMENT = {"absorbing_mode_ids": ["ceasefire", "peace_treaty", "russian_victory",
                                      "ukrainian_victory"],
               "rejected": [{"id": "frozen_conflict_stalemate", "why": "hostilities do not end"},
                            {"id": "nato_direct_escalation", "why": "escalation is not an end state"}]}

# pinned process-state classification (live evidence, 2026-07: Trump calls = feelers;
# summer offensive = campaign underway)
_PROCESS_STATES = {"process_states": [
    {"pathway": "cooperative_agreement", "state": "exploratory",
     "basis": "Trump holds separate calls with Putin and Zelenskyy (CBS)"},
    {"pathway": "unilateral_action", "state": "active",
     "basis": "Putin likely to escalate Ukraine war, despite Trump peace push (Reuters)"}]}

# pinned stance classification (live evidence + mode scoping) — CLASSIFICATION ONLY, no numbers
_INTENTIONS = {"intentions": [
    {"actor": "Vladimir_Putin", "stated_intention": "Escalate the war; no capitulation to a ceasefire",
     "basis_quote": "Putin likely to escalate Ukraine war, despite Trump peace push (Reuters)",
     "source": "evidence", "commitment_level": "committed_to_prevent", "target_mode": None,
     "pathway": "cooperative_agreement", "control": "veto", "capability": "high",
     "reliability": "high", "entails_direction": "no", "date": "2026-07-10"},
    {"actor": "Vladimir_Putin", "stated_intention": "Pursue full military objectives in Ukraine",
     "basis_quote": "objectives of the special military operation will be achieved",
     "source": "evidence", "commitment_level": "actively_pursuing", "target_mode": "russian_victory",
     "pathway": "unilateral_action", "control": "sole_authority", "capability": "high",
     "reliability": "high", "entails_direction": "no", "date": "2026-07-01"},
    {"actor": "Vladimir_Putin", "stated_intention": "Deny Ukraine any battlefield victory",
     "basis_quote": "Russia will never accept defeat on the battlefield",
     "source": "model_knowledge", "commitment_level": "committed_to_prevent",
     "target_mode": "ukrainian_victory", "pathway": "unilateral_action",
     "control": "operational_capability", "capability": "high", "reliability": "medium",
     "entails_direction": "no", "date": None},
    {"actor": "Volodymyr_Zelenskyy", "stated_intention": "No frozen conflict without security guarantees",
     "basis_quote": "we will not accept a frozen conflict without real security guarantees",
     "source": "evidence", "commitment_level": "conditionally_opposed", "target_mode": "ceasefire",
     "pathway": "cooperative_agreement", "control": "veto", "capability": "high",
     "reliability": "high", "entails_direction": "no", "date": "2026-07-08"},
    {"actor": "Volodymyr_Zelenskyy", "stated_intention": "Restore Ukraine's territorial integrity",
     "basis_quote": "we will liberate our land", "source": "evidence",
     "commitment_level": "actively_pursuing", "target_mode": "ukrainian_victory",
     "pathway": "unilateral_action", "control": "sole_authority", "capability": "medium",
     "reliability": "high", "entails_direction": "no", "date": "2026-06-28"},
    {"actor": "Donald_Trump", "stated_intention": "Broker an end to the war",
     "basis_quote": "Trump holds separate calls with Putin and Zelenskyy (CBS)",
     "source": "evidence", "commitment_level": "actively_pursuing", "target_mode": None,
     "pathway": "cooperative_agreement", "control": "informal_influence", "capability": "medium",
     "reliability": "high", "entails_direction": "yes", "date": "2026-07-12"},
    {"actor": "Verkhovna_Rada", "stated_intention": "Law prohibits negotiating with Putin",
     "basis_quote": "the 2022 NSDC decision bans negotiations with Putin (law in force)",
     "source": "evidence", "commitment_level": "committed_to_prevent", "target_mode": None,
     "pathway": "cooperative_agreement", "control": "agenda_setting", "capability": "high",
     "reliability": "high", "entails_direction": "no", "date": "2026-07-01"},
    {"actor": "Russian_General_Staff", "stated_intention": "Sustain offensive operations",
     "basis_quote": "summer offensive continues along the eastern front", "source": "evidence",
     "commitment_level": "actively_pursuing", "target_mode": "russian_victory",
     "pathway": "unilateral_action", "control": "operational_capability", "capability": "high",
     "reliability": "medium", "entails_direction": "no", "date": "2026-07-05"},
]}


def _fake_llm(prompt: str) -> str:
    """Pinned elicitations from the live compile — every consuming code path is the real one."""
    if "PASS 1" in prompt:
        return json.dumps(_MODE_PASSES[1])
    if "PASS 2" in prompt:
        return json.dumps(_MODE_PASSES[2])
    if "PASS 3" in prompt:
        return json.dumps(_MODE_PASSES[3])
    if "SATISFY the resolution criterion" in prompt:
        return json.dumps(_ENTAILMENT)
    if "STRATEGIC ACTOR" in prompt:
        return json.dumps(_INTENTIONS)
    if "dormant" in prompt:
        return json.dumps(_PROCESS_STATES)
    return "{}"


def build_ukraine_plan(n_particles: int) -> Plan:
    as_of = parse_time("2026-07-16")
    horizon = parse_time("2028-12-31")
    ents = [
        {"id": "Vladimir_Putin", "type": "person", "sensitivity": 0.95,
         "fields": {"roles": ["president_of_russia"], "goals": ["prevail_in_war"]}},
        {"id": "Volodymyr_Zelenskyy", "type": "person", "sensitivity": 0.95,
         "fields": {"roles": ["president_of_ukraine"], "goals": ["restore_territory"]}},
        {"id": "Donald_Trump", "type": "person", "sensitivity": 0.9,
         "fields": {"roles": ["president_of_united_states"], "goals": ["broker_deal"]}},
        {"id": "Verkhovna_Rada", "type": "institution", "sensitivity": 0.7,
         "fields": {"roles": ["ukrainian_parliament"]}},
        {"id": "Russian_General_Staff", "type": "institution", "sensitivity": 0.75,
         "fields": {"roles": ["military_command"]}},
        {"id": "Ukrainian_Ministry_of_Defense", "type": "institution", "sensitivity": 0.72,
         "fields": {"roles": ["military_command"]}},
        {"id": "Xi_Jinping", "type": "person", "sensitivity": 0.6,
         "fields": {"roles": ["president_of_china"]}},
        {"id": "Mark_Rutte", "type": "person", "sensitivity": 0.55,
         "fields": {"roles": ["nato_secretary_general"]}},
        {"id": "Olaf_Scholz", "type": "person", "sensitivity": 0.5, "fields": {"roles": ["chancellor"]}},
        {"id": "Joe_Biden", "type": "person", "sensitivity": 0.35, "fields": {"roles": ["former_president"]}},
        {"id": "Russian_Ministry_of_Defense", "type": "institution", "sensitivity": 0.5,
         "fields": {"roles": ["ministry"]}},
        {"id": "Federation_Council", "type": "institution", "sensitivity": 0.4,
         "fields": {"roles": ["russian_upper_house"]}},
        {"id": "State_Duma", "type": "institution", "sensitivity": 0.4,
         "fields": {"roles": ["russian_lower_house"]}},
    ]
    insts = [
        {"id": "Verkhovna_Rada", "rules": [
            {"kind": "quorum", "params": {"total": 450, "needed": 226, "members": []},
             "_provenance": "grounded_rules:evidence"}]},
        {"id": "Federation_Council", "rules": [
            {"kind": "quorum", "params": {"total": 178, "needed": 90, "members": []},
             "_provenance": "grounded_rules:model_knowledge"}]},
        {"id": "State_Duma", "rules": [
            {"kind": "quorum", "params": {"total": 450, "needed": 226, "members": []},
             "_provenance": "grounded_rules:model_knowledge"}]},
    ]
    rels = [
        {"src": "Donald_Trump", "rel": "communicates_with", "dst": "Vladimir_Putin"},
        {"src": "Donald_Trump", "rel": "communicates_with", "dst": "Volodymyr_Zelenskyy"},
        {"src": "Vladimir_Putin", "rel": "controls", "dst": "Russian_General_Staff"},
        {"src": "Volodymyr_Zelenskyy", "rel": "controls", "dst": "Ukrainian_Ministry_of_Defense"},
        {"src": "Verkhovna_Rada", "rel": "influences", "dst": "Volodymyr_Zelenskyy"},
        {"src": "Xi_Jinping", "rel": "influences", "dst": "Vladimir_Putin"},
    ]
    plan = Plan(
        question="When will the Russia-Ukraine conflict end?",
        as_of=as_of, horizon_ts=horizon,
        outcome_contract=SimpleNamespace(options=[]),          # replaced by convert_to_event_time
        entities=ents, populations=[], institutions=insts, relations=rels,
        quantities=[], latents=[], scheduled_events=[], stochastic_hazards=[],
        accepted_mechanisms=[{"mech_id": "actor_policy", "operator": "production_actor_policy",
                              "ontology_type": "decision", "causal_role": "strategic actor decisions",
                              "parameter_source": "tier-7 broad structural mixture",
                              "temporal_scale": "event", "calibration_status": "broad_prior",
                              "sensitivity": 0.9}],
        candidate_experimental_mechanisms=[], rejected_mechanisms=[],
        structural_hypotheses=[], actor_decisions=[], mechanism_choices=[],
        fallbacks_used=[], support_grade="exploratory", interpretations=[], omissions=[],
        degraded=False, fidelity_plan={}, uncertainty_plan={},
        compute_plan={"n_particles": n_particles}, provenance={"resolution_criterion": CRITERION},
        version=1, parent_version=0)
    return plan


def assemble(plan: Plan, llm) -> dict:
    """The post-evidence pipeline stages, in the runtime's order — every function is production code."""
    lineage = {}
    # 1. canonical mode decomposition (K=3 pinned passes reconciled by majority vote)
    modes, consensus = canonical_modes(question=plan.question, criterion=CRITERION,
                                       hypotheses=[], options=[], llm=llm, k_passes=3)
    plan._canonical_modes, plan._mode_consensus = modes, consensus
    lineage["mode_graph"] = consensus
    # 2. pathway-process grounding + declaration
    pws = sorted({mode_pathway(m) for m in modes})
    states = ground_process_states(plan.question, CRITERION, pws, as_of="2026-07-16",
                                   evidence_text="(pinned)", llm=llm)
    lineage["pathway_processes"] = declare_pathway_processes(plan, modes, grounding=states)
    # 3. mode-scoped stance grounding (real grounding code; pinned classifications)
    lineage["actor_intentions"] = ground_actor_intentions(plan, plan.question, criterion=CRITERION,
                                                          evidence_text="(pinned)", llm=llm,
                                                          modes=modes)
    # 3b. capability becomes a live, depletable capacity resource
    from swm.world_model_v2.world_dynamics import declare_actor_capacity
    lineage["actor_capacity"] = declare_actor_capacity(plan)
    # 4. LEGACY ABLATION comparison arm: the quarantined periodic scheduler (explicitly
    #    acknowledged) — production uses scenario-generated decision triggers instead
    lineage["trajectory_depth"] = legacy_periodic_review_ablation(
        plan, {"phase4_actor_policy": {"required": True}}, cadence_days=7.0,
        acknowledge=ABLATION_TOKEN)
    # 5. event-time conversion (real machinery; entailment filter pinned)
    lineage["event_time"] = convert_to_event_time(plan, CRITERION, lineage=lineage, llm=llm)
    return lineage


def _extract_story(branch, plan, sample_len=28):
    """Chronological human-readable trace of one particle from its StateDelta log."""
    rows = []
    for d in branch.log:
        when = _d(d.at)
        if d.event_type == "actor_action":
            act, actor = None, None
            for c in d.changes:
                if c["path"].endswith(".current_action"):
                    actor = c["path"].split(".")[0]
                    act = (c["after"] or {}).get("action_name")
            def _chan(path):
                return (path.split("[")[-1].rstrip("]")
                        .replace("pathway_progress:", "").replace("mode_progress:", "mode:"))
            moves = [f"{_chan(c['path'])} {c['before']}→{c['after']}" for c in d.changes
                     if "pathway_progress" in c["path"] or "mode_progress" in c["path"]]
            if act:
                rows.append(f"{when}  ACTION       {actor:<28} chose {act:<14}"
                            + (f" | process {'; '.join(moves)}" if moves else ""))
        elif d.event_type == "action_blocked":
            a = (d.uncertainty.get("action") or {})
            rows.append(f"{when}  BLOCKED      {a.get('actor_id', '?'):<28} tried {a.get('action_name', '?')}"
                        f" ({(d.reason_codes + ['?'])[1]})")
        elif d.event_type == "hazard_round":
            u = d.uncertainty
            tail = ""
            if any("absorbing_state_reached" in c["path"] for c in d.changes):
                tail = "  → ABSORBED"
            elif "provisional_pending_persistence" in d.reason_codes:
                tail = "  → PROVISIONAL (must hold 30d)"
            live = "  [hr recomputed from live stances]" if u.get("hr_source") == "live_recomputed" else ""
            rows.append(f"{when}  HAZARD       mode={d.reason_codes[0].split('=')[1]:<18}"
                        f" h={u.get('hazard'):.4f} state×{u.get('state_hazard_factor'):.3f}"
                        f" hr={u.get('sampled_hazard_ratio')}" + tail + live)
        elif d.event_type == "stance_review":
            for c in d.changes:
                actor = c["path"].split(".")[0]
                if "resources[capacity]" in c["path"]:
                    rows.append(f"{when}  ATTRITION    {actor:<28} capacity "
                                f"{c['before']} → {c['after']}  (contested pursuit drains)")
                else:
                    rule = next((r.split(":", 1)[1] for r in d.reason_codes
                                 if r.startswith(actor)), "?")
                    rows.append(f"{when}  STANCE       {actor:<28} "
                                f"{c['before']} → {c['after']}  ({rule})")
        elif d.event_type == "persistence_check":
            held = any("persisted" in r for r in d.reason_codes)
            rows.append(f"{when}  PERSISTENCE  "
                        + ("HELD ≥30d — criterion satisfied, absorption confirmed" if held
                           else "COLLAPSED — near-miss realized (temporary end-state failed)"))
        elif d.event_type == "absorption":
            rows.append(f"{when}  ABSORPTION   first passage observed (absorbed_at stamped)")
        elif d.event_type == "institutional_decision":
            u = d.uncertainty
            rows.append(f"{when}  INSTITUTION  yes={u.get('yes')}/{u.get('n_members')}"
                        f" needed={u.get('needed')} "
                        + ("PASSED" if any("absorbing" in c["path"] or c.get("after") == "passed"
                                           for c in d.changes) else "failed"))
    if len(rows) > sample_len:
        # center the excerpt on FIRST PASSAGE (the story's climax: provisional entry, persistence
        # outcome, absorption, and the stance dynamics around them), not the calendar's start
        anchor = next((i for i, r in enumerate(rows)
                       if "ABSORPTION" in r or "PERSISTENCE" in r or "PROVISIONAL" in r), None)
        head_n = 10
        if anchor is None or anchor < sample_len - 4:
            head, tail = rows[:sample_len - 8], rows[-8:]
            rows = head + [f"  … {len(rows) - sample_len} more events …"] + tail
        else:
            lo = max(head_n, anchor - (sample_len - head_n - 4))
            head = rows[:head_n]
            mid = rows[lo:anchor + 4]
            rows = (head + [f"  … {lo - head_n} more events …"] + mid
                    + ([f"  … {len(rows) - anchor - 4} more events …"]
                       if anchor + 4 < len(rows) else []))
    return rows


def run_ukraine(n_particles: int, seed: int = 0) -> dict:
    print("=" * 100)
    print("UKRAINE DEMO — When will the Russia-Ukraine conflict end?  "
          "(as-of 2026-07-16, horizon 2028-12-31)")
    print("=" * 100)
    plan = build_ukraine_plan(n_particles)
    lineage = assemble(plan, _fake_llm)
    cons = lineage["mode_graph"]
    et = lineage["event_time"]
    print(f"\n-- MODE GRAPH (canonical, {cons['n_sources']} sources, "
          f"agreement {cons['agreement']}) --")
    for m in plan._canonical_modes:
        hrm = et["hazard_ratio_by_mode"].get(m["id"], {})
        ds = m.get("decision_structure") or {}
        print(f"  {m['id']:<22} pathway={m.get('pathway', mode_pathway(m)):<24}"
              f" rule={ds.get('rule', '(default)'):<10} prior={m['prior']:.2f}"
              f" support={m.get('support')}/3+"
              f"  HR med={hrm.get('median')} [{hrm.get('lo80')}, {hrm.get('hi80')}]"
              f" binder={hrm.get('binding_actor')} (split={hrm.get('endogenous_split')})")
    print(f"  rejected (not absorbing): {et['rejected_non_absorbing_modes']}")
    print(f"  dropped by consensus vote: "
          f"{[d['id'] for d in cons['dropped_minority_candidates']]}")
    print(f"\n-- PATHWAY PROCESSES (grounded) --")
    for pw, rec in lineage["pathway_processes"]["declared"].items():
        print(f"  {rec['var']:<44} init={rec['initial']}  state={rec['state']}  basis={rec['basis']}")
    print(f"\n-- GROUNDED STANCES ({lineage['actor_intentions']['n_grounded']} stances, "
          f"{lineage['actor_intentions']['n_mode_scoped']} mode-scoped) --")
    for s in plan._intention_stances:
        print(f"  {s['actor']:<26} {s['commitment_level']:<22} pathway={s['pathway']:<24}"
              f" target={str(s.get('target_mode')):<18} control={s.get('control', s.get('controls_pathway')):<22}"
              f" rel={s['reliability']} cap={s['capability']}")
        print(f"      “{s['quote']}”")
    td = lineage["trajectory_depth"]
    n_dec = sum(1 for e in plan.scheduled_events if e["etype"] == "decision_opportunity")
    n_hz = sum(1 for e in plan.scheduled_events if e["etype"] == "hazard_round")
    print(f"\n-- SCHEDULE --  horizon {td['horizon_days']}d, cadence {td['cadence_days']}d, "
          f"{td['strategic_actors']} strategic actors × {td['decision_points']} decision points = "
          f"{n_dec} decision events; {n_hz} hazard rounds "
          f"({et['rounds_per_mode']}/mode × {len(et['modes'])} modes); "
          f"n_particles={plan.compute_plan['n_particles']}")
    sample = next(e for e in plan.scheduled_events if e["etype"] == "decision_opportunity")
    print(f"  candidate actions per review: "
          f"{[a['type'] for a in sample['payload']['actions']]}")

    print("\n-- RUNNING the production rollout (build_world → phase-4 policies → pathway writes → "
          "hazard rounds → absorption monitor) --")
    result, branches = run_from_plan(plan, llm=None, seed=seed)
    ev = result["event_time"]

    # ---------------- exact per-particle logs
    absorbed_idx = [i for i, b in enumerate(branches)
                    if plan.outcome_contract.readout(b.world) is not None]
    censored_idx = [i for i in range(len(branches)) if i not in absorbed_idx]
    picks = (absorbed_idx[:2] + censored_idx[:1]) or [0]
    for i in picks:
        b = branches[i]
        t_abs = plan.outcome_contract.readout(b.world)
        mode = b.world.quantities.get("absorbed_by")
        tag = (f"ABSORBED {_d(t_abs)} via {getattr(mode, 'value', None)}" if t_abs
               else "CENSORED beyond horizon")
        print(f"\n-- PARTICLE {i} — {tag} --")
        for row in _extract_story(b, plan):
            print("  " + row)

    # ---------------- aggregates
    actions_by_actor = defaultdict(Counter)
    prog_end = defaultdict(list)
    for b in branches:
        for d in b.log:
            if d.event_type == "actor_action":
                for c in d.changes:
                    if c["path"].endswith(".current_action") and isinstance(c["after"], dict):
                        actions_by_actor[c["path"].split(".")[0]][c["after"].get("action_name")] += 1
        for pw in ("cooperative_agreement", "unilateral_action"):
            q = b.world.quantities.get(progress_var(pw))
            if q is not None and isinstance(q.value, (int, float)):
                prog_end[pw].append(float(q.value))
    print("\n-- WHO DID WHAT (chosen actions across all particles) --")
    for actor, ctr in sorted(actions_by_actor.items()):
        top = ", ".join(f"{a}×{n}" for a, n in ctr.most_common(6))
        print(f"  {actor:<28} {top}")
    print("\n-- PATHWAY PROCESSES AT HORIZON (mean over particles; started at init above) --")
    for pw, vals in prog_end.items():
        print(f"  {progress_var(pw):<44} mean_end={sum(vals) / len(vals):.3f} "
              f"min={min(vals):.3f} max={max(vals):.3f}")

    # ---------------- terminal readout
    print("\n-- TERMINAL READOUT (pure first-passage readout; nothing drew an outcome) --")
    print(f"  P(conflict ends by 2028-12-31) = {result['distribution']['absorbed_by_horizon']:.3f}"
          f"   censored beyond horizon = {result['distribution']['censored_beyond_horizon']:.3f}")
    print("  CDF:", "  ".join(f"{_d(g)}:{c:.3f}" for g, c in
                              zip(ev["cdf_grid_ts"][1::2], ev["cdf"][1::2])))
    print("  first-passage quantiles:", {k: (_d(v) if v else "beyond_horizon")
                                         for k, v in ev["first_passage_quantiles_ts"].items()})
    print("  mode × absorption:", ev["mode_distribution"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    return {"question": plan.question, "lineage_mode_graph": cons, "event_time_report": et,
            "stances": plan._intention_stances,
            "pathway_processes": lineage["pathway_processes"],
            "result_distribution": result["distribution"], "event_time_readout": ev,
            "actions_by_actor": {a: dict(c) for a, c in actions_by_actor.items()},
            "pathway_progress_at_horizon": {k: round(sum(v) / len(v), 4)
                                            for k, v in prog_end.items()},
            "n_deltas": result["n_deltas"]}


# ================================================================ cross-domain fixtures (universality)
def _mini(question, criterion, modes, stances, process_states, entities, institutions,
          inst_event=None, extra_quantities=None, n_particles=80, horizon_days=200, seed=0):
    plan = Plan(question=question, as_of=parse_time("2026-07-16"),
                horizon_ts=parse_time("2026-07-16") + horizon_days * 86400.0,
                outcome_contract=SimpleNamespace(options=[]),
                entities=entities, populations=[], institutions=institutions, relations=[],
                quantities=list(extra_quantities or []), latents=[], scheduled_events=[],
                stochastic_hazards=[],
                accepted_mechanisms=[{"mech_id": "actor_policy", "operator": "production_actor_policy",
                                      "ontology_type": "decision", "causal_role": "actors",
                                      "parameter_source": "tier7", "temporal_scale": "event",
                                      "calibration_status": "broad_prior", "sensitivity": 0.8}]
                if entities else [],
                candidate_experimental_mechanisms=[], rejected_mechanisms=[],
                structural_hypotheses=[], actor_decisions=[], mechanism_choices=[],
                fallbacks_used=[], support_grade="exploratory", interpretations=[], omissions=[],
                degraded=False, fidelity_plan={}, uncertainty_plan={},
                compute_plan={"n_particles": n_particles}, provenance={}, version=1, parent_version=0)
    if inst_event:
        plan.scheduled_events.append(inst_event)
    plan._canonical_modes, plan._mode_consensus = modes, {"n_sources": 1, "agreement": 1.0,
                                                          "pinned": True}
    lineage = {}
    declare_pathway_processes(plan, modes, grounding=process_states)
    plan._intention_stances = stances
    for e in entities:                                          # write stances onto entities (grounded)
        recs = [s for s in stances if s["actor"] == e["id"]]
        if recs:
            e.setdefault("fields", {})["stances"] = recs
    if entities:
        legacy_periodic_review_ablation(plan, {"phase4_actor_policy": {"required": True}},
                                        cadence_days=14.0, acknowledge=ABLATION_TOKEN)
    rep = convert_to_event_time(plan, criterion, lineage=lineage, llm=None)
    plan.compute_plan["n_particles"] = n_particles              # keep fixtures small (floor is 200)
    result, branches = run_from_plan(plan, llm=None, seed=seed)
    return plan, rep, result


def run_cross_domain():
    print("\n" + "=" * 100)
    print("CROSS-DOMAIN FIXTURES — same engine, different decision structures / pathways")
    print("=" * 100)
    out = {}
    # ---- 1. BILL (institutional_procedure, MAJORITY — the most-opposed senator must NOT bind) ----
    stances = ([{"actor": f"Senator_{i}", "commitment_level": "actively_pursuing",
                 "reliability": "high", "capability": "high",
                 "pathway": "institutional_procedure", "control": "coalition_member"}
                for i in range(3)]
               + [{"actor": "Senator_Blocker", "commitment_level": "committed_to_prevent",
                   "reliability": "high", "capability": "high",
                   "pathway": "institutional_procedure", "control": "coalition_member"}])
    modes = [{"id": "bill_passes_senate", "prior": 1.0, "pathway": "institutional_procedure",
              "decision_structure": {"rule": "majority", "approvers": []}}]
    inst_event = {"etype": "institutional_decision",
                  "ts": parse_time("2026-07-16") + 90 * 86400.0, "participants": [],
                  "payload": {"institution_id": "Senate", "n_members": 100, "needed": 60,
                              "outcome_var": "outcome", "options": ["passes", "fails"],
                              "lean": "neutral"}}
    plan, rep, result = _mini(
        "When will the Senate pass the AI Safety Act?",
        {"resolves_yes_iff": "the Senate passes the AI Safety Act"},
        modes, stances,
        {"institutional_procedure": {"state": "active", "value": 0.5, "basis": "floor time set"}},
        [{"id": f"Senator_{i}", "type": "person", "sensitivity": 0.7, "fields": {}} for i in range(3)]
        + [{"id": "Senator_Blocker", "type": "person", "sensitivity": 0.75, "fields": {}},
           {"id": "Senate", "type": "institution", "sensitivity": 0.9, "fields": {}}],
        [{"id": "Senate", "rules": [{"kind": "quorum", "params": {"total": 100, "needed": 60,
                                                                  "members": []}}]}],
        inst_event=inst_event, n_particles=80, horizon_days=200)
    hr = rep["hazard_ratio_by_mode"]["bill_passes_senate"]
    print(f"\n[BILL] rule={hr['combination_rule']} HR med={hr['median']} binder={hr['binding_actor']}")
    print(f"       3 pursuing vs 1 blocking under MAJORITY → combined HR {hr['median']} > 1 "
          f"(most-opposed does NOT bind); absorbing institutional votes: "
          f"{rep['n_absorbing_institutional_decisions']}")
    print(f"       P(passes by horizon) = {result['distribution']['absorbed_by_horizon']:.3f}; "
          f"modes: {result['event_time']['mode_distribution']}")
    out["bill"] = {"hr": hr, "distribution": result["distribution"],
                   "mode_distribution": result["event_time"]["mode_distribution"]}
    assert hr["combination_rule"] == "majority" and hr["median"] > 1.0
    assert rep["n_absorbing_institutional_decisions"] == 1

    # ---- 2. PRODUCT LAUNCH (operational_execution, HIERARCHY — the CEO outweighs a resister) ----
    stances = [{"actor": "CEO", "commitment_level": "formally_committed", "reliability": "high",
                "capability": "high", "pathway": "operational_execution", "control": "sole_authority"},
               {"actor": "Lead_Engineer", "commitment_level": "conditionally_opposed",
                "reliability": "medium", "capability": "medium",
                "pathway": "operational_execution", "control": "operational_capability"}]
    modes = [{"id": "flagship_launch", "prior": 1.0, "pathway": "operational_execution",
              "decision_structure": {"rule": "hierarchy"}}]
    plan, rep, result = _mini(
        "When will Acme ship its flagship product?",
        {"resolves_yes_iff": "Acme's flagship product is generally available"},
        modes, stances,
        {"operational_execution": {"state": "advanced", "value": 0.7, "basis": "beta complete"}},
        [{"id": "CEO", "type": "person", "sensitivity": 0.9, "fields": {}},
         {"id": "Lead_Engineer", "type": "person", "sensitivity": 0.6, "fields": {}}],
        [], n_particles=80, horizon_days=200)
    hr = rep["hazard_ratio_by_mode"]["flagship_launch"]
    print(f"\n[LAUNCH] rule={hr['combination_rule']} HR med={hr['median']} binder={hr['binding_actor']}")
    print(f"         committed CEO under HIERARCHY beats one resister → HR > 1; advanced process "
          f"(0.7) amplifies hazards endogenously")
    print(f"         P(ships by horizon) = {result['distribution']['absorbed_by_horizon']:.3f}")
    out["launch"] = {"hr": hr, "distribution": result["distribution"]}
    assert hr["combination_rule"] == "hierarchy" and hr["median"] > 1.0

    # ---- 3. INFLATION THRESHOLD (world-driven — stances are NOT the causal variable) ----
    stances = [{"actor": "Fed_Chair", "commitment_level": "committed_to_prevent",
                "reliability": "high", "capability": "high",
                "pathway": "threshold_crossing", "control": "informal_influence"}]
    modes = [{"id": "inflation_below_2pct", "prior": 1.0, "pathway": "threshold_crossing"}]
    plan, rep, result = _mini(
        "When will US inflation fall below 2 percent?",
        {"resolves_yes_iff": "12-month CPI inflation prints below 2.0%"},
        modes, stances,
        {"threshold_crossing": {"state": "active", "value": 0.5, "basis": "disinflation trend"}},
        [], [],                                               # NO strategic actors in the loop
        extra_quantities=[{"name": "nonlinear_state", "qtype": "nonlinear_state",
                           "value": 0.62, "sd": 0.1}],
        n_particles=80, horizon_days=200)
    hr = rep["hazard_ratio_by_mode"]["inflation_below_2pct"]
    rounds = [e for e in plan.scheduled_events if e["etype"] == "hazard_round"]
    consumed = {c["var"] for e in rounds for c in e["payload"]["consume"]}
    print(f"\n[INFLATION] rule={hr['combination_rule']} HR med={hr['median']} "
          f"binder={hr['binding_level']}")
    print(f"            world-driven pathway: a Fed stance shrinks to ~no-effect under "
          f"AGGREGATION; hazards consume {sorted(consumed)}")
    print(f"            P(below 2% by horizon) = {result['distribution']['absorbed_by_horizon']:.3f}")
    out["inflation"] = {"hr": hr, "consumed": sorted(consumed),
                        "distribution": result["distribution"]}
    assert 0.85 <= hr["median"] <= 1.1                        # stances near-irrelevant, by design
    assert "nonlinear_state" in consumed                      # non-actor mechanism drives the clock
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ukraine = run_ukraine(args.particles, seed=args.seed)
    cross = run_cross_domain()
    OUT.write_text(json.dumps({"ukraine": ukraine, "cross_domain": cross}, indent=1, default=str))
    print(f"\nartifact → {OUT}")


if __name__ == "__main__":
    main()
