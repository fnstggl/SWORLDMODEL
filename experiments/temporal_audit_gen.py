"""Machine-generated audit of synthetic temporal assumptions in World Model V2 (§1 of the
event-driven temporal architecture task).

Every entry in ASSUMPTIONS was located by tracing the complete production timing path
(unified_runtime → fidelity/event_time conversion → materialize → rollout → generated_world /
semantic_consequences / phase4 / phase13 / individual_reaction). For each assumption the
registry records what the code did, whether production reached it, what real-world process it
approximated, and the required replacement. The script then GREPS THE LIVE TREE: an assumption
whose `disposition` is `removed_from_production` must no longer match in its production file —
so this audit is simultaneously the §1 inventory and the removal proof consumed by the
acceptance report. Run:

    PYTHONPATH=. python experiments/temporal_audit_gen.py [--out artifacts/temporal/audit.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# disposition vocabulary:
#   removed_from_production       — the pattern must be ABSENT from the production file
#   quarantined_legacy_ablation   — moved behind an explicit legacy/ablation door; the pattern
#                                   may exist only in the quarantine site
#   replaced_scenario_generated   — the mechanism now consumes the scenario temporal model; the
#                                   old fixed value must be absent
#   safety_limit_with_truncation  — allowed to remain ONLY as a service-protection limit whose
#                                   exhaustion produces simulation_status=temporally_truncated
#   exact_fact_retained           — correct behavior (real dates) retained on purpose
ASSUMPTIONS = [
    dict(id="periodic_strategic_review", file="swm/world_model_v2/fidelity.py",
         function="deepen_trajectory",
         pattern=r"periodic strategic review",
         current="2–14 evenly spaced 'periodic strategic review' decision events for actors[:6] "
                 "at cadence horizon/6 (critic override possible)",
         production_reachability="unified_runtime.simulate_world → deepen_trajectory (was lines 264–266)",
         approximates="recurring strategic decision opportunities of the process's real actors",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="scenario temporal model + DecisionTriggers: actors decide only when a real "
                     "trigger occurs (temporal_compiler + temporal_runtime)",
         disposition="quarantined_legacy_ablation",
         quarantine_file="swm/world_model_v2/legacy_ablations.py"),
    dict(id="six_actor_cap", file="swm/world_model_v2/fidelity.py", function="deepen_trajectory",
         pattern=r"actors\[:6\]",
         current="strategic = actors[:6] (top sensitivity)",
         production_reachability="same as periodic_strategic_review",
         approximates="'every real decision-holder' — truncated to six by rank",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="causal relevance / frontier discovery / authority — no fixed actor cap; "
                     "compute limits mark branches temporally_truncated instead",
         disposition="quarantined_legacy_ablation",
         quarantine_file="swm/world_model_v2/legacy_ablations.py"),
    dict(id="default_cadence_horizon_over_6", file="swm/world_model_v2/fidelity.py",
         function="deepen_trajectory", pattern=r"horizon_days\s*/\s*6\.0",
         current="cad = cadence_days or max(1.0, horizon_days/6.0); n_points clamp(2,14)",
         production_reachability="same as periodic_strategic_review",
         approximates="'typical days between meaningful decisions' fallback",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="no generic cadence anywhere; timing from generated triggers/processes",
         disposition="quarantined_legacy_ablation",
         quarantine_file="swm/world_model_v2/legacy_ablations.py"),
    dict(id="deepen_trajectory_call", file="swm/world_model_v2/unified_runtime.py",
         function="simulate_world", pattern=r"deepen_trajectory",
         current="unified_runtime called fidelity.deepen_trajectory on every run",
         production_reachability="default path",
         approximates="trajectory depth",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="compile_temporal_model + attach_temporal_events (LLM temporal compilation)",
         disposition="removed_from_production"),
    dict(id="hazard_round_grid_when", file="swm/world_model_v2/event_time.py",
         function="convert_to_event_time",
         pattern=r"for k in range\(1,\s*n_rounds \+ 1\)",
         current="n_rounds=clamp(6,40,horizon/21d) hazard_round events on an evenly spaced grid "
                 "per mode; stance_review events 1s before each",
         production_reachability="every when/categorical question",
         approximates="continuous resolution hazard of each outcome mode",
         evidence_grounded=False, fitted="curve shape only", scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="continuous-time first-passage scheduling (temporal_hazards): per-branch "
                     "Exp(1) threshold, cumulative intensity, state-preserving re-projection",
         disposition="removed_from_production"),
    dict(id="mode_cap_six", file="swm/world_model_v2/event_time.py",
         function="convert_to_event_time", pattern=r"for m in modes\[:6\]",
         current="only the first six modes got hazard chains",
         production_reachability="every when/categorical question",
         approximates="the question's real outcome modes",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="all canonical modes carry first-passage processes",
         disposition="removed_from_production"),
    dict(id="binary_residual_grid", file="swm/world_model_v2/event_time.py",
         function="convert_binary_to_event_time",
         pattern=r"round_ts = \[plan\.as_of \+ \(k / \(n_rounds \+ 1\)\) \* span",
         current="residual chain on evenly spaced grid at cadence horizon/10 (clamp 4..20)",
         production_reachability="every binary deadline question",
         approximates="residual resolution hazard",
         evidence_grounded=False, fitted="curve shape only", scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="one first-passage process with calibrated cumulative intensity "
                     "(-ln(1-target)), family-curve shaped",
         disposition="removed_from_production"),
    dict(id="stance_review_grid", file="swm/world_model_v2/event_time.py",
         function="convert_to_event_time", pattern=r'"etype":\s*"stance_review"',
         current="stance_review events scheduled on the hazard-round grid",
         production_reachability="every when/categorical question with stances",
         approximates="actors reconsidering their stances when something changes",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="event-driven stance updating: material state changes emit "
                     "stance_relevant_change events; elapsed-time hysteresis",
         disposition="removed_from_production"),
    dict(id="invocations_per_actor_5", file="swm/world_model_v2/generated_world.py",
         function="DEFAULT_BUDGETS", pattern=r'"max_invocations_per_actor":\s*5\b',
         current="hard cap: 5 LLM invocations per actor, exhaustion silently returns None",
         production_reachability="generated_actor_mediated_world (production default)",
         approximates="an actor's real number of decision situations",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="causal quiescence stopping; safety budget exhaustion → "
                     "simulation_status=temporally_truncated with actor/pending records",
         disposition="safety_limit_with_truncation"),
    dict(id="cascade_depth_8", file="swm/world_model_v2/generated_world.py",
         function="DEFAULT_BUDGETS", pattern=r'"max_cascade_depth":\s*8\b',
         current="hard cap: 8 cascade levels",
         production_reachability="generated mode",
         approximates="real cascade termination (quiescence)",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="quiescence; safety budget with truncation records",
         disposition="safety_limit_with_truncation"),
    dict(id="frontier_cap_8", file="swm/world_model_v2/generated_world.py",
         function="discover_causal_frontier", pattern=r"return out\[:8\]",
         current="causal frontier truncated to 8 actors",
         production_reachability="generated mode",
         approximates="every actor whose decision can materially change the trajectory",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="uncapped discovery; safety truncation recorded when hit",
         disposition="removed_from_production"),
    dict(id="direct_delivery_60s", file="swm/world_model_v2/generated_world.py",
         function="route_semantic_event", pattern=r'"default_delay_s",\s*60\.0',
         current="private observation delivered clock+60s",
         production_reachability="generated mode, every private observation",
         approximates="message transmission + delivery through the actual channel",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="ChannelTemporalModel stages (transmission/delivery/availability) from the "
                     "scenario temporal model; attention separated from delivery",
         disposition="replaced_scenario_generated"),
    dict(id="public_delivery_3600s", file="swm/world_model_v2/generated_world.py",
         function="route_semantic_event", pattern=r'"public_delay_s",\s*3600\.0',
         current="public observation delivered clock+1h to every recipient",
         production_reachability="generated mode, every public event",
         approximates="publication + gradual exposure spread",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="publication immediate at channel; per-actor exposure/attention from "
                     "profiles (exposure spreads, key actors can see earlier/later)",
         disposition="replaced_scenario_generated"),
    dict(id="reconsideration_1800s", file="swm/world_model_v2/generated_world.py",
         function="_invocation_event", pattern=r"delay_s:\s*float\s*=\s*1800\.0",
         current="actor invoked clock+30min after any triggering observation",
         production_reachability="generated mode, every actor invocation",
         approximates="noticing + interpreting + deciding-to-reconsider",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="attention events from ActorTemporalProfile (channel checking, urgency "
                     "interrupts, sleep/work windows) + DecisionTrigger provenance",
         disposition="replaced_scenario_generated"),
    dict(id="semcons_delivery_60s", file="swm/world_model_v2/semantic_consequences.py",
         function="_x_deliver_information", pattern=r'"delivery_delay_s",\s*60\.0',
         current="message delivery default 60s",
         production_reachability="fixed-v1 mode + generated-mode shared expander",
         approximates="channel transmission/delivery",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="channel timing resolved through the temporal model (resolve_timing)",
         disposition="replaced_scenario_generated"),
    dict(id="semcons_post_delivery_1800s", file="swm/world_model_v2/semantic_consequences.py",
         function="CommunicationDeliveryOperator", pattern=r"\+ 1800\.0",
         current="decision_opportunity scheduled 30min after message delivery",
         production_reachability="both consequence modes",
         approximates="recipient noticing and engaging with the message",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="attention event (delivery ≠ attention); actor sees the full available bundle",
         disposition="replaced_scenario_generated"),
    dict(id="semcons_reply_60s_6h", file="swm/world_model_v2/semantic_consequences.py",
         function="qualitative reply mapping", pattern=r'60\.0 if name == "reply_now" else 6 \* 3600\.0',
         current="reply_now→60s, reply_later→6h fixed mapping",
         production_reachability="both consequence modes",
         approximates="the actor's own stated response timing",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="actor temporal intent compiled through TimingSpec/calendar (§11)",
         disposition="replaced_scenario_generated"),
    dict(id="phase4_reaction_60s", file="swm/world_model_v2/phase4_execution.py",
         function="reaction scheduling", pattern=r'"reaction_delay_s",\s*60\.0',
         current="actor_reaction scheduled clock+60s",
         production_reachability="phase4 execution of communicate actions",
         approximates="target's reaction latency",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="delivery through channel + recipient attention",
         disposition="replaced_scenario_generated"),
    dict(id="background_daily_tick", file="swm/world_model_v2/rollout.py",
         function="RolloutEngine", pattern=r"background_every_days:\s*float\s*=\s*1\.0",
         current="background dynamics applied on a >=1-day threshold tick",
         production_reachability="every rollout",
         approximates="continuous processes (memory decay, attention drift)",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="advance_interval(world, start_ts, end_ts, ctx, rng): declared continuous "
                     "processes update over the exact elapsed interval between events",
         disposition="removed_from_production"),
    dict(id="crn_background_daily_tick", file="swm/world_model_v2/phase13/crn.py",
         function="MatchedRolloutEngine", pattern=r"background_every_days \* 86400\.0",
         current="same daily tick in the matched engine",
         production_reachability="phase13 counterfactuals",
         approximates="continuous processes",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="advance_interval in the matched engine (same as rollout)",
         disposition="removed_from_production"),
    dict(id="interventions_plus_1s", file="swm/world_model_v2/phase13/interventions.py",
         function="DecisionActionOperator", pattern=r"world\.clock\.now \+ 1\.0",
         current="every action's follow-up (message_delivered etc.) at clock+1s",
         production_reachability="phase13 legacy_fixed_v1 evaluations",
         approximates="transmission through the action's channel",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="channel-resolved delivery timing (temporal model aware)",
         disposition="replaced_scenario_generated"),
    dict(id="api_default_decision_point_1s", file="swm/world_model_v2/phase13/api.py",
         function="optimize_policy", pattern=r"clock\.now\) \+ 1\.0\]",
         current="missing decision points default to one synthetic point at start+1s",
         production_reachability="phase13 policy optimization",
         approximates="the policy's real decision opportunity",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="decision points anchored to real triggers (intervention initiation, "
                     "scenario anchors); synthetic point retained only as the policy's own "
                     "explicitly-labeled initiation event",
         disposition="replaced_scenario_generated"),
    dict(id="step_recheck_6h", file="swm/world_model_v2/phase13/scenario_actions/execution.py",
         function="_RECHECK_S", pattern=r"_RECHECK_S\s*=\s*21600\.0",
         current="conditional plan steps re-polled every 6h",
         production_reachability="generated-mode plan execution",
         approximates="the guard condition's underlying state changing",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="condition-watch triggers: the step wakes when a watched field changes "
                     "(state-change subscription), with a bounded fallback recheck labeled a "
                     "safety net",
         disposition="replaced_scenario_generated"),
    dict(id="step_gap_60s", file="swm/world_model_v2/phase13/scenario_actions/execution.py",
         function="_STEP_GAP_S", pattern=r"_STEP_GAP_S\s*=\s*60\.0",
         current="dependent steps fire parent-completion+60s",
         production_reachability="generated-mode plan execution",
         approximates="the dependent step's real earliest start",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="dependency triggers: the step fires when its parent's consequence lands",
         disposition="replaced_scenario_generated"),
    dict(id="midpoint_timing_variant", file="swm/world_model_v2/phase13/affordances.py",
         function="generate_affordances", pattern=r"0\.5 \* \(hz - now\)",
         current="'now vs mid-horizon' timing variants",
         production_reachability="phase13 legacy_fixed_v1",
         approximates="meaningful timing alternatives",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="timing variants anchored to real scenario anchors (deadlines, scheduled "
                     "events, availability windows) from the temporal model",
         disposition="replaced_scenario_generated"),
    dict(id="individual_history_1day_grid", file="swm/world_model_v2/individual_reaction.py",
         function="simulate_individual_reaction", pattern=r"86400\.0 \* \(len\(",
         current="prior interactions back-dated exactly 1 day apart",
         production_reachability="individual-reaction route",
         approximates="the relationship's real interaction timeline",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="real timestamps from supplied history (context['history'][i]['ts']); "
                     "unknown times labeled unresolved, not gridded",
         disposition="replaced_scenario_generated"),
    dict(id="attrition_per_review", file="swm/world_model_v2/world_dynamics.py",
         function="_contested_attrition", pattern=r'sampled_coupling\(world, "attrition_per_review"\)',
         current="capacity drained a fixed coupling per REVIEW ROUND (count-based)",
         production_reachability="every stance-carrying rollout",
         approximates="attrition over real elapsed time while a contested process is active",
         evidence_grounded=False, fitted="fittable coupling", scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="attrition_rate_per_day continuous process integrated over the exact "
                     "elapsed interval (advance_interval)",
         disposition="removed_from_production"),
    dict(id="stance_review_count_cooldown", file="swm/world_model_v2/world_dynamics.py",
         function="StanceReviewOperator", pattern=r"STANCE_REVIEW_COOLDOWN\s*=\s*1\b",
         current="stances may not change in consecutive REVIEWS (count hysteresis)",
         production_reachability="every stance-carrying rollout",
         approximates="real elapsed-time hysteresis / material-change thresholds",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=False, affects_terminal=True,
         replacement="elapsed-time hysteresis (min real seconds between changes) + "
                     "material-change trigger thresholds",
         disposition="removed_from_production"),
    dict(id="seq_insertion_tiebreak", file="swm/world_model_v2/events.py",
         function="EventQueue.schedule", pattern=r"self\._seq \+= 1",
         current="same-timestamp events resolved by monotonic insertion order",
         production_reachability="every rollout",
         approximates="causal dependency ordering / genuine simultaneity",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="same-time batch architecture: pop the full timestamp batch, causal-"
                     "dependency microsteps, canonical content order for independents, explicit "
                     "write-conflict handling (seq retained ONLY as heap stability inside the "
                     "batch pop, never as semantic order)",
         disposition="safety_limit_with_truncation"),
    dict(id="max_events_500", file="swm/world_model_v2/rollout.py",
         function="run_branch", pattern=r"max_events:\s*int\s*=\s*500",
         current="per-branch cap of 500 events; silent break when hit",
         production_reachability="every rollout",
         approximates="nothing — service protection",
         evidence_grounded=False, fitted=False, scenario_generated=False,
         affects_actor_calls=True, affects_terminal=True,
         replacement="kept as safety limit, but exhaustion now records "
                     "simulation_status=temporally_truncated with pending events",
         disposition="safety_limit_with_truncation"),
    dict(id="scheduled_facts_exact", file="swm/world_model_v2/scheduled_facts.py",
         function="attach_scheduled_facts", pattern=r'"etype": "scheduled_fact", "ts": f\["ts"\]',
         current="dated public facts execute deterministically at their real dates",
         production_reachability="every run with an LLM",
         approximates="the real public calendar — correctly",
         evidence_grounded=True, fitted=False, scenario_generated=True,
         affects_actor_calls=False, affects_terminal=True,
         replacement="none — this is the correct pattern the rest of the system now follows",
         disposition="exact_fact_retained"),
]


def _grep(path: Path, pattern: str) -> list:
    if not path.exists():
        return []
    out = []
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if rx.search(line):
            out.append({"line": i, "text": line.strip()[:160]})
    return out


def generate_audit() -> dict:
    rows, failures = [], []
    for a in ASSUMPTIONS:
        prod_hits = _grep(ROOT / a["file"], a["pattern"])
        row = dict(a)
        row["production_matches"] = prod_hits
        if a["disposition"] in ("removed_from_production", "replaced_scenario_generated"):
            row["removal_verified"] = not prod_hits
            if prod_hits:
                failures.append({"id": a["id"], "why": "pattern still present in production file",
                                 "matches": prod_hits[:3]})
        elif a["disposition"] == "quarantined_legacy_ablation":
            q = _grep(ROOT / a.get("quarantine_file", ""), a["pattern"]) \
                if a.get("quarantine_file") else []
            row["quarantine_matches"] = q
            row["removal_verified"] = not prod_hits
            if prod_hits:
                failures.append({"id": a["id"], "why": "pattern still in production file "
                                                       "(should live only in quarantine)",
                                 "matches": prod_hits[:3]})
        else:
            row["removal_verified"] = None                    # retained on purpose
        rows.append(row)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": commit, "n_assumptions": len(rows),
            "n_requiring_removal": sum(1 for a in ASSUMPTIONS
                                       if a["disposition"] in ("removed_from_production",
                                                               "replaced_scenario_generated",
                                                               "quarantined_legacy_ablation")),
            "n_removal_verified": sum(1 for r in rows if r.get("removal_verified") is True),
            "removal_failures": failures, "assumptions": rows,
            "all_removals_verified": not failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/temporal/temporal_assumption_audit.json")
    args = ap.parse_args()
    audit = generate_audit()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1))
    print(f"audit: {audit['n_assumptions']} assumptions, "
          f"{audit['n_removal_verified']}/{audit['n_requiring_removal']} removals verified, "
          f"failures={len(audit['removal_failures'])} → {out}")
    return 0 if audit["all_removals_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
