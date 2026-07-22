"""Single-individual reaction simulation — the same qualitative actor architecture, focused on
one person, THROUGH THE TEMPORAL MODEL (§25).

Route for questions like "how will this person react if I send this tonight?". The target
individual is AUTOMATICALLY Tier 1 (reaction_is_the_question). The stimulus is not teleported
into their head: it is SENT at the real send time, travels the channel (delivery), waits for
the person's real attention (their timezone, sleep/active windows, channel-checking habits and
sampled availability — user-supplied context strongly informs all of these), and only a NOTICED
stimulus produces a decision. Per-sample temporal outcomes are first-class and distinguishable:

    responded            — noticed and chose an observable response (with response time)
    read_but_deferred    — noticed, deliberately chose to wait / revisit later
    unread_by_horizon    — never noticed within the question's window (NOT "ignored")

History items may carry REAL timestamps ({"text": ..., "ts": ...} or (text, ts)); bare strings
keep relative order with the spacing recorded as an UNRESOLVED timing assumption — never a
fabricated 1-day grid presented as real.

STRICT SAMPLE INTEGRITY (§20/§33): a sample whose LLM decision fails after the §19.1 retry
ladder is recorded TRUNCATED (first-class §20 status, normally `truncated_provider_failure`) —
never excluded-and-continued with a numeric substitute, never silently resampled. The reported
distribution counts ONLY completed samples, carries an explicit note, and rides with §21
truncation bounds (the truncated share can swing any option). The recipient runs through the
SAME §9-§15 bounded-cognition stages as every other qualitative actor: the stimulus enters as
the availability set (obs_id 'stimulus', the question's channel, FULL exact text — §32), and
the decision call sees only the surviving material. Each sample's outcome additionally maps
onto the distinguishable §33 nonresponse states (bounded_cognition.NONRESPONSE_STATES) where
applicable, reported as `nonresponse_breakdown`."""
from __future__ import annotations

import time as _time

from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.qualitative_actor import (
    ActionClusterer, ActorDecisionUnavailable, ActorPolicyCalibrator,
    QualitativeActorPolicyRuntime, QualitativeConfig, QualitativeDecisionEngine,
    aggregate_actor_decisions,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.temporal_model import (ActorTemporalProfile, ScenarioTemporalModel,
                                               ChannelTemporalModel)

#: default observable-response menu: the messaging slice of the shared action ontology
DEFAULT_RESPONSE_ACTIONS = ("reply_now", "reply_later", "acknowledge", "clarify", "ignore")

#: §20 statuses for a per-sample truncation on this route: everything on the decision path is a
#: provider/parse/cognition failure except an exhausted actor budget (its own first-class kind)
_REASON_TO_BRANCH_STATUS = {
    "actor_llm_budget_exhausted": "truncated_actor_budget",
    "invocation_safety_budget_reached": "truncated_actor_budget",
    "llm_budget_exhausted": "truncated_actor_budget",
}


def _truncation_status(reason: str, detail: str = "") -> str:
    """Map a decision-failure reason onto the first-class §20 branch status for one SAMPLE.
    Budget exhaustion is `truncated_actor_budget`; every other failure on this route (provider
    down, unparseable after retries, no backend, cognition stage failure, refused numeric
    fallback) is `truncated_provider_failure` — the decision could not be produced."""
    if reason in _REASON_TO_BRANCH_STATUS:
        return _REASON_TO_BRANCH_STATUS[reason]
    if "budget" in f"{reason} {detail}".lower():
        return "truncated_actor_budget"
    return "truncated_provider_failure"


def _nonresponse_state(*, temporal_state: str = "", cognition: dict | None = None,
                       act_or_wait: str = "", observable_response: str = "",
                       blocked: bool = False) -> str:
    """Map one COMPLETED sample's outcome onto the §33 nonresponse vocabulary
    (bounded_cognition.NONRESPONSE_STATES) where applicable; '' means an actual response
    happened (not a nonresponse). Precedence mirrors the causal chain: an outside-mechanism
    block dominates (they tried), then never-reached attention ('unread' — on this route the
    temporal attention model owns noticing, so `unread_by_horizon` IS attention missing the
    stimulus; a bundle-level attention miss maps the same way), then working-memory
    displacement ('noticed_but_deprioritized'), then the decision itself (defer vs explicit
    no-reply)."""
    if blocked:
        return "response_blocked_by_outside_circumstances"
    if temporal_state == "unread_by_horizon":
        return "unread"
    cog = cognition or {}
    missed = {str((m or {}).get("obs_id", "")) for m in (cog.get("observations_missed") or [])
              if isinstance(m, dict)}
    if "stimulus" in missed:
        return "unread"                       # attention never registered the stimulus
    noticed = {str(n) for n in (cog.get("observations_noticed") or [])}
    active_sources = cog.get("working_memory_active_sources")
    if "stimulus" in noticed and isinstance(active_sources, list) \
            and "stimulus" not in {str(s) for s in active_sources}:
        return "noticed_but_deprioritized"    # working memory displaced it before the choice
    act = str(act_or_wait or "").lower()
    resp = str(observable_response or "").lower()
    if resp == "ignore" or act == "do_nothing":
        return "no_response_chosen"           # an explicit, deliberate no-reply
    if act in ("wait", "gather_information", "delegate") or resp in ("wait", "reply_later"):
        return "considered_but_deferred"
    return ""


def _history_entries(context: dict) -> tuple:
    """Normalize history to [(text, ts_or_None)] preserving order; count how many carried
    real timestamps (provenance honesty)."""
    out, n_real = [], 0
    for item in (context.get("history") or [])[:12]:
        if isinstance(item, dict):
            txt = str(item.get("text", item.get("content", "")))[:400]
            ts = item.get("ts", item.get("at"))
            if isinstance(ts, str):
                try:
                    from swm.world_model_v2.state import parse_time
                    ts = parse_time(ts)
                except (ValueError, TypeError):
                    ts = None
            ts = float(ts) if isinstance(ts, (int, float)) else None
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            txt, ts = str(item[0])[:400], (float(item[1])
                                           if isinstance(item[1], (int, float)) else None)
        else:
            txt, ts = str(item)[:400], None
        if ts is not None:
            n_real += 1
        out.append((txt, ts))
    return out, n_real


def _person_temporal_model(person_id: str, context: dict, *, now: float,
                           horizon_ts: float, channel: str) -> ScenarioTemporalModel:
    """A minimal per-question temporal model built ONLY from user-supplied context: timezone,
    sleep/active windows, channel checking, availability hypotheses. Nothing invented — absent
    fields stay absent and the attention model falls back to labeled broad bands."""
    tm = ScenarioTemporalModel(scenario_id=f"individual:{person_id}", as_of=now,
                               horizon_ts=horizon_ts)
    tz = str(context.get("timezone", "") or "")
    if "/" in tz:
        tm.timezones[person_id] = tz

    def _win(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                return (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                return None
        return None
    prof = ActorTemporalProfile(
        actor_id=person_id, timezone=tz if "/" in tz else "",
        sleep_window=_win(context.get("sleep_window")),
        active_window=_win(context.get("active_window")),
        workload_regime=str(context.get("workload", "") or "")[:24],
        channel_checking=({channel: dict(context["channel_check_gap"])}
                          if isinstance(context.get("channel_check_gap"), dict) else {}),
        relationship_priority=({str(context.get("relationship", "you")):
                                max(0.0, min(1.0, float(context["relationship_priority"])))}
                               if isinstance(context.get("relationship_priority"),
                                             (int, float)) else {}),
        latent_hypotheses=[h for h in (context.get("availability_hypotheses") or [])
                           if isinstance(h, dict) and h.get("state")],
        temporal_evidence=[f"user_context:{k}" for k in
                           ("timezone", "sleep_window", "active_window", "workload",
                            "channel_check_gap", "availability_hypotheses")
                           if context.get(k)],
        unresolved=[k for k in ("timezone", "sleep_window", "channel_check_gap")
                    if not context.get(k)],
        provenance="user_context")
    tm.actor_profiles[person_id] = prof
    if isinstance(context.get("channel_model"), dict):
        cm = context["channel_model"]
        tm.channels[channel] = ChannelTemporalModel(
            channel_id=channel, kind=str(cm.get("kind", channel))[:40],
            delivery=cm.get("delivery"), exposure=cm.get("exposure"),
            requires_attention=bool(cm.get("requires_attention", True)),
            provenance="user_context")
    tm.support_classification = "user_context_informed"
    return tm


def _mini_world(person_id: str, counterpart_id: str, context: dict, now: float,
                branch_id: str, history, n_real_ts: int) -> WorldState:
    w = WorldState("individual", branch_id, SimulationClock(now, now),
                   network=RelationGraph(), information=InformationLedger())
    person = Entity(person_id)
    person.set("roles", F([str(context.get("role", "person"))], status="observed"))
    if context.get("goals"):
        person.set("goals", F([str(g) for g in context["goals"]], status="inferred"))
    person.set("memory", F([t for t, _ in history][:12], status="observed"))
    person.set("past_actions", F([], status="observed"))
    counterpart = Entity(counterpart_id)
    counterpart.set("roles", F([str(context.get("your_role", "counterpart"))], status="observed"))
    w.entities = {person_id: person, counterpart_id: counterpart}
    w.network.add(counterpart_id, "communicates_with", person_id)
    # relationship history reaches the person as their OWN observed information at its REAL
    # times where supplied; unsupplied times keep order with spacing marked unresolved
    n = len(history)
    for i, (txt, ts) in enumerate(history):
        iid = f"history_{i}"
        at = ts if ts is not None else now - 86400.0 * (n - i)
        w.information.publish(InformationItem(iid, txt,
                                              source=str(context.get("relationship",
                                                                     counterpart_id)),
                                              created_at=at))
        w.information.expose(person_id, iid, at)
    return w


def simulate_individual_reaction(*, person_id: str, stimulus: str, context: dict | None = None,
                                 llm=None, counterpart_id: str = "you", channel: str = "message",
                                 n_hypotheses: int = 3, samples_per_hypothesis: int = 2,
                                 response_actions=DEFAULT_RESPONSE_ACTIONS, seed: int = 0,
                                 as_of: float | None = None, horizon_s: float = 7 * 86400.0,
                                 config: QualitativeConfig | None = None,
                                 calibrator: ActorPolicyCalibrator | None = None,
                                 scenario_schema=None, structural_frame: str = "") -> dict:
    """Simulate one person's reaction to one exact stimulus THROUGH the temporal model.

    ``context`` may supply: role, your_role, relationship, history (strings or {"text","ts"}),
    goals, timezone (IANA), sleep_window/active_window ([start_hour, end_hour] local),
    channel_check_gap (TimingSpec dict), relationship_priority (0..1),
    availability_hypotheses ([{state, prior, why}]), channel_model, urgency (0..1).

    ``structural_frame`` (structural-ensemble level-A uncertainty): ONE ensemble branch's
    hypothesized causal circumstance for the reaction (e.g. the relationship reading, attention
    and delivery, competing obligations). It conditions the hidden-state hypothesis SPACE the
    qualitative engine explores — always labeled a conjecture, never injected as observed fact,
    and never shown to the person as part of the stimulus. The default production route runs
    several such frames (structural_runtime._route_individual_reaction_ensemble), each with its
    own full sample budget."""
    from swm.world_model_v2.temporal_runtime import (channel_delivery_ts, compute_notice_ts,
                                                     get_stats, sample_temporal_latents)
    context = context or {}
    now = float(as_of if as_of is not None else _time.time())
    horizon_ts = now + max(3600.0, float(horizon_s))
    history, n_real_ts = _history_entries(context)
    tmodel = _person_temporal_model(person_id, context, now=now, horizon_ts=horizon_ts,
                                    channel=channel)
    cfg = config or QualitativeConfig(llm=llm, n_hypotheses=n_hypotheses,
                                      max_llm_calls=4 * n_hypotheses * samples_per_hypothesis)
    cfg.persistent = True
    if structural_frame:
        cfg.structural_frame = str(structural_frame)[:600]
    engine = QualitativeDecisionEngine(cfg)
    runtime = QualitativeActorPolicyRuntime(
        engine, mode="persistent_qualitative_llm_policy",
        tiers={person_id: {"tier": 1, "reasons": ["reaction_is_the_question"],
                           "selector": "individual-mode"}})
    urgency = max(0.0, min(1.0, float(context.get("urgency", 0.0) or 0.0)))
    outcomes, samples, truncated_samples = [], [], []
    nonresponse_breakdown: dict = {}
    total = max(1, n_hypotheses) * max(1, samples_per_hypothesis)
    n_unread = 0

    def _record_truncated(i, *, reason, detail, at):
        """§20/§33: one UNRESOLVED sample, first-class. Recorded exactly once (never resampled),
        never replaced by numeric anything, excluded from the distribution WITH its mass
        accounted in the truncation block."""
        status = _truncation_status(reason, detail)
        row = {"sample_index": i, "status": status, "reason": str(reason)[:80],
               "detail": str(detail)[:240], "at_ts": at,
               "hypothesis_id": "", "decision_source": "none_truncated",
               "temporal_state": "unresolved_truncated", "observable_response": "",
               "nonresponse_state": "", "trace_id": f"truncated_{i}"}
        truncated_samples.append(row)
        samples.append(dict(row))

    for i in range(total):
        world = _mini_world(person_id, counterpart_id, context, now,
                            branch_id=f"b{i:03d}", history=history, n_real_ts=n_real_ts)
        world.temporal_model = tmodel
        stats = get_stats(world)
        sample_temporal_latents(world, tmodel)
        if scenario_schema is not None:
            # generated actor-mediated mode for the individual route: the reply becomes a
            # scenario-typed ATTEMPT processed by the scenario's own mechanisms; without a
            # schema the runtime marks the branch execution_incomplete / structurally
            # under-modeled (never fixed-v1) — the reaction DISTRIBUTION, which is this
            # route's deliverable, still comes from the actor's own decisions either way
            import copy as _copy
            world.scenario_schema = _copy.deepcopy(scenario_schema)
        # ---- the stimulus TRAVELS: send → channel delivery → the person's real attention ----
        avail_ts, d_prov = channel_delivery_ts(world, tmodel, channel_id=channel, sent_ts=now,
                                               urgency=urgency, recipient=person_id,
                                               salt=f"stimulus:{i}", stats=stats)
        notice_ts, n_prov = compute_notice_ts(world, tmodel, actor_id=person_id,
                                              channel_id=channel, available_ts=avail_ts,
                                              urgency=urgency, sender=counterpart_id,
                                              stats=stats)
        if notice_ts > horizon_ts:
            # UNREAD within the window — a real, distinguishable outcome; no LLM is asked to
            # role-play a person who never saw the message (invariant 17; §25). §33: on this
            # route the temporal attention model owns noticing, so this IS "attention missed
            # the stimulus" → nonresponse state `unread`.
            n_unread += 1
            samples.append({"hypothesis_id": "", "decision_source": "not_invoked_unread",
                            "status": "completed",
                            "temporal_state": "unread_by_horizon",
                            "nonresponse_state": "unread",
                            "available_ts": avail_ts, "noticed_ts": None,
                            "delivery_provenance": d_prov, "notice_provenance": n_prov,
                            "observable_response": "", "trace_id": f"unread_{i}"})
            continue
        world.clock.advance_to(max(now, notice_ts))
        world.information.publish(InformationItem("stimulus", str(stimulus)[:800],
                                                  kind=channel or "message",
                                                  source=counterpart_id, created_at=now))
        world.information.expose(person_id, "stimulus", notice_ts)
        import datetime as _dt
        local = _dt.datetime.fromtimestamp(notice_ts).strftime("%H:%M")
        situation = (f"You just noticed (around {local}) this via {channel} from "
                     f"{context.get('relationship', counterpart_id)}: \"{str(stimulus)[:500]}\"")
        decision = {"situation": situation,
                    "candidate_actions": [{"name": a, "target": counterpart_id}
                                          for a in response_actions],
                    # §9-§15/§32/§33: the stimulus IS the availability set for the recipient's
                    # bounded-cognition stages — full exact text (never a summary slice), the
                    # question's channel, marked already-noticed because THIS route's temporal
                    # model (delivery→attention, §25) resolved noticing at notice_ts. The
                    # decision call then sees only the surviving cognition material.
                    "observation_bundle": [{
                        "iid": "stimulus", "channel": channel or "message",
                        "source": counterpart_id,
                        "content": str(stimulus)[:2000],
                        "urgency": urgency, "sent_ts": now,
                        "interrupting": True,
                        "exact_realized_message": True}],
                    "trigger": {"trigger_type": "newly_noticed_information",
                                "actor_id": person_id,
                                "why_now": n_prov, "provenance": "individual_reaction_route"}}
        try:
            selected, posterior, trace = runtime.decide(
                None, [world], person_id, decision=decision, seed=seed * 7919 + i)
        except ActorDecisionUnavailable as e:
            # §20/§33 strict integrity: the sample is TRUNCATED, first-class — never excluded-
            # and-continued with numeric anything, never silently resampled
            _record_truncated(i, reason=e.reason, detail=str(e), at=notice_ts)
            continue
        q = (posterior.provenance or {}).get("qualitative") or {}
        if q.get("decision_source") not in ("persistent_qualitative_llm", "stateless_llm"):
            # §33: on the personal-reaction route the numeric baseline arm is NEVER an
            # admissible substitute for the person's decision — even where the offline test
            # allowance (§19) lets other routes run it. The sample truncates instead.
            _record_truncated(
                i, reason=str(q.get("reason", "llm_failed_or_unparseable")),
                detail="numeric fallback refused on the personal-reaction route "
                       "(§33 strict sample integrity); sample recorded truncated",
                at=notice_ts)
            continue
        delta, _val = runtime.execute(world, selected, posterior, trace,
                                      seed=seed * 7919 + i)
        outcomes.append((posterior, trace))
        chosen = next((a for a in trace.candidate_actions
                       if a.get("action_id") == trace.sampled_action_id), {})
        deferred = q.get("act_or_wait") == "wait" or \
            str(chosen.get("action_name", "")) in ("wait", "reply_later", "ignore")
        blocked = any("blocked" in str(rc).lower()
                      for rc in (getattr(delta, "reason_codes", None) or []))
        nr = _nonresponse_state(
            temporal_state=("read_but_deferred" if deferred else "responded"),
            cognition=(posterior.provenance or {}).get("cognition") or {},
            act_or_wait=str(q.get("act_or_wait", "")),
            observable_response=str(chosen.get("action_name", "")), blocked=blocked)
        samples.append({
            "hypothesis_id": q.get("hypothesis_id", ""),
            "decision_source": q.get("decision_source", ""),
            "status": "completed",
            "temporal_state": ("read_but_deferred" if deferred else "responded"),
            "nonresponse_state": nr,
            "available_ts": avail_ts, "noticed_ts": notice_ts,
            "delivery_provenance": d_prov, "notice_provenance": n_prov,
            "availability_latent": str(getattr(world.quantities.get(
                f"temporal_latent:actor:{person_id}"), "value", "") or ""),
            "interpretation": q.get("situation_interpretation", {}),
            "internal_reaction": q.get("internal_reaction", ""),
            "observable_response": chosen.get("action_name", ""),
            "timing_intent": q.get("timing", ""),
            "target": (chosen.get("target") or {}).get("target_id", ""),
            "decision_summary": q.get("decision_summary", ""),
            "novel_action_unmodeled": bool(q.get("novel_action_unmodeled")),
            "trace_id": trace.trace_id,
            # §35.2: the per-sample bounded-cognition record (compact stage outputs)
            "cognition": {k: v for k, v in
                          (((posterior.provenance or {}).get("cognition")) or {}).items()
                          if k in ("model_family", "observations_noticed",
                                   "observations_missed", "working_memory_capacity",
                                   "memories_retrieved", "retrieval_failures",
                                   "options_considered", "actually_feasible_not_considered",
                                   "stage_traces")},
        })
    for s in samples:
        if s.get("nonresponse_state"):
            nonresponse_breakdown[s["nonresponse_state"]] = \
                nonresponse_breakdown.get(s["nonresponse_state"], 0) + 1
    agg = aggregate_actor_decisions(outcomes, clusterer=ActionClusterer(),
                                    calibrator=calibrator or
                                    ActorPolicyCalibrator.from_file())
    result = agg.get(person_id, {"raw_qualitative_simulation_distribution": {},
                                 "calibrated_distribution": {},
                                 "calibration_status": "unvalidated", "rows": []})
    raw = dict(result["raw_qualitative_simulation_distribution"])
    # ---- §20/§21 honest aggregation: the distribution covers ONLY completed samples --------
    n_truncated = len(truncated_samples)
    n_completed = total - n_truncated
    trunc_share = round(n_truncated / total, 6) if total else 0.0
    if n_unread and n_completed:
        # unread mass enters the distribution explicitly — "no response yet" is NOT "ignored"
        scale = (n_completed - n_unread) / n_completed
        raw = {k: round(v * scale, 4) for k, v in raw.items()}
        raw["unread_no_response_yet"] = round(n_unread / n_completed, 4)
    from swm.world_model_v2.truncation import honest_note, truncation_bounds
    dist_by_branch = {str(row.get("branch_id") or row.get("trace_id") or f"s{j}"):
                      row["cluster"] for j, row in enumerate(result.get("rows") or [])}
    for s in samples:
        if s.get("temporal_state") == "unread_by_horizon":
            dist_by_branch[str(s.get("trace_id"))] = "unread_no_response_yet"
    truncation_block = {
        "n_samples_truncated": n_truncated,
        "truncated_share": trunc_share,
        "statuses": {st: sum(1 for t in truncated_samples if t["status"] == st)
                     for st in sorted({t["status"] for t in truncated_samples})},
        "bounds_under_truncation": truncation_bounds(dist_by_branch, trunc_share,
                                                     sorted(raw) or
                                                     sorted(set(dist_by_branch.values()))),
        "distribution_note": (
            f"the distribution counts ONLY the {n_completed} completed samples; the truncated "
            f"share ({trunc_share}) is unresolved and could swing ANY option — see "
            "bounds_under_truncation" if n_truncated else
            "no samples were truncated; bounds collapse to the point distribution"),
        "honest_note": honest_note(),
    }
    return {
        "schema_version": "individual.reaction.v3_truncation_honest",
        "person_id": person_id, "stimulus": str(stimulus)[:800], "channel": channel,
        "n_hypotheses": n_hypotheses, "samples_per_hypothesis": samples_per_hypothesis,
        "samples": samples,
        "raw_qualitative_simulation_distribution": raw,
        "calibrated_distribution": result["calibrated_distribution"],
        "calibration_status": result["calibration_status"],
        "n_excluded_numeric_fallbacks": result.get("n_excluded_numeric_fallbacks", 0),
        "n_unread_by_horizon": n_unread,
        # §20/§33 first-class sample truncation accounting
        "n_samples_total": total,
        "n_samples_completed": n_completed,
        "n_samples_truncated": n_truncated,
        "truncated_samples": truncated_samples,
        "truncation": truncation_block,
        # §33 distinguishable nonresponse states over completed samples
        "nonresponse_breakdown": nonresponse_breakdown,
        "consequence_report": runtime.consequence_report,
        "llm_calls": engine.calls_used(),
        "temporal": {
            "temporal_model_hash": tmodel.temporal_model_hash(),
            "support_classification": tmodel.support_classification,
            "horizon_s": float(horizon_s),
            "history_timestamps": {"n_supplied_real": n_real_ts,
                                   "n_total": len(history),
                                   "unresolved_spacing": len(history) - n_real_ts},
            "profile_evidence": list(tmodel.actor_profiles[person_id].temporal_evidence),
            "profile_unresolved": list(tmodel.actor_profiles[person_id].unresolved)},
        "provenance": {"as_of": now, "tier_rule": "reaction_is_the_question → Tier 1",
                       "runtime": "QualitativeActorPolicyRuntime",
                       "temporal_route": "delivery→attention→decision (§25)",
                       "cognition_route": "stimulus availability set → bounded cognition "
                                          "(§9-§15; full exact stimulus text, §32) → decision "
                                          "sees only surviving material",
                       "sample_integrity": "failed decision ⇒ first-class truncated sample "
                                           "(§20/§33); numeric substitution and silent "
                                           "resampling are prohibited on this route",
                       "aggregation": "branch-selection counting (cluster-1.0)",
                       "structural_frame": getattr(cfg, "structural_frame", "") or ""},
    }
