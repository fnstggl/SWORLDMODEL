"""LLM-backed SCENARIO TEMPORAL COMPILATION + independent temporal critics (§4).

The default production runtime calls `compile_temporal_model` once per scenario (content-
addressed cache) with the generated causal world, actors, institutions, relationships,
scheduled facts, evidence, as-of, horizon, user context and intervention. The LLM is asked to
identify the scenario's REAL temporal structure — exact times where known, dependencies,
per-actor attention situations, decision triggers, procedural stages, continuous processes,
calendar restrictions, and honest unknowns. It is NEVER asked for one invented delay constant:
timing answers are TimingSpecs (exact | bounded range | qualitative regime | calendar
expression | dependency | unresolved) with provenance.

Two INDEPENDENT critic calls then check the twelve §4 failure classes (missing processes,
unrealistic speed/slowness, missing sleep/availability, missing institutional stages, missing
implementation lag, wrong simultaneity/ordering, invented precision, missing deadlines,
missing business-day/timezone effects, synthetic recurrence). Critic repairs are applied as
typed patches and recorded — never silently.

Every LLM call is traced (stage, prompt hash, raw response, parsed?, repairs, accepted/
rejected fields) into the model's `compilation_trace`; recurrence without a source is REFUSED
(§5); malformed timing collapses to `unresolved`, never to a made-up number.
"""
from __future__ import annotations

import hashlib
import json
import time as _time

from swm.world_model_v2.temporal_model import (ActorTemporalProfile, ChannelTemporalModel,
                                               ContinuousProcessSpec, InstitutionalProcessModel,
                                               InstitutionalStage, ScenarioTemporalModel,
                                               TIMING_REGIMES, TimingSpec)

_CACHE: dict = {}                                             # content-addressed, in-process

_TIMING_DOC = """A TIMING value is ALWAYS one JSON object of exactly one kind (never a bare number):
  {"kind":"exact","ts":"YYYY-MM-DDTHH:MM:SSZ","provenance":"evidence|user_context|model_knowledge"}
  {"kind":"range","lo_s":<seconds>,"hi_s":<seconds>,"provenance":"..."}   (bounded duration)
  {"kind":"regime","regime":"immediate|minutes|within_hour|hours|same_day|next_day|days|week|weeks|months"}
  {"kind":"calendar","calendar_expr":"tomorrow_morning|end_of_day|this_evening|next_business_day|next_morning_window","calendar_of":"<actor_or_institution_id>"}
  {"kind":"after_event","depends_on":"<event or condition>","lag":{...timing...}}
  {"kind":"unresolved","description":"<what is unknown and why>"}
Use "exact" ONLY for genuinely known real times. Use "unresolved" when you do not know —
NEVER invent precise durations."""

_WORLD_PROMPT = """You are the SCENARIO TEMPORAL COMPILER for a causal world simulation. Real timestamps already
exist; your job is the scenario's REAL temporal structure: WHY things happen WHEN they happen.

QUESTION: {q}
AS-OF: {as_of}   HORIZON: {horizon}
INTERVENTION (may be none): {intervention}
USER CONTEXT (may be none): {user_context}
ACTORS: {actors}
INSTITUTIONS: {institutions}
RELATIONS: {relations}
ALREADY-KNOWN SCHEDULED FACTS (exact, keep exact): {scheduled}
EVIDENCE: {ev}

{timing_doc}

Identify for THIS scenario (omit sections that genuinely do not apply — an empty list is a
valid answer; do not pad):
1. timezones: where each named actor/institution plausibly operates (IANA names). Unknown = omit.
2. calendars: working calendars that matter (business days/hours, weekend behavior). Only if relevant.
3. channels: the ACTUAL information channels between these actors (direct message, email, call,
   public post, filing, press, intermediary). For each: stage timings (transmission, delivery,
   moderation if any, exposure for broadcast), whether noticing is separate from delivery
   (almost always true), and situation modifiers (urgency, relationship, hour, weekday).
4. institutional_processes: REAL procedural stage machines (submission→queue→review→decision→
   implementation), each stage with entry condition, responsible holder, duration TIMING,
   working calendar, deadline. Only stages the institution really has.
5. continuous_processes: things evolving with elapsed time (fatigue, attention recovery,
   queue buildup, pressure) as {{process_id, writes, form: exponential_decay|linear_drift|
   exponential_approach, rate_per_day, target?, active_when?}}. Only if causally relevant.
6. deadlines: real deadlines with exact ts when known (else TIMING), what they bind, source.
7. dependencies: events that cannot happen before другие events (list {{event, depends_on, why}}).
8. recurring_obligations: ONLY real documented recurrences (a known weekly meeting, a scheduled
   committee session). Each MUST carry source + timezone + local time + participants + relevance.
   NO generic review cadences — if you cannot name the real recurring process, return none.
9. decision_trigger_sources: what would REALLY cause each actor to face a decision here
   (list {{actor, trigger_type, description}}); trigger_type is free text.
10. simultaneity: events that may genuinely co-occur and how conflicts resolve (institution/
    mechanism), vs events that look simultaneous but are causally ordered.
11. correlated_latents: shared temporal conditions affecting several parties at once (a holiday,
    an outage, a shared crisis workload) as {{latent_id, affects:[ids], hypotheses:[{{state,prior,why}}]}}.
12. temporal_uncertainties + unresolved_mechanisms: what timing you genuinely cannot know, and
    what evidence would resolve it.

Return ONLY JSON:
{{"timezones": {{"<id>": "<IANA tz>"}},
 "calendars": {{"<calendar_id>": {{"tz": "...", "business_days": [1,2,3,4,5], "open_hour": <h>,
                "close_hour": <h>, "holidays": ["YYYY-MM-DD"], "provenance": "..."}}}},
 "channels": {{"<channel_id>": {{"kind": "...", "transmission": {{...timing...}},
               "delivery": {{...timing...}}, "moderation": {{...timing...}}|null,
               "exposure": {{...timing...}}|null, "requires_attention": true,
               "modifiers": [{{"when": "...", "effect": "...", "factor": <mult>|null}}],
               "provenance": "..."}}}},
 "institutional_processes": [{{"process_id": "...", "institution_id": "...",
    "started_by": "<condition/event>", "initial_stages": ["<stage_id>"],
    "stages": [{{"stage_id": "...", "entry_condition": "...", "responsible": "...",
                 "duration": {{...timing...}}, "working_calendar": "<calendar_id>",
                 "deadline": {{...timing...}}|null, "output": "...",
                 "next_stages": ["..."], "creates_decision_for": "<actor id or empty>"}}]}}],
 "continuous_processes": [{{"process_id": "...", "writes": "<quantity name>",
    "form": "exponential_decay|linear_drift|exponential_approach", "rate_per_day": <num>,
    "target": <num>|null, "active_when": "<condition or empty>", "provenance": "..."}}],
 "deadlines": [{{"label": "...", "timing": {{...timing...}}, "binds": "<who/what>", "source": "..."}}],
 "dependencies": [{{"event": "...", "depends_on": "...", "why": "..."}}],
 "recurring_obligations": [{{"rule_id": "...", "freq": "weekly|biweekly|monthly_day|daily",
    "tz": "<IANA>", "local_hour": <h>, "weekday": <1-7>|null, "month_day": <1-28>|null,
    "participants": ["..."], "source": "<evidence/user context/documented — REQUIRED>",
    "relevance": "...", "confidence": <0..1>}}],
 "decision_trigger_sources": [{{"actor": "...", "trigger_type": "...", "description": "..."}}],
 "simultaneity": [{{"events": ["...", "..."], "relation": "independent|causally_ordered|conflict",
    "mechanism": "<how a genuine conflict resolves, or empty>"}}],
 "correlated_latents": [{{"latent_id": "...", "affects": ["..."],
    "hypotheses": [{{"state": "...", "prior": <0..1>, "why": "..."}}]}}],
 "temporal_uncertainties": [{{"about": "...", "why_unknown": "...", "impact": "..."}}],
 "unresolved_mechanisms": ["<timing mechanism you cannot model with current information>"],
 "assumptions": ["<each temporal assumption you made>"]}}"""

_ACTOR_PROMPT = """You are the ACTOR TEMPORAL PROFILER for a causal world simulation. For each actor below,
describe their plausible temporal situation IN THIS SCENARIO — availability, attention, and
response behavior. Use ONLY the context/evidence given plus general world knowledge about the
KIND of actor; do NOT invent precise personal schedules for real people without support —
unknown stays unknown, and mutually exclusive possibilities go into latent_hypotheses (they
will be sampled per simulated world and PERSIST).

QUESTION: {q}
AS-OF: {as_of}   HORIZON: {horizon}
USER CONTEXT: {user_context}
ACTORS TO PROFILE: {actors}
CHANNELS IN THIS SCENARIO: {channels}
EVIDENCE: {ev}

{timing_doc}

Return ONLY JSON:
{{"profiles": {{"<actor_id>": {{
   "timezone": "<IANA or empty if unknown>",
   "sleep_window": [<local_start_hour>, <local_end_hour>] | null,
   "active_window": [<h>, <h>] | null,
   "workload_regime": "<normal|busy|crisis|unknown>",
   "channel_checking": {{"<channel_id>": {{...timing (gap between checks)...}}}},
   "urgency_interrupt": {{"<channel_id>": {{"threshold": <0..1>, "why": "..."}}}},
   "relationship_priority": {{"<sender_id>": <0..1>}},
   "pending_obligations": ["..."], "deadline_awareness": ["..."],
   "batching_habit": "<empty|batches_low_urgency|handles_immediately|unknown>",
   "response_expectation": {{"<channel_id>": {{...timing...}}}},
   "latent_hypotheses": [{{"state": "available|asleep_offset|traveling|in_meetings|
       crisis_workload|<other>", "prior": <0..1>, "why": "...", "source": "..."}}],
   "temporal_evidence": ["<supporting evidence/context, if any>"],
   "unresolved": ["<what you do not know about this actor's time>"]}}}}}}"""

_CRITIC_A_PROMPT = """You are TEMPORAL CRITIC A — an independent reviewer of a scenario temporal model. Check ONLY:
1. MISSING temporal processes (a real channel/process/stage the model omits);
2. UNREALISTIC SPEED (things that could not plausibly happen this fast);
3. UNREALISTIC SLOWNESS;
4. missing SLEEP or availability constraints for individually simulated humans;
5. missing INSTITUTIONAL STAGES (a real procedure collapsed to one step);
6. missing IMPLEMENTATION LAG (decision treated as instantly complete).

QUESTION: {q}
AS-OF: {as_of} HORIZON: {horizon}
TEMPORAL MODEL (compact): {model}

Return ONLY JSON:
{{"findings": [{{"class": "missing_process|too_fast|too_slow|missing_availability|missing_stages|missing_implementation_lag",
   "where": "<model path/id>", "problem": "<one sentence>",
   "repair": {{"op": "add_channel|add_stage|add_continuous|adjust_timing|add_profile_field|note_only",
               "target": "<id/path>", "value": <JSON — for adjust_timing a TIMING object>}} | null}}]}}"""

_CRITIC_B_PROMPT = """You are TEMPORAL CRITIC B — an independent reviewer of a scenario temporal model. Check ONLY:
1. events treated as simultaneous that are CAUSALLY ORDERED;
2. events treated as ordered that are genuinely SIMULTANEOUS;
3. INVENTED PRECISION (exact times/durations without evidence — must become range/regime/unresolved);
4. MISSING DEADLINES that really bind this scenario;
5. missing BUSINESS-DAY or TIMEZONE effects;
6. SYNTHETIC RECURRENCE: recurring obligations with no real source (these must be REMOVED).

QUESTION: {q}
AS-OF: {as_of} HORIZON: {horizon}
TEMPORAL MODEL (compact): {model}

Return ONLY JSON:
{{"findings": [{{"class": "wrong_simultaneity|wrong_ordering|invented_precision|missing_deadline|missing_calendar_effect|synthetic_recurrence",
   "where": "<model path/id>", "problem": "<one sentence>",
   "repair": {{"op": "remove_recurrence|demote_to_unresolved|add_deadline|add_dependency|note_only",
               "target": "<id/path>", "value": <JSON>}} | null}}]}}"""


def _trace(model: ScenarioTemporalModel, *, stage: str, prompt: str, response,
           parsed_ok: bool, repairs=(), accepted=(), rejected=()):
    model.compilation_trace.append({
        "stage": stage, "at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "model": "configured_llm_callable",
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "prompt_chars": len(prompt),
        "response": (str(response)[:4000] if response is not None else None),
        "parsed_ok": bool(parsed_ok), "repairs": list(repairs),
        "accepted_fields": list(accepted), "rejected_fields": list(rejected)})


def _parse_ts(v, default=None):
    from swm.world_model_v2.state import parse_time
    try:
        return parse_time(v)
    except (ValueError, TypeError):
        return default


def _coerce_timing(raw, *, repairs: list, where: str) -> dict:
    """Raw LLM timing → validated TimingSpec dict. Bare numbers are REFUSED (invented
    precision) and become unresolved; exact kinds parse their timestamp."""
    if raw in (None, "", {}):
        return None
    if isinstance(raw, (int, float, str)) and not isinstance(raw, bool):
        repairs.append(f"{where}: bare timing value {str(raw)[:40]!r} refused → unresolved")
        return TimingSpec(kind="unresolved",
                          description=f"bare value {str(raw)[:60]!r} without structure").as_dict()
    spec = TimingSpec.from_dict(raw if isinstance(raw, dict) else {})
    if spec.kind == "exact":
        ts = _parse_ts(spec.ts if isinstance(spec.ts, (int, float)) else raw.get("ts"))
        if ts is None:
            repairs.append(f"{where}: unparseable exact ts → unresolved")
            return TimingSpec(kind="unresolved", description="unparseable exact ts").as_dict()
        spec.ts = ts
    if spec.kind == "unresolved" and isinstance(raw, dict) and raw.get("kind") not in (None, "unresolved"):
        repairs.append(f"{where}: malformed timing {json.dumps(raw)[:60]} → unresolved")
    return spec.as_dict()


def _plan_scenario_id(plan) -> str:
    try:
        return plan.plan_hash()
    except Exception:  # noqa: BLE001
        return hashlib.sha1(str(getattr(plan, "question", "")).encode()).hexdigest()[:12]


def compile_temporal_model(plan, *, llm, question: str = "", evidence_text: str = "",
                           user_context=None, intervention: str = "",
                           structural_model_id: str = "", seed: int = 0,
                           max_profiled_actors: int = None) -> ScenarioTemporalModel:
    """The default-on temporal compilation stage. Returns a ScenarioTemporalModel; with no LLM
    backend it returns a DEGRADED model (degraded='no_llm_backend') — recorded loudly by the
    runtime, never silently swapped for a fixed calendar."""
    q = question or str(getattr(plan, "question", ""))
    as_of, horizon_ts = float(plan.as_of), float(plan.horizon_ts)
    key = hashlib.sha256(
        f"{_plan_scenario_id(plan)}|{q}|{as_of}|{horizon_ts}|{structural_model_id}|"
        f"{json.dumps(user_context, sort_keys=True, default=str) if user_context else ''}|"
        f"{intervention}".encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    from swm.world_model_v2.state import rfc3339
    model = ScenarioTemporalModel(scenario_id=_plan_scenario_id(plan),
                                  structural_model_id=structural_model_id,
                                  as_of=as_of, horizon_ts=horizon_ts)
    # exact scheduled facts are copied through EXACTLY (they are already real)
    for e in (getattr(plan, "scheduled_events", None) or []):
        if isinstance(e, dict) and e.get("etype") == "scheduled_fact":
            p = e.get("payload") or {}
            model.scheduled_facts.append({"ts": float(e.get("ts", 0.0)),
                                          "fact": str(p.get("fact", ""))[:160],
                                          "source": str(p.get("source", ""))})
    if llm is None:
        model.degraded = "no_llm_backend"
        model.unresolved_mechanisms.append(
            "temporal compilation requires an LLM backend; no scenario temporal structure was "
            "generated — delivery/attention fall back to labeled broad regimes")
        _CACHE[key] = model
        return model

    from swm.engine.grounding import parse_json
    actors = [{"id": str(e.get("id")), "type": str(e.get("type", "person")),
               "role": str((e.get("fields") or {}).get("role", ""))[:60]}
              for e in (plan.entities or []) if isinstance(e, dict) and e.get("id")]
    insts = [str(i.get("id")) for i in (plan.institutions or []) if isinstance(i, dict)]
    rels = [(str(r.get("src")), str(r.get("rel")), str(r.get("dst")))
            for r in (plan.relations or []) if isinstance(r, dict)][:24]
    sched = [{"ts": rfc3339(float(e.get("ts", 0.0))), "etype": str(e.get("etype")),
              "what": str((e.get("payload") or {}).get("fact", ""))[:80]}
             for e in (getattr(plan, "scheduled_events", None) or [])
             if isinstance(e, dict) and e.get("etype") in ("scheduled_fact",
                                                           "institutional_decision")][:12]
    uc = json.dumps(user_context, default=str)[:1200] if user_context else "(none)"

    # ---------- stage 1: world temporal structure ----------
    prompt1 = _WORLD_PROMPT.format(q=q[:500], as_of=rfc3339(as_of), horizon=rfc3339(horizon_ts),
                                   intervention=str(intervention)[:300] or "(none)",
                                   user_context=uc, actors=actors[:40], institutions=insts[:12],
                                   relations=rels, scheduled=sched,
                                   ev=evidence_text[:2000] or "(none)", timing_doc=_TIMING_DOC)
    repairs, accepted, rejected = [], [], []
    raw1_txt = None
    try:
        raw1_txt = llm(prompt1)
        raw1 = parse_json(raw1_txt) or {}
    except Exception as e:  # noqa: BLE001
        raw1 = {}
        repairs.append(f"world stage failed: {type(e).__name__}: {e}"[:160])
    _ingest_world_structure(model, raw1, repairs=repairs, accepted=accepted, rejected=rejected)
    _trace(model, stage="temporal_world_structure", prompt=prompt1, response=raw1_txt,
           parsed_ok=bool(raw1), repairs=repairs, accepted=accepted, rejected=rejected)

    # ---------- stage 2: actor temporal profiles ----------
    profile_ids = [a["id"] for a in actors]
    if max_profiled_actors:
        profile_ids = profile_ids[:max_profiled_actors]        # explicit caller budget, recorded
        if len(actors) > len(profile_ids):
            model.unresolved_mechanisms.append(
                f"actor temporal profiles compiled for {len(profile_ids)}/{len(actors)} actors "
                f"(caller budget); remaining actors use latent-hypothesis-free attention")
    if profile_ids:
        prompt2 = _ACTOR_PROMPT.format(q=q[:500], as_of=rfc3339(as_of),
                                       horizon=rfc3339(horizon_ts), user_context=uc,
                                       actors=[a for a in actors if a["id"] in set(profile_ids)],
                                       channels=sorted(model.channels)[:16],
                                       ev=evidence_text[:1600] or "(none)",
                                       timing_doc=_TIMING_DOC)
        rep2, acc2 = [], []
        raw2_txt = None
        try:
            raw2_txt = llm(prompt2)
            raw2 = parse_json(raw2_txt) or {}
        except Exception as e:  # noqa: BLE001
            raw2 = {}
            rep2.append(f"actor stage failed: {type(e).__name__}: {e}"[:160])
        _ingest_actor_profiles(model, raw2, repairs=rep2, accepted=acc2)
        _trace(model, stage="temporal_actor_profiles", prompt=prompt2, response=raw2_txt,
               parsed_ok=bool(raw2), repairs=rep2, accepted=acc2)

    # ---------- stage 3+4: independent temporal critics ----------
    compact = json.dumps(model.as_dict(include_trace=False), default=str)[:6000]
    for stage, tmpl in (("temporal_critic_A", _CRITIC_A_PROMPT),
                        ("temporal_critic_B", _CRITIC_B_PROMPT)):
        pr = tmpl.format(q=q[:400], as_of=rfc3339(as_of), horizon=rfc3339(horizon_ts),
                         model=compact)
        repc, accc = [], []
        rawc_txt = None
        try:
            rawc_txt = llm(pr)
            rawc = parse_json(rawc_txt) or {}
        except Exception as e:  # noqa: BLE001
            rawc = {}
            repc.append(f"critic failed: {type(e).__name__}: {e}"[:160])
        n_applied = _apply_critic_findings(model, rawc.get("findings") or [],
                                           critic=stage, repairs=repc, accepted=accc)
        _trace(model, stage=stage, prompt=pr, response=rawc_txt, parsed_ok=bool(rawc),
               repairs=repc, accepted=accc)
        if n_applied:
            compact = json.dumps(model.as_dict(include_trace=False), default=str)[:6000]

    model.support_classification = ("scenario_generated_unvalidated"
                                    if not model.degraded else "degraded")
    _CACHE[key] = model
    return model


def _ingest_world_structure(model, raw, *, repairs, accepted, rejected):
    for k, v in (raw.get("timezones") or {}).items():
        if isinstance(v, str) and "/" in v:
            model.timezones[str(k)] = v
    if model.timezones:
        accepted.append(f"timezones:{len(model.timezones)}")
    for cid, c in (raw.get("calendars") or {}).items():
        if isinstance(c, dict) and c.get("tz"):
            model.calendars[str(cid)] = {
                "tz": str(c["tz"]),
                "business_days": tuple(int(d) for d in (c.get("business_days") or (1, 2, 3, 4, 5))
                                       if isinstance(d, (int, float)) and 1 <= int(d) <= 7),
                "open_hour": float(c.get("open_hour", 9.0) or 9.0),
                "close_hour": float(c.get("close_hour", 17.0) or 17.0),
                "holidays": tuple(str(h)[:10] for h in (c.get("holidays") or [])[:24]),
                "provenance": str(c.get("provenance", "scenario_generated"))[:120]}
    if model.calendars:
        accepted.append(f"calendars:{len(model.calendars)}")
    for cid, c in (raw.get("channels") or {}).items():
        if not isinstance(c, dict):
            continue
        model.channels[str(cid)] = ChannelTemporalModel(
            channel_id=str(cid), kind=str(c.get("kind", ""))[:40],
            transmission=_coerce_timing(c.get("transmission"), repairs=repairs,
                                        where=f"channel:{cid}.transmission"),
            delivery=_coerce_timing(c.get("delivery"), repairs=repairs,
                                    where=f"channel:{cid}.delivery"),
            moderation=_coerce_timing(c.get("moderation"), repairs=repairs,
                                      where=f"channel:{cid}.moderation"),
            exposure=_coerce_timing(c.get("exposure"), repairs=repairs,
                                    where=f"channel:{cid}.exposure"),
            requires_attention=bool(c.get("requires_attention", True)),
            modifiers=[m for m in (c.get("modifiers") or []) if isinstance(m, dict)][:8],
            provenance=str(c.get("provenance", "scenario_generated"))[:120])
    if model.channels:
        accepted.append(f"channels:{len(model.channels)}")
    for p in (raw.get("institutional_processes") or []):
        if not isinstance(p, dict) or not p.get("stages"):
            continue
        stages = []
        for s in p["stages"]:
            if not isinstance(s, dict) or not s.get("stage_id"):
                continue
            stages.append(InstitutionalStage(
                stage_id=str(s["stage_id"])[:60],
                institution_id=str(p.get("institution_id", ""))[:60],
                entry_condition=str(s.get("entry_condition", ""))[:160],
                responsible=str(s.get("responsible", ""))[:60],
                required_inputs=[str(x)[:60] for x in (s.get("required_inputs") or [])[:6]],
                earliest_start=_coerce_timing(s.get("earliest_start"), repairs=repairs,
                                              where=f"stage:{s['stage_id']}.earliest_start"),
                duration=_coerce_timing(s.get("duration"), repairs=repairs,
                                        where=f"stage:{s['stage_id']}.duration"),
                working_calendar=str(s.get("working_calendar", ""))[:40],
                deadline=_coerce_timing(s.get("deadline"), repairs=repairs,
                                        where=f"stage:{s['stage_id']}.deadline"),
                output=str(s.get("output", ""))[:120],
                next_stages=[str(x)[:60] for x in (s.get("next_stages") or [])[:6]],
                creates_decision_for=str(s.get("creates_decision_for", ""))[:60]))
        if stages:
            model.institutional_processes.append(InstitutionalProcessModel(
                process_id=str(p.get("process_id", f"proc_{len(model.institutional_processes)}"))[:60],
                institution_id=str(p.get("institution_id", ""))[:60], stages=stages,
                initial_stages=[str(x)[:60] for x in (p.get("initial_stages") or [])[:4]]
                or [stages[0].stage_id],
                started_by=str(p.get("started_by", ""))[:160]))
    if model.institutional_processes:
        accepted.append(f"institutional_processes:{len(model.institutional_processes)}")
    for p in (raw.get("continuous_processes") or []):
        if not isinstance(p, dict) or not p.get("process_id") or not p.get("writes"):
            continue
        try:
            rate = float(p.get("rate_per_day", 0.0) or 0.0)
        except (TypeError, ValueError):
            rejected.append(f"continuous:{p.get('process_id')}: non-numeric rate")
            continue
        form = str(p.get("form", "exponential_decay"))
        if form not in ("exponential_decay", "linear_drift", "exponential_approach", "logistic"):
            rejected.append(f"continuous:{p.get('process_id')}: unknown form {form!r}")
            continue
        model.continuous_processes.append(ContinuousProcessSpec(
            process_id=str(p["process_id"])[:60], writes=str(p["writes"])[:80], form=form,
            rate_per_day=rate,
            target=(float(p["target"]) if isinstance(p.get("target"), (int, float)) else None),
            active_when=str(p.get("active_when", ""))[:80],
            provenance=str(p.get("provenance", "scenario_generated"))[:120]))
    if model.continuous_processes:
        accepted.append(f"continuous_processes:{len(model.continuous_processes)}")
    for d in (raw.get("deadlines") or []):
        if isinstance(d, dict) and d.get("label"):
            model.deadlines.append({
                "label": str(d["label"])[:120],
                "timing": _coerce_timing(d.get("timing"), repairs=repairs,
                                         where=f"deadline:{d['label'][:30]}"),
                "binds": str(d.get("binds", ""))[:80], "source": str(d.get("source", ""))[:120]})
    model.dependencies = [d for d in (raw.get("dependencies") or [])
                          if isinstance(d, dict) and d.get("event") and d.get("depends_on")][:24]
    for r in (raw.get("recurring_obligations") or []):
        if not isinstance(r, dict):
            continue
        if not str(r.get("source", "")).strip():
            rejected.append(f"recurrence:{r.get('rule_id', '?')}: NO SOURCE — refused (§5: "
                            f"no evidence/scenario reason means no recurring review)")
            continue
        if str(r.get("freq")) not in ("weekly", "biweekly", "monthly_day", "daily"):
            rejected.append(f"recurrence:{r.get('rule_id', '?')}: unknown freq")
            continue
        try:
            model.recurring_obligations.append({
                "rule_id": str(r.get("rule_id", f"rec_{len(model.recurring_obligations)}"))[:60],
                "freq": str(r["freq"]), "tz": str(r.get("tz", "UTC")),
                "local_hour": float(r.get("local_hour", 9.0) or 9.0),
                "weekday": (int(r["weekday"]) if isinstance(r.get("weekday"), (int, float)) else None),
                "month_day": (int(r["month_day"]) if isinstance(r.get("month_day"), (int, float)) else None),
                "participants": [str(x)[:60] for x in (r.get("participants") or [])[:12]],
                "source": str(r["source"])[:160],
                "cancellation_conditions": [str(x)[:80] for x in
                                            (r.get("cancellation_conditions") or [])[:4]],
                "relevance": str(r.get("relevance", ""))[:120],
                "confidence": max(0.0, min(1.0, float(r.get("confidence", 0.6) or 0.6)))})
        except (TypeError, ValueError):
            rejected.append(f"recurrence:{r.get('rule_id', '?')}: malformed")
    model.decision_trigger_sources = [
        {"actor": str(t.get("actor", ""))[:60], "trigger_type": str(t.get("trigger_type", ""))[:48],
         "description": str(t.get("description", ""))[:160]}
        for t in (raw.get("decision_trigger_sources") or []) if isinstance(t, dict)][:40]
    model.simultaneity_rules = [s for s in (raw.get("simultaneity") or [])
                                if isinstance(s, dict) and s.get("events")][:16]
    for cl in (raw.get("correlated_latents") or []):
        if isinstance(cl, dict) and cl.get("latent_id") and cl.get("hypotheses"):
            model.correlated_latents.append({
                "latent_id": str(cl["latent_id"])[:60],
                "affects": [str(x)[:60] for x in (cl.get("affects") or [])[:24]],
                "hypotheses": [{"state": str(h.get("state", ""))[:60],
                                "prior": max(0.0, min(1.0, float(h.get("prior", 0.5) or 0.5))),
                                "why": str(h.get("why", ""))[:120]}
                               for h in cl["hypotheses"][:6] if isinstance(h, dict)]})
    model.temporal_uncertainties = [u for u in (raw.get("temporal_uncertainties") or [])
                                    if isinstance(u, dict)][:24]
    model.unresolved_mechanisms.extend(str(u)[:200] for u in
                                       (raw.get("unresolved_mechanisms") or [])[:16])
    model.assumptions = [str(a)[:200] for a in (raw.get("assumptions") or [])[:24]]


def _ingest_actor_profiles(model, raw, *, repairs, accepted):
    for aid, p in (raw.get("profiles") or {}).items():
        if not isinstance(p, dict):
            continue
        def _win(v):
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    return (float(v[0]), float(v[1]))
                except (TypeError, ValueError):
                    return None
            return None
        prof = ActorTemporalProfile(
            actor_id=str(aid),
            timezone=(str(p.get("timezone", "")) if "/" in str(p.get("timezone", "")) else ""),
            sleep_window=_win(p.get("sleep_window")),
            active_window=_win(p.get("active_window")),
            workload_regime=str(p.get("workload_regime", ""))[:24],
            channel_checking={str(k): _coerce_timing(v, repairs=repairs,
                                                     where=f"profile:{aid}.check:{k}")
                              for k, v in (p.get("channel_checking") or {}).items()
                              if isinstance(v, (dict, int, float, str))},
            urgency_interrupt={str(k): {"threshold": max(0.0, min(1.0, float(
                                            (v or {}).get("threshold", 0.8) or 0.8))),
                                        "why": str((v or {}).get("why", ""))[:100]}
                               for k, v in (p.get("urgency_interrupt") or {}).items()
                               if isinstance(v, dict)},
            relationship_priority={str(k): max(0.0, min(1.0, float(v)))
                                   for k, v in (p.get("relationship_priority") or {}).items()
                                   if isinstance(v, (int, float))},
            pending_obligations=[str(x)[:100] for x in (p.get("pending_obligations") or [])[:8]],
            deadline_awareness=[str(x)[:100] for x in (p.get("deadline_awareness") or [])[:8]],
            batching_habit=str(p.get("batching_habit", ""))[:40],
            response_expectation={str(k): _coerce_timing(v, repairs=repairs,
                                                         where=f"profile:{aid}.resp:{k}")
                                  for k, v in (p.get("response_expectation") or {}).items()},
            latent_hypotheses=[{"state": str(h.get("state", ""))[:48],
                                "prior": max(0.0, min(1.0, float(h.get("prior", 0.5) or 0.5))),
                                "why": str(h.get("why", ""))[:120],
                                "source": str(h.get("source", ""))[:100]}
                               for h in (p.get("latent_hypotheses") or [])[:6]
                               if isinstance(h, dict) and h.get("state")],
            temporal_evidence=[str(x)[:140] for x in (p.get("temporal_evidence") or [])[:6]],
            unresolved=[str(x)[:140] for x in (p.get("unresolved") or [])[:6]])
        model.actor_profiles[str(aid)] = prof
    if model.actor_profiles:
        accepted.append(f"actor_profiles:{len(model.actor_profiles)}")


def _apply_critic_findings(model, findings, *, critic, repairs, accepted) -> int:
    """Typed application of critic repairs. Unknown ops are recorded as findings only."""
    n_applied = 0
    for f in findings[:16]:
        if not isinstance(f, dict):
            continue
        rec = {"critic": critic, "class": str(f.get("class", ""))[:48],
               "where": str(f.get("where", ""))[:120],
               "problem": str(f.get("problem", ""))[:200], "applied": False}
        rep = f.get("repair") if isinstance(f.get("repair"), dict) else None
        if rep:
            op, target, value = str(rep.get("op", "")), str(rep.get("target", "")), rep.get("value")
            try:
                if op == "remove_recurrence":
                    before = len(model.recurring_obligations)
                    model.recurring_obligations = [
                        r for r in model.recurring_obligations if r.get("rule_id") != target]
                    rec["applied"] = len(model.recurring_obligations) < before
                elif op == "demote_to_unresolved":
                    model.unresolved_mechanisms.append(
                        f"{target}: demoted by {critic} — {rec['problem']}"[:200])
                    rec["applied"] = True
                elif op == "add_deadline" and isinstance(value, dict) and value.get("label"):
                    model.deadlines.append({
                        "label": str(value["label"])[:120],
                        "timing": _coerce_timing(value.get("timing"), repairs=repairs,
                                                 where=f"critic:{target}"),
                        "binds": str(value.get("binds", ""))[:80],
                        "source": f"{critic}"})
                    rec["applied"] = True
                elif op == "add_dependency" and isinstance(value, dict):
                    if value.get("event") and value.get("depends_on"):
                        model.dependencies.append({"event": str(value["event"])[:80],
                                                   "depends_on": str(value["depends_on"])[:80],
                                                   "why": str(value.get("why", critic))[:120]})
                        rec["applied"] = True
                elif op == "add_continuous" and isinstance(value, dict) \
                        and value.get("process_id") and value.get("writes"):
                    form = str(value.get("form", "exponential_decay"))
                    if form in ("exponential_decay", "linear_drift", "exponential_approach"):
                        model.continuous_processes.append(ContinuousProcessSpec(
                            process_id=str(value["process_id"])[:60],
                            writes=str(value["writes"])[:80], form=form,
                            rate_per_day=float(value.get("rate_per_day", 0.0) or 0.0),
                            active_when=str(value.get("active_when", ""))[:80],
                            provenance=f"{critic}"))
                        rec["applied"] = True
                elif op == "adjust_timing":
                    # timing adjustments land as recorded uncertainty, not silent mutation of
                    # arbitrary paths — the honest middle ground for a free-path repair
                    model.temporal_uncertainties.append(
                        {"about": target, "why_unknown": rec["problem"],
                         "impact": f"critic-adjusted timing: {json.dumps(value)[:120]}",
                         "critic": critic})
                    rec["applied"] = True
                elif op in ("add_channel", "add_stage", "add_profile_field", "note_only"):
                    rec["applied"] = False                     # advisory; kept as finding
            except Exception as e:  # noqa: BLE001
                repairs.append(f"critic repair {op} failed: {type(e).__name__}")
        model.critic_findings.append(rec)
        n_applied += 1 if rec["applied"] else 0
    if n_applied:
        accepted.append(f"{critic}:applied:{n_applied}")
    return n_applied


def attach_temporal_model(plan, model: ScenarioTemporalModel) -> dict:
    """Attach the compiled model to the plan and schedule its EXACT structure:
      * deadlines with exact timestamps → real `deadline` events;
      * sourced recurring obligations → their real occurrences inside the window (§5-gated:
        the compiler refused sourceless recurrence, so everything here carries provenance);
      * institutional processes/stages are NOT pre-scheduled — stages activate on entry (§17);
      * continuous processes/channels/profiles ride on the plan for the runtime.
    Returns a report for lineage."""
    from swm.world_model_v2.temporal_calendar import RecurrenceRule
    from swm.world_model_v2.events import register_event_type, event_type_registered
    plan.temporal_model = model
    plan.provenance = {**(getattr(plan, "provenance", None) or {}),
                       "temporal_model_hash": model.temporal_model_hash()}
    n_deadline, n_rec = 0, 0
    for d in model.deadlines:
        t = d.get("timing") or {}
        if t.get("kind") == "exact" and isinstance(t.get("ts"), (int, float)) \
                and plan.as_of < float(t["ts"]) <= plan.horizon_ts:
            plan.scheduled_events.append({
                "etype": "deadline", "ts": float(t["ts"]), "participants": [],
                "payload": {"label": d.get("label"), "binds": d.get("binds"),
                            "source": d.get("source"), "provenance": "temporal_model"}})
            n_deadline += 1
    if not event_type_registered("recurring_obligation"):
        register_event_type("recurring_obligation", scheduling="scheduled",
                            reads=("entities",), deltas=(),
                            parameter_source="sourced real recurrence (temporal model)",
                            validated=True)
    for r in model.recurring_obligations:
        try:
            rule = RecurrenceRule(rule_id=str(r["rule_id"]), freq=str(r["freq"]),
                                  tz=str(r.get("tz", "UTC")),
                                  local_hour=float(r.get("local_hour", 9.0)),
                                  weekday=r.get("weekday"), month_day=r.get("month_day"),
                                  participants=tuple(r.get("participants") or ()),
                                  source=str(r.get("source", "")),
                                  confidence=float(r.get("confidence", 0.6)))
        except (TypeError, ValueError, KeyError):
            continue
        ts, n_here = plan.as_of, 0
        while n_here < 400:
            try:
                ts = rule.next_occurrence(ts)
            except ValueError:
                break
            if ts > plan.horizon_ts:
                break
            plan.scheduled_events.append({
                "etype": "recurring_obligation", "ts": float(ts),
                "participants": list(rule.participants),
                "payload": {"rule_id": rule.rule_id, "source": rule.source,
                            "tz": rule.tz, "recurrence": r.get("freq"),
                            "relevance": r.get("relevance"),
                            "confidence": rule.confidence, "provenance": "temporal_model"}})
            n_here += 1
            n_rec += 1
    return {"temporal_model_hash": model.temporal_model_hash(),
            "n_channels": len(model.channels), "n_actor_profiles": len(model.actor_profiles),
            "n_institutional_processes": len(model.institutional_processes),
            "n_continuous_processes": len(model.continuous_processes),
            "n_deadline_events": n_deadline, "n_recurrence_events": n_rec,
            "n_recurrences_accepted": len(model.recurring_obligations),
            "n_critic_findings": len(model.critic_findings),
            "n_llm_calls": len(model.compilation_trace),
            "degraded": model.degraded or None,
            "unresolved": model.unresolved_mechanisms[:8]}
