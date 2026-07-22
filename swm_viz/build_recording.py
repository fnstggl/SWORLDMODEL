"""Assemble a unified, replayable ``recording`` from a real Lean V2 run.

Inputs are the verbatim capture (``swm_viz.recorder.Capture``) and the ``SimulationResult`` of
a real ``execution_profile="lean_v2"`` run. Everything here is a faithful *projection* of data
the simulation actually produced — the compiled world, the counted grounding, each actor's
modelled private-state hypotheses and their grounded weights, every LLM call verbatim, the
ordered decision/deliberation/vote trace, and the forecast decomposition. Nothing is invented.

The output is a single JSON-able dict with:
  meta, cast, institution, resolution, shared_conditions, outcome_reference_class,
  actor_states, relationships, llm_calls (full prompt+reply, attributed & classified),
  events (the ordered playback timeline), forecast, unresolved, preflight, challenger,
  budget, raw_provenance.
"""
from __future__ import annotations

import re

_YOU_ARE = re.compile(r"YOU ARE:\s*(.+?)\s+[—\-–]\s+", re.IGNORECASE)

# ---- stage/prompt -> (sub_type, phase) --------------------------------------------------
_STAGE_PHASE = {
    "structural_generation": ("blueprint", "compile"),
    "structural_compile": ("blueprint", "compile"),
    "targeted_repair": ("blueprint_repair", "compile"),
    "consequence_compile": ("consequence_compile", "compile"),
    "reference_class_grounding": ("grounding", "grounding"),
    "state_generation": ("state_generation", "states"),
    "schema_format_repair": ("schema_repair", "waves"),
    "actor_decision": ("decision", "waves"),  # refined by prompt content below
}


def _loose_json(text: str):
    """Best-effort JSON parse of a model reply (tolerant of prose around the object)."""
    import json
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        frag = text[s:e + 1]
        try:
            return json.loads(frag)
        except Exception:  # noqa: BLE001
            try:
                from swm.engine.grounding import parse_json
                return parse_json(text)
            except Exception:  # noqa: BLE001
                return None
    return None


def _classify(call: dict) -> tuple[str, str]:
    stage = call.get("stage", "")
    sub, phase = _STAGE_PHASE.get(stage, (stage or "call", "waves"))
    if stage == "actor_decision":
        p = call.get("prompt", "")
        if "SAME person continuing the SAME moment" in p:
            sub = "deliberation"
        elif "previous reply failed validation" in p:
            sub = "decision_reask"
        else:
            sub = "decision"
    return sub, phase


def _actor_from_prompt(prompt: str, cast_index: dict) -> tuple[str | None, str | None]:
    """Attribute a decision/deliberation prompt to an actor via its 'YOU ARE: <name>' line."""
    m = _YOU_ARE.search(prompt or "")
    if not m:
        return None, None
    name = m.group(1).strip()
    key = name.lower()
    if key in cast_index:
        return cast_index[key], name
    # relaxed match: name contains an actor's name/id token, or vice-versa
    for k, aid in cast_index.items():
        if k and (k in key or key in k):
            return aid, name
    return None, name


def build_recording(*, meta: dict, calls: list, blueprint_dict: dict,
                    result_dict: dict) -> dict:
    """Assemble the recording from plain, already-serialized data.

    ``calls`` is ``Capture.calls``; ``blueprint_dict`` is ``blueprint.as_dict()`` (or {});
    ``result_dict`` is the SimulationResult as a dict (scalar fields + ``provenance``). Taking
    dicts (not live objects) means a recording can be rebuilt from a saved raw dump without
    re-running the simulation.
    """
    prov_all = result_dict.get("provenance") or {}
    lv2 = prov_all.get("lean_v2") or {}
    bp_dict = blueprint_dict or {}

    grounding = lv2.get("grounding") or {}
    slice_m = lv2.get("slice") or {}
    actor_states_prov = lv2.get("actor_states") or {}
    state_post = lv2.get("state_posteriors") or {}
    unknown_mass = lv2.get("unknown_state_mass") or {}
    eng = lv2.get("engine_primary") or {}
    eng_ch = lv2.get("engine_challenger") or {}
    weight_law = eng.get("grounded_weight_law") or {}
    weight_ivals = eng.get("grounded_weight_intervals") or {}

    kept_actors = set(slice_m.get("kept_actors") or list(actor_states_prov.keys()))

    # -- cast -----------------------------------------------------------------------------
    cast = []
    cast_index: dict[str, str] = {}
    bp_actors = bp_dict.get("actors") or []
    if not bp_actors:
        # fall back to whatever the provenance exposes (ids only)
        bp_actors = [{"id": aid, "name": aid} for aid in actor_states_prov.keys()]
    for a in bp_actors:
        aid = str(a.get("id") or "")
        name = str(a.get("name") or aid)
        cast.append({
            "id": aid,
            "name": name,
            "role": a.get("role") or "",
            "authority": a.get("authority") or [],
            "discretion": a.get("discretion") or "",
            "aliases": a.get("aliases") or [],
            "kept": (aid in kept_actors) if kept_actors else True,
            "n_variants": len(a.get("private_state_variants") or []),
        })
        cast_index[name.lower()] = aid
        cast_index[aid.lower()] = aid
        for al in (a.get("aliases") or []):
            cast_index[str(al).lower()] = aid

    # -- institution + resolution + vote options ------------------------------------------
    institutions = bp_dict.get("institutions") or []
    resolution = bp_dict.get("resolution") or {}
    terminal = bp_dict.get("terminal") or {}
    vote_options = _vote_options(bp_dict)
    inst = {}
    if institutions:
        i0 = institutions[0]
        inst = {
            "id": i0.get("id") or "",
            "name": i0.get("name") or i0.get("id") or "institution",
            "members": i0.get("members") or [],
            "decision_rule": i0.get("decision_rule") or terminal.get("decision_rule") or "",
            "rule_params": i0.get("rule_params") or terminal.get("rule_params") or {},
            "procedure": i0.get("procedure") or [],
            "vote_options": vote_options,
        }

    # -- shared world conditions ----------------------------------------------------------
    shared_conditions = []
    for cid, sc in (grounding.get("shared_world_conditions") or {}).items():
        tbl = sc.get("table") or {}
        shared_conditions.append({
            "id": cid,
            "claim": sc.get("claim") or "",
            "states": sc.get("states") or [],
            "affects_actors": sc.get("affects_actors") or [],
            "rate": tbl.get("rate"),
            "n": tbl.get("n"),
            "interval": tbl.get("interval"),
        })
    outcome_rc = grounding.get("outcome_reference_class") or {}

    # -- actor states (the modelled private realities) + grounded weights ------------------
    actor_states = {}
    rel_edges = {}
    for aid, hyps in actor_states_prov.items():
        law = weight_law.get(aid) or {}
        ivals = weight_ivals.get(aid) or {}
        rows = []
        for h in hyps:
            sid = h.get("state_id") or ""
            wmid = law.get(sid)
            rng = ivals.get(sid)
            rows.append({
                "state_id": sid,
                "claim": h.get("claim") or "",
                "beliefs": h.get("beliefs") or [],
                "goals": h.get("goals") or [],
                "stances": h.get("stances") or [],
                "commitments": h.get("commitments") or [],
                "pressures": h.get("pressures") or "",
                "relationships": h.get("relationships") or {},
                "action_if_state": h.get("action_if_state") or "",
                "aligned_condition": h.get("aligned_condition") or {},
                "reversal_capable": bool(h.get("reversal_capable")),
                "supporting_evidence_ids": h.get("supporting_evidence_ids") or [],
                "weight_mid": wmid,
                "weight_range": rng,
            })
            for tgt, desc in (h.get("relationships") or {}).items():
                tid = cast_index.get(str(tgt).lower(), str(tgt))
                rel_edges.setdefault((aid, tid), desc)
        actor_states[aid] = {
            "hypotheses": rows,
            "unknown_mass": unknown_mass.get(aid),
        }
    relationships = [{"source": s, "target": t, "descriptor": d}
                     for (s, t), d in rel_edges.items() if s != t]

    # -- LLM calls: verbatim, attributed, classified --------------------------------------
    llm_calls = []
    for c in calls:
        sub, phase = _classify(c)
        actor_id = actor_name = None
        parsed = None
        if sub in ("decision", "deliberation", "decision_reask"):
            actor_id, actor_name = _actor_from_prompt(c.get("prompt", ""), cast_index)
            parsed = _decision_summary(_loose_json(c.get("reply", "")))
        elif sub in ("blueprint", "state_generation", "grounding"):
            parsed = None
        llm_calls.append({
            "seq": c["seq"],
            "stage": c["stage"],
            "sub_type": sub,
            "phase": phase,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "prompt": c["prompt"],
            "reply": c["reply"],
            "prompt_chars": c["prompt_chars"],
            "reply_chars": c["reply_chars"],
            "latency_s": c["latency_s"],
            "tier": c["tier"],
            "retried": c["retried"],
            "t_start": c["t_start"],
            "error": c.get("error", False),
            "parsed": parsed,
        })

    # -- forecast decomposition + final result --------------------------------------------
    fd = lv2.get("forecast_decomposition") or {}
    unresolved = lv2.get("unresolved") or {}
    preflight = lv2.get("preflight") or {}
    challenger = lv2.get("challenger") or {}

    forecast = {
        "headline_probability": result_dict.get("raw_probability"),
        "probability_source": result_dict.get("probability_source", ""),
        "conditional_on_resolved": result_dict.get("probability_conditional_on_resolved"),
        "uncertainty_interval": result_dict.get("uncertainty_interval"),
        "grounded_prior": fd.get("grounded_prior"),
        "simulation_conditional": fd.get("simulation_conditional"),
        "combined": fd.get("combined"),
        "method": fd.get("method"),
        "disagreement": fd.get("disagreement"),
        "notes": fd.get("notes") or [],
    }

    meta_out = {
        **meta,
        "status": result_dict.get("simulation_status", ""),
        "grounding_grade": result_dict.get("grounding_grade", ""),
        "confidence": result_dict.get("confidence", ""),
        "unresolved_mass": result_dict.get("unresolved_mass"),
        "weight_sensitive": result_dict.get("weight_sensitive"),
        "latency_s": result_dict.get("latency_s"),
        "causal_thesis": bp_dict.get("causal_thesis") or (lv2.get("blueprint") or {})
        .get("causal_thesis") or "",
        "n_llm_calls": len(llm_calls),
        "limitations": list(result_dict.get("limitations") or []),
    }

    # -- the ordered playback timeline ----------------------------------------------------
    events = _build_events(
        llm_calls=llm_calls, cast=cast, inst=inst, resolution=resolution,
        shared_conditions=shared_conditions, outcome_rc=outcome_rc,
        actor_states=actor_states, preflight=preflight, eng=eng, eng_ch=eng_ch,
        challenger=challenger, forecast=forecast, unresolved=unresolved,
        meta=meta_out, cast_index=cast_index, vote_options=vote_options)

    return {
        "meta": meta_out,
        "cast": cast,
        "institution": inst,
        "resolution": resolution,
        "shared_conditions": shared_conditions,
        "outcome_reference_class": outcome_rc,
        "actor_states": actor_states,
        "relationships": relationships,
        "llm_calls": llm_calls,
        "events": events,
        "forecast": forecast,
        "unresolved": unresolved,
        "preflight": preflight,
        "challenger": challenger,
        "budget": lv2.get("budget") or {},
        "gateway_manifest": lv2.get("gateway") or {},
        "decision_trace": eng.get("decision_trace") or [],
        "deliberations": eng.get("deliberations") or [],
        "raw_provenance": lv2,
    }


# ---------------------------------------------------------------------------- helpers
def _vote_options(bp_dict: dict) -> list:
    """Every real option a voter could choose, read from record_vote action templates."""
    opts = []
    for t in bp_dict.get("action_templates") or []:
        for eff in t.get("effects") or []:
            if eff.get("kind") == "record_vote":
                for o in (eff.get("params") or {}).get("options") or []:
                    if str(o) not in opts:
                        opts.append(str(o))
    if not opts:
        rp = (bp_dict.get("terminal") or {}).get("rule_params") or {}
        if rp.get("option"):
            opts = [str(rp["option"])]
    return opts


def _decision_summary(reply_obj) -> dict | None:
    if not isinstance(reply_obj, dict):
        return None
    dec = reply_obj.get("decision") or {}
    interp = reply_obj.get("interpretation") or {}
    return {
        "chosen_action": dec.get("chosen_action") or dec.get("act_or_wait") or "",
        "act_or_wait": dec.get("act_or_wait") or "",
        "vote_option": dec.get("vote_option") or "",
        "intended_effect": dec.get("intended_effect") or "",
        "summary": reply_obj.get("decision_summary")
        or reply_obj.get("reflection_summary") or "",
        "what_happened": interp.get("what_happened") or "",
        "unresolved_ambiguity": interp.get("unresolved_ambiguity") or "",
        "changed": reply_obj.get("changed"),
    }


def _primary_world_prefix(decision_trace: list) -> str:
    """The node prefix (e.g. 'w0_sw0') carrying the most decisions — the MAP world."""
    from collections import Counter
    counts = Counter()
    for d in decision_trace:
        node = str(d.get("node") or "")
        counts[node.split(".")[0]] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _build_events(*, llm_calls, cast, inst, resolution, shared_conditions, outcome_rc,
                  actor_states, preflight, eng, eng_ch, challenger, forecast, unresolved,
                  meta, cast_index, vote_options) -> list:
    events = []
    n = [0]

    def add(etype, **payload):
        events.append({"i": n[0], "type": etype, **payload})
        n[0] += 1

    by_phase = {}
    for c in llm_calls:
        by_phase.setdefault(c["phase"], []).append(c)

    def call_step(c, **extra):
        base = dict(seq=c["seq"], sub_type=c["sub_type"], phase=c["phase"],
                    actor_id=c["actor_id"], actor_name=c["actor_name"],
                    latency_s=c["latency_s"], tier=c["tier"],
                    prompt_chars=c["prompt_chars"], reply_chars=c["reply_chars"],
                    parsed=c.get("parsed"))
        base.update(extra)          # explicit per-call fields (decision/actor overrides) win
        add("llm_call", **base)

    # ---- compile ----
    add("phase", phase="compile", title="Compile the causal world")
    for c in by_phase.get("compile", []):
        call_step(c)
    add("world_ready", cast=[{"id": a["id"], "name": a["name"], "role": a["role"],
                              "kept": a["kept"]} for a in cast],
        thesis=meta.get("causal_thesis", ""),
        institution=inst, resolution=resolution, vote_options=vote_options)

    # ---- grounding ----
    add("phase", phase="grounding", title="Ground in counted reference classes")
    for c in by_phase.get("grounding", []):
        call_step(c)
    for sc in shared_conditions:
        add("condition", **sc)
    if outcome_rc:
        prov = outcome_rc.get("provenance") or {}
        add("outcome_class", quantity=outcome_rc.get("quantity"),
            rate=prov.get("rate_mean"), n=prov.get("denominator"),
            interval=outcome_rc.get("interval"))

    # ---- states ----
    add("phase", phase="states", title="Model each actor's private realities")
    for c in by_phase.get("states", []):
        call_step(c)
    for a in cast:
        st = actor_states.get(a["id"])
        if st:
            add("actor_states", actor_id=a["id"], actor_name=a["name"],
                hypotheses=st["hypotheses"], unknown_mass=st.get("unknown_mass"))

    # ---- preflight ----
    add("phase", phase="preflight", title="Prove the world can answer")
    add("preflight", verdict=preflight.get("verdict"),
        reached_terminal=(preflight.get("probe") or {}).get("reached_terminal"),
        blocking=preflight.get("blocking") or [])

    # ---- waves (the society acts) ----
    add("phase", phase="waves", title="Weighted causal waves — the society decides")
    decision_trace = eng.get("decision_trace") or []
    prefix = _primary_world_prefix(decision_trace)
    members_total = len(inst.get("members") or cast)

    # 1) every real moment of cognition, in captured order (decisions + deliberations + reasks)
    first_call_seq: dict[str, int] = {}
    for c in [x for x in llm_calls if x["phase"] == "waves"]:
        aid = c["actor_id"]
        aname = c["actor_name"] or next((a["name"] for a in cast if a["id"] == aid), aid)
        if c["sub_type"] == "decision":
            p = c.get("parsed") or {}
            if aid and aid not in first_call_seq:
                first_call_seq[aid] = c["seq"]
            call_step(c, decision=True, actor_id=aid, actor_name=aname,
                      chosen=p.get("chosen_action"), vote_option=p.get("vote_option"),
                      act_or_wait=p.get("act_or_wait"))
        else:  # deliberation / decision_reask / schema_repair
            call_step(c, actor_id=aid, actor_name=aname)

    # 2) the MAP-world vote tally forms — one authoritative vote per member
    tally: dict[str, str] = {}
    seen_actor = set()
    primary = [d for d in decision_trace if str(d.get("node", "")).startswith(prefix)]
    for d in primary:
        aid = d.get("actor")
        vote = (d.get("vote_option") or "").strip()
        if not aid or aid in seen_actor:
            continue
        seen_actor.add(aid)
        if not vote:
            continue  # a non-vote decision was already shown as its own cognition step
        tally[aid] = vote
        aname = next((a["name"] for a in cast if a["id"] == aid), aid)
        add("vote", actor_id=aid, actor_name=aname, vote_option=vote,
            variant=d.get("variant"), call_seq=first_call_seq.get(aid),
            tally=_tally_counts(tally), members_total=members_total)

    add("terminal",
        yes_mass=(eng.get("coalescer") or {}).get("yes_mass"),
        decision_rule=inst.get("decision_rule"),
        tally=_tally_counts(tally), members_total=members_total)

    # ---- challenger (only if it ran) ----
    if challenger.get("triggered"):
        add("phase", phase="challenger", title="Conditional structural challenger")
        for c in llm_calls:
            if c["phase"] == "challenger":
                call_step(c)
        add("challenger_result", spread=(eng_ch.get("coalescer") or {}).get("yes_mass"),
            reason=challenger.get("reason") or challenger.get("trigger"))

    # ---- forecast + result ----
    add("phase", phase="forecast", title="Decompose & report the forecast")
    add("forecast", **forecast)
    add("result", status=meta.get("status"), headline=forecast.get("headline_probability"),
        source=forecast.get("probability_source"),
        unresolved_mass=meta.get("unresolved_mass"),
        limitations=meta.get("limitations", []),
        by_cause=(unresolved.get("by_cause") or {}))
    return events


def _tally_counts(tally: dict) -> dict:
    out: dict[str, int] = {}
    for v in tally.values():
        out[v] = out.get(v, 0) + 1
    return out
