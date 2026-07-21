"""World-level forensic capture + human-readable rendering.

`dump_search_forensics` walks the matched arms AFTER a run and serializes, per candidate per
particle, what actually happened in that world: every semantic event with its exact content,
every actor's information exposures, the plan's own execution trace, every state-delta with
its reason codes and recorded decision summaries, and the goal-contract row. Nothing is
summarized into counters — the actual content is preserved so a human can inspect whether
the simulation makes real-world sense.

`render_forensic_md` composes the step-by-step FORENSIC.md for one example from the dumped
worlds + the role trace + the actor trace + the scenario report.
"""
from __future__ import annotations

import json
import os


def _deltas(branch) -> list:
    out = []
    for d in branch.log:
        row = {"at": getattr(d, "at", None), "operator": getattr(d, "operator", ""),
               "event_type": getattr(d, "event_type", ""),
               "reason_codes": list(getattr(d, "reason_codes", []))[:8],
               "changes": [{"path": str(c.get("path", c) if isinstance(c, dict) else c)[:160],
                            "after": str(c.get("after", ""))[:200]
                            if isinstance(c, dict) else ""}
                           for c in _change_rows(d)][:16]}
        unc = getattr(d, "uncertainty", None)
        if unc:
            row["notes"] = {k: (v if k == "kernel_quarantined" else str(v)[:240])
                            for k, v in unc.items()
                            if k in ("executed_action", "decision_summary", "stop_condition",
                                     "step_intent", "kernel_quarantined",
                                     "consequence_compiler")}
        out.append(row)
    return out


def _change_rows(d) -> list:
    ch = getattr(d, "changes", []) or []
    rows = []
    for c in ch:
        if isinstance(c, dict):
            rows.append(c)
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            rows.append({"path": c[0], "before": c[1], "after": c[2]})
        else:
            rows.append({"path": str(c)})
    return rows


def _exposures(world) -> dict:
    info = getattr(world, "information", None)
    out = {}
    if info is None:
        return out
    for actor in (getattr(world, "entities", {}) or {}):
        try:
            rows = []
            for item, exp in info.visible_to(actor, at=world.clock.now):
                rows.append({"item_id": getattr(item, "item_id", ""),
                             "content": str(getattr(item, "content", ""))[:400],
                             "source": getattr(item, "source", ""),
                             "observed_at": getattr(exp, "at", None)})
            if rows:
                out[actor] = rows
        except Exception:  # noqa: BLE001
            continue
    return out


def _records(world) -> list:
    return [{"record_id": o.object_id, "type": o.object_type, "status": o.status,
             "created_by": o.created_by, "fields": {k: (str(v)[:200]) for k, v in
                                                    o.attributes.items()},
             "visibility": o.visibility}
            for o in (getattr(world, "objects", {}) or {}).values()]


def dump_search_forensics(search, goal, out_dir: str, *, particles_per_candidate: int = 2):
    """Serialize the matched worlds' actual content. `particles_per_candidate` bounds file
    size; the count actually dumped is recorded so truncation is visible, never silent."""
    os.makedirs(out_dir, exist_ok=True)
    arms = getattr(search, "_arms", {}) or {}
    manifest = {"candidates": sorted(arms), "particles_per_arm": None,
                "particles_dumped_per_arm": particles_per_candidate,
                "note": "per-world content below is exact, not summarized; remaining "
                        "particles differ only by matched-world randomness"}
    path = os.path.join(out_dir, "forensic_worlds.jsonl")
    with open(path, "w") as f:
        for cid, arm in arms.items():
            manifest["particles_per_arm"] = len(arm.branches)
            evals = (search.report.evaluations or {}).get(cid, {})
            per_particle = evals.get("per_particle", [])
            for i, branch in enumerate(arm.branches[:particles_per_candidate]):
                world = branch.world
                from swm.world_model_v2.phase13.scenario_actions.execution import \
                    plan_execution_trace
                row = {"candidate_id": cid, "particle": i,
                       "semantic_events": [
                           {"type": e.get("semantic_type_id"),
                            "source": e.get("source_actor_id"),
                            "targets": e.get("direct_targets"),
                            "exact_content": str(e.get("exact_content", ""))[:600],
                            "visibility": e.get("intended_visibility"),
                            "at": e.get("occurred_at")}
                           for e in (getattr(world, "semantic_log", []) or [])],
                       "information_exposures": _exposures(world),
                       "records": _records(world),
                       "plan_execution": plan_execution_trace(world, cid),
                       "delta_log": _deltas(branch),
                       "goal_row": (per_particle[i] if i < len(per_particle) else {})}
                f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(out_dir, "forensic_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return path


# ---------------------------------------------------------------- human-readable rendering
def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def render_forensic_md(example_dir: str, *, title: str) -> str:
    """Compose FORENSIC.md from result.json + role_trace.jsonl + actor_trace.jsonl +
    forensic_worlds.jsonl. Every section quotes ACTUAL content."""
    res = json.load(open(os.path.join(example_dir, "result.json")))["result"]
    sr = res["provenance"]["scenario_report"]
    roles = _load_jsonl(os.path.join(example_dir, "role_trace.jsonl"))
    actor_rows = _load_jsonl(os.path.join(example_dir, "actor_trace.jsonl"))
    worlds = _load_jsonl(os.path.join(example_dir, "forensic_worlds.jsonl"))
    L = []
    a = L.append
    a(f"# Forensic trace — {title}\n")
    a("## 1. Decision contract\n```json\n"
      + json.dumps(json.load(open(os.path.join(example_dir, "contract.json"))), indent=1)
      + "\n```\n")
    a(f"## 2. Stated goal & missing preferences\n- goal: {sr.get('stated_goal', '')}\n"
      f"- missing preferences / unresolved tradeoffs: "
      f"{json.dumps(sr.get('missing_preferences', []))}\n"
      f"- goal predicates:\n```json\n"
      + json.dumps(sr["goal_contract"].get("predicates", []), indent=1)[:2400] + "\n```\n")
    a("## 3. Scenario-specific action language\n```json\n"
      + json.dumps(sr.get("action_language_summary", {}), indent=1)[:2600] + "\n```\n")
    a("## 4-5. Every candidate generated, and why\n")
    for c in sr.get("candidates", []):
        anc = sr.get("candidate_ancestry", {}).get(c["candidate_id"], {})
        a(f"### {c['candidate_id']} — {c.get('title', '')}\n"
          f"- proposed by: {anc.get('source', c.get('source'))}"
          + (f" (revision of {anc.get('parents')}: {anc.get('revision_reason')})"
             if anc.get("parents") else "")
          + f"\n- causal theory: {c.get('causal_theory', '')}\n")
        for s in c.get("steps", []):
            a(f"  - **{s.get('step_id')}**: {s.get('intent', '')}\n"
              f"    - targets {s.get('target_ids')} | channel {s.get('channel') or '—'} | "
              f"visibility {s.get('visibility')} | timing {s.get('timing_ts')}\n"
              + (f"    - exact content: “{s.get('exact_content')[:400]}”\n"
                 if s.get("exact_content") else "")
              + (f"    - conditions: {[cc.get('description') for cc in s.get('conditions', [])]}\n"
                 if s.get("conditions") else ""))
    a("## 6. Feasibility rejections (typed, exact)\n```json\n"
      + json.dumps(sr.get("rejected", []), indent=1)[:2400] + "\n```\n")
    a("## 7. Compiled direct effects (kernel ops per surviving step)\n```json\n"
      + json.dumps(sr.get("compiled_effects", {}), indent=1)[:4200] + "\n```\n")
    a("## 8-11. Per-world causal record (events → observations → actor choices → state)\n")
    for w in worlds[:8]:
        a(f"### {w['candidate_id']} — particle {w['particle']}\n")
        a("**Semantic events (exact content):**\n")
        for e in w["semantic_events"][:14]:
            a(f"- t={e.get('at')}: `{e.get('type')}` by {e.get('source')} → "
              f"{e.get('targets')} [{e.get('visibility')}]: “{e.get('exact_content')[:260]}”\n")
        a("\n**Who observed what:**\n")
        for actor, rows in list(w["information_exposures"].items())[:6]:
            for r in rows[:5]:
                a(f"- {actor} ← ({r.get('source')}): “{str(r.get('content'))[:200]}”\n")
        a("\n**Actor invocations & choices (from the delta log):**\n")
        for d in w["delta_log"]:
            notes = d.get("notes") or {}
            if notes.get("executed_action") or notes.get("decision_summary"):
                a(f"- {d['operator']}: {json.dumps(notes)[:300]}\n")
        a("\n**Resulting records (world state):**\n")
        for r in w["records"][:12]:
            a(f"- `{r['record_id']}` ({r['type']}/{r['status']}, by {r['created_by']}): "
              f"{json.dumps(r['fields'])[:220]}\n")
        a(f"\n**Plan execution here:** {json.dumps(w['plan_execution'])[:300]}\n")
        a(f"**Goal row:** success={w['goal_row'].get('success')}, "
          f"forbidden={w['goal_row'].get('forbidden_hit')}, "
          f"predicates={json.dumps(w['goal_row'].get('predicates', {}))[:240]}\n\n")
    a("## 12-13. Where each strategy first succeeded/failed (diagnosis)\n```json\n"
      + json.dumps(sr.get("trajectory_summaries", {}), indent=1)[:3600] + "\n```\n")
    a("## 14-15. Revisions and their fate\n```json\n"
      + json.dumps(sr.get("revisions", []), indent=1)[:1600]
      + "\n```\nRevision children appear in §4 with ancestry; a revision that worsened "
        "forbidden-state frequency is listed in §6 with code revision_worsened_forbidden.\n")
    a("## 16. Matched comparison between finalists\n```json\n"
      + json.dumps({k: v for k, v in sr.get("evaluations", {}).items()}, indent=1)[:3200]
      + "\n```\n")
    a(f"## 17. Final verdict\n- recommendation_kind: **{res.get('recommendation_kind')}** | "
      f"recommended: **{res.get('recommended')}**\n"
      f"- distinguishable finalists: {sr.get('finalists_distinguishable')}\n"
      f"- Pareto set: {sr.get('pareto')}\n"
      f"- adjudicator synthesis (blind): "
      f"{json.dumps((sr.get('trace_summary') or {}).get('by_role', {}))}\n"
      f"- support claim: best-supported among the considered feasible actions under the "
      f"stated goal, constraints, world hypotheses, and simulation support\n")
    a("## 18. Assumptions that could reverse the result\n"
      + "".join(f"- {x}\n" for x in (sr.get("reversal_conditions") or ["none recorded"])))
    a(f"\n## 19. Cost, coverage, approximation limits\n"
      f"- particles/arm: {sr.get('simulation_coverage', {}).get('n_particles_per_arm')} | "
      f"simulated arms: {sr.get('simulation_coverage', {}).get('n_simulated_arms')}\n"
      f"- LLM calls: planner/critic roles {sr.get('trace_summary', {}).get('n_llm_calls')} "
      f"+ actor-simulation calls {len(actor_rows)}\n"
      f"- latency_s: {res.get('latency_s')}\n"
      f"- stop reason: {sr.get('stop_reason')}\n"
      f"- unresolved semantics: {json.dumps(sr.get('unresolved_semantics', []))[:400]}\n"
      f"- forensic truncation: per-arm worlds dumped = "
      f"{json.load(open(os.path.join(example_dir, 'forensic_manifest.json')))['particles_dumped_per_arm'] if os.path.exists(os.path.join(example_dir, 'forensic_manifest.json')) else 'n/a'} "
      f"of {json.load(open(os.path.join(example_dir, 'forensic_manifest.json')))['particles_per_arm'] if os.path.exists(os.path.join(example_dir, 'forensic_manifest.json')) else 'n/a'}\n")
    a("\n## Raw traces\n- every planner/critic/adjudicator LLM call: `role_trace.jsonl`\n"
      "- every actor-simulation LLM call (what each invoked actor was SHOWN and ANSWERED, "
      "verbatim): `actor_trace.jsonl`\n- complete per-world dumps: `forensic_worlds.jsonl`\n")
    out = os.path.join(example_dir, "FORENSIC.md")
    with open(out, "w") as f:
        f.write("".join(L))
    return out
