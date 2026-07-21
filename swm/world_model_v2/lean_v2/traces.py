"""Full machine-readable trace + human-readable report writer (§14).

Writes to experiments/results/lean_v2_accuracy/<question_id>/: llm_calls.jsonl, shared_worlds.jsonl,
actor_states.jsonl, actor_decisions.jsonl, world_trajectories.jsonl, weight_provenance.json,
forecast_decomposition.json — plus report.md. Only prompts, raw outputs, model-produced
structured summaries, decisions, evidence links and execution traces are stored; no hidden
chain-of-thought is fabricated."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("experiments/results/lean_v2_accuracy")


def _wl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def write_traces(qid: str, *, gateway_rows, lean_v2_prov: dict, result_dict: dict) -> str:
    d = BASE / qid
    d.mkdir(parents=True, exist_ok=True)

    _wl(d / "llm_calls.jsonl", [
        {"call_id": i, "stage": r.get("stage"), "tier": r.get("tier"),
         "prompt_chars": r.get("prompt_chars"), "reply_chars": r.get("reply_chars"),
         "latency_s": r.get("latency_s"), "retried": r.get("retried")}
        for i, r in enumerate(gateway_rows or [])])

    shared = ((lean_v2_prov.get("grounding") or {}).get("shared_world_conditions") or {})
    _wl(d / "shared_worlds.jsonl",
        [{"condition_id": cid, **{k: sc.get(k) for k in
                                  ("claim", "affects_actors", "states", "evidence_ids")},
          "table": sc.get("table")} for cid, sc in shared.items()])

    states = (lean_v2_prov.get("actor_states") or {})
    _wl(d / "actor_states.jsonl",
        [{"actor_id": aid, **row} for aid, rows in states.items() for row in rows])

    eng = lean_v2_prov.get("engine_primary") or {}
    _wl(d / "actor_decisions.jsonl", eng.get("decision_trace") or [])

    _wl(d / "world_trajectories.jsonl", eng.get("node_audit_full")
        or (result_dict.get("resolution_report") or {}).get("per_node") or [])

    (d / "weight_provenance.json").write_text(json.dumps({
        "grounding": lean_v2_prov.get("grounding"),
        "state_posteriors": lean_v2_prov.get("state_posteriors"),
        "shared_condition_worlds": lean_v2_prov.get("shared_condition_worlds"),
        "unknown_state_mass": lean_v2_prov.get("unknown_state_mass"),
        "dependence": lean_v2_prov.get("dependence"),
        "no_label_derived_weights_invariant":
            lean_v2_prov.get("weight_invariant")}, indent=1, default=str))

    (d / "forecast_decomposition.json").write_text(json.dumps(
        lean_v2_prov.get("forecast_decomposition") or {}, indent=1, default=str))

    report = render_report(qid, lean_v2_prov=lean_v2_prov, result_dict=result_dict)
    (d / "report.md").write_text(report)
    return str(d)


def render_report(qid: str, *, lean_v2_prov: dict, result_dict: dict) -> str:
    fd = lean_v2_prov.get("forecast_decomposition") or {}
    eng = lean_v2_prov.get("engine_primary") or {}
    bud = lean_v2_prov.get("budget") or {}
    bp = lean_v2_prov.get("blueprint") or {}
    unresolved = lean_v2_prov.get("unresolved") or {}
    L = []
    L.append(f"# Lean V2 accuracy run — {qid}\n")
    L.append(f"**Question:** {result_dict.get('question', '')}\n")
    L.append(f"**Status:** {result_dict.get('simulation_status')} | "
             f"**probability:** {result_dict.get('raw_probability')} | "
             f"**source:** {result_dict.get('probability_source')} | "
             f"**grounding:** {result_dict.get('grounding_grade')} | "
             f"**confidence:** {result_dict.get('confidence')}\n")
    L.append("\n## Causal world\n")
    L.append(f"- thesis: {bp.get('causal_thesis', '')}")
    L.append(f"- actors: {bp.get('n_actors')} | action templates: "
             f"{bp.get('n_action_templates')}\n")
    L.append("## Shared world conditions\n")
    for cid, sc in ((lean_v2_prov.get("grounding") or {})
                    .get("shared_world_conditions") or {}).items():
        tbl = sc.get("table") or {}
        L.append(f"- **{cid}**: {sc.get('claim', '')} — counted rate "
                 f"{tbl.get('rate')} (n={tbl.get('n')})")
    L.append("\n## Forecast decomposition\n")
    L.append(f"- grounded prior: {fd.get('grounded_prior', {}).get('p')} "
             f"(n={fd.get('grounded_prior', {}).get('n')}, "
             f"source={fd.get('grounded_prior', {}).get('source')})")
    L.append(f"- simulation-conditional: {fd.get('simulation_conditional', {}).get('p')} "
             f"(resolved mass {fd.get('simulation_conditional', {}).get('resolved_mass')})")
    L.append(f"- combined: {fd.get('combined')} via {fd.get('method')}")
    L.append(f"- prior/simulation disagreement: {fd.get('disagreement')}")
    for n in fd.get("notes") or []:
        L.append(f"  - {n}")
    L.append("\n## Unresolved mass by cause\n")
    for c, m in (unresolved.get("by_cause") or {}).items():
        L.append(f"- {c}: {m} — {unresolved.get('treatments', {}).get(c, '')}")
    L.append("\n## Cost\n")
    L.append(f"- calls: {bud.get('calls')} | wall: {bud.get('wall_s')}s | "
             f"peak nodes: {bud.get('peak_weighted_nodes')}")
    L.append(f"- deliberations: {len(eng.get('deliberations') or [])} | "
             f"challenger: {lean_v2_prov.get('challenger', {}).get('triggered')}")
    for lim in (result_dict.get("limitations") or [])[:8]:
        L.append(f"- limitation: {lim}")
    return "\n".join(L) + "\n"
