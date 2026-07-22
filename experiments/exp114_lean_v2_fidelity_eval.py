"""EXP-114: the Lean V2 REAL-WORLD-FIDELITY five-question evaluation (PR #134).

The same five frozen BTF-3 questions as EXP-112/113 (Banxico → BoJ → visionOS → Wale → Hormuz),
sequentially, through the canonical `unified_runtime.simulate_world(..., "lean_v2")` — now carrying
the full fidelity architecture D1–D18: typed canonical options/resolution (D1,D4,D5,D6), reference-
class direction guard (D2), no alphabetical fallback (D3), faithful institution representation with
no threshold rescaling (D7), action-grounded weighting (D8), mindset↔external-event separation (D9),
verified reference cases (D10), canonical fact store (D11), shared-condition graph + tail
preservation (D12), actor knowledge packets (D13), deliberative convergence (D14), conservative
decision cache (D15), dimensional outcome mechanism (D16), structural-fidelity readiness (D17), and
self-contained traces (D18).

PHASE C (structural dry-audit): each question's compiled world is audited for STRUCTURE — actors,
institution size / seat power / roles, terminal variable+units+threshold, evidence coverage,
knowledge packets + leakage, action-mass weighting, deliberation stages, the outcome mechanism's
dimension, and the structural-fidelity verdict — printed and frozen BEFORE any outcome is joined.

§19 (cold-cache rerun): the persistent cache is CLEARED per question; one question at a time,
foreground; a HARD guard of 15 min / 150 calls (a trip finalizes the best labeled forecast, never a
relaunch). No outcome or prior probability is EVER passed to any stage — the frozen background
bundle is the benchmark's own as-of background, and the true `resolution` / SOTA prior are read only
by `_compare_after_freeze`, strictly after this arm's forecast is frozen and checkpointed.

Run one worker at a time, then merge:
    python -m experiments.exp114_lean_v2_fidelity_eval <i>   # i in 0..4, Banxico=0
    python -m experiments.exp114_lean_v2_fidelity_eval        # merge + measurement table + report
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

from experiments.exp101_btf3_pilot import _forecast_input, fetch_btf3
from experiments.exp107_btf3_full_fidelity_post127 import (MAX_TOKENS, MODEL, SEED, TEMPERATURE)
from experiments.exp113_lean_v2_completion_eval import NAMES, ORDER, _brier, _stored

CKPT = Path("experiments/results/exp114_checkpoints")
CACHE_ROOT = Path("experiments/results/exp114_cache")
SUMMARY = Path("experiments/results/exp114_fidelity_eval.json")
REPORT = Path("experiments/results/exp114_fidelity_eval.md")
#: §19 per-question HARD maximum — a trip finalizes the best labeled forecast, never a relaunch
GUARD_WALL_S, GUARD_CALLS = 900.0, 150     # 15 min / 150 calls
PRICE = {"input_per_m_cache_miss": 0.27, "input_per_m_cache_hit": 0.07, "output_per_m": 1.10}


# ---------------------------------------------------------------- Phase C: structural audit
def structural_audit(qid: str, lv2: dict, d: dict) -> dict:
    """The Phase C structural dry-audit — compiled-world STRUCTURE only, NO forecast, NO outcome."""
    bp = lv2.get("blueprint") or {}
    it = lv2.get("institution_terminal") or {}
    rep = it.get("representation") or {}
    ev = lv2.get("evidence_store") or {}
    pk = lv2.get("knowledge_packets") or {}
    scg = lv2.get("shared_condition_graph") or {}
    omd = lv2.get("outcome_mechanism_dimensions") or {}
    fid = lv2.get("structural_fidelity") or {}
    states = lv2.get("actor_states") or {}
    # deliberation stages present (institution votes): a per-combo resolution with a transcript
    delib = None
    per_combo = it.get("per_combo") or []
    if per_combo:
        tr = ((per_combo[0].get("resolution") or {}).get("transcript") or {})
        delib = {"institution_type": (per_combo[0].get("resolution") or {}).get("institution_type"),
                 "rounds_run": tr.get("rounds_run"), "material_changes": tr.get("material_changes"),
                 "n_messages": len(tr.get("messages") or [])}
    # terminal kind: institution_vote when a deliberative resolution ran; numeric when a mechanism
    # dimension was checked; else the readiness/round-trip record
    terminal_kind = ("institution_vote" if it.get("p_yes") is not None or rep
                     else "numeric_or_event" if omd
                     else (lv2.get("readiness") or {}).get("terminal_kind") or "unknown")
    audit = {
        "question_id": qid, "name": NAMES.get(qid, qid),
        "terminal_kind": terminal_kind,
        "actors": (bp.get("n_actors") if isinstance(bp, dict) else None) or len(states),
        "institution": {"id": rep.get("institution_id"), "rule": rep.get("rule"),
                        "modeled_members": len([u for u in (rep.get("decision_units") or [])
                                                if (u.get("provenance") or "").startswith("modeled")]),
                        "total_units": len(rep.get("decision_units") or [])} if rep else None,
        "representation": {"real_member_count": rep.get("real_member_count"),
                           "represented_voting_power": rep.get("represented_voting_power"),
                           "total_voting_power": rep.get("total_voting_power"),
                           "threshold": it.get("threshold"), "declared_threshold": it.get("declared_threshold"),
                           "n_decision_units": len(rep.get("decision_units") or []),
                           "candidates": rep.get("candidates"), "faithful": rep.get("faithful"),
                           "verdict": rep.get("verdict")},
        "terminal_variable_units_threshold": {
            "output_dimension": omd.get("dimension"), "required_dimension": omd.get("required_dimension"),
            "dimension_ok": omd.get("ok")},
        "evidence_coverage": {"n_facts": ev.get("n_facts"), "dropped_leakage": ev.get("n_dropped_leakage"),
                              "contradiction_groups": ev.get("contradiction_groups")},
        "knowledge_packets": {"n": pk.get("n"), "leakage_flags": pk.get("leakage_flags")},
        "shared_conditions": {"n_conditions": len((scg.get("conditions") or {})),
                              "correlated_actors": scg.get("correlated_actors"),
                              "preserved_tail_mass": scg.get("preserved_tail_mass"),
                              "n_joint_worlds": scg.get("n_joint_worlds")},
        "action_mass_weighting": {"actors_weighted": len(states),
                                  "mindset_separated": (lv2.get("mindset_separation") or {})
                                  .get("states_relabeled")},
        "deliberation": delib,
        "structural_fidelity_verdict": fid.get("verdict"),
        "structural_fidelity_checks": {k: v.get("verdict") for k, v in (fid.get("checks") or {}).items()},
    }
    # structural invariants (each pass/fail is about STRUCTURE, never the answer)
    inv = {}
    if audit["terminal_kind"] == "institution_vote":
        inv["faithful_roster"] = rep.get("verdict") == "ready" and (
            rep.get("real_member_count") is None
            or rep.get("total_voting_power") == rep.get("real_member_count"))
        inv["threshold_not_rescaled"] = (it.get("declared_threshold") is None
                                         or it.get("threshold") == it.get("declared_threshold"))
        inv["deliberation_ran"] = bool(delib)
    inv["evidence_present"] = (ev.get("n_facts") or 0) > 0
    inv["packets_built"] = (pk.get("n") or 0) > 0
    inv["no_packet_leakage"] = not (pk.get("leakage_flags"))
    inv["behavior_grounded"] = (audit["action_mass_weighting"]["actors_weighted"] or 0) > 0
    if omd:
        inv["outcome_dimension_ok"] = bool(omd.get("ok"))
    inv["structural_fidelity_not_broken"] = fid.get("verdict") in ("ready", "repairable")
    audit["structural_invariants"] = inv
    audit["structural_pass"] = all(inv.values())
    return audit


def print_structural_audit(a: dict):
    print(f"\n===== PHASE C — STRUCTURAL AUDIT: {a['name']} (no outcome) =====")
    print(f"  terminal_kind: {a['terminal_kind']}   actors: {a['actors']}")
    if a.get("institution"):
        inst = a["institution"]
        print(f"  institution {inst['id']}: {inst['modeled_members']} modeled / "
              f"{inst['total_units']} total units, rule={inst['rule']}")
    r = a["representation"]
    if a["terminal_kind"] == "institution_vote":
        print(f"  faithful representation: real={r['real_member_count']} "
              f"represented_power={r['total_voting_power']} threshold={r['threshold']} "
              f"(declared {r['declared_threshold']}) units={r['n_decision_units']} "
              f"verdict={r['verdict']}")
        if a["deliberation"]:
            dl = a["deliberation"]
            print(f"  deliberation: {dl['institution_type']} rounds={dl['rounds_run']} "
                  f"messages={dl['n_messages']} material_changes={dl['material_changes']}")
    t = a["terminal_variable_units_threshold"]
    print(f"  outcome dimension: {t['output_dimension']} vs required {t['required_dimension']} "
          f"(ok={t['dimension_ok']})")
    e = a["evidence_coverage"]
    print(f"  evidence: {e['n_facts']} facts ({e['dropped_leakage']} leakage-dropped), "
          f"contradictions={e['contradiction_groups']}")
    p = a["knowledge_packets"]
    print(f"  knowledge packets: {p['n']} built, leakage_flags={p['leakage_flags'] or 'none'}")
    print(f"  behavior grounded: {a['action_mass_weighting']['actors_weighted']} actors weighted, "
          f"mindset_relabeled={a['action_mass_weighting']['mindset_separated']}")
    print(f"  structural_fidelity: {a['structural_fidelity_verdict']}  "
          f"checks={a['structural_fidelity_checks']}")
    print(f"  STRUCTURAL INVARIANTS: {a['structural_invariants']}")
    print(f"  ==> STRUCTURAL PASS: {a['structural_pass']}")


# ---------------------------------------------------------------- §19 cold-cache run
def run_worker(i: int, *, cold: bool = True) -> dict:
    from datetime import datetime, timezone

    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    qid = ORDER[i]
    CKPT.mkdir(parents=True, exist_ok=True)
    cpath = CKPT / f"{qid}.json"
    if cpath.exists():
        print(f"{NAMES[qid]} already checkpointed — never re-run (delete to redo)")
        return json.loads(cpath.read_text())["metrics"]

    # §19 COLD CACHE: clear this question's persistent cache before the run
    cache_dir = CACHE_ROOT / NAMES[qid].lower()
    if cold and cache_dir.exists():
        shutil.rmtree(cache_dir)

    rows = {r["question_id"]: r for r in fetch_btf3()}
    q = _forecast_input(rows[qid])
    # NO outcome, NO prior probability in the evidence — only the benchmark's as-of background
    evidence = (f"Resolution criteria: {q['resolution_criteria']}\n\n"
                f"Background (as of {str(q['present_date'])[:10]}): {q['background']}")
    as_of_ts = datetime.fromisoformat(str(q["present_date"]).split(".")[0]) \
        .replace(tzinfo=timezone.utc).timestamp()
    bundle = frozen_background_bundle(q["question"], as_of_ts=as_of_ts, background=q["background"],
                                     resolution_criteria=q["resolution_criteria"], seed=SEED)
    calls, pending = [], []
    base = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE, usage_sink=pending.append)

    def llm(prompt, _c=calls, _u=pending):
        t = time.time()
        reply = base(prompt)
        _c.append({"i": len(_c), "prompt_chars": len(prompt), "reply_chars": len(reply or ""),
                   "latency_s": round(time.time() - t, 3), "usage": (_u.pop() if _u else None)})
        return reply

    t0 = time.time()
    res = simulate_world(
        q["question"], llm=llm, evidence=evidence, as_of=str(q["present_date"])[:10],
        horizon=str(q["expected_resolution_date"])[:10], seed=SEED, prebuilt_bundle=bundle,
        execution_policy={"lean_v2": {
            "budget": {"max_wall_s": GUARD_WALL_S, "max_calls": GUARD_CALLS},
            "backend_fingerprint": MODEL, "persistent_cache": True,
            "persistent_cache_dir": str(cache_dir), "qid": f"{qid}-fidelity", "max_workers": 6}},
        execution_profile="lean_v2")
    wall = time.time() - t0
    d = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(res.__dict__)
    lv2 = (d.get("provenance") or {}).get("lean_v2") or {}
    usage_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in calls)
    usage_out = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls)
    cache_hit = sum((c.get("usage") or {}).get("prompt_cache_hit_tokens", 0) for c in calls)
    cost = round((usage_in - cache_hit) / 1e6 * PRICE["input_per_m_cache_miss"]
                 + cache_hit / 1e6 * PRICE["input_per_m_cache_hit"]
                 + usage_out / 1e6 * PRICE["output_per_m"], 4)

    # PHASE C structural audit (no outcome) — printed + frozen
    audit = structural_audit(qid, lv2, d)

    metrics = {
        "question_id": qid, "name": NAMES[qid], "question": q["question"],
        "as_of": str(q["present_date"])[:10], "horizon": str(q["expected_resolution_date"])[:10],
        "simulation_status": d.get("simulation_status"),
        "headline_forecast": d.get("raw_probability"),
        "probability_source": d.get("probability_source"),
        "probability_conditional_on_resolved": d.get("probability_conditional_on_resolved"),
        "forecast_decomposition": lv2.get("forecast_decomposition"),
        "structural_audit": audit,
        "guard": {"wall_s": round(wall, 1), "n_calls": len(calls),
                  "passed_hard_stop": wall <= GUARD_WALL_S and len(calls) <= GUARD_CALLS},
        "cost_usd": cost, "input_tokens": usage_in, "output_tokens": usage_out,
        "provider_cache_hit_tokens": cache_hit}

    # FREEZE before joining any outcome
    cpath.write_text(json.dumps({"metrics": metrics, "simulation_result": d,
                                 "n_calls": len(calls)}, default=str))
    print_structural_audit(audit)
    print(f"\n----- {NAMES[qid]} FROZEN FORECAST (pre-outcome) -----")
    print(f"  status: {metrics['simulation_status']}  headline: {metrics['headline_forecast']}  "
          f"source: {metrics['probability_source']}")
    print(f"  guard: {metrics['guard']['wall_s']}s / {metrics['guard']['n_calls']} calls  "
          f"(hard-stop ok: {metrics['guard']['passed_hard_stop']})  cost ${cost}")
    _compare_after_freeze(qid, metrics, rows)
    return metrics


def _compare_after_freeze(qid: str, metrics: dict, rows: dict) -> dict:
    """Read the TRUE outcome + SOTA prior ONLY here, strictly after the forecast is frozen."""
    outcome = int(rows[qid]["resolution"])
    st = _stored(qid)
    p = metrics["headline_forecast"]
    print(f"\n  ----- {NAMES[qid]} AFTER-FREEZE OUTCOME JOIN (scoring only) -----")
    print(f"  outcome: {outcome}")
    print(f"  full-fidelity: {st.get('ff_p')}  (Brier {_brier(st.get('ff_p'), outcome)})")
    print(f"  Lean V1:       {st.get('l1_p')}  (Brier {_brier(st.get('l1_p'), outcome)})")
    print(f"  exp112 lean_v2:{st.get('exp112_p')}  (Brier {_brier(st.get('exp112_p'), outcome)})")
    print(f"  exp114 (D1-D18): {p}  (Brier {_brier(p, outcome)})")
    cmp = {"outcome": outcome, "ff_p": st.get("ff_p"), "l1_p": st.get("l1_p"),
           "exp112_p": st.get("exp112_p"), "exp114_p": p,
           "brier": {"full_fidelity": _brier(st.get("ff_p"), outcome),
                     "lean_v1": _brier(st.get("l1_p"), outcome),
                     "exp112_lean_v2": _brier(st.get("exp112_p"), outcome),
                     "exp114_fidelity": _brier(p, outcome)}}
    cpath = CKPT / f"{qid}.json"
    obj = json.loads(cpath.read_text())
    obj["after_freeze_comparison"] = cmp        # stored SEPARATELY, after the freeze write
    cpath.write_text(json.dumps(obj, default=str))
    return cmp


# ---------------------------------------------------------------- §20-22 measurement
def merge_and_report():
    rows = {r["question_id"]: r for r in fetch_btf3()}
    per = []
    for qid in ORDER:
        cpath = CKPT / f"{qid}.json"
        if not cpath.exists():
            print(f"missing checkpoint: {NAMES[qid]}")
            continue
        obj = json.loads(cpath.read_text())
        per.append(obj)
    SUMMARY.write_text(json.dumps({"per_question": per}, indent=1, default=str))
    _write_measurement_report(per)
    print(f"\nmeasurement written to {REPORT}")


def _write_measurement_report(per: list):
    L = ["# EXP-114 — Lean V2 real-world-fidelity five-question measurement", "",
         "Same five frozen BTF-3 questions, cold-cache, sequential, through "
         "`simulate_world(..., \"lean_v2\")` carrying the full D1–D18 fidelity architecture. "
         "No outcome or prior in any prompt; outcomes joined only after each forecast froze.", "",
         "## Measurement table", "",
         "| Q | status | headline | source | Brier (D1–D18) | Brier exp112 | Brier full-fid | "
         "Brier Lean V1 | calls | wall s | struct pass |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    briers = []
    for obj in per:
        m = obj["metrics"]
        c = obj.get("after_freeze_comparison") or {}
        b = c.get("brier") or {}
        briers.append(b.get("exp114_fidelity"))
        L.append(f"| {m['name']} | {m['simulation_status']} | {m['headline_forecast']} | "
                 f"{m['probability_source']} | {b.get('exp114_fidelity')} | "
                 f"{b.get('exp112_lean_v2')} | {b.get('full_fidelity')} | {b.get('lean_v1')} | "
                 f"{m['guard']['n_calls']} | {m['guard']['wall_s']} | "
                 f"{m['structural_audit']['structural_pass']} |")
    valid = [x for x in briers if isinstance(x, (int, float))]
    mean_b = round(sum(valid) / len(valid), 4) if valid else None
    L += ["", f"**Mean Brier (D1–D18, {len(valid)} scored):** {mean_b}", "",
          "## Per-question under-the-hood (structure, then frozen forecast, then outcome)", ""]
    for obj in per:
        m, a = obj["metrics"], obj["metrics"]["structural_audit"]
        c = obj.get("after_freeze_comparison") or {}
        L += [f"### {m['name']}", "",
              f"- terminal_kind: {a['terminal_kind']}; structural_pass: {a['structural_pass']}",
              f"- structural_invariants: {a['structural_invariants']}",
              f"- representation: {a['representation']}",
              f"- deliberation: {a['deliberation']}",
              f"- evidence: {a['evidence_coverage']}; packets: {a['knowledge_packets']}",
              f"- outcome dimension: {a['terminal_variable_units_threshold']}",
              f"- structural_fidelity: {a['structural_fidelity_verdict']} "
              f"{a['structural_fidelity_checks']}",
              f"- FROZEN forecast: {m['headline_forecast']} (source {m['probability_source']}, "
              f"status {m['simulation_status']})",
              f"- guard: {m['guard']}",
              f"- AFTER-FREEZE outcome: {c.get('outcome')}; Brier {(c.get('brier') or {})}", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_worker(int(sys.argv[1]))
    else:
        merge_and_report()
