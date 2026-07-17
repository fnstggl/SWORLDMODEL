"""Phase 13 controlled benchmark runner (Part 30A + Part 19 + Part 36 functional/search gates).

Executes the 200 independently specified controlled decision tasks through the CANONICAL Phase 13
API (`recommend_action` / `optimize_policy` — the same funnel production uses), measures every gate
against the tasks' KNOWN optima, and persists machine-readable artifacts:

    artifacts/phase13/controlled/manifest.json            task corpus (family, split, optimum)
    artifacts/phase13/controlled/results.jsonl            per-task rows (resumable; append-only)
    artifacts/phase13/controlled/search_correctness.json  Part-19 racing/hierarchical vs truth
    artifacts/phase13/controlled/gates.json               aggregate gate evaluation
    artifacts/phase13/controlled/locked_access_log.json   single-shot locked-test access record

Split governance (Part 32): development/calibration/validation run by default; the locked_test
split runs ONLY with --locked, exactly once — a second --locked invocation refuses (the access log
already exists) rather than reopening the set.

Resumability (Part 38): results.jsonl is append-only and keyed by task_id; a rerun skips completed
tasks deterministically (same per-task seed = sha256(task_id)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from swm.world_model_v2.phase13.api import recommend_action, optimize_policy, _evaluator
from swm.world_model_v2.phase13.controlled import (build_search_task, build_task, split_of,
                                                   task_ids, T0, DAY)
from swm.world_model_v2.phase13.crn import exogenous_trace
from swm.world_model_v2.phase13.ontology import ActionSchema
from swm.world_model_v2.phase13.policies import Policy
from swm.world_model_v2.phase13.search import SearchBudget, select_and_run

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "phase13", "controlled")


def _seed(task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big") % (2 ** 31)


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------- policies for the policy tasks
def _mk_action(aid: str, variant: str) -> ActionSchema:
    return ActionSchema(action_id=aid, actor="decider", operation="communicate", object="decider",
                        recipients=["decider"], content={"variant": variant, "quality": 0.0})


def adaptive_policy() -> Policy:
    """Waits for the reveal observation, then acts ONCE on the revealed side. Stateless: the
    already-acted guard reads the belief's prior_actions (per-branch), never a Python closure —
    closures would leak 'already fired' state across the arm's particles."""
    def decide(belief):
        if belief.get("prior_actions"):
            return None
        for o in belief.get("observations", []):
            iid = str(o.get("item_id", ""))
            if iid.startswith("reveal:"):
                side = iid.split(":", 1)[1]
                return _mk_action(f"act_{side}", side)
        return None                                      # nothing revealed yet -> wait
    return Policy(policy_id="adaptive_policy", decide=decide,
                  description="act on the revealed context; wait until revealed")


def greedy_blind_policy() -> Policy:
    """Commits ONCE at the first decision point without using any observation (one-step greedy)."""
    def decide(belief):
        if belief.get("prior_actions"):
            return None
        return _mk_action("act_blind", "blind")
    return Policy(policy_id="greedy_blind", decide=decide, description="commit immediately, blind")


# ---------------------------------------------------------------- per-task execution
def run_action_task(t: dict, seed: int) -> dict:
    r = recommend_action(t["problem"], t["ctx"], budget="standard", seed=seed)
    row = {"task_id": t["task_id"], "family": t["family"], "split": t["split"], "kind": "action",
           "recommended": r.recommended, "recommendation_kind": r.recommendation_kind,
           "optimum": t["optimum"]["action_id"], "hit": r.recommended == t["optimum"]["action_id"],
           "n_actions": len(r.evaluated), "search": r.search, "latency_s": r.latency_s,
           "seed": seed}
    # optimality gap vs the KNOWN per-action true values (only where E[U] defines the optimum)
    vals = t.get("values")
    if vals is not None and t.get("gap_evaluable", True):
        vopt = float(t["optimum"]["value"])
        vpick = float(vals.get(r.recommended, 0.0))       # do_nothing / gather_information = 0
        row["gap_abs"] = round(max(0.0, vopt - vpick), 6)
        row["gap_rel"] = round(max(0.0, vopt - vpick) / max(abs(vopt), 1e-9), 6)
    # CRN pairing evidence: every arm's hazard-sourced exogenous trace must match the reference
    man = (r.provenance or {}).get("crn_manifest", {})
    match = man.get("exogenous_trace_match_vs_reference", {})
    row["crn_match_min"] = min(match.values()) if match else None
    row["crn_all_paired"] = bool(match) and all(v == 1.0 for v in match.values())
    # variance reduction from matching (aggregate evidence, Part 36 counterfactual gate)
    vrs = [e.get("variance_reduction", {}).get("variance_reduction_factor")
           for e in r.evaluated if e.get("action_id") != r.reference_action]
    vrs = [v for v in vrs if isinstance(v, (int, float))]
    row["variance_reduction_median"] = sorted(vrs)[len(vrs) // 2] if vrs else None
    # feasibility gate: expected-infeasible actions must be rejected with typed reasons
    if t.get("expect_infeasible"):
        rejected = {f["action_id"]: [x["code"] for x in f["reasons"]]
                    for f in r.feasibility if not f["feasible"]}
        row["expected_infeasible_rejected"] = all(a in rejected for a in t["expect_infeasible"])
        row["infeasible_reasons"] = {a: rejected.get(a) for a in t["expect_infeasible"]}
    # Pareto gate for multi-stakeholder tasks
    if t.get("pareto_expected"):
        frontier = {p["action_id"] for p in r.pareto_frontier if p.get("on_frontier")}
        row["pareto_recovered"] = set(t["pareto_expected"]) <= frontier
    # no-abstention sanity on fully specified tasks
    row["abstained"] = r.recommendation_kind == "abstain"
    # active phase census (operator delta counts — proves effects flowed through the world)
    row["operator_delta_census"] = (r.active_phases or {}).get("operator_delta_census", {})
    return row


def run_policy_task(t: dict, seed: int) -> dict:
    r = optimize_policy(t["problem"], [adaptive_policy(), greedy_blind_policy()], t["ctx"],
                        seed=seed)
    by_id = {p["action_id"]: p for p in r.policies}
    ad, gr = by_id.get("adaptive_policy", {}), by_id.get("greedy_blind", {})
    row = {"task_id": t["task_id"], "family": t["family"], "split": t["split"], "kind": "policy",
           "recommended": r.recommended, "optimum": t["optimum"]["action_id"],
           "hit": r.recommended == t["optimum"]["action_id"],
           "adaptive_eu": ad.get("expected_utility"), "greedy_eu": gr.get("expected_utility"),
           "sequential_beats_greedy": (ad.get("expected_utility", 0) > gr.get("expected_utility", 0)),
           "paired_adaptive_vs_ref": ad.get("paired_vs_reference", {}),
           "latency_s": r.latency_s, "seed": seed}
    man = (r.provenance or {}).get("crn_manifest", {})
    match = man.get("exogenous_trace_match_vs_reference", {})
    row["crn_match_min"] = min(match.values()) if match else None
    row["crn_all_paired"] = bool(match) and all(v == 1.0 for v in match.values())
    # VOI gate on information-gathering tasks: knowing the hidden side must flip the best one-step
    # action, so EVSI > 0 and the layer recommends gathering before committing
    if t["family"] == "information_gathering":
        row["voi"] = run_voi_check(t, seed)
    return row


def run_voi_check(t: dict, seed: int) -> dict:
    ev = _evaluator(t["ctx"], seed=seed)
    sides = []
    for w in ev.particles():
        f = w.entities["environment"].get("latent_state", key="context")
        v = float(f.value) if f is not None and f.value is not None else 0.5
        sides.append("high" if v >= 0.5 else "low")
    p2 = t["problem"]
    p2.candidate_actions = [_mk_action("act_high", "high"), _mk_action("act_low", "low"),
                            _mk_action("act_blind", "blind")]
    p2.generated_action_permission = False
    r = recommend_action(p2, t["ctx"], budget="standard", seed=seed,
                         candidate_observations=[{"name": "context_reveal",
                                                  "signal_of_particle": sides, "cost": 0.0}])
    voi = r.value_of_information or {}
    best = (voi.get("candidates") or [{}])[0]
    return {"evpi": (voi.get("evpi") or {}).get("evpi"),
            "evsi_net": best.get("evsi_net"),
            "would_change_decision": best.get("would_change_decision"),
            "recommend_gathering": voi.get("recommend_gathering"),
            "recommendation_kind": r.recommendation_kind}


def run_determinism_check(task_id: str) -> dict:
    """Deterministic replay gate: identical config twice -> identical recommendation + utilities."""
    t1, t2 = build_task(task_id), build_task(task_id)
    s = _seed(task_id)
    r1 = recommend_action(t1["problem"], t1["ctx"], budget="standard", seed=s)
    r2 = recommend_action(t2["problem"], t2["ctx"], budget="standard", seed=s)
    eu1 = {e["action_id"]: e["expected_utility"] for e in r1.evaluated}
    eu2 = {e["action_id"]: e["expected_utility"] for e in r2.evaluated}
    return {"task_id": task_id, "same_recommendation": r1.recommended == r2.recommended,
            "identical_expected_utilities": eu1 == eu2}


# ---------------------------------------------------------------- Part 19: search correctness
def run_search_correctness() -> dict:
    """Racing + hierarchical vs exhaustive truth on enlarged known-optimum instances, with a
    budget-performance curve. The 200-task corpus itself runs exhaustive (small spaces); this block
    is where the approximate optimizers are MEASURED, not assumed."""
    out = {"racing": [], "hierarchical": [], "budget_curve": []}
    for n_arms, n_seeds in ((40, 6), (120, 4)):
        for s in range(n_seeds):
            t = build_search_task(n_arms, s)
            ev = _evaluator(t["ctx"], seed=_seed(t["task_id"]))
            bundle, diag = select_and_run(ev, list(t["problem"].candidate_actions), t["problem"],
                                          budget=SearchBudget.tiered("standard"))
            picked = _best_arm(bundle)
            vals = t["values"]
            vopt = t["optimum"]["value"]
            out["racing"].append({
                "task_id": t["task_id"], "n_arms": n_arms, "method": diag.method,
                "n_evaluated": diag.n_evaluated, "picked": picked,
                "recovered": picked == t["optimum"]["action_id"],
                "gap_rel": round(max(0.0, vopt - vals.get(picked, 0.0)) / max(abs(vopt), 1e-9), 6)})
    for s in range(2):
        t = build_search_task(240, s, n_families=2)
        ev = _evaluator(t["ctx"], seed=_seed(t["task_id"]))
        bundle, diag = select_and_run(ev, list(t["problem"].candidate_actions), t["problem"],
                                      budget=SearchBudget.tiered("production"))
        picked = _best_arm(bundle)
        vals, vopt = t["values"], t["optimum"]["value"]
        out["hierarchical"].append({
            "task_id": t["task_id"], "n_arms": 240, "method": diag.method,
            "n_evaluated": diag.n_evaluated, "picked": picked,
            "recovered": picked == t["optimum"]["action_id"],
            "gap_rel": round(max(0.0, vopt - vals.get(picked, 0.0)) / max(abs(vopt), 1e-9), 6)})
    # budget-performance curve: the same 60-arm instance at every tier
    t = build_search_task(60, 0)
    for tier in ("diagnostic", "standard", "production"):
        ev = _evaluator(t["ctx"], seed=_seed(t["task_id"]))
        bundle, diag = select_and_run(ev, list(t["problem"].candidate_actions), t["problem"],
                                      budget=SearchBudget.tiered(tier))
        picked = _best_arm(bundle)
        vals, vopt = t["values"], t["optimum"]["value"]
        out["budget_curve"].append({"tier": tier, "n_evaluated": diag.n_evaluated,
                                    "picked": picked,
                                    "gap_rel": round(max(0.0, vopt - vals.get(picked, 0.0))
                                                     / max(abs(vopt), 1e-9), 6)})
    return out


def _best_arm(bundle) -> str:
    def mean_readout(arm):
        xs = [float(o["readout"]) for o in arm.outcomes
              if isinstance(o.get("readout"), (int, float))]
        return sum(xs) / len(xs) if xs else 0.0
    scores = {aid: mean_readout(arm) for aid, arm in bundle.arms.items()
              if aid not in ("do_nothing", "gather_information", "defer")}
    return max(scores, key=scores.get) if scores else None


# ---------------------------------------------------------------- gates aggregation
def compute_gates(rows: list, search: dict, determinism: list) -> dict:
    act = [r for r in rows if r["kind"] == "action"]
    pol = [r for r in rows if r["kind"] == "policy"]
    gaps = sorted(r["gap_rel"] for r in act if "gap_rel" in r)
    crn_rows = [r for r in rows if r.get("crn_match_min") is not None]
    vrs = [r["variance_reduction_median"] for r in act
           if isinstance(r.get("variance_reduction_median"), (int, float))]
    feas = [r for r in act if "expected_infeasible_rejected" in r]
    pareto = [r for r in act if "pareto_recovered" in r]
    voi = [r["voi"] for r in pol if r.get("voi")]
    racing_rec = [x["recovered"] for x in search.get("racing", [])]
    racing_gap = sorted(x["gap_rel"] for x in search.get("racing", []))
    gates = {
        "n_tasks": len(rows), "n_action_tasks": len(act), "n_policy_tasks": len(pol),
        "exhaustive_recovery_rate": round(sum(1 for r in act if r["hit"]) / max(1, len(act)), 4),
        "recovery_gate_99pct": (sum(1 for r in act if r["hit"]) / max(1, len(act))) >= 0.99,
        "median_optimality_gap_rel": (gaps[len(gaps) // 2] if gaps else None),
        "gap_gate_1pct": (gaps[len(gaps) // 2] <= 0.01) if gaps else False,
        "crn_pairing_rate": round(sum(1 for r in crn_rows if r["crn_all_paired"])
                                  / max(1, len(crn_rows)), 4),
        "crn_gate_100pct": all(r["crn_all_paired"] for r in crn_rows) if crn_rows else False,
        "variance_reduction_median_of_medians": (sorted(vrs)[len(vrs) // 2] if vrs else None),
        "matched_variance_not_worse": (sorted(vrs)[len(vrs) // 2] >= 1.0) if vrs else False,
        "sequential_beats_greedy_rate": round(sum(1 for r in pol if r["sequential_beats_greedy"])
                                              / max(1, len(pol)), 4) if pol else None,
        "sequential_gate": all(r["sequential_beats_greedy"] for r in pol) if pol else False,
        "policy_recovery_rate": round(sum(1 for r in pol if r["hit"]) / max(1, len(pol)), 4)
        if pol else None,
        "feasibility_rejections_correct": all(r["expected_infeasible_rejected"] for r in feas)
        if feas else None,
        "pareto_recovery": all(r["pareto_recovered"] for r in pareto) if pareto else None,
        "voi_recommends_gathering_rate": round(sum(1 for v in voi if v.get("recommend_gathering"))
                                               / max(1, len(voi)), 4) if voi else None,
        "abstention_false_positives": sum(1 for r in act if r.get("abstained")),
        "racing_recovery_rate": round(sum(racing_rec) / max(1, len(racing_rec)), 4)
        if racing_rec else None,
        "racing_median_gap_rel": racing_gap[len(racing_gap) // 2] if racing_gap else None,
        "hierarchical": search.get("hierarchical"),
        "budget_curve": search.get("budget_curve"),
        "deterministic_replay": all(d["same_recommendation"] and d["identical_expected_utilities"]
                                    for d in determinism) if determinism else False,
        "search_diagnostics_present_rate": round(
            sum(1 for r in act if r.get("search", {}).get("method")) / max(1, len(act)), 4),
    }
    by_split = {}
    for r in rows:
        s = by_split.setdefault(r["split"], {"n": 0, "hits": 0})
        s["n"] += 1
        s["hits"] += 1 if r["hit"] else 0
    gates["by_split"] = {k: {"n": v["n"], "recovery": round(v["hits"] / v["n"], 4)}
                         for k, v in by_split.items()}
    return gates


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true",
                    help="run the locked_test split (exactly once; refuses on a second attempt)")
    ap.add_argument("--limit", type=int, default=0, help="cap tasks (debug only)")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    ids = task_ids()
    manifest = []
    for tid in ids:
        fam = tid.rsplit("_", 1)[0]
        manifest.append({"task_id": tid, "family": fam, "split": split_of(tid)})
    with open(os.path.join(ART, "manifest.json"), "w") as f:
        json.dump({"n_tasks": len(ids), "tasks": manifest,
                   "split_sizes": _split_sizes(manifest)}, f, indent=1)

    if args.locked:
        lock_path = os.path.join(ART, "locked_access_log.json")
        if os.path.exists(lock_path):
            print("REFUSING: locked_test already accessed (locked_access_log.json exists). "
                  "The locked set opens once; a failed locked run stays failed.")
            sys.exit(2)
        todo = [m["task_id"] for m in manifest if m["split"] == "locked_test"]
        res_path = os.path.join(ART, "results_locked.jsonl")
    else:
        todo = [m["task_id"] for m in manifest if m["split"] != "locked_test"]
        res_path = os.path.join(ART, "results.jsonl")

    done = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            done = {json.loads(line)["task_id"] for line in f if line.strip()}
    todo = [tid for tid in todo if tid not in done]
    if args.limit:
        todo = todo[:args.limit]

    t_start = _now()
    with open(res_path, "a") as f:
        for i, tid in enumerate(todo):
            t = build_task(tid)
            row = (run_policy_task if t.get("policy_task") else run_action_task)(t, _seed(tid))
            row["wall_ts"] = _now()
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(todo)}] {tid} hit={row['hit']} "
                      f"({_now() - t_start:.0f}s elapsed)", flush=True)

    rows = []
    with open(res_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.locked:
        with open(os.path.join(ART, "locked_access_log.json"), "w") as f:
            json.dump({"accessed_at": _now(), "n_tasks": len(rows),
                       "result_hash": hashlib.sha256(
                           json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16],
                       "note": "single locked-test access; this file's existence blocks reruns"},
                      f, indent=1)
        gates = compute_gates(rows, {}, [])
        with open(os.path.join(ART, "gates_locked.json"), "w") as f:
            json.dump(gates, f, indent=1)
    else:
        search = run_search_correctness()
        with open(os.path.join(ART, "search_correctness.json"), "w") as f:
            json.dump(search, f, indent=1)
        determinism = [run_determinism_check(tid) for tid in
                       ("discrete_01", "multi_actor_02", "constrained_03")]
        gates = compute_gates(rows, search, determinism)
        gates["determinism_checks"] = determinism
        with open(os.path.join(ART, "gates.json"), "w") as f:
            json.dump(gates, f, indent=1)
    print(json.dumps({k: v for k, v in gates.items()
                      if not isinstance(v, (list, dict))}, indent=1))


def _split_sizes(manifest):
    out = {}
    for m in manifest:
        out[m["split"]] = out.get(m["split"], 0) + 1
    return out


if __name__ == "__main__":
    main()
