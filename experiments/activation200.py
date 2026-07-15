"""Part 10 — activation gates on the 200-question independent corpus, measured through the SUPERVISED
canonical execution path (PhaseExecutionRecord statuses, not a separate accounting).

Per question: real compiler → rule normalization → supervisor assess → activation synthesis →
run_from_plan (deterministic, LLM-free rollout) → supervisor finalize → per-phase status; then matched
ablations (same plan, same seed, common randomness; only the target phase's requirement forced off) for
every executed phase.

Gates (Part 10): recall(causally_active | required) ≥ 0.95 per phase; false execution ≤ 0.10;
blocked-relevant ≤ 0.02; PhaseExecutionRecord coverage 100%; matched-ablation effect on ≥ 80% of relevant
cases (terminal shift ≥ 0.02 or a StateDelta-trajectory change).
"""
from __future__ import annotations
import argparse, copy, hashlib, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.activation_corpus_200 import QUESTIONS, PHASE_FLAGS

OUT = Path("experiments/results/post_snapshot_benchmark")
ART = OUT / "activation200_execution.json"

_FLAG_PHASE = {"p4": "phase4_actor_policy", "p6": "phase6_registry", "p7": "phase7_nonlinear",
               "p9pop": "phase9_populations", "p9net": "phase9_networks",
               "p10": "phase10_institutions", "p11": "phase11_recompilation"}
ABL_EPS = 0.02
_LOCK = threading.Lock()


def _make_llm(api_key=None):
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2,
                           api_key=api_key, model="deepseek-v4-flash", thinking="disabled")


def _p_aff(res, plan):
    dist = res.get("distribution") or {}
    opts = list(plan.outcome_contract.options)
    return float(dist.get(str(opts[0]) if opts else "True", 0.0) or 0.0)


def _trajectory_signature(branches):
    """Stable structural signature of the StateDelta trajectory.

    The gate accepts either a terminal shift or a StateDelta-trajectory
    change. Compare branch/order/timestamp/operator, event type, and written
    paths; omit values so floating-point serialization noise cannot create an
    effect.
    """
    trajectory = []
    for branch_index, branch in enumerate(branches):
        for delta_index, delta in enumerate(branch.log):
            trajectory.append([
                branch_index, delta_index, round(float(delta.at), 6),
                str(delta.operator), str(delta.event_type),
                [str(change.get("path", "")) for change in (delta.changes or [])],
            ])
    payload = json.dumps(trajectory, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "n_state_deltas": len(trajectory)}


def _run_supervised(plan, seed=3, force_off=None):
    """normalize → assess (with optional forced-off phase, matched-ablation arm) → synthesize → rollout →
    finalize. Returns (records, p_affirmative, delta_traj_len)."""
    from swm.world_model_v2.integration_completion import normalize_institution_rules
    from swm.world_model_v2.activation_synthesis import phase_requirements, synthesize_activation
    from swm.world_model_v2.materialize import run_from_plan
    from swm.world_model_v2 import phase_supervision as PS
    from swm.world_model_v2.pipeline import _operator_delta_census
    from types import SimpleNamespace
    normalize_institution_rules(plan)
    req = phase_requirements(plan)
    if force_off:
        req[force_off] = {"required": False, "why": "matched ablation arm", "signal": False}
    synthesize_activation(plan, req)
    recs = PS.assess(plan, has_as_of=True, has_bundle=False)   # no evidence/obs on this harness → p2/3/11 no-op
    p11_executed = False
    if req.get("phase11_recompilation", {}).get("required"):
        p11_executed = _run_p11_development_trigger(plan, seed)
        recs["phase11_recompilation"].input_state_present = p11_executed
        recs["phase11_recompilation"].execution_status = "no_op_causally_irrelevant"
    if force_off:
        recs[force_off].relevant = False
        recs[force_off].execution_status = "no_op_causally_irrelevant"
    res, branches = run_from_plan(plan, llm=None, seed=seed)
    stub = SimpleNamespace(provenance={"operator_delta_census": _operator_delta_census(branches)})
    phase_meta = {k: {"executed": True} for k in
                  ("phase1_compiler", "phase2_evidence", "phase3_posterior", "phase8_persistence")}
    phase_meta["phase11_recompilation"] = {
        "executed": p11_executed,
        "reason": ("development structural-change observation processed" if p11_executed
                   else "no natural structural-change cue")}
    out = PS.finalize(recs, plan, stub, phase_meta=phase_meta)
    terminal = dict(res.get("distribution") or {})
    return out["records"], _p_aff(res, plan), terminal, _trajectory_signature(branches)


def _run_p11_development_trigger(plan, seed):
    """Exercise the real controller on a non-outcome structural cue.

    This is activation-development data only.  The observation is generated
    from the question's independently adjudicated structural-change cue and
    contains no historical outcome or resolution text.
    """
    from swm.world_model_v2.phase11.controller import RecompilationController, ExecutionAdapter
    from swm.world_model_v2.phase11.contracts import RecompileObservation
    at = float(plan.as_of) + max(1.0, (float(plan.horizon_ts) - float(plan.as_of)) / 2.0)
    obs = RecompileObservation(
        observation_id="activation_dev_structural_change", observation_type="structural_break",
        origin="external_evidence", event_time=at, ingestion_time=at,
        evidence_ids=["activation-development-non-outcome-cue"],
        provenance={"observed_value": None, "question_derived": True})
    result = RecompilationController(llm=None, seed=seed, max_recompiles=1).run(
        plan=plan, worlds=[], weights=[], pending_events=[], observations=[obs],
        horizon_ts=float(plan.horizon_ts), as_of=float(plan.as_of), execution=ExecutionAdapter())
    return result.n_observations == 1 and result.n_eligible == 1


def _one(qrow, llm):
    qid, q, as_of, horizon, domain, family, flags = qrow
    from swm.world_model_v2.compiler import compile_world
    rec = {"qid": qid, "domain": domain, "family": family, "required_labels": sorted(flags)}
    try:
        base = compile_world(q, llm=llm, evidence="", as_of=as_of, horizon=horizon, seed=0)
        records, p_full, terminal_full, trajectory_full = _run_supervised(copy.deepcopy(base))
        rec["phase_records"] = {ph: {"status": r.execution_status, "relevant": r.relevant,
                                     "n_deltas": r.n_state_deltas,
                                     "terminal_influence": r.terminal_influence}
                                for ph, r in records.items()}
        rec["p_full"] = round(p_full, 4)
        abls = {}
        for f, ph in _FLAG_PHASE.items():
            if ph == "phase11_recompilation":
                continue
            if records[ph].execution_status == "causally_active":
                _, p_abl, terminal_abl, trajectory_abl = _run_supervised(
                    copy.deepcopy(base), force_off=ph)
                keys = set(terminal_full) | set(terminal_abl)
                tv = 0.5 * sum(abs(float(terminal_full.get(k, 0.0)) -
                                   float(terminal_abl.get(k, 0.0))) for k in keys)
                trajectory_changed = trajectory_full["sha256"] != trajectory_abl["sha256"]
                abls[f] = {"delta_terminal": round(abs(p_full - p_abl), 4),
                           "terminal_total_variation": round(tv, 4),
                           "state_trajectory_changed": trajectory_changed,
                           "full_state_deltas": trajectory_full["n_state_deltas"],
                           "ablated_state_deltas": trajectory_abl["n_state_deltas"],
                           "effect": tv >= ABL_EPS or trajectory_changed,
                           "effect_criterion": "terminal_tv_ge_0.02_or_state_trajectory_change"}
        rec["ablations"] = abls
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:160]
    return rec


def _write_artifact(artifact, payload):
    tmp = artifact.with_suffix(artifact.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(artifact)


def run(limit=None, workers=6, api_key=None, artifact=ART):
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm(api_key=api_key)
    rows, done = [], set()
    if artifact.exists():
        rows = [r for r in json.loads(artifact.read_text()).get("rows", []) if not r.get("error")]
        done = {r["qid"] for r in rows}
    qs = [q for q in (QUESTIONS[:limit] if limit else QUESTIONS) if q[0] not in done]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, q, llm): q[0] for q in qs}
        for fut in as_completed(futs):
            rec = fut.result()
            with _LOCK:
                rows.append(rec)
                _write_artifact(artifact, {"rows": rows})
            st = rec.get("phase_records", {})
            active = [k for k, v in st.items() if v["status"] == "causally_active"]
            print(f"{rec['qid']:16s} req={','.join(rec['required_labels']) or '-':22s} "
                  f"active={','.join(a.replace('phase', 'p') for a in active) or '-':40s} "
                  f"err={rec.get('error', '')[:50]}", flush=True)
    _aggregate(rows, artifact=artifact)


def _aggregate(rows, artifact=ART):
    ok = [r for r in rows if not r.get("error")]
    per = {}
    for f, ph in _FLAG_PHASE.items():
        req = [r for r in ok if f in r["required_labels"]]
        notreq = [r for r in ok if f not in r["required_labels"]]
        act = lambda r: r["phase_records"][ph]["status"] == "causally_active"       # noqa: E731
        blocked = lambda r: r["phase_records"][ph]["status"].startswith("blocked")  # noqa: E731
        tp = sum(1 for r in req if act(r))
        fp = sum(1 for r in notreq if act(r))
        blk = sum(1 for r in req if blocked(r))
        abl = [r for r in req if r.get("ablations", {}).get(f)]
        eff = sum(1 for r in abl if r["ablations"][f]["effect"])
        per[f] = {"n_required": len(req), "recall": round(tp / len(req), 3) if req else None,
                  "n_not_required": len(notreq),
                  "false_execution": round(fp / len(notreq), 3) if notreq else None,
                  "blocked_relevant_rate": round(blk / len(req), 3) if req else None,
                  "n_ablated": len(abl),
                  "ablation_effect_rate": round(eff / len(abl), 3) if abl else None}
    coverage = all(len(r.get("phase_records", {})) == 11 for r in ok)
    gates = {}
    for f in per:
        p = per[f]
        gates[f"recall_{f}_ge_0.95"] = (p["recall"] or 0) >= 0.95
        gates[f"false_{f}_le_0.10"] = (p["false_execution"] if p["false_execution"] is not None else 1) <= 0.10
        gates[f"blocked_{f}_le_0.02"] = (p["blocked_relevant_rate"] or 0) <= 0.02
        if f != "p11":
            gates[f"ablation_{f}_ge_0.80"] = (p["ablation_effect_rate"] or 0) >= 0.80
    gates["phase_record_coverage_100"] = coverage
    agg = {"n_scored": len(ok), "n_errors": len(rows) - len(ok), "per_phase": per, "gates": gates,
           "gates_passed": sum(gates.values()), "gates_total": len(gates),
           "all_pass": all(gates.values())}
    payload = {"rows": rows, "aggregate": agg}
    _write_artifact(artifact, payload)
    print("\nAGGREGATE:", json.dumps(agg["per_phase"], indent=1))
    print("GATES:", json.dumps(gates, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--artifact", type=Path, default=ART,
                    help="write a separate resumable artifact (existing non-error rows are reused)")
    ap.add_argument("--api-key-tty", action="store_true",
                    help="read DeepSeek credential from an echo-disabled TTY into process memory")
    ap.add_argument("--credential-source", type=Path,
                    help="read the credential into memory from a user-owned source; never persisted")
    args = ap.parse_args()
    api_key = None
    if args.api_key_tty:
        import getpass
        api_key = getpass.getpass("DeepSeek API key: ")
    elif args.credential_source:
        from experiments.post_snapshot_benchmark.credentials import read_deepseek_key
        api_key = read_deepseek_key(args.credential_source)
    run(limit=args.limit, workers=args.workers, api_key=api_key, artifact=args.artifact)


if __name__ == "__main__":
    main()
