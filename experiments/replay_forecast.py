"""Temporal Replay Laboratory — FORECASTER (sealed: never touches the resolution store).

Per event × cutoff × arm, run the FULL canonical runtime (simulate_world — every phase on the one path):

  arm blinded_current_llm      — evidence gathered strictly as-of by CODE (server-verified timestamps where
                                 available), then every LLM-visible text (question + evidence) pseudonymized
                                 through a causal-blinding mapping; leakage probes recorded per row.
  arm cutoff_prompted_unblinded— the current product path on real identities with the as-of cutoff. This arm
                                 is contamination_not_excluded BY CONSTRUCTION and is diagnostic only; it
                                 never enters the clean headline.

Every row is frozen with a content hash + the runtime fingerprint BEFORE any scoring. The scorer
(replay_score.py, REPLAY_SCORER=1, separate process) is the only component that reads outcomes.

Arm 1 (verified pre-cutoff model checkpoint) is NOT available for this backend; its absence is recorded in
the artifact rather than papered over.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from swm.replay.vault import public_events, freeze_hash, VAULT
from swm.replay.blinding import build_mapping, blind_question, blind_bundle
from swm.replay.probes import run_probes

OUT = Path("experiments/results/replay")
ART = OUT / "forecasts.json"
MAPS = VAULT / "blinding_mappings.json"      # forecaster-side (no outcomes); merged into sealed by scorer

ARMS = ("blinded_current_llm", "cutoff_prompted_unblinded")


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2200, temperature=0.2)


def _p_yes(res) -> float | None:
    """P(affirmative) from the SimulationResult (binary convenience projection, else the raw distribution's
    affirmative-first option)."""
    for attr in ("calibrated_probability", "raw_probability"):
        v = getattr(res, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    d = getattr(res, "raw_distribution", None) or {}
    if not d:
        return None
    keys = list(d)
    for k in keys:
        if str(k).lower() in ("true", "yes"):
            return float(d[k])
    return float(d[keys[0]])                                  # affirmative-first contract


def _bundle_for(question, as_of, horizon, llm, seed=0):
    """Strict as-of retrieval by CODE (the time-capsule stage). Returns the frozen bundle."""
    from swm.world_model_v2.compiler import compile_world
    from swm.world_model_v2.evidence_orchestrator import OrchestratorConfig, gather_evidence
    from swm.world_model_v2.evidence_requirements import requirements_from_plan
    plan = compile_world(question, llm=llm, evidence="", as_of=as_of, horizon=horizon, seed=seed)
    reqs = requirements_from_plan(plan, as_of_iso=as_of, question=question)
    return gather_evidence(question, as_of=as_of, requirements=reqs, llm=llm,
                           config=OrchestratorConfig(), plan_hash=plan.plan_hash(), seed=seed)


def run(limit=None, arms=ARMS, seed=0):
    from swm.world_model_v2.unified_runtime import simulate_world
    from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    fp = runtime_fingerprint()
    rows = []
    if ART.exists():
        rows = [r for r in json.loads(ART.read_text()).get("rows", []) if not r.get("error")]
    done = {(r["event_id"], r["cutoff"], r["arm"]) for r in rows}
    mappings = json.loads(MAPS.read_text())["mappings"] if MAPS.exists() else {}
    events = public_events()[:limit] if limit else public_events()
    for ev in events:
        for cutoff in ev.forecast_cutoffs:
            # shared per (event, cutoff): the frozen bundle + the blinding mapping + the probes
            bundle = None
            for arm in arms:
                if (ev.event_id, cutoff, arm) in done:
                    continue
                rec = {"event_id": ev.event_id, "cluster": ev.cluster, "cutoff": cutoff,
                       "horizon": ev.horizon, "arm": arm, "runtime_fingerprint": fp["fingerprint_hash"],
                       "arm1_pre_cutoff_model": "unavailable_for_backend (recorded, not papered over)"}
                try:
                    if arm == "blinded_current_llm":
                        if bundle is None:
                            bundle = _bundle_for(ev.question, cutoff, ev.horizon, llm, seed=seed)
                        mapping = mappings.get(ev.event_id) or build_mapping(ev.question, llm)
                        mappings[ev.event_id] = mapping
                        MAPS.write_text(json.dumps({"note": "pseudonym mappings (no outcomes)",
                                                    "mappings": mappings}, indent=1))
                        bq = blind_question(ev.question, mapping)
                        import copy as _copy
                        bb = blind_bundle(_copy.deepcopy(bundle), mapping)
                        rec["blinded_question"] = bq
                        rec["probes"] = run_probes(llm, real_question=ev.question,
                                                   blinded_question=bq, mapping=mapping)
                        res = simulate_world(bq, as_of=cutoff, horizon=ev.horizon, llm=llm,
                                             seed=seed, prebuilt_bundle=bb)
                    else:
                        res = simulate_world(ev.question, as_of=cutoff, horizon=ev.horizon, llm=llm,
                                             seed=seed)
                    rec.update({
                        "p_yes": _p_yes(res), "status": getattr(res, "simulation_status", ""),
                        "support_grade": getattr(res, "support_grade", ""),
                        "evidence_n_claims": len((getattr(res, "provenance", {}) or {})
                                                 .get("evidence_bundle_hash", "")) and None,
                        "manifest_executed": sorted(
                            k for k, v in ((getattr(res, "provenance", {}) or {})
                                           .get("active_component_manifest", {}) or {}).items()
                            if isinstance(v, dict) and v.get("executed"))})
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:180]
                rec["freeze_hash"] = freeze_hash({k: v for k, v in rec.items() if k != "freeze_hash"})
                rows.append(rec)
                ART.write_text(json.dumps(
                    {"note": "FROZEN forecasts — content-hashed before any resolution access. "
                             "Scored only by replay_score.py (REPLAY_SCORER=1).",
                     "rows": rows}, indent=1))
                print(f"{ev.event_id:16s} {cutoff} {arm:26s} p={rec.get('p_yes')} "
                      f"err={rec.get('error', '')[:60]}")
    print(f"\n{len(rows)} frozen forecast rows -> {ART}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N events")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
