"""Three live end-to-end forensic cases through the FINAL DEFAULT combined runtime
(structural ensemble x per-model temporal model x scenario-generated actions x causal
boundary). No enable flags; only the documented compute knob (n_particles) is set so the
cases finish in one sitting.

Prints, per case: structural models, temporal triggers, attempts, mechanisms,
observations, actor decisions, and (case 3) the cross-model action verdict.

Run:  PYTHONPATH=. python experiments/combined_runtime_forensics.py [case1|case2|case3]
"""
import json
import os
import sys
import time

OUT_DIR = "artifacts/combined_runtime_forensics"
os.makedirs(OUT_DIR, exist_ok=True)


def make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(max_tokens=2600, temperature=0.0)


def _sect(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def _dump(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[saved {path}]", flush=True)


def print_ensemble_trace(res, *, header):
    _sect(f"{header} — STRUCTURAL MODELS")
    prov = res.provenance or {}
    handle = getattr(res, "_ensemble_handle", None)
    se = getattr(res, "structural_ensemble", None) or prov.get("structural_ensemble") or {}
    models = se.get("models") if isinstance(se, dict) else None
    if models:
        for m in models:
            print(f"  - {m.get('model_id')}: [{m.get('promotion_status')}] "
                  f"{str(m.get('causal_thesis'))[:110]}")
    if isinstance(se, dict) and se.get("simulation_manifest"):
        for mid, man in se["simulation_manifest"].items():
            print(f"    budget {mid}: pilot={man.get('pilot_particles')} "
                  f"final={man.get('final_particles')} required={man.get('full_budget_required')} "
                  f"status={man.get('status')}")
    _sect(f"{header} — PER-MODEL TEMPORAL MODELS (compiled inside each structural model)")
    if handle is not None:
        for c in handle.surviving():
            p = c.executable_plan
            tm = getattr(p, "temporal_model", None)
            if tm is None:
                print(f"  - {c.model_id}: NO temporal model (degraded)")
                continue
            trig = [f"{t.get('actor')}: {t.get('trigger_type')}"
                    for t in (tm.decision_trigger_sources or [])][:6]
            print(f"  - {c.model_id}: hash={tm.temporal_model_hash()[:12]} "
                  f"channels={sorted(tm.channels)[:5]} "
                  f"institutional_processes={[pr.process_id for pr in tm.institutional_processes][:4]} "
                  f"deadlines={len(tm.deadlines)}")
            print(f"      decision-trigger sources: {trig}")
    _sect(f"{header} — ATTEMPTS / MECHANISMS / OBSERVATIONS / ACTOR DECISIONS (per model)")
    for mid, p in (prov.get("per_model_provenance") or {}).items():
        cr = (p.get("consequence_report") or {})
        ar = (p.get("actor_policy_report") or {})
        tr = (p.get("temporal_runtime") or {})
        print(f"  - {mid}:")
        print(f"      attempts={cr.get('action_attempts')} "
              f"intended_deliveries={cr.get('intended_deliveries')} "
              f"ACTUAL_deliveries={cr.get('actual_deliveries')} "
              f"mech invoked={cr.get('mechanisms_invoked')} ok={cr.get('mechanism_successes')} "
              f"fail={cr.get('mechanism_failures')} unresolved={cr.get('mechanism_unresolved')}")
        print(f"      observations_delivered={cr.get('observations_delivered')} "
              f"attention_events={cr.get('attention_events')} "
              f"actors_reconsidered={cr.get('actors_reconsidered')} "
              f"direct_human_writes={cr.get('human_reactions_written_directly')} "
              f"numeric_fallbacks={cr.get('numeric_fallbacks')}")
        if tr:
            print(f"      temporal runtime: {json.dumps(tr, default=str)[:220]}")
        dd = p.get("actor_decision_distributions") or {}
        for actor, dist in list(dd.items())[:4]:
            print(f"      decisions[{actor}]: {json.dumps(dist, default=str)[:160]}")
        if ar.get("mode"):
            print(f"      actor policy mode: {ar.get('mode')} warning={ar.get('warning')}")
    _sect(f"{header} — TERMINAL")
    print(f"  status={res.simulation_status}  distribution={res.raw_distribution}")
    agg = se.get("aggregation") if isinstance(se, dict) else None
    if agg:
        print(f"  cross-model aggregation: {json.dumps(agg, default=str)[:300]}")
    sens = se.get("sensitivity") if isinstance(se, dict) else None
    if sens:
        print(f"  structural sensitivity: {json.dumps(sens, default=str)[:300]}")


# ------------------------------------------------------------------ case 1: personal comms
def case1(llm):
    """Personal communication: an exact message travels channel → attention → decision,
    through the individual-reaction ensemble route (structural frames, per-frame budgets)."""
    import functools
    from swm.world_model_v2.unified_runtime import simulate_world as _sw_default
    # archival full-fidelity harness: pinned since the §25 default switch
    simulate_world = functools.partial(_sw_default, execution_profile="full_fidelity")
    t0 = time.time()
    res = simulate_world(
        "How will Dana react if I send this message tonight?",
        as_of="2026-07-18T21:30:00Z", horizon="2026-07-20T21:30:00Z", llm=llm, seed=11,
        user_context={"individual": {
            "person_id": "dana", "name": "dana",
            "stimulus": ("Hey Dana — I'm really sorry about how the planning meeting went "
                         "on Tuesday. I should have backed you up on the timeline. Can we "
                         "grab coffee this week and reset?"),
            "channel": "text_message", "relationship": "coworker and friend",
            "role": "senior engineer", "your_role": "project lead",
            "timezone": "America/New_York", "sleep_window": [23.0, 7.0],
            "history": ["argued at the planning meeting on Tuesday",
                        "usually reply to each other within hours",
                        "five years working together"],
            "urgency": 0.2, "n_hypotheses": 2, "samples_per_hypothesis": 2}})
    print(f"[case1 ran in {time.time()-t0:.0f}s]", flush=True)
    _sect("CASE 1 (personal communication) — ROUTE + STRUCTURAL FRAMES")
    se = getattr(res, "structural_ensemble", None) or {}
    print("route:", se.get("route"), " mode:", se.get("structural_mode"),
          " status:", res.simulation_status)
    for m in se.get("models") or []:
        print(f"  frame {m.get('model_id')} [{m.get('promotion_status')}]: "
              f"{str(m.get('causal_thesis'))[:120]}")
        print(f"        prediction: {json.dumps(m.get('prediction'), default=str)[:160]}")
    _sect("CASE 1 — TEMPORAL DELIVERY/ATTENTION + PER-SAMPLE ACTOR DECISIONS (per frame)")
    for mid, art in (se.get("per_frame_artifacts") or {}).items():
        print(f"  frame {mid}: temporal_summary="
              f"{json.dumps((art.get('provenance') or {}).get('temporal_route'), default=str)}")
        for s in (art.get("samples") or [])[:6]:
            print(f"    [{s.get('trace_id')}] state={s.get('temporal_state')} "
                  f"src={s.get('decision_source')} "
                  f"resp={str(s.get('observable_response'))[:56]!r}")
            if s.get("delivery_provenance"):
                print(f"        delivery={str(s.get('delivery_provenance'))[:80]} "
                      f"notice={str(s.get('notice_provenance'))[:80]}")
    print("mixture distribution:", json.dumps(res.raw_distribution, default=str)[:400])
    print("structural sensitivity:",
          json.dumps(se.get("structural_sensitivity"), default=str)[:220])
    _dump("case1_personal_communication.json",
          {"result": res.as_dict() if hasattr(res, "as_dict") else vars(res)})
    return res


# ------------------------------------------------------------------ case 2: institutional
def case2(llm):
    """Institutional decision through the full default ensemble runtime."""
    from swm.world_model_v2.unified_runtime import simulate_world
    t0 = time.time()
    res = simulate_world(
        "Will the Riverbend city zoning board approve the Harper Street shelter variance "
        "at its August session?",
        as_of="2026-07-18", horizon="2026-09-01", llm=llm, seed=7,
        user_context={"detail": "The shelter nonprofit filed the variance in June; two "
                                "residents' associations oppose it; the board has five "
                                "members and needs a simple majority; the city planner's "
                                "staff report is due before the August session."},
        execution_policy={"n_particles": 6})
    print(f"[case2 ran in {time.time()-t0:.0f}s]", flush=True)
    print_ensemble_trace(res, header="CASE 2 (institutional decision)")
    _dump("case2_institutional_decision.json",
          {"result": res.as_dict() if hasattr(res, "as_dict") else vars(res)})
    return res


# ------------------------------------------------------------------ case 3: best action
def case3(llm):
    """Multi-step best action: default ensemble world → Phase 13 across ALL models with
    scenario-generated candidates → cross-model verdict."""
    from swm.world_model_v2.unified_runtime import simulate_world
    from swm.world_model_v2.phase13.api import recommend_action
    from swm.world_model_v2.phase13.contracts import DecisionProblem
    t0 = time.time()
    res = simulate_world(
        "Will Priya's small robotics company win the Meridian District school-lab "
        "equipment contract this quarter?",
        as_of="2026-07-18", horizon="2026-10-01", llm=llm, seed=5,
        user_context={"detail": "Priya can demo the product, cut the price, or recruit the "
                                "district's STEM coordinator as a champion; the district "
                                "procurement office shortlists in August and the school "
                                "board ratifies in September; a larger competitor has an "
                                "existing relationship with the procurement office."},
        execution_policy={"n_particles": 6})
    print(f"[case3 world ran in {time.time()-t0:.0f}s]", flush=True)
    print_ensemble_trace(res, header="CASE 3 (world for best action)")
    problem = DecisionProblem(
        decision_id="win_contract", decision_maker="priya",
        authority=["vendor"], horizon="2026-10-01T00:00:00Z",
        context="win the Meridian District school-lab equipment contract this quarter")
    t1 = time.time()
    dec = recommend_action(problem, res, llm=llm, seed=5, n_particles=4,
                           goal_text="win the Meridian District school-lab equipment "
                                     "contract this quarter")
    print(f"[case3 decision ran in {time.time()-t1:.0f}s]", flush=True)
    _sect("CASE 3 — CROSS-MODEL ACTION VERDICT")
    print("recommended:", dec.recommended, " kind:", dec.recommendation_kind)
    se = (dec.provenance or {}).get("structural_ensemble") or {}
    print("winner_by_model:", json.dumps(se.get("winner_by_model"), default=str)[:400])
    print("stability:", json.dumps(se.get("recommendation_stability"), default=str)[:200])
    print("robust_action:", json.dumps(se.get("robust_action"), default=str)[:200])
    print("minimax_regret:", json.dumps(se.get("minimax_regret_across_models"),
                                        default=str)[:250])
    _sect("CASE 3 — PER-MODEL GENERATED CANDIDATES + CAUSAL BOUNDARY")
    for mid, r in (se.get("per_model_results") or {}).items():
        p = (r or {}).get("provenance") or {}
        sr = p.get("scenario_report") or {}
        cands = [c.get("candidate_id") if isinstance(c, dict) else str(c)
                 for c in (sr.get("candidates") or [])][:8]
        cb = p.get("causal_consequence_report") or {}
        print(f"  - {mid}: generated candidates={cands}")
        print(f"      steps_fired={cb.get('steps_fired')} mech invoked="
              f"{cb.get('mechanisms_invoked')} ok={cb.get('mechanism_successes')} "
              f"unresolved={cb.get('mechanism_unresolved')} "
              f"actual_deliveries={cb.get('actual_deliveries')}")
        print(f"      recommended in-model: {(r or {}).get('recommended')}")
    _dump("case3_best_action.json", {"decision": dec.as_dict()
                                     if hasattr(dec, "as_dict") else vars(dec)})
    return dec


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    llm = make_llm()
    if which in ("case1", "all"):
        case1(llm)
    if which in ("case2", "all"):
        case2(llm)
    if which in ("case3", "all"):
        case3(llm)
    print("\nDONE", flush=True)
