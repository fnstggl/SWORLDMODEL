"""§39/§40 reusable core-architecture ablations + cost accounting (offline, deterministic).

Each arm runs the SAME small scripted scenario through the real runtime seams with exactly one
architectural difference, then reports what changed: statuses, truncation, decision context,
LLM-call counts. These are ARCHITECTURE ablations measured with scripted backends — they
demonstrate mechanical behavior differences and cost deltas. They make NO accuracy-superiority
claim: superiority requires held-out outcomes, which this harness does not touch (§39).

Run:  PYTHONPATH=. python experiments/core_arch_ablations.py
Writes artifacts/core_arch/ablation_report.json and cost_report.json.
"""
import json
import os
import sys
import time

os.environ.pop("SWM_ALLOW_NUMERIC_BASELINE", None)      # strict default unless an arm opts in
os.environ.pop("SWM_ALLOW_GENERIC_PRIOR", None)

OUT_DIR = "artifacts/core_arch"
os.makedirs(OUT_DIR, exist_ok=True)


class ScriptedLLM:
    """Deterministic backend: routes by prompt markers; counts calls by stage."""

    def __init__(self, fail: bool = False):
        self.calls = {"attention": 0, "interpretation": 0, "search": 0, "decision": 0,
                      "other": 0}
        self.fail = fail
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if "ATTENTION process" in prompt:
            self.calls["attention"] += 1
            return json.dumps({"noticed": [{"obs_id": "ob0", "why": "directly relevant"}],
                               "missed": [{"obs_id": "ob1", "why": "buried under workload"}]})
        if "making private sense" in prompt:
            self.calls["interpretation"] += 1
            return json.dumps({"what_happened": "the committee asked for my decision",
                               "why_it_matters": "my project depends on it",
                               "perceived_sender_or_cause_intent": "genuine request",
                               "activated_memories": [], "active_belief": "",
                               "perceived_opportunities": ["settle it now"],
                               "perceived_threats": ["losing the room"],
                               "unresolved_ambiguity": "whether support holds"})
        if "options even OCCUR" in prompt:
            self.calls["search"] += 1
            return json.dumps({"options_recalled": ["approve"],
                               "options_generated": ["delay for one week"],
                               "options_screened_out": [{"option": "escalate",
                                                         "why_dismissed": "too aggressive"}],
                               "shortlist": ["approve", "delay for one week"]})
        self.calls["decision"] += 1
        if self.fail:
            raise RuntimeError("scripted provider failure")
        return json.dumps({
            "schema_version": "qualitative.actor.v1",
            "decision": {"act_or_wait": "act", "chosen_action": "approve", "target": "board",
                         "timing": "immediate", "observability": "public",
                         "intended_effect": "secure approval"},
            "decision_summary": "I approve now",
            "situation_interpretation": {"what_changed": "decision point"},
            "novel_action_proposal": {"present": False}})


def _world():
    from tests.test_llm_actor import world
    return world()


def _decision():
    from tests.test_llm_actor import DECISION
    return dict(DECISION)


def _runtime(llm, **cfg):
    from swm.world_model_v2.qualitative_actor import (QualitativeActorPolicyRuntime,
                                                      QualitativeConfig,
                                                      QualitativeDecisionEngine)
    defaults = dict(llm=llm, llm_hypotheses=False, n_hypotheses=2)
    defaults.update(cfg)
    return QualitativeActorPolicyRuntime(QualitativeDecisionEngine(QualitativeConfig(**defaults)),
                                         mode="persistent_qualitative_llm_policy")


def arm_bounded_vs_oneshot():
    """§39: one-shot actor prompt vs bounded-cognition pipeline."""
    out = {}
    for label, bc in (("one_shot_prompt", False), ("bounded_cognition", True)):
        os.environ["SWM_ALLOW_NUMERIC_BASELINE"] = "1"    # hypothesizer offline arm
        try:
            llm = ScriptedLLM()
            rt = _runtime(llm, bounded_cognition=bc)
            w = _world()
            w.branch_id = "b000"
            sel, post, _tr = rt.decide(None, [w], "alice", decision=_decision(), seed=1)
            cog = (post.provenance or {}).get("cognition") or {}
            out[label] = {
                "llm_calls_by_stage": dict(llm.calls),
                "total_llm_calls": sum(llm.calls.values()),
                "decision_prompt_has_full_ledger": "nothing new observed" not in
                                                   (llm.prompts[-1] if llm.prompts else ""),
                "observations_missed_recorded": len(cog.get("observations_missed", [])),
                "working_memory_capacity": cog.get("working_memory_capacity"),
                "options_considered": cog.get("options_considered"),
                "selected": sel.action_name}
        finally:
            os.environ.pop("SWM_ALLOW_NUMERIC_BASELINE", None)
    out["measured_difference"] = {
        "extra_calls_for_bounded_cognition":
            out["bounded_cognition"]["total_llm_calls"] - out["one_shot_prompt"]["total_llm_calls"],
        "note": "accuracy-increasing additional work (§40) — staged attention/interpretation/"
                "search calls; the decision call sees only surviving material"}
    return out


def arm_truncation_vs_numeric():
    """§39: numerical fallback vs honest truncation under provider failure."""
    from swm.world_model_v2.qualitative_actor import ActorDecisionUnavailable
    out = {}
    # strict (default): provider failure truncates
    llm = ScriptedLLM(fail=True)
    rt = _runtime(llm, bounded_cognition=False, llm_hypotheses=False)
    w = _world()
    w.branch_id = "b000"
    try:
        rt.decide(None, [w], "alice", decision=_decision(), seed=2)
        out["strict_default"] = {"behavior": "UNEXPECTED_COMPLETION"}
    except ActorDecisionUnavailable as e:
        out["strict_default"] = {"behavior": "branch_truncates", "reason": e.reason,
                                 "substitute_decision": None}
    # explicit numeric-baseline arm: legacy numeric fallback serves, loudly marked
    os.environ["SWM_ALLOW_NUMERIC_BASELINE"] = "1"
    try:
        llm2 = ScriptedLLM(fail=True)
        rt2 = _runtime(llm2, bounded_cognition=False)
        w2 = _world()
        w2.branch_id = "b001"
        sel2, post2, _ = rt2.decide(None, [w2], "alice", decision=_decision(), seed=2)
        q = (post2.provenance or {}).get("qualitative") or {}
        out["explicit_numeric_baseline"] = {"behavior": "numeric_fallback_served",
                                            "decision_source": q.get("decision_source"),
                                            "marked_excluded":
                                                q.get("excluded_from_qualitative_aggregation")}
    finally:
        os.environ.pop("SWM_ALLOW_NUMERIC_BASELINE", None)
    out["measured_difference"] = {"psychology_switch_on_default_path": False,
                                  "note": "the default branch STOPS; the numeric psychology "
                                          "exists only behind the explicit baseline marker"}
    return out


def arm_generic_prior_vs_under_modeled():
    """§39: generic outcome prior vs under-modeled classification."""
    from swm.world_model_v2.fallback import GenericOutcomeOperator
    from swm.world_model_v2.events import Event
    out = {}
    for label, allow in (("under_modeled_default", False), ("explicit_prior_baseline", True)):
        if allow:
            os.environ["SWM_ALLOW_GENERIC_PRIOR"] = "1"
        else:
            os.environ.pop("SWM_ALLOW_GENERIC_PRIOR", None)
        try:
            w = _world()
            w.branch_id = "b000"
            op = GenericOutcomeOperator()
            ev = Event(ts=w.clock.now, etype="resolve_outcome", participants=[],
                       payload={"outcome_var": "outcome", "family": "binary",
                                "lean": "neutral", "options": ["yes", "no"]})
            proposal = op.propose(w, ev, __import__("random").Random(3))
            delta = op.apply(w, proposal)
            resolved = "outcome" in w.quantities and w.quantities["outcome"].value is not None
            sup = getattr(getattr(w, "temporal_stats", None), "mechanism_suppressions", [])
            out[label] = {"terminal_written": resolved,
                          "suppressions_recorded": len(sup),
                          "delta_reasons": list(getattr(delta, "reason_codes", []))[:3]}
        finally:
            os.environ.pop("SWM_ALLOW_GENERIC_PRIOR", None)
    out["measured_difference"] = {
        "default_writes_terminal": out["under_modeled_default"]["terminal_written"],
        "note": "the default refuses the broad-prior terminal draw and records the missing "
                "mechanism; the explicit baseline arm still draws"}
    return out


def arm_outside_world():
    """§39: no outside-world process vs residual process (availability difference)."""
    from swm.world_model_v2.outside_world import (ArrivalModel, ExternalEventFamily,
                                                  OutsideWorldProcess, validate_entry)
    from swm.world_model_v2.boundary_monitor import schedule_outside_arrivals
    from swm.world_model_v2.events import EventQueue
    from types import SimpleNamespace
    out = {}
    for label, attach in (("no_residual", False), ("residual_process", True)):
        w = _world()
        w.branch_id = "b000"
        horizon = w.clock.now + 30 * 86400
        fam = validate_entry(ExternalEventFamily(
            family_id="platform_policy_change", description="platform policy shifts",
            marks=["the platform announced a fee change"],
            affected_boundary_components=["distribution channel"],
            impact_mechanism="observation_delivery",
            arrival=ArrivalModel(kind="observed_base_rate", rate_per_day=0.5,
                                 provenance="documented industry base rate (ablation fixture)")))
        proc = OutsideWorldProcess(boundary_id="wb_x", families=[fam] if attach else [],
                                   empty_residual_justification="" if attach else
                                   "ablation arm: residual deliberately removed")
        plan = SimpleNamespace(_outside_world=proc, horizon_ts=horizon)
        q = EventQueue(horizon_ts=horizon)
        n = schedule_outside_arrivals(plan, w, q)
        out[label] = {"outside_events_scheduled": n,
                      "queue_events": len(q.events)}
    out["measured_difference"] = {
        "events_only_with_residual": out["residual_process"]["outside_events_scheduled"] > 0
                                     and out["no_residual"]["outside_events_scheduled"] == 0,
        "note": "the explicit-boundary world without a residual process never experiences "
                "outside shocks — the ablation the residual exists to close"}
    return out


def main():
    t0 = time.time()
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "claims": "architecture-behavior differences only; NO accuracy superiority is "
                        "claimed (held-out outcome evaluation not run here — §39)",
              "arms": {}}
    for name, fn in (("bounded_vs_oneshot", arm_bounded_vs_oneshot),
                     ("truncation_vs_numeric", arm_truncation_vs_numeric),
                     ("generic_prior_vs_under_modeled", arm_generic_prior_vs_under_modeled),
                     ("outside_world_residual", arm_outside_world)):
        try:
            report["arms"][name] = fn()
            print(f"[arm] {name}: ok", flush=True)
        except Exception as e:  # noqa: BLE001 — a crashed arm is a visible failure
            report["arms"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[arm] {name}: FAILED {type(e).__name__}: {e}", flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "ablation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    # ---------------- §40 cost report: assembled from live forensic artifacts when present ----
    cost = {"note": "per-stage call counts from live forensic cases (CountingLLM meters + "
                    "ensemble cost manifests); ablation arms above separate the "
                    "accuracy-increasing additional work (staged cognition, boundary "
                    "generation+critics) from savings (event-driven attention, caching, "
                    "honest truncation stopping branches early)",
            "cases": {}}
    import glob
    for path in sorted(glob.glob("artifacts/core_arch_forensics/case*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            cost["cases"][os.path.basename(path)] = {
                "provider_cost": rec.get("provider_cost"),
                "ensemble_cost_manifest": rec.get("cost_manifest"),
                "runtime_s": rec.get("runtime_s"),
                "truncated_weight": (rec.get("truncation_report") or {}).get("truncated_weight"),
            }
        except Exception:  # noqa: BLE001
            continue
    with open(os.path.join(OUT_DIR, "cost_report.json"), "w", encoding="utf-8") as f:
        json.dump(cost, f, indent=1, default=str)
    ok = all("error" not in a for a in report["arms"].values())
    print(json.dumps({k: ("ok" if "error" not in v else v["error"])
                      for k, v in report["arms"].items()}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
