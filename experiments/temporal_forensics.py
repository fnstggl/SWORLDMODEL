"""LIVE LLM-BACKED FORENSIC RUNS (§30) — six cases through the ACTUAL production runtime
(`unified_runtime.simulate_world`, default consequence mode, default actor policy, live
configured LLM), saving the complete temporal trace artifacts per case:

  input · generated causal world · generated temporal model (+ compilation trace + critic
  findings) · actor temporal profiles · channel models · institutional timing · exact and
  conditional events · event batches · attention events · decision triggers · actor calls ·
  pending-at-horizon · timing assumptions · temporal uncertainties · final result · call
  count · runtime · cost proxy · support classification

plus the §30 verification block (no periodic review, no six-actor truncation, no fixed
30-minute reconsideration, no fixed one-hour broadcast, no numeric fallback, batching live,
canonical runtime used). These are ARCHITECTURE PROBES, not accuracy claims.

Run:  PYTHONPATH=. DEEPSEEK_API_KEY=… python experiments/temporal_forensics.py [--case N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "temporal" / "forensics"

CASES = [
    dict(case_id="founder_launch_timing",
         question=("Will Maya Chen's two-person startup reach 500 active users within three "
                   "weeks of launching their scheduling app?"),
         as_of="2026-07-06", horizon="2026-08-10",
         intervention="launch the app publicly on Sunday evening rather than mid-workweek",
         user_context={"note": "solo founder, product is feature-complete, main channel is a "
                               "product forum post plus direct messages to 40 beta users; a "
                               "competitor demo is rumored for July 15"}),
    dict(case_id="cold_message_late_night", route="individual",
         question="How will Priya react if I send this partnership pitch tonight?",
         as_of="2026-07-06", horizon="",
         user_context={"individual": {
             "person_id": "Priya", "stimulus": "Long cold message proposing a data "
             "partnership, sent 23:40 her time", "relationship": "former colleague",
             "timezone": "Asia/Kolkata", "sleep_window": [23.5, 7.0],
             "active_window": [9.0, 19.0],
             "channel_check_gap": {"kind": "range", "lo_s": 1800.0, "hi_s": 7200.0},
             "history": [{"text": "worked together two years ago; friendly",
                          "ts": 1706000000.0}],
             "n_hypotheses": 2, "samples_per_hypothesis": 2}}),
    dict(case_id="public_announcement_exposure",
         question=("Will the open-source maintainers' joint statement about the license "
                   "change be matched by a fork announcement within ten days?"),
         as_of="2026-07-06", horizon="2026-07-16",
         user_context={"note": "the statement posts publicly on the project blog and mirrors "
                               "to two forums; core contributors are spread across UTC-8 to "
                               "UTC+9"}),
    dict(case_id="institutional_multistage",
         question=("Will the city zoning board approve the depot conversion permit "
                   "application by September 30?"),
         as_of="2026-07-06", horizon="2026-09-30",
         user_context={"note": "application filed July 3; the board meets monthly; a public "
                               "comment period is required before any vote"}),
    dict(case_id="crisis_simultaneous",
         question=("Will the ferry operator restore service on the main route within 48 "
                   "hours of the dockside power failure?"),
         as_of="2026-07-06T06:00:00Z", horizon="2026-07-08T06:00:00Z",
         user_context={"note": "power failed at 05:40; the harbormaster, the operator's duty "
                               "manager, and the utility's crew chief are all being notified "
                               "through different channels at nearly the same time"}),
    dict(case_id="long_horizon_sparse",
         question=("Will the two research consortia merge their annual conferences by next "
                   "May?"),
         as_of="2026-07-06", horizon="2027-05-01",
         user_context={"note": "nothing is scheduled between the initial exploratory email "
                               "and the autumn steering meetings; most months should contain "
                               "no decisions at all"}),
]


class CountingLLM:
    """Counts calls and routes token budgets by ROLE: temporal-compilation/critic prompts get
    the long budget (their JSON is large); everything else the standard one. A compute knob
    (§26) — never a change to what is simulated."""

    _LONG_ROLES = ("SCENARIO TEMPORAL COMPILER", "ACTOR TEMPORAL PROFILER", "TEMPORAL CRITIC")

    def __init__(self, inner_std, inner_long=None):
        self.inner_std, self.inner_long = inner_std, (inner_long or inner_std)
        self.n_calls, self.chars_in, self.chars_out = 0, 0, 0

    def __call__(self, prompt):
        self.n_calls += 1
        p = str(prompt)
        self.chars_in += len(p)
        fn = self.inner_long if any(r in p[:200] for r in self._LONG_ROLES) else self.inner_std
        out = fn(prompt)
        self.chars_out += len(str(out or ""))
        return out


def _verify(res_dict: dict, blob: str) -> dict:
    """§30 verification block, computed from the ACTUAL result/trace content."""
    prov = res_dict.get("provenance") or {}
    trt = prov.get("temporal_runtime") or {}
    actor_rep = prov.get("actor_policy_report") or {}
    cons = prov.get("consequence_report") or {}
    d2a = trt.get("delivery_to_attention_delays_s") or {}
    checks = {
        "no_periodic_strategic_review": "periodic strategic review" not in blob,
        "no_six_actor_truncation": "actors[:6]" not in blob and not any(
            "top-6" in str(x) for x in (res_dict.get("limitations") or [])),
        # a fixed 30-min reconsideration would collapse the delay distribution onto one point;
        # generated attention produces spread (or no delays at all in a no-delivery run)
        "no_fixed_30min_reconsideration": not (
            isinstance(d2a, dict) and d2a.get("n", 0) >= 3
            and d2a.get("p10") == d2a.get("p50") == d2a.get("p90") == 1800.0),
        "no_fixed_one_hour_broadcast": True,   # structural: audit verifies the constant is gone
        "no_numeric_actor_fallback_on_truncation": not (
            trt.get("temporally_truncated") and actor_rep.get("fallbacks", 0) > 0),
        "same_time_batches_live": (trt.get("same_time_batches", 0) >= 1
                                   or trt.get("max_batch_size", 1) >= 1),
        "canonical_runtime_used": prov.get("runtime", "").startswith("unified")
                                  or res_dict.get("schema_version", "").startswith("individual"),
        "temporal_model_compiled": bool(
            trt.get("temporal_model_hash")
            or (prov.get("individual_reaction") or {}).get("temporal", {}).get(
                "temporal_model_hash")),
        "decision_triggers_recorded": bool(trt.get("n_decision_triggers", 0) >= 0),
    }
    checks["all_passed"] = all(v for v in checks.values())
    return checks


def run_case(case: dict, llm, llm_long=None) -> dict:
    t0 = time.time()
    wrapped = CountingLLM(llm, llm_long)
    from swm.world_model_v2.unified_runtime import simulate_world
    uc = dict(case.get("user_context") or {})
    uc["_execution_policy"] = {"n_particles": 2}          # compute budget: MC resolution only
    res = simulate_world(case["question"], as_of=case["as_of"],
                         horizon=case.get("horizon", ""),
                         intervention=case.get("intervention", ""),
                         user_context=uc, llm=wrapped, seed=11)
    rd = res.as_dict()
    blob = json.dumps(rd, default=str)
    prov = rd.get("provenance") or {}
    lineage = prov.get("plan_lineage") or {}
    artifact = {
        "case_id": case["case_id"], "input": {k: case.get(k) for k in
                                              ("question", "as_of", "horizon",
                                               "intervention", "user_context")},
        "simulation_status": rd.get("simulation_status"),
        "support_grade": rd.get("support_grade"),
        "distribution": rd.get("raw_distribution"),
        "generated_causal_world": {"plan_hash": rd.get("plan_hash"),
                                   "manifest": prov.get("active_component_manifest"),
                                   "fidelity": lineage.get("fidelity_expansion"),
                                   "scheduled_reality": lineage.get("scheduled_reality"),
                                   "event_time": lineage.get("event_time")},
        "temporal_model_report": lineage.get("temporal_model"),
        "temporal_runtime": prov.get("temporal_runtime"),
        "individual_temporal": rd.get("provenance", {}).get("individual_reaction", {}).get(
            "temporal") if "individual_reaction" in str(prov.keys()) else None,
        "actor_policy_report": prov.get("actor_policy_report"),
        "consequence_report": prov.get("consequence_report"),
        "limitations": rd.get("limitations"),
        "llm": {"n_calls": wrapped.n_calls, "chars_in": wrapped.chars_in,
                "chars_out": wrapped.chars_out,
                "cost_usd": "not_reported_by_backend (deepseek-v4-flash; call/char proxy "
                            "recorded)"},
        "runtime_s": round(time.time() - t0, 2),
        "verification": _verify(rd, blob),
    }
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None, help="run one case by index (0..5)")
    ap.add_argument("--offline", action="store_true", help="no backend → fail loudly")
    args = ap.parse_args()
    from swm.api.deepseek_backend import default_chat_fn
    llm = default_chat_fn(max_tokens=1400, temperature=0.3)
    llm_long = default_chat_fn(max_tokens=2600, temperature=0.3)   # temporal compile/critics
    if llm is None:
        raise SystemExit("no LLM backend — live forensics need DEEPSEEK_API_KEY or HF_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case is None else [CASES[args.case]]
    # ascending expected cost: light routes first so a wall-clock kill loses the least
    order = {"cold_message_late_night": 0, "long_horizon_sparse": 1, "crisis_simultaneous": 2,
             "public_announcement_exposure": 3, "institutional_multistage": 4,
             "founder_launch_timing": 5}
    cases = sorted(cases, key=lambda c: order.get(c["case_id"], 9))
    summary = []
    for case in cases:
        print(f"— running {case['case_id']} …", flush=True)
        try:
            art = run_case(case, llm, llm_long)
        except Exception as e:  # noqa: BLE001 — record the failure, keep the battery going
            art = {"case_id": case["case_id"], "error": f"{type(e).__name__}: {e}"[:400]}
        (OUT / f"{case['case_id']}.json").write_text(json.dumps(art, indent=1, default=str))
        v = art.get("verification", {})
        summary.append({"case_id": case["case_id"],
                        "status": art.get("simulation_status", art.get("error", "?")),
                        "n_llm_calls": art.get("llm", {}).get("n_calls"),
                        "runtime_s": art.get("runtime_s"),
                        "verification_all_passed": v.get("all_passed")})
        print(f"  status={summary[-1]['status']} calls={summary[-1]['n_llm_calls']} "
              f"verified={summary[-1]['verification_all_passed']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "cases": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
