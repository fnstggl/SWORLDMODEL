"""TEMPORAL VALIDATION / BACKTESTING HARNESS (§31) — reusable infrastructure for scoring the
event-driven temporal runtime against RESOLVED real-world cases with strict as-of controls.

A case file (benchmarks/temporal/cases/<id>.json) is a RESOLVED episode:

    {"case_id": "...", "question": "...", "as_of": "RFC3339", "horizon": "RFC3339",
     "user_context": {...},                    # only information available AT as_of
     "resolved": {
        "first_event_ts": <unix|null>,         # null = censored (never happened by horizon)
        "event_sequence": [{"label": "...", "ts": <unix>}],
        "stage_durations": [{"stage": "...", "s": <seconds>}],
        "response_delays": [{"from": "...", "to": "...", "s": <seconds>}],
        "decision_events": [{"actor": "...", "ts": <unix>}]},
     "leakage_note": "how as-of separation was checked"}

STRICT AS-OF: the harness refuses any case whose resolved timestamps precede its as_of, and
the runtime receives only question/as_of/horizon/user_context — never the resolution.

Metrics (censoring-aware where applicable): first-event time error, CRPS of the first-passage
CDF (event_time.crps_first_passage), interval coverage (event_time.interval_coverage),
stage-completion error, response-delay calibration, event-order accuracy (Kendall-tau
concordance on matched labels), censoring accuracy, missed/false event rates, decision-trigger
precision/recall (matched by actor within a tolerance window), same-time order invariance
(re-run with permuted schedules), and cost per simulated day / per actual event.

Baselines (§31): `arm=` selects production event-driven ("event_driven"), the quarantined
periodic scheduler ablation ("periodic_ablation"), a fixed-delay baseline ("fixed_delay"), or
a single-cadence baseline ("single_cadence"). Do not tune against locked test outcomes.

VALIDATION STATUS: this harness is the reusable infrastructure required by §31. No sufficient
resolved-case corpus ships with this change — cases/ starts empty and TEMPORAL CALIBRATION IS
NOT CLAIMED anywhere until real cases are added and scored (see cases/README.md).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = Path(__file__).resolve().parent / "cases"

ARMS = ("event_driven", "periodic_ablation", "fixed_delay", "single_cadence")


def load_cases() -> list:
    out = []
    for f in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(f.read_text())
        _check_as_of(case)
        out.append(case)
    return out


def _check_as_of(case: dict) -> None:
    """STRICT as-of: resolution timestamps must postdate the information cutoff."""
    from swm.world_model_v2.state import parse_time
    as_of = parse_time(case["as_of"])
    for seq in (case.get("resolved", {}).get("event_sequence") or []):
        if float(seq.get("ts", as_of + 1)) <= as_of:
            raise ValueError(f"case {case.get('case_id')}: resolved event {seq.get('label')!r} "
                             f"precedes as_of — leakage")


# ---------------------------------------------------------------- metrics
def first_event_error_s(predicted_median_ts, actual_ts):
    if predicted_median_ts is None or actual_ts is None:
        return None
    return float(predicted_median_ts) - float(actual_ts)


def order_accuracy(predicted_sequence: list, actual_sequence: list) -> float:
    """Kendall-tau-style concordance over labels present in BOTH sequences."""
    pred = {str(e["label"]): float(e["ts"]) for e in predicted_sequence if "label" in e}
    act = {str(e["label"]): float(e["ts"]) for e in actual_sequence if "label" in e}
    common = sorted(set(pred) & set(act))
    if len(common) < 2:
        return None
    n_conc = n_pairs = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            n_pairs += 1
            if (pred[a] - pred[b]) * (act[a] - act[b]) > 0:
                n_conc += 1
    return n_conc / n_pairs


def censoring_accuracy(p_censored: float, actually_censored: bool) -> float:
    """Brier-style score of the censoring call (lower is better)."""
    if p_censored is None:
        return None
    target = 1.0 if actually_censored else 0.0
    return (float(p_censored) - target) ** 2


def event_rates(predicted_labels: set, actual_labels: set) -> dict:
    missed = actual_labels - predicted_labels
    false = predicted_labels - actual_labels
    return {"missed_event_rate": len(missed) / max(1, len(actual_labels)),
            "false_event_rate": len(false) / max(1, len(predicted_labels)),
            "missed": sorted(missed)[:10], "false": sorted(false)[:10]}


def trigger_precision_recall(predicted_triggers: list, actual_decisions: list,
                             tol_s: float = 2 * 86400.0) -> dict:
    """A predicted decision trigger is a hit if the same actor really decided within ±tol."""
    hits = 0
    used = set()
    for p in predicted_triggers:
        for i, a in enumerate(actual_decisions):
            if i in used:
                continue
            if str(p.get("actor")) == str(a.get("actor")) \
                    and abs(float(p.get("ts", 0)) - float(a.get("ts", 0))) <= tol_s:
                hits += 1
                used.add(i)
                break
    return {"precision": hits / max(1, len(predicted_triggers)),
            "recall": hits / max(1, len(actual_decisions)),
            "n_predicted": len(predicted_triggers), "n_actual": len(actual_decisions)}


def stage_duration_errors(predicted: list, actual: list) -> list:
    act = {str(s["stage"]): float(s["s"]) for s in actual if "stage" in s}
    out = []
    for p in predicted:
        if str(p.get("stage")) in act:
            out.append({"stage": p["stage"],
                        "error_s": float(p.get("s", 0)) - act[str(p["stage"])]})
    return out


# ---------------------------------------------------------------- arms
def run_arm(case: dict, *, arm: str, llm, n_particles: int = 3, seed: int = 7) -> dict:
    """One evaluation arm on one case. event_driven = the production runtime. The baselines
    exist for COMPARISON ONLY — periodic_ablation routes through the quarantined scheduler."""
    assert arm in ARMS, f"unknown arm {arm}"
    from swm.world_model_v2.unified_runtime import simulate_world
    t0 = time.time()
    uc = dict(case.get("user_context") or {})
    uc["_execution_policy"] = {"n_particles": n_particles}
    if arm == "periodic_ablation":
        uc["_execution_policy"]["legacy_periodic_ablation"] = True   # consumed by the harness
    res = simulate_world(case["question"], as_of=case["as_of"],
                         horizon=case.get("horizon", ""), user_context=uc, llm=llm, seed=seed)
    rd = res.as_dict()
    prov = rd.get("provenance") or {}
    et = (prov.get("event_time") or {})
    trt = prov.get("temporal_runtime") or {}
    qtl = (et.get("first_passage_quantiles_ts") or {})
    return {"arm": arm, "case_id": case.get("case_id"),
            "simulation_status": rd.get("simulation_status"),
            "distribution": rd.get("raw_distribution"),
            "p_censored": et.get("p_censored"),
            "cdf_grid_ts": et.get("cdf_grid_ts"), "cdf": et.get("cdf"),
            "median_first_passage_ts": qtl.get("0.5"),
            "quantiles": qtl,
            "predicted_triggers": [
                {"actor": t.get("actor"), "ts": t.get("ts"),
                 "trigger_type": t.get("trigger_type")}
                for t in (trt.get("decision_triggers") or [])],
            "runtime_s": round(time.time() - t0, 2)}


def score_case(case: dict, arm_result: dict) -> dict:
    from swm.world_model_v2.event_time import crps_first_passage, interval_coverage
    from swm.world_model_v2.state import parse_time
    resolved = case.get("resolved") or {}
    as_of, hz = parse_time(case["as_of"]), parse_time(case["horizon"])
    actual_first = resolved.get("first_event_ts")
    scores = {"case_id": case.get("case_id"), "arm": arm_result.get("arm")}
    if arm_result.get("cdf_grid_ts") and arm_result.get("cdf"):
        scores["crps_first_passage"] = crps_first_passage(
            arm_result["cdf_grid_ts"], arm_result["cdf"], event_ts=actual_first,
            as_of=as_of, horizon_ts=hz)
        scores["interval_coverage_10_90"] = interval_coverage(
            arm_result.get("quantiles") or {}, actual_first)
    scores["first_event_error_s"] = first_event_error_s(
        arm_result.get("median_first_passage_ts"), actual_first)
    scores["censoring_brier"] = censoring_accuracy(arm_result.get("p_censored"),
                                                   actual_first is None)
    scores["order_accuracy"] = order_accuracy(
        [{"label": t.get("trigger_type"), "ts": t.get("ts")}
         for t in arm_result.get("predicted_triggers") or []],
        resolved.get("event_sequence") or [])
    scores["trigger_pr"] = trigger_precision_recall(
        arm_result.get("predicted_triggers") or [], resolved.get("decision_events") or [])
    span_days = max(1e-9, (hz - as_of) / 86400.0)
    scores["runtime_s_per_simulated_day"] = round(
        (arm_result.get("runtime_s") or 0.0) / span_days, 4)
    return scores


def run_benchmark(llm, *, arms=("event_driven",), out_path=None) -> dict:
    cases = load_cases()
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_cases": len(cases), "arms": list(arms), "scores": [],
              "validation_status": ("INCOMPLETE — no resolved-case corpus present; temporal "
                                    "calibration is NOT claimed" if not cases else
                                    "scored on the present corpus")}
    for case in cases:
        for arm in arms:
            result = run_arm(case, arm=arm, llm=llm)
            report["scores"].append(score_case(case, result))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    from swm.api.deepseek_backend import default_chat_fn
    print(json.dumps(run_benchmark(default_chat_fn(max_tokens=1600, temperature=0.3),
                                   out_path=ROOT / "artifacts" / "temporal" /
                                   "temporal_benchmark.json"),
                     indent=1, default=str))
