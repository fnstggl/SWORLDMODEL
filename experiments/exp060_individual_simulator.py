"""EXP-060: Level-1 individual simulator, validated on REAL persuasion data (ChangeMyView).

The full Level-1 assembly, put on real data and asked to earn its place:

  A. RESPONSE PREDICTION (does modeling the PERSON beat modeling only the message?). On 1,200 real CMV
     threads — an OP states a view, a challenger argues, the label is whether the OP was persuaded (a
     delta) — each row carries LLM-inferred PERSON variables (openness, skepticism, entrenchment) and
     MESSAGE variables (addresses-crux, evidence, clarity, respectfulness, expertise). We fit the
     StructuredResponseModel in ablation arms on a TEMPORAL split and score on the held-out future:
       base_rate | message_only (quality) | person_only (receptivity) | INDIVIDUAL (person x message).
     The Level-1 claim is that the INTERACTION (a strong argument moves an open mind far more than an
     entrenched one) beats a message-quality score that ignores who is reading — a real number.

  B. BEST MESSAGE (the action layer, on real do(x)). For OPs who received SEVERAL arguments with MIXED
     outcomes (some persuaded them, some did not), the person is fixed and only the message varies — a
     natural experiment. We rank each such OP's arguments with the individual model and ask: does its
     top-ranked argument actually persuade more often than picking at random? That is `best_message`
     scored causally.

  C. FORWARD STATE (the person as a dynamical system). One person, a sequence of contacts: we show the
     state (mood, attention, cognitive load, reciprocity) evolving so the SAME ask lands differently after
     a pushy opener than after a respectful one — the thing a static variable vector cannot do.

Backend: the structured/grounded response_fn (validated here). Production swaps in `llm_response_fn`
(the LLM reasons as the person) behind the identical interface.
Run: python -m experiments.exp060_individual_simulator
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from swm.api.individual_simulate import IndividualSimulator
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.simulation.response_model import StructuredResponseModel

COMMON = "experiments/results/exp021_cmv/cmv_common.json"
INFER = "experiments/results/exp021_cmv/cmv_inferences.json"
RESULT = "experiments/results/exp060_individual_simulator.json"


def _load():
    common = {r["id"]: r for r in json.loads(Path(COMMON).read_text())}
    infer = {r["id"]: r for r in json.loads(Path(INFER).read_text())}
    rows = []
    for rid, c in common.items():
        f = infer.get(rid)
        if not f:
            continue
        person = {"trait_openness": f["op_openness"], "skepticism": f["op_skepticism"],
                  "certainty_disposition": f["op_entrenchment"]}          # who the OP is
        message = {"addresses_crux": f["arg_addresses_crux"], "evidence": f["arg_evidence"],
                   "clarity": f["arg_clarity"], "politeness_disposition": f["arg_respectfulness"],
                   "expertise": f["arg_expertise"]}                        # the argument's properties
        rows.append({"op_id": c["op_id"], "ts": c["ts"], "person": person, "message": message,
                     "y": int(c["success"])})
    rows.sort(key=lambda r: r["ts"])                                       # temporal order (no leakage)
    return rows


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4), "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


def run():
    rows = _load()
    cut = int(0.7 * len(rows))
    train, test = rows[:cut], rows[cut:]
    train_rows = [(r["person"], {}, r["message"], r["y"]) for r in train]
    yte = [r["y"] for r in test]
    base = sum(r["y"] for r in train) / len(train)

    # --- A. ablation arms on the temporal holdout (features CMV can identify; state has no variation here) ---
    arms = {"message_only": ("quality",), "person_only": ("receptivity",),
            "INDIVIDUAL": ("receptivity", "quality", "interaction")}
    results = {"base_rate": _score(yte, [base] * len(test))}
    models = {}
    for name, feats in arms.items():
        m = StructuredResponseModel(features=feats).fit(train_rows)
        preds = [m(r["person"], {}, r["message"])["p"] for r in test]
        results[name] = _score(yte, preds)
        models[name] = m
    indiv = models["INDIVIDUAL"]
    comparison = {
        "individual_beats_message_only_logloss": round(
            results["message_only"]["log_loss"] - results["INDIVIDUAL"]["log_loss"], 4),
        "individual_beats_base_logloss": round(
            results["base_rate"]["log_loss"] - results["INDIVIDUAL"]["log_loss"], 4)}

    # --- B. best-message KPI on OPs with several TEST args and mixed outcomes (person fixed, msg varies) ---
    sim = IndividualSimulator(response_fn=indiv)
    by_op = defaultdict(list)
    for r in test:
        by_op[r["op_id"]].append(r)
    qualifying = {op: rs for op, rs in by_op.items() if len(rs) >= 2 and 0 < sum(x["y"] for x in rs) < len(rs)}
    hits, rand = 0, 0.0
    for op, rs in qualifying.items():
        pick = sim.best_message(rs[0]["person"], [x["message"] for x in rs])
        top_idx = pick["best"]["index"]
        hits += rs[top_idx]["y"]                                          # did the model's top arg persuade?
        rand += sum(x["y"] for x in rs) / len(rs)                         # random-pick success rate for this OP
    n_op = len(qualifying)
    best_kpi = {"n_ops": n_op,
                "model_precision@1": round(hits / n_op, 4) if n_op else None,
                "random_pick_rate": round(rand / n_op, 4) if n_op else None,
                "lift": round((hits - rand) / n_op, 4) if n_op else None}

    # --- C. forward-state demo: same closing ask after a PUSHY vs a RESPECTFUL opener ---
    # use the fitted person x message coefficients PLUS the grounded state gate (zero at rest, so the
    # fit is untouched) — now the transient state the opener leaves behind changes how the ask lands.
    stateful = StructuredResponseModel(features=("receptivity", "quality", "interaction"),
                                       state_gate=True, gate_strength=2.5).fit(train_rows)
    sim_state = IndividualSimulator(response_fn=stateful)
    # a marginal person (near the 50/50 line, where the transient state actually swings the decision)
    person = {"trait_openness": 0.45, "skepticism": 0.55, "goal_alignment": 0.45, "base_responsiveness": 0.45}
    ask = {"clarity": 0.8, "ask_directness": 0.8, "personalization": 0.6, "expertise": 0.6, "effort_cost": 0.4}
    pushy_opener = {"clarity": 0.5, "ask_directness": 0.6, "pushiness": 0.9, "personalization": 0.1,
                    "politeness_disposition": 0.1, "effort_cost": 0.5}
    kind_opener = {"clarity": 0.7, "ask_directness": 0.5, "pushiness": 0.1, "personalization": 0.8,
                   "politeness_disposition": 0.9, "effort_cost": 0.3}
    # a rapid follow-up (gap_steps=0): the person is still in the state the opener left them in
    pushy_thread = sim_state.simulate_thread(person, [pushy_opener, ask], gap_steps=0)
    kind_thread = sim_state.simulate_thread(person, [kind_opener, ask], gap_steps=0)
    forward = {
        "same_closing_ask_p_after_pushy_opener": pushy_thread["turns"][1]["p_respond"],
        "same_closing_ask_p_after_kind_opener": kind_thread["turns"][1]["p_respond"],
        "state_after_pushy": pushy_thread["turns"][1]["state_before"],
        "state_after_kind": kind_thread["turns"][1]["state_before"],
        "reads": "identical closing ask; the person the openers left behind differs (mood/attention), so "
                 "the same message lands differently — a static vector cannot express this."}

    out = {"data": "ChangeMyView (real persuasion; 1,200 threads, temporal split)",
           "n": len(rows), "n_test": len(test), "test_base_rate": round(sum(yte) / len(yte), 4),
           "A_response_prediction": results, "A_comparison": comparison,
           "B_best_message_causal": best_kpi, "C_forward_state": forward}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-060  Level-1 individual simulator on REAL persuasion data (ChangeMyView)")
    print(f"  n={len(rows)}  test={len(test)}  test base-rate={out['test_base_rate']}")
    print("  A. RESPONSE PREDICTION (log_loss, lower=better) — does modeling the PERSON help?")
    for name in ("base_rate", "message_only", "person_only", "INDIVIDUAL"):
        print(f"       {name:14s} log_loss={results[name]['log_loss']:.4f}  "
              f"brier={results[name]['brier']:.4f}  uplift@20={results[name]['uplift@20']:.4f}")
    print(f"     -> individual beats message-only by {comparison['individual_beats_message_only_logloss']:+.4f} "
          f"log-loss (person x message interaction earns its place)" if
          comparison['individual_beats_message_only_logloss'] > 0 else
          f"     -> individual NOT better than message-only ({comparison['individual_beats_message_only_logloss']:+.4f})")
    print("  B. BEST MESSAGE (real do(x): OP fixed, several args, mixed outcomes)")
    print(f"       {best_kpi['n_ops']} OPs | model precision@1={best_kpi['model_precision@1']} vs "
          f"random-pick {best_kpi['random_pick_rate']}  -> lift {best_kpi['lift']:+}")
    print("  C. FORWARD STATE (same closing ask, different opener):")
    print(f"       after PUSHY opener: p={forward['same_closing_ask_p_after_pushy_opener']}  | "
          f"after KIND opener: p={forward['same_closing_ask_p_after_kind_opener']}  "
          f"(state carries over — the person is a dynamical system)")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
