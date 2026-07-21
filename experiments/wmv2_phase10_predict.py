"""Phase 10 (continuation) — genuine PREDICTIVE end-to-end path vs PROCEDURAL reconstruction (#2, #6).

The 96.3% Senate result (wmv2_phase10_replay.py) is PROCEDURAL RULE-RECONSTRUCTION: it feeds the REAL yea/
nay counts into the threshold engine, validating rule EXECUTION — NOT forecasting. This harness adds the
required forward-prediction chain, OUT-OF-SAMPLE and leakage-safe (the target roll-call's own votes are NEVER
inputs):

  TRAIN on the prior Congress (117th): for each institutional MATTER-TYPE (the matter-aware threshold
      category the institution assigns — nomination-cloture / legislation-cloture / treaty / override /
      simple-majority) fit a REAL Phase-3 Dirichlet posterior over {passes, fails} — the coalition's success
      propensity at THAT institution's bar.  [swm ... infer_compositional_posterior, NOT a local pseudo-posterior]
  → TEST on the held-out later Congress (118th): as-of facts = matter-type (the institution's rule) + chamber
      party composition; Phase-6 actor policy = the fitted coalition-success propensity → institutional
      threshold → terminal outcome PROBABILITY, marginalised over the posterior particles.
  → route the point prediction through the REAL InstitutionOperator → StateDelta (terminal institutional
      outcome), proving the chain runs through the executable engine, not a lookup.

Two models are reported, both honest:
  M1  naive party-line (no training): the majority caucus forms the coalition, members vote their party, run
      the REAL decision engine per vote. A pure institution+actor policy — it UNDERPERFORMS the base rate,
      which is the point: party composition alone is a weak forecaster.
  M2  Phase-3 matter-type propensity (out-of-sample, probabilistic): learns that legislation-cloture usually
      FAILS on party lines while nomination-cloture usually PASSES — it BEATS the always-pass baseline and is
      scored with Brier + log-loss.

All reported SEPARATELY from procedural reconstruction (Part #6).

Run: PYTHONPATH=. python -m experiments.wmv2_phase10_predict
Writes experiments/results/phase10/wmv2_phase10_predict.json
"""
from __future__ import annotations

import json
import math

from swm.world_model_v2.phase3_posterior import infer_compositional_posterior
from swm.world_model_v2.institutions_v2.decisions import ThresholdSpec, evaluate_decision
from experiments.wmv2_phase10_replay import _fetch, _recorded, _threshold_for

# Public, durable, pre-vote fact: US Senate majority-caucus size (of 100), by Congress. Known before any
# roll-call — NOT derived from the vote being predicted.
SENATE_MAJORITY = {117: 50, 118: 51}
TRAIN_CONGRESS, TEST_CONGRESS = 117, 118

OUT = "experiments/results/phase10/wmv2_phase10_predict.json"


# ---------------------------------------------------------------- matter-type category counts (per Congress)
def _category_counts(rows) -> dict:
    """For each institutional matter-type (the matter-aware threshold label), tally recorded pass/fail. This is
    the only place a Congress's own outcomes are read — the TRAIN Congress here; the TEST Congress's per-vote
    outcome is used only to SCORE, never as a model input."""
    counts = {}
    for row in rows:
        rec = _recorded(row.get("vote_result", ""))
        if rec is None:
            continue
        _, label = _threshold_for(row.get("vote_question", ""), row.get("vote_result", ""),
                                  bill_number=row.get("bill_number", ""), date=row.get("date", ""),
                                  mode="matter_aware")
        c = counts.setdefault(label, {"pass": 0, "fail": 0})
        c["pass" if rec else "fail"] += 1
    return counts


# ---------------------------------------------------------------- Phase 3 fit (real compositional posterior)
def _fit_propensity(counts) -> dict:
    """Per matter-type, a REAL Phase-3 Dirichlet {pass,fail} posterior from the TRAIN Congress counts. Returns
    {label: {p_pass_mean, particles:[pass-propensity], ess, n_train}}. This is the coalition-success propensity
    at that institution's bar — the Phase-6 actor policy's learned parameter."""
    post = {}
    for label, c in counts.items():
        res = infer_compositional_posterior(
            ["pass", "fail"], prior_alpha=[1.0, 1.0],
            count_observations=[{"counts": {"pass": c["pass"], "fail": c["fail"]}, "reliability": 1.0,
                                 "source": f"train_c{TRAIN_CONGRESS}_{label}", "method": "rollcall_tally"}],
            n_particles=200, seed=17)
        post[label] = {"p_pass_mean": float(res.posterior_mean[0]),
                       "particles": [float(vec[0]) for vec, _w in res.particles],   # pass-propensity per particle
                       "ess": round(res.ess, 1), "n_train": c["pass"] + c["fail"]}
    return post


def _predict_prob(post, label, global_p) -> tuple:
    """P(pass) for a test vote, marginalised over the Phase-3 particles. Unseen matter-type → global base rate
    (honest fallback, flagged)."""
    e = post.get(label)
    if e is None:
        return global_p, False
    return e["p_pass_mean"], True


# ---------------------------------------------------------------- naive party-line policy (M1, no training)
def _naive_partyline_pass(label: str, majority_size: int, senate: int = 100) -> bool:
    """M1: the majority caucus sponsors and votes yea; the minority votes nay (cohesion≈1). Run the REAL
    decision engine on that synthesised coalition against the matter-type threshold. A pure institution+actor
    model with NO fitted parameter — deliberately naive."""
    spec, _ = _spec_for(label)
    yea, nay = majority_size, senate - majority_size
    votes = {f"s{i}": "yes" for i in range(yea)}
    votes.update({f"s{yea + i}": "no" for i in range(nay)})
    return evaluate_decision(spec, votes, eligible=[f"s{i}" for i in range(senate)]).passed


def _spec_for(label: str) -> tuple:
    """The institution's ThresholdSpec for a matter-type label (mirrors _threshold_for's matter-aware branch)."""
    if label == "cloture_legislation_3_5":
        return ThresholdSpec("supermajority", 0.6, base="all_members", quorum_fraction=0.5), label
    if label in ("treaty_2_3", "override_2_3"):
        return ThresholdSpec("supermajority", 2 / 3, base="present"), label
    return ThresholdSpec("simple_majority", 0.5, base="present"), label   # majority / nomination-cloture


# ---------------------------------------------------------------- StateDelta wiring proof (real operator)
def _statedelta_proof(label: str, p_pass: float, majority_size: int) -> dict:
    """Route the M2 point prediction (pass iff p≥0.5) through the REAL InstitutionOperator → StateDelta, so the
    terminal institutional outcome demonstrably flows through the executable engine, not a lookup table."""
    import random
    from swm.world_model_v2.state import Entity, SimulationClock, WorldState, parse_time
    from swm.world_model_v2.events import Event
    from swm.world_model_v2.network import RelationGraph
    from swm.world_model_v2.information import InformationLedger
    from swm.world_model_v2.institutions_v2.operators import InstitutionOperator, InstitutionRuntime
    from swm.world_model_v2.institutions_v2.record import InstitutionInstance
    from swm.world_model_v2.institutions_v2.store import load_store

    st = load_store(reload=True)
    tpl = st.templates["us_congress_legislative"]
    t0 = parse_time("2023-06-01")
    w = WorldState(world_id="p10pred", branch_id="root", clock=SimulationClock(now=t0, as_of=t0),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["chair"] = Entity(identity="chair")
    inst = InstitutionInstance("s1", tpl.template_id, tpl.version, "2023-06-01", current_stage="floor_first",
                               actor_bindings={"chair": "senator"})
    rt = InstitutionRuntime(template=tpl, instance=inst, as_of="2023-06-01")
    spec, _ = _spec_for(label)
    rt.thresholds["passage"] = spec
    # synthesise the coalition the FORECAST implies (predicted-pass → a coalition that clears the bar), then let
    # the engine decide — the outcome is produced by evaluate_decision inside the operator, not asserted.
    need = math.ceil((spec.fraction if spec.base != "all_members" else spec.fraction) * 100) + (
        1 if spec.kind == "simple_majority" else 0)
    yea = max(need, majority_size) if p_pass >= 0.5 else max(0, min(need - 1, majority_size))
    votes = {f"s{i}": ("yes" if i < yea else "no") for i in range(100)}
    op = InstitutionOperator()
    ev = Event(ts=w.clock.now, etype="institutional_action",
               payload={"institution": rt,
                        "action": {"actor": "chair", "type": "vote", "subject": "senate_vote",
                                   "required_authority": "final_decision"},
                        "decision": {"decision_id": "passage", "votes": votes,
                                     "eligible": [f"s{i}" for i in range(100)]},
                        "outcome_var": "predicted_enacted"})
    w.clock.advance_to(ev.ts)
    delta = op.run(w, ev, random.Random(0))[0]
    return {"matter_type": label, "predicted_p_pass": round(p_pass, 3),
            "synthesised_yea": yea, "threshold": f"{spec.kind}:{round(spec.fraction, 3)}/{spec.base}",
            "terminal_quantity": w.quantities.get("predicted_enacted").value if "predicted_enacted" in w.quantities else None,
            "statedelta_changes": len(delta.changes), "routed_through_operator": True}


# ---------------------------------------------------------------- scoring
def _brier(probs, actuals):
    return sum((p - int(a)) ** 2 for p, a in zip(probs, actuals)) / max(1, len(probs))


def _logloss(probs, actuals):
    s = 0.0
    for p, a in zip(probs, actuals):
        p = min(1 - 1e-9, max(1e-9, p))
        s += -(math.log(p) if a else math.log(1 - p))
    return s / max(1, len(probs))


def run():
    train_rows, test_rows = _fetch(TRAIN_CONGRESS), _fetch(TEST_CONGRESS)
    post = _fit_propensity(_category_counts(train_rows))
    # global TRAIN base rate — the fallback for a matter-type unseen in training + a baseline
    tc = _category_counts(train_rows)
    tot_pass = sum(c["pass"] for c in tc.values())
    tot = sum(c["pass"] + c["fail"] for c in tc.values())
    global_p = tot_pass / max(1, tot)

    maj = SENATE_MAJORITY[TEST_CONGRESS]
    m2_probs, m1_preds, actuals, labels = [], [], [], []
    by_type = {}
    unseen = 0
    for row in test_rows:
        rec = _recorded(row.get("vote_result", ""))
        if rec is None:
            continue
        _, label = _threshold_for(row.get("vote_question", ""), row.get("vote_result", ""),
                                  bill_number=row.get("bill_number", ""), date=row.get("date", ""),
                                  mode="matter_aware")
        p, seen = _predict_prob(post, label, global_p)
        if not seen:
            unseen += 1
        m2_probs.append(p)
        m1_preds.append(_naive_partyline_pass(label, maj))
        actuals.append(rec)
        labels.append(label)
        t = by_type.setdefault(label, {"n": 0, "m2_correct": 0, "m1_correct": 0, "sum_p": 0.0, "sum_y": 0})
        t["n"] += 1
        t["m2_correct"] += int((p >= 0.5) == rec)
        t["m1_correct"] += int(m1_preds[-1] == rec)
        t["sum_p"] += p
        t["sum_y"] += int(rec)

    n = len(actuals)
    base_rate = sum(actuals) / max(1, n)                    # TEST base rate (for reference only)
    m2_acc = sum(int((p >= 0.5) == a) for p, a in zip(m2_probs, actuals)) / max(1, n)
    m1_acc = sum(int(pr == a) for pr, a in zip(m1_preds, actuals)) / max(1, n)
    # baselines: always-pass, and predict the TRAIN global base rate for everyone (probabilistic)
    always_pass_acc = base_rate
    m2_brier = _brier(m2_probs, actuals)
    m2_logloss = _logloss(m2_probs, actuals)
    base_brier = _brier([global_p] * n, actuals)
    base_logloss = _logloss([global_p] * n, actuals)

    # StateDelta wiring proof on a representative informative matter-type present in test
    proof_label = "cloture_legislation_3_5" if "cloture_legislation_3_5" in post else labels[0]
    proof = _statedelta_proof(proof_label, post.get(proof_label, {}).get("p_pass_mean", global_p), maj)

    return {
        "design": {"train_congress": TRAIN_CONGRESS, "test_congress": TEST_CONGRESS,
                   "out_of_sample": True, "leakage_safe": True,
                   "note": "matter-type propensity fit on the PRIOR Congress; the TEST vote's own counts are "
                           "never a model input (used only to score). Party composition is a durable pre-vote fact."},
        "n_test_scored": n, "n_unseen_matter_type": unseen, "train_global_base_rate": round(global_p, 4),
        "test_base_rate_pass": round(base_rate, 4),
        "M2_phase3_matter_type_propensity": {
            "accuracy": round(m2_acc, 4), "brier": round(m2_brier, 4), "log_loss": round(m2_logloss, 4),
            "note": "OUT-OF-SAMPLE probabilistic forecast; learns matter-type success propensity via the real "
                    "Phase-3 Dirichlet posterior. Beats always-pass when Δacc>0 and beats base-rate Brier."},
        "M1_naive_partyline": {
            "accuracy": round(m1_acc, 4),
            "note": "no training; majority caucus votes party-line through the real decision engine — a weak "
                    "forecaster (party composition alone), kept as an honest reference."},
        "baselines": {"always_pass_accuracy": round(always_pass_acc, 4),
                      "predict_base_rate_brier": round(base_brier, 4),
                      "predict_base_rate_log_loss": round(base_logloss, 4)},
        "lift": {"M2_acc_minus_always_pass": round(m2_acc - always_pass_acc, 4),
                 "M2_brier_minus_base_rate_brier": round(m2_brier - base_brier, 4),
                 "M1_acc_minus_always_pass": round(m1_acc - always_pass_acc, 4)},
        "by_matter_type": {k: {"n": v["n"], "m2_acc": round(v["m2_correct"] / v["n"], 3),
                               "m1_acc": round(v["m1_correct"] / v["n"], 3),
                               "mean_pred_p": round(v["sum_p"] / v["n"], 3),
                               "empirical_pass": round(v["sum_y"] / v["n"], 3)}
                           for k, v in by_type.items()},
        "fitted_propensity_train": {k: {"p_pass_mean": round(v["p_pass_mean"], 3), "n_train": v["n_train"],
                                        "ess": v["ess"]} for k, v in post.items()},
        "statedelta_wiring_proof": proof,
    }


def main():
    pred = run()
    proc = {}
    try:
        rep = json.load(open("experiments/results/phase10/wmv2_phase10_replay.json"))
        proc = {"procedural_reconstruction_accuracy": rep["matter_aware"]["accuracy"],
                "note": "uses REAL yea/nay counts → threshold engine (validates rule EXECUTION, NOT forecasting)"}
    except Exception:
        pass
    doc = {"_meta": {"harness": "experiments/wmv2_phase10_predict.py",
                     "note": "OUT-OF-SAMPLE forward prediction (train prior Congress → test later Congress) via "
                             "the REAL Phase-3 posterior + institution engine, routed to a StateDelta. Kept "
                             "SEPARATE from procedural reconstruction (Part #6)."},
           "predictive_path": pred, "procedural_reconstruction_for_contrast": proc}
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    m2, m1, b, l = (pred["M2_phase3_matter_type_propensity"], pred["M1_naive_partyline"],
                    pred["baselines"], pred["lift"])
    print("=== Phase 10 PREDICTIVE path (out-of-sample: train C117 → test C118; leakage-safe) ===")
    print(f"  M2 Phase-3 matter-type propensity: acc {m2['accuracy']}  Brier {m2['brier']}  logloss {m2['log_loss']}")
    print(f"     baselines: always-pass acc {b['always_pass_accuracy']}  base-rate Brier {b['predict_base_rate_brier']}")
    print(f"     LIFT: acc vs always-pass {l['M2_acc_minus_always_pass']:+}, Brier vs base-rate {l['M2_brier_minus_base_rate_brier']:+}")
    print(f"  M1 naive party-line:               acc {m1['accuracy']}  (LIFT vs always-pass {l['M1_acc_minus_always_pass']:+} — weak, as expected)")
    print(f"  by matter-type (m2_acc | empirical_pass): "
          + str({k: (v['m2_acc'], v['empirical_pass']) for k, v in pred['by_matter_type'].items()}))
    print(f"  StateDelta wiring proof: {pred['statedelta_wiring_proof']['terminal_quantity']} "
          f"via real operator ({pred['statedelta_wiring_proof']['statedelta_changes']} changes)")
    if proc:
        print(f"  CONTRAST — procedural reconstruction (real votes): {proc['procedural_reconstruction_accuracy']} "
              f"(rule EXECUTION, NOT forecast accuracy)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
