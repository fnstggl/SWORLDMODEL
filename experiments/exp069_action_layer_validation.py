"""EXP-069: the general action layer + interventional scoreboard, re-earned on REAL data.

EXP-060 validated best-message on ChangeMyView with the OLD narrow `IndividualSimulator.best_message`
(precision@1 0.739 vs 0.518 random = +22pt causal lift). This re-earns that number through the NEW GENERIC
action layer (`swm/decision/best_action.py` — typed message interventions + best-arm racing) fed the SAME
validated per-person model, and scores it with the NEW interventional KPI module (`swm/eval/policy_regret.py`:
precision@1, policy regret, CATE-sign) — proving the generic layer reproduces the validated selection
end-to-end, not just in a unit test. It also runs the new scoreboard on the real Upworthy randomized A/B
tests (the interventional KPI-A), demonstrating the metrics on genuinely randomized `do(x)` data.

  A. CMV BEST-MESSAGE through the NEW layer. Fit the StructuredResponseModel (person × message) on the
     temporal-train split (identical to EXP-060). For each held-out OP that received SEVERAL arguments with
     MIXED outcomes (person fixed, message varies — a natural experiment), select the best argument with the
     generic `best_action` racing loop and ask: does its top pick persuade more than a random pick? Compared
     head-to-head with the old `best_message` path to confirm selection PARITY.

  B. UPWORTHY interventional scoreboard. On the real randomized headline A/B archive, a lexical headline→CTR
     policy picks an arm; the NEW `policy_regret` module scores realized policy value / regret + CATE-sign
     against the causal ground truth. (Lexical is the honest floor — the semantic ceiling, EXP-056, needs an
     LLM selector; here the point is that the scoreboard runs on real do(x) data.)

Run: python -m experiments.exp069_action_layer_validation
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

from swm.api.individual_simulate import IndividualSimulator
from swm.decision.action import Action
from swm.decision.best_action import best_action
from swm.decision.utility import Mean, identity
from swm.eval.policy_regret import cate_sign_accuracy, policy_regret, precision_at_1
from swm.simulation.response_model import StructuredResponseModel

CMV_COMMON = "experiments/results/exp021_cmv/cmv_common.json"
CMV_INFER = "experiments/results/exp021_cmv/cmv_inferences.json"
UPW = "experiments/results/exp054_upworthy/upworthy_parsed.json"
RESULT = "experiments/results/exp069_action_layer_validation.json"


def _load_cmv():
    common = {r["id"]: r for r in json.loads(Path(CMV_COMMON).read_text())}
    infer = {r["id"]: r for r in json.loads(Path(CMV_INFER).read_text())}
    rows = []
    for rid, c in common.items():
        f = infer.get(rid)
        if not f:
            continue
        person = {"trait_openness": f["op_openness"], "skepticism": f["op_skepticism"],
                  "certainty_disposition": f["op_entrenchment"]}
        message = {"addresses_crux": f["arg_addresses_crux"], "evidence": f["arg_evidence"],
                   "clarity": f["arg_clarity"], "politeness_disposition": f["arg_respectfulness"],
                   "expertise": f["arg_expertise"]}
        rows.append({"op_id": c["op_id"], "ts": c["ts"], "person": person, "message": message,
                     "y": int(c["success"])})
    rows.sort(key=lambda r: r["ts"])
    return rows


def _new_layer_pick(person, messages, model, seed=0):
    """Select the best message through the NEW generic action layer: each message is a typed do-operator, the
    outcome is the fitted model's response probability, best-arm racing returns the argmax."""
    def outcome_fn(action, rng):
        return model(person, {}, action.meta["message"])["p"], {}
    actions = [Action(f"arg_{i}", meta={"message": m}) for i, m in enumerate(messages)]
    res = best_action(outcome_fn, actions, identity(), objective=Mean(), max_per_arm=120, batch=24, seed=seed)
    return int(res.best.label.split("_")[1])


def run_cmv() -> dict:
    rows = _load_cmv()
    cut = int(0.7 * len(rows))
    train, test = rows[:cut], rows[cut:]
    train_rows = [(r["person"], {}, r["message"], r["y"]) for r in train]
    model = StructuredResponseModel(features=("receptivity", "quality", "interaction")).fit(train_rows)

    by_op = defaultdict(list)
    for r in test:
        by_op[r["op_id"]].append(r)
    qualifying = {op: rs for op, rs in by_op.items() if len(rs) >= 2 and 0 < sum(x["y"] for x in rs) < len(rs)}

    old_sim = IndividualSimulator(response_fn=model)                      # the EXP-060 path, for parity
    chosen_new, oracle_sets, chosen_reward, oracle_reward = [], [], [], []
    pred_deltas, true_deltas = [], []
    old_hits, rand = 0, 0.0
    for op, rs in qualifying.items():
        person, msgs = rs[0]["person"], [x["message"] for x in rs]
        idx_new = _new_layer_pick(person, msgs, model)
        idx_old = old_sim.best_message(person, msgs)["best"]["index"]
        chosen_new.append(idx_new)
        oracle_sets.append({j for j, x in enumerate(rs) if x["y"] == 1})  # arms that actually persuaded
        chosen_reward.append(rs[idx_new]["y"]); oracle_reward.append(max(x["y"] for x in rs))
        old_hits += rs[idx_old]["y"]
        rand += sum(x["y"] for x in rs) / len(rs)
        preds = [model(person, {}, m)["p"] for m in msgs]
        for i in range(len(rs)):                                           # pairwise CATE-sign within the OP
            for j in range(i + 1, len(rs)):
                if rs[i]["y"] == rs[j]["y"]:
                    continue
                pred_deltas.append(preds[i] - preds[j]); true_deltas.append(rs[i]["y"] - rs[j]["y"])

    n = len(qualifying)
    new_prec = precision_at_1(chosen_new, oracle_sets)                     # via the NEW scoreboard module
    return {"data": "ChangeMyView (real persuasion; temporal split)", "n_ops": n,
            "new_layer_precision@1": round(new_prec, 4),
            "old_path_precision@1": round(old_hits / n, 4) if n else None,
            "random_pick_rate": round(rand / n, 4) if n else None,
            "new_layer_lift": round(new_prec - rand / n, 4) if n else None,
            "selection_parity_with_old_path": round(sum(rs_y for rs_y in chosen_reward), 4) == old_hits,
            "policy_regret": round(policy_regret(chosen_reward, oracle_reward), 4),
            "cate_sign_accuracy": round(cate_sign_accuracy(pred_deltas, true_deltas), 4)}


def run_upworthy() -> dict:
    from experiments.exp054_interventional import _RidgeGD, _features
    tests = json.loads(Path(UPW).read_text())
    tr = [t for t in tests if (zlib.crc32(t["test_id"].encode()) % 1000) / 1000.0 >= 0.3]
    te = [t for t in tests if (zlib.crc32(t["test_id"].encode()) % 1000) / 1000.0 < 0.3]
    Xtr = [_features(a["headline"]) for t in tr for a in t["arms"]]
    ytr = [a["ctr"] for t in tr for a in t["arms"]]
    model = _RidgeGD().fit(Xtr, ytr)

    chosen, oracle, randp = [], [], []
    pred_deltas, true_deltas = [], []
    for t in te:
        arms = t["arms"]
        preds = [model.predict(_features(a["headline"])) for a in arms]
        ctrs = [a["ctr"] for a in arms]
        pick = max(range(len(arms)), key=lambda i: preds[i])
        chosen.append(ctrs[pick]); oracle.append(max(ctrs)); randp.append(sum(ctrs) / len(ctrs))
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                if abs(ctrs[i] - ctrs[j]) < 1e-9:
                    continue
                pred_deltas.append(preds[i] - preds[j]); true_deltas.append(ctrs[i] - ctrs[j])

    n = len(chosen)
    mp = lambda v: sum(v) / len(v)
    achievable = mp(oracle) - mp(randp)
    captured = mp(chosen) - mp(randp)
    return {"data": "Upworthy randomized headline A/B (real do(x))", "n_test": n,
            "policy_regret_vs_oracle": round(policy_regret(chosen, oracle), 6),
            "model_policy_ctr": round(mp(chosen), 5), "oracle_ctr": round(mp(oracle), 5),
            "random_ctr": round(mp(randp), 5),
            "fraction_achievable_uplift_captured": round(captured / achievable, 4) if achievable > 1e-9 else None,
            "cate_sign_accuracy": round(cate_sign_accuracy(pred_deltas, true_deltas), 4),
            "note": "lexical headline->CTR policy scored with the NEW policy_regret module on real randomized "
                    "A/B data; the semantic ceiling (EXP-056) needs an LLM selector — here the point is the "
                    "interventional scoreboard runs on genuine do(x) ground truth."}


def run_sequential() -> dict:
    """Part C: the SEQUENTIAL policy layer (best_policy) on the real CMV-fitted person model — does choosing a
    two-step PLAN (opener → ask) beat choosing a single message? The state the opener leaves behind changes
    how the identical ask lands (EXP-060 C), so best_policy should prefer the kind opener. This validates the
    reflexivity/sequence machinery (Component 6) on the real fitted dynamical model, not a synthetic one."""
    from swm.decision.policy import best_policy, individual_rollout, message_sequences
    from swm.decision.utility import Mean, identity

    rows = _load_cmv()
    cut = int(0.7 * len(rows))
    train_rows = [(r["person"], {}, r["message"], r["y"]) for r in rows[:cut]]
    # the same fit as EXP-060 C: person×message + a grounded state gate (zero at rest, so the fit is untouched)
    model = StructuredResponseModel(features=("receptivity", "quality", "interaction"),
                                    state_gate=True, gate_strength=2.5).fit(train_rows)

    person = {"trait_openness": 0.45, "skepticism": 0.55, "goal_alignment": 0.45, "base_responsiveness": 0.45}
    ask = {"clarity": 0.8, "ask_directness": 0.8, "personalization": 0.6, "expertise": 0.6, "effort_cost": 0.4}
    kind = {"clarity": 0.7, "ask_directness": 0.5, "pushiness": 0.1, "personalization": 0.8,
            "politeness_disposition": 0.9, "effort_cost": 0.3}
    pushy = {"clarity": 0.5, "ask_directness": 0.6, "pushiness": 0.9, "personalization": 0.1,
             "politeness_disposition": 0.1, "effort_cost": 0.5}

    rollout = individual_rollout(person, model, gap_steps=0, readout="last", respond="threshold")
    policies = message_sequences([kind, pushy], ask, opener_labels=["kind", "pushy"])
    res = best_policy(rollout, policies, identity(), objective=Mean(), max_per_arm=600, seed=0)
    p_after_kind, _ = rollout(policies[0], __import__("random").Random(0))
    p_after_pushy, _ = rollout(policies[1], __import__("random").Random(0))
    return {"data": "ChangeMyView (real fitted person×message model + state gate)",
            "best_plan": res.best.label, "picks_kind_opener": res.best.label.startswith("kind"),
            "same_ask_p_after_kind_opener": round(p_after_kind, 4),
            "same_ask_p_after_pushy_opener": round(p_after_pushy, 4),
            "reads": "identical closing ask; best_policy prefers the opener whose residue in the person's "
                     "state makes the ask land better — a plan a single-message layer cannot choose."}


def run() -> dict:
    cmv = run_cmv()
    upw = run_upworthy()
    seq = run_sequential()
    out = {"experiment": "EXP-069 action layer + scoreboard on real data", "cmv_best_message": cmv,
           "upworthy_interventional": upw, "sequential_policy": seq}
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-069  general action layer + interventional scoreboard, re-earned on REAL data")
    print("  A. CMV BEST-MESSAGE through the NEW generic action layer (best_action racing)")
    print(f"     {cmv['n_ops']} mixed-outcome OPs | NEW-layer precision@1={cmv['new_layer_precision@1']} vs "
          f"random {cmv['random_pick_rate']}  -> lift {cmv['new_layer_lift']:+}")
    print(f"     selection parity with old best_message path: {cmv['selection_parity_with_old_path']}  | "
          f"policy_regret={cmv['policy_regret']}  CATE-sign={cmv['cate_sign_accuracy']}")
    print("  B. UPWORTHY interventional scoreboard on real randomized A/B (NEW policy_regret module)")
    print(f"     {upw['n_test']} held-out experiments | model CTR {upw['model_policy_ctr']} vs random "
          f"{upw['random_ctr']} vs oracle {upw['oracle_ctr']}")
    print(f"     fraction of achievable uplift captured: {upw['fraction_achievable_uplift_captured']}  | "
          f"CATE-sign {upw['cate_sign_accuracy']} (chance 0.5)")
    print("  C. SEQUENTIAL POLICY (best_policy: opener -> ask) on the real fitted person model")
    print(f"     best plan: {seq['best_plan']}  | same ask lands p={seq['same_ask_p_after_kind_opener']} after "
          f"KIND vs p={seq['same_ask_p_after_pushy_opener']} after PUSHY opener (state carryover)")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
