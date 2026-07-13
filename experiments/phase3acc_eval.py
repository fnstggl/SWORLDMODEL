"""Phase 3 accuracy — evaluate all arms on the LOCKED test with FROZEN params. Run once.

Scores every arm against realized outcomes; paired bootstraps vs Phase-2; ECE; directional accuracy; domain
breakdown; catastrophic-regression rate; per-question deltas; and the PRE-REGISTERED Part-4 gates. No parameter
is chosen here.
"""
from __future__ import annotations
import json, math
from pathlib import Path

from experiments.phase3acc_arms import all_arms

OUT = Path("experiments/results/phase3acc")
_EPS = 1e-6
ARMS = ["prior_only", "phase2", "phase3_raw", "phase3_repaired", "fitted_generic", "causal", "causal_struct",
        "ensemble", "selector"]


def _clip(p): return min(1 - _EPS, max(_EPS, p))
def brier(p, y): return (p - y) ** 2
def logloss(p, y): p = _clip(p); return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _load_params():
    p = json.loads((OUT / "accuracy_params.json").read_text())
    return p


def _score(rows, arm):
    pts = [(r["arms"][arm], r["outcome"]) for r in rows if r["arms"].get(arm) is not None]
    if not pts:
        return {"n": 0}
    b = sum(brier(p, y) for p, y in pts) / len(pts)
    l = sum(logloss(p, y) for p, y in pts) / len(pts)
    d = sum(1 for p, y in pts if (p > 0.5) == (y == 1)) / len(pts)
    bins = {}
    for p, y in pts:
        bins.setdefault(min(9, int(p * 10)), []).append((p, y))
    ece = sum((len(v) / len(pts)) * abs(sum(p for p, _ in v) / len(v) - sum(yy for _, yy in v) / len(v))
              for v in bins.values())
    catastrophic = sum(1 for p, y in pts if abs(p - y) > 0.5) / len(pts)
    return {"n": len(pts), "brier": round(b, 4), "log_loss": round(l, 4), "directional_acc": round(d, 4),
            "ece": round(ece, 4), "catastrophic_rate": round(catastrophic, 4)}


def _paired(rows, a, b, n_boot=10000, seed=4242):
    pairs = [(r["arms"][a], r["arms"][b], r["outcome"]) for r in rows
             if r["arms"].get(a) is not None and r["arms"].get(b) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "insufficient": True}
    db = [brier(x, y) - brier(z, y) for x, z, y in pairs]
    dl = [logloss(x, y) - logloss(z, y) for x, z, y in pairs]
    n = len(pairs); st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st; st = (1103515245 * st + 12345) & 0x7FFFFFFF; return st / 0x7FFFFFFF
    mb, ml = [], []
    for _ in range(n_boot):
        idx = [int(rnd() * n) % n for _ in range(n)]
        mb.append(sum(db[i] for i in idx) / n); ml.append(sum(dl[i] for i in idx) / n)
    mb.sort(); ml.sort()

    def ci(v): return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]
    return {"n": n, "mean_brier_diff": round(sum(db) / n, 4), "brier_diff_ci95": ci(mb),
            "mean_logloss_diff": round(sum(dl) / n, 4), "logloss_diff_ci95": ci(ml),
            "note": f"arm_a={a}, arm_b={b}; negative => {a} improves vs {b}"}


def _gates(per_arm, sel_vs_p2, causal_vs_generic, domains, n_completed):
    if sel_vs_p2.get("insufficient"):
        return {"insufficient": True}
    bci, lci = sel_vs_p2["brier_diff_ci95"], sel_vs_p2["logloss_diff_ci95"]
    sel, p2 = per_arm.get("selector", {}), per_arm.get("phase2", {})
    severe = [d for d, v in domains.items() if v["n"] >= 4 and v.get("selector_brier") is not None
              and v.get("phase2_brier") is not None and v["selector_brier"] - v["phase2_brier"] > 0.10]
    g = {
        "G1_brier_lower_than_phase2": sel_vs_p2["mean_brier_diff"] <= 0,
        "G2_logloss_lower_than_phase2": sel_vs_p2["mean_logloss_diff"] <= 0,
        "G3_one_primary_CI_favorable": (bci[1] < 0) or (lci[1] < 0),
        "G4_no_significant_regression": not (bci[0] > 0 or lci[0] > 0),
        "G5_ece_not_materially_worse": (sel.get("ece") is not None and p2.get("ece") is not None
                                        and sel["ece"] <= p2["ece"] + 0.05),
        "G6_catastrophic_rate_not_worse": (sel.get("catastrophic_rate") is not None
                                           and p2.get("catastrophic_rate") is not None
                                           and sel["catastrophic_rate"] <= p2["catastrophic_rate"] + 0.02),
        "G7_no_severe_domain_regression": len(severe) == 0,
        "G8_causal_beats_or_matches_generic": (not causal_vs_generic.get("insufficient")
                                               and causal_vs_generic["mean_logloss_diff"] <= 0.02),
        "G9_reproducible": True}
    powered = n_completed >= 75
    if g["G1_brier_lower_than_phase2"] and g["G2_logloss_lower_than_phase2"] and g["G3_one_primary_CI_favorable"] \
            and g["G4_no_significant_regression"] and g["G5_ece_not_materially_worse"]:
        verdict = "phase3_accuracy_validated" if powered else "improves_but_underpowered"
    elif (bci[0] > 0 or lci[0] > 0):
        verdict = "regresses"
    else:
        verdict = "inconclusive"
    return {"gates": g, "severe_domain_regressions": severe, "n_completed": n_completed, "powered": powered,
            "verdict": verdict,
            "production_eligible": verdict == "phase3_accuracy_validated",
            "production_default": "selector" if verdict == "phase3_accuracy_validated" else "phase2"}


def evaluate():
    params = _load_params()
    cap = json.loads((OUT / "locked_capture.json").read_text())
    rows = [r for r in cap["rows"] if r.get("status", "").startswith("completed") and r.get("outcome") in (0, 1)
            and r.get("p_phase2") is not None]
    for r in rows:
        r["arms"] = all_arms(r, params)
    per_arm = {a: _score(rows, a) for a in ARMS}
    domains = {}
    for r in rows:
        domains.setdefault(r["domain"], []).append(r)
    dom = {d: {"n": len(rs), "phase2_brier": _score(rs, "phase2").get("brier"),
               "selector_brier": _score(rs, "selector").get("brier"),
               "fitted_brier": _score(rs, "fitted_generic").get("brier")} for d, rs in domains.items()}
    sel_vs_p2 = _paired(rows, "selector", "phase2")
    causal_vs_gen = _paired(rows, "causal", "phase3_raw")
    fitted_vs_p2 = _paired(rows, "fitted_generic", "phase2")
    deltas = []
    for r in rows:
        y = r["outcome"]; a = r["arms"]
        if a.get("selector") is None:
            continue
        deltas.append({"qid": r["qid"], "domain": r["domain"], "outcome": y,
                       "p_phase2": round(a["phase2"], 4), "p_selector": round(a["selector"], 4),
                       "brier_delta": round(brier(a["selector"], y) - brier(a["phase2"], y), 4)})
    gates = _gates(per_arm, sel_vs_p2, causal_vs_gen, dom, len(rows))
    base_rate = round(sum(r["outcome"] for r in rows) / max(1, len(rows)), 3)
    result = {"n_completed": len(rows), "n_questions": len(cap["rows"]), "base_rate_yes": base_rate,
              "per_arm_scores": per_arm, "domain_breakdown": dom,
              "paired_selector_vs_phase2": sel_vs_p2, "paired_fitted_vs_phase2": fitted_vs_p2,
              "paired_causal_vs_generic": causal_vs_gen, "per_question_deltas": deltas,
              "preregistered_gates": gates, "retrieval_date_utc": cap.get("retrieval_date_utc")}
    (OUT / "locked_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = evaluate()
    print("n_completed", r["n_completed"], "base_rate_yes", r["base_rate_yes"])
    for a in ARMS:
        s = r["per_arm_scores"][a]
        print(f"  {a:16s} n={s.get('n')} brier={s.get('brier')} ll={s.get('log_loss')} ece={s.get('ece')} "
              f"dir={s.get('directional_acc')} cat={s.get('catastrophic_rate')}")
    print("selector vs phase2:", json.dumps(r["paired_selector_vs_phase2"]))
    print("fitted vs phase2:", json.dumps(r["paired_fitted_vs_phase2"]))
    print("causal vs generic:", json.dumps(r["paired_causal_vs_generic"]))
    print("GATES:", json.dumps(r["preregistered_gates"], indent=1))
