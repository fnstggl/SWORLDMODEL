"""EXP-042: forward simulation of opinion change under an event — does the STRUCTURAL operator win?

The untested core of the thesis (EXP-040 was cross-sectional): couple a grounded actor to the
event-transition operator and simulate the outcome FORWARD, rather than averaging features. On
ChangeMyView each case is a real opinion-change event — an OP holds a view, an argument (the event) is
applied, and the ground truth is whether the view CHANGED (a delta).

The world-model form is multiplicative: the person GATES the event.
    P(change) ~ responsiveness(person) · impact(argument)
An open, non-entrenched mind moves under a strong argument; an entrenched or skeptical one does not — the
same argument has different effect depending on WHO receives it. A flat linear composite ("add up the
argument's quality and the person's openness") cannot represent that interaction; a simulation can.

Arms (no-cheat temporal split by timestamp; each fits a calibrated logistic on TRAIN, scores TEST):
  1. persistence           — base rate ("views rarely change" — the martingale analog)
  2. event-only composite  — argument-quality features only (no person)
  3. person-only           — responsiveness features only (no argument)
  4. additive (feature soup)— responsiveness ⊕ impact, linear, NO interaction
  5. STRUCTURAL SIMULATION  — additive + the coupled term responsiveness·impact (the operator)

Decisive test: does arm 5's interaction term beat arm 4 (the person really gates the event), and does the
structural simulation beat persistence + the event-only composite? Writes JSON.
Run: python -m experiments.exp042_forward_simulation
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.transition.readout import LogisticReadout
from swm.transition.unified_dynamics import responsiveness_from_map

RESULT = "experiments/results/exp042_forward_simulation.json"


class _VM:
    """Minimal VariableMap shim so the real operator (responsiveness_from_map) drives the simulation."""
    def __init__(self, d):
        self.d = d

    def get(self, k, default=0.0):
        return self.d.get(k, default)


def _f(v, default=0.5):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _features(inf):
    """From the inferred variables: the person's responsiveness (via the real operator) and the argument's
    persuasive impact, plus the coupled term the world model predicts is what matters."""
    # person -> responsiveness through the SAME operator the multi-step rollout uses
    vm = _VM({"openness_to_outreach": _f(inf.get("op_openness")),
              "skepticism": _f(inf.get("op_skepticism")),
              # entrenchment is a strong prior stance; the operator reads |prior_stance|
              "prior_stance": _f(inf.get("op_entrenchment"))})
    resp = responsiveness_from_map(vm)
    impact = sum(_f(inf.get(k)) for k in ("arg_addresses_crux", "arg_evidence", "arg_clarity",
                                          "arg_respectfulness", "arg_expertise")) / 5.0
    return resp, impact


def _fit_score(Xtr, ytr, Xte, yte):
    m = LogisticReadout(epochs=400, l2=0.5).fit(Xtr, ytr)
    p = [min(1 - 1e-6, max(1e-6, m.predict_proba(x))) for x in Xte]
    return {"log_loss": round(log_loss(yte, p), 4), "brier": round(brier_score(yte, p), 4),
            "ece": round(expected_calibration_error(yte, p), 4),
            "uplift@20": round(uplift_at_k(yte, p, 0.2), 4),
            "accuracy": round(sum(int((pi >= 0.5) == (yi == 1)) for pi, yi in zip(p, yte)) / len(yte), 4)}


def run():
    cases = json.loads(Path("data/cmv_common.json").read_text())
    inf = {}
    for fp in glob.glob("data/cmv_infer_*.json"):
        for r in json.loads(Path(fp).read_text()):
            inf[r["id"]] = r
    rows = [(s, inf[s["id"]]) for s in cases if s["id"] in inf]
    rows.sort(key=lambda z: z[0]["ts"])                    # temporal order -> no-cheat split
    feats = [(_features(i), s["success"]) for s, i in rows]
    n = len(feats); cut = int(0.7 * n)
    tr, te = feats[:cut], feats[cut:]
    ytr = [y for _, y in tr]; yte = [y for _, y in te]
    base = sum(ytr) / len(ytr)

    def X(rowset, kind):
        out = []
        for (resp, impact), _ in rowset:
            if kind == "event":
                out.append([impact])
            elif kind == "person":
                out.append([resp])
            elif kind == "additive":
                out.append([resp, impact])
            else:                                          # structural: main effects + the coupling term
                out.append([resp, impact, resp * impact])
        return out

    p_persist = [min(1 - 1e-6, max(1e-6, base))] * len(yte)
    arms = {
        "persistence": {"log_loss": round(log_loss(yte, p_persist), 4),
                        "brier": round(brier_score(yte, p_persist), 4),
                        "ece": round(expected_calibration_error(yte, p_persist), 4), "uplift@20": 0.0,
                        "accuracy": round(sum(int(yi == (base >= 0.5)) for yi in yte) / len(yte), 4)},
        "event_only_composite": _fit_score(X(tr, "event"), ytr, X(te, "event"), yte),
        "person_only": _fit_score(X(tr, "person"), ytr, X(te, "person"), yte),
        "additive_feature_soup": _fit_score(X(tr, "additive"), ytr, X(te, "additive"), yte),
        "structural_simulation": _fit_score(X(tr, "structural"), ytr, X(te, "structural"), yte),
    }
    # sharper, mechanistic test of gating: does argument IMPACT predict change MORE among responsive OPs?
    # (the model's claim is that the same argument moves an open mind and not an entrenched one.) Compare
    # the impact->success relationship in the high- vs low-responsiveness halves of the TEST set.
    def _slope(subset):
        if len(subset) < 20:
            return None
        mi = sum(im for im, _ in subset) / len(subset)
        my = sum(y for _, y in subset) / len(subset)
        num = sum((im - mi) * (y - my) for im, y in subset)
        den = sum((im - mi) ** 2 for im, y in subset)
        return round(num / den, 4) if den > 1e-9 else None

    te_pairs = [((resp, impact), y) for (resp, impact), y in te]
    med_resp = sorted(resp for (resp, _), _ in te_pairs)[len(te_pairs) // 2]
    hi = [(impact, y) for (resp, impact), y in te_pairs if resp >= med_resp]
    lo = [(impact, y) for (resp, impact), y in te_pairs if resp < med_resp]
    gating = {"impact_success_slope_high_responsiveness": _slope(hi),
              "impact_success_slope_low_responsiveness": _slope(lo),
              "n_high": len(hi), "n_low": len(lo)}

    d_interaction = round(arms["additive_feature_soup"]["log_loss"] - arms["structural_simulation"]["log_loss"], 4)
    d_persist = round(arms["persistence"]["log_loss"] - arms["structural_simulation"]["log_loss"], 4)
    d_event = round(arms["event_only_composite"]["log_loss"] - arms["structural_simulation"]["log_loss"], 4)

    out = {"dataset": "ChangeMyView", "n": n, "n_test": len(yte), "base_rate": round(base, 4),
           "coverage": len(rows), "arms": arms, "gating_analysis": gating,
           "interaction_gain_vs_additive": d_interaction,   # >0 => the coupling (person gates event) helps
           "structural_vs_persistence": d_persist, "structural_vs_event_only": d_event,
           "coupling_helps": d_interaction > 0, "beats_persistence": d_persist > 0}

    print(f"EXP-042 forward simulation of opinion change (CMV) — n={n} test={len(yte)} "
          f"base rate {base:.3f} coverage {len(rows)}")
    for k, v in arms.items():
        print(f"  {k:<26} log_loss {v['log_loss']}  brier {v['brier']}  acc {v['accuracy']}  up@20 {v['uplift@20']}")
    print(f"  -> coupling (person gates event) vs additive feature-soup: Δlog_loss {d_interaction:+.4f} "
          f"({'helps' if d_interaction > 0 else 'no gain'})")
    print(f"  -> structural simulation vs persistence: {d_persist:+.4f}; vs event-only: {d_event:+.4f}")
    print(f"  -> GATING: impact->change slope among responsive OPs "
          f"{gating['impact_success_slope_high_responsiveness']} (n={gating['n_high']}) vs entrenched "
          f"{gating['impact_success_slope_low_responsiveness']} (n={gating['n_low']})")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
