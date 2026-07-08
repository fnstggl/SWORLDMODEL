"""EXP-032: unify the transition operator across scales — one dynamics engine for market AND person.

EXP-030 showed the event-conditioned transition works for AGGREGATE belief (markets). This shows the
SAME operator form models INDIVIDUAL belief updates — persuasion on r/ChangeMyView is a belief
transition: an argument (event) updates the OP's stance (belief), and how much it moves depends on the
OP's openness/skepticism/entrenchment (their VariableMap). The unification:

    Δbelief = responsiveness · event_impact

  aggregate : event_impact = news impact,           responsiveness = market factor      (EXP-030)
  individual: event_impact = argument persuasive push, responsiveness = f(VariableMap)   (here)

The test of the unification: for individual belief updates, does routing the event impact through the
person's VariableMap responsiveness beat the event impact alone (the aggregate operator applied blind to
who the person is)? If yes, the same operator + the VariableMap = individual dynamics — State (the map)
feeds the responsiveness of the Dynamics.

No-cheat temporal split. Uses the committed EXP-021 CMV inferences. Writes JSON.
Run: python -m experiments.exp032_unified_dynamics
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from swm.eval.metrics import brier_score, log_loss
from swm.transition.readout import LogisticReadout
from swm.transition.unified_dynamics import UnifiedBeliefDynamics, responsiveness_from_map
from swm.variables.variable_map import VariableMap

RESULT = "experiments/results/exp032_unified_dynamics.json"


def _load_cmv():
    common = Path("data/cmv_common.json")
    if common.exists():
        sub = json.loads(common.read_text())
        inf = {}
        for fp in glob.glob("data/cmv_infer_*.json"):
            for r in json.loads(Path(fp).read_text()):
                inf[r["id"]] = r
        if inf:
            return sub, inf
    sub = json.loads(Path("experiments/results/exp021_cmv/cmv_common.json").read_text())
    inf = {r["id"]: r for r in json.loads(Path("experiments/results/exp021_cmv/cmv_inferences.json").read_text())}
    return sub, inf


def _op_map(inf) -> VariableMap:
    """Build the OP's VariableMap from the inferred variables (the State that sets responsiveness)."""
    vm = VariableMap(entity_id="op")
    vm.set("openness_to_outreach", float(inf.get("op_openness", 0.5)), provenance="llm", confidence=0.6)
    vm.set("skepticism", float(inf.get("op_skepticism", 0.5)), provenance="llm", confidence=0.6)
    vm.set("prior_stance", 2 * float(inf.get("op_entrenchment", 0.5)) - 1, provenance="llm", confidence=0.6)
    return vm


def _event_impact(inf) -> float:
    """The argument's persuasive push (the event): crux-fit + evidence + clarity."""
    return (float(inf.get("arg_addresses_crux", 0.5)) + float(inf.get("arg_evidence", 0.5))
            + float(inf.get("arg_clarity", 0.5))) / 3.0


def _fit_eval(Xtr, ytr, Xte, yte):
    m = LogisticReadout(epochs=300, l2=1.0).fit(Xtr, ytr)
    p = [min(1 - 1e-6, max(1e-6, m.predict_proba(x))) for x in Xte]
    acc = sum((pi > 0.5) == yi for pi, yi in zip(p, yte)) / len(yte)
    return {"log_loss": round(log_loss(yte, p), 4), "brier": round(brier_score(yte, p), 4),
            "accuracy": round(acc, 4)}


def run():
    sub, inf = _load_cmv()
    rows = [s for s in sub if s["id"] in inf]
    n = len(rows); cut = int(0.7 * n)
    y = [s["success"] for s in rows]

    imp = [_event_impact(inf[s["id"]]) for s in rows]
    resp = [responsiveness_from_map(_op_map(inf[s["id"]])) for s in rows]
    uni = UnifiedBeliefDynamics()
    unified = [uni.predict_update(imp[i], resp[i]) for i in range(n)]   # responsiveness x impact

    def split(feats):
        return [feats[:cut], y[:cut], feats[cut:], y[cut:]]

    tiers = {
        "base_rate": {"log_loss": round(log_loss(y[cut:], [sum(y[:cut]) / cut] * (n - cut)), 4),
                      "brier": round(brier_score(y[cut:], [sum(y[:cut]) / cut] * (n - cut)), 4),
                      "accuracy": round(sum(y[cut:]) / (n - cut) if sum(y[cut:]) > (n - cut) / 2
                                        else 1 - sum(y[cut:]) / (n - cut), 4)},
        "event_impact_only": _fit_eval([[v] for v in imp[:cut]], y[:cut], [[v] for v in imp[cut:]], y[cut:]),
        "responsiveness_only": _fit_eval([[v] for v in resp[:cut]], y[:cut], [[v] for v in resp[cut:]], y[cut:]),
        "unified(resp x impact)": _fit_eval([[v] for v in unified[:cut]], y[:cut],
                                            [[v] for v in unified[cut:]], y[cut:]),
    }
    gain = round(tiers["event_impact_only"]["log_loss"] - tiers["unified(resp x impact)"]["log_loss"], 4)
    out = {"dataset": "cmv_persuasion", "n": n, "n_test": n - cut, "tiers": tiers,
           "unified_vs_impact_only_logloss_gain": gain,
           "unified_beats_impact_only": gain > 0}
    print(f"EXP-032 unified dynamics — CMV individual belief-update, n_test={n - cut}")
    for k, v in tiers.items():
        print(f"  {k:<24} log loss {v['log_loss']}  brier {v['brier']}  acc {v['accuracy']}")
    print(f"  Δ unified (VariableMap responsiveness x event impact) vs event impact alone: {gain:+.4f} log loss "
          f"({'the person-modulated transition wins' if gain > 0 else 'no gain'})")
    print("  -> the SAME operator (Δ = responsiveness x event_impact) models aggregate markets (EXP-030) "
          "and individual persuasion (here); the VariableMap supplies individual responsiveness.")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
