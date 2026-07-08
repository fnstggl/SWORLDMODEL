"""EXP-082: state grounding — a calibrated weight times a GUESSED value is still a guess.

The corpus flywheel calibrated every variable's WEIGHT (elasticity). But the compiler still fills each
variable's VALUE — the current state of the world — from an LLM guess. This experiment isolates how much that
one thing costs, on the FOMC direction task (no-cheat, train-era-only weights), by holding the model FIXED and
changing ONLY where the feature VALUES come from:

  GUESSED  arm — the values the model would use with no current knowledge: the train-era MEAN of each feature
                 (the "typical world" prior). Same for every test month, so the model cannot tell the months
                 apart — a calibrated weight on a guessed value.
  GROUNDED arm — the SAME weights, but each high-leverage variable's value MEASURED as-of from real evidence
                 (a `DataGrounder` over the committed macro series), each carrying a CI.
  TRIAGE   arm — grounds ONLY the single highest-leverage variable (variance triage), leaves the rest guessed:
                 does grounding the few variables the outcome turns on capture most of the gain?

Then B. exercises the actual `StateGrounder.ground_spec` → compiler path: a `calibrated_readout` ModelSpec built
from the fitted weights, grounded for one hike month vs left at its guess, run through the Monte-Carlo — the
grounded spec's P(hike) should move toward the truth while the guessed spec sits at the base rate.

Run: python -m experiments.exp082_state_grounding
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.api.model_spec import ModelSpec, SpecVar
from swm.api.compiler import CompiledModel
from swm.api.state_grounding import DataGrounder, StateGrounder, ground_features
from swm.variables.calibrated_weights import CalibratedWeights, WeightPrior
from swm.variables.prior_registry import PriorRegistry

FOMC = "experiments/results/exp071/fomc_macro.json"
RESULT = "experiments/results/exp082_state_grounding.json"
FEATS = ["inflation", "unemployment", "recent_move"]
THR = 0.05


def _state(data, i):
    prev = data[max(0, i - 1)]["rate"]
    return {"inflation": data[i]["inflation"] / 10.0, "unemployment": data[i]["unemp"] / 10.0,
            "recent_move": max(-1.0, min(1.0, data[i]["rate"] - prev))}


def _clip(p):
    return min(1 - 1e-6, max(1e-6, p))


def _make_fetch(data, allow=None, sd=0.02):
    """A DataGrounder source: measure a variable's as-of value from the committed macro series.
    `as_of` is the month index. `allow` restricts which variables this source can ground (for the triage arm)."""
    def fetch(variable, as_of):
        if allow is not None and variable not in allow:
            return None
        st = _state(data, as_of)
        if variable not in st:
            return None
        return (st[variable], sd)
    return fetch


def _score(cw, feats_of, te_piv, move):
    """Directional accuracy + log-loss on held-out pivotal moves, where the feature vector for month i is
    produced by `feats_of(i)` (this is the ONLY thing that differs between arms)."""
    hit, ll = 0, []
    for i in te_piv:
        up = move(i) > 0
        p = cw.predict(feats_of(i))
        hit += 1 if (p > 0.5) == up else 0
        ll.append(-math.log(_clip(p if up else 1 - p)))
    n = len(te_piv)
    return {"accuracy": round(hit / n, 4), "logloss": round(sum(ll) / n, 4)}


def run() -> dict:
    data = json.loads(Path(FOMC).read_text())
    rates = [d["rate"] for d in data]
    n = len(rates)
    cut = int(0.6 * n)
    move = lambda i: rates[i + 1] - rates[i]

    # --- train the direction model on the train era only (identical recipe to EXP-079) ---
    reg = PriorRegistry.load()
    priors = [reg.prior_for(f, "rate_hike", fallback=WeightPrior(f, 0.0, 2.0)) for f in FEATS]
    tr_piv = [i for i in range(1, cut - 1) if abs(move(i)) > THR]
    Xtr = [[_state(data, i)[f] for f in FEATS] for i in tr_piv]
    ytr = [1 if move(i) > 0 else 0 for i in tr_piv]
    cw = CalibratedWeights(priors, temper_grid=(1.0, 4.0), epochs=150).fit(Xtr, ytr, tune=True)

    # the GUESS: what the model uses with no current knowledge — the train-era mean of each feature
    guess = {f: sum(row[j] for row in Xtr) / len(Xtr) for j, f in enumerate(FEATS)}
    # the highest-leverage variable by variance triage (weight^2 * Var) — the one the TRIAGE arm grounds
    top_var = cw.triage(Xtr)[0]["name"]

    te_piv = [i for i in range(cut, n - 1) if abs(move(i)) > THR]
    move_dir = move

    # GROUNDED: every high-leverage feature measured as-of from real evidence
    grounder_all = StateGrounder(grounders={f: DataGrounder(_make_fetch(data), name="fred") for f in FEATS})
    # TRIAGE: ground ONLY the single top-leverage variable, the rest fall back to the guess
    grounder_top = StateGrounder(grounders={top_var: DataGrounder(_make_fetch(data, allow={top_var}), name="fred")})

    def feats_guessed(i):
        return [guess[f] for f in FEATS]

    def feats_grounded(i):
        return ground_features(FEATS, grounder_all, as_of=i, guess=guess)

    def feats_triage(i):
        return ground_features(FEATS, grounder_top, as_of=i, guess=guess)

    guessed = _score(cw, feats_guessed, te_piv, move_dir)
    grounded = _score(cw, feats_grounded, te_piv, move_dir)
    triage = _score(cw, feats_triage, te_piv, move_dir)
    # SKILL of grounding = fractional log-loss reduction vs the guessed-state baseline (same weights)
    skill_grounded = round(1 - grounded["logloss"] / guessed["logloss"], 4)
    skill_triage = round(1 - triage["logloss"] / guessed["logloss"], 4)

    # --- B. the real StateGrounder.ground_spec -> compiler path, aggregated over held-out moves ---
    # The GUESSED spec is identical every month, so its compiler P(hike) is one number for all months; the
    # GROUNDED spec responds to the as-of world, so its P(hike) should be HIGH on months the Fed hiked and LOW
    # on months it cut. We report the mean over each, and one representative-month sample report for provenance.
    intercept = cw.model.b
    wsd = cw.model.weight_sd()
    sg = StateGrounder(grounders={f: DataGrounder(_make_fetch(data), name="fred") for f in FEATS})

    def readout_spec(values):                              # calibrated_readout: p = sigmoid(b + sum w*(x-0))
        vs = [SpecVar(name=f, value=values[f], est_sd=0.0, lo=-2.0, hi=2.0, center=0.0,
                      weight=cw.model.w[j], weight_sd=wsd[j], weight_source="fit") for j, f in enumerate(FEATS)]
        return ModelSpec(mechanism="calibrated_readout", variables=vs,
                         outcome={"event": {"op": ">", "value": 0.5}}, extra={"intercept": intercept})

    p_guessed = CompiledModel(readout_spec(guess)).run(n=6000, seed=1)["mean"]
    hike_ps, cut_ps = [], []
    sample_report = None
    for i in te_piv:
        gspec, rep = sg.ground_spec(readout_spec(guess), question="Will the Fed hike?", as_of=i)
        p = CompiledModel(gspec).run(n=6000, seed=1)["mean"]
        (hike_ps if move(i) > 0 else cut_ps).append(p)
        if sample_report is None and move(i) > 0 and p > 0.5:      # a clean hike the grounded spec calls right
            sample_report = {"month": data[i]["month"], "report": rep}
    p_grounded_hike = sum(hike_ps) / len(hike_ps) if hike_ps else None
    p_grounded_cut = sum(cut_ps) / len(cut_ps) if cut_ps else None

    res = {"data": "FOMC 1985-2026, no-cheat (train-era weights); model FIXED, only VALUES change",
           "n_held_out_pivotal_moves": len(te_piv), "top_leverage_var": top_var,
           "guess_values": {k: round(v, 4) for k, v in guess.items()},
           "A_grounded_vs_guessed": {
               "guessed_state": guessed, "grounded_state": grounded, "triage_grounded_top_only": triage,
               "grounding_skill_logloss": skill_grounded, "triage_skill_logloss": skill_triage,
               "accuracy_lift": round(grounded["accuracy"] - guessed["accuracy"], 4)},
           "B_ground_spec_compiler": {"p_hike_guessed_state_all_months": round(p_guessed, 4),
                                      "p_hike_grounded_on_actual_hike_months": round(p_grounded_hike, 4)
                                      if p_grounded_hike is not None else None,
                                      "p_hike_grounded_on_actual_cut_months": round(p_grounded_cut, 4)
                                      if p_grounded_cut is not None else None,
                                      "sample_grounded_report": sample_report}}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    a = res["A_grounded_vs_guessed"]
    print("EXP-082  state grounding: knowing the world vs guessing it (FOMC direction, weights FIXED)")
    print(f"  held-out pivotal moves: {res['n_held_out_pivotal_moves']}  | top-leverage var: {top_var}")
    print(f"  A. directional accuracy / log-loss (ONLY the feature VALUES differ):")
    print(f"     GUESSED  state  acc {a['guessed_state']['accuracy']}  ll {a['guessed_state']['logloss']}")
    print(f"     GROUNDED state  acc {a['grounded_state']['accuracy']}  ll {a['grounded_state']['logloss']}"
          f"   (+{a['accuracy_lift']} acc, skill {a['grounding_skill_logloss']} vs guessed)")
    print(f"     TRIAGE (top var only) acc {a['triage_grounded_top_only']['accuracy']}  "
          f"ll {a['triage_grounded_top_only']['logloss']}   (skill {a['triage_skill_logloss']})")
    b = res["B_ground_spec_compiler"]
    print(f"  B. ground_spec -> compiler: guessed spec P(hike) = {b['p_hike_guessed_state_all_months']} for EVERY "
          f"month (blind);")
    print(f"     grounded spec P(hike) = {b['p_hike_grounded_on_actual_hike_months']} on hike months vs "
          f"{b['p_hike_grounded_on_actual_cut_months']} on cut months (responds to the as-of world)")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
