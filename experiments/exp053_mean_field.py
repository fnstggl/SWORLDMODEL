"""EXP-053: does the mean-field COUPLED roll-forward beat the independent mean? (the audit's decisive test)

The simulation audit's mandate: `simulate_population` is `sum(ps)/n` — a mean of independent per-person
readouts. Making aggregation non-separable (each agent updates toward the evolving aggregate) earns the
word "simulate" ONLY if the coupled roll-forward beats that independent mean on a real outcome. This tests
it honestly, prepared for either verdict.

Two axes:
  A. CONTROLLED — a social cascade (threshold/social-proof adoption) has a genuine S-curve TRAJECTORY. A
     mean of independent predictions is a flat line; can the coupled model recover the trajectory (turning
     point) it cannot? Tests whether coupling captures EMERGENCE a composite cannot represent.
  B. REAL — GSS opinion shares. Holding the per-agent beliefs FIXED, does the coupled aggregation beat the
     independent mean (the current flagship) and persistence at predicting the true share? This is the
     honest "does interaction help the number" test.

Run: python -m experiments.exp053_mean_field
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from swm.simulation.mean_field import Agent, MeanFieldRollout, agents_from_cells
from experiments.datasets_gss import load
from experiments.exp045_population_rollout import ATTRS, _grounded_model, _predict_share, _share
from swm.variables.pooled_readout import encode, onehot_vocab

RESULT = "experiments/results/exp053_mean_field.json"


# ---- A. controlled cascade: does coupling recover an emergent S-curve a mean cannot? ----
def _true_threshold_cascade(n=400, steps=12, seed=0):
    """Granovetter-style threshold cascade: each agent adopts when the current adoption rate exceeds their
    personal threshold. Produces a real S-curve. Deterministic (seeded)."""
    import random
    rng = random.Random(seed)
    thresh = sorted(rng.random() * 0.6 for _ in range(n))   # heterogeneous thresholds in [0,0.6)
    adopted = [t < 0.05 for t in thresh]                    # a small seed already adopted
    traj = [sum(adopted) / n]
    for _ in range(steps):
        rate = sum(adopted) / n
        adopted = [a or (thresh[i] <= rate) for i, a in enumerate(adopted)]
        traj.append(sum(adopted) / n)
    return traj


def _controlled():
    true = _true_threshold_cascade()
    steps = len(true) - 1
    init_rate = true[0]
    # independent mean: no dynamics — predicts the initial rate for every step (a flat line)
    flat = [init_rate] * len(true)
    flat_mae = sum(abs(a - b) for a, b in zip(flat, true)) / len(true)
    # coupled mean-field with social proof; fit k_proof/k_social by a small grid to the FIRST HALF only
    half = len(true) // 2
    best, best_e = None, 1e9
    for kp in (0.2, 0.4, 0.7, 1.0, 1.5):
        for ks in (0.0, 0.1, 0.3):
            agents = [Agent(belief=init_rate, responsiveness=0.5, influence=1.0) for _ in range(200)]
            # bandwagon adoption: proof_center=0 so pull grows with the adoption rate (S-curve)
            mf = MeanFieldRollout(k_social=ks, k_event=0.0, k_proof=kp, proof_center=0.0)
            traj, _ = mf.roll(agents, steps)
            e = sum(abs(traj[i] - true[i]) for i in range(half)) / half
            if e < best_e:
                best_e, best = e, (kp, ks)
    agents = [Agent(belief=init_rate, responsiveness=0.5, influence=1.0) for _ in range(200)]
    traj, _ = MeanFieldRollout(k_social=best[1], k_event=0.0, k_proof=best[0], proof_center=0.0).roll(agents, steps)
    coupled_mae = sum(abs(traj[i] - true[i]) for i in range(len(true))) / len(true)
    return {"true_trajectory": [round(x, 3) for x in true],
            "coupled_trajectory": [round(x, 3) for x in traj],
            "independent_mean_flat_mae": round(flat_mae, 4), "coupled_trajectory_mae": round(coupled_mae, 4),
            "coupling_recovers_cascade": coupled_mae < flat_mae, "fit": {"k_proof": best[0], "k_social": best[1]}}


# ---- B. real GSS: does coupled aggregation beat the independent mean on the true share? ----
def _cell_agents(model, vocab, year_rows):
    """Segment the year's population into demographic cells; each cell is an agent with belief = the
    grounded readout P, responsiveness ~ openness (younger/educated move more), influence ~ 1."""
    groups = defaultdict(list)
    for r in year_rows:
        key = (r["demo"].get("age"), r["demo"].get("ideology"), r["demo"].get("degree"))
        groups[key].append(r)
    resp = {"18-29": 0.55, "30-49": 0.4, "50-64": 0.3, "65+": 0.2}
    cells = []
    for key, rs in groups.items():
        b = sum(_predict_share(model, vocab, [r]) for r in rs) / len(rs)
        age = key[0]
        # influence = cell SIZE (population-proportional) so the independent-mean baseline equals the true
        # cross-sectional mean (fair, EXP-045-calibrated); only the COUPLING dynamics is the variable.
        cells.append((b, resp.get(age, 0.35), len(rs)))
    return agents_from_cells(cells)


def _gss(steps=3):
    recs = load()
    by_year = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r)
    years = sorted(by_year)
    items = ["cappun", "gunlaw", "grass", "abany", "homosex", "premarsx", "natheal", "natenvir", "natfare"]
    err = {"persistence": [], "independent_mean": [], "coupled": []}
    for item in items:
        iy = [y for y in years if _share(by_year[y], item) is not None]
        shares = {y: _share(by_year[y], item) for y in iy}
        for t in iy:
            tr_years = [y for y in iy if y < t]
            if len(tr_years) < 4:
                continue
            last = tr_years[-1]
            model, vocab = _grounded_model([r for y in tr_years for r in by_year[y]], item)
            if model is None:
                continue
            agents = _cell_agents(model, vocab, by_year[t])
            indep = sum(a.influence * a.belief for a in agents) / sum(a.influence for a in agents)
            # coupled: roll the SAME agents forward (conformity + social proof), read the aggregate
            mf = MeanFieldRollout(k_social=0.2, k_event=0.0, k_proof=0.15)
            _, coupled = mf.roll(agents, steps)
            actual = shares[t]
            err["persistence"].append(abs(shares[last] - actual))
            err["independent_mean"].append(abs(indep - actual))
            err["coupled"].append(abs(coupled - actual))
    return {m: round(sum(v) / len(v), 4) for m, v in err.items() if v}, len(err["coupled"])


def run():
    controlled = _controlled()
    gss, n_gss = _gss()
    out = {"A_controlled_cascade": controlled, "B_gss_aggregate": {"mae": gss, "n_forecasts": n_gss,
           "coupled_beats_independent_mean": gss["coupled"] < gss["independent_mean"],
           "coupled_beats_persistence": gss["coupled"] < gss["persistence"]}}

    print("EXP-053 mean-field coupling vs independent mean")
    print("  A. CONTROLLED cascade (does coupling recover an emergent S-curve a mean cannot?):")
    print(f"     independent-mean (flat) trajectory MAE {controlled['independent_mean_flat_mae']}")
    print(f"     coupled trajectory MAE            {controlled['coupled_trajectory_mae']}  "
          f"(recovers cascade: {controlled['coupling_recovers_cascade']})")
    print(f"     true:    {controlled['true_trajectory']}")
    print(f"     coupled: {controlled['coupled_trajectory']}")
    print("  B. REAL GSS aggregate (does coupled aggregation beat the independent mean on the true share?):")
    for m in ("persistence", "independent_mean", "coupled"):
        print(f"     {m:<18} MAE {gss[m]}")
    print(f"     -> coupled beats independent mean: {out['B_gss_aggregate']['coupled_beats_independent_mean']}; "
          f"beats persistence: {out['B_gss_aggregate']['coupled_beats_persistence']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
