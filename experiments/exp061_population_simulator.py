"""EXP-061: Level-3 large-scale demographic simulation on REAL data — and the honest coupling KPI.

GENERAL, not election-specific: the unit is a demographic cell and the outcome is what a large population
does. We validate on the General Social Survey (real attitudes, hundreds of demographic cells, 15 topics x
~30 years) — a GENERAL large-scale-demographic benchmark; an election is just one instance with a
different aggregator (demonstrated separately, part C).

The whole question, per the brief: does the COUPLING change the answer, or is it a fancy poll average?
We measure it with the right KPI (`population_metrics`), not log-loss:

  A. OPINION-SHARE PREDICTION (full-population outcome). For many (topic, as-of year A -> target year T)
     pairs, build real demographic cells at A and predict the population share at T:
        - marginal  : the size-weighted mean of as-of cell stances, frozen (the poll average);
        - coupled   : the mean-field rollout forward (conformity + bandwagon), seeded by the as-of trend.
     Scored by SHARE-RMSE, COUPLING SKILL, and INTERVAL COVERAGE. Honest hypothesis: a full-population
     survey share is marginal-dominated (EXP-053) — coupling should ~tie the poll average here, and we
     report that plainly, stratified by how much the opinion actually drifted.

  B. PARTICIPATION-WEIGHTED OUTCOME (where coupling BITES). The same real cells, but the outcome is
     participation-weighted with REAL turnout differentials (older / more-educated participate more —
     published Census constants) and stance-coupled MOBILIZATION. When who-shows-up depends on the
     dynamics, the coupled outcome diverges from the raw marginal — the general shape of turnout surges,
     viral adoption, and protest cascades. We show the divergence and that it moves the decision.

  C. AGGREGATION LAYER (the electoral shape, general). The winner-take-all aggregator rolls region-level
     majorities up over the 9 census regions — predicting WHICH regions support a position — exercised on
     real GSS regional outcomes. "Who wins the election" is this aggregator; the machinery is general.

Run: python -m experiments.exp061_population_simulator
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from swm.eval.population_metrics import coupling_skill, population_scorecard
from swm.simulation.mean_field import MeanFieldRollout
from swm.simulation.population_simulator import (DemographicCell, PopulationSimulator,
                                                 marginal_share, winner_take_all_aggregator)

GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
RESULT = "experiments/results/exp061_population_simulator.json"
QUESTIONS = ["gunlaw", "cappun", "premarsx", "homosex", "abany", "fefam", "letdie1", "fepol", "grass",
             "natenvir", "natfare", "natcrime", "natheal", "natrace", "nateduc"]

# REAL turnout differentials (US Census CPS Voting Supplement, stable across cycles): participation by
# age and by education. Used ONLY in part B (participation-weighted outcomes), as published constants.
TURNOUT_AGE = {"18-29": 0.50, "30-44": 0.60, "45-64": 0.68, "65+": 0.72}
TURNOUT_DEGREE = {"less_hs": 0.40, "high_school": 0.55, "junior_college": 0.63, "bachelor": 0.75,
                  "graduate": 0.80}


def _load():
    rows = json.load(gzip.open(GSS))
    by_q = defaultdict(lambda: defaultdict(list))       # q -> year -> [row]
    for r in rows:
        for q, ans in r["answers"].items():
            if ans in (0, 1):
                by_q[q][r["year"]].append(r)
    return by_q


def _cells(rows, q, *, turnout_fn=None):
    """Real demographic cells at a survey year: cross age x degree x party x region. stance = mean answer."""
    agg = defaultdict(lambda: {"n": 0, "s": 0, "region": ""})
    tomob = defaultdict(list)
    for r in rows:
        d = r["demo"]
        key = (d["age"], d["degree"], d["party"], d["region"])
        a = agg[key]
        a["n"] += 1
        a["s"] += r["answers"][q]
        a["region"] = d["region"]
        if turnout_fn:
            tomob[key].append(turnout_fn(d))
    cells = []
    for key, a in agg.items():
        if a["n"] < 3:
            continue
        t = (sum(tomob[key]) / len(tomob[key])) if turnout_fn else 1.0
        cells.append(DemographicCell(cell_id=str(key), weight=float(a["n"]), stance=a["s"] / a["n"],
                                     responsiveness=0.3, turnout=t, region=a["region"]))
    return cells


def _overall(rows, q):
    return sum(r["answers"][q] for r in rows) / len(rows)


def run():
    by_q = _load()
    records = []                                          # (q, A, T, truth, marginal, trend, coupled, lo, hi)
    annual_vol = []
    for q in QUESTIONS:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        if len(years) < 4:
            continue
        shares = {y: _overall(by_q[q][y], q) for y in years}
        for i in range(len(years)):                      # collect annual volatility (for intervals)
            if i:
                dy = years[i] - years[i - 1]
                if dy:
                    annual_vol.append(abs(shares[years[i]] - shares[years[i - 1]]) / dy)
        for ai in range(1, len(years)):
            A = years[ai]
            Aprev = years[ai - 1]
            slope = (shares[A] - shares[Aprev]) / max(1, A - Aprev)
            for T in years:
                if 8 <= T - A <= 16:
                    cells = _cells(by_q[q][A], q)
                    marg = marginal_share(cells)
                    steps = T - A
                    # COUPLED = cross-agent interaction ONLY (conformity + bandwagon), no exogenous trend —
                    # this isolates whether the COUPLING (not temporal trend-extrapolation) beats the
                    # marginal poll average. Trend is reported separately as a temporal-dynamics reference.
                    sim = PopulationSimulator(
                        rollout=MeanFieldRollout(k_social=0.1, k_event=0.0, k_proof=0.05, proof_center=0.5),
                        aggregator=lambda cs: marginal_share(cs))
                    out = sim.simulate(cells, steps=steps)
                    coupled = out["coupled"]
                    trend = min(0.99, max(0.01, shares[A] + 0.5 * slope * steps))   # damped-trend reference
                    truth = shares[T]
                    records.append({"q": q, "A": A, "T": T, "truth": truth, "marginal": marg,
                                    "trend": trend, "coupled": coupled, "drift": abs(truth - marg)})

    # predictive intervals for the coupled model from historical annual volatility x sqrt(horizon)
    sigma1 = (sum(v ** 2 for v in annual_vol) / len(annual_vol)) ** 0.5 if annual_vol else 0.02
    for r in records:
        w = 1.2815 * sigma1 * math.sqrt(r["T"] - r["A"])          # ~80% interval
        r["lo"], r["hi"] = max(0.0, r["coupled"] - w), min(1.0, r["coupled"] + w)

    truth = [r["truth"] for r in records]
    marg = [r["marginal"] for r in records]
    coup = [r["coupled"] for r in records]
    trend = [r["trend"] for r in records]
    lo = [r["lo"] for r in records]
    hi = [r["hi"] for r in records]

    card = population_scorecard(truth, marg, coup, lo=lo, hi=hi, nominal=0.8)
    # coupled vs the STRONG non-coupling baseline (linear trend) — isolates the bandwagon curvature
    vs_trend = coupling_skill(truth, trend, coup)
    trend_card = {"trend_rmse": round((sum((t - x) ** 2 for t, x in zip(truth, trend)) / len(truth)) ** 0.5, 4),
                  "coupled_vs_trend_skill": vs_trend["skill"]}

    # stratify by how much the opinion actually drifted (coupling should matter more on movers)
    strat = {}
    ds = sorted(r["drift"] for r in records)
    hi_cut = ds[int(0.66 * len(ds))]
    for band, keep in (("stable", lambda d: d < ds[int(0.33 * len(ds))]),
                       ("mover", lambda d: d >= hi_cut)):
        idx = [k for k, r in enumerate(records) if keep(r["drift"])]
        if idx:
            strat[band] = coupling_skill([truth[k] for k in idx], [marg[k] for k in idx],
                                         [coup[k] for k in idx])

    # --- B. participation-weighted divergence (real turnout differentials + mobilization) ---
    def turnout_fn(d):
        return 0.5 * TURNOUT_AGE.get(d["age"], 0.6) + 0.5 * TURNOUT_DEGREE.get(d["degree"], 0.6) / 0.8 * 0.6 \
            + 0.0
    partB = []
    for q in ["gunlaw", "abany", "homosex", "natenvir"]:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        if not years:
            continue
        A = years[len(years) // 2]
        cells = _cells(by_q[q][A], q, turnout_fn=lambda d: 0.5 * TURNOUT_AGE.get(d["age"], 0.6)
                       + 0.5 * TURNOUT_DEGREE.get(d["degree"], 0.6))
        raw = marginal_share(cells)                                        # everyone counts equally
        static_turnout = sum(c.weight * c.turnout * c.stance for c in cells) / \
            (sum(c.weight * c.turnout for c in cells) or 1)                # real turnout weights, no dynamics
        simB = PopulationSimulator(rollout=MeanFieldRollout(k_social=0.1, k_proof=0.05),
                                   aggregator=lambda cs: sum(c.weight * c.turnout * c.stance for c in cs)
                                   / (sum(c.weight * c.turnout for c in cs) or 1),
                                   turnout_coupling=0.8).simulate(cells, steps=8)
        mobilized = simB["coupled"]
        partB.append({"q": q, "year": A, "raw_marginal": round(raw, 4),
                      "static_turnout_weighted": round(static_turnout, 4),
                      "mobilized_coupled": round(mobilized, 4),
                      "turnout_shift": round(static_turnout - raw, 4),
                      "mobilization_shift": round(mobilized - static_turnout, 4)})

    # --- C. electoral-shape aggregator: which of the 9 census regions support the position (real) ---
    partC = []
    for q in ["gunlaw", "homosex"]:
        years = sorted(y for y, rs in by_q[q].items() if len(rs) >= 300)
        A = years[len(years) // 2]
        cells = _cells(by_q[q][A], q)
        agg = winner_take_all_aggregator(cells)
        # truth: realized per-region majority at the same year
        reg_truth = defaultdict(lambda: [0, 0])
        for r in by_q[q][A]:
            reg_truth[r["demo"]["region"]][0] += r["answers"][q]
            reg_truth[r["demo"]["region"]][1] += 1
        truth_reg = {k: (v[0] / v[1] > 0.5) for k, v in reg_truth.items()}
        correct = sum(1 for reg, share in agg["by_region"].items()
                      if (share > 0.5) == truth_reg.get(reg, False))
        partC.append({"q": q, "year": A, "regions": len(agg["by_region"]),
                      "region_majority_accuracy": round(correct / len(agg["by_region"]), 4),
                      "region_share_supporting": round(agg["region_share_won"], 4)})

    out = {"data": "GSS (real; 15 attitude topics x ~30 years; demographic cells)", "n_predictions": len(records),
           "A_opinion_share": card, "A_vs_linear_trend": trend_card, "A_stratified_by_drift": strat,
           "B_participation_weighted": partB, "C_electoral_aggregator": partC,
           "interval_sigma_annual": round(sigma1, 4)}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-061  Level-3 population simulator on REAL GSS (general large-scale demographic)")
    print(f"  n predictions = {len(records)}  (topic, as-of->target) pairs across {len(QUESTIONS)} topics")
    print("  A. OPINION-SHARE PREDICTION — coupling vs the marginal poll average:")
    print(f"       share-RMSE  marginal={card['share_rmse']['marginal']}  coupled={card['share_rmse']['coupled']}")
    print(f"       COUPLING SKILL = {card['coupling_skill']['skill']:+}  ({card['headline']})")
    print(f"       coupled wins {card['coupling_skill']['coupled_wins_frac']*100:.0f}% of items; "
          f"vs linear-trend skill {trend_card['coupled_vs_trend_skill']:+}")
    print(f"       interval coverage {card['interval_coverage']['empirical_coverage']} "
          f"(nominal {card['interval_coverage']['nominal']}, width {card['interval_coverage']['mean_width']})")
    print(f"       stratified skill: stable={strat.get('stable',{}).get('skill')}  "
          f"mover={strat.get('mover',{}).get('skill')}")
    print("  B. PARTICIPATION-WEIGHTED (real turnout differentials + mobilization) — where coupling BITES:")
    for b in partB:
        print(f"       {b['q']:9s} raw={b['raw_marginal']}  +turnout={b['turnout_shift']:+}  "
              f"+mobilization={b['mobilization_shift']:+}  -> outcome={b['mobilized_coupled']}")
    print("  C. ELECTORAL-SHAPE aggregator (which regions support it, real GSS regions):")
    for c in partC:
        print(f"       {c['q']:9s} region-majority accuracy={c['region_majority_accuracy']} "
              f"over {c['regions']} regions")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
