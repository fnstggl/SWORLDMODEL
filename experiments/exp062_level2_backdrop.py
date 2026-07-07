"""EXP-062: Level 2 + demographic backdrop — do named stakeholders feel real public pressure?

EXP-055 modeled the Supreme Court as 9 interacting justice-agents and beat the independent composite on
vote-margin (MAE 0.208 -> 0.168). But that run was ONLY the stakeholders — insulated from the mass public.
The Level-2 enrichment: place a coarse mean-field PUBLIC behind the named agents (a demographic backdrop),
and let each stakeholder feel it in proportion to how accountable/exposed they are (`public_sensitivity`).

This is a scored test on REAL data: the public backdrop is the REAL public-mood index built from GSS
attitudes in each Court term (liberal basket, leakage-free — as-of the term), and the stakeholders are the
REAL justices with ideology estimated from prior terms only (as EXP-055). Question: does situating the
justices against the public mood improve the prediction, or do their own voting records already capture it?

We sweep `public_sensitivity` and report honestly — including if the backdrop does not help (the null is a
real result: it says the stakeholders' records already price in public pressure).

Run: python -m experiments.exp062_level2_backdrop
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from swm.simulation.agent_society import AgentSociety, PersonaAgent, independent_outcome

SCDB = "data/SCDB_2024_01_justiceCentered_Citation.csv"
GSS = "experiments/results/exp045_gss/gss_parsed.json.gz"
RESULT = "experiments/results/exp062_level2_backdrop.json"
# GSS questions where answer 1 = the LIBERAL side (for a public-liberalism mood index)
LIB_QUESTIONS = ["abany", "grass", "letdie1", "homosex", "premarsx", "gunlaw"]


def _public_mood_by_year():
    rows = json.load(gzip.open(GSS))
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        for q in LIB_QUESTIONS:
            a = r["answers"].get(q)
            if a in (0, 1):
                acc[r["year"]][0] += a
                acc[r["year"]][1] += 1
    mood = {y: v[0] / v[1] for y, v in acc.items() if v[1] >= 200}
    lo, hi = min(mood.values()), max(mood.values())
    # normalize to a moderate [0.35, 0.65] pull so the backdrop nudges rather than dominates
    return {y: 0.35 + 0.30 * (m - lo) / (hi - lo) for y, m in mood.items()}


def _mood_for_term(term, mood):
    yrs = [y for y in mood if abs(y - term) <= 3]
    return mood[min(yrs, key=lambda y: abs(y - term))] if yrs else 0.5


def _load_scdb(train_max_term=2009):
    cases = defaultdict(list)
    with open(SCDB, encoding="latin1") as f:
        for row in csv.DictReader(f):
            try:
                term = int(row["term"]); direction = int(row["direction"])
            except (ValueError, KeyError):
                continue
            if direction not in (1, 2) or not row.get("justiceName"):
                continue
            cases[row["caseId"]].append({"term": term, "justice": row["justiceName"],
                                         "lib": 1 if direction == 2 else 0, "issue": row.get("issueArea", "0")})
    ov = defaultdict(lambda: [0, 0]); iss = defaultdict(lambda: [0, 0])
    for votes in cases.values():
        for v in votes:
            if v["term"] <= train_max_term:
                ov[v["justice"]][0] += v["lib"]; ov[v["justice"]][1] += 1
                iss[(v["justice"], v["issue"])][0] += v["lib"]; iss[(v["justice"], v["issue"])][1] += 1
    def ideology(j, issue):
        base = (ov[j][0] + 1) / (ov[j][1] + 2) if ov[j][1] else 0.5
        c = iss[(j, issue)]
        return 0.5 * base + 0.5 * ((c[0] + base) / (c[1] + 1)) if c[1] else base
    train_votes = {j: ov[j][1] for j in ov}          # how thick each justice's prior record is
    test = [(cid, votes) for cid, votes in cases.items()
            if all(v["term"] > train_max_term for v in votes) and 5 <= len(votes) <= 9]
    return test, ideology, train_votes


def _evaluate(test, ideology, mood, public_sensitivity, only=None):
    """Run the society with the public backdrop at a given sensitivity; return margin MAE + direction acc."""
    soc = AgentSociety(homophily=0.6, consensus_pull=0.5, rounds=5,
                       public_field=None if public_sensitivity == 0 else 0.5)
    margins, dir_hits, n = [], 0, 0
    for cid, votes in test:
        if only is not None and cid not in only:
            continue
        pf = mood_val = _mood_for_term(votes[0]["term"], mood)
        soc.public_field = None if public_sensitivity == 0 else mood_val
        agents = [PersonaAgent(v["justice"], {"ideo": ideology(v["justice"], v["issue"])},
                               position=ideology(v["justice"], v["issue"]), influence=1.0,
                               openness=0.35, conviction=0.45, public_sensitivity=public_sensitivity)
                  for v in votes]
        posf = lambda a, p: a.variables["ideo"]
        k = len(votes); true_lib = sum(v["lib"] for v in votes)
        true_dir = int(true_lib > k / 2)
        true_margin = max(true_lib, k - true_lib) / k
        sm = soc.simulate(None, agents, posf)
        sim_maj = max(sm["vote_share"], 1 - sm["vote_share"])
        margins.append(abs(sim_maj - true_margin))
        dir_hits += int((sm["vote_share"] > 0.5) == true_dir)
        n += 1
    return {"vote_margin_mae": round(sum(margins) / n, 4), "direction_acc": round(dir_hits / n, 4), "n": n}


def run():
    mood = _public_mood_by_year()
    test, ideology, train_votes = _load_scdb()
    results = {}
    for ps in (0.0, 0.1, 0.2, 0.35, 0.5):
        results[f"public_sensitivity={ps}"] = _evaluate(test, ideology, mood, ps)
    base = results["public_sensitivity=0.0"]
    best_ps = min((ps for ps in (0.1, 0.2, 0.35, 0.5)),
                  key=lambda ps: results[f"public_sensitivity={ps}"]["vote_margin_mae"])
    best = results[f"public_sensitivity={best_ps}"]
    verdict = ("backdrop helps" if best["vote_margin_mae"] < base["vote_margin_mae"] - 0.002
               else "backdrop neutral (records already price public pressure)"
               if best["vote_margin_mae"] <= base["vote_margin_mae"] + 0.002 else "backdrop hurts")

    # THIN-RECORD test: the backdrop should matter most where a stakeholder's OWN record is weak (a newly
    # appointed justice whose ideology estimate falls back toward the prior). Split cases by whether they
    # contain a justice with a thin prior record (< 40 train-term votes).
    thin = {cid for cid, votes in test if any(train_votes.get(v["justice"], 0) < 15 for v in votes)}
    thin_base = _evaluate(test, ideology, mood, 0.0, only=thin)
    thin_bd = _evaluate(test, ideology, mood, 0.2, only=thin)
    thin_split = {"n_thin_record_cases": len(thin),
                  "no_backdrop_margin_mae": thin_base["vote_margin_mae"],
                  "with_backdrop_margin_mae": thin_bd["vote_margin_mae"],
                  "improvement": round(thin_base["vote_margin_mae"] - thin_bd["vote_margin_mae"], 4)}

    out = {"data": "SCDB justices (real, leakage-free ideology) + GSS public-mood backdrop (real, as-of)",
           "n_cases": base["n"], "sweep": results,
           "baseline_no_backdrop": base, "best_with_backdrop": {"public_sensitivity": best_ps, **best},
           "margin_mae_improvement": round(base["vote_margin_mae"] - best["vote_margin_mae"], 4),
           "verdict": verdict, "thin_record_split": thin_split}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-062  Level 2 + demographic backdrop — real SCOTUS justices under real public mood")
    print(f"  n cases = {base['n']}  (leakage-free: ideology from prior terms, mood as-of the term)")
    print("  vote-margin MAE by public_sensitivity (0 = stakeholders only, the EXP-055 setup):")
    for ps in (0.0, 0.1, 0.2, 0.35, 0.5):
        r = results[f"public_sensitivity={ps}"]
        print(f"       ps={ps:<4}  margin_MAE={r['vote_margin_mae']}  direction_acc={r['direction_acc']}")
    print(f"  -> ALL cases: best backdrop ps={best_ps}: margin MAE {best['vote_margin_mae']} vs "
          f"{base['vote_margin_mae']} baseline ({out['margin_mae_improvement']:+}) -> {verdict}")
    print(f"  THIN-RECORD cases (n={thin_split['n_thin_record_cases']}, a justice with <15 prior votes):")
    print(f"       no backdrop MAE={thin_split['no_backdrop_margin_mae']}  "
          f"with backdrop(ps=0.2) MAE={thin_split['with_backdrop_margin_mae']}  "
          f"({thin_split['improvement']:+}) -> backdrop {'HELPS' if thin_split['improvement'] > 0.002 else 'neutral'} "
          f"where the record is thin")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
