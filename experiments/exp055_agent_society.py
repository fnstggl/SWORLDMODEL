"""EXP-055: agent-based simulation — does modelling AGENTS-WHO-INTERACT beat the composite? (institutional)

The audit's flagship failure was that `simulate_population` is a mean of independent regressions. This
validates the real alternative — `AgentSociety`, where persona agents take positions and INTERACT — on two
axes:

  A. CONTROLLED mechanism proofs: emergent outcomes a composite CANNOT produce —
     (1) an influential minority FLIPS a vote the independent count would lose;
     (2) deliberation drives CONSENSUS (a split body converges);
     (3) homophily produces POLARIZATION (blocs harden).

  B. REAL institutional agents — the Supreme Court (SCDB, per-justice votes). Model the 9 justices as
     agents (ideology from PRIOR terms, leakage-free), simulate their deliberation (ideological blocs +
     a consensus pull), and predict the decision direction, the vote MARGIN, and unanimity — vs the
     INDEPENDENT baseline (each justice votes their ideology, majority wins = the composite). This
     directly answers the pushback: institutional events ARE populations of modelable agents.

Run: python -m experiments.exp055_agent_society
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from swm.simulation.agent_society import AgentSociety, PersonaAgent, independent_outcome

RESULT = "experiments/results/exp055_agent_society.json"
SCDB = "data/SCDB_2024_01_justiceCentered_Citation.csv"


# ---- A. controlled mechanism proofs ----
def _controlled():
    prop = None
    # (1) influential minority flips a vote: 6 weak-conviction agents lean NO (0.45), 3 high-influence,
    #     high-conviction agents strongly YES (0.9). Independent count: NO wins (6>3). Deliberation: the
    #     influential bloc pulls the persuadable majority over the line.
    agents = ([PersonaAgent(f"weak{i}", {"v": 0.5}, position=0.45, influence=1.0, openness=0.6, conviction=0.2)
               for i in range(6)]
              + [PersonaAgent(f"leader{i}", {"v": 0.5}, position=0.95, influence=4.0, openness=0.1, conviction=0.9)
                 for i in range(3)])
    pf = lambda a, p: a.position
    indep = independent_outcome(agents, pf, prop)
    soc = AgentSociety(homophily=0.0, consensus_pull=0.0, rounds=8).simulate(prop, agents, pf)
    flip = {"independent_passes": indep["passes"], "simulated_passes": soc["passes"],
            "independent_share": round(indep["vote_share"], 3), "simulated_share": round(soc["vote_share"], 3),
            "minority_flipped_outcome": indep["passes"] != soc["passes"]}

    # (2) deliberation -> consensus: a split body (half 0.2, half 0.8) with a consensus pull converges
    split = [PersonaAgent(f"lo{i}", {"v": 0.1}, position=0.2, openness=0.5, conviction=0.2) for i in range(5)] + \
            [PersonaAgent(f"hi{i}", {"v": 0.9}, position=0.8, openness=0.5, conviction=0.2) for i in range(5)]
    con = AgentSociety(homophily=0.0, consensus_pull=0.7, rounds=10).simulate(None, split, lambda a, p: a.position)
    spread = max(con["final_positions"]) - min(con["final_positions"])
    consensus = {"initial_spread": 0.6, "final_spread": round(spread, 3), "converged": spread < 0.3}

    # (3) homophily -> polarization: two blocs listen mostly to their own kind; the gap persists/hardens
    blocs = [PersonaAgent(f"L{i}", {"v": 0.1}, position=0.35, openness=0.6, conviction=0.2) for i in range(5)] + \
            [PersonaAgent(f"R{i}", {"v": 0.9}, position=0.65, openness=0.6, conviction=0.2) for i in range(5)]
    pol = AgentSociety(homophily=1.0, consensus_pull=0.0, confidence_bound=0.5,
                       rounds=10).simulate(None, blocs, lambda a, p: a.position)
    fp = pol["final_positions"]
    gap = (sum(fp[5:]) / 5) - (sum(fp[:5]) / 5)
    polar = {"initial_bloc_gap": 0.3, "final_bloc_gap": round(gap, 3), "stayed_polarized": gap > 0.2}

    return {"minority_flip": flip, "consensus": consensus, "polarization": polar,
            "all_emergent_effects_present": flip["minority_flipped_outcome"] and consensus["converged"]
            and polar["stayed_polarized"]}


# ---- B. real Supreme Court ----
def _load_scdb(train_max_term=2009):
    cases = defaultdict(list)
    with open(SCDB, encoding="latin1") as f:
        for row in csv.DictReader(f):
            try:
                term = int(row["term"]); direction = int(row["direction"])  # 1 cons, 2 lib
            except (ValueError, KeyError):
                continue
            if direction not in (1, 2) or not row.get("justiceName"):
                continue
            cases[row["caseId"]].append({"term": term, "justice": row["justiceName"],
                                         "lib": 1 if direction == 2 else 0,
                                         "issue": row.get("issueArea", "0"),
                                         "dd": row.get("decisionDirection", "0")})
    # per-justice ideology from TRAIN terms only (overall + per issue-area liberal rate)
    ov = defaultdict(lambda: [0, 0]); iss = defaultdict(lambda: [0, 0])
    for cid, votes in cases.items():
        for v in votes:
            if v["term"] <= train_max_term:
                ov[v["justice"]][0] += v["lib"]; ov[v["justice"]][1] += 1
                iss[(v["justice"], v["issue"])][0] += v["lib"]; iss[(v["justice"], v["issue"])][1] += 1
    def ideology(j, issue):
        base = (ov[j][0] + 1) / (ov[j][1] + 2) if ov[j][1] else 0.5
        c = iss[(j, issue)]
        return 0.5 * base + 0.5 * ((c[0] + base) / (c[1] + 1)) if c[1] else base
    test = [(cid, votes) for cid, votes in cases.items()
            if all(v["term"] > train_max_term for v in votes) and 5 <= len(votes) <= 9]
    return test, ideology


def _scotus():
    test, ideology = _load_scdb()
    soc = AgentSociety(homophily=0.6, consensus_pull=0.5, rounds=5)
    ind_dir, sim_dir, ind_margin, sim_margin = 0, 0, [], []
    ind_vote, sim_vote, nvotes = 0, 0, 0
    n = 0
    for cid, votes in test:
        agents = [PersonaAgent(v["justice"], {"ideo": ideology(v["justice"], v["issue"])},
                               position=ideology(v["justice"], v["issue"]),
                               influence=1.0, openness=0.35, conviction=0.45) for v in votes]
        pf = lambda a, p: a.variables["ideo"]
        true_lib = sum(v["lib"] for v in votes); k = len(votes)
        true_dir = int(true_lib > k / 2)
        true_margin = max(true_lib, k - true_lib) / k          # majority fraction (unanimity=1.0)
        ind = independent_outcome(agents, pf, None)
        sm = soc.simulate(None, agents, pf)
        ind_dir += int((ind["vote_share"] > 0.5) == true_dir)
        sim_dir += int((sm["vote_share"] > 0.5) == true_dir)
        ind_maj = max(ind["vote_share"], 1 - ind["vote_share"])
        sim_maj = max(sm["vote_share"], 1 - sm["vote_share"])
        ind_margin.append(abs(ind_maj - true_margin)); sim_margin.append(abs(sim_maj - true_margin))
        # per-justice vote accuracy
        for a, v in zip(agents, votes):
            nvotes += 1
            ind_vote += int((a.variables["ideo"] > 0.5) == v["lib"])
        for a, v in zip(agents, votes):
            sim_vote += int((a.position > 0.5) == v["lib"])
        n += 1
    return {"n_cases": n, "n_justice_votes": nvotes,
            "decision_direction_acc": {"independent": round(ind_dir / n, 4), "agent_sim": round(sim_dir / n, 4)},
            "vote_margin_mae": {"independent": round(sum(ind_margin) / n, 4), "agent_sim": round(sum(sim_margin) / n, 4)},
            "individual_vote_acc": {"independent": round(ind_vote / nvotes, 4), "agent_sim": round(sim_vote / nvotes, 4)}}


def run():
    controlled = _controlled()
    scotus = _scotus()
    out = {"A_controlled": controlled, "B_supreme_court": scotus,
           "agent_sim_beats_independent_on_margin":
               scotus["vote_margin_mae"]["agent_sim"] < scotus["vote_margin_mae"]["independent"]}

    print("EXP-055 agent-based simulation vs the composite")
    print("  A. CONTROLLED mechanism proofs (emergent outcomes a mean cannot produce):")
    f = controlled["minority_flip"]
    print(f"     (1) influential minority: independent passes {f['independent_passes']} "
          f"-> simulated passes {f['simulated_passes']}  (flipped: {f['minority_flipped_outcome']})")
    print(f"     (2) deliberation -> consensus: spread 0.6 -> {controlled['consensus']['final_spread']} "
          f"(converged: {controlled['consensus']['converged']})")
    print(f"     (3) homophily -> polarization: bloc gap 0.3 -> {controlled['polarization']['final_bloc_gap']} "
          f"(stayed polarized: {controlled['polarization']['stayed_polarized']})")
    print("  B. REAL Supreme Court (justice-agents, SCDB, leakage-free ideology from prior terms):")
    print(f"     n_cases={scotus['n_cases']}  n_votes={scotus['n_justice_votes']}")
    for k in ("decision_direction_acc", "vote_margin_mae", "individual_vote_acc"):
        print(f"     {k:<24} independent {scotus[k]['independent']}   agent_sim {scotus[k]['agent_sim']}")
    print(f"     -> agent-sim beats independent on vote-margin: {out['agent_sim_beats_independent_on_margin']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
