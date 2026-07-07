"""EXP-070: the persistent World substrate — a coupled two-scale world, and does coupling earn its place?

Builds the first persistent, coupled world (`swm/world/substrate.py`) and holds it to the discipline the
brief demands: score COUPLED vs SEPARATE before scaling up.

  A. CROSS-SCALE FEEDBACK the substrate captures and separate models cannot. A bank-run world: an
     environment RUMOR -> depositors' (individuals) withdrawal intent -> the BANK (institution) distress ->
     which feeds BACK into the rumor and the depositors. With the edges wired, a single shock CASCADES to
     failure; with the edges cut (each scale alone), the same shock fizzles. This is emergent cross-scale
     contagion — non-separable by construction, and the reason a shared world can differ from independent
     models.

  B. A REAL scored two-scale test: individuals -> the institution they sit in. The Supreme Court as a World
     of 9 justice entities (individual scale: each carries an ideology STATE that DRIFTS as their record
     accumulates) coupled UP into the Court entity (institution scale: a committee vote over the justices'
     current states). We score, on real SCDB cases (leakage-free), whether coupling the individual-scale
     DYNAMICS up to the institution beats treating each justice as a STATIC input:
        SEPARATE : justice ideology frozen at its train-era value        (no individual dynamics)
        COUPLED  : justice ideology is the as-of running estimate         (individual drift, coupled up)
     Same committee mechanism both ways; the only difference is whether the lower scale is simulated and
     wired up. If COUPLED wins, the shared world earns its place here; if it ties, we do not scale up here.

Run: python -m experiments.exp070_world_substrate
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from swm.simulation.agent_society import AgentSociety, PersonaAgent
from swm.world.substrate import Entity, World, rollout

SCDB = "data/SCDB_2024_01_justiceCentered_Citation.csv"
RESULT = "experiments/results/exp070_world_substrate.json"


# ============================ A. cross-scale feedback: a bank run ============================
def _bank_world(n_dep=12, k_env=1.0, k_peer=0.7, coupled=True):
    w = World()
    # rumor absorbs the exogenous shock, decays, and is re-fed by bank distress (contagion)
    w.add(Entity("rumor", "environment", {"level": 0.0},
                 step_fn=lambda s, inp, dt, rng: {"level": min(1.0, 0.95 * s["level"]
                         + 0.6 * inp.get("bank_distress", 0.0) + inp.get("shock", 0.0))}))
    for i in range(n_dep):
        w.add(Entity(f"dep{i}", "individual", {"intent": 0.1},
                     step_fn=lambda s, inp, dt, rng: {"intent": min(1.0, max(0.0, s["intent"] + 0.4 * (
                         k_env * inp.get("rumor", 0.0) + k_peer * inp.get("bank_distress", 0.0) - s["intent"])))}))

    def bank_step(s, inp, dt, rng):
        intents = [v for k, v in inp.items() if k.startswith("dep")]
        return {"distress": (sum(i > 0.5 for i in intents) / len(intents)) if intents else 0.0}
    w.add(Entity("bank", "institution", {"distress": 0.0}, step_fn=bank_step,
                 readout_fn=lambda s, inp, rng: "FAILED" if s["distress"] > 0.5 else "stable"))
    if coupled:
        for i in range(n_dep):
            w.couple("rumor", f"dep{i}", lambda s: {"rumor": s["level"]})
            w.couple("bank", f"dep{i}", lambda s: {"bank_distress": s["distress"]})
            w.couple(f"dep{i}", "bank", (lambda j: (lambda s: {f"dep{j}": s["intent"]}))(i))
        w.couple("bank", "rumor", lambda s: {"bank_distress": s["distress"]})
    return w


def _bank_run():
    shock = {"rumor": {"shock": 0.8}}                         # a one-time exogenous rumor at t=0
    out = {}
    for label, coupled in (("coupled_world", True), ("separate_scales", False)):
        # run 14 steps; inject the shock on the first step only
        w = _bank_world(coupled=coupled)
        rng = random.Random(0)
        traj = []
        for step in range(14):
            w.advance(1.0, external=(shock if step == 0 else None), rng=rng)
            traj.append(round(w.entities["bank"].state["distress"], 3))
        out[label] = {"final_state": w.query("bank"), "bank_distress_trajectory": traj,
                      "final_rumor": round(w.entities["rumor"].state["level"], 3)}
    out["reads"] = ("identical shock: the COUPLED world cascades to bank FAILURE via rumor->depositors->"
                    "bank->rumor feedback; with the scales SEPARATE the shock fizzles and the bank stays "
                    "stable. Cross-scale contagion is not reachable by simulating the scales independently.")
    return out


# ============================ B. real two-scale: justices -> the Court ============================
def _load_scdb():
    cases = defaultdict(list)
    with open(SCDB, encoding="latin1") as f:
        for row in csv.DictReader(f):
            try:
                term = int(row["term"]); d = int(row["direction"])
            except (ValueError, KeyError):
                continue
            if d not in (1, 2) or not row.get("justiceName"):
                continue
            cases[row["caseId"]].append({"term": term, "j": row["justiceName"], "lib": 1 if d == 2 else 0})
    return cases


def _committee_vote(ideologies):
    agents = [PersonaAgent(f"j{i}", {"ideo": io}, position=io, influence=1.0, openness=0.35, conviction=0.45)
              for i, io in enumerate(ideologies)]
    out = AgentSociety(homophily=0.6, consensus_pull=0.5, rounds=5).simulate(None, agents, lambda a, p: a.variables["ideo"])
    return out["vote_share"]


def _scotus_two_scale(train_max=2009):
    cases = _load_scdb()
    # static (train-era) ideology per justice, and the running as-of estimate
    train = defaultdict(lambda: [0, 0])
    for votes in cases.values():
        for v in votes:
            if v["term"] <= train_max:
                train[v["j"]][0] += v["lib"]; train[v["j"]][1] += 1
    def static_ideo(j):
        return (train[j][0] + 1) / (train[j][1] + 2) if train[j][1] else 0.5

    # build the World: a justice entity per justice (drifting ideology), advanced term by term
    world = World()
    running = defaultdict(lambda: [0, 0])
    for j in train:
        world.add(Entity(j, "individual", {"lib": train[j][0], "n": train[j][1]},
                         step_fn=lambda s, inp, dt, rng: {"lib": s["lib"] + inp.get("term_lib", 0),
                                                          "n": s["n"] + inp.get("term_n", 0)}))
        running[j] = [train[j][0], train[j][1]]

    # order test cases by term; query as-of BEFORE folding that term's votes in
    test_cases = [(cid, votes) for cid, votes in cases.items()
                  if all(v["term"] > train_max for v in votes) and 5 <= len(votes) <= 9]
    by_term = defaultdict(list)
    for cid, votes in test_cases:
        by_term[votes[0]["term"]].append((cid, votes))

    def score(ideo_of):
        dir_hit, margins, n = 0, [], 0
        for cid, votes in test_cases:
            k = len(votes); true_lib = sum(v["lib"] for v in votes)
            true_dir = int(true_lib > k / 2); true_margin = max(true_lib, k - true_lib) / k
            vs = _committee_vote([ideo_of(v["j"], votes[0]["term"]) for v in votes])
            dir_hit += int((vs > 0.5) == true_dir)
            margins.append(abs(max(vs, 1 - vs) - true_margin)); n += 1
        return {"direction_acc": round(dir_hit / n, 4), "margin_mae": round(sum(m for m in margins) / n, 4),
                "n": n}

    # SEPARATE: frozen train ideology (no individual dynamics)
    sep = score(lambda j, term: static_ideo(j))
    # COUPLED: as-of running ideology (advance the justice entities through terms via the World)
    asof_cache = {}
    for term in sorted(by_term):
        # snapshot as-of ideology (from strictly-prior terms) for every justice
        for j in world.entities:
            st = world.entities[j].state
            asof_cache[(j, term)] = (st["lib"] + 1) / (st["n"] + 2) if st["n"] else static_ideo(j)
        # fold in THIS term's observed votes (advance the individual scale one tick)
        term_tally = defaultdict(lambda: [0, 0])
        for cid, votes in cases.items():
            for v in votes:
                if v["term"] == term:
                    term_tally[v["j"]][0] += v["lib"]; term_tally[v["j"]][1] += 1
        ext = {j: {"term_lib": term_tally[j][0], "term_n": term_tally[j][1]} for j in world.entities}
        world.advance(1.0, external=ext)
    coup = score(lambda j, term: asof_cache.get((j, term), static_ideo(j)))
    return {"separate_static_justices": sep, "coupled_asof_drift": coup,
            "margin_improvement": round(sep["margin_mae"] - coup["margin_mae"], 4),
            "direction_improvement": round(coup["direction_acc"] - sep["direction_acc"], 4),
            "verdict": ("coupling the individual drift up to the institution HELPS"
                        if coup["margin_mae"] < sep["margin_mae"] - 0.002 else
                        "coupling ties separate here (do not scale up on this case)")}


def run():
    A = _bank_run()
    B = _scotus_two_scale()
    out = {"A_cross_scale_feedback": A, "B_real_two_scale_scored": B}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-070  persistent World substrate — coupled two-scale world, scored coupled-vs-separate")
    print("  A. CROSS-SCALE FEEDBACK (bank run: rumor->depositors->bank->rumor):")
    print(f"     COUPLED world  -> bank {A['coupled_world']['final_state']}  "
          f"(distress traj {A['coupled_world']['bank_distress_trajectory'][:6]}...)")
    print(f"     SEPARATE scales-> bank {A['separate_scales']['final_state']}  "
          f"(distress traj {A['separate_scales']['bank_distress_trajectory'][:6]}...)")
    print("     -> identical shock; coupling produces contagion-to-failure that separate models cannot.")
    print("  B. REAL TWO-SCALE (justices -> the Court, SCDB, leakage-free):")
    s, c = B["separate_static_justices"], B["coupled_asof_drift"]
    print(f"     SEPARATE (static justices): margin MAE {s['margin_mae']}  direction acc {s['direction_acc']}")
    print(f"     COUPLED  (as-of drift up) : margin MAE {c['margin_mae']}  direction acc {c['direction_acc']}")
    print(f"     -> margin {B['margin_improvement']:+}, direction {B['direction_improvement']:+}  |  {B['verdict']}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
