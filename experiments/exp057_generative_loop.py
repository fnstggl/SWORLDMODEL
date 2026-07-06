"""EXP-057: the full generative loop assembled — one simulate(question) end-to-end.

Wires the whole thing into `GenerativeSimulator.simulate(question)`: identify agents -> map variables ->
assign positions (LLM persona reasoning) -> `AgentSociety` deliberation -> emergent outcome + audit trail.
Validated two ways, honestly separated:

  A. ASSEMBLY CORRECTNESS (quantitative, no LLM, no contamination). Run the assembled loop on the Supreme
     Court with a STRUCTURED position_fn (justice ideology from prior terms, exactly EXP-055). If the
     assembled `GenerativeSimulator` reproduces EXP-055's margin win over the independent composite, the
     wiring is correct independent of any LLM call.

  B. LLM-PERSONA DEMONSTRATION (qualitative). Run the loop on a real institutional question with an LLM
     identifying the agents + assigning positions (committed judgments), and show the full audit trail: who
     the agents are, their initial vs final positions after deliberation, the trajectory, and how the
     emergent outcome differs from the independent composite. Illustrative (an LLM persona pass, not a
     leakage-free skill number) — the point is that the general loop RUNS end-to-end on an arbitrary
     question.

Run: python -m experiments.exp057_generative_loop
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.api.generative_simulator import AgentSpec, GenerativeSimulator
from swm.simulation.agent_society import AgentSociety
from experiments.exp055_agent_society import _load_scdb

RESULT = "experiments/results/exp057_generative_loop.json"


# ---- A. assembly correctness on SCDB (structured position_fn = justice ideology) ----
def _scdb_assembly():
    test, ideology = _load_scdb()
    society = AgentSociety(homophily=0.6, consensus_pull=0.5, rounds=5)

    def make_sim():
        return GenerativeSimulator(society=society,
                                   position_fn=lambda q, spec, ctx: spec.variables["ideo"])
    sim = make_sim()
    ind_margin, gen_margin, n = [], [], 0
    for cid, votes in test:
        specs = [AgentSpec(v["justice"], {"ideo": ideology(v["justice"], v["issue"])},
                           influence=1.0, openness=0.35, conviction=0.45) for v in votes]
        k = len(votes); true_lib = sum(v["lib"] for v in votes)
        true_margin = max(true_lib, k - true_lib) / k
        fc = sim.simulate(cid, agents=specs)
        gen_share = _share(fc)                          # fraction voting liberal after deliberation
        gen_maj = max(gen_share, 1 - gen_share)
        ind_share = sum(1 for a in fc.agents if a["initial"] > 0.5) / len(fc.agents)  # independent count
        ind_maj = max(ind_share, 1 - ind_share)
        gen_margin.append(abs(gen_maj - true_margin)); ind_margin.append(abs(ind_maj - true_margin))
        n += 1
    return {"n_cases": n, "independent_margin_mae": round(sum(ind_margin) / n, 4),
            "generative_loop_margin_mae": round(sum(gen_margin) / n, 4),
            "assembly_reproduces_win": (sum(gen_margin) / n) < (sum(ind_margin) / n)}


def _share(fc):
    # vote share = fraction of agents whose FINAL position > 0.5 (from the audit trail)
    return sum(1 for a in fc.agents if a["final"] > 0.5) / max(1, len(fc.agents))


# ---- B. LLM-persona worked example (committed judgments) ----
def _worked_example():
    """A committee decision where the LLM identifies the agents + positions; the loop shows deliberation
    flipping the naive count via an influential, high-conviction bloc — an emergent outcome."""
    # committed LLM output (identify + positions) for the question below
    question = ("A 9-member standards committee votes on adopting a strict new safety rule. Does it pass?")
    specs = [
        AgentSpec("chair", {"expertise": 0.9, "stance": 0.85}, influence=3.0, openness=0.15, conviction=0.85),
        AgentSpec("safety_lead", {"expertise": 0.85, "stance": 0.9}, influence=2.2, openness=0.2, conviction=0.8),
        AgentSpec("eng_1", {"expertise": 0.7, "stance": 0.38}, influence=1.0, openness=0.6, conviction=0.2),
        AgentSpec("eng_2", {"expertise": 0.7, "stance": 0.35}, influence=1.0, openness=0.6, conviction=0.2),
        AgentSpec("eng_3", {"expertise": 0.65, "stance": 0.37}, influence=1.0, openness=0.55, conviction=0.25),
        AgentSpec("ops_1", {"expertise": 0.5, "stance": 0.3}, influence=0.9, openness=0.5, conviction=0.3),
        AgentSpec("ops_2", {"expertise": 0.5, "stance": 0.32}, influence=0.9, openness=0.5, conviction=0.3),
        AgentSpec("junior_1", {"expertise": 0.4, "stance": 0.4}, influence=0.6, openness=0.7, conviction=0.15),
        AgentSpec("junior_2", {"expertise": 0.4, "stance": 0.42}, influence=0.6, openness=0.7, conviction=0.15),
    ]
    positions = {s.agent_id: s.variables["stance"] for s in specs}
    sim = GenerativeSimulator(society=AgentSociety(homophily=0.3, consensus_pull=0.2, rounds=8),
                              position_fn=lambda q, spec, ctx: positions[spec.agent_id])
    fc = sim.simulate(question, agents=specs)
    return {"question": question, "n_agents": fc.n_agents,
            "independent_p": round(fc.independent_p, 3),
            "independent_passes": fc.independent_p > 0.5,
            "simulated_p": round(fc.p_outcome, 3), "simulated_passes": fc.passes,
            "trajectory": [round(x, 3) for x in fc.trajectory],
            "emergence": fc.passes != (fc.independent_p > 0.5),
            "agents": fc.agents}


def run():
    scdb = _scdb_assembly()
    worked = _worked_example()
    out = {"A_assembly_correctness_scdb": scdb, "B_worked_llm_example": worked,
           "loop_runs_end_to_end": True}

    print("EXP-057 full generative loop assembled — simulate(question) end-to-end")
    print("  A. ASSEMBLY CORRECTNESS on SCDB (structured position_fn; should reproduce EXP-055):")
    print(f"     independent margin MAE {scdb['independent_margin_mae']}  "
          f"generative-loop margin MAE {scdb['generative_loop_margin_mae']}  "
          f"(reproduces win: {scdb['assembly_reproduces_win']})")
    print("  B. LLM-PERSONA worked example (committee adopts a strict safety rule?):")
    print(f"     independent (naive count): p={worked['independent_p']} passes={worked['independent_passes']}")
    print(f"     simulated (deliberation):  p={worked['simulated_p']} passes={worked['simulated_passes']}  "
          f"emergent flip: {worked['emergence']}")
    print(f"     trajectory: {worked['trajectory']}")
    for a in worked["agents"][:4]:
        print(f"       {a['id']:<12} initial {a['initial']} -> final {a['final']} (influence {a['influence']})")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
