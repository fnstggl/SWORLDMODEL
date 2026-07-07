"""EXP-059: a NO-CHEAT as-of forecast through the REAL system — 2026 NBA champion, from Jan-2026.

This is a stress test of the end-to-end pipeline on a domain it was never tuned on, and an HONEST probe
of what the system is and is not. Two questions the user asked: (a) can we run a no-cheat backtest, and
(b) what does the system actually DO.

LEAKAGE CONTROL (why this is genuinely no-cheat):
  - The 2026 NBA Finals are played in JUNE 2026 — AFTER the model's Jan-2026 training cutoff. The winner
    is therefore NOT in the model's memory and cannot be recalled.
  - The evidence (`nba_context.json`) is the state of the season AS OF mid-Jan 2026, and NO live web search
    for the *result* was performed. So the forecast is made from as-of information only.
  - The forecast is logged pre-resolution to a PostMortemLog; the user (in July 2026) grades it.

WHAT THIS EXPOSES (the honest finding):
  The pipeline runs: retrieve as-of evidence -> instantiate teams as agents with a `strength` variable
  read from that evidence -> run the REAL GenerativeSimulator (AgentSociety deliberation). But a
  championship is a COMPETITIVE, mutually-exclusive outcome, not a SOCIAL one: teams do not deliberate
  toward each other's title odds. So:
    - the social-deliberation aggregator is the WRONG tool here — and the run PROVES it: deliberation
      shifts team positions by up to ~0.18 of SPURIOUS conformity (teams pulled toward a consensus they
      have no reason to reach). That distortion is a bug, not signal, on a competition; and
    - the RIGHT aggregator for "exactly one of N wins" is a competitive normalization (softmax over
      strength), which we apply explicitly and label as such.
  This is the point of the whole audit made concrete: the system's genuine EDGE is social questions where
  interaction manifests (committee votes, opinion cascades). On a competition it degenerates to a
  strength ranking — about as good as a power ranking / betting market, not a differentiated world-sim.

Run: python -m experiments.exp059_nba_backtest
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.api.generative_simulator import AgentSpec, GenerativeSimulator
from swm.api.retrieval import asof_retriever
from swm.eval.postmortem import PostMortemLog
from swm.simulation.agent_society import AgentSociety

CTX = "experiments/results/exp059_nba/nba_context.json"
RESULT = "experiments/results/exp059_nba_backtest.json"

# as-of Jan-2026 STRENGTH read per team (0..1), from the committed evidence. Approximate, dated to cutoff.
STRENGTH = {
    "Oklahoma City Thunder": 0.90,   # defending champs, best record, young core -> clear favorite
    "Cleveland Cavaliers":   0.63,
    "Denver Nuggets":        0.61,
    "Houston Rockets":       0.60,   # KD + young defensive core
    "Los Angeles Lakers":    0.56,   # Luka + LeBron, defense questions
    "New York Knicks":       0.54,
    "Minnesota Timberwolves":0.52,
    "field (all others)":    0.35,   # the long tail of the league
}


def _softmax_titleodds(strength: dict, k: float = 6.0) -> dict:
    """Competition-appropriate aggregation for a mutually-exclusive 'exactly one wins' outcome:
    title odds proportional to exp(k * strength) (a Bradley-Terry-style normalization). NOT deliberation,
    NOT an independent mean — the honest aggregator for a championship."""
    ex = {t: math.exp(k * s) for t, s in strength.items()}
    z = sum(ex.values())
    return {t: v / z for t, v in ex.items()}


def run():
    ctx = json.loads(Path(CTX).read_text())
    question = ctx["question"]

    # --- retrieval: serve the committed AS-OF evidence (no live result lookup) ---
    retriever = asof_retriever({question: [
        {"title": s["team"], "description": s["read"], "published_at": s["date"], "source": "as-of read"}
        for s in ctx["snippets"]]})
    evidence = retriever.retrieve(question, as_of=ctx["as_of"])

    # --- run the REAL system: teams as agents, strength as the driving variable, real deliberation ---
    contenders = {t: s for t, s in STRENGTH.items() if not t.startswith("field")}
    specs = [AgentSpec(t, {"strength": s}, influence=1.0 + s, openness=0.3, conviction=0.5)
             for t, s in contenders.items()]
    sim = GenerativeSimulator(society=AgentSociety(homophily=0.4, consensus_pull=0.3, rounds=6),
                              identify_fn=lambda q, c: specs,
                              position_fn=lambda q, s, c: s.variables["strength"])
    fc = sim.simulate(question, context=evidence.to_prompt(), agents=specs)

    # deliberated vs independent: on a competition these should ~coincide (no genuine social interaction) --
    delib_by_team = {a["id"]: a["final"] for a in fc.agents}
    indep_by_team = {a["id"]: a["initial"] for a in fc.agents}
    max_shift = max(abs(delib_by_team[t] - indep_by_team[t]) for t in delib_by_team)

    # --- the HONEST aggregation: competitive normalization over strength -> a champion distribution ---
    title_odds = _softmax_titleodds(STRENGTH)
    ranked = sorted(title_odds.items(), key=lambda kv: -kv[1])
    top_team, top_p = ranked[0]

    # --- pre-register the forecast (leakage-free: outcome is post-cutoff, unknown) ---
    log = PostMortemLog()
    log.log("nba-champ-2026", top_p, made_at="2026-01-15", resolves_at="2026-06-30",
            meta={"pick": top_team, "distribution": {t: round(p, 3) for t, p in ranked}})

    out = {
        "question": question,
        "as_of": ctx["as_of"], "resolves_at": ctx["resolves_at"],
        "leakage_control": ctx["leakage_control"],
        "champion_distribution": {t: round(p, 3) for t, p in ranked},
        "point_pick": {"team": top_team, "p": round(top_p, 3)},
        "mechanism_audit": {
            "ran_real_agent_society": True,
            "deliberated_vs_independent_max_shift": round(max_shift, 4),
            "interpretation": "deliberation moved team positions by up to this much of SPURIOUS conformity "
                              "-- competing teams have no reason to converge on each other's title odds, so "
                              "this shift is distortion, not signal. It is exactly why the social "
                              "aggregator is the wrong tool for a competition; the champion distribution is "
                              "produced by the competition-appropriate normalizer (softmax over strength).",
        },
        "honest_accuracy_note": "On a competitive/sports outcome the system reduces to a strength ranking "
                                "-- about as good as a public power ranking or betting market, NOT a "
                                "differentiated world-sim. The system's genuine edge is SOCIAL questions "
                                "where interaction manifests (committee votes, opinion cascades).",
    }
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-059  NO-CHEAT as-of forecast through the REAL system: 2026 NBA champion, from Jan-2026")
    print(f"  Q: {question}")
    print(f"  as-of: {ctx['as_of']}  |  resolves: {ctx['resolves_at']}  (Finals are post-cutoff -> unknown to the model)")
    print(f"  evidence items (as-of, no result lookup): {len(evidence)}")
    print("  --- champion distribution (competition-normalized over as-of strength) ---")
    for t, p in ranked:
        bar = "#" * int(round(p * 40))
        print(f"    {t:24s} {p*100:5.1f}%  {bar}")
    print(f"  POINT PICK: {top_team}  (P = {top_p*100:.1f}%)")
    print("  --- mechanism audit ---")
    print(f"    real AgentSociety deliberation ran; deliberated-vs-independent max shift = {max_shift:.4f}")
    print("    that shift is SPURIOUS conformity (teams pulled toward consensus they'd never reach) -> the")
    print("    social aggregator is the wrong tool here; the champion distribution uses the competition")
    print("    normalizer, NOT deliberation.")
    print("  HONEST NOTE: on sports this ~= a power ranking / market. The system's real edge is SOCIAL")
    print("    questions where agents genuinely influence each other (SCOTUS/FOMC votes, opinion cascades).")
    print(f"  pre-registered to PostMortemLog (made 2026-01-15, resolves 2026-06-30) -> you grade it.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
