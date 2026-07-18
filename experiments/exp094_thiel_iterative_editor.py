"""EXP-094 — full-draft generator vs ITERATIVE EDITOR on the Thiel email, same evaluator, honest verdict.

Two search methods over the same wording decision (the full-draft path from exp092 is unchanged;
the iterative editor is the new, experimental second path):

  A. FULL-DRAFT   — strategy-diverse complete drafts -> gates -> persona-ensemble ranking
                    (optimize_cold_outreach; the v3 approach).
  B. ITER-EDITOR  — the exacting-human-editor loop (iterative_editor.py): strongest seed ->
                    whole-message diagnosis -> per-location materially-different alternatives
                    (keep/rewrite/shorten/reframe/merge/DELETE/insert) -> independent FULL-MESSAGE
                    comparative judge -> whole-email rescore guard -> endgame sweeps (deletion sweep,
                    reorder, add-beat, shorten, replace-ask, new opening, reframe) -> beam +
                    informed rewrite + crossover. Complete machine-readable edit trace.

FINAL COMPARISON under the SAME evaluator for every candidate (same persona hypotheses — loaded
from the exp092 run for comparability — same outcome utilities, same fact guards, same labels):
best iterative-editor candidates, the full-draft winner from THIS run, the recorded v3 winner, the
plain human baseline, and the previously rejected debate-bait message. Reported as
best-supported-among-tested with hypothesis fragility and within-noise honesty; no claim of a
theoretical or unique optimum.

Run:  PYTHONPATH=. python experiments/exp094_thiel_iterative_editor.py            (live)
      PYTHONPATH=. python experiments/exp094_thiel_iterative_editor.py --offline  (mechanics smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.iterative_editor import IterativeEditor
from swm.decision.llm_moves import SenderBrief, llm_draft_proposer
from swm.decision.message_pipeline import optimize_cold_outreach, recipient_from_world
from swm.decision.outreach_contract import plain_baseline_draft, validate
from swm.decision.persona_response import (DEFAULT_OUTCOME_UTILITIES, PersonaDossier,
                                           ensemble_evaluate, fragility_report,
                                           specialize_hypotheses)
from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_v4")
V3_ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_v3")

AURELIUS_SENDER = SenderBrief(
    sender="Beckett",
    thesis="AI infrastructure has a planning problem disguised as a power problem: schedulers "
           "optimize the next placement, but nothing chooses the fleet's best trajectory over time",
    ask="permission to send the one-page technical memo",
    facts=[
        "17 years old, starting Princeton in the fall",
        "building Aurelius (runaurelius.com): constraint-aware orchestration for AI infrastructure",
        "a predictive world model forecasts power constraints, simulates candidate scheduling "
        "decisions, and ranks them by economic outcome before execution",
        "+724% average SLA-safe goodput per dollar vs the production scheduler, in simulated replay "
        "of ~1.5M requests of public production traces",
        "-84% GPU-hours in the same replay",
        "results are simulated replay of public traces, not a production deployment",
        "working with a small batch of infrastructure operators in read-only shadow mode",
    ])

_EVIDENCE = [
    {"title": "The Thiel Fellowship pays people to skip college",
     "snippet": "backs young founders who drop out; discovered founders via cold email; took a "
                "meeting with a teenager who had an unusual, contrarian thesis"},
    {"title": "Peter Thiel, the contrarian",
     "snippet": "provocative, heterodox, iconoclast; a skeptic who challenges consensus; famously "
                "skeptical of elite university prestige and the higher-ed status game"},
    {"title": "How to actually reach Peter Thiel",
     "snippet": "hard to reach, screens heavily, rarely responds to cold outreach unless the pitch "
                "is genuinely contrarian and specific"},
    {"title": "Founders Fund and definite optimism",
     "snippet": "billionaire investor and chairman; looks for definite, specific plans and secrets; "
                "dislikes generic status-seeking outreach"},
    {"title": "thielfellowship.org (live fetch 2026-07-17)",
     "snippet": "gives $250,000 and mentorship to young people to skip or stop out of college"},
]

# recorded reference candidates (verbatim from prior runs; also read from artifacts when present)
V3_WINNER_FALLBACK = (
    "I'm Beckett, 17, building Aurelius (runaurelius.com) — constraint-aware orchestration for AI "
    "infrastructure. The thesis: AI infrastructure has a planning problem disguised as a power "
    "problem, where schedulers optimize the next placement but nothing chooses the fleet's best "
    "trajectory over time. In simulated replay of ~1.5M requests from public production traces, a "
    "predictive world model that forecasts power constraints and ranks candidate decisions by "
    "economic outcome achieved -84% GPU-hours versus the production scheduler. May I send you the "
    "one-page technical memo? — Beckett")

DEBATE_BAIT = (
    "Peter, treating data center power as a static budget ignores that dynamic scheduling against "
    "grid forecasts cut GPU-hours by 84% in our simulated replay of public production traces. "
    "Which assumption in that claim is wrong?")


def _load_v3():
    winner = V3_WINNER_FALLBACK
    hyps = None
    try:
        r = json.load(open(os.path.join(V3_ART, "result.json")))
        t = (r.get("stage_b", {}).get("ranked") or [])
        for row in t:
            if row.get("arm") == r.get("stage_b_winner"):
                winner = row.get("text") or winner
                break
    except Exception:  # noqa: BLE001
        pass
    try:
        hyps = json.load(open(os.path.join(V3_ART, "hypotheses.json")))["recipient_hypotheses"]
    except Exception:  # noqa: BLE001
        hyps = None
    return winner, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()

    world = World(store=EventStore(":memory:"),
                  resolver=PublicFigureResolver(search_fn=lambda q: _EVIDENCE))
    chat = None if args.offline else default_chat_fn(max_tokens=800, temperature=0.2)
    recipient = recipient_from_world(world, "peter_thiel", name="Peter Thiel",
                                     domain="AI infrastructure", ask="cold outreach")
    dossier = PersonaDossier.for_public_figure(world, "peter_thiel", name="Peter Thiel",
                                               role="investor, Founders Fund")
    if not dossier.evidence:
        dossier.evidence = [(e["title"], e["snippet"]) for e in _EVIDENCE]
    v3_winner, saved_hyps = _load_v3()
    hyps = saved_hyps or (specialize_hypotheses(chat, dossier) if chat
                          else specialize_hypotheses(None, dossier))
    notes = "Peter Thiel; screens heavily; treat all cold inbound as barely read"

    # ---------------- ARM A: the (unchanged) full-draft generator --------------------------------
    slate = optimize_cold_outreach(
        recipient, sender_brief=AURELIUS_SENDER, chat_fn=chat, recipient_notes=notes,
        k_drafts=6, n_mc=200, dossier=dossier, hypotheses=hyps, persona_draws=args.draws,
        persona_top_k=3, method="slate")   # this experiment compares the SLATE path vs the editor
    fulldraft_winner = slate.candidates.get(slate.winner, {}).get("text", "")

    # ---------------- ARM B: the iterative editor ------------------------------------------------
    seeds = [{"label": "plain_baseline", "text": plain_baseline_draft(AURELIUS_SENDER, "Peter Thiel")}]
    if chat is not None:
        proposer = llm_draft_proposer(chat, recipient_notes=notes, sender=AURELIUS_SENDER)
        # ask for strategy-diverse seeds (the proposer prompt already enforces the contract)
        for i, d in enumerate(proposer({"identity_legibility": 1.0, "claim_believability": 1.0,
                                        "next_step_clarity": 1.0, "adversarial_framing": 0.0,
                                        "cognitive_effort": 0.0, "convenience_selling": 0.0,
                                        "credential_signaling": 0.0}, k=5)):
            seeds.append({"label": f"seed_{i}", "text": d})
    editor = IterativeEditor(
        chat, sender_brief=AURELIUS_SENDER, recipient_notes=notes,
        dossier_text=dossier.render(), recipient_vars=recipient.vars,
        base_mean=recipient.base_mean, max_passes=2, beam_size=3, max_llm_calls=110,
        trace_path=os.path.join(ART, "edit_trace.jsonl"))
    ed_out = editor.run(seeds)
    editor_best = [s for s in ed_out["beam"][:2]]

    # ---------------- FINAL COMPARISON: same persona evaluator for every candidate ---------------
    candidates = {"iter_editor_best": editor_best[0].text if editor_best else "",
                  "fulldraft_winner": fulldraft_winner,
                  "v3_winner_recorded": v3_winner,
                  "plain_baseline": plain_baseline_draft(AURELIUS_SENDER, "Peter Thiel"),
                  "debate_bait_rejected": DEBATE_BAIT}
    if len(editor_best) > 1 and editor_best[1].text != editor_best[0].text:
        candidates["iter_editor_second"] = editor_best[1].text
    results, contract_verdicts = {}, {}
    for name, text in candidates.items():
        if not text:
            continue
        contract_verdicts[name] = validate(text, AURELIUS_SENDER).as_dict()
        if chat is not None:
            results[name] = ensemble_evaluate(chat, dossier, hyps, text,
                                              draws_per_hypothesis=args.draws)
    frag = fragility_report(results, DEFAULT_OUTCOME_UTILITIES) if results else {}

    editor_vs_fulldraft = None
    if results.get("iter_editor_best") and results.get("fulldraft_winner"):
        eu_e = results["iter_editor_best"].expected_utility(DEFAULT_OUTCOME_UTILITIES)
        eu_f = results["fulldraft_winner"].expected_utility(DEFAULT_OUTCOME_UTILITIES)
        n = results["iter_editor_best"].n_draws or 1
        noise = 1.5 / (n ** 0.5)
        editor_vs_fulldraft = {
            "editor_eu": round(eu_e, 4), "fulldraft_eu": round(eu_f, 4),
            "delta": round(eu_e - eu_f, 4), "noise_scale": round(noise, 4),
            "material": abs(eu_e - eu_f) > noise,
            "verdict": ("iterative editor materially stronger" if eu_e - eu_f > noise else
                        "full-draft materially stronger" if eu_f - eu_e > noise else
                        "indistinguishable at this draw count")}

    payload = {
        "experiment": "exp094_thiel_iterative_editor", "mode": "live" if chat else "offline",
        "hypotheses_source": "thiel_v3 saved" if saved_hyps else "specialized this run",
        "editor": {"beam": [{"label": s.label, "internal_score": s.value, "text": s.text}
                            for s in ed_out["beam"]],
                   "llm_calls": ed_out["llm_calls"],
                   "n_trace_steps": len(ed_out["trace"]),
                   "n_rejected_local_improvements": sum(
                       1 for r in ed_out["trace"]
                       if str(r.get("reject_reason", "")).startswith("local improvement"))},
        "fulldraft": {"winner": slate.winner, "text": fulldraft_winner,
                      "within_noise": slate.within_noise, "fragility": slate.fragility},
        "final_comparison": {
            "persona": {k: v.summary(DEFAULT_OUTCOME_UTILITIES) for k, v in results.items()},
            "fragility": frag, "contracts": contract_verdicts,
            "editor_vs_fulldraft": editor_vs_fulldraft},
        "label": "model_based_judgment — same UNCALIBRATED persona evaluator for every candidate; "
                 "'best-supported among tested', never a theoretical optimum",
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ART, "result.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)

    strongest_rejected = [r for r in ed_out["trace"]
                          if str(r.get("reject_reason", "")).startswith("local improvement")][:5]
    with open(os.path.join(ART, "rejected_improvements.json"), "w") as f:
        json.dump(strongest_rejected, f, indent=1, default=str)

    print("=" * 88)
    print("EXP-094  full-draft vs ITERATIVE EDITOR (same persona evaluator)")
    print("=" * 88)
    print("\n[EDITOR] best (internal score %.3f, %d LLM calls, %d trace steps):"
          % (editor_best[0].value if editor_best else -1, ed_out["llm_calls"],
             len(ed_out["trace"])))
    if editor_best:
        print("  " + editor_best[0].text.replace(". ", ".\n  "))
    print("\n[FULL-DRAFT] winner (%s):" % slate.winner)
    print("  " + fulldraft_winner.replace(". ", ".\n  "))
    if results:
        print("\n[FINAL — persona EU, same evaluator]")
        for k, v in sorted(results.items(),
                           key=lambda kv: -kv[1].expected_utility(DEFAULT_OUTCOME_UTILITIES)):
            print("  %-24s EU=%.3f  byH=%s" % (
                k, v.expected_utility(DEFAULT_OUTCOME_UTILITIES),
                {h: round(x, 2) for h, x in v.by_hypothesis_utility(DEFAULT_OUTCOME_UTILITIES).items()}))
        print("\n  fragility:", {k: frag.get(k) for k in ("winner", "fragile",
                                                          "within_noise_of_winner")})
        if editor_vs_fulldraft:
            print("\n[VERDICT]", editor_vs_fulldraft["verdict"],
                  f"(editor {editor_vs_fulldraft['editor_eu']} vs "
                  f"full-draft {editor_vs_fulldraft['fulldraft_eu']}, "
                  f"noise ±{editor_vs_fulldraft['noise_scale']})")
    print("\nartifacts ->", os.path.relpath(ART), " wall=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
