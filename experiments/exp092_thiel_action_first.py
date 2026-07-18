"""EXP-092 — the Thiel outreach decision, ACTION-FIRST, with a qualitative persona ensemble.

What changed vs exp090 (each item answers a diagnosed failure of that run):

  1. ACTION SPACE BEFORE WORDING. The decision starts one level up: cold email vs cold text vs
     permission-ask vs full memo vs warm introduction via an operator vs having the operator
     forward the memo vs routing through a Founders Fund partner vs waiting for pilot evidence vs
     not contacting yet. "The best action may not be a better sentence."
  2. QUALITATIVE PERSONA ENSEMBLE AS THE BEHAVIORAL ENGINE. Every action is an ARRIVAL CONTEXT for
     the same first-person simulation: "You are Peter Thiel — [qualitative dossier: beliefs,
     incentives, dispositions, evidence quotes — no invented numbers]. [inbox-reality hypothesis].
     This arrives: […]. What do you actually do?" Outcomes are drawn from a categorical menu and
     COUNTED across draws × hypotheses — never asked for as probabilities. This replaces the
     circular numeric-trait scoring loop.
  3. COMPETING INBOX-REALITY HYPOTHESES with honest priors (assistant screens 0.35, intros-only
     0.25, reads-own-bursts 0.15, evidence-first 0.15, ignores-all-cold 0.10). A recommendation
     that wins under only one hypothesis is flagged fragile, never trusted.
  4. VALENCED OUTCOME VECTOR — no_response / dismissive (a cost) / curious / requests_material /
     refers_to_other / meeting_offer — user-weighted; "any reply" is not the objective.
  5. WORDING OPTIMIZED ONLY INSIDE THE CHOSEN ACTION, through the corrected cold-outreach path
     (content contract, fact guard, strategy-diverse whole drafts, register + cold-read gates,
     persona-ensemble ranking, 'best-supported among tested' output).
  6. PHASE 13 is the uncertainty/governance layer, not the behavior: hypothesis-stratified matched
     particles draw outcomes from the MEASURED persona-count distributions per (action, hypothesis),
     with CRN pairing, robust/CVaR/fragility reporting and a prospective-ledger freeze. Provenance
     marks the behavioral source as persona_ensemble_counts (model-based judgment, uncalibrated).

Assumption registry (explicit, uncertain, reported): P(operator agrees to intro) = 0.55 ± 0.2;
P(operator agrees to forward) = 0.7 ± 0.15; FF-partner pass-through discount = 0.75; wait-for-pilot
delay discount = 0.85 with P(pilot confirms) = 0.6. These are priors about BECKETT'S side of each
path, separate from Peter's response behavior.

Run:  PYTHONPATH=. python experiments/exp092_thiel_action_first.py            (live)
      PYTHONPATH=. python experiments/exp092_thiel_action_first.py --offline  (structure smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.llm_moves import SenderBrief
from swm.decision.message_pipeline import optimize_cold_outreach, recipient_from_world
from swm.decision.outreach_contract import plain_baseline_draft
from swm.decision.persona_response import (DEFAULT_OUTCOME_UTILITIES, PersonaDossier,
                                           ensemble_evaluate, fragility_report,
                                           specialize_hypotheses)
from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_v3")

# sender facts verbatim from runaurelius.com (fetched 2026-07-17) + user-stated bio
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
    ],
)

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
     "snippet": "gives $250,000 and mentorship to young people to skip or stop out of college; "
                "'College can be good for learning about what's been done before, but it can also "
                "discourage you from doing something new'"},
]

# ---------------------------------------------------------------- Beckett-side path assumptions
PATH_ASSUMPTIONS = {
    "p_operator_agrees_intro": {"mean": 0.55, "sd": 0.20,
                                "basis": "operators in read-only shadow mode know the work; an "
                                         "intro request is a real but plausible favor"},
    "p_operator_agrees_forward": {"mean": 0.70, "sd": 0.15,
                                  "basis": "forwarding a memo is a smaller favor than a vouched intro"},
    "ff_partner_passthrough_discount": {"value": 0.75,
                                        "basis": "value realized only if the partner routes it "
                                                 "onward or engages themselves"},
    "wait_delay_discount": {"value": 0.85, "basis": "option/competition cost of ~8 weeks"},
    "p_pilot_confirms": {"mean": 0.6, "sd": 0.15,
                         "basis": "shadow-mode replay results transferring to a live operator pilot"},
}


def build_actions(probe_email: str, permission_ask: str, memo_line: str):
    """Each action = (id, recipient_kind, arrival_context, message, prerequisite, discount).
    prerequisite = (assumption_key or None); the same persona engine reads every arrival."""
    intro_note = ("One of the infrastructure operators you respect forwards this with a note: "
                  "\"I've had read-only access to this kid's scheduler for a month. The numbers "
                  "are real on our traces. Worth two minutes.\"")
    return [
        {"id": "cold_email_direct", "recipient": "thiel",
         "arrival": "A cold email from a stranger arrives in your inbox.",
         "message": probe_email, "prereq": None, "discount": 1.0},
        {"id": "cold_text_personal_number", "recipient": "thiel",
         "arrival": "A text message from an unknown number appears on your PERSONAL phone — a "
                    "number you have never given out publicly.",
         "message": permission_ask, "prereq": None, "discount": 1.0},
        {"id": "permission_ask_email", "recipient": "thiel",
         "arrival": "A cold email from a stranger arrives in your inbox.",
         "message": permission_ask, "prereq": None, "discount": 1.0},
        {"id": "full_memo_email", "recipient": "thiel",
         "arrival": "A cold email from a stranger arrives with a one-page technical memo attached.",
         "message": probe_email + " " + memo_line, "prereq": None, "discount": 1.0},
        {"id": "operator_intro_email", "recipient": "thiel",
         "arrival": intro_note, "message": probe_email,
         "prereq": "p_operator_agrees_intro", "discount": 1.0},
        {"id": "operator_forwards_memo", "recipient": "thiel",
         "arrival": "One of the infrastructure operators you respect forwards a one-page technical "
                    "memo: \"this beat our scheduler in replay; the summary is one page.\"",
         "message": memo_line, "prereq": "p_operator_agrees_forward", "discount": 1.0},
        {"id": "ff_partner_route", "recipient": "ff_partner",
         "arrival": "A cold email from a stranger arrives in your inbox.",
         "message": probe_email, "prereq": None,
         "discount": PATH_ASSUMPTIONS["ff_partner_passthrough_discount"]["value"]},
        {"id": "wait_for_pilot_then_email", "recipient": "thiel",
         "arrival": "A cold email from a stranger arrives. The attached memo cites a COMPLETED "
                    "operator pilot with named (redacted) infrastructure operators confirming the "
                    "replay results on live traffic.",
         "message": probe_email, "prereq": "p_pilot_confirms",
         "discount": PATH_ASSUMPTIONS["wait_delay_discount"]["value"]},
        {"id": "do_nothing", "recipient": None, "arrival": "", "message": "",
         "prereq": None, "discount": 1.0},
    ]


def action_expected_utility(action, ens, utilities) -> dict:
    """EU(action) = P(prereq) × discount × EU_persona + explicit assumption provenance."""
    if action["recipient"] is None:
        return {"eu": 0.0, "eu_by_hypothesis": {}, "prereq_p": 1.0, "note": "reference"}
    pre = 1.0
    if action["prereq"]:
        pre = PATH_ASSUMPTIONS[action["prereq"]]["mean"]
    eu_p = ens.expected_utility(utilities)
    return {"eu": round(pre * action["discount"] * eu_p, 4),
            "eu_persona_given_arrival": round(eu_p, 4),
            "eu_by_hypothesis": ens.by_hypothesis_utility(utilities),
            "prereq_p": pre, "discount": action["discount"],
            "outcome_dist": {k: round(v, 3) for k, v in ens.outcome_dist().items()}}


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

    # the QUALITATIVE dossier: evidence text + user context; never numeric traits
    dossier = PersonaDossier.for_public_figure(world, "peter_thiel", name="Peter Thiel",
                                               role="investor, Founders Fund")
    if not dossier.evidence:
        dossier.evidence = [(e["title"], e["snippet"]) for e in _EVIDENCE]
    ff_dossier = PersonaDossier.from_user_context(
        "a Founders Fund partner focused on AI infrastructure",
        "you evaluate deep-tech deals; you forward genuinely interesting infrastructure work to "
        "the relevant partner or Peter when it is credible and concrete; you ignore hype",
        role="venture partner")

    hyps = specialize_hypotheses(chat, dossier) if chat else specialize_hypotheses(None, dossier)
    with open(os.path.join(ART, "hypotheses.json"), "w") as f:
        json.dump({"recipient_hypotheses": hyps, "path_assumptions": PATH_ASSUMPTIONS}, f, indent=1)

    # --- probe messages for STAGE A (action comparison uses the same fixed, contract-valid probes)
    probe_email = plain_baseline_draft(AURELIUS_SENDER, "Peter Thiel")
    permission_ask = ("Peter, I'm Beckett, 17, building Aurelius, a constraint-aware orchestration "
                      "for AI infrastructure. In replays of ~1.5M requests of public production "
                      "traces it beat the production scheduler on SLA-safe goodput per dollar. "
                      "May I send you the one-page memo? Beckett")
    memo_line = ("The one-page memo: schedulers optimize the next placement; Aurelius simulates "
                 "fleet trajectories against forecast power constraints and picks by economics — "
                 "+724% SLA-safe goodput per dollar and -84% GPU-hours vs the production scheduler "
                 "in simulated replay of ~1.5M requests of public production traces.")

    actions = build_actions(probe_email, permission_ask, memo_line)
    utilities = DEFAULT_OUTCOME_UTILITIES

    # ---------------- STAGE A: action-level comparison through the persona ensemble ----------------
    stage_a = {}
    ens_by_action = {}
    for a in actions:
        if a["recipient"] is None:
            stage_a[a["id"]] = {"eu": 0.0, "note": "reference (not contacting yet keeps the option)"}
            continue
        dos = dossier if a["recipient"] == "thiel" else ff_dossier
        hy = hyps if a["recipient"] == "thiel" else specialize_hypotheses(None, ff_dossier)
        if chat:
            ens = ensemble_evaluate(chat, dos, hy, a["message"], arrival_context=a["arrival"],
                                    draws_per_hypothesis=args.draws)
        else:
            from swm.decision.persona_response import PersonaEnsembleResult
            ens = PersonaEnsembleResult(priors={h["id"]: h["prior"] for h in hy})
            for h in hy:
                ens.counts[h["id"]] = {"no_response": 1}
                ens.n_draws += 1
        ens_by_action[a["id"]] = ens
        stage_a[a["id"]] = action_expected_utility(a, ens, utilities)

    frag = fragility_report({k: v for k, v in ens_by_action.items()}, utilities)
    ranked_actions = sorted(((k, v.get("eu", 0.0)) for k, v in stage_a.items()),
                            key=lambda kv: -kv[1])
    with open(os.path.join(ART, "stage_a_actions.json"), "w") as f:
        json.dump({"per_action": stage_a, "fragility": frag,
                   "ranked": ranked_actions,
                   "label": "model_based_judgment — persona-ensemble counts under competing "
                            "inbox hypotheses × explicit path assumptions; UNCALIBRATED"}, f, indent=1)

    # ---------------- STAGE B: wording INSIDE the best direct-send action --------------------------
    best_direct = next((k for k, _ in ranked_actions
                        if k in ("permission_ask_email", "cold_email_direct", "full_memo_email")),
                       "permission_ask_email")
    arrival = next(a["arrival"] for a in actions if a["id"] == best_direct)
    slate = optimize_cold_outreach(
        recipient, sender_brief=AURELIUS_SENDER, chat_fn=chat,
        recipient_notes="Peter Thiel — screens heavily; treat all cold inbound as barely read",
        k_drafts=8, n_mc=300, dossier=dossier, hypotheses=hyps, persona_draws=args.draws,
        persona_top_k=4, arrival_context=arrival,
        method="slate")   # this experiment demonstrates the v3 slate path; exp095 is reply-first

    payload = {"experiment": "exp092_thiel_action_first",
               "mode": "live" if chat else "offline",
               "stage_a": {"ranked_actions": ranked_actions, "fragility": frag},
               "stage_b_winner": slate.winner,
               "stage_b": slate.summary(),
               "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(ART, "result.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)

    # ---------------- ledger freeze (prospective; no fabricated outcomes) --------------------------
    from swm.world_model_v2.phase13.ledger import DecisionLedger
    led = DecisionLedger(os.path.join(ART, "ledger.jsonl"))
    led._append({"kind": "frozen_decision", "decision_id": "thiel_outreach_v3",
                 "stage_a_recommendation": ranked_actions[0][0] if ranked_actions else None,
                 "stage_a_ranked": ranked_actions,
                 "stage_b_winner": slate.winner,
                 "stage_b_winner_text": slate.candidates.get(slate.winner, {}).get("text"),
                 "hypotheses": [h["id"] for h in hyps],
                 "label": "model_based_judgment (uncalibrated persona ensemble)",
                 "chosen_real_action": None, "realized_outcome": None})

    print("=" * 88)
    print("EXP-092  Thiel outreach — ACTION-FIRST, qualitative persona ensemble")
    print("=" * 88)
    print("\n[STAGE A] action ranking (EU = P(prereq) × discount × persona EU):")
    for k, v in ranked_actions:
        extra = stage_a[k]
        byh = extra.get("eu_by_hypothesis", {})
        print(f"  {k:28} EU={v:+.3f}  byH={ {h: round(x, 2) for h, x in byh.items()} }")
    print("  fragility:", {kk: frag.get(kk) for kk in ("winner", "fragile",
                                                       "winner_wins_under_hypotheses")})
    print(f"\n[STAGE B] wording inside {best_direct} — winner: {slate.winner}")
    w = slate.candidates.get(slate.winner, {})
    print("  " + str(w.get("text", "")).replace(". ", ".\n  "))
    if "persona" in w:
        print("\n  persona verdict:", json.dumps(w["persona"], indent=2)[:600])
    print("\n  within noise of winner:", slate.within_noise)
    print("\n[HONESTY]", slate.honesty[:400])
    print("\nartifacts ->", os.path.relpath(ART), " wall=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
