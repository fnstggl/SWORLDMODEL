"""EXP-095 — the DEFAULT reply-first architecture on the same Thiel inputs, fully traced.

Same inputs as exp092/094 (same sender facts from runaurelius.com, same recipient evidence, same
saved inbox-reality hypotheses). The pipeline is the new default (`optimize_cold_outreach`
method="reply_first"): desired replies → backward requirements → beat structures → blind
outcome-ranked beat search → capped wording pass → three separated judges (truth / human-language /
blind persona outcome) → ONE recommended message, no simulated percentages in human-facing output.

EVERY LLM interaction is captured twice over:
  * artifacts/phase13/thiel_v5/llm_trace.jsonl   — every raw call (prompt + response), any seam
  * artifacts/phase13/thiel_v5/plan_trace.jsonl  — the planner's stage-labeled calls + internal
                                                    outcome counts (machine-readable only)
so a reader can replay exactly what each model call saw and returned, end to end.

Run:  PYTHONPATH=. python experiments/exp095_thiel_reply_first.py            (live)
      PYTHONPATH=. python experiments/exp095_thiel_reply_first.py --offline  (structure smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.llm_moves import SenderBrief
from swm.decision.message_pipeline import optimize_cold_outreach, recipient_from_world
from swm.decision.persona_response import PersonaDossier, specialize_hypotheses
from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_v5")
V3_ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_v3")

AURELIUS_SENDER = SenderBrief(
    sender="Beckett",
    thesis="AI infrastructure has a planning problem disguised as a power problem: schedulers "
           "optimize the next placement, but nothing chooses the fleet's best trajectory over time",
    ask="get Peter to ask for the one-page technical memo or engage with the idea",
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


class TracingChat:
    """Record EVERY raw LLM interaction (prompt + response + kwargs + latency), whatever seam
    called it — the planner, the truth judge, the language judge, the persona ensemble, the
    wording editor. This is the exp095 under-the-hood artifact."""

    def __init__(self, fn, path):
        self.fn, self.path, self.n = fn, path, 0
        self.default_max_tokens = getattr(fn, "default_max_tokens", None)
        self.default_temperature = getattr(fn, "default_temperature", None)

    def __call__(self, prompt, **kw):
        t0 = time.time()
        out = self.fn(prompt, **kw) if kw else self.fn(prompt)
        self.n += 1
        with open(self.path, "a") as f:
            f.write(json.dumps({"call": self.n, "kwargs": kw,
                                "wall_s": round(time.time() - t0, 2),
                                "prompt": prompt, "response": out}) + "\n")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    for f in ("llm_trace.jsonl", "plan_trace.jsonl"):
        p = os.path.join(ART, f)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()

    world = World(store=EventStore(":memory:"),
                  resolver=PublicFigureResolver(search_fn=lambda q: _EVIDENCE))
    chat = None
    if not args.offline:
        backend = default_chat_fn(max_tokens=800, temperature=0.2)
        if backend is not None:
            chat = TracingChat(backend, os.path.join(ART, "llm_trace.jsonl"))
    recipient = recipient_from_world(world, "peter_thiel", name="Peter Thiel",
                                     domain="AI infrastructure", ask="cold outreach")
    dossier = PersonaDossier.for_public_figure(world, "peter_thiel", name="Peter Thiel",
                                               role="investor, Founders Fund")
    if not dossier.evidence:
        dossier.evidence = [(e["title"], e["snippet"]) for e in _EVIDENCE]
    try:
        hyps = json.load(open(os.path.join(V3_ART, "hypotheses.json")))["recipient_hypotheses"]
        hyp_src = "thiel_v3 saved (same as exp092/094)"
    except Exception:  # noqa: BLE001
        hyps = specialize_hypotheses(chat, dossier)
        hyp_src = "specialized this run"

    result = optimize_cold_outreach(
        recipient, sender_brief=AURELIUS_SENDER, chat_fn=chat,
        recipient_notes="Peter Thiel; screens heavily; treat all cold inbound as barely read",
        dossier=dossier, hypotheses=hyps, persona_draws=3,
        trace_path=os.path.join(ART, "plan_trace.jsonl"))          # method defaults to reply_first

    payload = {"experiment": "exp095_thiel_reply_first",
               "mode": "live" if chat else "offline", "hypotheses_source": hyp_src,
               "result": result.summary(),
               "n_raw_llm_calls": chat.n if chat else 0,
               "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(ART, "result.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)

    from swm.world_model_v2.phase13.ledger import DecisionLedger
    DecisionLedger(os.path.join(ART, "ledger.jsonl"))._append({
        "kind": "frozen_decision", "decision_id": "thiel_outreach_v5_reply_first",
        "recommendation": result.candidates[result.winner]["text"],
        "label": result.honesty, "chosen_real_action": None, "realized_outcome": None})

    print("=" * 88)
    print("EXP-095  reply-first (DEFAULT) on the same Thiel inputs")
    print("=" * 88)
    print("\n[THE ONE RECOMMENDED MESSAGE]")
    print("  " + result.candidates[result.winner]["text"].replace(". ", ".\n  "))
    print("\n[ORIGIN]", result.candidates[result.winner].get("origin"))
    print("[LABEL]", result.honesty)
    print("[RAW LLM CALLS]", chat.n if chat else 0,
          "| planner-labeled calls in plan_trace.jsonl")
    print("\nartifacts ->", os.path.relpath(ART), " wall=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
