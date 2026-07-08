"""EXP-084 — the message optimizer, end to end, on the Peter Thiel outreach question.

This is the whole thesis of the change in one run: instead of an LLM writing a few emails and the world
model picking the least-bad one, the world model is the OBJECTIVE and the email is CONSTRUCTED by search:

    public-figure inference (web)  ->  L1 optimal strategy (variable space, no text)
                                   ->  L2 assemble the email move-by-move (beam search, scorer selects)
                                   ->  L3 Monte-Carlo evaluate under the recipient's hidden state

No global "write me an email" call ever happens. The optimizer authors. Everything is `unvalidated`
(coarse world-knowledge priors) — trust the ranking and the lever directions first.

Run:  PYTHONPATH=. python experiments/exp084_message_optimizer.py
"""
from __future__ import annotations

import json

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.llm_moves import SenderBrief
from swm.decision.message_pipeline import optimize_for_world
from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

# The sender's REAL facts — the LLM proposer may use these and must not invent beyond them.
THIEL_SENDER = SenderBrief(
    sender="Beckett",
    thesis="the expensive part of AI is shifting from training models to running them (inference), and "
           "almost nobody is building infrastructure for that",
    ask="a one-line reaction on whether the thesis is wrong",
    facts=["17 years old", "building AI infrastructure that lowers inference cost",
           "got into Princeton but questioning whether to go",
           "previously featured in the NYT for an affordable-housing startup"],
)

# illustrative web evidence about the recipient (a live search_fn replaces this fixture)
_EVIDENCE = [
    {"title": "The Thiel Fellowship pays people to skip college",
     "snippet": "backs young founders who drop out; discovered founders via cold email; took a meeting "
                "with a teenager who had an unusual, contrarian thesis"},
    {"title": "Peter Thiel, the contrarian",
     "snippet": "provocative, heterodox, iconoclast; a skeptic who challenges consensus; famously "
                "skeptical of elite university prestige and the higher-ed status game"},
    {"title": "How to actually reach Peter Thiel",
     "snippet": "hard to reach, screens heavily, rarely responds to cold outreach unless the pitch is "
                "genuinely contrarian and specific"},
    {"title": "Founders Fund and definite optimism",
     "snippet": "billionaire investor and chairman; looks for definite, specific plans and secrets; "
                "dislikes generic status-seeking outreach"},
]


def main():
    world = World(store=EventStore(":memory:"),
                  resolver=PublicFigureResolver(search_fn=lambda q: _EVIDENCE))
    # Live LLM if a backend key is set (DeepSeek): the LLM WRITES the candidate moves and JUDGES them.
    # Otherwise the offline sentence bank + lexical critic run — same architecture, lower text quality.
    chat_fn = default_chat_fn(max_tokens=220, temperature=0.2)
    mode = "LIVE LLM proposer + critic" if chat_fn else "OFFLINE bank + lexical critic (no LLM key set)"
    print("mode:", mode)
    result = optimize_for_world(world, "peter_thiel", name="Peter Thiel", domain="AI infrastructure",
                                ask="cold outreach from a 17-year-old founder",
                                sender_brief=THIEL_SENDER, chat_fn=chat_fn, beam=3 if chat_fn else 6,
                                n_mc=2000)

    print("=" * 80)
    print("EXP-084  message optimizer — 'what should I actually send Peter Thiel?'")
    print("=" * 80)

    print("\n[L1] OPTIMAL STRATEGY (variable space, text-free)")
    spec = result.spec
    print("  predicted reply  mean=%.3f  lower-bound(q20)=%.3f" % (spec.mean, spec.lower_bound))
    for k, v in spec.strategy.items():
        print(f"     {k:>22}: {v:.2f}")
    print("  top drivers (why):")
    for d in spec.drivers[:8]:
        print(f"     {d['term']:>22}: {d['contribution']:+.3f}")

    print("\n[L2] CONSTRUCTED EMAIL (assembled move-by-move, then passed through the critic gate)")
    print("  " + result.email.text.replace(". ", ".\n  "))
    crit = result.email.critique
    if crit is not None:
        print("\n[CRITIC] coherence=%.2f naturalness=%.2f (source=%s)  flags=%d"
              % (crit.coherence, crit.naturalness, crit.source, len(crit.flags())))
        for f in crit.flags():
            print("   ⚠ ", f["issue"], "—", f["sentence"][:70])

    print("\n[L3] MONTE-CARLO EVALUATION (fraction of recipient-trajectories that reply)")
    ev = result.evaluation
    print("  p(reply) fraction = %.3f   mean = %.3f   80%% interval = [%.3f, %.3f]   grade=%s"
          % (ev.p_reply, ev.p_mean, ev.interval80[0], ev.interval80[1], ev.grade))

    print("\n[CONTRAST] the same evaluator on naive drafts:")
    for label, b in result.baselines.items():
        print(f"  {label:>24}: reply_mean={b['mc'].p_mean:.3f}")
        print(f"      \"{b['text'][:72]}...\"")

    print("\n[HONESTY]", result.summary()["honesty"])


if __name__ == "__main__":
    main()
