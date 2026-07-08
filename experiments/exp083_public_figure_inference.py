"""EXP-083 — public-figure inference, end to end (the Peter Thiel outreach question, for real).

This runs the actual pipeline, not a narration of it:

    name -> PublicFigureResolver (search online + infer variables, web provenance)
         -> World.predict / World.compare  (no fitted readout -> UNVALIDATED inference, never a block)

The web evidence below is a small FIXTURE standing in for what a live `search_fn` would return — the
snippets are illustrative, not asserted facts, and everything downstream is labeled by provenance and
confidence. In production you pass a real `search_fn(query) -> [{title, snippet, url, date}]` (and
optionally an LLM `infer_fn`); with neither, the resolver degrades to a transparent prior.

Run:  python experiments/exp083_public_figure_inference.py
"""
from __future__ import annotations

import json

from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

# --- fixture: illustrative web evidence about the recipient (replace with a live search_fn) --------
_EVIDENCE = [
    {"title": "The Thiel Fellowship pays people to skip college",
     "snippet": "backs young founders who drop out; has discovered founders via cold email; took a "
                "meeting with a teenager who had an unusual thesis"},
    {"title": "Peter Thiel, the contrarian investor",
     "snippet": "provocative, heterodox, iconoclast; a skeptic who challenges consensus; famously "
                "skeptical of elite university prestige and the higher ed bubble / status game"},
    {"title": "How to actually reach Peter Thiel",
     "snippet": "hard to reach, screens heavily, rarely responds to cold outreach unless the pitch is "
                "genuinely contrarian and specific; gatekept inbox"},
    {"title": "Founders Fund and definite optimism",
     "snippet": "billionaire investor and chairman; looks for definite, specific plans and secrets; "
                "dislikes generic status-seeking outreach"},
]


def search_fn(_query: str):
    """Stand-in for a web backend. A real one would query per `_query`; the fixture ignores it."""
    return _EVIDENCE


DRAFTS = {
    "A_contrarian_low_friction":
        "Peter — I'm 17, got into Princeton, and I'm fairly sure going is the wrong trade. I'm "
        "building AI infra for [the specific bottleneck]. The secret I'm betting on: [one contrarian "
        "claim most people in AI would disagree with]. Not asking for money or a meeting — is that "
        "thesis obviously wrong to you? One line back and I'll leave you alone.",
    "B_credential_parade_pushy":
        "Dear Mr. Thiel, I'm a 17-year-old Princeton admit and I was recently featured in the NYT for "
        "my affordable-housing startup. I'd love to urgently set up a call about my AI infrastructure "
        "company. Please respond ASAP — just following up and circling back per my last note.",
}


def main():
    resolver = PublicFigureResolver(search_fn=search_fn)   # swap for a live backend in production
    world = World(store=EventStore(":memory:"), resolver=resolver)

    print("=" * 78)
    print("EXP-083  public-figure inference — 'what should I send Peter Thiel?'")
    print("=" * 78)

    # 1) resolve the recipient from what's observable online
    profile = resolver.resolve("Peter Thiel", domain="AI infrastructure",
                               ask="cold outreach from a 17-year-old founder")
    print("\n[1] RESOLVED RECIPIENT (provenance: web, inferred not asserted)")
    print(json.dumps(profile.summary(), indent=2))

    # 2) rank the drafts — no fitted readout, so this is an UNVALIDATED inference, not a refusal
    ranking = world.compare("peter_thiel", list(DRAFTS.values()), name="Peter Thiel")
    label = {v: k for k, v in DRAFTS.items()}
    print("\n[2] RANKED DRAFTS  (grade: %s)" % ranking["calibration"]["grade"])
    for r in ranking["ranked"]:
        print(f"  p(reply)={r['p_mean']:.3f}  {label[r['text']]}")
        for d in r["drivers"]:
            print(f"       {d['feature']:>28}: {d['contribution']:+.3f}")

    # 3) the honest footer — what grade this is and how to earn a better one
    best = world.predict("peter_thiel", DRAFTS["A_contrarian_low_friction"], name="Peter Thiel")
    print("\n[3] HONESTY")
    print("   grade:", best["calibration"]["grade"])
    print("   note :", best["calibration"]["note"])
    print("   base responsiveness n_effective:",
          best["provenance"]["base_responsiveness_n_effective"])


if __name__ == "__main__":
    main()
