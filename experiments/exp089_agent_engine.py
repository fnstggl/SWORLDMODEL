"""EXP-089: the grounded-agent engine, live — the questions that exposed the old system, re-asked.

The old front door's failures on these exact questions (documented in the vision-gap analysis): NY-10
returned "P(event)=0.957" about an abstract logit naming no candidate (and missed that the primary had
ALREADY HAPPENED); the headline question returned p=0.38 about copy-properties of a headline that was
never written. The agent engine must: (1) notice NY-10 resolved (Lander d. Goldman, 2026-06-24) with a
citation; (2) return ACTUAL ranked headline texts; (3) give a scenario-specific p(reply) for one named
person × one exact email, from N grounded runs, never a base rate; (4) on a genuinely open race, return a
distribution over NAMED candidates with per-persona WHY audit, flagged ungraded until the class is
backtested.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp089_agent_engine
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RESULT = "experiments/results/exp089_agent_engine.json"

COLD_EMAIL = """Subject: contrarian take on defense-tech vertical software

Peter — I run a 4-person team building procurement software the primes can't build internally
(we've closed 2 pilot contracts with Tier-1 suppliers in 90 days, $340k ACV). Everyone says sell to
startups; we think the primes' broken procurement IS the moat. You wrote the book on secrets —
this one's hiding in plain sight. 15 minutes to show you the numbers?"""


def run():
    from swm.engine.front_door import agent_world_model
    wm = agent_world_model(branches=2, max_rounds=2)

    out = {}
    for tag, kwargs in [
        ("ny10_primary", {"question": "Who will win the Democratic primary in NY-10?"}),
        ("airpods_headline", {"question": "What is the best landing page headline for Apple to "
                                          "maximize AirPods Max sales on the web?"}),
        ("thiel_cold_email", {"question": "Will Peter Thiel respond to this cold email?",
                              "recipient": "Peter Thiel", "message": COLD_EMAIL}),
        ("open_race", {"question": "Who will win the 2026 New York governor's race?"}),
    ]:
        t0 = time.time()
        try:
            res = wm.simulate(**kwargs)
        except Exception as e:                             # a live run must record, never crash the batch
            res = {"error": f"{type(e).__name__}: {e}"}
        res["elapsed_s"] = round(time.time() - t0, 1)
        out[tag] = res
        print(f"\n=== {tag} ({res.get('elapsed_s')}s) ===")
        print("  mechanism :", res.get("mechanism"))
        print("  headline  :", res.get("headline"))
        if res.get("distribution"):
            print("  dist      :", json.dumps(res["distribution"]))
        for a in (res.get("ranked_artifacts") or [])[:5]:
            print(f"    {a['p_engage']:.0%}  {a['text']}  [{a.get('angle')}]")
        if res.get("abstain"):
            print("  ABSTAINED :", res.get("abstain_reason"))
        g = res.get("grounding") or {}
        print(f"  grounding : {g.get('n_passages', '?')} passages, coverage {g.get('coverage', '?')}, "
              f"missing {len(g.get('missing') or [])}")
        print("  calibration:", (res.get("calibration") or {}).get("grade"))

    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
