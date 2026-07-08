"""EXP-086: live grounding — the router grounds the ACTUAL present, over real keyless backends + a real LLM.

EXP-085 proved the coverage architecture on fixtures. This runs it LIVE: real Coinbase market data, real
DuckDuckGo/Wikipedia retrieval, and DeepSeek as both the value-extractor AND the variable→series matcher
(LLM-inferred matching — no hardcoded token rules). It grounds a spread of real variables across domains right
now and prints each grounded value with its provenance and CI.

It is a live SMOKE/DEMO, not a fixed metric: results move with the world and depend on network + the
DEEPSEEK_API_KEY. If no LLM key or the network is down it skips gracefully (so CI never breaks). Run it with a
key set to see today's world grounded end-to-end.

Run: DEEPSEEK_API_KEY=... python -m experiments.exp086_live_grounding
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RESULT = "experiments/results/exp086_live_grounding.json"

# (domain, question, variable) — a spread across domains; some hit the live market source, some the LLM+web
# retrieval, and the LLM resolver decides which. Paraphrases probe that matching is semantic, not lexical.
PROBES = [
    ("markets", "Will BTC top $100k this year?", "bitcoin price"),
    ("markets", "Ethereum outlook", "the price of ether"),          # paraphrase -> eth_usd (semantic match)
    ("markets", "Solana momentum", "SOL/USD spot price"),
    ("macro", "US recession odds in a year?", "current US unemployment rate"),
    ("macro", "Is inflation cooling?", "current US CPI inflation rate, percent"),
    ("macro", "Fed path", "current federal funds target rate"),
    ("demography", "US population trend", "current US population"),
    ("public_health", "US longevity", "current US life expectancy in years"),
    ("energy", "Clean energy transition", "current renewable share of US electricity, percent"),
]


def run() -> dict:
    from swm.api.live_grounding import json_llm, live_router
    if json_llm() is None:
        print("EXP-086  live grounding — SKIPPED (no DEEPSEEK_API_KEY / HF_TOKEN configured)")
        return {"skipped": "no LLM backend"}

    router = live_router()
    rows, grounded, via_struct, via_retr = [], 0, 0, 0
    for dom, q, var in PROBES:
        try:
            gv = router.ground(var, question=q)
        except Exception as e:                                       # a live failure must not kill the run
            gv = None
            print(f"    ({var}: {str(e)[:60]})")
        if gv is not None:
            grounded += 1
            kind = "retrieval" if gv.source.startswith("retrieval") else "structured"
            via_struct += kind == "structured"; via_retr += kind == "retrieval"
            rows.append({"domain": dom, "variable": var, "value": gv.value, "sd": gv.sd,
                         "source": gv.source, "kind": kind})
        else:
            rows.append({"domain": dom, "variable": var, "grounded": False})

    res = {"n_probes": len(PROBES), "grounded": grounded, "coverage": round(grounded / len(PROBES), 3),
           "via_structured": via_struct, "via_retrieval": via_retr, "rows": rows,
           "note": "live smoke — values move with the world; provenance and CI are the point"}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print("EXP-086  LIVE grounding over real backends (Coinbase market + DDG/Wikipedia + DeepSeek extractor,")
    print("         LLM-inferred matching) — grounding today's world:")
    for r in rows:
        if r.get("grounded", True):
            print(f"    {r['domain']:13s} {r['variable'][:40]:40s} = {r['value']:>14.3f}  "
                  f"±{r['sd']:<10.3f} [{r['kind']:10s}] {r['source']}")
        else:
            print(f"    {r['domain']:13s} {r['variable'][:40]:40s} = (ungrounded — honest)")
    print(f"  coverage {res['coverage']*100:.0f}%  ({via_struct} live-structured + {via_retr} LLM-retrieval)")
    print(f"  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run()
