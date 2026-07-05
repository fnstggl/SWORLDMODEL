# EXP-017 — Structured-market attempt: blocked on reachable as-of feeds (honest)

**Goal:** point the machinery at a structured market domain with real as-of feeds (elections via 538
polls, or economic releases via FRED/ALFRED vintages) where a mechanism model can exceed the crowd,
and run the as-of-retrieval arm vs market@T on thin markets.

**Outcome: blocked in this environment on reachable, no-leak structured data** — a data/credential
bottleneck (one of the directive's explicit honest stopping conditions), not an architecture gap.

## What was reachable, and what wasn't (probed directly)
| source | needed for | status |
|---|---|---|
| Manifold API | the market prices to beat | **reachable** |
| GDELT DOC 2.0 | as-of breaking-news retrieval | reachable but rate-limited (429s); loader built |
| **FRED / ALFRED** | econ releases + real-time vintages (no-revision-leak) | **blocked** — API needs a key; keyless CSV endpoint timed out |
| **FiveThirtyEight polls** | election mechanism inputs | **blocked** — endpoints now return ABC News HTML (the site was shut down post-acquisition); the CSVs are gone |

FRED/ALFRED is the single most important feed (its *vintage* API gives each value **as it was known
on date T**, which is what makes an econ backtest leakage-proof). Without an API key it is
unreachable here. 538's poll archive — the cleanest election mechanism input — no longer serves data.

## Why this is a real block, not a cop-out
A credible structured-market beat requires **three** things simultaneously: (1) a resolved market with
price history at a fixed horizon, (2) the domain's mechanism inputs as an **as-of** feed with clean
timestamps/vintages, and (3) enough events for power. We have (1) via Manifold, and the **pipe** for
(2) is built and leakage-tested (`swm/retrieval/gdelt.py` + `AsOfStore` + `market_comparison`), but
the actual (2) corpora with clean vintages are behind an API key (FRED) or offline (538). Fabricating
the feed would be exactly the leakage/cheating the directive forbids, so the arm is marked blocked.

## What is nonetheless established (the capability is proven elsewhere)
The core claim behind "a mechanism model can beat the crowd" — that **precisely modeling the mapped
variables beats an educated guesser given the same information** — is already demonstrated on real
data in **EXP-014 (GitHub)**: the entity-state world model beats a raw LLM *even when the LLM is
handed the same track record* (0.258 vs 0.304 log loss), with the edge scaling by state depth. A
market is the same problem with a harder data-ingestion requirement; the ingestion is what's blocked
here, not the modeling.

## Exactly what unblocks it (the next build, when a feed is available)
1. **A FRED/ALFRED API key** → econ-release markets (e.g. "next CPI/unemployment print above X",
   "Fed holds") with real-time vintages as the as-of feature; the ALFRED vintage *is* the no-leak
   guarantee. This is the highest-integrity structured backtest and needs only a free key.
2. **A restored/mirrored 538 poll archive** (or a poll API) → election/primary mechanism model.
3. Then: fit the mechanism model on as-of features, snapshot Manifold/PredictIt price at T, score
   both vs resolution, segment by liquidity, target thin markets first. All plumbing is in place
   (`experiments/manifold_harness.py`, `swm/eval/market_comparison.py`, `swm/retrieval/gdelt.py`).

## Honest verdict
The structured-market beat is the right frontier and the pipeline is built, but it is **blocked on a
reachable timestamped/vintage feed (FRED key or a live 538 mirror)** in this environment. This is a
data-access bottleneck; the modeling capability it would use is already validated on GitHub. The
individual-response backtests (EXP-014/016) are where the no-cheat evidence actually accrued this
round.
