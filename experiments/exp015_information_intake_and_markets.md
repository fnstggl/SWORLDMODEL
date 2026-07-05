# EXP-015 — Information intake and beating prediction markets: the honest path

Directly addresses the thesis: *if we truly model the moving parts of the world with the relevant
variables mapped, and take in the same information the market participants had, we should beat
individuals educated-guessing.* This lays out what is true, what we measured, and the concrete build
— no hype.

## What we already proved (the thesis holds where we can test it)
On **real state-rich data** (GitHub, EXP-014) the world model beats a raw LLM **even when the LLM is
handed the same information** (log loss 0.258 vs 0.304), and the edge scales with how much entity
state exists. So the core claim — *precise modeling of the mapped variables beats an educated
guesser given the same inputs* — is confirmed where the variables are observable and the entity
recurs. Markets are the hardest instance of the same claim.

## The one caveat that must be internalized
**A prediction-market price is already an information-aggregation world model.** It integrates every
participant's information *and* their models into one number. So "ingest the same information" gets
you to **parity**, not superiority — and EXP-006 measured exactly this: at a fair 48h horizon we
lose overall (Brier 0.260 vs 0.178) because of **information staleness**, but on the
**information-symmetric** subset (where neither side has decisive late news) we are at **near-parity**
(0.247 vs 0.234, 52% head-to-head). Matching the pipe removes the staleness penalty and takes us to
parity. Beating the market requires an edge *beyond* information parity.

## Where a real world model can exceed the market (three real edges)
1. **Model the mechanism better than the crowd** — on *structured* questions whose moving parts are
   mappable: elections (polls + fundamentals, à la 538), economic releases (leading indicators),
   epidemics (SEIR), sports (player/team stats), scheduled corporate/legal events. Here explicit
   simulation of the actual variables beats vibes-betting. **This is the fundable frontier** and it
   is the same machinery as our HN simulation engine and GitHub entity model — just pointed at a
   domain whose mechanism we map.
2. **Be faster / broader than thin crowds** — most Manifold markets have <25 bettors; a model that
   ingests news as-of and updates beats a thin, slow crowd. Our thin-market segment is already our
   closest to the market.
3. **Exploit systematic market biases** — longshot bias, overreaction to news, round-number
   clustering. A calibrated model can harvest these.

Information intake is *necessary* for all three (you can't beat a mechanism you can't see the inputs
to), but it is *sufficient* only for reaching parity on pure information-race questions.

## What is built now (the leakage-proof intake pipe)
The plumbing the thesis needs already exists and is leakage-tested:
- `swm/retrieval/asof_store.py` — a store that **physically cannot return an item dated after the
  forecast horizon**; untimestamped items are refused at insert.
- `swm/retrieval/gdelt.py` (**new**) — `GDELTLoader`: pulls GDELT's free, timestamped global news
  index into the AsOfStore, hard-gated at the horizon (`enddatetime<=T`, drops any `seendate>T`,
  refuses articles without a publish time). This is the as-of news pipe for markets.
- `swm/retrieval/news_context.py` + `swm/eval/leakage.py` — the guards + gate + tests.
- `swm/eval/market_comparison.py` — fair price-at-fixed-horizon scoring, segmented by liquidity.

## What is still needed (the concrete next build, ranked)
1. **A timestamped, no-revision data corpus per domain.** Free, backtestable sources with clean
   as-of timestamps:
   - **GDELT** (news/events; loader built) — general breaking-news intake.
   - **FRED/ALFRED** — economic releases with *real-time vintages* (the value as it was known then;
     no revision leakage). Essential for econ questions.
   - **SEC EDGAR** — timestamped filings (corporate/legal events).
   - **538 / poll aggregators** — elections.
2. **Per-domain mechanism models** — the moving parts mapped as an explicit simulation (our engine
   pattern), not a generic classifier.
3. **The as-of retrieval arm vs market@T** — run: no-retrieval, LLM + as-of retrieval, world model +
   as-of retrieval, market@T; segment by liquidity; **target thin markets first** (edge 2) and
   structured markets (edge 1).
4. **A market-bias exploitation layer** on top of the calibrated model (edge 3).

## Honest expectation
- On **pure breaking-news races**: reaching **parity** by matching the pipe is realistic; *beating*
  the market there is hard because the market has the same news faster.
- On **structured, mechanism-driven questions**: a correctly-built world model that maps the
  variables can **exceed** the crowd — this is where "model the moving parts" genuinely pays, and it
  is the same edge we already demonstrated on GitHub (precise state modeling > guessing over the same
  inputs).
- The bottleneck is **data ingestion + per-domain mechanism modeling**, exactly as the thesis says —
  and the leakage-proof pipe for it is now in place. The next experiment is a structured domain
  (elections or econ) with real as-of feeds, where the win is most plausible and fully backtestable.
