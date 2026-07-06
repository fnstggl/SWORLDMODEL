# EXP-058 — Retrieval + a leakage-free LLM-loop as a *measured* general predictor

**One-line:** the generative loop (EXP-057) needed a real input; this wires **retrieval** into
the front of it and a **forward scoring log** onto the back, closing the pipeline into a live
forecaster — and it answers the cutoff question directly.

---

## The cutoff question (the thing that motivated this)

> *"this should work for all general requests when we need to retrieve context or variables from
> the internet not only in this chat/for tests — because on the API I don't think the cutoff
> matters anymore? or am I wrong?"*

**You are right.** The training cutoff limits what the model has *memorized*, not what the system
can *do*. Two separate things get conflated, so let's keep them apart:

1. **Capability** — *can it forecast a question about the future?* The cutoff is irrelevant.
   Retrieval supplies the current evidence; the model reasons over what it just read, not over what
   it happened to memorize. Proof, committed in this experiment: the FOMC context
   (`fomc_context.json`) contains facts dated **June 2026** — a *new* Fed Chair (Kevin Warsh), the
   3.5–3.75% June range, the July-hike odds. Those are **after the Jan-2026 training cutoff**; the
   model could not have memorized them. The loop still produces a coherent forecast from them
   because they were *retrieved*, not recalled.

2. **Measurement** — *can we honestly score it?* This is the **only** place the cutoff bites, and
   only for *tests*. A backtest against a **known** outcome can be gamed two ways: the model
   *recalls* the answer (it was in training), or search *retrieves* the resolved result. Neither is
   a real forecast. So a *test* has to be built to exclude both.

The resolution is that there are exactly **two clean measurement paths**, and this harness
implements both:

| Path | When | Why it's leakage-free |
|---|---|---|
| **FORWARD** | production / any genuinely-future question | the event hasn't happened — **there is no answer to leak** |
| **AS-OF BACKTEST** | offline eval on post-cutoff resolutions | `asof_retriever` serves only evidence *dated before* resolution — no memorization (post-cutoff) and no retrieved outcome |

**So for a real user request on the API, there is nothing to worry about.** You are forecasting a
future that has not happened; leakage is a concern that exists *only* when you are grading yourself
against an outcome you could already know. In production, you never are. The cutoff caveats in the
code and reports are about keeping *our own benchmarks* honest — they are not a limit on the product.

---

## What EXP-058 builds

### 1. Retrieval layer — `swm/api/retrieval.py`

The generative loop instantiates agents and has them reason, but its `context` was a stub. This is
the real input. Same pluggable pattern as every other LLM-touching part of the system:

- `web_search_retriever(search_fn)` — **PRODUCTION.** `search_fn(query) -> [{title, snippet, date,
  source}]` over any web-search backend. Resilient: a search that errors returns empty rather than
  crashing the forecast.
- `asof_retriever(news_by_question)` — **LEAKAGE-FREE EVAL.** Serves committed as-of news (dated
  before resolution) so a backtest cannot see the future.

`retrieve(question, as_of)` returns a bounded, timestamped `RetrievedContext` with `.to_prompt()`
for the identify/position calls.

### 2. Live forecaster — `swm/eval/live_forecast.py`

`LiveForecaster.forecast(question, fid, made_at, resolves_at, as_of)` = **retrieve → simulate →
forecast → log**. It logs to a `PostMortemLog` with the resolution date so the skill number accrues
*on resolution* — made-before, scored-after, leakage-free by construction. `resolve(fid, outcome)`
and `skill()` close the loop.

### 3. The live run — `experiments/exp058_live_forecast.py`

A genuinely-future question, forecast end-to-end from retrieved post-cutoff evidence.

**Q: Will the FOMC raise the federal funds rate at its July 28-29, 2026 meeting?**

- **Evidence retrieved:** 5 snippets, all dated June 2026 (federalreserve.gov, Forbes, Wells Fargo,
  bondsavvy) — expected-unchanged guidance, elevated inflation nudging hike odds up, new Chair
  Warsh signalling strategic ambiguity.
- **Agents:** the loop instantiates the FOMC as 12 members from that evidence — Chair Warsh
  (hawkish-leaning, high influence, patient), a 3-member hawk bloc, a 5-member moderate majority, a
  3-member dove bloc. Position = P(this member votes to hike in July).
- **Deliberation:** `AgentSociety(homophily=0.4, consensus_pull=0.3, rounds=6)` — the committee
  deliberates with influence-weighting and homophily.

**Forecast: P(hike in July) = 0.333** (deliberated vote share) → **the FOMC leans HOLD**, consistent
with the retrieved "expected unchanged" evidence, while pricing the real hawkish minority. The
trajectory is stable across rounds (the moderate majority holds; the hawk bloc doesn't have the
influence to flip consensus) — the honest read for a committee whose base case is "no change, option
preserved."

For a **vote** question the outcome is the *majority*, so we report the deliberated vote share (the
fraction of members leaning to hike), not the mean position — the correct aggregation for a committee
decision.

**Logged** `fid=fomc-2026-07`, made 2026-07-06, resolves 2026-07-29 → a leakage-free skill number
accrues the moment the FOMC meets. Nothing to leak: the meeting hasn't happened.

---

## Tests — `tests/test_retrieval_live.py` (5, all pass)

- retriever bounds + prompt formatting
- `web_search_retriever` resilience (a throwing search never crashes the pipeline)
- `asof_retriever` serves committed as-of news at the right `as_of`
- live forecaster logs with resolution metadata (`made_at < resolves_at` — leakage-free by
  construction)
- forecast → resolve → score: the full loop closes with `leakage_free == True`

---

## Where this leaves the system

The pipeline is now **end-to-end and general**: any question → retrieve real context → identify the
deciding agents + map their variables from that context → deliberate → emergent forecast → log for
honest scoring. The retrieval front door means it is **not** limited to the datasets we happened to
cache, and **not** limited by the training cutoff for *capability* — only for *measurement*, which
the forward log and as-of retriever handle by construction.

Swap `asof_retriever` for `web_search_retriever(<live search>)` and the same code forecasts any live
question on the API.
