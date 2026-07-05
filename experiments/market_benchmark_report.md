# Market benchmark report (spec Phase 6)

**Directive:** keep Manifold as a hard benchmark, but only fair no-cheat versions — market price at a
fixed horizon (not close), timestamp-bounded retrieval, no post-resolution leakage, segmented by
liquidity/uncertainty/information. Run: no-retrieval, raw LLM + as-of retrieval, world model + as-of
retrieval, market @ fixed horizon. Report whether retrieval closes the information gap.

**Headline:** at a **fair 48h information horizon the market beats the model** (Brier 0.178 vs 0.260,
EXP-006), and the binding constraint is **information staleness, not reasoning**. The one lever that
could close it — **as-of news retrieval** — is now **built and leakage-tested end to end**, but its
**content is BLOCKED-ON-CORPUS** (no licensed timestamped news source in this environment). No fake
news was injected to manufacture a win. On the **information-symmetric** subset the model is already
at near-parity, which is the honest, winnable slice.

## The fair test (unchanged, re-expressed through the reusable module)
`swm/eval/market_comparison.py` now holds the scoring the EXP-006 script inlined: snapshot the market
probability at a fixed lead T (reconstructed from bet history strictly before T), the predictor
forecasts at T blind to the price, both scored against resolution, segmented by liquidity and market
uncertainty. `experiments/market_harness.py` drives it.

### Result (EXP-006, 140 resolved binary markets, post-cutoff, ≥8 bettors, lead=48h)
| segment | n | model Brier | market@48h Brier | model beats market |
|---|---|---|---|---|
| ALL | 140 | 0.260 | **0.178** | 41% |
| thin (<25 bettors) | 106 | 0.272 | 0.175 | 40% |
| deep (≥25 bettors) | 34 | 0.221 | 0.189 | 44% |
| **market-uncertain (0.25–0.75)** | 93 | 0.247 | 0.234 | **52%** |

The market's edge concentrates on the ~40 markets **already information-determined by 48h** (mid-2026
news the model's Jan-2026 cutoff cannot see: "US–Iran deal", "ECB June rate hike", a specific
earthquake toll). On the **market-uncertain** subset — where neither side has decisive news — the
model is at **near-parity and wins the head-to-head 52%**. This is an information deficit, not a
reasoning deficit, which is why EXP-006's reasoning-side iterations (diverse ensembles, extremizing)
did not close it.

## The four arms
| arm | status |
|---|---|
| 1. no retrieval (question text only) | **RUN** — reproduces the fair loss above (model 0.260 vs market 0.178); near-parity on the uncertain subset. |
| 2. raw LLM + as-of retrieval | **BLOCKED-ON-CORPUS** — plumbing built + leakage-tested; needs a timestamped news source. |
| 3. world model + as-of retrieval | **BLOCKED-ON-CORPUS** — same. |
| 4. market @ fixed horizon (48h) | **RUN** — reconstructed from bet history strictly before T. |

## As-of retrieval: built, leakage-proof, blocked only on content
What is real and tested (`tests/test_general_swm.py`):
- `swm/retrieval/asof_store.py` — every item is timestamped; a query with `as_of=T` **physically
  cannot** return an item dated after T; an untimestamped item is **rejected at insert**; a query
  without an as_of is **rejected**.
- `swm/retrieval/news_context.py` — `NewsContext` retrieves as-of; `reject_untimestamped()` refuses
  any external/live news item lacking a publish time (so a naive web search — which would surface the
  post-resolution outcome — cannot be wired in by accident). `LiveNewsAdapter.IMPLEMENTED = False`.
- `swm/eval/leakage.py` — a real gate (was a stub): temporal check, retrieval check, content-hash
  dedup, label-separation. Tests assert future items and label leakage are caught.

**Why arms 2/3 are blocked, honestly:** closing an information gap requires feeding the predictor
*real* news up to T. Fabricating "news" that carries signal about the real resolution would be
me hand-labeling outcomes — exactly the cheating the directive forbids. So the arms are marked
blocked rather than faked. They are ready the moment a timestamped corpus (GDELT, a news API with
publish times, a snapshotted archive) is back-filled into the `AsOfStore`.

## Does retrieval close the information gap?
- On the **market** (this report): **unknown, blocked on a news corpus** — but the diagnosis
  (EXP-006) says it is the *only* thing that can, and the near-parity uncertain subset shows the gap
  is specifically informational.
- On **HN** (EXP-009, where leakage-free as-of retrieval *is* available as author/domain history):
  retrieval **did not help** — the raw LLM already prices HN dynamics from pretraining, so as-of
  context added overconfidence, not signal. Retrieval helps only where the predictor genuinely lacks
  the information *and* a timestamped source of it exists.

## Strategic read (per the audit, unchanged and reinforced)
"Beat liquid prediction markets on public resolvable events" is the hardest bar in forecasting and
the one arena engineered to be unbeatable by a static-knowledge model. Keep it as a **calibration
KPI on the information-symmetric subset** (we're ~at parity) and pursue as-of retrieval to push it —
but the product moat is the **counterfactual, no-market-exists** questions (will *this* email get a
reply, will *this* post land) where no aggregator has priced the answer. That is where EXP-004's
decision lift and the individual model point.
