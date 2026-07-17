# Free evidence breadth — why "more Google RSS" is not the answer

Thirteen articles for a world war was the measured thinness. The fix is not more of the same
modality: Google News RSS returns **recent news prose**, and most of what a causal world model
needs is not news prose. Of the nine evidence categories the requirements layer already declares,
RSS can only genuinely serve two:

| Category | What actually answers it | Free source (no paid APIs) |
|---|---|---|
| leadership statements | news prose | Google News RSS + **GDELT DOC 2.0** (date-bounded, no key) |
| domestic politics | news prose + polling tables | GDELT + **Wikipedia "Opinion polling for …" pages** |
| public opinion | polling aggregation pages | **Wikipedia structured pages** (not news) |
| battlefield information | event datasets + campaign assessments | **UCDP GED** (free research token) + **ISW feed** + ReliefWeb |
| weapons & aid schedules | structured lists | **Wikipedia "List of military aid to …"**, government/defence RSS |
| military capacity | order-of-battle pages, structured facts | **Wikipedia "Order of battle …"** + **Wikidata** |
| economic constraints | macro series | **World Bank API** (no key; GDP/inflation/military-expenditure/reserves) |
| alliance commitments | membership/treaty facts | **Wikidata SPARQL** (member-of, with time qualifiers) |
| leadership incentives | office-holder terms, election calendars | **Wikidata** (positions + start dates) + Wikipedia |
| negotiation history | timelines | **Wikipedia "Timeline of …" pages** + GDELT historical windows |

## The implementation (`evidence_connectors_free.py`)

Seven connectors, all returning the same `(items, RetrievalTrace)` contract as the existing news
connector, each enforcing the as-of hygiene its source supports (date-bounded query parameters
where the API has them — GDELT `enddatetime`, ReliefWeb `filter[value][to]`, World Bank year
window, Wikidata statement-start qualifiers — claimed-pubdate filtering where only the feed
speaks), and each **degrading to a recorded failure trace** when a host is blocked: breadth must
never make gathering fragile.

The **FreeSourceRouter** maps every `EvidenceRequirement` to the connectors that can answer its
*category* (statement / opinion / quantity / calendar / capability — lexical detection over the
requirement's own text, no scenario lists) plus its *domain* (conflict → UCDP + ReliefWeb;
economy → World Bank + IMF/ECB feeds). Routes are ordered breadth-first-then-structured so the
per-requirement cost cap (`max_free_routes_per_req`, default 4) never starves the structured
source that answers what news prose cannot.

Live measurement (this container, conflict question, 3 requirements): **29 documents across 5
source types from 9 connectors** vs ~13 news-only before — with GDELT and the ISW direct feed
blocked by this environment's network policy and UCDP's token unset. Traces record all three
honestly (`network_error`, `auth_required`). In an environment whose allowlist includes
`api.gdeltproject.org` and `understandingwar.org`, breadth rises further with zero code change.

## What stays out, and why

* **Paid APIs** (TradingEconomics, IISS Military Balance, ACLED commercial tier) — excluded by
  policy. The free stack covers the same categories at lower resolution.
* **UCDP** requires a token that is free for research (`UCDP_ACCESS_TOKEN`); unset ⇒ the source
  is skipped with an `auth_required` trace. A free registration is allowed; a paid dependency is
  not.
* **High-frequency personal-activity counters** (the "Musk tweet count" question class) have no
  free structured source here; those questions run on posterior + hazard dynamics with the
  option-space repair, and the forecast is honest about that support grade.

## The next increments this unlocks

1. `fit_intention_hr.py` can swap its archived-news retrieval to GDELT's date-bounded search —
   the paired-date invariant is native to the API, which removes the biggest corpus bottleneck
   for fitting effect sizes.
2. Wikipedia polling/order-of-battle pages are parseable into declared-quantity measurements
   (the `structured_fields` slot on requirements already exists for this).
3. The curated feed registry is data, not code — adding a ministry or central-bank feed is one
   tuple.
