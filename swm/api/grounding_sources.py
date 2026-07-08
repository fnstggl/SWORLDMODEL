"""Grounding COVERAGE — grounders for real, live evidence across ALL general domains, and a router that
matches any high-leverage variable to the best one.

State grounding (EXP-082) and rate grounding (EXP-084) proved the LEVER: measuring a variable's current value
(and its recent rate) from evidence, instead of guessing, is where the accuracy is. But they used hand-wired
grounders over committed datasets. To ground an ARBITRARY question, the system needs COVERAGE — and coverage
across every domain, not a few. The architecture that makes that tractable is two-tier:

  1. TYPED STRUCTURED SOURCES for the domains where precise, as-of numeric series exist — macro (FRED),
     markets (crypto/equities/rates), polls (approval/elections), sports (Elo/standings), product analytics,
     demography/health/energy. Each is a `StructuredSource`: a set of variable ALIASES + a pluggable `fetch`
     (point value + CI) and optional `fetch_series` (a recent trajectory, for RATE grounding). The backend is
     injectable, so the SAME source runs against a live API in production and a committed fixture in tests.
  2. A UNIVERSAL RetrievalGrounder fallback (swm.api.retrieval_grounding) for everything the structured
     sources don't cover — the long tail of any domain, grounded from as-of web evidence + a CALIBRATED LLM
     value-extractor. This is what makes coverage GENERAL rather than a fixed menu.

`GroundingRouter` ties them together: it semantically matches a variable to the best structured source, and
falls back to retrieval. It is itself a `Grounder`, so dropping it in as `StateGrounder(default=router)` gives
the state layer broad coverage for free; `ground_series` feeds the rate layer (`TransitionOperator.ground_gain`)
the same way. One router grounds any question, end to end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.api.state_grounding import GroundedValue, Grounder
from swm.variables.embedding_registry import _cos

# generic quantity words carry no domain meaning — matching on them ("...rate", "...index") produces false
# positives (case RATE ~ inflation RATE), so they are stripped before lexical matching.
_GENERIC = {"rate", "index", "level", "current", "the", "of", "and", "a", "an", "total", "per", "status",
            "price", "share", "growth", "number", "value", "amount", "score", "ratio", "average"}


def _content(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2 and t not in _GENERIC}


def _lexical_recall(variable, phrase):
    """Fraction of the alias's DISTINCTIVE tokens present in the variable — requires a real content match, not
    just a shared generic word. (Synonyms with no token overlap need a real `embed_fn`; this is the robust,
    offline default that never fires spuriously.)"""
    vc, pc = _content(variable), _content(phrase)
    return len(vc & pc) / len(pc) if pc else 0.0


def _embed_score(embed_fn, a, b):
    va, vb = embed_fn(a), embed_fn(b)
    return _cos(va, vb) if va is not None and vb is not None else 0.0


@dataclass
class StructuredSource(Grounder):
    """A typed numeric source for one domain. `aliases` maps a canonical series key to the phrases a variable
    might be called; `fetch(key, as_of) -> (value, sd) | None` measures the current value with a CI;
    `fetch_series(key, as_of, window) -> [values] | None` returns a recent trajectory for rate grounding.
    The backends are injected, so this runs live in production and against a fixture offline."""
    name: str
    domain: str
    aliases: dict                              # canonical_key -> [alias phrases]
    fetch: object = None                       # (key, as_of) -> (value, sd) | None
    fetch_series: object = None                # (key, as_of, window) -> [value, ...] | None
    embed_fn: object = None                    # optional real sentence embedder for SEMANTIC (synonym) matching
    threshold: float = 0.6

    def match(self, variable, question=None):
        """Best (canonical_key, score) for this variable, or None if nothing clears the threshold. The default
        matcher is distinctive-content-token recall (robust offline, never fires on a shared generic word); a
        real `embed_fn` upgrades it to semantic matching so synonyms with no token overlap still resolve."""
        best_key, best = None, 0.0
        for key, phrases in self.aliases.items():
            for ph in phrases:
                s = (_embed_score(self.embed_fn, variable, ph) if self.embed_fn is not None
                     else _lexical_recall(variable, ph))
                if s > best:
                    best_key, best = key, s
        return (best_key, best) if best >= self.threshold else None

    def ground(self, variable, question=None, as_of=None):
        m = self.match(variable, question)
        if m is None or self.fetch is None:
            return None
        r = self.fetch(m[0], as_of)
        return GroundedValue(float(r[0]), float(r[1]), f"{self.name}:{m[0]}") if r is not None else None

    def ground_series(self, variable, question=None, as_of=None, window=6):
        m = self.match(variable, question)
        if m is None or self.fetch_series is None:
            return None
        seq = self.fetch_series(m[0], as_of, window)
        return (m[0], [float(v) for v in seq]) if seq else None


@dataclass
class GroundingRouter(Grounder):
    """Routes a variable to the best-matching structured source, falling back to a universal retrieval
    grounder. Being a `Grounder`, it drops straight into `StateGrounder(default=router)` (state layer) and its
    `ground_series` feeds `TransitionOperator.ground_gain` (rate layer) — one router grounds any question."""
    sources: list = field(default_factory=list)     # list[StructuredSource]
    retrieval: object = None                         # a universal RetrievalGrounder (the long-tail fallback)

    def _best_source(self, variable, question):
        best = None
        for s in self.sources:
            m = s.match(variable, question)
            if m is not None and (best is None or m[1] > best[2]):
                best = (s, m[0], m[1])
        return best

    def ground(self, variable, question=None, as_of=None):
        b = self._best_source(variable, question)
        if b is not None:
            gv = b[0].ground(variable, question, as_of)
            if gv is not None:
                return gv
        return self.retrieval.ground(variable, question, as_of) if self.retrieval is not None else None

    def ground_series(self, variable, question=None, as_of=None, window=6):
        """A recent trajectory for RATE grounding, from the best structured source that carries a series."""
        b = self._best_source(variable, question)
        if b is not None:
            seq = b[0].ground_series(variable, question, as_of, window)
            if seq is not None:
                return seq
        return None

    def route_report(self, variables, question=None, as_of=None):
        """Which grounder each variable resolves to — the coverage picture for a question."""
        out = []
        for v in variables:
            b = self._best_source(v, question)
            gv = self.ground(v, question, as_of)
            out.append({"variable": v, "grounded": gv is not None,
                        "via": (gv.source.split(":")[0] if gv is not None else None),
                        "kind": ("structured" if b is not None and gv is not None
                                 and not gv.source.startswith("retrieval")
                                 else ("retrieval" if gv is not None else None)),
                        "matched_source": (b[0].name if b is not None else None),
                        "match_score": round(b[2], 3) if b is not None else None})
        return out

    def coverage(self, variables, question=None, as_of=None):
        r = self.route_report(variables, question, as_of)
        n = len(r) or 1
        return {"n_variables": len(r), "grounded": sum(x["grounded"] for x in r),
                "coverage": round(sum(x["grounded"] for x in r) / n, 3),
                "via_structured": sum(1 for x in r if x["kind"] == "structured"),
                "via_retrieval": sum(1 for x in r if x["kind"] == "retrieval"), "detail": r}


# ---- source builders for the main structured domains (aliases only; inject a live or fixture backend) --------
# Each returns a StructuredSource with the canonical keys + alias phrases for a domain; pass `fetch`
# (and `fetch_series` for rate grounding) as the live API client or a committed fixture reader.

def macro_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    return StructuredSource("fred", "macro", {
        "inflation": ["inflation", "inflation rate", "cpi", "consumer price index", "price growth"],
        "unemployment": ["unemployment", "unemployment rate", "jobless rate", "joblessness"],
        "fed_funds_rate": ["fed funds rate", "policy rate", "interest rate", "federal funds rate"],
        "gdp_growth": ["gdp growth", "economic growth", "real gdp", "output growth"],
        "consumer_sentiment": ["consumer sentiment", "consumer confidence", "sentiment index"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def markets_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    return StructuredSource("markets", "markets", {
        "btc_usd": ["bitcoin price", "btc price", "bitcoin", "btc usd"],
        "sp500": ["s&p 500", "sp500", "stock market index", "s and p 500"],
        "vix": ["vix", "volatility index", "market volatility", "fear index"],
        "oil_wti": ["oil price", "crude oil", "wti", "wti crude"],
        "us10y": ["10 year treasury yield", "10y yield", "treasury yield", "bond yield"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def polls_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    return StructuredSource("polls", "politics", {
        "pres_approval": ["presidential approval", "president approval rating", "approval rating", "job approval"],
        "generic_ballot": ["generic ballot", "congressional ballot", "party vote share"],
        "right_track": ["right track wrong track", "direction of the country", "right direction"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def sports_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    return StructuredSource("sports", "sports", {
        "team_elo": ["team elo", "elo rating", "team strength", "power rating"],
        "win_pct": ["win percentage", "winning percentage", "record", "win rate"],
        "playoff_odds": ["playoff odds", "playoff probability", "chance of making the playoffs"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def product_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    return StructuredSource("product", "product", {
        "adoption": ["adoption", "adoption rate", "penetration", "market penetration", "share of users"],
        "mau": ["monthly active users", "mau", "active users", "user base"],
        "churn": ["churn", "churn rate", "cancellation rate", "attrition"],
        "conversion": ["conversion rate", "signup conversion", "conversion"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def indicators_source(fetch=None, fetch_series=None, **kw) -> StructuredSource:
    """Public-interest indicators — demography / health / energy / climate / crime (the civic long tail)."""
    return StructuredSource("indicators", "public", {
        "population": ["population", "total population", "number of people"],
        "life_expectancy": ["life expectancy", "expected lifespan"],
        "renewable_share": ["renewable energy share", "renewables share", "share of renewables"],
        "co2_emissions": ["co2 emissions", "carbon emissions", "emissions"],
        "crime_rate": ["crime rate", "violent crime rate", "homicide rate"],
        "vaccination_rate": ["vaccination rate", "immunization rate", "vaccine coverage"],
    }, fetch=fetch, fetch_series=fetch_series, **kw)


def default_sources(fetch=None, fetch_series=None, **kw) -> list:
    """The full structured menu — one call to cover macro, markets, politics, sports, product, and civic
    indicators. Inject one `fetch`/`fetch_series` (a backend that dispatches on `source:key`) or wire each."""
    return [macro_source(fetch, fetch_series, **kw), markets_source(fetch, fetch_series, **kw),
            polls_source(fetch, fetch_series, **kw), sports_source(fetch, fetch_series, **kw),
            product_source(fetch, fetch_series, **kw), indicators_source(fetch, fetch_series, **kw)]
