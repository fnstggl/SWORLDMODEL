"""Deterministic causal-scale classification + benchmark composition (no LLM anywhere —
a frontier model must not perform case-dependent selection, and the period model is not needed
for a frozen keyword rubric)."""
from __future__ import annotations

import re

SCALES = ("single_decision_maker", "small_group_decision", "multi_actor_strategic",
          "institutional_process", "broad_aggregate", "mixed_scale")

# frozen keyword rubric — first matching category wins within each tier; mixed_scale triggers
# when both an elite-decision cue and an aggregate cue appear
_ELITE = ("president", "prime minister", "ceo", "chancellor", "governor", "judge", "justice",
          "chair", "pope", "musk", "trump", "putin", "zelensky", " xi ", "biden", "harris",
          "netanyahu", "resign", "veto", "pardon", "nominate", "appoint", " fire ", " fired",
          "step down", "sign the", "sign into law", "announce", "candidate")
_SMALL_GROUP = ("supreme court", "court rule", "board", "cabinet", "committee", "federal reserve",
                "fomc", "ftc", " sec ", "doj", "opec", "security council", "confirm", "ruling",
                "rate cut", "rate hike", "cut rates", "raise rates", "central bank")
_INSTITUTIONAL = ("congress", "senate", "house of representatives", "parliament", "legislation",
                  " bill ", " act ", " law ", "approve", "approval", "ratif", "regulator", "fda",
                  "european union", " eu ", "impeach", "government shutdown", "budget", "treaty",
                  "amendment", " pass ")
_MULTI_ACTOR = ("ceasefire", "war", "peace", "invade", "invasion", "strike ", "attack",
                "negotiat", "agreement between", "merger", "acquisition", "acquire", "nato",
                "sanction", "conflict", "hostage", "deal", "summit", "talks", "alliance", "ban ",
                "banned", "tariff")
_AGGREGATE = ("election", "win the", "votes", "poll", "turnout", "recession", "inflation", "gdp",
              "unemployment", "sales exceed", "more than", "at least", "exceed", "adoption",
              "users", "population", "countries", "recognize", "membership", "market share",
              "percent", "%", "million", "billion", "majority of")


def _hit(q: str, toks) -> bool:
    return any(t in q for t in toks)


def classify_scale(question: str) -> tuple:
    """(primary, secondary[]) — deterministic, frozen before selection."""
    q = " " + str(question).lower() + " "
    elite, small = _hit(q, _ELITE), _hit(q, _SMALL_GROUP)
    inst, multi, agg = _hit(q, _INSTITUTIONAL), _hit(q, _MULTI_ACTOR), _hit(q, _AGGREGATE)
    secondary = [s for s, h in (("single_decision_maker", elite), ("small_group_decision", small),
                                ("institutional_process", inst), ("multi_actor_strategic", multi),
                                ("broad_aggregate", agg)) if h]
    if (elite or small or inst) and agg:
        primary = "mixed_scale"
    elif small:
        primary = "small_group_decision"
    elif inst:
        primary = "institutional_process"
    elif elite:
        primary = "single_decision_maker"
    elif multi:
        primary = "multi_actor_strategic"
    elif agg:
        primary = "broad_aggregate"
    else:
        primary = "multi_actor_strategic"                     # conservative default for geopolitics
    return primary, [s for s in secondary if s != primary]


# frozen BEFORE selection (protocol): minimum primary-category representation in the 100
QUOTAS = {"single_decision_maker": 15, "small_group_decision": 15, "multi_actor_strategic": 20,
          "institutional_process": 15, "broad_aggregate": 20, "mixed_scale": 15}

_EXCLUDE_TOKENS = (
    # sports / games
    " nba ", " nfl ", " mlb ", " nhl ", " ufc ", "premier league", "champions league",
    "world cup", "grand slam", "wimbledon", "super bowl", "playoff", "vs.", " vs ", "match",
    "tournament", "olympic", "f1 ", "grand prix", "world series", "espn", "score",
    # pure price thresholds
    "bitcoin", " btc", " eth", "ethereum", "solana", " doge", "crypto", "price of", "stock",
    " s&p", "nasdaq", "all-time high", "market cap", "$", "strategic reserve of",
    # celebrity trivia / entertainment metrics
    "taylor swift", "album", "box office", "spotify", "grammy", "oscars", "mrbeast",
    # mechanical calendar
    "mention", "tweet", "say the word", "post ", "times will")


def excluded(question: str) -> str | None:
    q = " " + str(question).lower() + " "
    for t in _EXCLUDE_TOKENS:
        if t in q:
            return f"excluded_token:{t.strip()}"
    return None


def proper_nouns(question: str, k: int = 8) -> list:
    """Deterministic entity extraction for query generation (no LLM): capitalized multi/single
    word runs, minus leading interrogatives."""
    stop = {"Will", "Which", "What", "When", "Who", "How", "Before", "By", "The", "A", "An",
            "In", "On", "At", "Of", "For", "To", "And", "Or", "Is", "Are", "Does", "Do", "Yes",
            "No", "Not", "After", "During"}
    runs = re.findall(r"(?:[A-Z][\w&.'-]*)(?:\s+(?:of|the|and|for|de|von|al)?\s*[A-Z][\w&.'-]*)*",
                      str(question))
    out, seen = [], set()
    for r in runs:
        words = [w for w in r.strip().split() if w not in stop]   # strip stopwords anywhere
        cand = " ".join(words).strip()
        if len(cand) >= 3 and cand.lower() not in seen:
            seen.add(cand.lower())
            out.append(cand)
    return out[:k]


def composition_report(cases: list) -> dict:
    by = lambda key: {v: sum(1 for c in cases if c.get(key) == v)  # noqa: E731
                      for v in sorted({c.get(key) for c in cases})}
    horizons = [round((c["resolution_deadline_ts"] - c["question_open_ts"]) / 86400.0)
                for c in cases]
    return {"n": len(cases), "by_causal_scale": by("causal_scale"), "by_domain": by("domain"),
            "by_split": by("split"),
            "horizon_days": {"min": min(horizons), "median": sorted(horizons)[len(horizons) // 2],
                             "max": max(horizons)},
            "quotas_frozen": QUOTAS,
            "quota_satisfied": {s: sum(1 for c in cases if c.get("causal_scale") == s) >= q
                                for s, q in QUOTAS.items()}}
