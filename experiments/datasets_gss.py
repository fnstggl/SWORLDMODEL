"""General Social Survey (GSS) — longitudinal population opinion, 1972-2024 (the multi-step substrate).

OpinionQA is a single cross-section; to test the untested thesis axis — simulate a POPULATION FORWARD
over time and predict opinion CHANGE — we need the same questions asked of fresh cross-sections across
many years, with individual demographics each wave. The GSS is the canonical source: 75,699 respondents
over 34 survey years, a stable core of demographic + attitude items, purpose-built for time-trend study
(NORC, cumulative file 1972-2024 Release 3).

This loader reads only the needed columns from the Stata cumulative file (pyreadstat, offline) and writes
a compact per-respondent JSON cache — the same pattern as the OpinionQA loader (parse once, commit the
cache, keep runtime pure-Python). Each record: {uid, year, demo{...}, answers{item: 0/1}}.

Download (once, ~48 MB zip -> 598 MB .dta; gitignored, not committed):
  curl -sSL -o data/GSS_stata.zip \
    https://www.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip
  unzip -o data/GSS_stata.zip gss7224_r3.dta -d data/
The parsed cache under experiments/results/exp045_gss/ makes the experiment reproducible without the .dta.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

DTA = "data/gss7224_r3.dta"
CACHE = "experiments/results/exp045_gss/gss_parsed.json.gz"

# attitude items -> a rule mapping the raw GSS code to a binary "1" pole (else 0); None/other codes drop.
# favor/oppose & yes/no items: 1 is the affirmative pole. nat* spending items: 1 == "too little" (spend more).
_ITEMS = {
    "cappun": lambda v: 1 if v == 1 else (0 if v == 2 else None),      # favor death penalty
    "gunlaw": lambda v: 1 if v == 1 else (0 if v == 2 else None),      # favor gun permit
    "grass": lambda v: 1 if v == 1 else (0 if v == 2 else None),       # marijuana should be legal
    "abany": lambda v: 1 if v == 1 else (0 if v == 2 else None),       # abortion for any reason
    "letdie1": lambda v: 1 if v == 1 else (0 if v == 2 else None),     # allow incurable to die
    "fepol": lambda v: 1 if v == 1 else (0 if v == 2 else None),       # agree women not suited for politics
    "fefam": lambda v: 1 if v in (1, 2) else (0 if v in (3, 4) else None),   # agree women stay home
    "homosex": lambda v: 1 if v == 1 else (0 if v in (2, 3, 4) else None),   # homosexuality always wrong
    "premarsx": lambda v: 1 if v in (1, 2) else (0 if v in (3, 4) else None),  # premarital sex wrong
    "natheal": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),      # spend more on health
    "natenvir": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),     # spend more on environment
    "natfare": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),      # spend more on welfare
    "natcrime": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),     # spend more on crime
    "nateduc": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),      # spend more on education
    "natrace": lambda v: 1 if v == 1 else (0 if v in (2, 3) else None),      # spend more on Black Americans
}

_DEMO_COLS = ["age", "sex", "race", "region", "degree", "partyid", "polviews", "relig", "attend",
              "marital", "income"]


def _age(v):
    if v is None:
        return "unknown"
    v = float(v)
    return "18-29" if v < 30 else ("30-49" if v < 50 else ("50-64" if v < 65 else "65+"))


def _party(v):
    if v is None:
        return "unknown"
    v = int(v)
    return "democrat" if v <= 2 else ("independent" if v == 3 else ("republican" if v <= 6 else "unknown"))


def _polviews(v):
    if v is None:
        return "unknown"
    v = int(v)
    return "liberal" if v <= 3 else ("moderate" if v == 4 else ("conservative" if v <= 7 else "unknown"))


def _cat(v, mapping):
    return mapping.get(int(v), "unknown") if v is not None else "unknown"


_SEX = {1: "male", 2: "female"}
_RACE = {1: "white", 2: "black", 3: "other"}
_DEGREE = {0: "lt_highschool", 1: "highschool", 2: "junior_college", 3: "bachelor", 4: "graduate"}
_RELIG = {1: "protestant", 2: "catholic", 3: "jewish", 4: "none", 5: "other"}
_MARITAL = {1: "married", 2: "widowed", 3: "divorced", 4: "separated", 5: "never_married"}


def _attend(v):
    if v is None:
        return "unknown"
    v = int(v)
    return "low" if v <= 1 else ("medium" if v <= 4 else "high")


def parse(subsample=None, seed=0):
    import pyreadstat
    cols = ["year", "id"] + _DEMO_COLS + list(_ITEMS)
    df, _ = pyreadstat.read_dta(DTA, usecols=cols, encoding="latin1")
    recs = []

    def g(row, c):
        v = row[c]
        try:
            import math
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
        except Exception:
            pass
        return v

    for _, row in df.iterrows():
        yr = g(row, "year")
        if yr is None:
            continue
        demo = {"age": _age(g(row, "age")), "sex": _cat(g(row, "sex"), _SEX),
                "race": _cat(g(row, "race"), _RACE), "region": str(g(row, "region") or "unknown"),
                "degree": _cat(g(row, "degree"), _DEGREE), "party": _party(g(row, "partyid")),
                "ideology": _polviews(g(row, "polviews")), "relig": _cat(g(row, "relig"), _RELIG),
                "attendance": _attend(g(row, "attend")), "marital": _cat(g(row, "marital"), _MARITAL),
                "income": str(int(g(row, "income"))) if g(row, "income") is not None else "unknown"}
        answers = {}
        for item, rule in _ITEMS.items():
            v = g(row, item)
            b = rule(int(v)) if v is not None else None
            if b is not None:
                answers[item] = b
        if answers:
            recs.append({"uid": f"{int(yr)}_{int(g(row, 'id') or 0)}", "year": int(yr),
                         "demo": demo, "answers": answers})
    if subsample and len(recs) > subsample:
        import random
        rng = random.Random(seed); rng.shuffle(recs); recs = recs[:subsample]
    return recs


def build_cache(subsample=None):
    recs = parse(subsample=subsample)
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt", encoding="utf-8") as f:
        json.dump(recs, f)
    print(f"wrote {len(recs)} GSS respondent-records -> {CACHE}")
    return recs


def load():
    if Path(CACHE).exists():
        with gzip.open(CACHE, "rt", encoding="utf-8") as f:
            return json.load(f)
    return build_cache()


if __name__ == "__main__":
    build_cache()
