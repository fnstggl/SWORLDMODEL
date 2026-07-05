"""Map a survey respondent's demographics -> the same value-variables the world model conditions on.

This is the VariableMap architecture applied to a person described by structured demographics (the
OpinionQA / Pew ATP setting): the KNOWN attributes (age, religion, ideology, income, …) are mapped to
the INFERRED latent value-variables (religiosity, traditionalism, individualism, …) that actually drive
opinions — the same 10 value dimensions used for population opinion in EXP-023. The mapping is a
transparent, grounded heuristic (Inglehart value axes, standard political-science associations), so it
is non-circular (it never sees the survey answer) and interpretable.

Returns a dict {value_dim: float}. Unsigned dims are in [0,1]; signed dims (economic_left) in [-1,1].
"""
from __future__ import annotations

VALUE_DIMS = ["religiosity", "traditionalism", "individualism", "trust_institutions", "openness_change",
              "national_pride", "economic_left", "social_progressive", "hierarchy_respect",
              "survival_vs_selfexpression"]

# ordinal encodings of demographic levels onto a 0..1 scale (unknown -> None => neutral)
_RELIGION = {"atheist": 0.02, "agnostic": 0.12, "nothing in particular": 0.2, "unitarian": 0.35,
             "jewish": 0.45, "buddhist": 0.4, "hindu": 0.55, "roman catholic": 0.65, "orthodox": 0.7,
             "protestant": 0.72, "mormon": 0.85, "muslim": 0.85, "other": 0.5}
_ATTEND = {"never": 0.0, "seldom": 0.25, "a few times a year": 0.4, "once or twice a month": 0.6,
           "once a week": 0.85, "more than once a week": 1.0}
_IDEOLOGY = {"very liberal": 0.0, "liberal": 0.2, "moderate": 0.5, "conservative": 0.8,
             "very conservative": 1.0}   # 0 = left, 1 = right
_PARTY = {"democrat": 0.1, "independent": 0.5, "other": 0.5, "republican": 0.9}
_AGE = {"18-29": 0.1, "30-49": 0.4, "50-64": 0.7, "65+": 0.95}   # older = higher
_EDU = {"less than high school": 0.0, "high school graduate": 0.25, "some college, no degree": 0.45,
        "associate's degree": 0.55, "college graduate/some postgrad": 0.8, "postgraduate": 1.0}
_INCOME = {"less than $30,000": 0.1, "$30,000-$50,000": 0.35, "$50,000-$75,000": 0.55,
           "$75,000-$100,000": 0.75, "$100,000 or more": 1.0}


def _lvl(table, key, default=0.5):
    return table.get((key or "").strip(), default)


def _clip01(x):
    return max(0.0, min(1.0, x))


def demographic_to_values(demo: dict) -> dict:
    """Structured demographics -> the 10 value-variables (the on-architecture latent profile)."""
    rel = _lvl(_RELIGION, demo.get("religion"), 0.5)
    att = _lvl(_ATTEND, demo.get("attendance"), 0.4)
    ideo = _lvl(_IDEOLOGY, demo.get("ideology"), 0.5)          # 0 left .. 1 right
    party = _lvl(_PARTY, demo.get("party"), 0.5)
    age = _lvl(_AGE, demo.get("age"), 0.5)
    edu = _lvl(_EDU, demo.get("education"), 0.5)
    inc = _lvl(_INCOME, demo.get("income"), 0.5)

    religiosity = _clip01(0.6 * rel + 0.4 * att)
    right = 0.5 * ideo + 0.5 * party                            # composite left-right
    return {
        "religiosity": religiosity,
        "traditionalism": _clip01(0.45 * right + 0.3 * religiosity + 0.25 * age),
        "individualism": _clip01(0.5 * right + 0.25 * inc + 0.25 * (1 - religiosity)),
        "trust_institutions": _clip01(0.5 + 0.3 * (edu - 0.5)),   # weak: education-linked, else neutral
        "openness_change": _clip01(0.4 * (1 - right) + 0.3 * (1 - age) + 0.3 * edu),
        "national_pride": _clip01(0.55 * right + 0.2 * age + 0.25 * religiosity),
        "economic_left": max(-1.0, min(1.0, 2 * (0.5 * (1 - right) + 0.3 * (1 - inc) + 0.2 * (1 - edu)) - 1)),
        "social_progressive": _clip01(0.45 * (1 - right) + 0.3 * (1 - religiosity) + 0.25 * (1 - age)),
        "hierarchy_respect": _clip01(0.45 * right + 0.3 * religiosity + 0.25 * age),
        "survival_vs_selfexpression": _clip01(0.4 * inc + 0.3 * edu + 0.3 * (1 - age)),
    }


def value_vector(demo: dict) -> list:
    v = demographic_to_values(demo)
    return [float(v[d]) for d in VALUE_DIMS]
