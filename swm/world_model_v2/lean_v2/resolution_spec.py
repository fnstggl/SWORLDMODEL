"""Deterministic resolution parser + typed terminal kinds (D4, D5).

The frozen resolution criterion is the source of truth for what YES means. The blueprint LLM may
propose a terminal, but it may never *overwrite* the resolution; if the deterministic
`ResolutionSpec` parsed from the criterion disagrees with the blueprint terminal, readiness must
fail and repair. This module is pure, deterministic, and universal — it encodes no question.

Terminal kinds (a terminal transform may only act within its own kind):

    BOOLEAN_EVENT     an event either happens by a deadline or does not (announcement, occurrence)
    INSTITUTION_VOTE  a body votes; YES is a vote/seat count or rule over members
    NUMERIC_THRESHOLD a measured numeric variable compared to a threshold in stated units
    CATEGORICAL_STATE a categorical world state equals a specific value
    FIRST_PASSAGE     a variable first crosses a level within a window
    DEADLINE_ABSENCE  the complement of a deadline-bounded event (explicit NO path)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

BOOLEAN_EVENT = "BOOLEAN_EVENT"
INSTITUTION_VOTE = "INSTITUTION_VOTE"
NUMERIC_THRESHOLD = "NUMERIC_THRESHOLD"
CATEGORICAL_STATE = "CATEGORICAL_STATE"
FIRST_PASSAGE = "FIRST_PASSAGE"
DEADLINE_ABSENCE = "DEADLINE_ABSENCE"

RESOLUTION_SPEC_VERSION = "lean_v2.resolution_spec.v1"


@dataclass
class ResolutionSpec:
    terminal_kind: str = BOOLEAN_EVENT
    measured_variable: str = ""
    unit: str = ""
    comparator: str = ">="                 # >= | > | <= | < | ==
    threshold: float = None
    threshold_units: str = ""              # "votes" | "seats" | "" (raw numeric)
    aggregation_window: str = "level"      # any_day | average | total | level
    observation_window: str = ""
    resolution_deadline: str = ""
    yes_condition: str = ""
    no_condition: str = ""
    # institution-vote specifics
    vote_rule: str = ""                    # majority | unanimity | threshold | ""
    vote_threshold: float = None           # absolute vote/seat count when stated
    vote_of_total: int = None              # N in "majority of N" / "N-of-N"
    source_text: str = ""
    parse_notes: list = field(default_factory=list)
    version: str = RESOLUTION_SPEC_VERSION

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("terminal_kind", "measured_variable", "unit", "comparator", "threshold",
                 "threshold_units", "aggregation_window", "observation_window",
                 "resolution_deadline", "yes_condition", "no_condition", "vote_rule",
                 "vote_threshold", "vote_of_total", "parse_notes", "version")}


_NUM = r"(\d+(?:\.\d+)?)"
# comparator phrasings → (comparator, is_lower_bound)
_GE = (r"(?:at least|no fewer than|no less than|greater than or equal to|"
       r"greater than or equal|minimum of|>=)\s+" + _NUM,
       r"" + _NUM + r"\s*(?:or more|or greater|or above|\+|or higher)")
_GT = r"(?:more than|greater than|above|over|exceeds?|exceeding|>)\s+" + _NUM
_LT = r"(?:fewer than|less than|under|below|<)\s+" + _NUM
_LE = (r"(?:at most|no more than|less than or equal to|maximum of|<=)\s+" + _NUM,
       r"" + _NUM + r"\s*(?:or fewer|or less)")


def _find_threshold(text: str):
    """Return (comparator, threshold) or (None, None). Order matters: bound phrasings first."""
    for pat in _GE:
        m = re.search(pat, text, re.I)
        if m:
            return ">=", float(m.group(1))
    m = re.search(_GT, text, re.I)
    if m:
        return ">", float(m.group(1))
    for pat in _LE:
        m = re.search(pat, text, re.I)
        if m:
            return "<=", float(m.group(1))
    m = re.search(_LT, text, re.I)
    if m:
        return "<", float(m.group(1))
    return None, None


def _aggregation(text: str) -> str:
    t = text.lower()
    if re.search(r"any (?:single |one )?day|on any day|single-day|daily peak", t):
        return "any_day"
    if re.search(r"average|mean|on average", t):
        return "average"
    if re.search(r"cumulativ|in total|total of|aggregate|over the (?:period|window)|summed", t):
        return "total"
    return "level"


def _vote_rule(text: str):
    """Return (rule, absolute_threshold, of_total) for vote phrasings, else (\"\",None,None)."""
    t = text.lower()
    m = re.search(r"unanimous(?:ly)?(?:\s+(\d+)\s*[-–]?\s*of\s*[-–]?\s*(\d+))?", t)
    if m or "unanimity" in t:
        of_total = int(m.group(2)) if (m and m.group(2)) else None
        return "unanimity", None, of_total
    m = re.search(r"majority of\s+(\d+)", t)
    if m:
        n = int(m.group(1))
        return "majority", None, n
    m = re.search(r"at least\s+(\d+)\s+(?:of\s+(\d+)\s+)?(?:votes|members|seats|mps)", t)
    if m:
        thr = float(m.group(1))
        of_total = int(m.group(2)) if m.group(2) else None
        return "threshold", thr, of_total
    m = re.search(r"(\d+)\s*[-–]\s*of\s*[-–]?\s*(\d+)", t)   # "5-of-9"
    if m:
        return "threshold", float(m.group(1)), int(m.group(2))
    if "simple majority" in t or re.search(r"\bmajority\b", t):
        return "majority", None, None
    return "", None, None


def _unit(text: str, variable: str) -> str:
    for u in ("tankers per day", "tankers/day", "transits per day", "transits", "tankers",
              "barrels per day", "barrels", "votes", "seats", "members", "percent", "%",
              "basis points", "bps", "usd", "dollars", "ships", "vessels", "days"):
        if u in text.lower():
            return u
    return ""


def parse_resolution(resolution_criteria: str, *, question: str = "", horizon: str = "",
                     terminal_kind_hint: str = "") -> ResolutionSpec:
    """Deterministically parse the frozen resolution criterion into a typed `ResolutionSpec`.
    Universal: no question-specific logic. `terminal_kind_hint` (e.g. the blueprint's declared
    kind) only breaks genuine ties; the parsed structure otherwise governs."""
    text = str(resolution_criteria or "")
    spec = ResolutionSpec(source_text=text[:600])
    low = text.lower()

    # deadline / window
    md = re.search(r"by\s+(\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2},? \d{4})", text)
    if md:
        spec.resolution_deadline = md.group(1)
    elif horizon:
        spec.resolution_deadline = str(horizon)[:10]

    # institution vote?
    rule, vthr, of_total = _vote_rule(text)
    comparator, threshold = _find_threshold(text)
    votish = bool(rule) or bool(re.search(r"\bvote|ballot|elect|board|parliament|council|"
                                          r"committee|members?\b", low))

    if votish and (rule or re.search(r"\bvotes?\b|\bseats?\b|\bmembers?\b", low)):
        spec.terminal_kind = INSTITUTION_VOTE
        spec.vote_rule = rule or ("threshold" if threshold is not None else "majority")
        spec.vote_threshold = vthr if vthr is not None else (
            threshold if (comparator and threshold is not None
                          and re.search(r"votes?|seats?|members?", low)) else None)
        spec.vote_of_total = of_total
        spec.threshold_units = "votes" if "vote" in low else ("seats" if "seat" in low else "")
        spec.yes_condition = text[:300]
        spec.no_condition = "the vote rule is not satisfied by the deadline"
        return spec

    # numeric threshold?
    if threshold is not None:
        spec.terminal_kind = NUMERIC_THRESHOLD
        spec.comparator = comparator
        spec.threshold = threshold
        spec.aggregation_window = _aggregation(text)
        # measured variable: the noun phrase around the number / before the comparator
        mv = re.search(r"(?:number of|count of|daily|total)?\s*([a-z][a-z \-]{2,40}?)\s*"
                       r"(?:per day|reach|reaches|of|hits|cross|exceed|at least|more than|"
                       r"fewer than|or more)", low)
        spec.measured_variable = (mv.group(1).strip() if mv else "").strip() or "measured value"
        spec.unit = _unit(text, spec.measured_variable)
        if re.search(r"first (?:time|day)|first reaches|first crosses", low):
            spec.terminal_kind = FIRST_PASSAGE
        spec.yes_condition = f"{spec.measured_variable} {comparator} {threshold} " \
                             f"({spec.aggregation_window})"
        spec.no_condition = f"{spec.measured_variable} does not {comparator} {threshold} " \
                            f"by {spec.resolution_deadline or 'the deadline'}"
        return spec

    # categorical state?
    mc = re.search(r"resolves?\s+yes\s+if\s+(.+?)\s+(?:is|equals|becomes)\s+(.+?)[\.\n]", low)
    if mc:
        spec.terminal_kind = CATEGORICAL_STATE
        spec.measured_variable = mc.group(1).strip()
        spec.yes_condition = f"{mc.group(1).strip()} == {mc.group(2).strip()}"
        spec.no_condition = "the categorical state does not equal the YES value"
        return spec

    # default: boolean deadline-bounded event
    spec.terminal_kind = terminal_kind_hint if terminal_kind_hint in (
        BOOLEAN_EVENT, DEADLINE_ABSENCE) else BOOLEAN_EVENT
    spec.yes_condition = text[:300] or "the event occurs by the deadline"
    spec.no_condition = "the event does not occur by the deadline (event_absent)"
    spec.parse_notes.append("no numeric threshold or vote rule parsed → boolean deadline event")
    return spec


def spec_matches_blueprint(spec: ResolutionSpec, bp) -> dict:
    """Compare the parsed ResolutionSpec against the blueprint terminal. Returns
    {ok, mismatches} — a mismatch means readiness must fail and repair (the blueprint may not
    overwrite the resolution)."""
    term = bp.terminal or {}
    tk = str(term.get("kind") or "")
    mism = []
    kind_map = {"institution_vote": INSTITUTION_VOTE, "event_occurs": BOOLEAN_EVENT,
                "state_predicate": None}   # predicate may be numeric or categorical
    expected = kind_map.get(tk, None)
    if spec.terminal_kind == INSTITUTION_VOTE and tk != "institution_vote":
        mism.append(f"resolution is an institution vote but blueprint terminal kind is '{tk}'")
    if spec.terminal_kind == NUMERIC_THRESHOLD:
        # the blueprint must carry the numeric threshold, not a boolean flag
        rp = term.get("rule_params") or {}
        bp_thr = rp.get("threshold") or term.get("threshold")
        if bp_thr is None:
            mism.append("resolution is a numeric threshold but blueprint terminal carries no "
                        "numeric threshold (would collapse to a boolean event)")
    return {"ok": not mism, "mismatches": mism, "spec_kind": spec.terminal_kind,
            "blueprint_kind": tk, "expected_kind": expected}
