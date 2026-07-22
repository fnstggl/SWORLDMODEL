"""Faithful world representation (D7) — the institution a terminal vote runs through must be
represented with its REAL decision-makers and voting power, never collapsed into a handful of
equal votes and never "fixed" by rescaling the real threshold.

The EXP-113 failures this eliminates:
  * BoJ: a 9-member board modeled as 5 units (4 principals + one `other_members` bloc casting a
    single vote); the real "5 of 9" majority silently rescaled to "≥3 of 5".
  * Wale: a 50-seat parliament modeled as 5 "members" that were the rival candidates themselves,
    so the electorate became the candidates and the majority coalition was one bloc vote; "26 of
    50" rescaled to "≥3 of 5".

The typed representation makes the real arithmetic explicit and REQUIRES it to reconcile:

    real voting power  ==  represented voting power  ==  the terminal threshold's denominator

A `DecisionUnit` is either an INDIVIDUAL (one seat, one vote) or a seat-weighted BLOC (N seats,
emitting a DISTRIBUTION/COUNT of member votes — never one ordinary vote). Candidates are typed
separately from voters, so a candidate is never mistaken for the electorate. When the modeled
roster does not reconcile with the real body, `repair_representation` expands the roster (small
bodies → every seat as an individual unit; large bodies → seat-weighted blocs) rather than
rescaling reality. If it still cannot reconcile, readiness fails.

Universal: nothing here is question-specific. The real body size and threshold denominator are
read from the frozen resolution (`ResolutionSpec.vote_of_total`) and, failing that, deterministic
`N-member` / `N-seat` / `N members` phrases in the sealed evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm_key

REPRESENTATION_VERSION = "lean_v2.representation.v1"

#: at or below this modeled size, individual identities matter and every seat is an INDIVIDUAL
#: voting unit; above it, the body is represented by seat-weighted blocs. This is a fidelity
#: boundary, NOT a decisive-actor cap — no real voter is ever deleted; a large body keeps its
#: full seat count via blocs.
INDIVIDUAL_BODY_MAX = 15

# ------------------------------------------------------------------ roles
CANDIDATE = "candidate"
VOTER = "voter"
ADVISER = "adviser"
OBSERVER = "observer"
LEADER = "leader"
COALITION_LEADER = "coalition_leader"


@dataclass
class DecisionUnit:
    unit_id: str
    kind: str = "individual"                 # individual | bloc
    member_ids: list = field(default_factory=list)   # modeled actor ids this unit represents
    seat_count: int = 1                      # voting power (individual=1; bloc=N seats)
    roles: list = field(default_factory=list)
    is_candidate: bool = False
    is_voter: bool = True                    # participates in the terminal tally
    represents_label: str = ""
    provenance: str = ""

    @property
    def voting_power(self) -> int:
        return self.seat_count if self.is_voter else 0

    def as_dict(self) -> dict:
        return {"unit_id": self.unit_id, "kind": self.kind, "member_ids": list(self.member_ids),
                "seat_count": self.seat_count, "roles": list(self.roles),
                "is_candidate": self.is_candidate, "is_voter": self.is_voter,
                "voting_power": self.voting_power, "represents_label": self.represents_label,
                "provenance": self.provenance}


@dataclass
class WorldRepresentationSpec:
    institution_id: str = ""
    real_member_count: int = None            # true body size (seats)
    real_member_count_source: str = ""
    represented_voting_power: int = 0        # sum of voter seat_counts
    decision_units: list = field(default_factory=list)   # DecisionUnit
    rule: str = "majority"                   # majority | unanimity | threshold | single | all_option
    threshold: float = None                  # absolute votes/seats needed for YES
    threshold_units: str = "votes"           # votes | seats
    target_option: str = ""                  # the option that must reach the threshold (if any)
    quorum: int = None
    candidates: list = field(default_factory=list)       # unit_ids that are candidates
    faithful: bool = False
    verdict: str = "not_ready"               # ready | repairable | not_ready
    repairs: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    version: str = REPRESENTATION_VERSION

    def voter_units(self) -> list:
        return [u for u in self.decision_units if u.is_voter]

    def total_voting_power(self) -> int:
        return sum(u.voting_power for u in self.decision_units)

    def as_dict(self) -> dict:
        return {"institution_id": self.institution_id,
                "real_member_count": self.real_member_count,
                "real_member_count_source": self.real_member_count_source,
                "represented_voting_power": self.represented_voting_power,
                "total_voting_power": self.total_voting_power(),
                "decision_units": [u.as_dict() for u in self.decision_units],
                "rule": self.rule, "threshold": self.threshold,
                "threshold_units": self.threshold_units, "target_option": self.target_option,
                "quorum": self.quorum, "candidates": list(self.candidates),
                "faithful": self.faithful, "verdict": self.verdict,
                "repairs": self.repairs, "diagnostics": self.diagnostics,
                "version": self.version}


# ------------------------------------------------------------------ real body size inference
_SIZE_PATTS = [
    r"(\d+)\s*[- ]?(?:member|seat|person|strong)\b",
    r"(?:all|of the|the)\s+(\d+)\s+(?:members|seats|lawmakers|mps|legislators|governors|"
    r"directors|justices|board members)",
    r"(?:board|council|committee|parliament|assembly|panel|bench)\s+of\s+(\d+)",
    r"(\d+)\s*[- ]?(?:vote|votes)\s+(?:body|chamber|house|parliament)",
]


def infer_real_body_size(resolution_spec, evidence_text: str, grounding: dict = None):
    """(count, source) for the real body size. Resolution `vote_of_total` first (authoritative:
    'majority of 9', '26 of 50'), then deterministic 'N-member'/'N-seat' phrases in the sealed
    evidence, then None. Never guessed."""
    if getattr(resolution_spec, "vote_of_total", None):
        return int(resolution_spec.vote_of_total), "resolution:vote_of_total"
    ev = str(evidence_text or "")
    best = None
    for patt in _SIZE_PATTS:
        for m in re.finditer(patt, ev, re.I):
            n = int(m.group(1))
            if 2 <= n <= 1000:
                best = n if best is None else max(best, n)   # prefer the largest stated body
    if best is not None:
        return best, "evidence:body_size_phrase"
    return None, "unavailable"


# ------------------------------------------------------------------ role inference (typed)
def _looks_like_candidate(actor: dict, question: str, target_option: str) -> bool:
    """A modeled actor is a candidate when the terminal outcome is about a PERSON taking office
    and this actor is that person or a named rival option. Deterministic, general — driven by
    the actor's name appearing as a terminal option / in the question's office phrasing."""
    name = norm_key(actor.get("name") or actor.get("id"))
    if not name:
        return False
    role = norm_key(actor.get("role"))
    if any(k in role for k in ("candidate", "contender", "nominee", "aspirant",
                               "leadership contender", "prime minister candidate")):
        return True
    # the actor's own name is a vote option → they are being voted ON (a candidate), which means
    # they are not merely one elector among the body
    if target_option and name and name in norm_key(target_option):
        return True
    return False


def _safe_threshold(raw):
    """Parse a rule_params threshold that may be a number or a phrase (">50%", "majority of 9",
    "5"). Returns a float (a percentage becomes its fraction) or None — never raises. A fraction
    <1 is an explicit share; a bare integer is an absolute vote/seat count."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    s = str(raw or "").strip()
    if not s or any(k in s.lower() for k in ("majority", "unanimity", "consensus", "none",
                                             "half", "plurality")):
        return None                                # a rule word, not an absolute count
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:                                          # a percentage → its fraction
        return float(m.group(1)) / 100.0
    m = re.search(r"\d+(?:\.\d+)?", s)             # else the first bare number (absolute count)
    return float(m.group(0)) if m else None


def _bloc_seat_count(unit_label: str, real_total: int, individual_seats: int,
                     n_blocs: int) -> int:
    """Seats a bloc represents = (real_total − individually-modeled seats) shared across blocs.
    A bloc NEVER collapses to one vote."""
    remaining = max(0, (real_total or 0) - individual_seats)
    if n_blocs <= 0:
        return 0
    return max(1, remaining // n_blocs)


# ------------------------------------------------------------------ build
def build_representation(bp, resolution_spec, *, evidence_text: str = "",
                         grounding: dict = None) -> WorldRepresentationSpec:
    """Construct the faithful representation for the terminal institution. Pure/deterministic.
    Does NOT rescale anything — it records the real vs represented arithmetic and (via
    validate/repair) reconciles them or fails."""
    term = bp.terminal or {}
    spec = WorldRepresentationSpec()
    if str(term.get("kind") or "") != "institution_vote":
        spec.faithful = True
        spec.verdict = "ready"
        spec.diagnostics.append("non-vote terminal: representation not applicable")
        return spec
    inst = bp.institution_by_id(term.get("institution_id")) or {}
    spec.institution_id = inst.get("id") or term.get("institution_id") or ""
    spec.rule = str(term.get("decision_rule") or inst.get("decision_rule") or "majority")
    rp = term.get("rule_params") or {}
    spec.target_option = str(rp.get("option") or "")
    # the threshold may be a bare number, or a phrase like ">50%", "majority", "5 of 9" — parse the
    # numeric part safely (a fraction stays a fraction; a bare integer stays absolute); never crash
    vt = getattr(resolution_spec, "vote_threshold", None)
    spec.threshold = float(vt) if isinstance(vt, (int, float)) else _safe_threshold(rp.get("threshold"))
    spec.threshold_units = getattr(resolution_spec, "threshold_units", "") or "votes"
    q = getattr(resolution_spec, "vote_of_total", None)
    real, src = infer_real_body_size(resolution_spec, evidence_text, grounding)
    spec.real_member_count = real
    spec.real_member_count_source = src

    question = getattr(bp, "question", "") or bp.resolution.get("interpretation", "")
    modeled = [bp.actor_by_id(m) or {"id": m, "name": m} for m in (inst.get("members") or [])]
    # classify candidates vs voters
    for a in modeled:
        is_cand = _looks_like_candidate(a, question, spec.target_option)
        roles = [CANDIDATE] if is_cand else [VOTER]
        u = DecisionUnit(unit_id=a.get("id"), kind="individual", member_ids=[a.get("id")],
                         seat_count=1, roles=roles, is_candidate=is_cand,
                         is_voter=not is_cand,   # a pure candidate is not the electorate
                         represents_label=a.get("name") or a.get("id"),
                         provenance="modeled_actor")
        spec.decision_units.append(u)
        if is_cand:
            spec.candidates.append(u.unit_id)
    spec.represented_voting_power = spec.total_voting_power()
    return spec


# ------------------------------------------------------------------ validate
def validate_representation(spec: WorldRepresentationSpec) -> WorldRepresentationSpec:
    """Reconcile real vs represented voting power vs the threshold denominator. Sets
    verdict ∈ {ready, repairable, not_ready}. Never rescales the threshold."""
    d = spec.diagnostics
    if not spec.institution_id:
        spec.verdict, spec.faithful = "ready", True
        return spec
    rep_power = spec.total_voting_power()
    spec.represented_voting_power = rep_power
    real = spec.real_member_count
    # if we have no real size, we cannot prove fidelity — but we can proceed only when the
    # modeled roster already looks whole (no candidate-as-electorate and threshold satisfiable)
    if real is None:
        if spec.threshold is not None and spec.threshold > rep_power:
            d.append(f"threshold {spec.threshold} exceeds represented voting power {rep_power} "
                     f"and the real body size is unknown — cannot reconcile")
            spec.verdict = "repairable"
            return spec
        d.append("real body size unavailable; proceeding on the modeled roster (no rescale)")
        spec.verdict = "ready" if _electorate_ok(spec, d) else "repairable"
        spec.faithful = spec.verdict == "ready"
        return spec
    # we know the real body size — represented voting power MUST equal it
    if rep_power != real:
        d.append(f"represented voting power {rep_power} != real body size {real} "
                 f"(roster collapse) — repair required, threshold must NOT be rescaled")
        spec.verdict = "repairable"
        return spec
    if spec.threshold is not None and spec.threshold > real:
        d.append(f"threshold {spec.threshold} exceeds real body size {real} — invalid")
        spec.verdict = "not_ready"
        return spec
    spec.verdict = "ready" if _electorate_ok(spec, d) else "repairable"
    spec.faithful = spec.verdict == "ready"
    return spec


def _electorate_ok(spec: WorldRepresentationSpec, diagnostics: list) -> bool:
    """The electorate must contain real voters, not only candidates."""
    voters = spec.voter_units()
    if not voters:
        diagnostics.append("no voter units — the electorate was modeled entirely as "
                           "candidates/non-voters")
        return False
    # if EVERY modeled unit is a candidate, the electorate is missing
    if all(u.is_candidate for u in spec.decision_units):
        diagnostics.append("every modeled unit is a candidate — the electorate is absent")
        return False
    return True


# ------------------------------------------------------------------ repair
def repair_representation(spec: WorldRepresentationSpec) -> WorldRepresentationSpec:
    """Reconcile the roster to the real body WITHOUT rescaling the threshold. Small bodies gain
    individual unnamed-seat units for the missing members; large bodies are represented by
    seat-weighted blocs; a body modeled entirely as candidates gains a real electorate bloc."""
    real = spec.real_member_count
    if spec.verdict == "ready":
        return spec
    if real is None:
        # can't repair a size we don't know — but we can at least give a candidate-only body a
        # neutral electorate so the vote is decided by voters, not by candidates voting for self
        if spec.decision_units and all(u.is_candidate for u in spec.decision_units):
            spec.decision_units.append(_electorate_bloc(spec, seats=max(1, len(spec.candidates)),
                                                        label="electorate (size unknown)"))
            spec.repairs.append("added an electorate bloc: a candidate-only roster cannot elect "
                                "itself")
        spec.represented_voting_power = spec.total_voting_power()
        return validate_representation(spec)

    rep_power = spec.total_voting_power()
    # separate the candidates (kept, non-voting) from the modeled voters
    modeled_voter_seats = sum(u.voting_power for u in spec.decision_units)
    missing = real - modeled_voter_seats
    if missing <= 0:
        spec.represented_voting_power = spec.total_voting_power()
        return validate_representation(spec)

    if real <= INDIVIDUAL_BODY_MAX:
        # small body: model every missing member as its own INDIVIDUAL unnamed-seat voter —
        # never a single bloc that casts one vote for 4 people (the BoJ collapse)
        for i in range(missing):
            spec.decision_units.append(DecisionUnit(
                unit_id=f"__seat_{spec.institution_id}_{i}", kind="individual",
                member_ids=[], seat_count=1, roles=[VOTER], is_voter=True,
                represents_label=f"unmodeled member seat {i + 1} of {real}",
                provenance="roster_repair:individual_seat"))
        spec.repairs.append(f"expanded the roster: added {missing} individual voter seats so the "
                            f"body has its real {real} members (small-body fidelity)")
    else:
        # large body: represent the missing seats as seat-weighted BLOC(s) that emit a
        # DISTRIBUTION of member votes weighted by seat_count — never one ordinary vote. Reuse an
        # existing modeled bloc-like voter if present, else create one.
        bloc = DecisionUnit(
            unit_id=f"__bloc_{spec.institution_id}", kind="bloc", member_ids=[],
            seat_count=int(missing), roles=[VOTER], is_voter=True,
            represents_label=f"remaining {missing} of {real} seats (seat-weighted bloc)",
            provenance="roster_repair:seat_weighted_bloc")
        spec.decision_units.append(bloc)
        spec.repairs.append(f"expanded the roster: added a seat-weighted bloc of {missing} seats "
                            f"(large-body fidelity) — it emits a vote distribution over its "
                            f"seats, never one vote")
    spec.represented_voting_power = spec.total_voting_power()
    return validate_representation(spec)


def _electorate_bloc(spec, *, seats: int, label: str) -> DecisionUnit:
    return DecisionUnit(unit_id=f"__electorate_{spec.institution_id}", kind="bloc",
                        member_ids=[], seat_count=int(seats), roles=[VOTER], is_voter=True,
                        represents_label=label, provenance="roster_repair:electorate")


def ensure_faithful_representation(bp, resolution_spec, *, evidence_text: str = "",
                                   grounding: dict = None) -> WorldRepresentationSpec:
    """Build → validate → (repair → re-validate). The one entry point. Returns a spec whose
    verdict the readiness gate consumes; NEVER rescales the terminal threshold."""
    spec = build_representation(bp, resolution_spec, evidence_text=evidence_text,
                                grounding=grounding)
    spec = validate_representation(spec)
    if spec.verdict == "repairable":
        spec = repair_representation(spec)
    spec.faithful = spec.verdict == "ready"
    return spec
