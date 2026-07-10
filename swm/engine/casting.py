"""Casting — Stage 1: construct the belief state by identifying WHO decides and WHAT the answer space is.

The LLM's job here is deliberately NOT to invent variables-with-weights (the old compiler's sin). It is to
answer three questions a director would ask:
  - What social process is this? (one person reacting / a collective choosing among options / a population
    adopting / an artifact being optimized against an audience)
  - Who are the REAL actors — named individuals (the actual candidates, justices, the recipient) or
    representative segments (voter cells with weights)? Actors must come from the grounded dossier; a cast
    of abstractions is rejected.
  - What is the NATIVE answer space — the literal set of named options, a yes/no on a concrete event, or
    real artifact texts to be generated and ranked? The simulation's state space must BE the answer space,
    so the output always answers the actual question.

Casting also fixes REAL TIME: the resolution date (from the dossier when the evidence names one) and the
cadence at which the situation plausibly changes — so a roll-forward of one week simulates one week.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.engine.grounding import parse_json

PROCESSES = ("individual_reaction", "collective_choice", "population_share", "artifact_optimization")


@dataclass
class Actor:
    name: str                          # a real person ("Brad Lander") or a concrete segment
    kind: str = "named"                # named | segment
    weight: float = 1.0                # segment share / actor influence on the outcome
    role: str = ""                     # why this actor matters (from the dossier)
    n_variants: int = 1                # diversity: concrete persona draws for a segment


@dataclass
class Cast:
    process: str
    answer_space: dict                 # {"type": "named_options"|"binary"|"artifacts", "options": [...]}
    actors: list = field(default_factory=list)
    horizon_days: float = 30.0
    resolve_by: str = ""               # real date when the outcome is known, if the evidence names one
    cadence_days: float = 7.0          # how often the public state meaningfully updates
    interaction: str = ""              # who observes whom (drives the between-round public signal)
    rationale: str = ""

    def options(self) -> list:
        return list(self.answer_space.get("options", []) or [])

    def as_dict(self) -> dict:
        return {"process": self.process, "answer_space": self.answer_space,
                "actors": [{"name": a.name, "kind": a.kind, "weight": round(a.weight, 4),
                            "role": a.role, "n_variants": a.n_variants} for a in self.actors],
                "horizon_days": self.horizon_days, "resolve_by": self.resolve_by,
                "cadence_days": self.cadence_days, "interaction": self.interaction,
                "rationale": self.rationale}


_CAST_PROMPT = """You are the CASTING DIRECTOR of a social world simulation. Do not forecast. Identify the
real social process, the real actors, and the answer space, from the grounded evidence ONLY.

QUESTION: {q}
TODAY: {today}
GROUNDED SCENE (cited facts — the only actors you may cast are the ones in evidence):
{scene}

Rules:
- process: one of {processes}.
  individual_reaction = ONE specific person reacting to a specific stimulus (an email, an offer).
  collective_choice = an identified set of deciders/voters choosing among NAMED options (election,
  court, committee, award).
  population_share = a population's aggregate level/share of something over time.
  artifact_optimization = the question asks WHICH text/design/action is best — the answer is
  generated artifacts ranked by simulated audience response.
- answer_space must literally answer the question: for collective_choice give the NAMED options (the
  actual candidates from the evidence); for individual_reaction {{"type":"binary","options":["responds",
  "does_not_respond"]}}; for artifact_optimization {{"type":"artifacts","options":[]}} (artifacts are
  generated later).
- actors: the people/segments whose behavior GENERATES the outcome. For an election: 4-8 voter segments
  (concrete: geography x ideology x turnout propensity, weights summing to ~1 from the district's real
  makeup) — the candidates are OPTIONS, not actors, unless their in-race conduct matters. For a committee:
  the named members. For individual_reaction: the one named person. Give each segment n_variants 2-3
  (distinct concrete personas will be drawn); named individuals n_variants 3-5 (latent-state draws).
- horizon/time: resolve_by = the real resolution date if the evidence names one (election day, decision
  date); horizon_days = days from today until then; cadence_days = how often the public state meaningfully
  moves (news cycle) for THIS situation.
- interaction: one sentence — what public signal do these actors see between rounds (polls, endorsements,
  media, each other's statements)?

Return ONLY JSON:
{{"process": "...", "answer_space": {{"type": "...", "options": [...]}},
  "actors": [{{"name": "...", "kind": "named|segment", "weight": <0..1>, "role": "...",
              "n_variants": <int>}}],
  "resolve_by": "<YYYY-MM-DD or ''>", "horizon_days": <float>, "cadence_days": <float>,
  "interaction": "...", "rationale": "<one sentence>"}}"""


def allocate_variants(actors, *, budget: int = None, floor: int = 2, cap: int = 6):
    """Gap 3 — hierarchical proportional sampling. Equal variants-per-segment means 3 personas represent a
    90% segment and 3 represent a 10% one: the majority's behavioral diversity is under-sampled exactly where
    it moves the outcome most. Reallocate the SAME total persona budget ∝ segment weight, with a floor so
    small-but-real segments never collapse to one stereotype (their within-segment variance still matters)
    and a cap so one segment can't consume the budget. Named individuals keep their latent-state draws."""
    segs = [a for a in actors if a.kind == "segment"]
    if len(segs) < 2:
        return actors
    budget = budget or sum(a.n_variants for a in segs)
    budget = max(budget, floor * len(segs))
    w = sum(a.weight for a in segs) or 1.0
    for a in segs:                                         # largest-remainder proportional allocation
        a.n_variants = floor
    remaining = budget - floor * len(segs)
    shares = [(a, (a.weight / w) * remaining) for a in segs]
    for a, s in shares:
        a.n_variants += int(s)
    left = remaining - sum(int(s) for _, s in shares)
    for a, s in sorted(shares, key=lambda x: -(x[1] - int(x[1])))[:max(0, left)]:
        a.n_variants += 1
    for a in segs:
        a.n_variants = min(cap, a.n_variants)
    return actors


@dataclass
class CastingDirector:
    llm: object

    def cast(self, question: str, scene_brief: str, *, today: str = "") -> Cast:
        today = today or __import__("time").strftime("%Y-%m-%d")
        raw = parse_json(self.llm(_CAST_PROMPT.format(
            q=question, today=today, scene=scene_brief, processes=json.dumps(PROCESSES)))) or {}
        process = raw.get("process") if raw.get("process") in PROCESSES else "collective_choice"
        space = raw.get("answer_space") or {}
        if not isinstance(space, dict) or "type" not in space:
            space = {"type": "binary", "options": ["yes", "no"]}
        actors = []
        for a in raw.get("actors", []) or []:
            try:
                actors.append(Actor(name=str(a["name"]), kind=a.get("kind", "segment"),
                                    weight=float(a.get("weight", 1.0)), role=str(a.get("role", ""))[:200],
                                    n_variants=max(1, min(5, int(a.get("n_variants", 2))))))
            except (KeyError, TypeError, ValueError):
                continue
        total = sum(a.weight for a in actors) or 1.0
        for a in actors:                                   # weights are shares of the outcome, normalized
            a.weight = a.weight / total
        allocate_variants(actors)                          # gap 3: samples ∝ segment weight, not equal-per-cell
        return Cast(process=process, answer_space=space, actors=actors,
                    horizon_days=float(raw.get("horizon_days", 30.0) or 30.0),
                    resolve_by=str(raw.get("resolve_by", "") or ""),
                    cadence_days=max(0.5, float(raw.get("cadence_days", 7.0) or 7.0)),
                    interaction=str(raw.get("interaction", ""))[:300],
                    rationale=str(raw.get("rationale", ""))[:300])
