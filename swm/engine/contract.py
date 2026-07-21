"""The typed answer CONTRACT + the measurable GROUNDING SCORE — vision gaps 1 & 2.

Gap 1 (typed outcome contract): every supported question family has a formal OutcomeSpace schema, VALIDATED
BEFORE simulation begins (a bad cast fails loudly instead of degrading into a generic scalar) and CHECKED
AFTER (the produced Forecast must conform to its declared space — distributions sum to ~1 over the declared
options, ranked artifacts carry real text + a metric, response mode carries a scenario-specific p). This is
the mechanical guarantee that future additions can't regress the native-answer principle.

Gap 2 (grounding score): "grounded" must be measurable, not vibes. Score each dossier on five observable
axes — actor coverage (were the named actors actually retrieved?), decisive-fact presence (is there a
directional standing?), freshness (how close is the newest evidence to as-of?), corroboration (do >=2
independent sources back the facts?), and as-of validity (did anything post-date the cutoff?). The composite
gates confidence: a low grounding score should shrink toward the base rate / abstain, and it ships in every
Forecast's grounding report so a consumer can knowingly distrust a thinly-grounded number.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

# the closed set of supported answer families — adding a family means adding a schema HERE first
FAMILIES = ("binary", "categorical", "named_options", "ranked_artifacts", "response_probability",
            "reach_distribution")


class ContractViolation(ValueError):
    """A cast or forecast that does not conform to its declared OutcomeSpace. Fail loudly, never degrade."""


def validate_outcome_space(space: dict) -> dict:
    """Validate an OutcomeSpace BEFORE simulation. Returns the (normalized) space or raises ContractViolation."""
    if not isinstance(space, dict) or not space.get("type"):
        raise ContractViolation(f"outcome space missing/untyped: {space!r}")
    t = space["type"]
    if t not in FAMILIES:
        raise ContractViolation(f"unknown outcome family {t!r} — supported: {FAMILIES}")
    opts = space.get("options") or []
    if t in ("categorical", "named_options"):
        named = [o for o in opts if isinstance(o, str) and o.strip()]
        if len(named) < 2:
            raise ContractViolation(f"{t} needs >=2 NAMED options, got {opts!r}")
        if any(o.lower() in ("option a", "option b", "candidate a", "candidate b", "other") for o in named):
            raise ContractViolation(f"{t} options look like abstractions, not real names: {named!r}")
    elif t == "binary":
        if opts and sorted(str(o).lower() for o in opts) != ["no", "yes"]:
            raise ContractViolation(f"binary options must be yes/no, got {opts!r}")
        space["options"] = ["yes", "no"]
    elif t == "ranked_artifacts":
        if not space.get("metric"):
            space["metric"] = "p_engage"                      # explicit target metric (gap 9 ties into this)
    elif t == "response_probability":
        space["options"] = ["responds", "does_not_respond"]
    elif t == "reach_distribution":
        pass                                                   # thresholds are produced by the simulator
    return space


def check_forecast_conforms(f: dict) -> list:
    """Check a produced Forecast dict against its declared answer_space AFTER simulation. Returns a list of
    violations (empty = conforms). Abstentions always conform — refusing is a valid native answer."""
    if f.get("abstain"):
        return []
    space = f.get("answer_space") or {}
    t = space.get("type")
    t = "ranked_artifacts" if t == "artifacts" else t     # legacy alias used by the artifact mode
    v = []
    dist = f.get("distribution") or {}
    if t in ("categorical", "named_options", "binary", "response_probability"):
        declared = set(space.get("options") or [])
        if not dist:
            v.append("no distribution produced for a distribution-typed question")
        else:
            extra = set(dist) - declared if declared else set()
            if extra:
                v.append(f"distribution has options outside the declared space: {sorted(extra)}")
            s = sum(dist.values())
            if not (0.98 <= s <= 1.02):
                v.append(f"distribution sums to {s:.3f}, not ~1")
    elif t == "ranked_artifacts":
        arts = f.get("ranked_artifacts") or []
        if not arts:
            v.append("no ranked artifacts produced for an artifact question")
        elif not all(a.get("text", "").strip() for a in arts):
            v.append("ranked artifacts must carry the ACTUAL candidate texts")
    elif t == "reach_distribution":
        if not dist:
            v.append("no reach distribution produced")
    return v


# ---------------------------------------------------------------- gap 2: the grounding score
@dataclass
class GroundingScore:
    actor_coverage: float = 0.0        # fraction of cast/named actors that appear in the retrieved facts
    decisive_fact: float = 0.0         # 1.0 = a DIRECTIONAL standing was established, 0.5 = standing text only
    freshness: float = 0.0             # newest evidence age vs as-of (1.0 <= 7d, decays to 0 at ~120d)
    corroboration: float = 0.0         # fraction of facts backed by >=2 distinct sources (source diversity)
    asof_valid: float = 1.0            # 1.0 = nothing post-dates the cutoff (should always hold; verified)
    composite: float = 0.0
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return {"actor_coverage": round(self.actor_coverage, 3), "decisive_fact": self.decisive_fact,
                "freshness": round(self.freshness, 3), "corroboration": round(self.corroboration, 3),
                "asof_valid": self.asof_valid, "composite": round(self.composite, 3), **(
                    {"detail": self.detail} if self.detail else {})}


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return _time.mktime(_time.strptime(str(s)[:10], fmt))
        except (ValueError, TypeError):
            continue
    return None


def score_grounding(dossier, *, actors=None, as_of: str = "") -> GroundingScore:
    """Compute the five-axis grounding score from an already-built SceneDossier. Cheap (no LLM calls)."""
    facts = dossier.facts or []
    text_all = " ".join(f"{f.get('fact', '')} {f.get('detail', '')}" for f in facts).lower()
    gs = GroundingScore()
    # actor coverage — of the actors we know matter, how many did retrieval actually establish?
    names = [a for a in (actors or getattr(dossier, "actors_hint", None) or [])
             if isinstance(a, str) and a.strip()]
    if names:
        found = sum(1 for n in names if any(tok in text_all for tok in [n.lower()] +
                                            ([n.split()[-1].lower()] if len(n.split()) > 1 else [])))
        gs.actor_coverage = found / len(names)
        gs.detail["actors_missing"] = [n for n in names if n.lower() not in text_all
                                       and (len(n.split()) < 2 or n.split()[-1].lower() not in text_all)]
    else:
        gs.actor_coverage = 0.5                              # no cast yet — neutral
    # decisive fact — the directional standing is THE fact that decides most questions
    gs.decisive_fact = 1.0 if getattr(dossier, "standing_directional", False) else \
        (0.5 if getattr(dossier, "standing", "") else 0.0)
    # freshness — newest fact date vs as_of
    asof_ts = _parse_date(as_of) or _time.time()
    dates = [d for d in (_parse_date(f.get("date")) for f in facts) if d is not None]
    if dates:
        age_days = max(0.0, (asof_ts - max(dates)) / 86400.0)
        gs.freshness = max(0.0, min(1.0, 1.0 - (age_days - 7.0) / 113.0))   # 1.0 <=7d → 0.0 at 120d
        gs.detail["newest_fact_age_days"] = round(age_days, 1)
        # as-of validity: a fact dated AFTER the cutoff is a leak — should never happen (retrieval drops them)
        leaks = sum(1 for d in dates if d > asof_ts + 86400)
        gs.asof_valid = 0.0 if leaks else 1.0
        if leaks:
            gs.detail["post_asof_facts"] = leaks
    # corroboration — source diversity across facts
    sources = [str(f.get("source", "")).split(":")[0] for f in facts if f.get("source")]
    if facts:
        gs.corroboration = min(1.0, len(set(s for s in sources if s)) / max(2.0, len(facts) * 0.4))
    gs.composite = round(0.30 * gs.actor_coverage + 0.30 * gs.decisive_fact + 0.15 * gs.freshness +
                         0.15 * gs.corroboration + 0.10 * gs.asof_valid, 3)
    return gs
