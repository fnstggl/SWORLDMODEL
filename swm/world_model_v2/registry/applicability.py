"""Real applicability scoring — the compiler must not select a mechanism because its name sounds relevant.

score_applicability(record, scenario) → ApplicabilityScore with per-axis subscores, an overall score,
hard-exclusion reasons, and the transport-risk widening the instantiation must apply. The scenario
descriptor is a plain dict the compiler assembles from the WorldExecutionPlan draft:

    {"domain": "political_participation", "population_kind": "online_social",
     "time_scale": "hours", "available_state": ["network","entities","information"],
     "available_data": ["activity_log"], "institutional": False}

Axes (each 0..1): domain_match, population_match, time_scale_match, data_availability,
variable_compatibility, institutional_compatibility, evidence_quality (from the record's validation
history — a family whose only support is a citation scores low). transport_risk maps to a mandatory
uncertainty widening factor for any pack fitted outside the scenario's domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.registry.record import MechanismRecord

WIDEN_BY_RISK = {"low": 1.25, "medium": 1.75, "high": 2.5}


@dataclass
class ApplicabilityScore:
    family_id: str
    overall: float
    subscores: dict
    hard_exclusions: list = field(default_factory=list)
    transport_widening: float = 1.0
    pack_id: str = ""                          # best-matching pack, if any
    pack_is_transported: bool = True
    note: str = ""

    def usable(self, threshold: float = 0.45) -> bool:
        return not self.hard_exclusions and self.overall >= threshold


def _match(tags: list, value: str) -> float:
    if not value:
        return 0.5
    tags = [t.lower() for t in (tags or [])]
    v = value.lower()
    if "*" in tags:
        return 0.6                              # wildcards are weaker evidence than a declared match
    if v in tags:
        return 1.0
    if any(v in t or t in v for t in tags):
        return 0.75
    return 0.0


def score_applicability(rec: MechanismRecord, scenario: dict) -> ApplicabilityScore:
    app = rec.applicability
    sub, excl = {}, []
    domain = str(scenario.get("domain", ""))

    if any(_match([d], domain) >= 0.75 for d in app.excluded_domains):
        excl.append(f"domain {domain!r} is explicitly excluded for {rec.family_id}")
    for cond in app.exclusion_conditions:
        flag = str(cond).split(":")[0].strip()
        if scenario.get(flag) is True:
            excl.append(f"exclusion condition met: {cond}")

    sub["domain_match"] = _match(app.domains, domain)
    sub["population_match"] = _match(app.population_kinds, str(scenario.get("population_kind", ""))) \
        if app.population_kinds else 0.5
    sub["time_scale_match"] = _match(app.time_scales, str(scenario.get("time_scale", ""))) \
        if app.time_scales else 0.5

    have_state = set(scenario.get("available_state") or [])
    need_state = set(app.requires_state or [])
    missing_state = need_state - have_state
    sub["variable_compatibility"] = 1.0 if not need_state else \
        max(0.0, 1.0 - len(missing_state) / len(need_state))
    if missing_state:
        excl.append(f"required state absent: {sorted(missing_state)}")

    have_data = set(scenario.get("available_data") or [])
    need_data = set(app.requires_data or [])
    sub["data_availability"] = 1.0 if not need_data else \
        max(0.0, 1.0 - len(need_data - have_data) / len(need_data))

    sub["institutional_compatibility"] = 1.0
    if rec.ontology_type == "institutional" and not scenario.get("institutional", False):
        sub["institutional_compatibility"] = 0.0
        excl.append("institutional mechanism in a scenario with no institution in scope")

    # evidence quality: what the validation HISTORY earned, not what the citation list claims
    recs = list(rec.validation) + [v for p in rec.packs for v in p.validation]
    passed = [v for v in recs if v.passed]
    failed = [v for v in recs if v.passed is False]
    eq = 0.15                                   # citation-only floor
    if any(v.kind == "posterior_predictive" for v in passed):
        eq = max(eq, 0.45)
    if any(v.kind == "held_out" for v in passed):
        eq = max(eq, 0.7)
    if any(v.kind == "transfer" for v in passed):
        eq = max(eq, 0.9)
    if failed and not passed:
        eq = 0.05
    sub["evidence_quality"] = eq

    # pack matching: prefer a pack fitted in this domain; otherwise transported with widening
    best_pack, transported = "", True
    for p in rec.packs:
        if _match([p.domain], domain) >= 0.75:
            best_pack, transported = p.pack_id, False
            break
    if not best_pack and rec.packs:
        best_pack = rec.packs[0].pack_id

    weights = {"domain_match": 0.2, "population_match": 0.1, "time_scale_match": 0.1,
               "variable_compatibility": 0.2, "data_availability": 0.15,
               "institutional_compatibility": 0.05, "evidence_quality": 0.2}
    overall = sum(weights[k] * sub[k] for k in weights)
    widening = 1.0 if not transported else WIDEN_BY_RISK.get(app.transport_risk, 2.5)
    return ApplicabilityScore(
        family_id=rec.family_id, overall=round(overall, 3), subscores={k: round(v, 3) for k, v in sub.items()},
        hard_exclusions=excl, transport_widening=widening, pack_id=best_pack,
        pack_is_transported=transported,
        note=("transported pack — parameter sd multiplied by "
              f"{widening} per transport risk {app.transport_risk!r}") if transported and best_pack else "")


def rank_mechanisms(store, scenario: dict, *, threshold: float = 0.45,
                    statuses=("locally_validated", "transfer_validated", "production_eligible")) -> list:
    """The compiler's selection call: score every family whose lifecycle status allows execution, return
    usable ones ranked. Families below threshold or hard-excluded return in `rejected` with reasons —
    selection provenance is part of the plan."""
    usable, rejected = [], []
    for rec in store.records.values():
        if rec.status not in statuses:
            rejected.append({"family_id": rec.family_id, "reason": f"status={rec.status} not executable "
                            f"in production selection (allowed: {statuses})"})
            continue
        s = score_applicability(rec, scenario)
        if s.usable(threshold):
            usable.append(s)
        else:
            rejected.append({"family_id": rec.family_id,
                             "reason": ("; ".join(s.hard_exclusions) or f"overall {s.overall} < {threshold}"),
                             "subscores": s.subscores})
    usable.sort(key=lambda s: -s.overall)
    return [{"selected": [s.__dict__ for s in usable], "rejected": rejected}][0]
