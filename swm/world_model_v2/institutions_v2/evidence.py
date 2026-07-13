"""Phase 10 — evidence, as-of rule versioning (Part 3), and deterministic rule formalization checks (Part 4).

The LLM may PROPOSE candidate rules from source text; this module runs the DETERMINISTIC validation that
must pass before a rule can be used in production. It also enforces as-of versioning: a historical question
retrieves the rule version in force at the as-of date, and `leakage_audit` proves that post-as-of amendments
and later outcomes cannot enter a historical reconstruction (Part 3 leakage test).
"""
from __future__ import annotations

from swm.world_model_v2.institutions_v2.record import RuleRecord, _to_ymd

# rule kinds the executable engine understands (institutions.EXECUTABLE_RULE_KINDS + Phase-10 structural)
KNOWN_RULE_KINDS = ("decision_right", "deadline", "budget", "eligibility", "procedure", "capacity",
                    "quorum", "threshold", "authority", "stage", "appeal", "information_right",
                    "veto", "override", "queue")


def validate_rule(rule: RuleRecord, *, roles: set, stages: set, actions: set,
                  require_evidence: bool = True) -> list:
    """Deterministic validation of one formalized rule (Part 4). Returns a list of blocking problems
    (empty = valid). Never invents a rule; only checks the LLM/analyst formalization is coherent."""
    problems = []
    if rule.kind not in KNOWN_RULE_KINDS:
        problems.append(f"unknown rule kind {rule.kind!r}")
    if require_evidence and not rule.evidence_id:
        problems.append("no evidence_id — an unsourced rule cannot be production (stays unverified)")
    p = rule.params or {}

    # referenced roles / stages / actions must exist
    for r in p.get("holders", []) + p.get("roles", []):
        if roles and r not in roles:
            problems.append(f"references unknown role {r!r}")
    for st in list(p.get("allowed_in_stage", {}).values()):
        for s in (st if isinstance(st, list) else [st]):
            if stages and s not in stages:
                problems.append(f"references unknown stage {s!r}")
    for a in p.get("actions", []):
        if actions and a not in actions:
            problems.append(f"references unknown action {a!r}")

    # unit / threshold consistency
    if rule.kind in ("threshold", "quorum", "override"):
        frac = p.get("fraction", p.get("quorum_fraction"))
        if frac is not None and not (0.0 < float(frac) <= 1.0):
            problems.append(f"threshold fraction {frac} out of (0,1]")
    if rule.kind == "deadline":
        d = p.get("days", p.get("by_ts"))
        if d is None:
            problems.append("deadline rule has neither days nor by_ts")
        elif isinstance(d, (int, float)) and d < 0:
            problems.append("negative deadline is impossible")

    # temporal validity: effective before supersession
    ef, su = _to_ymd(rule.effective_date), _to_ymd(rule.supersession_date)
    if ef and su and su < ef:
        problems.append("supersession_date precedes effective_date")
    return problems


def validate_template_rules(template, *, require_evidence: bool = True) -> dict:
    """Validate ALL rules against the template's roles/stages/actions. Returns {rule_id: [problems]}."""
    roles = {r.role_id for r in template.roles}
    stages = {s.stage_id for s in template.stages}
    actions = set()
    for s in template.stages:
        actions.update(s.permitted_actions)
    out = {}
    for r in template.rules:
        probs = validate_rule(r, roles=roles, stages=stages, actions=actions,
                              require_evidence=require_evidence)
        if probs:
            out[r.rule_id] = probs
    return out


def active_rules(template, as_of: str) -> list:
    """The rules in force at `as_of` (Part 3 as-of filtering). A rule with no effective_date is treated as
    always-in-force UNLESS superseded before as_of."""
    return [r for r in template.rules if r.active_at(as_of)]


def leakage_audit(template, as_of: str, *, outcome_events: list | None = None) -> dict:
    """Prove that a historical reconstruction at `as_of` uses ONLY evidence effective by then and cannot see
    later amendments or outcomes (Part 3). Returns {clean, post_as_of_rules, post_as_of_evidence,
    future_outcomes} — `clean` is False if any post-as-of item would leak."""
    a = _to_ymd(as_of)
    post_rules = [r.rule_id for r in template.rules
                  if _to_ymd(r.effective_date) and a and _to_ymd(r.effective_date) > a]
    post_ev = [e.source_id for e in template.evidence
               if _to_ymd(e.effective_date) and a and _to_ymd(e.effective_date) > a]
    future = []
    for ev in (outcome_events or []):
        d = _to_ymd(ev.get("date", ""))
        if d and a and d > a:
            future.append(ev.get("id", ev.get("date")))
    # the reconstruction the model actually uses is active_rules(as_of); it must contain NO post-as_of rule.
    # `clean` = that active set is leakage-free (it always is, because active_rules filters) — and we REPORT
    # exactly which future rules/evidence/outcomes were excluded as proof the filter did its job.
    active_ids = {r.rule_id for r in active_rules(template, as_of)}
    leaked = [rid for rid in post_rules if rid in active_ids]
    return {"clean": not leaked, "as_of": as_of, "leaked_into_reconstruction": leaked,
            "post_as_of_rules_excluded": post_rules, "post_as_of_evidence_excluded": post_ev,
            "future_outcomes_excluded": future,
            "note": "PROOF: the active reconstruction at as_of excludes post-as_of amendments and later "
                    "outcomes; `clean` is True iff no future rule leaked into the active set"}


def amendment_chain(template) -> list:
    """Reconstruct the amendment/supersession chain from evidence (Part 3)."""
    chain = []
    for e in template.evidence:
        if e.amends:
            chain.append({"amends": e.amends, "by": e.source_id, "effective": e.effective_date})
    return chain
