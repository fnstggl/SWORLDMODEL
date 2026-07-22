"""Integration completion — requirement inference + spec normalization so causally-required phases activate.

The Part-A activation-chain audit found concrete breaks between the compiler's output and the runtime's
instantiation contracts. The highest-confidence, verifiable break is Phase 10: the compiler emits institutions
with rule KINDS (voting_rule / confirmation_process / committee_vote / …) that are NOT in the runtime's closed
`EXECUTABLE_RULE_KINDS` set, so `materialize` silently DROPS them and the RuleSystem is empty (ornamental).

`normalize_institution_rules` maps those compiler kinds onto the executable set (preserving params) so declared
institutions get REAL executable rules — turning an ornamental institution into one the runtime can execute.
It is conservative: it never invents institutions, only makes a DECLARED one executable.

`infer_required_phases` gives an independent, structural requirement judgment used by the completeness
validator and the active-component manifest's `relevance` field — it is NOT the ground-truth relevance label
(those live in the benchmark, Part B) and it does not fabricate specs.
"""
from __future__ import annotations

from swm.world_model_v2.institutions import EXECUTABLE_RULE_KINDS

# compiler rule-kind → executable rule kind (params preserved). Anything unmapped → generic executable procedure.
_RULE_KIND_MAP = {
    "voting_rule": "quorum", "vote": "quorum", "majority_vote": "quorum", "supermajority": "quorum",
    "confirmation_process": "procedure", "committee_vote": "procedure", "floor_vote": "procedure",
    "ratification": "procedure", "approval_process": "procedure", "review": "procedure",
    "hearing": "procedure", "stage": "procedure", "reading": "procedure",
    "veto": "decision_right", "authority": "decision_right", "agenda_control": "decision_right",
    "certification": "decision_right", "signature": "decision_right",
    "timeline": "deadline", "schedule": "deadline", "sunset": "deadline",
    "funding": "budget", "appropriation": "budget",
    "membership": "eligibility", "standing": "eligibility", "jurisdiction": "eligibility",
    "throughput": "capacity", "docket": "capacity",
}


def normalize_institution_rules(plan):
    """Rewrite each declared institution's rule kinds onto the EXECUTABLE set so the RuleSystem is not empty.
    Returns a report: {institutions, rules_total, rules_already_executable, rules_normalized, rules_unmapped}.
    Mutates plan.institutions in place. Idempotent."""
    rep = {"institutions": 0, "rules_total": 0, "already_executable": 0, "normalized": 0, "unmapped_to_procedure": 0}
    for inst in (getattr(plan, "institutions", []) or []):
        if not isinstance(inst, dict):
            continue
        rep["institutions"] += 1
        for ru in (inst.get("rules") or []):
            if not isinstance(ru, dict):
                continue
            rep["rules_total"] += 1
            kind = str(ru.get("kind", "procedure"))
            if kind in EXECUTABLE_RULE_KINDS:
                rep["already_executable"] += 1
                continue
            mapped = _RULE_KIND_MAP.get(kind)
            if mapped is None:
                mapped = "procedure"                          # generic executable fallback (never drop the rule)
                rep["unmapped_to_procedure"] += 1
            else:
                rep["normalized"] += 1
            ru["_original_kind"] = kind
            ru["kind"] = mapped
    return rep


def executable_rule_count(plan):
    """How many institution rules would the runtime actually execute (kind in the executable set)."""
    n = 0
    for inst in (getattr(plan, "institutions", []) or []):
        if isinstance(inst, dict):
            n += sum(1 for ru in (inst.get("rules") or [])
                     if isinstance(ru, dict) and str(ru.get("kind")) in EXECUTABLE_RULE_KINDS)
    return n


def infer_required_phases(plan):
    """Independent structural requirement judgment from the compiled plan's typed sections (NOT the benchmark
    ground truth). Used for the manifest `relevance` field + the completeness validator."""
    req = {}
    req["phase10_institutions"] = bool(getattr(plan, "institutions", []))
    req["phase9_populations"] = bool(getattr(plan, "populations", []))
    req["phase9_networks"] = bool(getattr(plan, "relations", []))
    req["phase4_actor_policy"] = bool(getattr(plan, "actor_decisions", [])) or any(
        str(m.get("operator", "")).endswith(("decision", "policy")) for m in
        (getattr(plan, "accepted_mechanisms", []) or []) if isinstance(m, dict))
    ops = " ".join(str(m.get("operator", "")) for m in (getattr(plan, "accepted_mechanisms", []) or [])
                   if isinstance(m, dict))
    req["phase7_nonlinear"] = "nonlinear" in ops
    req["phase6_registry"] = any(k in ops for k in ("mechanism", "contagion", "diffusion", "belief", "resource"))
    return req


def completeness_diagnostics(plan):
    """Deterministic post-compiler completeness check (Part J): flag institutions declared but not executable.
    Returns [{issue, detail, severity}]. It does NOT silently repair beyond rule-kind normalization."""
    diags = []
    insts = getattr(plan, "institutions", []) or []
    if insts and executable_rule_count(plan) == 0:
        diags.append({"issue": "institution_declared_but_no_executable_rule",
                      "detail": f"{len(insts)} institution(s) with no executable rule kind — RuleSystem would be "
                                "empty (ornamental) until normalize_institution_rules runs",
                      "severity": "high"})
    has_inst_op = any(str(m.get("operator", "")) in ("institution_action", "institutional_vote")
                      for m in (getattr(plan, "accepted_mechanisms", []) or []) if isinstance(m, dict))
    if insts and not has_inst_op:
        diags.append({"issue": "institution_declared_but_no_operator",
                      "detail": "institutions present but no institution_action/institutional_vote operator is "
                                "named, so no institutional_action event will fire (execution gap)",
                      "severity": "high"})
    return diags
