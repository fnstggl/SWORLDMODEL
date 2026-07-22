"""D10 — verified reference cases + separated prior/behavior layers.

A counted rate is only as honest as the cases behind it. Full fidelity let the model propose
historical cases with placeholder sources, unverifiable or invented quotes, mismatched
actor/action pairs, and vague thematic gestures, then counted them as if real. And it mixed three
distinct things — the OUTCOME base rate, an actor's ACTION tendency, and an actor's private STATE —
so outcome history leaked in as private-state "evidence."

This module makes every case EARN its count:

  * `source_available`  — a real citable source, not a placeholder (example.com, TODO, "a study");
  * `quote_verified`    — the source_quote actually appears in the PERMITTED evidence (normalized
    substring or near-exact token match) — an LLM-invented quote fails this;
  * `date_verified`     — a parseable date strictly before as_of (no leakage);
  * `action_typed`      — a concrete actor/role AND a concrete observed_action (not vague-thematic),
    so the case can be typed to an action class (this feeds D8's `action_option_id`);
  * `layer`             — outcome | action_baseline | state_hypothesis; a case may only be counted
    in its own layer, so outcome history can never weight a private state.

A case counts iff every applicable check passes; otherwise it is EXCLUDED with a recorded reason.
Universal: the checks are content checks against the sealed evidence, never question-specific."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key, parse_day

REFERENCE_VERIFICATION_VERSION = "lean_v2.reference_verification.v1"

# the three layers that must never be conflated
LAYER_OUTCOME = "outcome"                 # base rate of the terminal outcome
LAYER_ACTION = "action_baseline"          # how an actor/role tends to ACT
LAYER_STATE = "state_hypothesis"          # an actor's private mindset

#: placeholder / non-source markers — a "source" matching any of these is not a real citation
_PLACEHOLDER_SOURCE = re.compile(
    r"\b(example\.(com|org)|todo|tbd|n/?a|placeholder|lorem|foo|bar|various|unknown|"
    r"a study|some report|reports say|sources say|it is said|anecdot)\b", re.I)

#: vague-thematic actions that cannot be typed to an action class
_VAGUE_ACTION = re.compile(
    r"^(acted|responded|reacted|behaved|did something|was involved|participated|engaged|"
    r"took action|moved|proceeded|handled it|dealt with it)\.?$", re.I)

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s or "").strip().lower())


@dataclass
class VerifiedReferenceCase:
    case_id: str
    layer: str
    description: str = ""
    source: str = ""
    source_quote: str = ""
    date: str = ""
    actor_or_role: str = ""
    decision_type: str = ""
    observed_action: str = ""              # the action class this case counts (→ D8 action_option_id)
    outcome: bool = False
    # verification flags
    source_available: bool = False
    quote_verified: bool = False
    date_verified: bool = False
    action_typed: bool = False
    included: bool = False
    exclusion_reason: str = ""
    version: str = REFERENCE_VERIFICATION_VERSION

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("case_id", "layer", "description", "source", "source_quote", "date",
                 "actor_or_role", "decision_type", "observed_action", "outcome",
                 "source_available", "quote_verified", "date_verified", "action_typed",
                 "included", "exclusion_reason", "version")}


def _quote_appears(quote: str, evidence_norm: str) -> bool:
    """The quote is verified when its normalized text is a substring of the normalized evidence,
    or ≥ 0.85 of its content tokens appear in the evidence (paraphrase-robust but not a free pass —
    a fabricated quote shares little with the real evidence)."""
    q = _norm(quote)
    if len(q) < 8:
        return False                       # too short to verify anything meaningful
    if q in evidence_norm:
        return True
    qtoks = [t for t in q.split() if len(t) > 2]
    if not qtoks:
        return False
    ev_tokens = set(evidence_norm.split())
    hit = sum(1 for t in qtoks if t in ev_tokens)
    return hit / len(qtoks) >= 0.85


def verify_reference_case(raw: dict, *, evidence_text: str, as_of: str, layer: str,
                          case_id: str = "") -> VerifiedReferenceCase:
    """Verify one proposed case against the permitted evidence. Returns a VerifiedReferenceCase
    with each check's result and a single inclusion decision + recorded exclusion reason."""
    evidence_norm = _norm(evidence_text)
    vc = VerifiedReferenceCase(
        case_id=case_id or f"vc_{abs(hash(_norm(str(raw)))) % 10**8}", layer=layer,
        description=norm(raw.get("description"), 240), source=norm(raw.get("source"), 160),
        source_quote=norm(raw.get("basis_quote") or raw.get("source_quote"), 300),
        date=str(raw.get("date") or "")[:10],
        actor_or_role=norm(raw.get("actor_or_role") or raw.get("actor_id"), 120),
        decision_type=norm(raw.get("decision_type"), 120),
        observed_action=norm(raw.get("observed_action") or raw.get("action_option_id"), 120),
        outcome=bool(raw.get("outcome")))

    # source availability — a real citation, not a placeholder
    vc.source_available = bool(vc.source) and not _PLACEHOLDER_SOURCE.search(vc.source)
    # quote verification — the quote must be traceable in the permitted evidence
    vc.quote_verified = bool(vc.source_quote) and _quote_appears(vc.source_quote, evidence_norm)
    # date — parseable and strictly pre as_of (no leakage)
    dd, da = parse_day(vc.date), parse_day(as_of)
    vc.date_verified = dd is not None and (da is None or dd < da)
    # action typing — a concrete action class (only required for the action layer)
    typed = bool(vc.observed_action) and not _VAGUE_ACTION.match(vc.observed_action.strip())
    vc.action_typed = typed if layer == LAYER_ACTION else True

    reasons = []
    if not vc.source_available:
        reasons.append("no real source (placeholder/absent)")
    if not vc.quote_verified:
        reasons.append("quote not verifiable in the permitted evidence")
    if not vc.date_verified:
        reasons.append("date unparseable, absent, or post-as_of (leakage)")
    if not vc.action_typed:
        reasons.append("action vague/untyped — cannot map to an action class")
    vc.included = not reasons
    vc.exclusion_reason = "" if vc.included else "; ".join(reasons)
    return vc


def verify_cases(raw_cases: list, *, evidence_text: str, as_of: str, layer: str) -> list:
    """Verify a list of proposed cases; returns [VerifiedReferenceCase] (included + excluded, all
    with reasons) for full auditability."""
    out = []
    for i, c in enumerate(raw_cases or []):
        if not isinstance(c, dict):
            continue
        out.append(verify_reference_case(c, evidence_text=evidence_text, as_of=as_of,
                                         layer=layer, case_id=f"{layer}_{i}"))
    return out


def counted_rate(verified: list) -> dict:
    """Beta-binomial rate over the INCLUDED cases only, plus the typed dominant action class (the
    D8 `action_option_id`). Excluded cases never touch the numerator/denominator."""
    from swm.world_model_v2.lean_v2.grounding import _beta_binomial
    incl = [c for c in verified if c.included]
    num = sum(1 for c in incl if c.outcome)
    den = len(incl)
    mean, interval = _beta_binomial(num, den) if den else (None, (0.0, 1.0))
    # the action class this rate counts = the most common observed_action among included cases
    actions = {}
    for c in incl:
        if c.observed_action:
            actions[norm_key(c.observed_action)] = actions.get(norm_key(c.observed_action), 0) + 1
    action_option_id = max(actions, key=actions.get) if actions else ""
    return {"rate_mean": mean, "rate_interval": list(interval), "numerator": num,
            "denominator": den, "n_considered": len(verified),
            "n_excluded": len(verified) - den, "action_option_id": action_option_id,
            "excluded": [{"case_id": c.case_id, "reason": c.exclusion_reason}
                         for c in verified if not c.included]}
