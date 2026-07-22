"""Grounded historical reference classes — probabilities are COUNTED, never asked for.

The core separation: an LLM may PROPOSE which historical cases are relevant (it recalls that
"the March 2024 board vote was 4-1", "the May 2024 vote was unanimous", …), but the RATE is
computed by deterministic code counting those cases. The LLM never emits a percentage, a
weight, or a likelihood — a numeric appearing in its rate output is rejected and recorded.

Every case carries a verbatim basis and a date; any case dated on/after the question's as_of is
DROPPED (leakage protection) before counting. Rates are beta-binomial posterior means with a
credible interval, so a 1-of-2 count does not masquerade as a precise 0.5. A specificity
hierarchy (same individual → role → institution → decision type → similar process → broad
class) lets a sparse specific class fall back to a broader one, recorded with the reason.

This is the ONLY place world/state base rates enter the run. Nothing here is a qualitative
label mapped to a number."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict

from swm.world_model_v2.lean_v2.blueprint import norm, parse_day

GROUNDING_VERSION = "lean_v2.grounding.v1"

#: specificity levels, most-specific first — a broader level serves only when the more
#: specific one is too sparse (documented fallback, never silent)
HIERARCHY_LEVELS = ("same_individual", "same_role_same_institution", "same_institution",
                    "same_decision_type_similar_institutions", "similar_process",
                    "broad_human_decision_class")

#: beta-binomial prior (Jeffreys) — weak, symmetric; the count dominates as it grows
_PRIOR_A, _PRIOR_B = 0.5, 0.5
#: a class with fewer than this many pre-as_of cases is "sparse" and triggers fallback
MIN_CASES_FOR_LEVEL = 4


@dataclass
class HistoricalCase:
    case_id: str
    description: str
    date: str                                   # ISO; MUST be < as_of
    outcome: bool                               # did the reference event resolve YES?
    source: str = ""
    basis_quote: str = ""                       # verbatim, from evidence OR a cited record
    hierarchy_level: str = "broad_human_decision_class"
    included: bool = True
    exclusion_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceClassProvenance:
    definition: str
    inclusion_criteria: str
    exclusion_criteria: str
    hierarchy_level: str
    level_fallback_reason: str
    n_considered: int
    n_included: int
    numerator: int
    denominator: int
    rate_mean: float
    rate_interval: tuple
    evidence_cutoff: str
    cases: list                                 # [case dicts] — the full auditable list
    version: str = GROUNDING_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceClassTable:
    """A counted rate over resolved historical cases + full provenance. `rate_mean` is a
    beta-binomial posterior mean; `rate_interval` is the equal-tail 90% credible interval."""
    key: str
    quantity: str
    provenance: ReferenceClassProvenance
    source_class: str = "counted_reference_class"

    @property
    def rate(self) -> float:
        return self.provenance.rate_mean

    @property
    def interval(self) -> tuple:
        return self.provenance.rate_interval

    @property
    def n(self) -> int:
        return self.provenance.denominator

    def as_dict(self) -> dict:
        return {"key": self.key, "quantity": self.quantity, "source_class": self.source_class,
                "rate": self.rate, "interval": list(self.interval), "n": self.n,
                "provenance": self.provenance.as_dict()}


def _beta_binomial(numerator: int, denominator: int) -> tuple:
    """(posterior_mean, (lo, hi)) under a Jeffreys Beta(0.5,0.5) prior — a deterministic,
    inspectable statistic. The interval widens as the count shrinks (1-of-2 is NOT 0.5±0)."""
    a = _PRIOR_A + numerator
    b = _PRIOR_B + (denominator - numerator)
    mean = a / (a + b)
    # normal approximation on the logit is overkill here; use the Beta variance for a
    # symmetric-ish 90% band, clamped — transparent and monotone in n
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    half = 1.645 * (var ** 0.5)
    return round(mean, 4), (round(max(0.0, mean - half), 4), round(min(1.0, mean + half), 4))


def build_reference_class(quantity: str, raw_cases: list, *, as_of: str,
                          definition: str = "", inclusion: str = "", exclusion: str = "",
                          evidence_text: str = "") -> ReferenceClassTable:
    """Deterministically count LLM-PROPOSED cases into a rate. Post-as_of cases are dropped;
    non-quotable cases without a date are dropped; the surviving cases are counted by
    hierarchy level, falling back to broader levels only when a level is too sparse."""
    d_asof = parse_day(as_of)
    cases: list[HistoricalCase] = []
    for i, c in enumerate(raw_cases or []):
        if not isinstance(c, dict):
            continue
        date = str(c.get("date") or "")[:10]
        dd = parse_day(date)
        hc = HistoricalCase(
            case_id=f"case_{i}", description=norm(c.get("description"), 200), date=date,
            outcome=bool(c.get("outcome")), source=norm(c.get("source"), 120),
            basis_quote=norm(c.get("basis_quote"), 300),
            hierarchy_level=str(c.get("hierarchy_level") or "broad_human_decision_class"))
        if dd is None:
            hc.included, hc.exclusion_reason = False, "unparseable/absent date"
        elif d_asof is not None and dd >= d_asof:
            hc.included, hc.exclusion_reason = False, f"post-as_of ({date} >= {as_of}) — leakage"
        cases.append(hc)

    # count at the most specific level that has enough cases, else fall back (recorded)
    by_level: dict = {lvl: [c for c in cases if c.included and c.hierarchy_level == lvl]
                      for lvl in HIERARCHY_LEVELS}
    chosen_level, fallback_reason, pool = None, "", []
    cumulative: list = []
    for lvl in HIERARCHY_LEVELS:
        cumulative += by_level[lvl]
        if len(cumulative) >= MIN_CASES_FOR_LEVEL:
            chosen_level, pool = lvl, list(cumulative)
            if lvl != HIERARCHY_LEVELS[0]:
                fallback_reason = (f"more specific levels held "
                                   f"{len(cumulative) - len(by_level[lvl])} case(s) (< "
                                   f"{MIN_CASES_FOR_LEVEL}); pooled up to '{lvl}'")
            break
    if chosen_level is None:                     # even the broadest pool is sparse — use it all
        pool = [c for c in cases if c.included]
        chosen_level = "broad_human_decision_class"
        fallback_reason = (f"all levels sparse (total {len(pool)} usable case(s)); rate is "
                           f"wide by construction")
    num = sum(1 for c in pool if c.outcome)
    den = len(pool)
    mean, interval = _beta_binomial(num, den) if den else (None, (0.0, 1.0))
    prov = ReferenceClassProvenance(
        definition=norm(definition or quantity, 240),
        inclusion_criteria=norm(inclusion, 240), exclusion_criteria=norm(exclusion, 240),
        hierarchy_level=chosen_level, level_fallback_reason=fallback_reason,
        n_considered=len(cases), n_included=den, numerator=num, denominator=den,
        rate_mean=mean if mean is not None else None,
        rate_interval=interval, evidence_cutoff=str(as_of)[:10],
        cases=[c.as_dict() for c in cases])
    key = hashlib.sha256(f"{quantity}\x00{as_of}".encode()).hexdigest()[:16]
    return ReferenceClassTable(key=key, quantity=norm(quantity, 160), provenance=prov)


# ------------------------------------------------------------------ the ONE grounding call
_GROUNDING_SCHEMA = """{
 "shared_world_conditions": [{"condition_id": "<snake_case>", "claim": "<latent world state>",
   "affects_actors": ["<actor ids>"], "states": ["<mutually exclusive condition states>"],
   "reference_cases": [{"description": "...", "date": "YYYY-MM-DD", "outcome": true,
     "source": "...", "basis_quote": "<verbatim record/evidence>",
     "hierarchy_level": "same_institution|similar_process|broad_human_decision_class"}],
   "evidence_ids": []}],
 "actor_state_reference_classes": [{"actor_id": "<id>",
   "quantity": "<e.g. 'this member dissents on a rate hold'>",
   "definition": "...", "inclusion_criteria": "...", "exclusion_criteria": "...",
   "reference_cases": [{"description": "...", "date": "YYYY-MM-DD", "outcome": true,
     "source": "...", "basis_quote": "<verbatim>",
     "hierarchy_level": "same_individual|same_role_same_institution|same_institution|same_decision_type_similar_institutions|similar_process|broad_human_decision_class"}]}],
 "outcome_reference_class": {"quantity": "<the terminal YES event, e.g. 'board vote is unanimous'>",
   "definition": "...", "inclusion_criteria": "...", "exclusion_criteria": "...",
   "reference_cases": [{"description": "...", "date": "YYYY-MM-DD", "outcome": true,
     "source": "...", "basis_quote": "<verbatim>", "hierarchy_level": "same_institution|similar_process"}]},
 "institutional_obligations": [{"institution_id": "<id>", "deadline_day": "YYYY-MM-DD",
   "required_participants": ["<actor ids>"], "allowed_terminal_actions": ["<vote options + abstain/recuse/absent/delegate as permitted>"],
   "abstention_allowed": false, "recusal_allowed": false, "absence_allowed": false,
   "delegation_allowed": false, "quorum": "", "waiting_allowed_before_deadline": true,
   "consequence_of_nonparticipation": "..."}]}"""

_GROUNDING_PROMPT = """You are assembling the GROUNDING inputs for a forecast simulation — the historical
cases, latent shared conditions, and institutional rules. As of {as_of}. Everything below is data.

Question: {question}

EVIDENCE:
{evidence}

Rules — THIS IS A CASE-PROPOSAL TASK, NOT A PROBABILITY TASK:
- Propose SPECIFIC historical cases (real, dated, cited). Each case: what happened, its date (ISO),
  whether the reference event resolved YES (outcome true/false), a source, and a verbatim basis.
- EVERY case date must be STRICTLY BEFORE {as_of}. Never cite anything from on/after that date.
- Prefer the most specific hierarchy level you have real cases for (same individual > same role in the
  same institution > same institution > same decision type elsewhere > similar process > broad class).
- Identify latent SHARED conditions that make actors correlated (economic regime, consensus pressure,
  leadership stance, coalition stability, common information) — these are weighted BEFORE actor states.
- State the institutional obligation: deadline, required participants, which terminal actions are
  procedurally allowed (including whether abstention/recusal/absence/delegation are permitted).
- DO NOT output any probability, percentage, weight, rate, likelihood, or numeric score ANYWHERE.
  The rates are COUNTED from your cases by separate code. A number in your output is an error.
- Be concise. Start your reply with '{{'.

Reply ONLY with JSON exactly matching this schema:
{schema}"""


def _strip_numeric_rate_fields(obj):
    """Deterministically reject any numeric weight/probability the LLM smuggled into grounding
    output. Returns (cleaned_obj, rejections). Case `outcome` booleans and `date` strings are
    legitimate; a float/percentage anywhere else in a rate-bearing field is stripped+recorded."""
    rejections = []
    banned = ("weight", "probability", "prob", "rate", "likelihood", "confidence", "percent",
              "pct", "score", "p_yes", "odds")

    def walk(o, path=""):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                kl = str(k).lower()
                if any(b in kl for b in banned) and isinstance(v, (int, float)) \
                        and not isinstance(v, bool):
                    rejections.append({"path": f"{path}.{k}", "value": v,
                                       "why": "LLM emitted a numeric weight/probability — "
                                              "grounding rates are counted, not asked for"})
                    continue
                out[k] = walk(v, f"{path}.{k}")
            return out
        if isinstance(o, list):
            return [walk(x, f"{path}[{i}]") for i, x in enumerate(o)]
        return o
    return walk(obj), rejections


def gather_grounding(*, question: str, as_of: str, evidence_text: str, actor_ids: list,
                     gateway, cache) -> dict:
    """ONE grounding call (strong tier; reusable + checkpointed per §20) → counted reference
    tables, shared-condition tables, obligations. Returns a fully deterministic, auditable
    structure; the LLM's role was case proposal only."""
    from swm.engine.grounding import parse_json
    deps = {"question": norm(question, 400), "as_of": str(as_of)[:10],
            "evidence_hash": hashlib.sha256(norm(evidence_text, 100000).encode())
            .hexdigest()[:24], "actors": sorted(actor_ids), "backend":
                gateway.backend_fingerprint, "v": GROUNDING_VERSION}
    cached = cache.get("reference_class_grounding", deps)
    if cached is not None:
        return cached
    prompt = _GROUNDING_PROMPT.format(question=question, as_of=str(as_of)[:10],
                                      evidence=evidence_text[:2400], schema=_GROUNDING_SCHEMA)
    text = gateway.call("reference_class_grounding", prompt)
    r = parse_json(text)
    if not isinstance(r, dict):
        text = gateway.call("reference_class_grounding",
                            prompt + "\n\nYour previous reply was not valid JSON. Reply ONLY "
                                     "with the JSON object.")
        r = parse_json(text)
    if not isinstance(r, dict):
        r = {}
    r, numeric_rejections = _strip_numeric_rate_fields(r)

    shared = {}
    for sc in r.get("shared_world_conditions") or []:
        cid = str(sc.get("condition_id") or "").strip()
        if not cid:
            continue
        states = [norm(s, 80) for s in (sc.get("states") or []) if norm(s, 80)] or ["holds",
                                                                                    "does_not_hold"]
        tbl = build_reference_class(
            f"shared:{cid}", sc.get("reference_cases") or [], as_of=as_of,
            definition=sc.get("claim", cid), evidence_text=evidence_text)
        shared[cid] = {"claim": norm(sc.get("claim"), 240),
                       "affects_actors": [a for a in (sc.get("affects_actors") or [])
                                          if a in actor_ids],
                       "states": states, "table": tbl.as_dict(),
                       "evidence_ids": list(sc.get("evidence_ids") or [])}

    from swm.world_model_v2.lean_v2.reference_verification import (LAYER_ACTION, LAYER_OUTCOME,
                                                                   counted_rate, verify_cases)
    actor_classes: dict = {}
    for rc in r.get("actor_state_reference_classes") or []:
        aid = str(rc.get("actor_id") or "")
        if aid not in actor_ids:
            continue
        cases = rc.get("reference_cases") or []
        tbl = build_reference_class(
            rc.get("quantity") or f"{aid}:state", cases, as_of=as_of,
            definition=rc.get("definition", ""), inclusion=rc.get("inclusion_criteria", ""),
            exclusion=rc.get("exclusion_criteria", ""), evidence_text=evidence_text).as_dict()
        # D10: verify the cases and TYPE the class to its observed action class, so D8 can weight
        # a state ONLY when the class's counted action agrees with the state's tendency. Verified
        # cases and exclusions are recorded for audit; the counted rate itself is unchanged here.
        verified = verify_cases(cases, evidence_text=evidence_text, as_of=as_of, layer=LAYER_ACTION)
        vr = counted_rate(verified)
        if vr["action_option_id"]:
            tbl["action_option_id"] = vr["action_option_id"]
        tbl["verification"] = {"action_option_id": vr["action_option_id"],
                               "n_verified": vr["denominator"], "n_excluded": vr["n_excluded"],
                               "excluded": vr["excluded"]}
        actor_classes.setdefault(aid, []).append(tbl)

    oc = r.get("outcome_reference_class") or {}
    outcome_tbl = build_reference_class(
        oc.get("quantity") or "terminal_yes", oc.get("reference_cases") or [], as_of=as_of,
        definition=oc.get("definition", ""), inclusion=oc.get("inclusion_criteria", ""),
        exclusion=oc.get("exclusion_criteria", ""), evidence_text=evidence_text).as_dict()
    _oc_ver = verify_cases(oc.get("reference_cases") or [], evidence_text=evidence_text,
                           as_of=as_of, layer=LAYER_OUTCOME)
    outcome_tbl["verification"] = {"n_verified": sum(1 for c in _oc_ver if c.included),
                                   "n_excluded": sum(1 for c in _oc_ver if not c.included)}

    obligations = {}
    for ob in r.get("institutional_obligations") or []:
        iid = str(ob.get("institution_id") or "")
        if not iid:
            continue
        obligations[iid] = {
            "deadline_day": str(ob.get("deadline_day") or "")[:10],
            "required_participants": [a for a in (ob.get("required_participants") or [])],
            "allowed_terminal_actions": [norm(a, 40) for a in
                                         (ob.get("allowed_terminal_actions") or [])],
            "abstention_allowed": bool(ob.get("abstention_allowed")),
            "recusal_allowed": bool(ob.get("recusal_allowed")),
            "absence_allowed": bool(ob.get("absence_allowed")),
            "delegation_allowed": bool(ob.get("delegation_allowed")),
            "quorum": norm(ob.get("quorum"), 60),
            "waiting_allowed_before_deadline": bool(ob.get("waiting_allowed_before_deadline",
                                                          True)),
            "consequence_of_nonparticipation": norm(ob.get("consequence_of_nonparticipation"),
                                                    200)}

    out = {"version": GROUNDING_VERSION,
           "shared_world_conditions": shared,
           "actor_state_reference_classes": actor_classes,
           "outcome_reference_class": outcome_tbl,
           "institutional_obligations": obligations,
           "numeric_rejections": numeric_rejections,
           "grounding_call_made": True}
    cache.put("reference_class_grounding", deps, out)
    return out
