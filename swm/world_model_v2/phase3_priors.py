"""Prior construction with provenance + transport-risk inflation — Phase 3 (Part B).

A posterior is only as honest as its prior. This module builds the outcome-rate prior with an explicit
provenance chain and, crucially, INFLATES its variance for transport risk: a reference class estimated from a
different population/time/context must not masquerade as a tight, well-identified prior for THIS scenario.

LLM-inference contract (enforced):
  * the LLM may PROPOSE a reference-class descriptor and a QUALITATIVE transport-risk level (semantic mapping).
    It may NOT mint the base rate, the pseudo-counts, or the inflation factor.
  * the base rate comes from PROVIDED reference-class data (successes/total) when available; absent data it
    falls back to the qualitative-lean broad Beta (labeled generic_weakly_informative — NOT a reference class).
  * transport risk maps to a FIXED retained-effective-sample fraction (severe transport → almost no effective
    data → a near-flat prior). More transport risk ALWAYS widens, never narrows.

The result is a `PriorSpec` carrying the Beta parameters, the effective sample size actually retained, and the
full provenance a reviewer needs to trace the prior back to its source (or to "no reference class found").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.fallback import LEAN_BETA

#: qualitative transport risk → fraction of the reference-class effective sample size RETAINED. A prior lifted
#: from a poorly-matching reference class keeps little of its apparent precision. FIXED — never LLM-minted.
TRANSPORT_RETAINED = {"none": 1.0, "low": 0.65, "moderate": 0.4, "high": 0.22, "severe": 0.08}
#: a reference-class prior is never allowed to be tighter than this effective sample size (keeps it broad).
MAX_EFFECTIVE_N = 40.0
TRANSPORT_RISKS = tuple(TRANSPORT_RETAINED)


@dataclass
class PriorSpec:
    family: str = "beta"
    alpha: float = 1.0
    beta: float = 1.0
    source_class: str = "generic_weakly_informative"        # "reference_class" | "generic_weakly_informative"
    reference_class: str = ""
    transport_risk: str = "high"
    retained_effective_n: float = 0.0
    raw_effective_n: float = 0.0
    provenance: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    def as_dict(self):
        d = self.__dict__.copy()
        d["mean"] = round(self.mean, 5)
        return d


def reference_class_prior(reference_class: str, successes: float, total: float, *,
                          transport_risk: str = "high", lean: str = "neutral") -> PriorSpec:
    """Build a Beta prior from reference-class data (successes/total), then INFLATE its variance by the
    transport-risk retained-sample fraction. The reference-class MEAN is preserved; its PRECISION is discounted
    (that is the whole point of transport-risk widening). `successes`/`total` are DATA, not LLM output."""
    total = max(0.0, float(total))
    successes = max(0.0, min(total, float(successes)))
    if total <= 0:                                          # no usable data → fall back to the generic lean prior
        return generic_lean_prior(lean, reason=f"reference class {reference_class!r} had no usable base-rate data")
    raw_rate = successes / total
    retained = TRANSPORT_RETAINED.get(transport_risk, 0.22)
    eff_n = min(MAX_EFFECTIVE_N, total * retained)          # transport-discounted effective sample size
    # a Laplace-smoothed Beta centered on the reference rate with the RETAINED effective sample size
    a = raw_rate * eff_n + 1.0
    b = (1.0 - raw_rate) * eff_n + 1.0
    return PriorSpec(
        family="beta", alpha=round(a, 4), beta=round(b, 4), source_class="reference_class",
        reference_class=reference_class, transport_risk=transport_risk,
        retained_effective_n=round(eff_n, 3), raw_effective_n=round(total, 3),
        provenance={"reference_rate": round(raw_rate, 4), "successes": successes, "total": total,
                    "transport_retained_fraction": retained,
                    "widening": f"effective N {total:.0f} → {eff_n:.1f} (transport_risk={transport_risk})",
                    "rule": "reference-class mean preserved; precision discounted by transport-risk fraction",
                    "llm_role": "proposed reference-class descriptor + qualitative transport risk ONLY"})


def generic_lean_prior(lean: str, *, reason: str = "no reference class identified") -> PriorSpec:
    """The fallback prior when no reference class is available: the fixed qualitative-lean broad Beta. Labeled
    generic (high transport risk by construction) so it is NEVER mistaken for reference-class evidence."""
    a, b = LEAN_BETA.get(lean, (1.0, 1.0))
    return PriorSpec(
        family="beta", alpha=float(a), beta=float(b), source_class="generic_weakly_informative",
        reference_class="", transport_risk="high", retained_effective_n=0.0, raw_effective_n=0.0,
        provenance={"source": f"qualitative lean {lean!r} → fixed broad Beta({a},{b})", "reason": reason,
                    "class": "generic_weakly_informative",
                    "transport_risk": "high (no held-out-validated reference class)",
                    "llm_role": "proposed qualitative lean ONLY; numbers fixed"})


_REF_PROMPT = """You are picking a REFERENCE CLASS for a base rate. Do NOT give any numbers. Name the most apt
reference class of comparable past situations, and judge how well it TRANSPORTS to this specific scenario.
Reply ONLY JSON: {{"reference_class": "<short descriptor or empty if none apt>",
"transport_risk": "none|low|moderate|high|severe", "why": "<one line>"}}
Higher transport risk = the reference class differs more from THIS scenario (different era, population, regime).

QUESTION: {question}
SCENARIO CONTEXT: {context}"""


def propose_reference_class(question: str, *, llm, context: str = "") -> dict:
    """LLM proposes a reference-class descriptor + qualitative transport risk (semantic mapping only). Returns
    {reference_class, transport_risk, why}. Never returns a number. Safe defaults on failure."""
    if llm is None:
        return {"reference_class": "", "transport_risk": "high", "why": "no llm"}
    from swm.engine.grounding import parse_json
    try:
        raw = parse_json(llm(_REF_PROMPT.format(question=question, context=context or "n/a"))) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    tr = str(raw.get("transport_risk", "high"))
    return {"reference_class": str(raw.get("reference_class", ""))[:120],
            "transport_risk": tr if tr in TRANSPORT_RETAINED else "high",
            "why": str(raw.get("why", ""))[:160]}


def build_outcome_rate_prior(plan, *, llm=None, reference_data: dict = None) -> PriorSpec:
    """Construct the outcome-rate prior for a plan. If `reference_data` supplies {reference_class, successes,
    total} (DATA — e.g. from a dataset connector), build a transport-inflated reference-class prior; otherwise
    ask the LLM only to NAME a reference class + transport risk, and — absent numeric data — fall back to the
    generic lean prior (honestly labeled). The lean comes from the compiler provenance."""
    lean = str((plan.provenance or {}).get("outcome_lean", "neutral"))
    if reference_data and float(reference_data.get("total", 0) or 0) > 0:
        return reference_class_prior(
            str(reference_data.get("reference_class", "provided_reference_data")),
            float(reference_data.get("successes", 0) or 0), float(reference_data["total"]),
            transport_risk=str(reference_data.get("transport_risk", "high")), lean=lean)
    proposal = propose_reference_class(plan.question, llm=llm)
    # a named reference class with NO numeric data cannot mint a base rate — we keep the generic lean prior but
    # RECORD the proposed class + its transport risk in provenance (so a data connector can fill it in later).
    spec = generic_lean_prior(lean, reason="reference class proposed but no base-rate data available")
    spec.provenance["proposed_reference_class"] = proposal["reference_class"]
    spec.provenance["proposed_transport_risk"] = proposal["transport_risk"]
    spec.provenance["proposed_why"] = proposal["why"]
    return spec
