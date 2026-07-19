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
#: an LLM-ESTIMATED base rate (no held-out data) is capped MUCH tighter — it grounds the prior MEAN
#: continuously but stays weakly-informative, so it can never masquerade as a data-backed prior.
MAX_LLM_EFFECTIVE_N = 10.0
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


def grounded_estimate_prior(reference_class: str, base_rate: float, *, transport_risk: str = "moderate",
                            n_examples: float = 6.0, is_recurrence: bool = False, why: str = "") -> PriorSpec:
    """Build a CONTINUOUS Beta prior centred on an LLM-ESTIMATED base rate (or a calendar recurrence rate),
    with a bounded, transport-discounted effective sample size. Distinct from `reference_class_prior`
    (held-out DATA): here the base rate is a grounded ESTIMATE, so it is capped at MAX_LLM_EFFECTIVE_N and
    labelled `llm_estimated_reference`/`recurrence` — it moves the prior MEAN off the coarse 5-value lean
    grid, but stays weakly-informative and never a data-backed certainty. A strong recurrence (annual event
    that reliably happens) legitimately carries low transport risk and a high base rate."""
    p = min(0.98, max(0.02, float(base_rate)))
    retained = TRANSPORT_RETAINED.get(transport_risk, 0.4)
    eff_n = min(MAX_LLM_EFFECTIVE_N, max(1.0, float(n_examples)) * retained)
    a = p * eff_n + 1.0
    b = (1.0 - p) * eff_n + 1.0
    src = "recurrence" if is_recurrence else "llm_estimated_reference"
    return PriorSpec(
        family="beta", alpha=round(a, 4), beta=round(b, 4), source_class=src,
        reference_class=reference_class, transport_risk=transport_risk,
        retained_effective_n=round(eff_n, 3), raw_effective_n=round(float(n_examples), 3),
        provenance={"estimated_base_rate": round(p, 4), "is_recurrence": is_recurrence,
                    "n_examples_estimated": n_examples, "transport_retained_fraction": retained,
                    "widening": f"effective N capped at {MAX_LLM_EFFECTIVE_N} → {eff_n:.1f}",
                    "why": why[:160],
                    "rule": "continuous grounded MEAN, bounded precision — an ESTIMATE, not held-out data",
                    "llm_role": "proposed reference class + bounded base-rate estimate + transport risk"})


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


_ESTIMATE_PROMPT = """You are grounding the STARTING probability (base rate) for a forecasting question in a
reference class of comparable past situations. Reason like a superforecaster's OUTSIDE VIEW.

QUESTION: {question}
AS-OF: {as_of}
{recurrence_block}
Decide:
1. reference_class: the most apt class of comparable past cases (short descriptor), or "" if none.
2. is_recurrence: true if the question hinges on a RELIABLY RECURRING or SCHEDULED event (an annual
   release/conference, a body that meets on a fixed calendar, a regular filing). Such events have a HIGH,
   well-identified base rate.
3. base_rate: the historical frequency the YES outcome occurs in that class, 0..1. For a strong recurrence
   this is high (e.g. an annual product cycle that has happened every year ≈ 0.9-0.97). Be honest and
   calibrated; do NOT default to 0.5.
4. n_examples: roughly how many past comparable cases your rate is based on (integer; be conservative).
5. transport_risk: none|low|moderate|high|severe — how much this class differs from THIS scenario
   (a reliable recurrence with no disruption = low; a loose analogy across eras = high).

Return ONLY JSON: {{"reference_class": "...", "is_recurrence": true|false, "base_rate": <0..1>,
"n_examples": <int>, "transport_risk": "none|low|moderate|high|severe", "why": "<one line>"}}"""


def estimate_reference_base_rate(question: str, *, llm, as_of: str = "", recurrence: dict = None) -> dict:
    """LLM OUTSIDE-VIEW estimate: name a reference class and estimate its base rate (bounded), flag
    recurrences, judge transport risk. The estimate is grounded world-knowledge, NOT held-out data — the
    caller discounts it heavily (grounded_estimate_prior). A `recurrence` hint from the calendar layer
    (e.g. {"base_rate": 0.9, "strength": 0.8}) is passed to the model as prior context. Safe defaults on
    failure; returns {} when nothing usable."""
    if llm is None:
        return {}
    from swm.engine.grounding import parse_json
    rblock = ""
    if recurrence and recurrence.get("base_rate") is not None:
        rblock = (f"CALENDAR SIGNAL: a scheduled/recurring-event analysis suggests a base rate near "
                  f"{float(recurrence['base_rate']):.2f} (pattern strength {recurrence.get('strength', 0)}). "
                  f"Weigh this; override only with good reason.\n")
    try:
        raw = parse_json(llm(_ESTIMATE_PROMPT.format(question=question, as_of=as_of or "n/a",
                                                     recurrence_block=rblock))) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    if not raw or raw.get("base_rate") is None:
        return {}
    try:
        br = min(0.98, max(0.02, float(raw["base_rate"])))
    except (TypeError, ValueError):
        return {}
    tr = str(raw.get("transport_risk", "high"))
    return {"reference_class": str(raw.get("reference_class", ""))[:120],
            "is_recurrence": bool(raw.get("is_recurrence")),
            "base_rate": br,
            "n_examples": max(1.0, min(60.0, float(raw.get("n_examples", 5) or 5))),
            "transport_risk": tr if tr in TRANSPORT_RETAINED else "high",
            "why": str(raw.get("why", ""))[:160]}


def build_outcome_rate_prior(plan, *, llm=None, reference_data: dict = None, recurrence: dict = None,
                             curated_lookup=None) -> PriorSpec:
    """Construct the outcome-rate prior for a plan, GROUNDED where possible (evidence order, best first):

      1. `reference_data` {reference_class, successes, total}  — held-out DATA (connector) → tightest.
      2. `curated_lookup(plan)` → curated reference-class table row (real data) → transport-inflated Beta.
      3. `recurrence` hint (calendar) + LLM outside-view ESTIMATE → continuous grounded prior (bounded).
      4. generic qualitative-lean Beta — the weak 5-value fallback ONLY when nothing can be grounded.

    Steps 1-3 give a CONTINUOUS, evidence-grounded mean; step 4 is the coarse LLM-lean guess kept only as a
    last resort. The lean comes from compiler provenance."""
    lean = str((plan.provenance or {}).get("outcome_lean", "neutral"))
    # 1. real reference DATA (a dataset/connector supplied successes/total)
    if reference_data and float(reference_data.get("total", 0) or 0) > 0:
        return reference_class_prior(
            str(reference_data.get("reference_class", "provided_reference_data")),
            float(reference_data.get("successes", 0) or 0), float(reference_data["total"]),
            transport_risk=str(reference_data.get("transport_risk", "high")), lean=lean)
    # 2. curated reference-class table (real data, human-vetted)
    if curated_lookup is not None:
        try:
            row = curated_lookup(plan)
        except Exception:  # noqa: BLE001
            row = None
        if row and float(row.get("total", 0) or 0) > 0:
            return reference_class_prior(
                str(row.get("reference_class", "curated_reference")),
                float(row.get("successes", 0) or 0), float(row["total"]),
                transport_risk=str(row.get("transport_risk", "moderate")), lean=lean)
    # 3. LLM outside-view base-rate ESTIMATE (recurrence-aware) → continuous grounded prior
    est = estimate_reference_base_rate(plan.question, llm=llm,
                                       as_of=str((plan.provenance or {}).get("as_of", "")),
                                       recurrence=recurrence)
    if est:
        # a strong calendar recurrence sets a low transport floor (a reliable annual event transports well)
        tr = "low" if (est["is_recurrence"] and est["transport_risk"] in ("moderate", "high")) \
            else est["transport_risk"]
        return grounded_estimate_prior(est["reference_class"], est["base_rate"], transport_risk=tr,
                                       n_examples=est["n_examples"], is_recurrence=est["is_recurrence"],
                                       why=est["why"])
    # 4. weak fallback: the coarse qualitative-lean Beta (honestly labeled)
    spec = generic_lean_prior(lean, reason="no reference class or base-rate estimate available")
    return spec
