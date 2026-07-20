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
#: influence is DISCOUNTED BY EVIDENCE QUALITY, not flat: a recurrence COUNTED from sourced history may be
#: a relatively strong (not certain) prior; a base rate from the model's incomplete memory stays broad and
#: weakly weighted. An unsupported precise point estimate is never allowed to be tight (that is the whole
#: point of the cap). Keyed by the estimator's self-reported evidence quality.
QUALITY_MAX_N = {"sourced": 34.0, "model_memory": 8.0}
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
                            n_examples: float = 6.0, is_recurrence: bool = False,
                            evidence_quality: str = "model_memory", why: str = "") -> PriorSpec:
    """Build a CONTINUOUS Beta prior centred on a grounded base-rate ESTIMATE, with an effective sample size
    DISCOUNTED BY EVIDENCE QUALITY (not a flat heavy discount that would throw away real signal):

      * evidence_quality="sourced"       — the rate is counted from sourced history (retrieved past cases, a
        verified recurrence). May be a relatively STRONG (never certain) prior: cap QUALITY_MAX_N['sourced'].
      * evidence_quality="model_memory"  — the model's incomplete recall. BROAD, weakly weighted: cap
        QUALITY_MAX_N['model_memory'].

    In both cases the mean is continuous (off the 5-value lean grid) and transport risk still widens. This is
    distinct from `reference_class_prior` (explicit successes/total DATA). The LLM may PROPOSE the rate but its
    INFLUENCE depends on the evidence behind it — an unsupported precise number can never be tight."""
    p = min(0.98, max(0.02, float(base_rate)))
    retained = TRANSPORT_RETAINED.get(transport_risk, 0.4)
    cap = QUALITY_MAX_N.get(evidence_quality, MAX_LLM_EFFECTIVE_N)
    eff_n = min(cap, max(1.0, float(n_examples)) * retained)
    a = p * eff_n + 1.0
    b = (1.0 - p) * eff_n + 1.0
    src = "recurrence" if is_recurrence else "llm_estimated_reference"
    return PriorSpec(
        family="beta", alpha=round(a, 4), beta=round(b, 4), source_class=src,
        reference_class=reference_class, transport_risk=transport_risk,
        retained_effective_n=round(eff_n, 3), raw_effective_n=round(float(n_examples), 3),
        provenance={"estimated_base_rate": round(p, 4), "is_recurrence": is_recurrence,
                    "evidence_quality": evidence_quality, "n_examples_estimated": n_examples,
                    "transport_retained_fraction": retained,
                    "widening": f"effective N capped at {cap} (evidence_quality={evidence_quality}) → {eff_n:.1f}",
                    "why": why[:160],
                    "rule": "continuous grounded MEAN; precision scaled by EVIDENCE QUALITY, not flat",
                    "llm_role": "proposed reference class + base-rate estimate; influence set by evidence behind it"})


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

The single most important discipline: A PLAUSIBLE STORY IS NOT EVIDENCE THAT A SPECIFIC EVENT WILL HAPPEN
BEFORE A DEADLINE. Most specific proposed/attempted events do NOT complete within a short window. Raise the
probability only when comparable events COMMONLY happen within comparable time, OR this specific case is
materially advancing toward completion, OR it is a reliable recurrence.

QUESTION: {question}
AS-OF: {as_of}
TIME REMAINING until the deadline: about {horizon_days} days.
{recurrence_block}
You may PROPOSE a numerical base rate, but you must NEVER present an unsupported guess as precise historical
fact, and you must NEVER invent a numeric "completion percentage". Ground everything in comparable cases.

Decide:
1. reference_class: the most apt class of comparable past cases (short descriptor), or "" if none.
2. is_recurrence: true ONLY if the question hinges on a RELIABLY RECURRING or SCHEDULED event (an annual
   release/conference, a body that meets on a fixed calendar) whose next instance falls inside the window.
3. stage: the CURRENT qualitative stage of this specific event, from evidence/world-knowledge — one of:
   "mere_proposal_or_speculation" | "formally_initiated" | "scheduled_or_on_calendar" |
   "advanced_prerequisites_met" | "essentially_decided_awaiting_formality" | "recurring_due_this_window" |
   "blocked_or_stalled" | "unknown". Do NOT invent a number for it.
4. base_rate: among comparable past cases AT THIS SAME STAGE, the fraction that actually REACHED the YES
   outcome WITHIN about this much remaining time. Count cases that had not happened by a comparable horizon
   as NOT-yet (censored ⇒ they count toward "did not happen in time"), not as if they eventually succeeded.
   A vague proposal with a short deadline is usually LOW; an essentially-decided or recurring-due case is
   HIGH. Be honest and calibrated; do NOT default to 0.5 and do NOT default to optimism.
5. status_quo: one sentence on what happens if NO decisive change occurs before the deadline (delay,
   blockage, cancellation, unresolved process) — and whether the status quo is the likely outcome here.
6. n_examples: roughly how many comparable past cases at this stage your rate is based on (integer;
   conservative — count real instances, not vague impressions).
7. transport_risk: none|low|moderate|high|severe — how much this class differs from THIS scenario.
8. evidence_quality: "sourced" if you can point to specific comparable cases (named instances) you are
   confident about — else "model_memory" (a general impression).

Return ONLY JSON: {{"reference_class": "...", "is_recurrence": true|false, "stage": "...",
"base_rate": <0..1>, "status_quo": "...", "n_examples": <int>,
"transport_risk": "none|low|moderate|high|severe", "evidence_quality": "sourced|model_memory",
"why": "<one line grounding the rate in comparable cases at this stage within this window>"}}"""

#: stages where the deadline-conditioned base rate is the load-bearing signal (specific one-off events).
#: "recurring_due_this_window" and "essentially_decided_awaiting_formality" legitimately support HIGH rates;
#: the rest are where over-prediction of occurrence lives and the outside view must dominate.
_OCCURRENCE_STAGES = {"mere_proposal_or_speculation", "formally_initiated", "scheduled_or_on_calendar",
                      "advanced_prerequisites_met", "blocked_or_stalled", "unknown"}


def estimate_reference_base_rate(question: str, *, llm, as_of: str = "", horizon_days=None,
                                 recurrence: dict = None) -> dict:
    """DEADLINE-AWARE OUTSIDE-VIEW estimate (§8-9): name a reference class and estimate the base rate as the
    fraction of comparable cases AT THE SAME STAGE that reached YES WITHIN the remaining time — grounded in
    comparable-case timing, never a hardcoded decay or invented completion %. Returns the qualitative stage
    and the status-quo consideration too. The estimate is grounded world-knowledge, NOT held-out data — the
    caller discounts it by evidence quality (grounded_estimate_prior). Safe defaults on failure; {} when
    nothing usable."""
    if llm is None:
        return {}
    from swm.engine.grounding import parse_json
    rblock = ""
    if recurrence and recurrence.get("base_rate") is not None:
        rblock = (f"CALENDAR SIGNAL: a scheduled/recurring-event analysis suggests a base rate near "
                  f"{float(recurrence['base_rate']):.2f} (pattern strength {recurrence.get('strength', 0)}). "
                  f"Weigh this; override only with good reason.\n")
    hd = "unknown" if horizon_days is None else str(int(max(0, round(float(horizon_days)))))
    try:
        raw = parse_json(llm(_ESTIMATE_PROMPT.format(question=question, as_of=as_of or "n/a",
                                                     horizon_days=hd, recurrence_block=rblock))) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    if not raw or raw.get("base_rate") is None:
        return {}
    try:
        br = min(0.98, max(0.02, float(raw["base_rate"])))
    except (TypeError, ValueError):
        return {}
    tr = str(raw.get("transport_risk", "high"))
    eq = str(raw.get("evidence_quality", "model_memory")).lower()
    stage = str(raw.get("stage", "unknown")).lower().strip()
    return {"reference_class": str(raw.get("reference_class", ""))[:120],
            "is_recurrence": bool(raw.get("is_recurrence")),
            "stage": stage if stage else "unknown",
            "base_rate": br,
            "status_quo": str(raw.get("status_quo", ""))[:200],
            "deadline_conditioned": horizon_days is not None,
            "horizon_days": None if horizon_days is None else int(max(0, round(float(horizon_days)))),
            "n_examples": max(1.0, min(60.0, float(raw.get("n_examples", 5) or 5))),
            "transport_risk": tr if tr in TRANSPORT_RETAINED else "high",
            "evidence_quality": eq if eq in ("sourced", "model_memory") else "model_memory",
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
    # 3. DEADLINE-AWARE outside-view base-rate ESTIMATE (recurrence + stage + status-quo) → grounded prior.
    #    The remaining time is derived deterministically from the plan (horizon − as_of); the base rate is
    #    the fraction of comparable cases AT THIS STAGE that reached YES within that window (§8-9).
    horizon_days = None
    try:
        a0 = float(getattr(plan, "as_of", 0.0) or 0.0)
        h0 = float(getattr(plan, "horizon_ts", 0.0) or 0.0)
        if h0 > a0 > 0:
            horizon_days = (h0 - a0) / 86400.0
    except Exception:  # noqa: BLE001
        horizon_days = None
    est = estimate_reference_base_rate(plan.question, llm=llm,
                                       as_of=str((plan.provenance or {}).get("as_of", "")),
                                       horizon_days=horizon_days, recurrence=recurrence)
    if est:
        # a strong calendar recurrence DUE this window transports well (low risk); everything else keeps the
        # model's judged transport risk — no blanket optimism, no blanket pessimism.
        tr = "low" if (est["is_recurrence"] and est["transport_risk"] in ("moderate", "high")) \
            else est["transport_risk"]
        spec = grounded_estimate_prior(est["reference_class"], est["base_rate"], transport_risk=tr,
                                       n_examples=est["n_examples"], is_recurrence=est["is_recurrence"],
                                       evidence_quality=est.get("evidence_quality", "model_memory"),
                                       why=est["why"])
        spec.provenance.update({"stage": est.get("stage"), "status_quo": est.get("status_quo"),
                                "deadline_conditioned": est.get("deadline_conditioned"),
                                "horizon_days": est.get("horizon_days"),
                                "occurrence_class": est.get("stage") in _OCCURRENCE_STAGES})
        return spec
    # 4. weak fallback: the coarse qualitative-lean Beta (honestly labeled)
    spec = generic_lean_prior(lean, reason="no reference class or base-rate estimate available")
    return spec
