"""Nonlinear applicability & transport — Phase 7, Parts 14 + 15.

Two gates stand between "a validated nonlinear form exists" and "use it in THIS scenario":

  APPLICABILITY (Part 14): is the nonlinear phenomenon even supported here? Do the required history and
  context exist? Is the subgroup big enough? Does the time scale / network / institution context match? If
  the honest answer is "not enough support", the verdict is KEEP LINEAR — the flexible form is not chosen just
  because it exists.

  TRANSPORT (Part 15): nonlinear transport is especially dangerous. A spline or Hill curve fitted on one input
  range says nothing reliable about inputs outside that range; a threshold learned in one population may sit
  elsewhere in another. This classifies the move as interpolation / mild / strong / structural extrapolation
  and, for strong extrapolation, widens uncertainty, lowers support, or refuses the nonlinear transport.

Verdicts are typed so the compiler can act on them; nothing here silently upgrades a prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

APPLICABILITY_VERDICTS = ("nonlinear_applicable", "applicable_widened", "keep_linear",
                          "structural_alternatives", "experimental", "incompatible", "insufficient_support")
EXTRAPOLATION = ("interpolation", "mild_extrapolation", "strong_extrapolation", "structural_extrapolation",
                 "unsupported_regime_transfer")


@dataclass
class SupportBox:
    """The observed input support the form was fitted on (per input key: [lo, hi] and the fitted regime set)."""
    ranges: dict = field(default_factory=dict)       # {input_key: [lo, hi]}
    regimes: tuple = ()                              # regime ids the fit covered
    n_train: int = 0
    subgroup_min_n: int = 0

    def as_dict(self):
        return {"ranges": self.ranges, "regimes": list(self.regimes), "n_train": self.n_train,
                "subgroup_min_n": self.subgroup_min_n}


@dataclass
class ApplicabilityResult:
    verdict: str
    reasons: list = field(default_factory=list)
    extrapolation: str = "interpolation"
    uncertainty_widening: float = 1.0
    support_grade_delta: int = 0                      # 0 keep, −1 lower one grade, −2 lower two
    keep_linear_fallback: bool = False

    def as_dict(self):
        return {"verdict": self.verdict, "reasons": self.reasons, "extrapolation": self.extrapolation,
                "uncertainty_widening": round(self.uncertainty_widening, 3),
                "support_grade_delta": self.support_grade_delta,
                "keep_linear_fallback": self.keep_linear_fallback}


def classify_extrapolation(inputs: dict, support: SupportBox) -> tuple:
    """Return (level, detail) for how far the runtime inputs sit outside the fitted support."""
    worst, detail = "interpolation", []
    for key, rng in support.ranges.items():
        if key not in inputs or inputs[key] is None:
            continue
        x = float(inputs[key])
        lo, hi = float(rng[0]), float(rng[1])
        span = (hi - lo) or 1.0
        if lo <= x <= hi:
            continue
        over = (x - hi) / span if x > hi else (lo - x) / span
        detail.append({"input": key, "value": x, "range": [lo, hi], "overshoot_frac": round(over, 3)})
        lvl = "mild_extrapolation" if over <= 0.25 else "strong_extrapolation"
        worst = _worse(worst, lvl)
    return worst, detail


def _worse(a, b):
    return a if EXTRAPOLATION.index(a) >= EXTRAPOLATION.index(b) else b


def evaluate_applicability(*, phenomenon_supported: bool, history_available: bool,
                           context_available: bool, inputs: dict, support: SupportBox,
                           subgroup_n: int | None = None, regime_known: bool = True,
                           time_scale_match: bool = True, monotonicity_ok: bool = True,
                           has_structural_alternatives: bool = False) -> ApplicabilityResult:
    """The Part-14 gate. Conservative by construction: any missing precondition drops to keep_linear rather
    than pushing a flexible form onto unsupported ground."""
    reasons = []
    # hard incompatibilities first
    if not phenomenon_supported:
        return ApplicabilityResult("keep_linear", ["nonlinear phenomenon not supported by evidence here"],
                                   keep_linear_fallback=True, support_grade_delta=0)
    if not monotonicity_ok:
        return ApplicabilityResult("incompatible",
                                   ["required monotonicity assumption violated in this context"],
                                   keep_linear_fallback=True)
    if not time_scale_match:
        reasons.append("time-scale mismatch vs fitted pack")
    # extrapolation
    level, detail = classify_extrapolation(inputs, support)
    widen, grade = 1.0, 0
    if level == "mild_extrapolation":
        widen, reasons = 1.3, reasons + [f"mild extrapolation: {detail}"]
    elif level == "strong_extrapolation":
        widen, grade = 1.8, -1
        reasons.append(f"STRONG extrapolation beyond fitted support: {detail}")
    # missing preconditions → prefer the simpler form
    if not history_available:
        reasons.append("required event history unavailable → history terms cannot fire")
    if not context_available:
        reasons.append("required context unavailable")
    if subgroup_n is not None and subgroup_n < max(1, support.subgroup_min_n):
        reasons.append(f"subgroup n={subgroup_n} < min {support.subgroup_min_n} → shrink to pooled/linear")
    if not regime_known:
        reasons.append("regime identity uncertain → marginalize or widen")
        widen = max(widen, 1.5)
    # decide
    if not history_available and not context_available:
        return ApplicabilityResult("keep_linear", reasons or ["no history/context to condition on"],
                                   extrapolation=level, uncertainty_widening=widen,
                                   support_grade_delta=grade, keep_linear_fallback=True)
    if level == "strong_extrapolation":
        v = "applicable_widened" if (history_available or context_available) else "insufficient_support"
        return ApplicabilityResult(v, reasons, extrapolation=level, uncertainty_widening=widen,
                                   support_grade_delta=grade, keep_linear_fallback=True)
    if has_structural_alternatives:
        return ApplicabilityResult("structural_alternatives",
                                   reasons + ["≥2 forms plausible — branch, do not collapse"],
                                   extrapolation=level, uncertainty_widening=widen)
    if reasons:
        return ApplicabilityResult("applicable_widened", reasons, extrapolation=level,
                                   uncertainty_widening=widen, support_grade_delta=grade,
                                   keep_linear_fallback=True)
    return ApplicabilityResult("nonlinear_applicable", ["within support; phenomenon + history + context ok"],
                               extrapolation=level, uncertainty_widening=widen)


def transport_check(*, input_support_overlap: float, threshold_shift: bool, regime_mismatch: bool,
                    population_mismatch: bool, platform_mismatch: bool, outcome_def_mismatch: bool) -> dict:
    """Part-15 transport verdict for moving a fitted nonlinear pack to a new domain. Overlap in [0,1]."""
    flags = []
    widen = 1.0
    if input_support_overlap < 0.5:
        flags.append("low input-support overlap"); widen *= 1.5
    if threshold_shift:
        flags.append("threshold location likely shifts across domains"); widen *= 1.3
    if regime_mismatch:
        flags.append("regime mismatch — unsupported regime transfer"); widen *= 1.6
    if population_mismatch:
        flags.append("population mismatch"); widen *= 1.2
    if platform_mismatch:
        flags.append("platform mismatch"); widen *= 1.2
    if outcome_def_mismatch:
        flags.append("outcome definition mismatch — comparison may be invalid"); widen *= 1.4
    transportable = input_support_overlap >= 0.5 and not (regime_mismatch or outcome_def_mismatch)
    return {"transportable": transportable, "uncertainty_widening": round(min(widen, 3.0), 3),
            "flags": flags,
            "recommendation": ("transport with widened uncertainty" if transportable and flags
                               else "transportable" if transportable
                               else "refuse nonlinear transport — keep linear / domain-restrict")}
