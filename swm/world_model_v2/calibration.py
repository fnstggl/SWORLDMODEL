"""Calibration, abstention, uncertainty decomposition and the direct-forecast critic — Phase 12 (Tier G).

Every V2 forecast passes through here to become a user-facing result. The contract:
  raw distribution + context → calibrated distribution + confidence grade + abstention decision +
  uncertainty decomposition + sensitivity contributors + omitted high-impact variables +
  structural disagreement + direct-model disagreement + calibration provenance.

CALIBRATION GOVERNANCE: calibrators are fitted on a TRAIN/VAL split ONLY (fit_* never sees test); each
carries its fit provenance and is versioned. Conditioned calibration keys on domain/horizon/mechanism-
status with partial pooling to the global calibrator when a cell is small.

ABSTENTION is signal-driven, not a fixed threshold: unsupported high-sensitivity variables, weak evidence
(leakage grade / few items), structural-hypothesis disagreement, no applicable validated mechanism,
out-of-distribution, poor expected calibration. Graded outputs: supported / supported_with_limitations /
low_confidence / abstain / unresolvable.

The direct-forecast CRITIC compares the simulation to grounded-direct and ensemble baselines on identical
evidence; it flags disagreement and possible failure modes but NEVER overwrites the simulation number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ------------------------------------------------------------------ calibrators (fit on train/val only)
def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0
    fitted_on: str = ""
    version: str = "1.0"

    def apply(self, p: float) -> float:
        return _sig(self.a * _logit(p) + self.b)


def fit_platt(pairs, *, iters=500, lr=0.2, l2=1e-3, fitted_on="") -> PlattCalibrator:
    """Logistic (Platt) calibration on (p, y). Shrinks toward identity (a=1,b=0) via L2 so a small cell
    cannot overfit. Train/val pairs ONLY."""
    if not pairs:
        return PlattCalibrator(fitted_on="empty")
    a, b, n = 1.0, 0.0, len(pairs)
    for _ in range(iters):
        ga, gb = 0.0, 0.0
        for p, y in pairs:
            z = a * _logit(p) + b
            e = _sig(z) - y
            ga += e * _logit(p)
            gb += e
        a -= lr * (ga / n + l2 * (a - 1.0))
        b -= lr * (gb / n + l2 * b)
    return PlattCalibrator(a=round(a, 5), b=round(b, 5), n_fit=n, fitted_on=fitted_on)


@dataclass
class IsotonicCalibrator:
    """Isotonic regression (pool-adjacent-violators) — monotone, nonparametric. Best with more data."""
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    n_fit: int = 0
    fitted_on: str = ""
    version: str = "1.0"

    def apply(self, p: float) -> float:
        if not self.xs:
            return p
        import bisect
        i = bisect.bisect_left(self.xs, p)
        if i == 0:
            return self.ys[0]
        if i >= len(self.xs):
            return self.ys[-1]
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        t = (p - x0) / max(1e-9, x1 - x0)
        return y0 + t * (y1 - y0)


def fit_isotonic(pairs, *, fitted_on="") -> IsotonicCalibrator:
    if len(pairs) < 10:
        return IsotonicCalibrator(fitted_on="too_small")
    pts = sorted(pairs)
    xs = [p for p, _ in pts]
    ys = [float(y) for _, y in pts]
    # PAV
    w = [1.0] * len(ys)
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            new = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            ys[i] = new
            del ys[i + 1]
            wi = w[i] + w[i + 1]
            del w[i + 1]
            xs_merge = xs[i]
            del xs[i + 1]
            xs[i] = xs_merge
            w[i] = wi
            if i > 0:
                i -= 1
        else:
            i += 1
    return IsotonicCalibrator(xs=xs, ys=ys, n_fit=len(pairs), fitted_on=fitted_on)


@dataclass
class ConditionedCalibrator:
    """Calibration keyed on a context cell (domain/horizon/mechanism-status), each cell a Platt calibrator,
    with partial pooling to a global calibrator when the cell is small (n < min_cell)."""
    cells: dict = field(default_factory=dict)                 # key -> PlattCalibrator
    global_cal: PlattCalibrator = field(default_factory=PlattCalibrator)
    min_cell: int = 30
    version: str = "1.0"

    def apply(self, p: float, key: str) -> float:
        cal = self.cells.get(key)
        if cal is None or cal.n_fit < self.min_cell:
            g = self.global_cal.apply(p)
            if cal is None:
                return g
            w = cal.n_fit / self.min_cell                     # blend toward global for small cells
            return w * cal.apply(p) + (1 - w) * g
        return cal.apply(p)

    def provenance(self, key: str) -> dict:
        cal = self.cells.get(key)
        return {"cell": key, "cell_n": (cal.n_fit if cal else 0),
                "used": ("cell" if cal and cal.n_fit >= self.min_cell else "pooled_or_global"),
                "global_n": self.global_cal.n_fit, "version": self.version}


def fit_conditioned(pairs_with_key, *, min_cell=30, fitted_on="") -> ConditionedCalibrator:
    """pairs_with_key: [(p, y, key)]. Fits a global Platt + per-key Platt. Train/val only."""
    by_key = {}
    for p, y, k in pairs_with_key:
        by_key.setdefault(k, []).append((p, y))
    cells = {k: fit_platt(v, fitted_on=f"{fitted_on}:{k}") for k, v in by_key.items()}
    g = fit_platt([(p, y) for p, y, _ in pairs_with_key], fitted_on=f"{fitted_on}:global")
    return ConditionedCalibrator(cells=cells, global_cal=g, min_cell=min_cell)


# ------------------------------------------------------------------ calibration metrics
def ece(pairs, *, bins=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> float:
    """Expected calibration error over confidence buckets (weighted by bucket mass)."""
    if not pairs:
        return None
    n = len(pairs)
    tot = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [(p, y) for p, y in pairs if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        tot += (len(b) / n) * abs(conf - acc)
    return round(tot, 4)


def reliability_table(pairs, *, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list:
    """Per-bucket (predicted, observed, n) — the reliability diagram data. Buckets per the spec."""
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [(p, y) for p, y in pairs if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if b:
            out.append({"bucket": f"{lo:.0%}-{hi:.0%}", "mean_pred": round(sum(p for p, _ in b) / len(b), 3),
                        "observed_freq": round(sum(y for _, y in b) / len(b), 3), "n": len(b)})
    return out


# ------------------------------------------------------------------ uncertainty decomposition
def decompose_uncertainty(branches, *, structural_posterior=None, evidence_grade="") -> dict:
    """Decompose predictive uncertainty from the rollout's terminal branches (weighted). Reports the
    variance contributed by within-world randomness (aleatory), between-particle spread (state/parameter),
    and structural-hypothesis disagreement (model) — reported SEPARATELY, never merged into one number."""
    import statistics
    # binary/categorical: use the terminal outcome frequency spread; continuous: variance
    vals = []
    for b in branches:
        um = getattr(b.world, "uncertainty_meta", {}) or {}
        vals.append((b.weight, um))
    n = len(branches) or 1
    out = {"n_particles": n}
    # structural (model) uncertainty from the hypothesis posterior entropy, if present
    if structural_posterior:
        ps = list(structural_posterior.values())
        ent = -sum(p * math.log(max(1e-9, p)) for p in ps)
        out["structural_model_uncertainty"] = round(ent / math.log(max(2, len(ps))), 4)  # normalized [0,1]
        out["structural_posterior"] = structural_posterior
    else:
        out["structural_model_uncertainty"] = 0.0
    # state/parameter spread: dispersion of sampled latents across particles (mean normalized sd)
    latent_series = {}
    for b in branches:
        for k, v in (getattr(b.world, "uncertainty_meta", {}).get("sampled", {}) or {}).items():
            if isinstance(v, (int, float)):
                latent_series.setdefault(k, []).append(float(v))
    disp = []
    for k, xs in latent_series.items():
        if len(xs) > 1:
            m = sum(xs) / len(xs)
            sd = statistics.pstdev(xs)
            rng = (max(xs) - min(xs)) or 1.0
            disp.append(sd / rng)
    out["state_parameter_uncertainty"] = round(sum(disp) / len(disp), 4) if disp else 0.0
    out["evidence_uncertainty"] = {"A (leaked)": 1.0, "F (leaked)": 1.0}.get(evidence_grade, None) or (
        {"A (mostly server-verified timestamps)": 0.1, "B (mixed timestamp basis)": 0.3,
         "C (claimed timestamps only)": 0.5, "C (nonzero as-of slack)": 0.6}.get(evidence_grade, 0.5))
    return out


# ------------------------------------------------------------------ support grading (no-abstention)
#: The new axes live in swm/world_model_v2/result.py. This module maps epistemic SIGNALS to a support
#: grade + recommendation + limitations — signals NEVER refuse a forecast, they downgrade support.
def grade_support(*, worst_mechanism_tier: int = 6, structural_model_uncertainty: float = 0.0,
                  n_evidence_items: int = 0, unsupported_high_sensitivity: int = 0,
                  transport_risk: str = "medium", intervention_requested: bool = False) -> dict:
    """Map signals → {support_grade, recommendation_status, limitations}. Support grade follows the
    weakest mechanism tier; structural disagreement / thin evidence / unsupported high-sensitivity
    variables add limitations and can pull the grade down one step, but NEVER produce a refusal."""
    from swm.world_model_v2.fallback import TIER_SUPPORT_GRADE
    grade = TIER_SUPPORT_GRADE.get(worst_mechanism_tier, "highly_speculative")
    order = ["empirically_supported", "transfer_supported", "exploratory", "highly_speculative"]
    idx = order.index(grade)
    limitations = []
    if structural_model_uncertainty > 0.6:
        limitations.append("structural hypotheses disagree materially")
        idx = min(len(order) - 1, idx + 1)
    if unsupported_high_sensitivity > 0:
        limitations.append(f"{unsupported_high_sensitivity} high-sensitivity variable(s) supported only "
                           f"by broad priors")
    if n_evidence_items == 0:
        limitations.append("no admissible as-of evidence — prior-driven simulation")
    if transport_risk == "high" and idx < order.index("exploratory"):
        limitations.append("parameters transported with widened uncertainty")
    grade = order[idx]
    if not intervention_requested:
        rec = "not_requested"
    elif grade == "empirically_supported":
        rec = "eligible"
    elif grade == "transfer_supported":
        rec = "limited"
    else:
        rec = "withheld"
    return {"support_grade": grade, "recommendation_status": rec, "limitations": limitations}


# ------------------------------------------------------------------ DEPRECATED abstention policy
# Retained for backward compatibility + its tests. NOT used on the no-abstention production path — the
# pipeline derives support grade from the plan's mechanism tiers (fallback.overall_support_grade) and uses
# grade_support() above. `decide_abstention` no longer gates whether a forecast is produced.
ABSTAIN_GRADES = ("supported", "supported_with_limitations", "low_confidence", "abstain", "unresolvable")


@dataclass
class AbstentionDecision:
    grade: str
    reasons: list = field(default_factory=list)
    triggered_signals: dict = field(default_factory=dict)

    def as_dict(self):
        return {"grade": self.grade, "reasons": self.reasons, "signals": self.triggered_signals}


def decide_abstention(*, unsupported_high_sensitivity: int = 0, evidence_grade: str = "",
                      n_evidence_items: int = 0, structural_model_uncertainty: float = 0.0,
                      has_applicable_validated_mechanism: bool = True, out_of_distribution: bool = False,
                      compiler_abstained: bool = False, unresolved_terminal_share: float = 0.0,
                      transport_risk: str = "medium") -> AbstentionDecision:
    """Signal-driven grading. Any hard signal → abstain/unresolvable; soft signals downgrade support."""
    signals = {"unsupported_high_sensitivity": unsupported_high_sensitivity,
               "evidence_grade": evidence_grade, "n_evidence_items": n_evidence_items,
               "structural_model_uncertainty": round(structural_model_uncertainty, 3),
               "has_applicable_validated_mechanism": has_applicable_validated_mechanism,
               "out_of_distribution": out_of_distribution, "compiler_abstained": compiler_abstained,
               "unresolved_terminal_share": round(unresolved_terminal_share, 3),
               "transport_risk": transport_risk}
    reasons = []
    if compiler_abstained:
        return AbstentionDecision("unresolvable", ["compiler could not type the world slice"], signals)
    if unresolved_terminal_share > 0.5:
        return AbstentionDecision("abstain",
                                  [f"{unresolved_terminal_share:.0%} of terminal worlds unresolved — the "
                                   "causal chain likely did not execute"], signals)
    if str(evidence_grade).startswith("F"):
        return AbstentionDecision("abstain", ["evidence failed the leakage audit"], signals)
    if not has_applicable_validated_mechanism:
        reasons.append("no applicable VALIDATED mechanism — only experimental/implemented forms available")
    if unsupported_high_sensitivity > 0:
        reasons.append(f"{unsupported_high_sensitivity} high-sensitivity variable(s) unsupported by evidence")
    if out_of_distribution:
        reasons.append("scenario is out of the validated distribution")
    if structural_model_uncertainty > 0.6:
        reasons.append("structural hypotheses disagree materially")
    if n_evidence_items == 0:
        reasons.append("no admissible as-of evidence — prediction is prior-only")
    # grade from the count/severity of soft signals
    hard = sum([not has_applicable_validated_mechanism, out_of_distribution,
                structural_model_uncertainty > 0.6, unsupported_high_sensitivity >= 2])
    if hard >= 2:
        return AbstentionDecision("abstain", reasons or ["multiple support signals failed"], signals)
    if hard == 1 or reasons:
        return AbstentionDecision("low_confidence" if hard else "supported_with_limitations",
                                  reasons, signals)
    if transport_risk == "high":
        return AbstentionDecision("supported_with_limitations",
                                  ["parameters transported with widened uncertainty (high transport risk)"],
                                  signals)
    return AbstentionDecision("supported", [], signals)


# ------------------------------------------------------------------ direct-forecast critic
@dataclass
class CriticReport:
    v2_p: float
    direct_p: float | None
    ensemble_p: float | None
    disagreement: float | None
    flags: list = field(default_factory=list)

    def as_dict(self):
        return {"v2_p": self.v2_p, "direct_p": self.direct_p, "ensemble_p": self.ensemble_p,
                "disagreement": self.disagreement, "flags": self.flags}


def run_critic(v2_p: float, *, direct_p=None, ensemble_p=None, v2_sharpness=None) -> CriticReport:
    """Compare the simulation to grounded-direct / ensemble baselines on identical evidence. Flags
    disagreement and possible failure modes. NEVER overwrites v2_p (the simulation stays the answer)."""
    flags = []
    dis = None
    refs = [x for x in (direct_p, ensemble_p) if x is not None]
    if refs:
        dis = max(abs(v2_p - r) for r in refs)
        if dis > 0.3:
            flags.append(f"large disagreement with direct/ensemble ({dis:.2f}) — check compiler/mechanisms")
        if all(abs(v2_p - 0.5) > abs(r - 0.5) for r in refs) and abs(v2_p - 0.5) > 0.4:
            flags.append("V2 much sharper than direct baselines — verify the sharpness is earned")
    if v2_sharpness is not None and v2_sharpness > 0.45 and not refs:
        flags.append("sharp V2 forecast with no baseline to cross-check")
    return CriticReport(v2_p=v2_p, direct_p=direct_p, ensemble_p=ensemble_p,
                        disagreement=(round(dis, 3) if dis is not None else None), flags=flags)


# ------------------------------------------------------------------ the user-facing result contract
def build_result(raw_p, *, calibrator=None, cal_key="", branches=None, structural_posterior=None,
                 evidence_grade="", abstention: AbstentionDecision = None, critic: CriticReport = None,
                 sensitivity=None, omitted_high_impact=None) -> dict:
    """Assemble the full result contract. raw_p is the terminal-state probability (binary) or None."""
    cal_p = calibrator.apply(raw_p, cal_key) if (calibrator is not None and raw_p is not None
                                                 and hasattr(calibrator, "apply")) else raw_p
    return {
        "raw_probability": raw_p,
        "calibrated_probability": round(cal_p, 4) if cal_p is not None else None,
        "confidence_grade": abstention.grade if abstention else "supported",
        "abstention": abstention.as_dict() if abstention else None,
        "uncertainty_decomposition": (decompose_uncertainty(branches, structural_posterior=structural_posterior,
                                                            evidence_grade=evidence_grade)
                                      if branches else None),
        "sensitivity_contributors": sensitivity or [],
        "omitted_high_impact_variables": omitted_high_impact or [],
        "structural_disagreement": structural_posterior,
        "direct_model_disagreement": critic.as_dict() if critic else None,
        "calibration_provenance": (calibrator.provenance(cal_key) if hasattr(calibrator, "provenance")
                                   else {"calibrator": type(calibrator).__name__ if calibrator else None}),
    }
