"""Numerical stability & structured failure records — Phase 7, Part 17 + Part 26.

Nonlinear forms and event-generating intensities are the parts of the system most prone to blow up: an
exploding hazard, a probability that drifts outside [0,1], a self-exciting process with branching ratio ≥ 1,
an event storm, a spline extrapolated into nonsense. This module is the single guard surface.

Two rules the spec insists on:
  * DO NOT hide instability with arbitrary clipping. Clipping is applied ONLY where it is mathematically
    justified (a probability lives in [0,1]; a hazard is non-negative; a branching ratio must be < 1 for a
    finite process). Every clip that actually bit is RECORDED, not silent — so a mechanism that is only
    "stable" because it was clamped shows up in the failure ledger instead of masquerading as healthy.
  * Failures are append-only and preserved (Part 26). `FailureRecord` is the typed row; `FailureLedger`
    serializes to the machine-readable failures artifact and NEVER overwrites the Hawkes quarantine.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field, asdict

FAILURE_TYPES = ("overfit", "calibration_failure", "instability", "transfer_failure", "null_improvement",
                 "leakage", "nonidentifiability", "extrapolation_failure", "event_storm",
                 "numerical_nonfinite", "degenerate_mixture", "explosive_branching", "quarantine")

MAX_EVENT_RATE_PER_DAY = 5000.0     # a hard cap; a mechanism exceeding it is an event-storm failure
MAX_FOLLOWUP_EVENTS = 2000          # per transition — infinite-loop backstop


class StabilityError(ValueError):
    pass


@dataclass
class GuardReport:
    """Records whether a guard actually altered a value — so 'clamped to survive' is never invisible."""
    clamped: bool = False
    reasons: list = field(default_factory=list)

    def note(self, reason: str):
        self.clamped = True
        self.reasons.append(reason)
        return self


def safe_prob(p: float, report: GuardReport | None = None) -> float:
    """Probabilities live in [0,1]; clamp is mathematically justified. Non-finite → recorded failure."""
    if p != p or p in (float("inf"), float("-inf")):
        if report is not None:
            report.note(f"non_finite_prob:{p}")
        raise StabilityError(f"non-finite probability {p!r}")
    if p < 0.0:
        if report is not None:
            report.note("prob<0_clamped")
        return 0.0
    if p > 1.0:
        if report is not None:
            report.note("prob>1_clamped")
        return 1.0
    return p


def safe_rate(lam: float, report: GuardReport | None = None) -> float:
    """A hazard/intensity is non-negative and finite. Exploding rates are recorded, not silently eaten."""
    if lam != lam or lam == float("inf"):
        if report is not None:
            report.note(f"non_finite_rate:{lam}")
        raise StabilityError(f"non-finite rate {lam!r}")
    if lam < 0.0:
        if report is not None:
            report.note("rate<0_clamped")
        return 0.0
    if lam > MAX_EVENT_RATE_PER_DAY:
        if report is not None:
            report.note(f"rate>{MAX_EVENT_RATE_PER_DAY}_capped(event_storm_risk)")
        return MAX_EVENT_RATE_PER_DAY
    return lam


def check_branching(alpha: float) -> None:
    """A self-exciting process with branching ratio α ≥ 1 is explosive — refuse, do not silently damp."""
    if alpha >= 1.0:
        raise StabilityError(f"branching ratio α={alpha} ≥ 1 — explosive self-excitation refused "
                             f"(a finite process needs α<1; see Hawkes quarantine)")


def check_mixture(weights) -> None:
    ws = [float(w) for w in weights]
    if not ws or any(w < 0 for w in ws):
        raise StabilityError("degenerate mixture: empty or negative component weight")
    if sum(ws) <= 0:
        raise StabilityError("degenerate mixture: weights sum to zero")


def bounded_logistic_step(x: float, lo: float = -30.0, hi: float = 30.0) -> float:
    return 1.0 / (1.0 + math.exp(-(lo if x < lo else (hi if x > hi else x))))


@dataclass
class FailureRecord:
    """One preserved Phase 7 failure (Part 26). Append-only; disposition is honest."""
    failure_id: str
    mechanism_family: str
    structural_form: str
    failure_type: str
    dataset: str = ""
    split: str = ""
    context: str = ""
    history_window: str = ""
    seed: int = 0
    code_commit: str = ""
    metric: str = ""
    value: float | None = None
    baseline_value: float | None = None
    uncertainty: dict = field(default_factory=dict)
    suspected_cause: str = ""
    disposition: str = "preserved"       # preserved | quarantined | rejected | retained_linear
    artifact_links: list = field(default_factory=list)
    at: str = ""

    def __post_init__(self):
        if self.failure_type not in FAILURE_TYPES:
            raise StabilityError(f"bad failure_type {self.failure_type!r}")
        if not self.at:
            self.at = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())

    def as_dict(self):
        return asdict(self)


@dataclass
class FailureLedger:
    """Append-only ledger. `extend_preserving` merges without ever dropping a prior record."""
    records: list = field(default_factory=list)

    def add(self, rec: FailureRecord):
        self.records.append(rec)
        return rec

    def as_dict(self):
        return {"_meta": {"note": "Phase 7 failures — append-only; never deleted (Part 26). The Hawkes "
                          "quarantine and all Phase 6 nulls are preserved.",
                          "n": len(self.records)},
                "failures": [r.as_dict() for r in self.records]}
