"""Nonlinear composition & stability — Phase 7, Part 16.

Nonlinear mechanisms are far more dangerous to compose than linear ones: two saturating terms can double-count
the same ceiling; two fatigue terms can drive a response negative; a self-exciting intensity feeding its own
follow-up events can run away; two thresholds can fight; incompatible monotonicities can cancel into nonsense.
This module DETECTS those hazards from the mechanism instances' declared phenomena BEFORE rollout, orders
execution, and provides the rollout-time stability checks (bounded feedback, event-rate cap, repeated-rollout
divergence) that keep a nonlinear world from exploding.

It reasons over the forms' `phenomena` tags (from `forms.py`) plus each instance's target state path, so it
never needs to execute a mechanism to know two of them both saturate the same quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.nonlinear.forms import get_form


@dataclass
class MechanismSlot:
    """One nonlinear mechanism instance about to enter a world, as composition sees it."""
    mechanism_id: str
    form_id: str
    target_path: str                     # the state/quantity it writes
    inputs: tuple = ()                   # input keys it reads
    precedence: int = 100                # lower runs first
    phenomena: tuple = ()                # filled from the form if empty


def _phenomena(slot: MechanismSlot):
    if slot.phenomena:
        return set(slot.phenomena)
    try:
        return set(get_form(slot.form_id).phenomena)
    except Exception:
        return set()


def detect_conflicts(slots: list) -> dict:
    """Return the typed hazards among a set of nonlinear mechanism slots (Part 16 detection list)."""
    conflicts = []
    by_target = {}
    for s in slots:
        by_target.setdefault(s.target_path, []).append(s)
    for target, group in by_target.items():
        phs = [(_phenomena(s), s) for s in group]
        # duplicate saturation / fatigue / reinforcement on the SAME target = double counting
        for tag in ("saturation", "fatigue", "reinforcement", "diminishing_returns"):
            hit = [s.mechanism_id for ph, s in phs if tag in ph]
            if len(hit) >= 2:
                conflicts.append({"type": f"duplicate_{tag}", "target": target, "mechanisms": hit,
                                  "risk": "double counting — mediate or nest, do not add"})
        # multiple thresholds competing on one target
        thr = [s.mechanism_id for ph, s in phs if "threshold" in ph or "tipping" in ph]
        if len(thr) >= 2:
            conflicts.append({"type": "competing_thresholds", "target": target, "mechanisms": thr,
                              "risk": "order-dependent tipping — declare precedence"})
        # incompatible monotonicity on one target
        monos = {get_form(s.form_id).monotonicity for _, s in phs if _safe_mono(s)}
        if "increasing" in monos and "decreasing" in monos:
            conflicts.append({"type": "incompatible_monotonicity", "target": target,
                              "mechanisms": [s.mechanism_id for _, s in phs],
                              "risk": "opposing directions cancel — is this a genuine inverted-U? refit"})
    # self-excitation feedback (potential runaway)
    se = [s.mechanism_id for s in slots if "self_excitation" in _phenomena(s)]
    if se:
        conflicts.append({"type": "self_excitation_feedback", "mechanisms": se,
                          "risk": "unbounded event generation — cap rate + require branching α<1"})
    return {"n_slots": len(slots), "conflicts": conflicts, "clean": not conflicts}


def _safe_mono(slot):
    try:
        get_form(slot.form_id)
        return True
    except Exception:
        return False


def execution_order(slots: list) -> list:
    """Deterministic precedence order (Part 16): explicit precedence, then mediation heuristic (state-setting
    before rate-setting), then id for stability."""
    def key(s):
        ph = _phenomena(s)
        # thresholds/regime that GATE others run first; self-exciting runs last (it schedules events)
        gate = 0 if (ph & {"threshold", "regime", "tipping", "hysteresis"}) else 1
        late = 1 if ("self_excitation" in ph) else 0
        return (s.precedence, gate, late, s.mechanism_id)
    return sorted(slots, key=key)


@dataclass
class StabilityMonitor:
    """Rollout-time guard against runaway nonlinear feedback (Part 16 + Part 17)."""
    max_events_per_day: float = 5000.0
    max_total_followups: int = 2000
    _emitted: int = field(default=0)
    _history: list = field(default_factory=list)

    def admit_followups(self, n: int, *, window_days: float) -> tuple:
        """Return (allowed_n, note). Caps endogenous event generation to keep a self-exciting world finite."""
        self._emitted += n
        rate = n / max(1e-6, window_days)
        if self._emitted > self.max_total_followups:
            return 0, "followup_cap_reached(event_storm_guard)"
        if rate > self.max_events_per_day:
            allowed = int(self.max_events_per_day * window_days)
            return max(0, allowed), f"rate_capped({rate:.0f}/day > {self.max_events_per_day})"
        return n, ""

    def check_divergence(self, series: list, *, tol: float = 1e6) -> dict:
        """Detect explosive trajectories under repeated rollout (values blowing up / oscillating unboundedly)."""
        if not series:
            return {"stable": True}
        mx = max(abs(float(v)) for v in series)
        stable = mx < tol and all(v == v for v in series)     # finite + bounded
        return {"stable": stable, "max_abs": mx,
                "note": "" if stable else "divergent/non-finite trajectory — refuse, record instability"}
