"""Transport-risk engine — Phase 6, Part 5.

A parameter pack is fitted/estimated in ONE empirical context (a domain, population, geography, period,
platform, outcome definition, …). Using it in a different scenario is TRANSPORT, and transport is risky.
This module scores that risk per axis and returns a decision + an uncertainty-inflation factor:

    profile = assess_transport(pack_context, scenario)
    profile.decision  ∈ {transport_direct, transport_widened, experimental, reject}
    profile.widening  — multiply parameter sd / broaden the outcome dispersion by this
    profile.reasons   — per-axis risk with the decisive failures called out

The engine is deliberately PESSIMISTIC: an unstated axis defaults to medium risk, a decisive mismatch
(incompatible outcome definition, missing required variable, intervention-type change) can force `reject`
so the compiler falls through to a weaker-but-honest mechanism rather than a confidently-wrong strong one.
No number here is minted for the forecast — this only widens uncertainty; the point estimate is the pack's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: axes and their weight in the overall transport risk (sum need not be 1; normalized internally)
AXES = {
    "population": 1.5, "domain": 1.5, "platform": 1.0, "geography": 0.7, "period": 0.8,
    "outcome_definition": 1.5, "action_space": 1.0, "exposure_process": 1.0,
    "network_structure": 0.8, "time_scale": 0.8, "measurement": 0.7, "intervention_type": 1.2,
}
#: axes that can VETO transport (a mismatch here forces reject/experimental regardless of the average)
DECISIVE = ("outcome_definition", "intervention_type")

WIDEN_MAX = 3.0


@dataclass
class TransportProfile:
    overall_risk: float                       # 0 (identical context) .. 1 (fully out of context)
    per_axis: dict                            # {axis: {"risk": r, "why": str}}
    decision: str                             # transport_direct | transport_widened | experimental | reject
    widening: float                           # uncertainty inflation factor (>=1)
    decisive_failures: list = field(default_factory=list)
    note: str = ""

    def as_dict(self):
        return {"overall_risk": round(self.overall_risk, 3), "decision": self.decision,
                "widening": round(self.widening, 3), "decisive_failures": self.decisive_failures,
                "per_axis": {k: {"risk": round(v["risk"], 3), "why": v["why"]}
                             for k, v in self.per_axis.items()}, "note": self.note}


def _axis_risk(pack_val, scen_val) -> tuple[float, str]:
    """Risk on one axis from two free-text context descriptors. Missing either side → medium (0.5,
    pessimistic). Exact match → 0. Token overlap → partial. No overlap → high."""
    if not pack_val and not scen_val:
        return 0.5, "both unspecified → default medium risk"
    if not pack_val or not scen_val:
        return 0.5, "one side unspecified → medium risk"
    a = str(pack_val).lower()
    b = str(scen_val).lower()
    if a == b:
        return 0.0, "exact match"
    ta, tb = set(a.replace("_", " ").split()), set(b.replace("_", " ").split())
    if a in b or b in a:
        return 0.15, "substring match"
    jac = len(ta & tb) / max(1, len(ta | tb))
    if jac >= 0.5:
        return 0.3, f"strong token overlap ({jac:.2f})"
    if jac > 0.0:
        return 0.6, f"weak token overlap ({jac:.2f})"
    return 0.9, "no overlap → out of context"


def assess_transport(pack_context: dict, scenario: dict, *,
                     missing_required_vars: list | None = None) -> TransportProfile:
    """pack_context: the pack's empirical context keyed by AXES (domain/population/platform/…). scenario:
    the compiler's scenario descriptor (same keys where known). missing_required_vars: variables the
    family needs that the scenario cannot supply (a decisive data gap)."""
    per_axis, wsum, rsum, decisive = {}, 0.0, 0.0, []
    for axis, weight in AXES.items():
        r, why = _axis_risk(pack_context.get(axis), scenario.get(axis))
        per_axis[axis] = {"risk": r, "why": why}
        rsum += weight * r
        wsum += weight
        if axis in DECISIVE and r >= 0.85:
            decisive.append(f"{axis}: {why}")
    overall = rsum / wsum if wsum else 0.5

    missing = list(missing_required_vars or [])
    if missing:
        decisive.append(f"required variables unavailable in scenario: {missing}")

    # decision
    if decisive:
        # a decisive mismatch: transport is not defensible at full strength
        decision = "reject" if (len(decisive) >= 2 or missing) else "experimental"
        widening = WIDEN_MAX
    elif overall <= 0.12:
        decision, widening = "transport_direct", 1.0
    elif overall <= 0.5:
        decision = "transport_widened"
        widening = 1.0 + 2.0 * overall           # 0.12→1.24 … 0.5→2.0
    else:
        decision = "experimental"
        widening = min(WIDEN_MAX, 1.0 + 2.0 * overall)
    note = ("direct — context matches" if decision == "transport_direct" else
            "widen parameter uncertainty by the factor; point estimate unchanged"
            if decision == "transport_widened" else
            "treat as experimental — broad uncertainty; do not present as calibrated"
            if decision == "experimental" else
            "reject transport — a decisive axis is incompatible; fall through to a weaker honest mechanism")
    return TransportProfile(overall_risk=overall, per_axis=per_axis, decision=decision,
                            widening=round(widening, 3), decisive_failures=decisive, note=note)


def pack_context_from_record(pack) -> dict:
    """Extract a transport context dict from a ParameterPack (best-effort; unstated axes stay empty so the
    engine treats them pessimistically)."""
    ctx = {"domain": getattr(pack, "domain", ""), "population": getattr(pack, "population", ""),
           "time_scale": getattr(pack, "time_scale", "")}
    # packs may carry a richer context in a `context` dict on values or transport_note
    extra = getattr(pack, "context", None)
    if isinstance(extra, dict):
        ctx.update({k: v for k, v in extra.items() if k in AXES})
    return ctx
