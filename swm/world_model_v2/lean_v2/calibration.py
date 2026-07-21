"""Actor-action calibration + the prior↔simulation reliability combiner.

Two separate calibration concerns, both empirical, neither rewriting a real actor decision:

  * `ActorActionReliabilityModel` — how much to trust a simulated actor's action, estimated on
    RESOLVED historical decisions (held out from the five evaluation questions). It may widen
    behavioral uncertainty (grounded action-policy variants) or scale the reliability of a
    simulation result; it may NOT overwrite the action with a hardcoded rule and it never
    invents a "100% real-world probability" for a single sampled action.

  * `ForecastReliabilityCombiner` — combines the grounded prior and the simulation-conditional
    forecast through EMPIRICALLY CALIBRATED reliability, never a fixed 70/30 blend. Fit only
    from resolved cases outside {Banxico, BoJ, visionOS, Wale, Hormuz}, with strict temporal +
    outcome-leakage protection. When too few independent cases exist to fit a defensible
    combiner, it does NOT invent one: it exposes prior and simulation separately, gives the
    feasible combined range, and refuses to let Lean V2 become the default.

Both always expose the grounded prior, the simulation-conditional forecast, and the combined
forecast separately — disagreement is never hidden."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

CALIBRATION_VERSION = "lean_v2.calibration.v1"

#: the five evaluation questions — their outcomes may NEVER enter combiner training
EVAL_QIDS = ("cfb43147-d9d2-5bd9-903f-f449e9a5aecf",   # Banxico
             "7279494c-a775-5a57-a5f2-ac22252fb286",   # BoJ
             "5c0765ed-cbd1-5af5-bce0-adbfebd4e0f6",   # visionOS
             "741b4bed-7502-5cd2-9cbe-949fbc70f857",   # Wale
             "017e64ef-7354-56c4-8a4d-e27121bc639a")   # Hormuz
EVAL_NAMES = ("banxico", "boj", "visionos", "wale", "hormuz")

_COMBINER_PATH = Path("swm/world_model_v2/lean_v2/data/combiner_weights.json")


# ------------------------------------------------------------------ action calibration
@dataclass
class ActorActionReliabilityModel:
    """Reliability strata for simulated actions, keyed by (institution_type, role, decision,
    state, evidence_completeness). Values are calibrated accuracies from resolved cases; absent
    strata fall back to the global reliability with the fallback recorded. Loaded from a
    committed dataset when present; otherwise an explicit `unavailable` model that widens
    behavioral uncertainty rather than pretending precision."""
    strata: dict = field(default_factory=dict)
    global_reliability: float = None
    n_cases: int = 0
    available: bool = False
    biases: dict = field(default_factory=dict)   # consensus_bias / dissent_bias / deferral ...

    def reliability_for(self, *, institution_type: str = "", role: str = "",
                        decision_type: str = "", state_type: str = "",
                        evidence_completeness: str = "") -> tuple:
        """(reliability, provenance). Reliability in [0,1]; None when no calibration exists —
        the caller then represents behavioral uncertainty by grounded variants, not a number."""
        if not self.available:
            return None, {"source": "no_calibration_dataset",
                          "treatment": "behavioral uncertainty via grounded action-policy "
                                       "variants; no invented action probability"}
        for key in (f"{institution_type}|{role}|{decision_type}|{state_type}",
                    f"{institution_type}|{role}|{decision_type}",
                    f"{institution_type}|{role}", institution_type):
            if key in self.strata:
                return self.strata[key]["reliability"], {"source": "calibrated_stratum",
                                                         "key": key,
                                                         "n": self.strata[key].get("n")}
        return self.global_reliability, {"source": "global_fallback", "n": self.n_cases}

    def as_dict(self) -> dict:
        return asdict(self)


def load_action_reliability() -> ActorActionReliabilityModel:
    p = Path("swm/world_model_v2/lean_v2/data/action_reliability.json")
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return ActorActionReliabilityModel(
                strata=d.get("strata") or {}, global_reliability=d.get("global_reliability"),
                n_cases=d.get("n_cases") or 0, available=True, biases=d.get("biases") or {})
        except Exception:  # noqa: BLE001
            pass
    return ActorActionReliabilityModel(available=False)


# ------------------------------------------------------------------ forecast combination
@dataclass
class GroundedPriorForecast:
    p: float = None
    source: str = ""                             # outcome_reference_class | ...
    n: int = 0
    interval: tuple = None
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationConditionalForecast:
    p: float = None                              # P(yes | resolved mass)
    resolved_mass: float = 0.0
    interval: tuple = None
    weight_sensitive: bool = False
    dependence_sensitive: bool = False
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastReliabilityFeatures:
    prior_n: int = 0
    prior_specificity: str = ""
    resolved_mass: float = 0.0
    unknown_state_mass: float = 0.0
    evidence_coverage: float = 0.0
    structural_sensitivity: bool = False
    weight_sensitive: bool = False
    dependence_sensitive: bool = False
    prior_sim_divergence: float = 0.0
    horizon_days: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastCombinationReport:
    grounded_prior: dict
    simulation_conditional: dict
    combined: float
    combined_interval: tuple
    method: str
    reliability_features: dict
    sim_weight: float = None
    prior_weight: float = None
    disagreement: float = 0.0
    fixed_blend_used: bool = False               # MUST stay False
    combiner_available: bool = False
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class ForecastReliabilityCombiner:
    """Empirically-calibrated combiner. If a committed, leakage-audited weight file exists it
    is used (a transparent logistic reliability model over the features); otherwise the combiner
    is UNAVAILABLE and the report exposes prior + simulation separately with the feasible range
    — never a fixed blend, never a hidden 0.5."""

    def __init__(self):
        self.weights = None
        self.meta = {}
        if _COMBINER_PATH.exists():
            try:
                d = json.loads(_COMBINER_PATH.read_text())
                # refuse a combiner whose training touched any eval outcome
                trained = set(d.get("training_qids") or [])
                if trained & set(EVAL_QIDS):
                    self.meta = {"rejected": "combiner training included an eval qid — refused"}
                else:
                    self.weights = d.get("weights")
                    self.meta = {k: v for k, v in d.items() if k != "weights"}
            except Exception:  # noqa: BLE001
                self.meta = {"error": "combiner weights unreadable"}

    @property
    def available(self) -> bool:
        return bool(self.weights)

    def combine(self, prior: GroundedPriorForecast, sim: SimulationConditionalForecast,
                feats: ForecastReliabilityFeatures) -> ForecastCombinationReport:
        pri_p, sim_p = prior.p, sim.p
        disagreement = (abs((pri_p if pri_p is not None else 0.5)
                            - (sim_p if sim_p is not None else 0.5))
                        if pri_p is not None and sim_p is not None else 0.0)
        rep = ForecastCombinationReport(
            grounded_prior=prior.as_dict(), simulation_conditional=sim.as_dict(),
            combined=None, combined_interval=None, method="", reliability_features=feats.as_dict(),
            disagreement=round(disagreement, 4), combiner_available=self.available)

        if sim_p is None and pri_p is None:
            rep.method = "no_forecast"
            return rep
        if sim_p is None:
            rep.combined, rep.method = pri_p, "prior_only_simulation_unavailable"
            rep.combined_interval = prior.interval
            rep.notes.append("simulation produced no resolved forecast; prior served, labeled")
            return rep
        if pri_p is None:
            rep.combined, rep.method = sim_p, "simulation_only_no_grounded_prior"
            rep.combined_interval = sim.interval
            rep.notes.append("no grounded prior available; simulation served, labeled")
            return rep

        if self.available:
            w_sim = self._reliability_weight(feats)
            rep.sim_weight = round(w_sim, 4)
            rep.prior_weight = round(1 - w_sim, 4)
            rep.combined = round(w_sim * sim_p + (1 - w_sim) * pri_p, 4)
            rep.method = "calibrated_logistic_reliability"
            lo = min(pri_p, sim_p, (sim.interval or [sim_p])[0] if sim.interval else sim_p)
            hi = max(pri_p, sim_p, (sim.interval or [sim_p, sim_p])[1]
                     if sim.interval else sim_p)
            rep.combined_interval = (round(lo, 4), round(hi, 4))
            rep.notes.append(f"combined via committed leakage-audited combiner "
                             f"(sim reliability weight {w_sim:.3f}); prior and simulation "
                             f"remain visible above")
            return rep

        # NO combiner: expose both, give the feasible range, do NOT invent a blend
        rep.method = "combiner_unavailable_range_only"
        rep.combined = None
        rep.combined_interval = (round(min(pri_p, sim_p), 4), round(max(pri_p, sim_p), 4))
        rep.notes.append("no leakage-audited reliability combiner is fitted — prior and "
                         "simulation are reported separately with the feasible combined range; "
                         "no fixed blend is applied and Lean V2 must not become default on "
                         "this basis")
        return rep

    def _reliability_weight(self, f: ForecastReliabilityFeatures) -> float:
        """Transparent logistic over the reliability features → simulation weight in [0.05,
        0.95]. Coefficients come from the committed file; this function only evaluates them."""
        x = {"bias": 1.0, "resolved_mass": f.resolved_mass,
             "prior_n": math.log1p(f.prior_n), "unknown_state_mass": f.unknown_state_mass,
             "weight_sensitive": 1.0 if f.weight_sensitive else 0.0,
             "dependence_sensitive": 1.0 if f.dependence_sensitive else 0.0,
             "structural_sensitivity": 1.0 if f.structural_sensitivity else 0.0,
             "prior_sim_divergence": f.prior_sim_divergence,
             "evidence_coverage": f.evidence_coverage}
        z = sum(self.weights.get(k, 0.0) * v for k, v in x.items())
        return max(0.05, min(0.95, 1.0 / (1.0 + math.exp(-z))))

    def manifest(self) -> dict:
        return {"version": CALIBRATION_VERSION, "available": self.available,
                "meta": self.meta,
                "eval_qids_excluded": list(EVAL_QIDS)}
