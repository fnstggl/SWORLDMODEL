"""Semantic-feature validation registry — Phase 5.

The audit's RC5: LLM semantic features enter production unvalidated, and the portfolio showed they can
HURT (Upworthy interpretation dims −0.060 [−0.12,−0.007]; Enron interpretation channel ns; BehaviorBench
no_interp ≈ full). This registry makes the semantic channel EVIDENCE-GATED: a feature reaches production
only after passing reliability + incremental-held-out-value + transport checks; otherwise it is quarantined
or rejected, and production execution selects features by registry status + domain.

The lifecycle mirrors the mechanism registry: proposed → operationally_defined → reliability_validated →
locally_predictive → incrementally_predictive → transport_validated → production_eligible, with
domain_restricted / quarantined / rejected. Failures are preserved. Seeded with the REAL prior results so
the harmful/null features are quarantined from day one.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict

SEMANTIC_STATUSES = ("proposed", "operationally_defined", "reliability_validated", "locally_predictive",
                     "incrementally_predictive", "transport_validated", "production_eligible",
                     "domain_restricted", "quarantined", "rejected")


@dataclass
class SemanticFeature:
    feature_id: str
    construct: str                            # exact construct definition
    observable_input: str
    support: str                              # e.g. "[0,1]"
    causal_role: str                          # which mechanism consumes it
    downstream_mechanisms: list = field(default_factory=list)
    expected_direction: str = ""
    known_confounds: list = field(default_factory=list)      # e.g. ["message length", "sender identity"]
    domain_limits: list = field(default_factory=list)
    status: str = "proposed"
    status_reason: str = ""
    # measured evidence
    inter_prompt_agreement: float | None = None
    inter_model_agreement: float | None = None
    incremental_value: dict = field(default_factory=dict)    # domain -> {delta, ci95, controls}
    transport: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    version: str = "1.0"

    def __post_init__(self):
        if self.status not in SEMANTIC_STATUSES:
            raise ValueError(f"{self.feature_id}: bad status {self.status!r}")

    def promotion_blockers(self, target: str) -> list:
        order = {s: i for i, s in enumerate(
            ("proposed", "operationally_defined", "reliability_validated", "locally_predictive",
             "incrementally_predictive", "transport_validated", "production_eligible"))}
        if target in ("quarantined", "rejected", "domain_restricted"):
            return []
        b = []
        if order.get(target, 0) >= order["operationally_defined"] and not self.construct.strip():
            b.append("no construct definition")
        if order.get(target, 0) >= order["reliability_validated"]:
            if (self.inter_prompt_agreement or 0) < 0.6:
                b.append(f"inter-prompt agreement {self.inter_prompt_agreement} < 0.6")
        if order.get(target, 0) >= order["incrementally_predictive"]:
            # needs a domain where the incremental held-out delta is beneficial with CI excluding 0
            good = [d for d, v in self.incremental_value.items()
                    if v.get("delta") is not None and v.get("beneficial")]
            if not good:
                b.append("no domain with SIGNIFICANT beneficial incremental held-out value")
        if order.get(target, 0) >= order["production_eligible"]:
            if not self.transport.get("passed"):
                b.append("transport not validated")
        return b

    def as_dict(self):
        return asdict(self)


class SemanticRegistry:
    def __init__(self):
        self.features: dict[str, SemanticFeature] = {}

    def register(self, f: SemanticFeature):
        self.features[f.feature_id] = f
        return f

    def set_status(self, fid: str, status: str, *, reason: str):
        f = self.features[fid]
        b = f.promotion_blockers(status)
        if b:
            raise ValueError(f"{fid} → {status} blocked: {'; '.join(b)}")
        f.status, f.status_reason = status, reason
        return f

    def record_incremental(self, fid: str, domain: str, *, delta: float, ci95: list, controls: list,
                           beneficial: bool):
        """delta = held-out metric change from ADDING the feature after controlling for nonsemantic
        features. beneficial=True iff it improves AND the CI excludes 0. Negative/null → a failure record."""
        f = self.features[fid]
        f.incremental_value[domain] = {"delta": delta, "ci95": ci95, "controls": controls,
                                       "beneficial": beneficial}
        if not beneficial:
            f.failures.append({"kind": "incremental_value", "domain": domain, "delta": delta, "ci95": ci95})

    def production_features(self, domain: str) -> list:
        out = []
        for f in self.features.values():
            if f.status == "production_eligible":
                out.append(f.feature_id)
            elif f.status == "domain_restricted" and domain in [d for d in f.incremental_value
                                                                if f.incremental_value[d].get("beneficial")]:
                out.append(f.feature_id)
        return out

    def summary(self):
        by = {}
        for f in self.features.values():
            by[f.status] = by.get(f.status, 0) + 1
        return {"n": len(self.features), "by_status": by,
                "quarantined": [f.feature_id for f in self.features.values() if f.status == "quarantined"],
                "production": [f.feature_id for f in self.features.values()
                               if f.status == "production_eligible"]}


# ------------------------------------------------------------------ reliability measurement (real)
def inter_run_agreement(runs: list) -> float:
    """runs: list of feature-vectors (same items, repeated LLM calls / prompts / models). Returns mean
    per-dimension intraclass-correlation-like agreement = 1 − mean(within-item variance)/total variance."""
    if len(runs) < 2:
        return None
    n_items = len(runs[0])
    vec = bool(n_items) and isinstance(runs[0][0], (list, tuple))
    n_dims = len(runs[0][0]) if vec else 1

    def cell(r, it, d):
        return runs[r][it][d] if vec else runs[r][it]

    agrees = []
    for d in range(n_dims):
        within, allv = [], []
        for it in range(n_items):
            vals = [cell(r, it, d) for r in range(len(runs))]
            within.append(statistics.pvariance(vals) if len(vals) > 1 else 0.0)
            allv += vals
        total = statistics.pvariance(allv) if len(allv) > 1 else 1e-9
        agrees.append(max(0.0, 1.0 - (sum(within) / len(within)) / max(1e-9, total)))
    return round(sum(agrees) / len(agrees), 4)


# ------------------------------------------------------------------ seed with the REAL prior evidence
def seed_registry() -> SemanticRegistry:
    """Every feature from actor_cognition.FEATURE_DIMS, seeded with the portfolio's measured evidence.
    The interpretation channel's held-out record is null-or-harmful, so those features are QUARANTINED —
    they may run in experimental arms but are excluded from production selection (evidence-based gating)."""
    from swm.world_model_v2.actor_cognition import FEATURE_DIMS

    reg = SemanticRegistry()
    confounds = {"urgency": ["message length", "thread position"],
                 "obligation": ["sender seniority", "explicit ask"],
                 "relevance_to_goals": ["recipient role", "keyword overlap"],
                 "relationship_salience": ["sender identity", "prior interaction count"]}
    for dim in FEATURE_DIMS:
        reg.register(SemanticFeature(
            feature_id=f"interp.{dim}", construct=f"actor's perceived {dim} of an incoming item",
            observable_input="the item text + actor-observable context", support="[0,1]",
            causal_role="modulates action utility / timing hazard within bounds",
            downstream_mechanisms=["information_interpretation", "typed_action_policy"],
            expected_direction="higher → more/faster engagement",
            known_confounds=confounds.get(dim, ["surface features (length, sender)"]),
            status="operationally_defined",
            status_reason="defined in FEATURE_DIMS with a clamped [0,1] extraction schema"))
    # REAL measured incremental value (preserved from the portfolio artifacts):
    for dim in FEATURE_DIMS:
        reg.record_incremental(f"interp.{dim}", "upworthy_headline", delta=+0.060,
                               ci95=[0.007, 0.12], controls=["surface features (length, punctuation, "
                               "clickbait markers)"], beneficial=False)   # positive delta = WORSE (W1) → harmful
        reg.record_incremental(f"interp.{dim}", "enron_messaging", delta=+0.006,
                               ci95=[-0.006, 0.019], controls=["fitted metadata anchor"], beneficial=False)
        reg.record_incremental(f"interp.{dim}", "behaviorbench_games", delta=0.0,
                               ci95=[-0.005, 0.005], controls=["structural preference model"],
                               beneficial=False)
    # quarantine the whole interpretation channel from production selection (harmful/null everywhere tested)
    for dim in FEATURE_DIMS:
        reg.set_status(f"interp.{dim}", "quarantined",
                       reason="held-out incremental value null-or-HARMFUL on every domain tested "
                              "(Upworthy +0.060 W1 harmful [CI excl 0]; Enron ns; BehaviorBench ~0). "
                              "May run in experimental arms; EXCLUDED from production selection.")
    return reg
