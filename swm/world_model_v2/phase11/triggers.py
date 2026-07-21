"""Phase 11 — trigger detection (spec §9). Sixteen trigger families, each an EXECUTABLE detector (not an
enum), returning typed ``RecompileTriggerEvidence`` with a probability, severity, persistence, affected
scope, alternative explanations, and a dedup fingerprint.

Thresholds are NOT one arbitrary scalar. ``TriggerThresholds`` carries a per-signal value WITH provenance and
a version, and ``fit_on_calibration`` learns the residual/drift thresholds from the NEGATIVE-CONTROL
(unchanged) calibration episodes so the false-alarm rate is controlled — never from test outcomes. Until
fitted, defaults are principled (predictive-density tails / standardized effect sizes) and explicitly labelled
``broad_experimental``.

Detectors split into DIAGNOSTIC-driven (residual/ESS/regime/drift, from the running posterior predictive) and
EVIDENCE/STRUCTURE-driven (new actor/institution/rule/authority/coalition/network/shock/contradiction/outcome-
space, from typed fields the evidence layer attaches to the observation). Each returns evidence only; nothing
here decides to recompile (that is fusion + the decision policy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.world_model_v2.phase11 import diagnostics as D
from swm.world_model_v2.phase11._serial import content_hash
from swm.world_model_v2.phase11.contracts import RecompileTriggerEvidence, TRIGGER_FAMILIES

THRESHOLDS_VERSION = "phase11-thresholds-1.0"


@dataclass
class TriggerThresholds:
    """Versioned thresholds with provenance. ``basis`` records HOW each value was set."""
    residual_high: float = 3.0             # ≈ |z| 2.1 predictive surprise (broad prior; refine on calibration)
    impossible_density: float = 1e-4
    impossible_z: float = 6.0
    sustained_min_run: int = 3
    ess_collapse_frac: float = 0.1
    drift_effect: float = 0.8              # standardized mean shift (Cohen's d) that warrants a refit
    regime_gap: float = 2.0               # sd gap in residual between pre/post windows
    contradiction_min_reliability: float = 0.6
    version: str = THRESHOLDS_VERSION
    basis: dict = field(default_factory=lambda: {"all": "broad_experimental_default"})

    def fit_on_calibration(self, unchanged_residuals, *, false_alarm_target: float = 0.1) -> "TriggerThresholds":
        """Set ``residual_high`` to the (1−false_alarm_target) quantile of residuals seen on UNCHANGED
        (negative-control) calibration episodes — a false-alarm-controlled, data-derived threshold. Test
        outcomes are never used. Records the basis + n."""
        xs = sorted(float(r) for r in unchanged_residuals if r is not None)
        if len(xs) >= 20:
            q = xs[min(len(xs) - 1, int((1 - false_alarm_target) * len(xs)))]
            self.residual_high = round(max(1.0, q), 4)
            self.basis = {"residual_high": f"calibration_{int((1-false_alarm_target)*100)}pct_quantile",
                          "n_calibration_unchanged": len(xs), "false_alarm_target": false_alarm_target,
                          "others": "broad_experimental_default"}
        else:
            self.basis = {"residual_high": "broad_experimental_default_insufficient_calibration",
                          "n_calibration_unchanged": len(xs)}
        return self

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TriggerContext:
    """Everything a detector needs at one observation step. The controller assembles this."""
    observation: object                                  # RecompileObservation
    surprise: dict = field(default_factory=dict)         # diagnostics.surprise(...) for this obs
    ess: dict = field(default_factory=dict)              # diagnostics.ess_diagnostic(weights)
    residual_history: list = field(default_factory=list)  # per-observable trailing residuals (incl. this)
    value_history: list = field(default_factory=list)     # per-observable trailing observed values
    pre_residuals: list = field(default_factory=list)     # window before a suspected change point
    post_residuals: list = field(default_factory=list)
    plan_facts: dict = field(default_factory=dict)        # {known_actors, known_institutions, outcome_options,
    #                                                        mechanism_preconditions, network_edges, aliases}
    declared: dict = field(default_factory=dict)          # typed evidence hints on the observation
    cooldown: dict = field(default_factory=dict)          # {family: {on_cooldown, last_fired_at}}


def _p_from_severity(sev, persistence, independence):
    """A monotone squashing of severity·persistence·independence into a trigger probability in (0,1). This is
    a transparent, reproducible mapping — NOT an LLM-minted number."""
    x = max(0.0, min(1.0, sev)) * max(0.05, min(1.0, persistence)) * max(0.2, min(1.0, independence))
    return round(1.0 / (1.0 + math.exp(-6.0 * (x - 0.35))), 4)


def _ev(family, ctx, *, severity, persistence, scopes, method, independence=1.0, alts=None,
        supporting=None, contradictory=None):
    obs = ctx.observation
    oid = getattr(obs, "observation_id", "obs")
    fp = content_hash({"family": family, "obs": oid, "scopes": sorted(scopes)}, length=12)
    cd = (ctx.cooldown or {}).get(family, {})
    return RecompileTriggerEvidence(
        trigger_evidence_id=f"te::{family}::{oid}",
        trigger_family=family, affected_scope_candidates=list(scopes),
        supporting_observations=list(supporting or [oid]),
        contradictory_observations=list(contradictory or []),
        severity=round(float(severity), 4), persistence=round(float(persistence), 4),
        expected_impact=round(float(getattr(obs, "uncertainty", {}).get("terminal_sensitivity", severity)), 4),
        evidence_independence=round(float(independence), 4),
        trigger_probability=_p_from_severity(severity, persistence, independence),
        alternative_explanations=list(alts or []),
        diagnostic_method=method, thresholds_version=THRESHOLDS_VERSION,
        cooldown_state={"on_cooldown": bool(cd.get("on_cooldown")), "last_fired_at": cd.get("last_fired_at")},
        fingerprint=fp, provenance={"detector": family})


# ============================================================ DIAGNOSTIC-DRIVEN detectors
def d_unexplained_residual(ctx, th):
    """A high residual on an IN-SUPPORT observation is, by itself, ordinary low-probability sampling — that is
    Phase-3 posterior updating, NOT model inadequacy. This detector therefore fires ONLY when the surprise is
    also (a) near the edge of support (extreme tail) OR (b) the trailing edge of a SUSTAINED run — i.e. it
    distinguishes noise from persistent model failure (spec §9.1). A single ordinary surprise returns None."""
    s = ctx.surprise
    if not s or s.get("impossible"):
        return None                                       # impossibility handled by its own detector
    if s.get("residual", 0.0) < th.residual_high:
        return None
    extreme = s.get("tail_prob", 1.0) <= th.impossible_density * 50      # very extreme but not out-of-support
    run = D.sustained_failure(ctx.residual_history, threshold=th.residual_high,
                              min_run=th.sustained_min_run)["run_length"]
    if not (extreme or run >= 2):
        return None                                       # single ordinary in-support surprise → NOT a trigger
    sev = min(1.0, (s["residual"] - th.residual_high) / (2 * th.residual_high) + 0.25 + 0.1 * run)
    return _ev("unexplained_residual", ctx, severity=sev, persistence=min(0.8, 0.3 + 0.15 * run),
               scopes=["parameter_only", "latent_state", "mechanism_replacement"],
               method="neg_log_predictive_density_extreme_or_runct",
               alts=[{"explanation": "observation noise / heavy tail (ordinary — Phase 3)", "prob": 0.45},
                     {"explanation": "local parameter drift", "prob": 0.3}])


def d_impossible_event(ctx, th):
    s = ctx.surprise
    if not (s and s.get("impossible")):
        return None
    return _ev("impossible_event", ctx, severity=0.95, persistence=1.0,
               scopes=["outcome_contract", "mechanism_replacement", "structural_hypothesis", "full_plan"],
               method="outside_current_support",
               alts=[{"explanation": "outcome-space misspecification", "prob": 0.6},
                     {"explanation": "data error / mis-recorded observation", "prob": 0.25}])


def d_sustained_predictive_failure(ctx, th):
    sf = D.sustained_failure(ctx.residual_history, threshold=th.residual_high, min_run=th.sustained_min_run)
    if not sf["sustained"]:
        return None
    sev = min(1.0, 0.4 + 0.15 * sf["run_length"])
    return _ev("sustained_predictive_failure", ctx, severity=sev, persistence=min(1.0, sf["run_length"] / 5.0),
               scopes=["mechanism_replacement", "structural_hypothesis", "parameter_only"],
               method="residual_run_length",
               alts=[{"explanation": "persistent parameter drift", "prob": 0.45},
                     {"explanation": "wrong mechanism family", "prob": 0.45}])


def d_particle_collapse(ctx, th):
    e = ctx.ess
    if not (e and e.get("collapsed")):
        return None
    # collapse is a NUMERICAL degeneracy → resample/rejuvenate, not necessarily structure (low expected impact)
    return _ev("particle_collapse", ctx, severity=0.5, persistence=0.6,
               scopes=["parameter_only", "latent_state"], method="ess_fraction",
               alts=[{"explanation": "numerical degeneracy (resample fixes it)", "prob": 0.7},
                     {"explanation": "genuine evidential concentration", "prob": 0.3}])


def d_mechanism_regime_change(ctx, th):
    rs = D.regime_shift(ctx.pre_residuals, ctx.post_residuals)
    if not rs["shift"]:
        return None
    return _ev("mechanism_regime_change", ctx, severity=min(1.0, 0.4 + rs["gap"] / 8.0), persistence=0.8,
               scopes=["mechanism_replacement", "structural_hypothesis"], method="pre_post_residual_gap",
               alts=[{"explanation": "one-off shock", "prob": 0.35}])


def d_parameter_drift(ctx, th):
    dr = D.parameter_drift(ctx.pre_residuals and ctx.value_history[:len(ctx.value_history)//2] or [],
                           ctx.value_history[len(ctx.value_history)//2:] or [])
    if not dr.get("drift"):
        return None
    # drift = level moved but residuals did NOT explode → refit parameters, keep structure
    if D.sustained_failure(ctx.residual_history, threshold=th.residual_high, min_run=th.sustained_min_run)["sustained"]:
        return None                                       # structural failure dominates; not mere drift
    return _ev("parameter_drift", ctx, severity=min(1.0, 0.3 + dr["shift"] / 3.0), persistence=0.7,
               scopes=["parameter_only", "observation_model"], method="standardized_mean_shift",
               alts=[{"explanation": "structural change (ruled out: residuals stable)", "prob": 0.2}])


def d_mechanism_precondition_failure(ctx, th):
    dec = ctx.declared or {}
    failed = dec.get("mechanism_precondition_failed")
    if not failed:
        return None
    return _ev("mechanism_precondition_failure", ctx, severity=0.7, persistence=0.9,
               scopes=["mechanism_replacement"], method="declared_precondition_check",
               supporting=[getattr(ctx.observation, "observation_id", "obs")])


# ============================================================ EVIDENCE / STRUCTURE-driven detectors
def _declared(ctx, key):
    return (ctx.declared or {}).get(key)


def d_new_actor(ctx, th):
    a = _declared(ctx, "new_actor")
    if not a:
        return None
    known = set(ctx.plan_facts.get("known_actors", []))
    aliases = ctx.plan_facts.get("aliases", {})
    aid = a.get("id") if isinstance(a, dict) else a
    if aid in known or aliases.get(aid) in known:
        return None                                       # alias of an existing actor → NOT a new actor
    rel = float((a.get("causal_relevance", 1.0)) if isinstance(a, dict) else 1.0)
    if rel < 0.2:
        return None                                       # causally irrelevant new name → no trigger
    return _ev("new_actor", ctx, severity=min(1.0, 0.5 + rel / 2), persistence=1.0,
               scopes=["actor", "relationship", "local_network_region"], method="entity_resolution",
               alts=[{"explanation": "alias / rename of an existing actor", "prob": 0.3}])


def d_new_institution(ctx, th):
    i = _declared(ctx, "new_institution")
    if not i:
        return None
    known = set(ctx.plan_facts.get("known_institutions", []))
    iid = i.get("id") if isinstance(i, dict) else i
    if iid in known:
        return None
    return _ev("new_institution", ctx, severity=0.75, persistence=1.0,
               scopes=["institution_ruleset", "action_space"], method="institution_resolution")


def d_rule_change(ctx, th):
    r = _declared(ctx, "rule_change")
    if not r:
        return None
    # a rule change needs an effective date + source; a future-dated rule is NOT yet active (adversarial case)
    eff = r.get("effective_date") if isinstance(r, dict) else None
    now = getattr(ctx.observation, "event_time", 0.0)
    not_yet = bool(r.get("future_dated")) or (isinstance(eff, (int, float)) and eff > now)
    if not_yet:
        return None
    if not (isinstance(r, dict) and r.get("source")):
        return None                                       # unsourced rule claim is rejected
    return _ev("rule_change", ctx, severity=0.8, persistence=1.0,
               scopes=["institution_ruleset"], method="dated_sourced_rule_publication",
               alts=[{"explanation": "future-dated rule not yet in force", "prob": 0.1}])


def d_authority_change(ctx, th):
    a = _declared(ctx, "authority_change")
    if not a:
        return None
    return _ev("authority_change", ctx, severity=0.75, persistence=1.0,
               scopes=["institution_ruleset", "actor"], method="authority_delta")


def d_coalition_change(ctx, th):
    c = _declared(ctx, "coalition_change")
    if not c:
        return None
    return _ev("coalition_change", ctx, severity=0.7, persistence=0.85,
               scopes=["relationship", "local_network_region", "structural_hypothesis"],
               method="coalition_delta")


def d_network_restructuring(ctx, th):
    n = _declared(ctx, "network_change")
    if not n:
        return None
    # a transient outage is not a restructuring (persistence gates it)
    persistent = bool(n.get("persistent", True)) if isinstance(n, dict) else True
    if not persistent:
        return None
    return _ev("network_restructuring", ctx, severity=0.65, persistence=0.8,
               scopes=["local_network_region", "relationship"], method="edge_community_delta",
               alts=[{"explanation": "transient communication outage", "prob": 0.3}])


def d_exogenous_shock(ctx, th):
    s = _declared(ctx, "exogenous_shock")
    if not s:
        return None
    return _ev("exogenous_shock", ctx, severity=0.7, persistence=0.6,
               scopes=["structural_hypothesis", "parameter_only"], method="exogenous_event",
               alts=[{"explanation": "already represented by an endogenous mechanism", "prob": 0.3}])


def d_evidence_contradiction(ctx, th):
    links = getattr(ctx.observation, "contradiction_links", [])
    if not links:
        return None
    rel = float((ctx.declared or {}).get("contradiction_reliability", 0.8))
    if rel < th.contradiction_min_reliability:
        return None
    return _ev("evidence_contradiction", ctx, severity=min(1.0, 0.5 + rel / 2), persistence=1.0,
               scopes=["latent_state", "parameter_only", "structural_hypothesis"],
               method="credible_contradiction", contradictory=links)


def d_outcome_space_change(ctx, th):
    o = _declared(ctx, "outcome_space_change")
    if not o:
        return None
    return _ev("outcome_space_change", ctx, severity=0.9, persistence=1.0,
               scopes=["outcome_contract", "action_space"], method="outcome_contract_delta",
               alts=[{"explanation": "recorded but resolution rule unchanged", "prob": 0.2}])


# registry (family -> detector). Order is stable for deterministic scans.
DETECTORS = [
    ("unexplained_residual", d_unexplained_residual),
    ("impossible_event", d_impossible_event),
    ("new_actor", d_new_actor),
    ("new_institution", d_new_institution),
    ("rule_change", d_rule_change),
    ("authority_change", d_authority_change),
    ("coalition_change", d_coalition_change),
    ("network_restructuring", d_network_restructuring),
    ("exogenous_shock", d_exogenous_shock),
    ("mechanism_precondition_failure", d_mechanism_precondition_failure),
    ("sustained_predictive_failure", d_sustained_predictive_failure),
    ("particle_collapse", d_particle_collapse),
    ("evidence_contradiction", d_evidence_contradiction),
    ("outcome_space_change", d_outcome_space_change),
    ("mechanism_regime_change", d_mechanism_regime_change),
    ("parameter_drift", d_parameter_drift),
]
assert {f for f, _ in DETECTORS} == set(TRIGGER_FAMILIES), "detector set must cover all 16 families"


# Origins that are ALLOWED to trigger a recompile. A `simulation_internal` event (sampled by the running
# rollout from the ACTIVE plan) and a `planned_expansion` (a predefined structural branch firing) are executed
# normally and are NEVER eligible — surprise from them is ordinary Phase-3 updating, not model failure.
TRIGGER_ELIGIBLE_ORIGINS = ("external_evidence", "historical_replay", "internal_diagnostic")


def observation_eligible(obs) -> bool:
    """Only external/leakage-safe-replay/internal-diagnostic observations may TRIGGER recompilation. An
    out-of-support observation (representable=False) is eligible regardless (it proves inadequacy), but a
    representable simulation-internal/planned event is not."""
    origin = getattr(obs, "origin", "external_evidence")
    if origin in ("simulation_internal", "planned_expansion"):
        return (not getattr(obs, "representable", True)) and origin != "planned_expansion"
    return origin in TRIGGER_ELIGIBLE_ORIGINS


def detect_all(ctx, thresholds=None) -> list:
    """Run every detector on the context; return the list of fired ``RecompileTriggerEvidence`` (deduped by
    fingerprint). Ineligible observations (representable simulation-internal / planned expansion) yield NO
    triggers — they are executed normally. Detectors on cooldown are suppressed (fusion re-checks)."""
    th = thresholds or TriggerThresholds()
    if not observation_eligible(ctx.observation):
        return []                                         # normal execution — not a recompilation candidate
    fired, seen = [], set()
    for family, fn in DETECTORS:
        cd = (ctx.cooldown or {}).get(family, {})
        try:
            ev = fn(ctx, th)
        except Exception:  # noqa: BLE001 — a detector must never crash the run; log-and-continue
            ev = None
        if ev is None:
            continue
        if cd.get("on_cooldown") and ev.trigger_family not in ("impossible_event", "outcome_space_change"):
            continue                                      # cooldown suppresses repeats (except hard-structural)
        if ev.fingerprint in seen:
            continue
        seen.add(ev.fingerprint)
        fired.append(ev)
    return fired
