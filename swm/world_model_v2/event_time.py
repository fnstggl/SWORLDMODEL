"""EVENT-TIME contract architecture — every resolvable question as a first-passage problem (universal).

READOUT, NOT RESOLVER. The answer mechanism distinguishes two things:
  1. a READOUT — translating the simulated world into the format the question asks for;
  2. a RESOLVER — an additional decision or random draw that DECLARES the answer.
You need the readout; you should not have the resolver. Mechanisms simulate events through time;
when the target condition first becomes true, absorption is OBSERVED; every related forecast (the
deadline probability, the timing distribution, the mode of resolution) is derived from those same
simulated trajectories. Nothing at the terminal draws an outcome.

The components, on the existing event/StateDelta plane:
 1. `EventTimeContract` — terminal readout = per-particle FIRST-PASSAGE times into the absorbing state;
    projects a censoring-aware survival curve, CDF, quantiles, P(censored), and the JOINT mode×time
    distribution. With `binary_options` it ALSO projects the question's own two options:
    P(yes) = F(deadline) (occurrence questions) or 1 − F(deadline) (remains/still-in-state questions) —
    monotone and mutually coherent across cutoffs by construction.
 2. `AbsorptionMonitorOperator` — runs on EVERY event; when the absorbing predicate first holds, stamps
    `absorbed_at`/`absorbed_by` and freezes them (first passage only). Mechanisms never "resolve the
    outcome"; they change the world and absorption is OBSERVED.
 3. `HazardRoundOperator` — every scheduled round (cadence layer) carries a per-round success hazard.
    Three parameterizations: (a) fitted family hazard curve h(t) × the mode's per-branch SAMPLED stance
    hazard ratio ("when" questions); (b) CALIBRATED — the per-particle target absorbed-mass is drawn
    from the evidence-updated POSTERIOR rate particles and spread over the rounds as exponents shaped
    by the family curve; (c) `success_prob` — an outcome-entailing DATED FACT absorbs at its real date
    with probability = fact confidence. All three consume upstream causal state RELATIVELY
    (multiplicative, no-effect-centered) — the pathway-process quantities written by the simulated
    actors' own ACTIONS move these hazards: the endogenous half of the clock.
 4. Fitted time-to-event families — `fit_survival_pack` (calibration split ONLY) fits discrete hazards
    over lifetime-fraction buckets from archived price paths. `family_hazard_curve` serves h(t).
 5. Unification — `convert_to_event_time` rewires "when" questions; `convert_binary_to_event_time`
    rewires binary deadline questions through the SAME machinery: the terminal resolver events
    (resolve_outcome / aggregate_outcome_resolution) are removed, outcome-entailing facts and
    institutional decisions become absorbing writers at their real dates, and the answer is read out
    of the trajectories. Universal: routing is linguistic + structural, never scenario-specific.

Causal layering (see `mode_graph`): the question's modes form a typed graph — each mode is reached
through a PATHWAY (actor-driven: cooperative/unilateral/institutional/operational/competitive; or
WORLD-DRIVEN: threshold_crossing, diffusion_adoption, market_aggregation, physical_process,
stochastic_external, resource_depletion, cascade_failure, scheduled_transition) and carries a
DECISION STRUCTURE from which the stance-combination rule is DERIVED — "most-opposed binds" is the
unanimity case, not a universal law. Stances are MODE-SCOPED (`stance(actor, mode)`), condition the
Phase-4 ACTION policies (behavior), and actions write pathway-process state the hazards consume — so
the direct stance→hazard multiplier is only the residual channel, log-split by
ENDOGENOUS_STANCE_SPLIT when the behavioral channel is live.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)
from swm.world_model_v2.mode_graph import (ENDOGENOUS_STANCE_SPLIT, LEGACY_LEVELS, canon_level,
                                           canonical_modes, combine_stances, mode_pathway,
                                           pathway_of, progress_var)

SURV_PACK = Path("experiments/replay_vault_v3/family_survival_pack.json")
INTENTION_HR_PACK = Path("experiments/replay_vault_v3/intention_hr_pack.json")
_BUCKETS = 5                                                  # lifetime-fraction hazard buckets
_Z80 = 1.2816                                                 # 80% two-sided normal quantile

# DOCUMENTED PRIORS — NOT FITTED. Hazard ratio a stated stance implies for the hazards of modes on
# the pathway the stance concerns, as (median, lo80, hi80). The taxonomy is UNIVERSAL — stances are
# classified against the resolution criterion's resolving state (a deal, a resignation, a rate cut,
# a bill, a launch), never against "agreement" specifically. Deliberately conservative (centered
# nearer 1.0 than any crushed point value); each PARTICLE samples its own ratio from the lognormal
# these bounds define, so the uncertainty over the effect size survives into the terminal CDF
# instead of being collapsed to a point coefficient. Replaced wholesale when intention_hr_pack.json
# exists (fit from resolved cases via fit_intention_hazard_ratios — see
# experiments/replay_v3/fit_intention_hr.py for the statement→hazard-change corpus builder).
INTENTION_HR_PRIORS = {
    "committed_to_prevent": (0.55, 0.30, 0.90),
    "conditionally_opposed": (0.78, 0.50, 1.10),
    "weakly_opposed": (0.90, 0.65, 1.15),
    "neutral": (1.00, 0.80, 1.25),
    "inclined_toward": (1.35, 1.00, 1.90),
    "actively_pursuing": (1.70, 1.15, 2.50),
    "formally_committed": (2.10, 1.30, 3.20),
}
# legacy agreement-specific labels map onto the universal taxonomy (old packs/transcripts)
_LEGACY_LEVELS = LEGACY_LEVELS

# SENSITIVITY-HARNESS OVERRIDES (experiments only, never production defaults): force the agreement /
# non-agreement mode hazard ratio to a point value so assumption-dependence is measurable.
AGREEMENT_HR_OVERRIDE = None
VICTORY_HR_OVERRIDE = None


def _hr_table() -> dict:
    if INTENTION_HR_PACK.exists():
        try:
            pack = json.loads(INTENTION_HR_PACK.read_text())
            return {k: tuple(v) for k, v in (pack.get("hazard_ratios") or {}).items()} or dict(INTENTION_HR_PRIORS)
        except Exception:  # noqa: BLE001
            return dict(INTENTION_HR_PRIORS)
    return dict(INTENTION_HR_PRIORS)


def mode_hazard_ratio(stances: list, pathway: str = "cooperative_agreement", *,
                      mode: dict = None) -> dict:
    """One hazard-ratio DISTRIBUTION for a mode, combined from the grounded stances by the mode's
    DECISION STRUCTURE (mode_graph.combine_stances) — unanimity/veto, majority, hierarchy, unilateral,
    weakest-link, cumulative pressure, or aggregation. "Most-opposed binds" is the unanimity case,
    not a universal law. Effect sizes come from the fitted pack when it exists, else the documented
    priors; reliability/capability/graded-control shrinks happen inside the combiner."""
    return combine_stances(stances, pathway, mode=mode, hr_table=_hr_table())


def agreement_hazard_ratio(stances: list) -> dict:
    """Back-compat wrapper: the cooperative-pathway (unanimity) case of mode_hazard_ratio."""
    return mode_hazard_ratio(stances, "cooperative_agreement")


def fit_intention_hazard_ratios(rows: list, *, pool_strength: float = 8.0) -> dict:
    """Fit statement-class hazard ratios from RESOLVED historical cases. Each row:
    {commitment_level, hazard_ratio} where hazard_ratio = (resolving-state hazard after the statement) /
    (hazard before), measured on archived paths of resolved cases. Partial pooling toward 1.0 (no
    effect). Writes nothing — caller persists to INTENTION_HR_PACK. Until such labeled data exists the
    documented INTENTION_HR_PRIORS above serve, and are reported as priors."""
    import statistics
    out = {}
    by = {}
    for r in rows:
        lvl = canon_level(r.get("commitment_level"))
        hr = r.get("hazard_ratio")
        if lvl and isinstance(hr, (int, float)) and hr > 0:
            by.setdefault(lvl, []).append(math.log(float(hr)))
    for lvl, logs in by.items():
        k = pool_strength / (len(logs) + pool_strength)
        med = math.exp((1 - k) * statistics.median(logs))            # pooled toward log(1.0)=0
        sd = statistics.pstdev(logs) if len(logs) > 1 else 0.35
        out[lvl] = (round(med, 4), round(med * math.exp(-_Z80 * sd), 4),
                    round(med * math.exp(_Z80 * sd), 4))
    return {"version": "intention-hr-1.0", "fit_on": "resolved historical statement/outcome pairs",
            "n_rows": len(rows), "hazard_ratios": {k: list(v) for k, v in out.items()}}


# ---------------------------------------------------------------- 1. the contract
class EventTimeContract:
    """First-passage terminal contract — a pure READOUT. project(branches) reads absorbed_at/absorbed_by
    from each terminal world; nothing here decides or draws. With `binary_options` the projection also
    answers the question in its own two options: P(occurrence option) = F(deadline_ts); a survival-polarity
    question (`occurrence_resolves="no"`) maps absorption to the negative option."""
    family = "event_time"

    def __init__(self, *, as_of: float, horizon_ts: float, resolves_iff: str = "",
                 modes: list = None, readout_var: str = "absorbed_at",
                 binary_options: list = None, occurrence_resolves: str = "yes",
                 deadline_ts: float = None):
        self.as_of, self.horizon_ts = float(as_of), float(horizon_ts)
        self.resolution_rule = resolves_iff[:300]
        self.modes = list(modes or [])
        self.readout_var = readout_var
        self.binary_options = [str(o) for o in (binary_options or [])][:2]
        self.occurrence_resolves = "no" if str(occurrence_resolves) == "no" else "yes"
        self.deadline_ts = float(deadline_ts) if deadline_ts else float(horizon_ts)
        self.options = (list(self.binary_options) if len(self.binary_options) == 2
                        else ["absorbed_by_horizon", "censored_beyond_horizon"])

    def validate(self):
        if self.horizon_ts <= self.as_of:
            raise ValueError("event_time contract needs horizon > as_of")
        if not (self.as_of < self.deadline_ts <= self.horizon_ts):
            self.deadline_ts = self.horizon_ts
        return self

    def readout(self, world):
        q = world.quantities.get("absorbed_at")
        v = getattr(q, "value", None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        # belt: a writer set the absorbing predicate on the branch's LAST event, so the monitor never got
        # a later event to stamp on — the write time on the predicate quantity is the first-passage time
        flag = world.quantities.get("absorbing_state_reached")
        if getattr(flag, "value", None):
            ts = getattr(flag, "timestamp", None)
            return float(ts) if isinstance(ts, (int, float)) and ts > 0 else None
        return None

    def _mode_of(self, world) -> str:
        m = world.quantities.get("absorbed_by") or world.quantities.get("absorbing_mode")
        return str(getattr(m, "value", None) or "unspecified")

    def project(self, branches) -> dict:
        times, modes = [], {}                                  # times: [(t, weight)]
        n = sum(max(0.0, float(getattr(b, "weight", 1.0))) for b in branches) or 1.0
        for b in branches:
            w = max(0.0, float(getattr(b, "weight", 1.0)))
            t = self.readout(b.world)
            if isinstance(t, (int, float)) and t > 0:
                times.append((float(t), w))
                mid = self._mode_of(b.world)
                modes[mid] = modes.get(mid, 0.0) + w
        times.sort(key=lambda x: x[0])
        p_absorbed = sum(w for _, w in times) / n
        span = self.horizon_ts - self.as_of
        grid = [self.as_of + k / 10.0 * span for k in range(1, 11)]
        cdf = [round(sum(w for t, w in times if t <= g) / n, 4) for g in grid]
        qtl = {}
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            acc, val = 0.0, None
            for t, w in times:
                acc += w
                if acc >= q * n:
                    val = round(t, 1)
                    break
            qtl[str(q)] = val                                  # None = beyond horizon
        et = {"n_particles": len(branches), "n_absorbed": len(times),
              "p_censored": round(1 - p_absorbed, 4),
              "cdf_grid_ts": [round(g, 0) for g in grid], "cdf": cdf,
              "survival": [round(1 - c, 4) for c in cdf],
              "first_passage_quantiles_ts": qtl,
              "mode_distribution": {k: round(v / n, 4) for k, v in modes.items()}}
        if len(self.binary_options) == 2:
            # the question's own answer is a READOUT of the same trajectories: F(deadline), polarity-mapped
            p_occ = sum(w for t, w in times if t <= self.deadline_ts) / n
            p_yes = p_occ if self.occurrence_resolves == "yes" else 1.0 - p_occ
            dist = {self.binary_options[0]: round(p_yes, 4),
                    self.binary_options[1]: round(1.0 - p_yes, 4)}
            et["p_event_by_deadline"] = round(p_occ, 4)
            et["deadline_ts"] = round(self.deadline_ts, 0)
            et["occurrence_resolves"] = self.occurrence_resolves
        else:
            dist = {"absorbed_by_horizon": round(p_absorbed, 4),
                    "censored_beyond_horizon": round(1 - p_absorbed, 4)}
        return {"distribution": dist, "family": "event_time",
                "n_deltas": sum(len(b.log) for b in branches),
                "readout": "terminal_states", "event_time": et}

    def cdf_at(self, deadline_ts: float, branches) -> float:
        n = sum(max(0.0, float(getattr(b, "weight", 1.0))) for b in branches) or 1.0
        acc = 0.0
        for b in branches:
            t = self.readout(b.world)
            if isinstance(t, (int, float)) and t <= deadline_ts:
                acc += max(0.0, float(getattr(b, "weight", 1.0)))
        return acc / n


# ---------------------------------------------------------------- censoring-aware scoring
# Pure functions — the frozen event-time vault scorer and any future benchmark call these; no I/O.
def crps_first_passage(cdf_grid_ts: list, cdf: list, *, event_ts=None, as_of: float,
                       horizon_ts: float) -> float:
    """CRPS of a first-passage CDF against a realized event time, censoring-aware.

    The forecast is the stepwise CDF F on `cdf_grid_ts` (right-continuous between grid points; F=0
    before the first point; F stays at its last value up to the horizon). `event_ts=None` means the
    true event was CENSORED (did not occur by horizon): the target CDF is 0 on the whole window, so
    CRPS = ∫ F(t)² dt / span — probability mass placed inside the window is penalized, exactly.
    With an observed event time T: CRPS = [∫_{as_of}^{T} F² + ∫_{T}^{horizon} (1−F)²] / span.
    Normalized by the window span so scores are comparable across questions. Lower is better."""
    lo, hi = float(as_of), float(horizon_ts)
    if hi <= lo or not cdf_grid_ts or len(cdf_grid_ts) != len(cdf):
        raise ValueError("crps_first_passage needs a nonempty grid inside a positive window")
    knots = [lo] + [float(t) for t in cdf_grid_ts] + [hi]
    vals = [0.0] + [max(0.0, min(1.0, float(c))) for c in cdf]           # F on [knot_i, knot_{i+1})
    total = 0.0
    t_ev = float(event_ts) if isinstance(event_ts, (int, float)) else None
    for i in range(len(knots) - 1):
        a, b = knots[i], min(knots[i + 1], hi)
        if b <= a:
            continue
        f = vals[min(i, len(vals) - 1)]
        if t_ev is None or t_ev >= b:            # target still 0 on [a,b)
            total += (b - a) * f * f
        elif t_ev <= a:                          # target already 1 on [a,b)
            total += (b - a) * (1.0 - f) * (1.0 - f)
        else:                                    # event inside this piece
            total += (t_ev - a) * f * f + (b - t_ev) * (1.0 - f) * (1.0 - f)
    return total / (hi - lo)


def interval_coverage(quantiles: dict, event_ts, *, lo_q: str = "0.1", hi_q: str = "0.9"):
    """Was the realized event time inside the forecast's [lo_q, hi_q] first-passage interval?
    Censoring-aware: a censored outcome (event_ts None) is covered iff the UPPER quantile is already
    beyond the horizon (None) — the forecast itself put the upper tail outside the window."""
    lo, hi = quantiles.get(lo_q), quantiles.get(hi_q)
    if event_ts is None:
        return hi is None
    if lo is None:
        return False                              # forecast said beyond-horizon; event happened inside
    t = float(event_ts)
    return (t >= float(lo)) and (hi is None or t <= float(hi))


# ---------------------------------------------------------------- 2. the absorption monitor
class AbsorptionMonitorOperator(TransitionOperator):
    """Universal first-passage observer: after any event, if the canonical absorbing predicate holds
    (quantities[absorbing_state_reached] truthy — mechanisms WRITE it; they never resolve outcomes) and no
    absorption is stamped yet, stamp absorbed_at = clock.now and absorbed_by, immutably (first passage)."""
    name = "absorption_monitor"

    def applicable(self, world, event):
        q = world.quantities.get("absorbing_state_reached")
        stamped = world.quantities.get("absorbed_at")
        return bool(getattr(q, "value", None)) and getattr(stamped, "value", None) in (None, 0)

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action={"at": world.clock.now},
                                  reason_codes=["first_passage"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        register_quantity_type("absorbed_at", units="unix_ts")
        register_quantity_type("absorbed_by", units="mode")
        world.quantities["absorbed_at"] = Quantity(name="absorbed_at", qtype="absorbed_at",
                                                   value=float(world.clock.now),
                                                   timestamp=world.clock.now)
        mode_q = world.quantities.get("absorbing_mode")
        world.quantities["absorbed_by"] = Quantity(name="absorbed_by", qtype="absorbed_by",
                                                   value=str(getattr(mode_q, "value", None) or "unspecified"),
                                                   timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="absorption", operator=self.name,
                       reason_codes=["first_passage_observed"])
        return d.change("quantities[absorbed_at]", None, float(world.clock.now))


# ---------------------------------------------------------------- 3. hazard-mode rounds
class HazardRoundOperator(TransitionOperator):
    """One causal round of a process that may enter the absorbing state. Three parameterizations of the
    per-round success hazard (all inside the MECHANISM — the readout never decides):

      curve      payload = {mode, base_hazard | hazard_bucket_curve, hr {median,lo80,hi80} (or legacy
                 intention_factor), consume}: fitted family curve(t) × per-branch SAMPLED stance hazard
                 ratio ("when" questions).
      calibrated payload["calibration"] = {exponent, absorb_from, fact_floor, posterior_rate_particles?,
                 fallback_rate?, fallback_provenance?, lean}: ONE target absorbed-mass per particle is
                 drawn from the posterior rate particles (fallback: fitted family rate, then broad
                 lean-Beta), polarity-mapped, budgeted down by absorbing-fact mass, and spread across the
                 chain as h_k = 1-(1-target)^exponent_k (Σ exponent = 1 ⇒ the chain's total absorbed mass
                 reproduces the target exactly before structural modulation).
      fact       payload["success_prob"]: an outcome-entailing dated fact fires at ITS date with
                 probability = confidence — a calendar fact is not intention- or crowd-modulated.

    Hazard parameterizations consume causal state RELATIVELY (`_consume_state_hazard`): consumed
    quantities written by upstream mechanisms — including the pathway-process quantities the simulated
    actors' own ACTIONS move (phase4_execution) — act as a bounded multiplicative modifier centered at
    no-effect (state 0.5 → ×1). The absolute blend in consume_state_rate (p' = (1-Σw)p + Σw·v) is
    correct for a one-shot probability decision but destroys timing structure when applied to a small
    per-round hazard — a mid-level state would swamp the hazard base every round.

    On success: writes absorbing_state_reached + absorbing_mode at THIS round's date."""
    name = "hazard_round"

    def applicable(self, world, event):
        q = world.quantities.get("absorbed_at")
        flag = world.quantities.get("absorbing_state_reached")
        return event.etype == "hazard_round" and getattr(q, "value", None) in (None, 0) \
            and not getattr(flag, "value", None)               # first passage: never re-fire past absorption

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=[f"mode={event.payload.get('mode', '?')}"])

    def _sampled_hr(self, world, mode, hr):
        """One hazard-ratio draw PER BRANCH per mode (lognormal from the prior's 80% interval),
        persisted on the world so every round in this branch sees the same sampled effect size —
        the uncertainty over the coefficient becomes cross-particle spread in the terminal CDF."""
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import _branch_rng
        qname = f"sampled_intention_hr:{mode}"
        q = world.quantities.get(qname)
        if q is not None and isinstance(getattr(q, "value", None), (int, float)):
            return float(q.value)
        med = max(1e-6, float(hr.get("median", 1.0)))
        lo, hi = float(hr.get("lo80", med)), float(hr.get("hi80", med))
        sigma = (math.log(max(hi, 1e-6)) - math.log(max(lo, 1e-6))) / (2 * _Z80) if hi > lo else 0.0
        rng = _branch_rng(world, f"hr:{mode}")
        val = med * math.exp(sigma * rng.gauss(0.0, 1.0)) if sigma > 0 else med
        val = max(0.05, min(3.0, val))
        register_quantity_type("sampled_intention_hr", units="hazard_ratio")
        world.quantities[qname] = Quantity(name=qname, qtype="sampled_intention_hr", value=val,
                                           timestamp=world.clock.now)
        return val

    @staticmethod
    def _consume_state_hazard(world, h, consume):
        """RELATIVE consumption for hazards: consumed causal state acts as a bounded MULTIPLICATIVE
        modifier centered at no-effect (state 0.5 → ×1). An entry may set `invert: true`
        (survival-polarity chains: pro-YES state must SUPPRESS the state-breaking hazard, v ↦ 1−v).
        Weight 0.45 at an extreme state moves the hazard ×2^0.45 ≈ 1.37 (or ÷); a weight-1.0
        pathway-process channel reaches ×2 — deliberately bounded; total factor clamped to [0.25, 4]."""
        used, logf = [], 0.0
        for m in (consume or []):
            var, w = str(m.get("var", "")), float(m.get("weight", 0.0) or 0.0)
            q = world.quantities.get(var)
            v = getattr(q, "value", None)
            if w <= 0.0 or not isinstance(v, (int, float)):
                continue
            v = max(0.0, min(1.0, float(v)))
            if m.get("invert"):
                v = 1.0 - v
            logf += w * (v - 0.5) * 2.0 * math.log(2.0)
            used.append(var)
        if not used:
            return h, [], 1.0
        f = max(0.25, min(4.0, math.exp(logf)))
        return h * f, used, f

    def _calibrated_target(self, world, cal: dict) -> tuple:
        """Per-particle target absorbed-mass: posterior draw (one per branch — the SAME salt across the
        branch's rounds) → hypothesis-lean shift → polarity map → absorbing-fact residual budget."""
        from swm.world_model_v2.fallback import LEAN_BETA, _apply_lean_shift, _beta_sample
        from swm.world_model_v2.phase_consumers import _branch_rng, _draw_rate
        rng = _branch_rng(world, "event_time_target")
        parts = cal.get("posterior_rate_particles")
        if parts:
            r, src = _draw_rate(parts, rng), "posterior"
        elif isinstance(cal.get("fallback_rate"), (int, float)):
            r, src = float(cal["fallback_rate"]), str(cal.get("fallback_provenance")
                                                      or "fitted_family_prior")
        else:
            av, bv = LEAN_BETA.get(str(cal.get("lean", "neutral")), (1.0, 1.0))
            r, src = _beta_sample(rng, av, bv), "prior_beta"
        lean_h = (getattr(world, "uncertainty_meta", None) or {}).get("hypothesis_lean")
        if lean_h:                                            # competing structures shift the target rate
            r = _apply_lean_shift(max(0.0, min(1.0, r)), str(lean_h))
        r = max(0.0, min(1.0, r))
        t = r if str(cal.get("absorb_from", "rate")) == "rate" else 1.0 - r
        c = max(0.0, min(0.999, float(cal.get("fact_floor", 0.0) or 0.0)))
        if c > 0.0:                                           # absorbing facts already carry mass c
            t = max(0.0, (t - c) / (1.0 - c))
        return max(0.0, min(0.98, t)), src

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import _branch_rng
        a = proposal.action
        frac = min(0.999, max(0.0, (world.clock.now - float(a.get("as_of", world.clock.now)))
                              / max(1.0, float(a.get("span_s", 1.0)))))
        unc = {"lifetime_fraction": round(frac, 3)}
        used, sfac, hr_used = [], 1.0, None
        if a.get("success_prob") is not None:                 # dated entailing fact — fires at ITS date
            h = max(0.0, min(0.999, float(a["success_prob"])))
            unc["rate_source"] = "entailed_fact_confidence"
        elif isinstance(a.get("calibration"), dict):          # posterior-calibrated residual chain
            t, src = self._calibrated_target(world, a["calibration"])
            exp = max(0.0, float(a["calibration"].get("exponent", 0.0) or 0.0))
            h = 1.0 - (1.0 - t) ** exp
            if isinstance(a.get("hr"), dict):                 # distributional stance ratio (sampled)
                hr_used = self._sampled_hr(world, str(a.get("mode")), a["hr"])
                h *= hr_used
            else:
                h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
            h, used, sfac = self._consume_state_hazard(world, h, a.get("consume") or [])
            h = max(0.0, min(0.95, h))
            unc.update({"rate_source": src, "target_mass": round(t, 4)})
        else:                                                 # fitted family curve ("when" questions)
            curve = a.get("hazard_bucket_curve") or []
            h = float(curve[min(int(frac * _BUCKETS), _BUCKETS - 1)]) if curve \
                else float(a.get("base_hazard", 0.05))
            if isinstance(a.get("hr"), dict):                 # distributional stance ratio (sampled)
                hr_used = self._sampled_hr(world, str(a.get("mode")), a["hr"])
                h *= hr_used
            else:                                             # legacy point factor
                h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
            h, used, sfac = self._consume_state_hazard(world, h, a.get("consume") or [])
            h = max(0.0, min(0.95, h))
        unc.update({"hazard": round(h, 4), "consumed": used,
                    "state_hazard_factor": round(sfac, 4),
                    "sampled_hazard_ratio": (round(hr_used, 4) if hr_used is not None else None),
                    "intention_factor": a.get("intention_factor")})
        rng2 = _branch_rng(world, f"hz:{a.get('mode')}:{world.clock.now}")
        d = StateDelta(at=world.clock.now, event_type="hazard_round", operator=self.name,
                       reason_codes=proposal.reason_codes + [f"h={round(h, 4)}"],
                       uncertainty=unc)
        if rng2.random() < h:
            register_quantity_type("absorbing_state_reached", units="bool")
            register_quantity_type("absorbing_mode", units="mode")
            world.quantities["absorbing_state_reached"] = Quantity(
                name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                timestamp=world.clock.now)
            world.quantities["absorbing_mode"] = Quantity(
                name="absorbing_mode", qtype="absorbing_mode", value=str(a.get("mode", "unspecified")),
                timestamp=world.clock.now)
            d.change("quantities[absorbing_state_reached]", None, True)
        return d


register_operator("absorption_monitor", AbsorptionMonitorOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="pure observation of the absorbing predicate (first passage)",
                  validated=True)
register_operator("hazard_round", HazardRoundOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="fitted family hazard curve × sampled stance hazard ratio × consumed "
                                   "causal state (bounded, inside the mechanism)", validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("hazard_round", "absorption"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("quantities",), deltas=("quantities",),
                            parameter_source="event-time architecture", validated=True)


# ---------------------------------------------------------------- 4. fitted survival pack
def fit_survival_pack(worlds_with_paths: list, *, pool_strength: float = 6.0) -> dict:
    """worlds_with_paths: [{question, lifetime_fraction_resolved | None}] — calibration split ONLY.
    `lifetime_fraction_resolved` = the effective-resolution proxy (preferred: the market's true
    resolution timestamp; else the STICKY 0.9 crossing — see replay_v3/fit_survival_pack.py), as a
    fraction of market lifetime; None = never (censored). Fits discrete per-bucket hazards with
    partial pooling toward the global curve."""
    from swm.world_model_v2.family_hazards import classify_family

    def _hazards(rows):
        alive = len(rows)
        hs = []
        for b in range(_BUCKETS):
            lo, hi = b / _BUCKETS, (b + 1) / _BUCKETS
            events = sum(1 for f in rows if f is not None and lo <= f < hi)
            hs.append(events / alive if alive else 0.0)
            alive -= events
        return hs
    all_f = [w.get("lifetime_fraction_resolved") for w in worlds_with_paths]
    g = _hazards(all_f)
    fams = {}
    for w in worlds_with_paths:
        fams.setdefault(classify_family(w["question"]), []).append(w.get("lifetime_fraction_resolved"))
    out = {}
    for fam, rows in fams.items():
        hf = _hazards(rows)
        k = pool_strength / (len(rows) + pool_strength)
        out[fam] = {"n": len(rows), "hazards": [round((1 - k) * a + k * b, 4) for a, b in zip(hf, g)]}
    return {"version": "family-survival-1.0", "fit_on": "calibration split only",
            "global_hazards": [round(x, 4) for x in g], "families": out}


def family_hazard_curve(question: str) -> tuple:
    from swm.world_model_v2.family_hazards import classify_family
    fam = classify_family(question)
    if not SURV_PACK.exists():
        return None, fam, "no_pack"
    pack = json.loads(SURV_PACK.read_text())
    ent = (pack.get("families") or {}).get(fam)
    if ent:
        return ent["hazards"], fam, f"fitted_family_survival(n={ent['n']})"
    return pack.get("global_hazards"), fam, "global_pooled"


# ---------------------------------------------------------------- 5. plan conversion (unification)
_WHEN_TOKENS = ("when will", "when does", "how long until", "how soon", "by what date", "what date will")


def is_when_question(question: str) -> bool:
    q = str(question).lower()
    return any(t in q for t in _WHEN_TOKENS)


def _mode_pathway(mode) -> str:
    """The causal pathway a mode is reached through (mode_graph owns the taxonomy + fallbacks)."""
    return mode_pathway(mode)


def _mode_requires_agreement(mode) -> bool:
    return _mode_pathway(mode) == "cooperative_agreement"


def _filter_absorbing_modes(modes: list, criterion: dict, llm) -> tuple:
    """A structural hypothesis is only an ABSORBING mode if the state it describes satisfies the parsed
    resolution criterion (NATO escalation does not END a conflict). LLM entailment judgment, universal;
    on any failure every mode is kept (never blocks the forecast)."""
    iff = (criterion or {}).get("resolves_yes_iff")
    if llm is None or not iff or len(modes) < 2:
        return modes, []
    try:
        from swm.engine.grounding import parse_json
        raw = parse_json(llm(
            "Which of these candidate end-states SATISFY the resolution criterion when reached?\n"
            f"CRITERION (resolves yes iff): {iff}\n"
            f"CANDIDATE MODES: {[m['id'] for m in modes]}\n"
            'Return ONLY JSON: {"absorbing_mode_ids": ["..."], '
            '"rejected": [{"id": "...", "why": "<does not satisfy the criterion>"}]}')) or {}
        ids = {str(x) for x in (raw.get("absorbing_mode_ids") or [])}
        kept = [m for m in modes if m["id"] in ids]
        rejected = [m["id"] for m in modes if m["id"] not in ids]
        return (kept, rejected) if kept else (modes, [])
    except Exception:  # noqa: BLE001 — entailment filtering must never block the forecast
        return modes, []


def _ensure_event_time_mechanisms(plan, curve_src: str):
    """Accept the timing machinery on the plan. ORDER MATTERS: absorbing writers (hazard_round) must come
    BEFORE the absorption monitor in the operator list so the monitor observes a same-event write and
    stamps first passage at the write's own clock time, not one event late."""
    for mech_id, op in (("hazard_rounds", "hazard_round"), ("absorption_monitor", "absorption_monitor")):
        if not any(x.get("operator") == op for x in plan.accepted_mechanisms if isinstance(x, dict)):
            plan.accepted_mechanisms.append({
                "mech_id": mech_id, "ontology_type": "event_time", "operator": op,
                "causal_role": "first-passage timing machinery",
                "parameter_source": curve_src if op == "hazard_round" else "observation",
                "temporal_scale": "event", "calibration_status": curve_src, "sensitivity": 1.0})


def _declare_readout_quantities(plan):
    """Declare the readout quantities on the plan so build_world registers them and the contract's
    readout binds at materialization (value None until first passage stamps it)."""
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    for name in ("absorbed_at", "absorbing_state_reached"):
        if name not in declared:
            plan.quantities.append({"name": name, "qtype": name, "value": None, "sd": None})


def _endogenous_consume(plan, mode, base_consumed: list) -> tuple:
    """The mode's state-consumption channels — the ENDOGENOUS half of the hazard clock:
      * its OWN pathway-process quantity (weight 1.0): the quantity the simulated actors' actions,
        institutional stage reviews and world-driven consumers write each round — declared by
        mode_graph.declare_pathway_processes; a dormant process suppresses the hazard, an advanced
        one raises it;
      * every OTHER declared pathway process at a smaller cross weight (0.25): resolution pressure
        spills over — a collapsing battlefield forces parties to the table ("negotiations begin
        because battlefield state changed"), advancing institutional stages raise unilateral urgency;
      * for WORLD-DRIVEN pathways (threshold/diffusion/market/physical/…): the plan's declared
        nonlinear state and population aggregates, when present — non-actor mechanisms drive those
        hazards; stances barely touch them (mode_graph handles that on the HR side).
    Returns (consume list, endogenous_channel_live)."""
    consume = [dict(m) for m in (base_consumed or []) if isinstance(m, dict)]
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    pw = _mode_pathway(mode)
    live = False
    own = progress_var(pw)
    if own in declared:
        consume.append({"var": own, "weight": 1.0})
        live = True
    for other_pw in sorted({p for p in (getattr(plan, "_declared_pathways", None) or [])
                            if p != pw}):
        v = progress_var(other_pw)
        if v in declared:
            consume.append({"var": v, "weight": 0.25})
    if not pathway_of(pw).actor_driven:
        for q in (getattr(plan, "quantities", []) or []):
            name = str(q.get("name", "")) if isinstance(q, dict) else ""
            if name.startswith("population_aggregate:") or name == "nonlinear_state":
                if not any(c.get("var") == name for c in consume):
                    consume.append({"var": name, "weight": 0.35})
    return consume, live


def convert_to_event_time(plan, criterion: dict, *, lineage: dict = None, llm=None) -> dict:
    """Replace the compiled point contract with an EventTimeContract and schedule the timing machinery:
    per-mode hazard_round chains at the trajectory cadence, the absorption monitor, and per-mode stance
    hazard-ratio DISTRIBUTIONS combined by each mode's decision structure. Universal: built only from
    the plan's own structure + the canonical mode graph."""
    # canonical mode set: compiler hypotheses + K-pass elicitation reconciled by mode_graph (compile-
    # variance fix). If unified_runtime already computed it (before intention grounding, so stances
    # could be mode-scoped), reuse it — no second elicitation.
    modes = list(getattr(plan, "_canonical_modes", None) or [])
    consensus = dict(getattr(plan, "_mode_consensus", None) or {})
    if not modes:
        modes, consensus = canonical_modes(
            question=getattr(plan, "question", ""), criterion=criterion,
            hypotheses=list(getattr(plan, "structural_hypotheses", []) or []),
            options=list(getattr(plan.outcome_contract, "options", []) or []), llm=llm)
    modes, rejected_modes = _filter_absorbing_modes(modes, criterion, llm)
    z = sum(m["prior"] for m in modes) or 1.0
    curve, fam, curve_src = family_hazard_curve(getattr(plan, "question", ""))
    # grounded-stance effect, MODE-SCOPED and structure-combined: each mode's hazard-ratio
    # DISTRIBUTION is combined from the stances concerning THAT mode (or its pathway) under the
    # mode's decision structure — never a point coefficient invented by the LLM, and never a
    # universal "most-opposed binds" shortcut.
    stances = list(getattr(plan, "_intention_stances", []) or [])

    def _pathway_hr(pathway, mode=None):
        if pathway == "cooperative_agreement" and AGREEMENT_HR_OVERRIDE is not None:
            v = float(AGREEMENT_HR_OVERRIDE)                  # sensitivity harness only
            return {"median": v, "lo80": v, "hi80": v, "binding_actor": "OVERRIDE",
                    "binding_level": "sensitivity_override", "binding_pathway": pathway,
                    "combination_rule": "override"}
        if pathway != "cooperative_agreement" and VICTORY_HR_OVERRIDE is not None:
            v = float(VICTORY_HR_OVERRIDE)                    # sensitivity harness only
            return {"median": v, "lo80": v, "hi80": v, "binding_actor": "OVERRIDE",
                    "binding_level": "sensitivity_override", "binding_pathway": pathway,
                    "combination_rule": "override"}
        return mode_hazard_ratio(stances, pathway, mode=mode)
    agr_hr = _pathway_hr("cooperative_agreement")             # reported for auditability
    span = plan.horizon_ts - plan.as_of
    horizon_days = max(1.0, span / 86400.0)
    n_rounds = max(4, min(20, int(horizon_days / max(1.0, horizon_days / 10.0))))
    base_consumed = list(getattr(plan, "_consumed_state", []) or [])
    n_ev = 0
    mode_hr = {}
    for m in modes[:6]:
        share = m["prior"] / z
        # each mode consumes the stance hazard ratio of ITS OWN causal pathway under ITS decision
        # structure, and the endogenous state channels of its pathway processes
        pathway = _mode_pathway(m)
        hr_m = _pathway_hr(pathway, mode=m)
        consume_m, endo_live = _endogenous_consume(plan, m, base_consumed)
        if endo_live and hr_m.get("combination_rule") != "override":
            # ANTI-DOUBLE-COUNT: when the behavioral channel is live (stances condition the Phase-4
            # policies whose actions move the consumed pathway process), part of the stance's total
            # effect flows through behavior — the DIRECT multiplier keeps only the residual share
            # (log-split, documented; sensitivity-harness variable).
            s = ENDOGENOUS_STANCE_SPLIT
            hr_m = dict(hr_m, median=round(math.exp(s * math.log(max(1e-6, hr_m["median"]))), 4),
                        lo80=round(math.exp(s * math.log(max(1e-6, hr_m["lo80"]))), 4),
                        hi80=round(math.exp(s * math.log(max(1e-6, hr_m["hi80"]))), 4),
                        endogenous_split=s)
        mode_hr[m["id"]] = {k: hr_m.get(k) for k in ("median", "lo80", "hi80", "binding_actor",
                                                     "binding_level", "combination_rule",
                                                     "endogenous_split")}
        mode_hr[m["id"]]["pathway"] = pathway
        agreement = pathway == "cooperative_agreement"
        for k in range(1, n_rounds + 1):
            ts = plan.as_of + (k / (n_rounds + 1)) * span
            # per-round per-mode hazard: bucket hazard spread over the bucket's rounds (n_rounds/_BUCKETS)
            # weighted by the mode's structural share — summed over modes+rounds it reproduces the fitted
            # family curve exactly (no len(modes) inflation)
            plan.scheduled_events.append({
                "etype": "hazard_round", "ts": ts, "participants": [],
                "payload": {"mode": m["id"], "base_hazard": 0.5 * share / n_rounds,
                            "hazard_bucket_curve": ([h * share * _BUCKETS / n_rounds for h in curve]
                                                    if curve else None),
                            "hr": {k2: hr_m.get(k2) for k2 in ("median", "lo80", "hi80")},
                            "requires_agreement": agreement, "pathway": pathway,
                            "as_of": plan.as_of, "span_s": span,
                            "consume": consume_m}})
            n_ev += 1
    _ensure_event_time_mechanisms(plan, curve_src)
    plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") != "resolve_outcome"]
    _declare_readout_quantities(plan)
    plan.outcome_contract = EventTimeContract(as_of=plan.as_of, horizon_ts=plan.horizon_ts,
                                              resolves_iff=str((criterion or {}).get("resolves_yes_iff",
                                                                                     ""))[:300],
                                              modes=[m["id"] for m in modes]).validate()
    # a first-passage CDF needs particle resolution a point forecast does not: floor the particle
    # count for EVERY event-time contract (rollout is LLM-free per particle, so this is cheap)
    cp = getattr(plan, "compute_plan", None)
    n_particles = None
    if isinstance(cp, dict):
        cp["n_particles"] = max(int(cp.get("n_particles", 30) or 30), 200)
        n_particles = cp["n_particles"]
    rep = {"modes": [m["id"] for m in modes], "rejected_non_absorbing_modes": rejected_modes,
           "mode_consensus": consensus or None,
           "n_particles": n_particles,
           "n_hazard_rounds": n_ev, "rounds_per_mode": n_rounds,
           "agreement_hazard_ratio": agr_hr, "hazard_ratio_by_mode": mode_hr,
           "hazard_ratio_source": ("fitted_pack" if INTENTION_HR_PACK.exists()
                                   else "documented_priors_unfitted"),
           "n_grounded_stances": len(stances),
           "declared_pathways": sorted(getattr(plan, "_declared_pathways", None) or []),
           "family": fam, "hazard_curve_source": curve_src}
    if lineage is not None:
        lineage["event_time"] = rep
    return rep


# ---------------------------------------------------------------- 6. binary unification (readout, no resolver)
#: question phrasings whose YES state is a STATE PERSISTING to the deadline — there the absorbing event is
#: the state-BREAKING event and its occurrence resolves NO (survival polarity). Lexical fallback only; the
#: resolution-criterion parser's `event_polarity` wins when present.
_SURVIVAL_TOKENS = ("remain", "stay ", "stays ", "still be", "still the", "still in", "retain",
                    "keep his", "keep her", "keep its", "keep their", "keeps his", "keeps her",
                    "keeps its", "continue as", "continues as", "continue to be", "hold onto",
                    "holds onto", "in office on", "in office through", "in office at", "survive as",
                    "survives as", "through the end of", "hold on to", "stay above", "stay below",
                    "remain above", "remain below")


def _lexical_event_polarity(question: str) -> str:
    q = " " + str(question).lower()
    return "occurrence_resolves_no" if any(t in q for t in _SURVIVAL_TOKENS) \
        else "occurrence_resolves_yes"


def _event_polarity(question: str, criterion: dict) -> tuple:
    """Does the FIRST OCCURRENCE of the decisive event resolve the question YES (event-by-deadline) or NO
    (remains/still-in-state)? Criterion parser's judgment wins; lexical fallback keeps this universal
    when no LLM ran."""
    pol = str((criterion or {}).get("event_polarity") or "")
    if pol in ("occurrence_resolves_yes", "occurrence_resolves_no"):
        return pol, "criterion_parser"
    return _lexical_event_polarity(question), "lexical"


def _mass_weights_from_curve(curve) -> list:
    """Fitted bucket hazards → normalized per-bucket EVENT-MASS weights (the SHAPE of when resolutions
    happen in the window). Uniform when no curve was fit."""
    if not curve or len(curve) != _BUCKETS or not any(h > 0 for h in curve):
        return [1.0 / _BUCKETS] * _BUCKETS
    surv, mass = 1.0, []
    for h in curve:
        h = max(0.0, min(1.0, float(h)))
        mass.append(surv * h)
        surv *= (1.0 - h)
    z = sum(mass) or 1.0
    return [m / z for m in mass]


def convert_binary_to_event_time(plan, criterion: dict, *, lineage: dict = None, llm=None,
                                 cadence_days: float = None) -> dict:
    """Rewire a binary deadline question so the answer is DERIVED from the simulation instead of declared
    by a terminal resolver. Universal — built only from the plan's own structure:

      * the resolver events (resolve_outcome, aggregate_outcome_resolution) are REMOVED — no component
        draws the outcome;
      * outcome-entailing dated facts become absorbing events at their REAL dates (probability = fact
        confidence; the residual chain is budgeted down by their mass);
      * an institutional decision that decides THIS outcome becomes an absorbing writer at its scheduled
        date (payload flag) — when one exists it IS the resolution process and no residual chain runs;
      * otherwise a residual hazard chain spreads the per-particle posterior-drawn target mass over the
        window (shaped by the fitted family curve), so the evidence-updated calibration that used to bias
        the terminal coin now parameterizes causal dynamics whose first passage is OBSERVED — and the
        chain consumes the declared pathway-process quantities, so the simulated actors' ACTIONS move
        the binary answer through the same endogenous channel as timing questions;
      * P(yes) = F(deadline) (or 1−F for survival-polarity questions) — a pure readout, coherent across
        cutoffs by construction, with the timing curve and resolution-mode marginal from the SAME worlds.

    Returns a report; {"skipped": ...} (no mutation) for non-binary contracts."""
    contract = getattr(plan, "outcome_contract", None)
    family = str(getattr(contract, "family", ""))
    options = [str(o) for o in (getattr(contract, "options", None) or [])]
    if family not in ("binary", "response_occurrence") or len(options) != 2:
        rep = {"skipped": f"family={family or '?'} n_options={len(options)} — binary conversion "
                          f"applies to 2-option binary/response_occurrence contracts only"}
        if lineage is not None and "event_time" not in (lineage or {}):
            lineage["event_time"] = rep
        return rep

    question = str(getattr(plan, "question", ""))
    polarity, polarity_src = _event_polarity(question, criterion)
    absorb_dir = "yes" if polarity == "occurrence_resolves_yes" else "no"

    # ---- deadline: the criterion's parsed deadline when it falls inside the window, else the horizon ----
    deadline_ts = float(plan.horizon_ts)
    d = (criterion or {}).get("deadline")
    if d:
        try:
            from swm.world_model_v2.state import parse_time
            ts = parse_time(str(d)[:10])
            if plan.as_of < ts < plan.horizon_ts:
                deadline_ts = ts
        except (ValueError, TypeError):
            pass
    span = deadline_ts - plan.as_of

    # ---- the resolver's outcome variable (needed to recognize the deciding institution) ----
    rev = next((e for e in plan.scheduled_events if e.get("etype") == "resolve_outcome"), None)
    resolve_var = str(((rev or {}).get("payload") or {}).get("outcome_var", "outcome"))
    lean = str(((rev or {}).get("payload") or {}).get("lean")
               or (getattr(plan, "provenance", None) or {}).get("outcome_lean", "neutral"))

    # ---- absorbing facts: outcome-entailing dated facts whose direction IS the absorbing side ----
    absorbing_facts, opposite_facts = [], []
    for e in plan.scheduled_events:
        if e.get("etype") != "scheduled_fact":
            continue
        p = e.get("payload") or {}
        if not p.get("outcome_entailing") or p.get("entailed_direction") not in ("yes", "no"):
            continue
        if not (plan.as_of <= float(e.get("ts", 0.0)) <= deadline_ts):
            continue
        (absorbing_facts if p["entailed_direction"] == absorb_dir else opposite_facts).append(e)
    fact_floor = 1.0
    for e in absorbing_facts:
        fact_floor *= (1.0 - max(0.0, min(0.999, float((e.get("payload") or {}).get("confidence", 0.6)))))
    fact_floor = round(1.0 - fact_floor, 4)                    # combined mass of the absorbing facts

    # ---- absorbing institution: a declared decision procedure that decides THIS outcome. Only an
    #      institution that can decide INSIDE the deadline suppresses the residual chain; a vote scheduled
    #      after the deadline still absorbs (it shapes the timing curve) but leaves the residual paths ----
    inst_events = [e for e in plan.scheduled_events
                   if e.get("etype") == "institutional_decision"
                   and str((e.get("payload") or {}).get("outcome_var", "")) == resolve_var]
    inst_in_window = [e for e in inst_events if float(e.get("ts", 0.0)) <= deadline_ts]

    # ---- consumed state moves onto the residual chain (facts that absorb leave the channel; survival
    #      polarity inverts the channel: pro-YES state suppresses the state-breaking hazard) ----
    consumed = [dict(m) for m in (getattr(plan, "_consumed_state", None) or []) if isinstance(m, dict)]
    if absorbing_facts:
        consumed = [m for m in consumed if m.get("var") != "fact_entailment"]
    # the endogenous channel is universal: declared pathway processes (moved by the simulated actors'
    # actions) drive the binary residual chain exactly as they drive when-question hazards
    declared_q = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    endo_channels = []
    for pw in sorted(getattr(plan, "_declared_pathways", None) or []):
        v = progress_var(pw)
        if v in declared_q and not any(m.get("var") == v for m in consumed):
            endo_channels.append(v)
            consumed.append({"var": v, "weight": 0.6})
    if polarity == "occurrence_resolves_no":
        consumed = [{**m, "invert": True} for m in consumed]

    # ---- residual chain geometry: rounds at the trajectory cadence, exponents shaped by the family curve
    curve, fam, curve_src = family_hazard_curve(question)
    from swm.world_model_v2.family_hazards import family_base_rate
    fbr, _fam2, fbr_src = family_base_rate(question)
    horizon_days = max(1.0, span / 86400.0)
    cad = max(0.5, float(cadence_days)) if cadence_days else max(1.0, horizon_days / 10.0)
    n_rounds = max(4, min(20, int(round(horizon_days / cad))))
    weights = _mass_weights_from_curve(curve)
    round_ts = [plan.as_of + (k / (n_rounds + 1)) * span for k in range(1, n_rounds + 1)]
    buckets = [min(int(((t - plan.as_of) / max(1.0, span)) * _BUCKETS), _BUCKETS - 1) for t in round_ts]
    per_bucket = {b: buckets.count(b) for b in set(buckets)}
    exponents = [weights[b] / per_bucket[b] for b in buckets]
    z = sum(exponents) or 1.0
    exponents = [x / z for x in exponents]                     # Σ = 1 ⇒ chain mass = target exactly

    calibration_base = {"absorb_from": ("rate" if absorb_dir == "yes" else "one_minus_rate"),
                        "fact_floor": fact_floor, "lean": lean,
                        "posterior_rate_particles": ([[float(r), float(w)] for r, w in
                                                      (getattr(plan, "posterior_rate_particles", None)
                                                       or [])] or None),
                        "fallback_rate": (float(fbr) if isinstance(fbr, (int, float)) else None),
                        "fallback_provenance": fbr_src}

    # ---------------- mutations (everything above computed first so a failure leaves the plan intact) ----
    n_fact_ev = 0
    for e in absorbing_facts:
        p = e.get("payload") or {}
        plan.scheduled_events.append({
            "etype": "hazard_round", "ts": max(plan.as_of + 1.0, float(e.get("ts", plan.as_of + 1.0))),
            "participants": [],
            "payload": {"mode": f"entailed_fact:{str(p.get('kind', 'fact'))[:24]}",
                        "success_prob": max(0.0, min(0.999, float(p.get("confidence", 0.6)))),
                        "as_of": plan.as_of, "span_s": span,
                        "fact": str(p.get("fact", ""))[:160], "source": p.get("source")}})
        n_fact_ev += 1
    for e in inst_events:
        pl = e.setdefault("payload", {})
        pl["absorbing"] = True
        pl["absorbing_mode"] = f"institutional:{str(pl.get('institution_id', ''))[:28]}"
    n_resid = 0
    if not inst_in_window:
        for ts, exp in zip(round_ts, exponents):
            plan.scheduled_events.append({
                "etype": "hazard_round", "ts": ts, "participants": [],
                "payload": {"mode": "resolution", "intention_factor": 1.0,
                            "as_of": plan.as_of, "span_s": span, "consume": consumed,
                            "calibration": {**calibration_base, "exponent": exp}}})
            n_resid += 1
    removed = [e for e in plan.scheduled_events
               if e.get("etype") in ("resolve_outcome", "aggregate_outcome_resolution")]
    plan.scheduled_events = [e for e in plan.scheduled_events
                             if e.get("etype") not in ("resolve_outcome", "aggregate_outcome_resolution")]
    _ensure_event_time_mechanisms(plan, curve_src if n_resid else "structural (facts/institutions absorb)")
    _declare_readout_quantities(plan)
    modes = ([f"entailed_fact:{str((e.get('payload') or {}).get('kind', 'fact'))[:24]}"
              for e in absorbing_facts]
             + [f"institutional:{str((e.get('payload') or {}).get('institution_id', ''))[:28]}"
                for e in inst_events]
             + (["resolution"] if n_resid else []))
    plan.outcome_contract = EventTimeContract(
        as_of=plan.as_of, horizon_ts=plan.horizon_ts,
        resolves_iff=str((criterion or {}).get("resolves_yes_iff", "")
                         or getattr(contract, "resolution_rule", ""))[:300],
        modes=modes, binary_options=options,
        occurrence_resolves=("yes" if absorb_dir == "yes" else "no"),
        deadline_ts=deadline_ts).validate()
    rep = {"contract": "binary_first_passage", "options": options,
           "event_polarity": polarity, "polarity_source": polarity_src,
           "deadline_ts": round(deadline_ts, 0),
           "n_absorbing_fact_events": n_fact_ev, "fact_floor": fact_floor,
           "n_opposite_direction_facts": len(opposite_facts),
           "absorbing_institutions": [str((e.get("payload") or {}).get("institution_id", ""))
                                      for e in inst_events],
           "n_residual_rounds": n_resid,
           "residual_skipped_reason": ("institutional decision is the resolution path"
                                       if inst_in_window else None),
           "consumed_state": consumed, "endogenous_channels": endo_channels,
           "family": fam, "hazard_curve_source": curve_src,
           "fallback_rate_provenance": fbr_src,
           "posterior_calibrated": bool(calibration_base["posterior_rate_particles"]),
           "n_resolver_events_removed": len(removed)}
    if lineage is not None:
        lineage["event_time"] = rep
    return rep
