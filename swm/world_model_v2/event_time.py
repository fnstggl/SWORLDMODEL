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
import re
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)
from swm.world_model_v2.mode_graph import (ENDOGENOUS_STANCE_SPLIT, LEGACY_LEVELS, PROGRESS_PREFIX,
                                           canon_level, canonical_modes, combine_stances,
                                           mode_pathway, pathway_of, progress_var)

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


def _hr_table(pathway: str = "") -> dict:
    """Effect-size table: fitted pack when present (pathway-STRATIFIED estimates overlay the pooled
    ones when the pack carries them — a refusal's effect on a deal differs from its effect on a
    bill), else the documented priors."""
    if INTENTION_HR_PACK.exists():
        try:
            pack = json.loads(INTENTION_HR_PACK.read_text())
            base = {k: tuple(v) for k, v in (pack.get("hazard_ratios") or {}).items()}
            strat = ((pack.get("hazard_ratios_by_pathway") or {}).get(str(pathway)) or {})
            base.update({k: tuple(v) for k, v in strat.items()})
            return base or dict(INTENTION_HR_PRIORS)
        except Exception:  # noqa: BLE001
            return dict(INTENTION_HR_PRIORS)
    return dict(INTENTION_HR_PRIORS)


def hr_pack_info() -> dict:
    """Staleness/provenance surfacing: which effect-size source is serving, and how old it is —
    stamped into every event-time conversion report."""
    if INTENTION_HR_PACK.exists():
        try:
            p = json.loads(INTENTION_HR_PACK.read_text())
            return {"source": "fitted_pack", "fitted_at": p.get("fitted_at"),
                    "n_rows": p.get("n_rows"),
                    "stratified": bool(p.get("hazard_ratios_by_pathway"))}
        except Exception:  # noqa: BLE001
            pass
    return {"source": "documented_priors_unfitted", "fitted_at": None, "n_rows": None,
            "stratified": False}


def mode_hazard_ratio(stances: list, pathway: str = "cooperative_agreement", *,
                      mode: dict = None) -> dict:
    """One hazard-ratio DISTRIBUTION for a mode, combined from the grounded stances by the mode's
    DECISION STRUCTURE (mode_graph.combine_stances) — unanimity/veto, majority, hierarchy, unilateral,
    weakest-link, cumulative pressure, or aggregation. "Most-opposed binds" is the unanimity case,
    not a universal law. Effect sizes come from the fitted pack when it exists (pathway-stratified
    when fitted), else the documented priors; reliability/capability/graded-control shrinks happen
    inside the combiner."""
    return combine_stances(stances, pathway, mode=mode, hr_table=_hr_table(pathway))


def agreement_hazard_ratio(stances: list) -> dict:
    """Back-compat wrapper: the cooperative-pathway (unanimity) case of mode_hazard_ratio."""
    return mode_hazard_ratio(stances, "cooperative_agreement")


def fit_intention_hazard_ratios(rows: list, *, pool_strength: float = 8.0) -> dict:
    """Fit statement-class hazard ratios from RESOLVED historical cases. Each row:
    {commitment_level, hazard_ratio, pathway?} where hazard_ratio = (resolving-state hazard after
    the statement) / (hazard before), measured (placebo-controlled) on archived paths of resolved
    cases. Partial pooling toward 1.0 (no effect); rows carrying a pathway additionally produce
    PATHWAY-STRATIFIED estimates pooled toward the unstratified level estimate (a refusal moves a
    deal differently than a bill). Writes nothing — caller persists to INTENTION_HR_PACK. Until
    such labeled data exists the documented INTENTION_HR_PRIORS above serve, reported as priors."""
    import statistics

    def _fit(logs, k_pool, center):
        k = k_pool / (len(logs) + k_pool)
        med = math.exp((1 - k) * statistics.median(logs) + k * center)
        sd = statistics.pstdev(logs) if len(logs) > 1 else 0.35
        return (round(med, 4), round(med * math.exp(-_Z80 * sd), 4),
                round(med * math.exp(_Z80 * sd), 4))
    by, by_pw = {}, {}
    for r in rows:
        lvl = canon_level(r.get("commitment_level"))
        hr = r.get("hazard_ratio")
        if lvl and isinstance(hr, (int, float)) and hr > 0:
            by.setdefault(lvl, []).append(math.log(float(hr)))
            pw = str(r.get("pathway") or "").strip().lower()
            if pw:
                by_pw.setdefault(pw, {}).setdefault(lvl, []).append(math.log(float(hr)))
    out = {lvl: _fit(logs, pool_strength, 0.0) for lvl, logs in by.items()}
    out_pw = {}
    for pw, levels in by_pw.items():
        for lvl, logs in levels.items():
            if lvl in out:                                    # pool strata toward the pooled estimate
                out_pw.setdefault(pw, {})[lvl] = _fit(logs, pool_strength,
                                                      math.log(max(1e-6, out[lvl][0])))
    return {"version": "intention-hr-1.1", "fit_on": "resolved historical statement/outcome pairs",
            "n_rows": len(rows), "hazard_ratios": {k: list(v) for k, v in out.items()},
            "hazard_ratios_by_pathway": {pw: {k: list(v) for k, v in lv.items()}
                                         for pw, lv in out_pw.items()}}


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
                 deadline_ts: float = None, categorical_options: list = None,
                 mode_option_map: dict = None):
        self.as_of, self.horizon_ts = float(as_of), float(horizon_ts)
        self.resolution_rule = resolves_iff[:300]
        self.modes = list(modes or [])
        self.readout_var = readout_var
        self.binary_options = [str(o) for o in (binary_options or [])][:2]
        # categorical unification: the question's own >2 options project as the absorbed_by
        # marginal (mode_option_map: canonical mode id → original option label), with the honest
        # residual "none_of_the_options_by_horizon" mass — no option is ever force-picked in a
        # world that reached none of them
        self.categorical_options = [str(o) for o in (categorical_options or [])]
        self.mode_option_map = dict(mode_option_map or {})
        self.occurrence_resolves = "no" if str(occurrence_resolves) == "no" else "yes"
        self.deadline_ts = float(deadline_ts) if deadline_ts else float(horizon_ts)
        if len(self.binary_options) == 2:
            self.options = list(self.binary_options)
        elif self.categorical_options:
            self.options = self.categorical_options + ["none_of_the_options_by_horizon"]
        else:
            self.options = ["absorbed_by_horizon", "censored_beyond_horizon"]

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
        elif self.categorical_options:
            # categorical readout: WHICH mode produced absorption, mapped back to the question's own
            # option labels; worlds that reached none of them are honest residual mass, never a draw
            dist = {o: 0.0 for o in self.categorical_options}
            unmapped = 0.0
            for mid, wsum in modes.items():
                label = self.mode_option_map.get(mid)
                if label is None and mid in dist:
                    label = mid
                if label in dist:
                    dist[label] = round(dist[label] + wsum / n, 4)
                else:
                    unmapped += wsum / n
            dist["none_of_the_options_by_horizon"] = round(1 - p_absorbed + unmapped, 4)
            et["unmapped_absorbed_share"] = round(unmapped, 4)
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
        prov = world.quantities.get("provisional_absorbing_mode")
        return event.etype == "hazard_round" and getattr(q, "value", None) in (None, 0) \
            and not getattr(flag, "value", None) \
            and not getattr(prov, "value", None)               # first passage; and while a provisional
                                                               # end-state holds, the world IS in that
                                                               # candidate state — hazards pause until
                                                               # it confirms or collapses

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=[f"mode={event.payload.get('mode', '?')}"])

    def _sampled_hr(self, world, mode, hr, *, salt=""):
        """One hazard-ratio draw PER BRANCH per mode (lognormal from the prior's 80% interval),
        persisted on the world so every round in this branch sees the same sampled effect size —
        the uncertainty over the coefficient becomes cross-particle spread in the terminal CDF.
        `salt` carries the stance-state hash: when stance dynamics rewrite the records, the draw
        refreshes (a NEW effect size for the NEW stance state)."""
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import _branch_rng
        qname = f"sampled_intention_hr:{mode}" + (f":{salt}" if salt else "")
        q = world.quantities.get(qname)
        if q is not None and isinstance(getattr(q, "value", None), (int, float)):
            return float(q.value)
        med = max(1e-6, float(hr.get("median", 1.0)))
        lo, hi = float(hr.get("lo80", med)), float(hr.get("hi80", med))
        sigma = (math.log(max(hi, 1e-6)) - math.log(max(lo, 1e-6))) / (2 * _Z80) if hi > lo else 0.0
        rng = _branch_rng(world, f"hr:{mode}:{salt}")
        val = med * math.exp(sigma * rng.gauss(0.0, 1.0)) if sigma > 0 else med
        val = max(0.05, min(3.0, val))
        register_quantity_type("sampled_intention_hr", units="hazard_ratio")
        world.quantities[qname] = Quantity(name=qname, qtype="sampled_intention_hr", value=val,
                                           timestamp=world.clock.now)
        return val

    def _resolve_hr(self, world, a) -> tuple:
        """The mode's stance hazard ratio for THIS round: re-derived from the CURRENT entity stance
        records when they carry a mode_def and the stance state has changed since conversion (the
        stance-review operator rewrites records mid-trajectory), else the baked distribution. The
        endogenous split is applied at RUNTIME as a per-branch sampled exponent. Returns
        (sampled_ratio | None, provenance dict)."""
        if not isinstance(a.get("hr"), dict):
            return None, {}
        hr, prov = dict(a["hr"]), {"hr_source": "baked"}
        salt = ""
        if isinstance(a.get("mode_def"), dict):
            from swm.world_model_v2.world_dynamics import live_capacity, live_stances, stance_state_hash
            stances = live_stances(world)
            h_now = stance_state_hash(stances)
            if stances and h_now != str(a.get("stances_hash", "")):
                live = combine_stances(stances, str(a.get("pathway", "")),
                                       mode=a["mode_def"], hr_table=_hr_table(),
                                       live_capacity=live_capacity(world))
                if live.get("median") is not None:
                    hr = {k: live[k] for k in ("median", "lo80", "hi80")}
                    prov = {"hr_source": "live_recomputed", "binding_actor": live.get("binding_actor")}
                    salt = h_now
        if a.get("endogenous_live"):
            from swm.world_model_v2.world_dynamics import sampled_coupling
            s = sampled_coupling(world, "endogenous_stance_split")
            hr = {k: math.exp(s * math.log(max(1e-6, float(hr.get(k, 1.0)))))
                  for k in ("median", "lo80", "hi80")}
            prov["endogenous_split"] = round(s, 4)
        val = self._sampled_hr(world, str(a.get("mode")), hr, salt=salt)
        return val, prov

    @staticmethod
    def _consume_state_hazard(world, h, consume):
        """RELATIVE consumption for hazards: consumed causal state acts as a bounded MULTIPLICATIVE
        modifier centered at no-effect (state 0.5 → ×1). An entry may set `invert: true`
        (survival-polarity chains: pro-YES state must SUPPRESS the state-breaking hazard, v ↦ 1−v).
        An entry may name a `coupling` (own_pathway_weight / cross_pathway_weight /
        world_state_weight): its weight is then the PER-BRANCH SAMPLED coupling constant —
        structural-coefficient uncertainty propagates into the CDF instead of being a point choice.
        Weight 0.45 at an extreme state moves the hazard ×2^0.45 ≈ 1.37 (or ÷); a weight-1.0
        pathway-process channel reaches ×2 — deliberately bounded; total factor clamped to [0.25, 4]."""
        used, logf = [], 0.0
        for m in (consume or []):
            var, w = str(m.get("var", "")), float(m.get("weight", 0.0) or 0.0)
            if m.get("coupling"):
                from swm.world_model_v2.world_dynamics import sampled_coupling
                w = sampled_coupling(world, str(m["coupling"]))
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
        hr_prov = {}
        if a.get("success_prob") is not None:                 # dated entailing fact — fires at ITS date
            h = max(0.0, min(0.999, float(a["success_prob"])))
            unc["rate_source"] = "entailed_fact_confidence"
        elif isinstance(a.get("calibration"), dict):          # posterior-calibrated residual chain
            t, src = self._calibrated_target(world, a["calibration"])
            exp = max(0.0, float(a["calibration"].get("exponent", 0.0) or 0.0))
            h = 1.0 - (1.0 - t) ** exp
            if isinstance(a.get("hr"), dict):                 # distributional stance ratio (sampled)
                hr_used, hr_prov = self._resolve_hr(world, a)
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
                hr_used, hr_prov = self._resolve_hr(world, a)
                h *= hr_used
            else:                                             # legacy point factor
                h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
            h, used, sfac = self._consume_state_hazard(world, h, a.get("consume") or [])
            h = max(0.0, min(0.95, h))
        unc.update({"hazard": round(h, 4), "consumed": used,
                    "state_hazard_factor": round(sfac, 4),
                    "sampled_hazard_ratio": (round(hr_used, 4) if hr_used is not None else None),
                    "intention_factor": a.get("intention_factor"), **hr_prov})
        rng2 = _branch_rng(world, f"hz:{a.get('mode')}:{world.clock.now}")
        d = StateDelta(at=world.clock.now, event_type="hazard_round", operator=self.name,
                       reason_codes=proposal.reason_codes + [f"h={round(h, 4)}"],
                       uncertainty=unc)
        if rng2.random() < h:
            persistence_s = float(a.get("persistence_s", 0.0) or 0.0)
            if persistence_s > 0.0:
                # the criterion requires the end-state to HOLD: write a PROVISIONAL absorption and
                # schedule the persistence check — the near-miss ("temporary ceasefire that
                # collapses") is a real possible event, not a parser annotation
                register_quantity_type("provisional_absorbing_mode", units="mode")
                world.quantities["provisional_absorbing_mode"] = Quantity(
                    name="provisional_absorbing_mode", qtype="provisional_absorbing_mode",
                    value=str(a.get("mode", "unspecified")), timestamp=world.clock.now)
                d.change("quantities[provisional_absorbing_mode]", None, str(a.get("mode")))
                d.reason_codes.append("provisional_pending_persistence")
                d.follow_up_events.append({
                    "etype": "persistence_check", "ts": world.clock.now + persistence_s,
                    "participants": [],
                    "payload": {"mode": str(a.get("mode")), "pathway": str(a.get("pathway", ""))}})
            else:
                register_quantity_type("absorbing_state_reached", units="bool")
                register_quantity_type("absorbing_mode", units="mode")
                world.quantities["absorbing_state_reached"] = Quantity(
                    name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                    timestamp=world.clock.now)
                world.quantities["absorbing_mode"] = Quantity(
                    name="absorbing_mode", qtype="absorbing_mode",
                    value=str(a.get("mode", "unspecified")), timestamp=world.clock.now)
                d.change("quantities[absorbing_state_reached]", None, True)
        return d


register_operator("absorption_monitor", AbsorptionMonitorOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="pure observation of the absorbing predicate (first passage)",
                  validated=True)
register_operator("hazard_round", HazardRoundOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="dated outcome-entailing facts (success_prob at their real "
                                   "dates); legacy grid parameterizations survive only in "
                                   "explicit ablations", validated=True)


# ---------------------------------------------------------------- 3b. continuous-time first passage
# (§15–§16): the production replacement for evenly spaced hazard-round grids. Each mode (or the
# binary residual) is ONE process: per-branch persistent Exp(1) threshold (particle-rooted →
# matched across counterfactual arms), piecewise cumulative intensity from the fitted family
# curve, live modulation = sampled stance hazard ratio × consumed causal state. State changes
# preserve accumulated hazard + threshold and only re-project the crossing.

def _fp_sampled_hr(world, spec) -> float:
    """The branch's persistent sampled stance hazard ratio for a mode process — re-derived from
    LIVE stances when the stance state has changed (salt = live stance hash), endogenous-split
    when the behavioral channel is live. Particle-rooted stream: matched arms share the draw."""
    from swm.world_model_v2.temporal_model import particle_rng
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    hr = dict(spec.get("hr") or {})
    if not hr or hr.get("median") is None:
        return 1.0
    salt = ""
    if isinstance(spec.get("mode_def"), dict):
        from swm.world_model_v2.world_dynamics import live_capacity, live_stances, stance_state_hash
        stances = live_stances(world)
        h_now = stance_state_hash(stances)
        if stances and h_now != str(spec.get("stances_hash", "")):
            live = combine_stances(stances, str(spec.get("pathway", "")),
                                   mode=spec["mode_def"], hr_table=_hr_table(),
                                   live_capacity=live_capacity(world))
            if live.get("median") is not None:
                hr = {k: live[k] for k in ("median", "lo80", "hi80")}
                salt = h_now
    if spec.get("endogenous_live"):
        from swm.world_model_v2.world_dynamics import sampled_coupling
        s = sampled_coupling(world, "endogenous_stance_split")
        hr = {k: math.exp(s * math.log(max(1e-6, float(hr.get(k, 1.0)))))
              for k in ("median", "lo80", "hi80")}
    qname = f"sampled_intention_hr:{spec.get('mode')}" + (f":{salt}" if salt else "")
    q = world.quantities.get(qname)
    if q is not None and isinstance(getattr(q, "value", None), (int, float)):
        return float(q.value)
    med = max(1e-6, float(hr.get("median", 1.0)))
    lo, hi = float(hr.get("lo80", med)), float(hr.get("hi80", med))
    sigma = (math.log(max(hi, 1e-6)) - math.log(max(lo, 1e-6))) / (2 * _Z80) if hi > lo else 0.0
    rng = particle_rng(world, f"fp_hr:{spec.get('mode')}:{salt}")
    val = med * math.exp(sigma * rng.gauss(0.0, 1.0)) if sigma > 0 else med
    val = max(0.05, min(3.0, val))
    register_quantity_type("sampled_intention_hr", units="hazard_ratio")
    world.quantities[qname] = Quantity(name=qname, qtype="sampled_intention_hr", value=val,
                                       timestamp=world.clock.now)
    return val


def _fp_consume_factor(world, spec) -> float:
    """Bounded multiplicative state modulation — same relative semantics as
    HazardRoundOperator._consume_state_hazard (×2 per full weight at an extreme state,
    clamped [0.25, 4])."""
    logf, used = 0.0, 0
    for m in (spec.get("consume") or []):
        var, w = str(m.get("var", "")), float(m.get("weight", 0.0) or 0.0)
        if m.get("coupling"):
            from swm.world_model_v2.world_dynamics import sampled_coupling
            w = sampled_coupling(world, str(m["coupling"]))
        q = world.quantities.get(var)
        v = getattr(q, "value", None)
        if w <= 0.0 or not isinstance(v, (int, float)):
            continue
        v = max(0.0, min(1.0, float(v)))
        if m.get("invert"):
            v = 1.0 - v
        logf += w * (v - 0.5) * 2.0 * math.log(2.0)
        used += 1
    return max(0.25, min(4.0, math.exp(logf))) if used else 1.0


def _fp_modulation(world, st) -> float:
    spec = st.payload.get("spec") or {}
    if spec.get("kind") == "calibrated_resolution":
        base = max(0.0, min(2.0, float(spec.get("intention_factor", 1.0) or 1.0)))
        return base * _fp_consume_factor(world, spec)
    return _fp_sampled_hr(world, spec) * _fp_consume_factor(world, spec)


def _fp_target_mass(world, cal: dict) -> tuple:
    """Per-particle calibrated target absorbed-mass — the same posterior→fallback→lean ladder
    as the legacy calibrated chains, drawn from the PARTICLE stream so matched arms share the
    target (§23)."""
    from swm.world_model_v2.fallback import LEAN_BETA, _apply_lean_shift, _beta_sample
    from swm.world_model_v2.phase_consumers import _draw_rate
    from swm.world_model_v2.temporal_model import particle_rng
    rng = particle_rng(world, "event_time_target")
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
    if lean_h:
        r = _apply_lean_shift(max(0.0, min(1.0, r)), str(lean_h))
    r = max(0.0, min(1.0, r))
    t = r if str(cal.get("absorb_from", "rate")) == "rate" else 1.0 - r
    c = max(0.0, min(0.999, float(cal.get("fact_floor", 0.0) or 0.0)))
    if c > 0.0:
        t = max(0.0, (t - c) / (1.0 - c))
    return max(0.0, min(0.98, t)), src


def ensure_first_passage_state(world, spec: dict):
    """Get-or-create the branch's CumulativeHazardState for one plan-level process spec, with
    per-particle base rates (mode share × family curve; or the calibrated target's cumulative
    intensity) and live initial modulation."""
    from swm.world_model_v2.temporal_hazards import (ensure_hazard_state,
                                                     rates_from_target_mass)
    pid = str(spec.get("process_id"))
    store = getattr(world, "temporal_hazards", None) or {}
    if pid in store:
        return store[pid]
    as_of = float(spec.get("as_of", world.clock.now))
    span_s = max(1.0, float(spec.get("span_s", 1.0)))
    curve = spec.get("curve")
    if spec.get("kind") == "calibrated_resolution":
        target, src = _fp_target_mass(world, spec.get("calibration") or {})
        base_rates = rates_from_target_mass(target, span_s=span_s, curve=curve)
        extra = {"target_mass": round(target, 4), "rate_source": src}
    else:
        share = max(0.0, min(1.0, float(spec.get("share", 1.0) or 1.0)))
        bucket_days = (span_s / _BUCKETS) / 86400.0
        if curve:
            base_rates = [share * (-math.log(1.0 - max(0.0, min(0.999999, float(h)))))
                          / max(1e-9, bucket_days) for h in curve]
        else:
            lam_total = 0.5 * share                       # legacy no-curve prior mass, exact
            base_rates = [lam_total / _BUCKETS / max(1e-9, bucket_days)] * _BUCKETS
        extra = {"share": share}
    st = ensure_hazard_state(world, pid, as_of=as_of, horizon_ts=as_of + span_s,
                             base_rates=base_rates, reads=spec.get("reads") or (),
                             payload={"etype": "first_passage", "spec": dict(spec),
                                      "mode": spec.get("mode"), **extra,
                                      "modulation_hook": "event_time_mode"})
    st.modulation = _fp_modulation(world, st)
    return st


class FirstPassageOperator(TransitionOperator):
    """Executes a first-passage crossing: the process's cumulative hazard reached its branch
    threshold at THIS real timestamp. Persistence-window criteria write a PROVISIONAL
    absorption + schedule the persistence check; otherwise the absorbing state is written
    directly. On a later persistence COLLAPSE the process RESUMES with a fresh threshold
    segment (memoryless continuation — accumulated exposure preserved, invariant 26)."""
    name = "first_passage"

    def applicable(self, world, event):
        q = world.quantities.get("absorbed_at")
        flag = world.quantities.get("absorbing_state_reached")
        prov = world.quantities.get("provisional_absorbing_mode")
        return event.etype == "first_passage" and getattr(q, "value", None) in (None, 0) \
            and not getattr(flag, "value", None) and not getattr(prov, "value", None)

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=[f"mode={event.payload.get('mode', '?')}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        pid = str(a.get("hazard_process_id", ""))
        st = (getattr(world, "temporal_hazards", None) or {}).get(pid)
        spec = (st.payload.get("spec") if st is not None else None) or a.get("spec") or {}
        if st is not None:
            st.accumulate_to(world.clock.now)
            st.fired = True
        d = StateDelta(at=world.clock.now, event_type="first_passage", operator=self.name,
                       reason_codes=[f"mode={spec.get('mode', a.get('mode', '?'))}",
                                     "threshold_crossed"],
                       uncertainty={"process_id": pid,
                                    "accumulated_hazard": (round(st.accumulated, 4)
                                                           if st is not None else None),
                                    "threshold": (round(st.threshold, 4)
                                                  if st is not None else None),
                                    "n_reprojections": (st.n_reprojections
                                                        if st is not None else None)})
        persistence_s = float(spec.get("persistence_s", 0.0) or 0.0)
        mode = str(spec.get("mode", a.get("mode", "unspecified")))
        if persistence_s > 0.0:
            register_quantity_type("provisional_absorbing_mode", units="mode")
            world.quantities["provisional_absorbing_mode"] = Quantity(
                name="provisional_absorbing_mode", qtype="provisional_absorbing_mode",
                value=mode, timestamp=world.clock.now)
            d.change("quantities[provisional_absorbing_mode]", None, mode)
            d.reason_codes.append("provisional_pending_persistence")
            d.follow_up_events.append({
                "etype": "persistence_check", "ts": world.clock.now + persistence_s,
                "participants": [],
                "payload": {"mode": mode, "pathway": str(spec.get("pathway", "")),
                            "first_passage_process_id": pid}})
        else:
            register_quantity_type("absorbing_state_reached", units="bool")
            register_quantity_type("absorbing_mode", units="mode")
            world.quantities["absorbing_state_reached"] = Quantity(
                name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                timestamp=world.clock.now)
            world.quantities["absorbing_mode"] = Quantity(
                name="absorbing_mode", qtype="absorbing_mode", value=mode,
                timestamp=world.clock.now)
            d.change("quantities[absorbing_state_reached]", None, True)
        return d


def resume_first_passage_after_collapse(world, process_id: str) -> tuple:
    """A provisional end-state COLLAPSED: the mode's process resumes with a fresh threshold
    segment above the exposure already accumulated (memoryless continuation; particle-rooted
    draw, deterministic per collapse count). Returns (state, next_crossing_ts | None)."""
    from swm.world_model_v2.temporal_model import particle_rng
    st = (getattr(world, "temporal_hazards", None) or {}).get(str(process_id))
    if st is None:
        return None, None
    st.accumulate_to(world.clock.now)
    n_collapses = int(st.payload.get("n_collapses", 0)) + 1
    st.payload["n_collapses"] = n_collapses
    rng = particle_rng(world, f"fp_resume:{process_id}:{n_collapses}")
    st.threshold = st.accumulated + rng.expovariate(1.0)
    st.fired = False
    st.generation += 1
    st.modulation = _fp_modulation(world, st)
    return st, st.project_crossing()


register_operator("first_passage", FirstPassageOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="continuous-time cumulative-hazard first passage: per-branch "
                                   "Exp(1) threshold (particle-matched), fitted family curve "
                                   "intensity × sampled stance hazard ratio × consumed causal "
                                   "state; state changes preserve accumulated hazard",
                  validated=True)

from swm.world_model_v2.temporal_hazards import register_modulation_hook  # noqa: E402
register_modulation_hook("event_time_mode", _fp_modulation)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("hazard_round", "absorption"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("quantities",), deltas=("quantities",),
                            parameter_source="event-time architecture", validated=True)

# the world-dynamics layer (stance reviews, persistence checks, sampled couplings) is PART OF the
# event-time machinery: importing event_time must register its operators and event types, or the
# scheduled stance_review/persistence_check events would be dead weight in any process that did not
# import the runtime (the offline demo caught exactly this)
import swm.world_model_v2.world_dynamics  # noqa: E402,F401


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
    """Accept the timing machinery on the plan. ORDER MATTERS: absorbing writers (first_passage
    crossings; dated-fact hazard_round events) must come BEFORE the absorption monitor in the
    operator list so the monitor observes a same-event write and stamps first passage at the
    write's own clock time, not one event late."""
    for mech_id, op in (("first_passage_processes", "first_passage"),
                        ("hazard_rounds", "hazard_round"),
                        ("absorption_monitor", "absorption_monitor")):
        if not any(x.get("operator") == op for x in plan.accepted_mechanisms if isinstance(x, dict)):
            plan.accepted_mechanisms.append({
                "mech_id": mech_id, "ontology_type": "event_time", "operator": op,
                "causal_role": ("continuous-time first-passage scheduling"
                                if op == "first_passage" else "first-passage timing machinery"),
                "parameter_source": curve_src if op != "absorption_monitor" else "observation",
                "temporal_scale": "event", "calibration_status": curve_src, "sensitivity": 1.0})


def _declare_readout_quantities(plan):
    """Declare the readout quantities on the plan so build_world registers them and the contract's
    readout binds at materialization (value None until first passage stamps it)."""
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    for name in ("absorbed_at", "absorbing_state_reached"):
        if name not in declared:
            plan.quantities.append({"name": name, "qtype": name, "value": None, "sd": None})


_PERSISTENCE_RX = re.compile(r"(?:>=|at least|for(?: a minimum of)?)\s*(\d{1,3})\s*(?:consecutive\s*)?"
                             r"(day|week|month)s?", re.IGNORECASE)
_PERSIST_UNIT_S = {"day": 86400.0, "week": 7 * 86400.0, "month": 30 * 86400.0}


def criterion_persistence_s(criterion: dict) -> float:
    """A resolution criterion that requires the end-state to HOLD ("no active hostilities for >=30
    consecutive days") makes near-misses real: parse the persistence window. Explicit
    `persistence_days` from the parser wins; else the phrase pattern in resolves_yes_iff. 0 = none."""
    pd = (criterion or {}).get("persistence_days")
    if isinstance(pd, (int, float)) and pd > 0:
        return float(pd) * 86400.0
    m = _PERSISTENCE_RX.search(str((criterion or {}).get("resolves_yes_iff", "")))
    if m:
        return float(m.group(1)) * _PERSIST_UNIT_S[m.group(2).lower()]
    return 0.0


def _endogenous_consume(plan, mode, base_consumed: list) -> tuple:
    """The mode's state-consumption channels — the ENDOGENOUS half of the hazard clock:
      * its OWN process quantity (sampled own_pathway_weight): the per-mode channel on CONTESTED
        (non-shared) pathways — russian_victory and ukrainian_victory evolve separately — else the
        shared pathway process; written by the simulated actors' actions, institutional stage
        reviews and world-driven consumers;
      * every OTHER declared pathway process at the sampled cross weight: resolution pressure
        spills over — a collapsing battlefield forces parties to the table ("negotiations begin
        because battlefield state changed"), advancing institutional stages raise unilateral urgency;
      * for WORLD-DRIVEN pathways (threshold/diffusion/market/physical/…): the plan's declared
        nonlinear state and population aggregates, when present — non-actor mechanisms drive those
        hazards; stances barely touch them (mode_graph handles that on the HR side).
    Channel weights carry a `coupling` name — the operator resolves them to PER-BRANCH SAMPLED
    coupling constants, so the structural coefficients are distributions, not point choices.
    Returns (consume list, endogenous_channel_live)."""
    consume = [dict(m) for m in (base_consumed or []) if isinstance(m, dict)]
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    pw = _mode_pathway(mode)
    live = False
    own_mode_var = f"mode_progress:{pw}:{mode.get('id')}" if isinstance(mode, dict) else ""
    if own_mode_var in declared:                              # contested pathway: the mode's OWN channel
        consume.append({"var": own_mode_var, "weight": 1.0, "coupling": "own_pathway_weight"})
        live = True
        if progress_var(pw) in declared:                      # the pathway aggregate stays as spillover
            consume.append({"var": progress_var(pw), "weight": 0.25,
                            "coupling": "cross_pathway_weight"})
    elif progress_var(pw) in declared:
        consume.append({"var": progress_var(pw), "weight": 1.0, "coupling": "own_pathway_weight"})
        live = True
    for other_pw in sorted({p for p in (getattr(plan, "_declared_pathways", None) or [])
                            if p != pw}):
        v = progress_var(other_pw)
        if v in declared:
            consume.append({"var": v, "weight": 0.25, "coupling": "cross_pathway_weight"})
    if not pathway_of(pw).actor_driven:
        for q in (getattr(plan, "quantities", []) or []):
            name = str(q.get("name", "")) if isinstance(q, dict) else ""
            if name.startswith("population_aggregate:") or name == "nonlinear_state":
                if not any(c.get("var") == name for c in consume):
                    consume.append({"var": name, "weight": 0.35, "coupling": "world_state_weight"})
    return consume, live


def declare_contested_mode_channels(plan, modes: list) -> dict:
    """On CONTESTED (non-shared) pathways each mode gets its own process channel
    `mode_progress:<pathway>:<mode_id>` — the two sides' campaigns evolve separately (universal:
    rival victories, rival candidates, competing products). Initialized from the pathway grounding.
    Shared pathways keep the one shared process (talks ARE one process). Also records the shared
    pathways' PRINCIPALS (decision-structure approvers) as a world quantity so execution can weight
    principals' moves above bystanders'. Active only when the plan's endogenous process layer is
    (declared pathways exist) — a bare plan without stance/process grounding stays minimal.
    Idempotent."""
    if not getattr(plan, "_declared_pathways", None):
        return {"mode_channels": {}, "principals": {}, "skipped": "no declared pathway processes"}
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    init_by_pw = {}
    for q in plan.quantities:
        if isinstance(q, dict) and str(q.get("name", "")).startswith(PROGRESS_PREFIX):
            init_by_pw[str(q["name"])[len(PROGRESS_PREFIX):]] = q.get("value", 0.5)
    added, principals = {}, {}
    for m in (modes or []):
        pw = _mode_pathway(m)
        if not pathway_of(pw).shared_process and pathway_of(pw).actor_driven:
            var = f"mode_progress:{pw}:{m.get('id')}"
            if var not in declared:
                plan.quantities.append({"name": var, "qtype": "mode_progress",
                                        "value": round(float(init_by_pw.get(pw, 0.5)), 3),
                                        "sd": 0.15})
                declared.add(var)
                added[str(m.get('id'))] = var
        ds = (m or {}).get("decision_structure") or {}
        for a in (ds.get("approvers") or []):
            principals.setdefault(pw, set()).add(str(a))
    for pw, names in principals.items():
        var = f"pathway_principals:{pw}"
        if var not in declared:
            plan.quantities.append({"name": var, "qtype": "pathway_principals",
                                    "value": "|".join(sorted(names)), "sd": None})
    return {"mode_channels": added,
            "principals": {k: sorted(v) for k, v in principals.items()}}


def convert_to_event_time(plan, criterion: dict, *, lineage: dict = None, llm=None,
                          categorical_options: list = None) -> dict:
    """Replace the compiled point contract with an EventTimeContract and schedule the timing machinery:
    per-mode hazard_round chains at the trajectory cadence, stance-review rounds (stance DYNAMICS),
    the absorption monitor, the persistence checker (near-miss semantics), and per-mode stance
    hazard-ratio DISTRIBUTIONS combined by each mode's decision structure. With
    `categorical_options` the SAME machinery answers a categorical question: the distribution over
    the question's own options is the absorbed_by marginal (plus honest none-by-horizon mass).
    Universal: built only from the plan's own structure + the canonical mode graph."""
    # canonical mode set: compiler hypotheses + K-pass elicitation reconciled by mode_graph (compile-
    # variance fix). If unified_runtime already computed it (before intention grounding, so stances
    # could be mode-scoped), reuse it — no second elicitation.
    modes = list(getattr(plan, "_canonical_modes", None) or [])
    consensus = dict(getattr(plan, "_mode_consensus", None) or {})
    if not modes:
        modes, consensus = canonical_modes(
            question=getattr(plan, "question", ""), criterion=criterion,
            hypotheses=list(getattr(plan, "structural_hypotheses", []) or []),
            options=list(categorical_options
                         or getattr(plan.outcome_contract, "options", []) or []), llm=llm)
    modes, rejected_modes = _filter_absorbing_modes(modes, criterion, llm)
    z = sum(m["prior"] for m in modes) or 1.0
    curve, fam, curve_src = family_hazard_curve(getattr(plan, "question", ""))
    # contested pathways get PER-MODE process channels (rival campaigns evolve separately) and the
    # shared pathways record their PRINCIPALS for execution-time weighting
    contested_rep = declare_contested_mode_channels(plan, modes)
    # a criterion that requires the end-state to HOLD makes near-misses real trajectory events
    persistence_s = criterion_persistence_s(criterion)
    # grounded-stance effect, MODE-SCOPED and structure-combined: each mode's hazard-ratio
    # DISTRIBUTION is combined from the stances concerning THAT mode (or its pathway) under the
    # mode's decision structure — never a point coefficient invented by the LLM, and never a
    # universal "most-opposed binds" shortcut.
    stances = list(getattr(plan, "_intention_stances", []) or [])
    from swm.world_model_v2.world_dynamics import coupling_pack_info, stance_state_hash
    baked_hash = stance_state_hash(stances)

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
    base_consumed = list(getattr(plan, "_consumed_state", []) or [])
    n_ev = 0
    mode_hr = {}
    if not hasattr(plan, "first_passage_processes") or plan.first_passage_processes is None:
        plan.first_passage_processes = []
    for m in modes:                     # ALL canonical modes — no fixed mode cap (§7 analogue)
        share = m["prior"] / z
        # each mode consumes the stance hazard ratio of ITS OWN causal pathway under ITS decision
        # structure, and the endogenous state channels of its pathway processes
        pathway = _mode_pathway(m)
        hr_m = _pathway_hr(pathway, mode=m)
        consume_m, endo_live = _endogenous_consume(plan, m, base_consumed)
        # ANTI-DOUBLE-COUNT: when the behavioral channel is live (stances condition the Phase-4
        # policies whose actions move the consumed process), the DIRECT multiplier keeps only the
        # residual share. The split exponent is SAMPLED PER BRANCH at runtime
        # (coupling `endogenous_stance_split`) — the payload carries the UNSPLIT distribution +
        # the endogenous_live flag; the report shows the prior-median-split for auditability.
        endo_split_applied = endo_live and hr_m.get("combination_rule") != "override"
        rep_hr = dict(hr_m)
        if endo_split_applied:
            s = ENDOGENOUS_STANCE_SPLIT
            rep_hr = dict(hr_m,
                          median=round(math.exp(s * math.log(max(1e-6, hr_m["median"]))), 4),
                          lo80=round(math.exp(s * math.log(max(1e-6, hr_m["lo80"]))), 4),
                          hi80=round(math.exp(s * math.log(max(1e-6, hr_m["hi80"]))), 4),
                          endogenous_split=s)
        mode_hr[m["id"]] = {k: rep_hr.get(k) for k in ("median", "lo80", "hi80", "binding_actor",
                                                       "binding_level", "combination_rule",
                                                       "endogenous_split")}
        mode_hr[m["id"]]["pathway"] = pathway
        mode_hr[m["id"]]["split_applied_at"] = ("runtime_sampled" if endo_split_applied else None)
        agreement = pathway == "cooperative_agreement"
        mode_def = {"id": m["id"], "pathway": pathway}
        if isinstance(m.get("decision_structure"), dict):
            mode_def["decision_structure"] = m["decision_structure"]
        # CONTINUOUS-TIME FIRST PASSAGE (§15): one process per mode — per-branch Exp(1)
        # threshold (particle-rooted: matched across counterfactual arms), cumulative intensity
        # from the fitted family curve × the mode's structural share, live modulation from the
        # sampled stance hazard ratio and consumed causal state. State changes preserve the
        # accumulated hazard and the threshold and only re-project the crossing (§16). No
        # evenly spaced grid; no visible integration rounds.
        reads = sorted({str(c.get("var")) for c in consume_m if isinstance(c, dict)
                        and c.get("var")})
        if stances:
            reads.append("stances")                       # stance rewrites re-derive the HR
        plan.first_passage_processes.append({
            "process_id": f"mode:{m['id']}", "kind": "mode_transition",
            "mode": m["id"], "pathway": pathway, "mode_def": mode_def, "share": share,
            "curve": list(curve) if curve else None,
            "hr": {k2: hr_m.get(k2) for k2 in ("median", "lo80", "hi80")},
            "endogenous_live": endo_split_applied, "stances_hash": baked_hash,
            "persistence_s": persistence_s, "requires_agreement": agreement,
            "as_of": plan.as_of, "span_s": span, "consume": consume_m, "reads": reads})
        n_ev += 1
    # STANCE DYNAMICS are EVENT-DRIVEN (§13): the temporal runtime emits
    # stance_relevant_change when a batch's writes cross a stance-rule threshold or move a
    # watched process var materially — no review grid, no review-count cooldowns. The
    # stance_dynamics mechanism below makes the operator live for those events.
    n_reviews = 0
    # a scheduled institutional decision becomes an ABSORBING WRITER for the institutional-pathway
    # mode (pass ⇒ absorbed at the vote's real date; fail ⇒ the world stays unabsorbed): the declared
    # procedure executes INSIDE the trajectory instead of writing a dead outcome variable no
    # event-time readout consumes
    inst_modes = [m for m in modes if _mode_pathway(m) == "institutional_procedure"]
    n_inst_absorbing = 0
    if inst_modes:
        for e in plan.scheduled_events:
            if e.get("etype") == "institutional_decision":
                pl = e.setdefault("payload", {})
                pl["absorbing"] = True
                pl.setdefault("absorbing_mode", inst_modes[0]["id"])
                n_inst_absorbing += 1
    _ensure_event_time_mechanisms(plan, curve_src)
    for mech_id, op, role in (
            ("stance_dynamics", "stance_review",
             "in-run stance updates (ripeness/winning/exhaustion/bandwagon) conditioning policies "
             "and hazard ratios"),
            ("persistence_semantics", "persistence_check",
             "near-miss realization: provisional end-states confirm or collapse")):
        if (op != "persistence_check" or persistence_s > 0) and (op != "stance_review" or stances):
            if not any(x.get("operator") == op for x in plan.accepted_mechanisms
                       if isinstance(x, dict)):
                mech = {"mech_id": mech_id, "ontology_type": "world_dynamics", "operator": op,
                        "causal_role": role, "parameter_source": "documented world-dynamics rules / "
                        "sampled coupling priors (fittable)", "temporal_scale": "scheduled",
                        "calibration_status": "documented_priors", "sensitivity": 0.8}
                # ORDER: the persistence check WRITES the absorbing state — like every absorbing
                # writer it must precede the monitor so first passage stamps at the confirmation's
                # own clock time, not one event late
                mon = next((i for i, x in enumerate(plan.accepted_mechanisms)
                            if isinstance(x, dict) and x.get("operator") == "absorption_monitor"),
                           None)
                if mon is not None:
                    plan.accepted_mechanisms.insert(mon, mech)
                else:
                    plan.accepted_mechanisms.append(mech)
    plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") != "resolve_outcome"]
    _declare_readout_quantities(plan)
    # categorical unification: the question's own options project as the absorbed_by marginal
    opt_map = None
    if categorical_options:
        from swm.world_model_v2.mode_graph import _canon_mode_id
        opt_map = {}
        for o in categorical_options:
            opt_map[_canon_mode_id(str(o))] = str(o)
    plan.outcome_contract = EventTimeContract(as_of=plan.as_of, horizon_ts=plan.horizon_ts,
                                              resolves_iff=str((criterion or {}).get("resolves_yes_iff",
                                                                                     ""))[:300],
                                              modes=[m["id"] for m in modes],
                                              categorical_options=(list(categorical_options)
                                                                   if categorical_options else None),
                                              mode_option_map=opt_map).validate()
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
           "n_first_passage_processes": n_ev, "scheduling": "continuous_first_passage",
           "stance_updating": ("event_driven_material_change" if stances else None),
           "n_stance_reviews": n_reviews,
           "persistence_window_days": round(persistence_s / 86400.0, 1) if persistence_s else 0,
           "contested_mode_channels": contested_rep,
           "agreement_hazard_ratio": agr_hr, "hazard_ratio_by_mode": mode_hr,
           "hazard_ratio_source": ("fitted_pack" if INTENTION_HR_PACK.exists()
                                   else "documented_priors_unfitted"),
           "hr_pack": hr_pack_info(),
           "coupling_source": coupling_pack_info(),
           "categorical_options": list(categorical_options) if categorical_options else None,
           "n_grounded_stances": len(stances),
           "n_absorbing_institutional_decisions": n_inst_absorbing,
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


def convert_binary_to_event_time(plan, criterion: dict, *, lineage: dict = None,
                                 llm=None) -> dict:
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

    # ---- residual process: ONE continuous-time first-passage process whose per-particle
    #      cumulative intensity is −ln(1−target) shaped by the fitted family curve (§15) —
    #      exact mass conservation with no evenly spaced grid and no cadence parameter
    curve, fam, curve_src = family_hazard_curve(question)
    from swm.world_model_v2.family_hazards import family_base_rate
    fbr, _fam2, fbr_src = family_base_rate(question)

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
        if not hasattr(plan, "first_passage_processes") or plan.first_passage_processes is None:
            plan.first_passage_processes = []
        reads = sorted({str(c.get("var")) for c in consumed if isinstance(c, dict)
                        and c.get("var")})
        plan.first_passage_processes.append({
            "process_id": "resolution", "kind": "calibrated_resolution", "mode": "resolution",
            "curve": list(curve) if curve else None, "calibration": dict(calibration_base),
            "deadline_ts": deadline_ts, "as_of": plan.as_of, "span_s": span,
            "consume": consumed, "reads": reads})
        n_resid = 1
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
           "n_residual_processes": n_resid, "scheduling": "continuous_first_passage",
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
