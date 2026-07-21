"""EVENT-TIME contract architecture — every resolvable question as a first-passage problem (universal).

READOUT, NOT RESOLVER. The answer mechanism distinguishes two things:
  1. a READOUT — translating the simulated world into the format the question asks for;
  2. a RESOLVER — an additional decision or random draw that DECLARES the answer.
You need the readout; you should not have the resolver. Mechanisms simulate events through time;
when the target condition first becomes true, absorption is OBSERVED; every related forecast (the
deadline share, the timing distribution, the mode of resolution) is derived from those same
simulated trajectories. Nothing at the terminal draws an outcome.

§NAP — NO ARBITRARY NUMERIC REALITY. This module used to carry the numerical social layer:
stance-level hazard-ratio priors (never fitted), a 0.6 endogenous split, LLM-classified process
labels mapped onto a 0.15…0.85 progress bar consumed by hazards, LLM mode priors normalized into
first-passage intensity shares, and a calibrated-target ladder that fell through from the evidence
posterior to a 40-world keyword family rate and then to a broad lean-Beta. ALL of that is gone
from production (tables buried in legacy_numeric_ablations; the ladder's family/Beta rungs
deleted). What remains is honest:

  * ABSORBING WRITERS with approved provenance — an evidence-cited outcome-entailing dated fact
    absorbs at its REAL date; a declared institutional decision absorbs at its scheduled date
    through its real rule over its real members (generated-mode member votes); scenario-generated
    actor-mediated mechanisms write the absorbing predicate through typed state.
  * ONE residual outcome process, ONLY when the evidence-updated posterior exists (≥1 effective
    as-of observation) — the target mass is drawn from the posterior particles and spread
    uniformly over the window (deterministic arithmetic; registered in the numeric-provenance
    ledger). No posterior ⇒ NO residual process: the mechanism is recorded UNRESOLVED and the
    unabsorbed mass classifies `unresolved_mechanism`, never `resolved_no`.
  * BRANCH TERMINAL CATEGORIES in the projection — resolved_yes / resolved_no /
    censored_by_real_horizon / unresolved_mechanism — with honest bounds (min supported yes
    share; max possible yes share given unresolved branches). Unresolved mass is NEVER
    normalized away, and simulated frequencies are labeled `simulated_scenario_frequency`.

Stances remain QUALITATIVE facts on actor records: they condition each actor's own situated LLM
cognition (the behavior channel — qualitative_actor), and a literal binding instrument may
constrain feasibility (resolution_criteria). They never multiply a hazard.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)
from swm.world_model_v2.mode_graph import (canon_level, canonical_modes, mode_pathway,
                                           pathway_of)
from swm.world_model_v2.numeric_provenance import (NumericProvenanceRejected, ledger_of,
                                                   plan_ledger_of, plan_record_unresolved,
                                                   record_unresolved_mechanism,
                                                   unresolved_mechanisms_of,
                                                   fitted_artifact_eligible)

SURV_PACK = Path("experiments/replay_vault_v3/family_survival_pack.json")
_BUCKETS = 5                                                  # lifetime-fraction hazard buckets
_Z80 = 1.2816                                                 # 80% two-sided normal quantile

#: the honest label for every share counted from LLM role-play particles (§NAP): these are
#: frequencies of outcomes across simulated scenario branches, not calibrated real-world
#: probabilities — calibration is a separate, later, held-out exercise.
FREQUENCY_SEMANTICS = "simulated_scenario_frequency"


def hr_pack_info() -> dict:
    """Stance→hazard effect sizes are QUARANTINED from production (§NAP): the unfitted
    INTENTION_HR_PRIORS are buried in legacy_numeric_ablations, and no fitted pack currently
    passes the provenance contract (numeric_provenance.fitted_artifact_eligible). A future fitted
    artifact carrying training population, outcome definition, data cutoff, n, held-out metrics,
    calibration, domain/transport restrictions and architecture version could restore a numerical
    stance channel through the gate."""
    return {"source": "quarantined_no_production_stance_hazard_channel", "fitted_at": None,
            "n_rows": None, "stratified": False}


def fit_intention_hazard_ratios(rows: list, *, pool_strength: float = 8.0) -> dict:
    """Fit statement-class hazard ratios from RESOLVED historical cases — the OFFLINE fitting
    utility for a future stance-effect artifact. Each row: {commitment_level, hazard_ratio,
    pathway?}. Writes nothing. NOTE (§NAP): the artifact this produces is NOT production-eligible
    until it also carries the full provenance contract (training population, outcome definition,
    frozen data cutoff, n ≥ the support floor, held-out metrics, calibration diagnostics,
    domain/transport restrictions, architecture version) and passes
    numeric_provenance.fitted_artifact_eligible."""
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
    answers the question in its own two options, WITH branch terminal categories: a branch whose
    outcome-relevant mechanism carries an unresolved record must not read out as resolved — its mass
    stays explicit `unresolved_mechanism` mass, and the projection reports honest bounds instead of
    normalizing it away (§NAP)."""
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

    @staticmethod
    def _branch_unresolved(world) -> list:
        return unresolved_mechanisms_of(world)

    def project(self, branches) -> dict:
        times, modes = [], {}                                  # times: [(t, weight)]
        n = sum(max(0.0, float(getattr(b, "weight", 1.0))) for b in branches) or 1.0
        unresolved_w = 0.0
        unresolved_mechs = {}
        censored_modeled_w = 0.0
        for b in branches:
            w = max(0.0, float(getattr(b, "weight", 1.0)))
            t = self.readout(b.world)
            if isinstance(t, (int, float)) and t > 0:
                times.append((float(t), w))
                mid = self._mode_of(b.world)
                modes[mid] = modes.get(mid, 0.0) + w
            else:
                unres = self._branch_unresolved(b.world)
                if unres:
                    unresolved_w += w
                    for u in unres[:4]:
                        key = str(u.get("mechanism", "?"))
                        unresolved_mechs[key] = unresolved_mechs.get(key, 0) + 1
                else:
                    censored_modeled_w += w
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
        unresolved_share = round(unresolved_w / n, 4)
        et = {"n_particles": len(branches), "n_absorbed": len(times),
              "p_censored": round(1 - p_absorbed, 4),
              "censored_modeled_share": round(censored_modeled_w / n, 4),
              "unresolved_share": unresolved_share,
              "unresolved_mechanisms": sorted(unresolved_mechs,
                                              key=lambda k: -unresolved_mechs[k])[:6],
              "frequency_semantics": FREQUENCY_SEMANTICS,
              "cdf_grid_ts": [round(g, 0) for g in grid], "cdf": cdf,
              "survival": [round(1 - c, 4) for c in cdf],
              "first_passage_quantiles_ts": qtl,
              "mode_distribution": {k: round(v / n, 4) for k, v in modes.items()}}
        if len(self.binary_options) == 2:
            # the question's own answer is a READOUT of the same trajectories, with unresolved
            # branch mass kept explicit: absorbed-by-deadline mass resolves the occurrence side;
            # unabsorbed mass resolves the other side ONLY on branches whose outcome mechanisms
            # were all modeled — otherwise it is unresolved_mechanism mass, never a "no"
            occ_w = sum(w for t, w in times if t <= self.deadline_ts)
            # branches that absorbed AFTER the deadline still answered the deadline question
            # through a modeled mechanism (the event demonstrably did not occur in time)
            late_w = sum(w for t, w in times if t > self.deadline_ts)
            nonocc_w = censored_modeled_w + late_w
            p_occ = occ_w / n
            p_nonocc = nonocc_w / n
            if self.occurrence_resolves == "yes":
                p_yes, p_no = p_occ, p_nonocc
            else:
                p_yes, p_no = p_nonocc, p_occ
            dist = {self.binary_options[0]: round(p_yes, 4),
                    self.binary_options[1]: round(p_no, 4)}
            if unresolved_share > 0:
                dist["unresolved_mechanism"] = unresolved_share
            et["p_event_by_deadline"] = round(p_occ, 4)
            et["deadline_ts"] = round(self.deadline_ts, 0)
            et["occurrence_resolves"] = self.occurrence_resolves
            et["branch_terminals"] = {
                "resolved_yes": round(p_yes, 4), "resolved_no": round(p_no, 4),
                "censored_by_real_horizon": round(censored_modeled_w / n, 4),
                "unresolved_mechanism": unresolved_share}
            et["bounds"] = {
                self.binary_options[0]: {"min_supported": round(p_yes, 4),
                                         "max_possible": round(min(1.0, p_yes + unresolved_share), 4)},
                self.binary_options[1]: {"min_supported": round(p_no, 4),
                                         "max_possible": round(min(1.0, p_no + unresolved_share), 4)}}
            resolved_total = p_yes + p_no
            et["resolved_conditional"] = (
                {self.binary_options[0]: round(p_yes / resolved_total, 4),
                 self.binary_options[1]: round(p_no / resolved_total, 4)}
                if resolved_total > 0 else None)
        elif self.categorical_options:
            # categorical readout: WHICH mode produced absorption, mapped back to the question's own
            # option labels; worlds that reached none of them are honest residual mass, never a draw —
            # and worlds with unresolved mechanisms are unresolved mass, never "none of the options"
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
            dist["none_of_the_options_by_horizon"] = round(censored_modeled_w / n + unmapped, 4)
            if unresolved_share > 0:
                dist["unresolved_mechanism"] = unresolved_share
            et["unmapped_absorbed_share"] = round(unmapped, 4)
        else:
            dist = {"absorbed_by_horizon": round(p_absorbed, 4),
                    "censored_beyond_horizon": round(censored_modeled_w / n, 4)}
            if unresolved_share > 0:
                dist["unresolved_mechanism"] = unresolved_share
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


# ---------------------------------------------------------------- 3. provenance-gated event rounds
class HazardRoundOperator(TransitionOperator):
    """A dated event that may enter the absorbing state — ONLY under approved numeric provenance.

    The single production parameterization is `success_prob` with a `numeric_provenance` block whose
    source class is approved (observed_measurement / institutional_rule / explicit_user_input /
    derived_deterministic / eligible fitted artifact): an evidence-cited outcome-entailing dated
    fact absorbs deterministically at ITS real date (success_prob=1.0; the extraction-confidence
    LABEL rides qualitatively in provenance — it is never a Bernoulli parameter, §NAP).

    The legacy parameterizations — fitted-curve hazard grids, calibrated target-mass chains with
    stance hazard ratios, intention factors, and consumed progress state — are NOT executable in
    production: a payload carrying them is refused, the mechanism is recorded UNRESOLVED on the
    branch, and the branch is preserved (its unabsorbed mass classifies `unresolved_mechanism`
    at readout, never `resolved_no`)."""
    name = "hazard_round"

    def applicable(self, world, event):
        q = world.quantities.get("absorbed_at")
        flag = world.quantities.get("absorbing_state_reached")
        prov = world.quantities.get("provisional_absorbing_mode")
        return event.etype == "hazard_round" and getattr(q, "value", None) in (None, 0) \
            and not getattr(flag, "value", None) \
            and not getattr(prov, "value", None)               # first passage; and while a provisional
                                                               # end-state holds, the world IS in that
                                                               # candidate state — rounds pause until
                                                               # it confirms or collapses

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=[f"mode={event.payload.get('mode', '?')}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import _branch_rng
        a = proposal.action
        mode = str(a.get("mode", "unspecified"))
        d = StateDelta(at=world.clock.now, event_type="hazard_round", operator=self.name,
                       reason_codes=list(proposal.reason_codes))
        if a.get("success_prob") is None:
            # legacy curve/calibrated parameterization — no approved provenance exists for it
            record_unresolved_mechanism(
                world, mechanism=f"hazard_round:{mode}",
                why="legacy hazard-grid/calibrated parameterization carries no approved numeric "
                    "provenance (§NAP); the transition mechanism is unresolved",
                missing="production-eligible transition intensity for this process")
            d.reason_codes.append("numeric_provenance_rejected_unresolved")
            return d
        prov = a.get("numeric_provenance") or {}
        try:
            p = ledger_of(world).approve(
                name=f"absorbing_fact:{mode}", value=float(a["success_prob"]),
                units="deterministic_occurrence", causal_role="dated outcome-entailing fact "
                "absorbs at its real date", source_class=str(prov.get("source_class", "")),
                consumer="hazard_round.success_prob",
                evidence_id=str(prov.get("evidence_id", "")),
                fitted_on=prov.get("artifact") or {})
        except NumericProvenanceRejected as e:
            record_unresolved_mechanism(
                world, mechanism=f"absorbing_fact:{mode}",
                why=f"fact-occurrence input rejected: {e}",
                missing="evidence-cited scheduled fact or institutional schedule")
            d.reason_codes.append("numeric_provenance_rejected_unresolved")
            return d
        h = max(0.0, min(1.0, float(p)))
        d.uncertainty = {"success_prob": round(h, 4), "provenance": dict(prov),
                         "fact": str(a.get("fact", ""))[:160]}
        rng2 = _branch_rng(world, f"hz:{mode}:{world.clock.now}")
        if h >= 1.0 or rng2.random() < h:
            persistence_s = float(a.get("persistence_s", 0.0) or 0.0)
            if persistence_s > 0.0:
                # the criterion requires the end-state to HOLD: write a PROVISIONAL absorption and
                # schedule the persistence check — the near-miss ("temporary ceasefire that
                # collapses") is a real possible event, not a parser annotation
                register_quantity_type("provisional_absorbing_mode", units="mode")
                world.quantities["provisional_absorbing_mode"] = Quantity(
                    name="provisional_absorbing_mode", qtype="provisional_absorbing_mode",
                    value=mode, timestamp=world.clock.now)
                d.change("quantities[provisional_absorbing_mode]", None, mode)
                d.reason_codes.append("provisional_pending_persistence")
                d.follow_up_events.append({
                    "etype": "persistence_check", "ts": world.clock.now + persistence_s,
                    "participants": [],
                    "payload": {"mode": mode, "pathway": str(a.get("pathway", ""))}})
            else:
                register_quantity_type("absorbing_state_reached", units="bool")
                register_quantity_type("absorbing_mode", units="mode")
                world.quantities["absorbing_state_reached"] = Quantity(
                    name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                    timestamp=world.clock.now)
                world.quantities["absorbing_mode"] = Quantity(
                    name="absorbing_mode", qtype="absorbing_mode",
                    value=mode, timestamp=world.clock.now)
                d.change("quantities[absorbing_state_reached]", None, True)
        return d


register_operator("absorption_monitor", AbsorptionMonitorOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="pure observation of the absorbing predicate (first passage)",
                  validated=True)
register_operator("hazard_round", HazardRoundOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="dated outcome-entailing facts with APPROVED numeric provenance "
                                   "(evidence-cited schedules absorb at their real dates); legacy "
                                   "grid/calibrated parameterizations are refused and recorded "
                                   "unresolved (§NAP)", validated=True)


# ---------------------------------------------------------------- 3b. continuous-time first passage
# The residual outcome process: per-branch persistent Exp(1) threshold, cumulative intensity from
# a provenance-approved target mass spread uniformly over the window. §NAP: NO stance hazard
# ratios, NO mode shares, NO consumed progress state, NO family/lean fallbacks — the ONLY
# admissible parameterization is the evidence-updated posterior (or a future artifact that passes
# the fitted-eligibility gate), registered in the branch's numeric-provenance ledger.

def _fp_target_mass(world, cal: dict) -> tuple:
    """Per-particle target absorbed-mass drawn from the EVIDENCE-UPDATED posterior rate particles
    (particle-rooted stream: matched arms share the target). Raises NumericProvenanceRejected when
    no posterior exists — there is no family-rate rung and no lean-Beta rung (§NAP)."""
    from swm.world_model_v2.phase_consumers import _draw_rate
    from swm.world_model_v2.temporal_model import particle_rng
    parts = cal.get("posterior_rate_particles")
    if not parts:
        raise NumericProvenanceRejected(
            "residual outcome process has no evidence-updated posterior; family-rate and "
            "lean-Beta fallbacks are removed from production (§NAP)",
            name="residual_target_mass", consumer="first_passage.calibrated_resolution")
    rng = particle_rng(world, "event_time_target")
    r = max(0.0, min(1.0, _draw_rate(parts, rng)))
    prior_prov = cal.get("prior_provenance") or {}
    prior_note = (f"prior behind the posterior: {prior_prov.get('source_class', 'unrecorded')}"
                  + (f" (evidence_quality={prior_prov['evidence_quality']}" if
                     prior_prov.get("evidence_quality") else "")
                  + (f", retained_effective_n={prior_prov['retained_effective_n']})"
                     if prior_prov.get("retained_effective_n") is not None
                     else (")" if prior_prov.get("evidence_quality") else "")))
    ledger_of(world).approve(
        name="residual_target_mass", value=round(r, 4), units="absorbed_mass_share",
        causal_role="evidence-updated posterior outcome rate parameterizing the residual "
                    "outcome process", source_class="derived_deterministic",
        consumer="first_passage.calibrated_resolution",
        evidence_id=str(cal.get("posterior_evidence_id", "phase3_posterior")),
        fitted_on={"prior_provenance": prior_prov} if prior_prov else {},
        applicability="posterior requires ≥1 effective as-of observation; its unfitted prior is a "
                      "recorded remaining assumption — " + prior_note)
    t = r if str(cal.get("absorb_from", "rate")) == "rate" else 1.0 - r
    c = max(0.0, min(0.999, float(cal.get("fact_floor", 0.0) or 0.0)))
    if c > 0.0:                                           # absorbing facts already carry mass c
        t = max(0.0, (t - c) / (1.0 - c))
    return max(0.0, min(0.98, t)), "posterior"


def ensure_first_passage_state(world, spec: dict):
    """Get-or-create the branch's CumulativeHazardState for one plan-level process spec. ONLY the
    `calibrated_resolution` kind with an evidence posterior is production-executable; any other
    spec (mode-share intensity, stance-ratio modulation, curve shares) is refused: the mechanism
    is recorded UNRESOLVED on the branch and None is returned (callers skip scheduling — the
    branch is preserved, never silently absorbed or silently resolved-no)."""
    from swm.world_model_v2.temporal_hazards import (ensure_hazard_state,
                                                     rates_from_target_mass)
    pid = str(spec.get("process_id"))
    store = getattr(world, "temporal_hazards", None) or {}
    if pid in store:
        return store[pid]
    as_of = float(spec.get("as_of", world.clock.now))
    span_s = max(1.0, float(spec.get("span_s", 1.0)))
    if spec.get("kind") != "calibrated_resolution":
        record_unresolved_mechanism(
            world, mechanism=f"first_passage:{pid}",
            why="mode-transition intensity has no approved numeric provenance (mode shares, "
                "stance ratios and progress consumption are removed from production, §NAP)",
            missing="empirically fitted, validated transition model for this process")
        return None
    try:
        target, src = _fp_target_mass(world, spec.get("calibration") or {})
    except NumericProvenanceRejected as e:
        record_unresolved_mechanism(
            world, mechanism=f"first_passage:{pid}", why=str(e),
            missing="evidence-updated posterior or production-eligible fitted rate")
        return None
    base_rates = rates_from_target_mass(target, span_s=span_s, curve=None)
    st = ensure_hazard_state(world, pid, as_of=as_of, horizon_ts=as_of + span_s,
                             base_rates=base_rates, reads=(),
                             payload={"etype": "first_passage", "spec": dict(spec),
                                      "mode": spec.get("mode"),
                                      "target_mass": round(target, 4), "rate_source": src,
                                      "modulation_hook": "event_time_mode"})
    st.modulation = 1.0                       # §NAP: no stance/progress modulation channel exists
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
    st.modulation = 1.0
    return st, st.project_crossing()


register_operator("first_passage", FirstPassageOperator(), requires=("quantities",),
                  modifies=("quantities",), temporal_scale="event",
                  parameter_source="continuous-time cumulative-hazard first passage: per-branch "
                                   "Exp(1) threshold (particle-matched); intensity ONLY from a "
                                   "provenance-approved evidence posterior (§NAP: no stance "
                                   "ratios, no shares, no progress modulation)",
                  validated=True)

from swm.world_model_v2.temporal_hazards import register_modulation_hook  # noqa: E402
register_modulation_hook("event_time_mode", lambda world, st: 1.0)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("hazard_round", "absorption"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("quantities",), deltas=("quantities",),
                            parameter_source="event-time architecture", validated=True)

# the world-dynamics layer (observational persistence semantics) is PART OF the event-time
# machinery: importing event_time must register its operators and event types, or the scheduled
# persistence_check events would be dead weight in any process that did not import the runtime
import swm.world_model_v2.world_dynamics  # noqa: E402,F401


# ---------------------------------------------------------------- 4. fitted survival pack
def fit_survival_pack(worlds_with_paths: list, *, pool_strength: float = 6.0) -> dict:
    """worlds_with_paths: [{question, lifetime_fraction_resolved | None}] — calibration split ONLY.
    OFFLINE fitting utility. NOTE (§NAP): the resulting pack is DIAGNOSTIC — it does not carry the
    held-out metrics, calibration diagnostics, domain restrictions or transport checks that
    numeric_provenance.fitted_artifact_eligible requires, so it cannot parameterize a production
    process until those are added and pass."""
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


def family_survival_pack_eligibility() -> dict:
    """§NAP eligibility verdict for the on-disk family survival pack — always computed through the
    ONE gate (fitted_artifact_eligible), never assumed. The current pack is a keyword-family
    reference class with no held-out metrics/calibration/transport check: it FAILS, and is
    therefore diagnostic-only."""
    if not SURV_PACK.exists():
        return {"exists": False, "eligible": False, "why": "no pack on disk"}
    try:
        pack = json.loads(SURV_PACK.read_text())
    except Exception as e:  # noqa: BLE001
        return {"exists": True, "eligible": False, "why": f"unreadable: {type(e).__name__}"}
    ok, why = fitted_artifact_eligible(pack)
    return {"exists": True, "eligible": ok, "why": why,
            "families": {k: v.get("n") for k, v in (pack.get("families") or {}).items()}}


# ---------------------------------------------------------------- 5. plan conversion (unification)
_WHEN_TOKENS = ("when will", "when does", "how long until", "how soon", "by what date", "what date will")


def is_when_question(question: str) -> bool:
    q = str(question).lower()
    return any(t in q for t in _WHEN_TOKENS)


def _mode_pathway(mode) -> str:
    """The causal pathway a mode is reached through (mode_graph owns the taxonomy + fallbacks)."""
    return mode_pathway(mode)


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


def _ensure_event_time_mechanisms(plan, param_src: str):
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
                "causal_role": ("provenance-gated residual outcome process"
                                if op == "first_passage" else "first-passage timing machinery"),
                "parameter_source": param_src if op != "absorption_monitor" else "observation",
                "temporal_scale": "event", "calibration_status": param_src, "sensitivity": 1.0})


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
    `persistence_days` from the parser wins; else the phrase pattern in resolves_yes_iff. 0 = none.
    Provenance: institutional_rule — the number is the criterion's own literal text."""
    pd = (criterion or {}).get("persistence_days")
    if isinstance(pd, (int, float)) and pd > 0:
        return float(pd) * 86400.0
    m = _PERSISTENCE_RX.search(str((criterion or {}).get("resolves_yes_iff", "")))
    if m:
        return float(m.group(1)) * _PERSIST_UNIT_S[m.group(2).lower()]
    return 0.0


def record_pathway_principals(plan, modes: list) -> dict:
    """Record the shared pathways' PRINCIPALS (decision-structure approvers) as a world quantity so
    execution can distinguish principals' moves from bystanders' QUALITATIVELY. This is the
    surviving, non-numeric remnant of the old contested-mode-channel declaration: the numeric
    `mode_progress:*` channels (0.5 init, sd 0.15) no longer exist (§NAP). Idempotent."""
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    principals = {}
    for m in (modes or []):
        pw = _mode_pathway(m)
        ds = (m or {}).get("decision_structure") or {}
        for a in (ds.get("approvers") or []):
            principals.setdefault(pw, set()).add(str(a))
    for pw, names in principals.items():
        var = f"pathway_principals:{pw}"
        if var not in declared:
            plan.quantities.append({"name": var, "qtype": "pathway_principals",
                                    "value": "|".join(sorted(names)), "sd": None})
    return {"principals": {k: sorted(v) for k, v in principals.items()}}


def convert_to_event_time(plan, criterion: dict, *, lineage: dict = None, llm=None,
                          categorical_options: list = None) -> dict:
    """Replace the compiled point contract with an EventTimeContract — the answer becomes a READOUT
    of first-passage times. §NAP: the conversion schedules NO invented intensity. Mode transitions
    happen through: (a) declared institutional decisions executing inside the trajectory (absorbing
    writers at their real dates, decided by their real rules over their real members); (b)
    evidence-cited outcome-entailing dated facts absorbing at their real dates; (c) scenario-
    generated actor-mediated mechanisms writing the absorbing predicate through typed state. A mode
    with none of those channels is recorded UNRESOLVED (plan_record_unresolved) and its mass stays
    explicit `unresolved_mechanism` mass at readout — never a hazard synthesized from stance labels,
    LLM mode priors, family curves or progress bars. With `categorical_options` the SAME machinery
    answers a categorical question via the absorbed_by marginal."""
    modes = list(getattr(plan, "_canonical_modes", None) or [])
    consensus = dict(getattr(plan, "_mode_consensus", None) or {})
    if not modes:
        modes, consensus = canonical_modes(
            question=getattr(plan, "question", ""), criterion=criterion,
            hypotheses=list(getattr(plan, "structural_hypotheses", []) or []),
            options=list(categorical_options
                         or getattr(plan.outcome_contract, "options", []) or []), llm=llm)
    modes, rejected_modes = _filter_absorbing_modes(modes, criterion, llm)
    principals_rep = record_pathway_principals(plan, modes)
    persistence_s = criterion_persistence_s(criterion)
    led = plan_ledger_of(plan)
    if persistence_s > 0:
        led.approve(name="criterion_persistence_window", value=persistence_s, units="seconds",
                    causal_role="the resolution criterion's own literal persistence requirement",
                    source_class="institutional_rule", consumer="persistence_check",
                    evidence_id="resolution_criterion")
    # §NAP: the fitted survival pack is consulted ONLY through the eligibility gate — and the
    # current pack fails it (keyword reference class, no held-out metrics), so no curve serves.
    pack_verdict = family_survival_pack_eligibility()
    led.reject(name="family_survival_curve", value=None, units="bucket_hazards",
               causal_role="generic timing curve for social-process resolution",
               source_class="keyword_family_rate", consumer="convert_to_event_time",
               why=f"pack ineligible: {pack_verdict.get('why')}")
    # a scheduled institutional decision becomes an ABSORBING WRITER for the institutional-pathway
    # mode (pass ⇒ absorbed at the vote's real date; fail ⇒ the world stays unabsorbed): the declared
    # procedure executes INSIDE the trajectory instead of writing a dead outcome variable
    inst_modes = [m for m in modes if _mode_pathway(m) == "institutional_procedure"]
    n_inst_absorbing = 0
    if inst_modes:
        for e in plan.scheduled_events:
            if e.get("etype") == "institutional_decision":
                pl = e.setdefault("payload", {})
                pl["absorbing"] = True
                pl.setdefault("absorbing_mode", inst_modes[0]["id"])
                n_inst_absorbing += 1
    # every mode WITHOUT a concrete modeled channel is an unresolved transition mechanism — named,
    # preserved, never replaced by an invented number
    unresolved_modes = []
    for m in modes:
        pw = _mode_pathway(m)
        has_institution = pw == "institutional_procedure" and n_inst_absorbing > 0
        if not has_institution:
            plan_record_unresolved(
                plan, mechanism=f"mode_transition:{m['id']}",
                why=f"no production-eligible transition model for pathway {pw!r}: transitions "
                    f"must come from scenario-generated events, institutional decisions, or "
                    f"evidence-cited scheduled facts (§NAP removed mode-share/stance-ratio/"
                    f"progress-bar intensity)",
                missing="scenario-specific transition mechanism or eligible fitted model")
            unresolved_modes.append(m["id"])
    _ensure_event_time_mechanisms(plan, "provenance_gated")
    # persistence semantics stay: the check is OBSERVATIONAL (world_dynamics) — a provisional
    # end-state confirms iff it still holds when its window completes in the simulated world
    if persistence_s > 0:
        if not any(x.get("operator") == "persistence_check" for x in plan.accepted_mechanisms
                   if isinstance(x, dict)):
            mech = {"mech_id": "persistence_semantics", "ontology_type": "world_dynamics",
                    "operator": "persistence_check",
                    "causal_role": "near-miss realization: provisional end-states confirm or "
                                   "collapse by OBSERVATION of the simulated world",
                    "parameter_source": "observational (no survival coin, §NAP)",
                    "temporal_scale": "scheduled", "calibration_status": "observational",
                    "sensitivity": 0.8}
            mon = next((i for i, x in enumerate(plan.accepted_mechanisms)
                        if isinstance(x, dict) and x.get("operator") == "absorption_monitor"),
                       None)
            if mon is not None:
                plan.accepted_mechanisms.insert(mon, mech)
            else:
                plan.accepted_mechanisms.append(mech)
    plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") != "resolve_outcome"]
    _declare_readout_quantities(plan)
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
           "scheduling": "provenance_gated_event_time",
           "unresolved_mode_transitions": unresolved_modes,
           "principals": principals_rep.get("principals"),
           "persistence_window_days": round(persistence_s / 86400.0, 1) if persistence_s else 0,
           "n_absorbing_institutional_decisions": n_inst_absorbing,
           "stance_hazard_channel": "removed_quarantined (§NAP)",
           "hr_pack": hr_pack_info(),
           "family_survival_pack": pack_verdict,
           "categorical_options": list(categorical_options) if categorical_options else None,
           "numeric_provenance_manifest": led.manifest()}
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


def convert_binary_to_event_time(plan, criterion: dict, *, lineage: dict = None,
                                 llm=None) -> dict:
    """Rewire a binary deadline question so the answer is DERIVED from the simulation instead of
    declared by a terminal resolver. §NAP semantics:

      * the resolver events (resolve_outcome, aggregate_outcome_resolution) are REMOVED — no
        component draws the outcome;
      * EVIDENCE-CITED outcome-entailing dated facts become deterministic absorbing events at
        their REAL dates (approved provenance; the LLM extraction-confidence number is NEVER a
        Bernoulli parameter). Model-knowledge facts do not absorb — their channel is recorded
        unresolved;
      * an institutional decision that decides THIS outcome becomes an absorbing writer at its
        scheduled date, decided by its real rule (generated-mode member votes);
      * ONE residual outcome process exists ONLY when the evidence-updated posterior exists —
        the family-rate and lean-Beta rungs are deleted. Without it, the residual mechanism is
        recorded UNRESOLVED and unabsorbed branch mass classifies `unresolved_mechanism` at
        readout (bounds reported), never `resolved_no`;
      * P(yes)/P(no) are simulated-scenario frequencies over the RESOLVED mass only, with
        min/max bounds over the unresolved mass — nothing normalized away.

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
    led = plan_ledger_of(plan)

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

    # ---- absorbing facts: EVIDENCE-CITED outcome-entailing dated facts absorb deterministically
    #      at their real dates; model-knowledge facts stay visible state but cannot absorb ----
    absorbing_facts, opposite_facts, ungrounded_facts = [], [], []
    for e in plan.scheduled_events:
        if e.get("etype") != "scheduled_fact":
            continue
        p = e.get("payload") or {}
        # only STRICTLY entailing facts claim the deterministic-absorption channel. A weak/moderate
        # influence nudge (the recurrence-aware schema) is calendar CONTEXT — it reaches the forecast
        # through the recurrence-aware prior and actor calendar knowledge, never as an absorber.
        strictly = p.get("strictly_entailing", p.get("outcome_entailing"))
        if not strictly or p.get("entailed_direction") not in ("yes", "no"):
            continue
        if not (plan.as_of <= float(e.get("ts", 0.0)) <= deadline_ts):
            continue
        if str(p.get("source")) != "evidence" or not p.get("evidence_quote"):
            ungrounded_facts.append(e)
            continue
        (absorbing_facts if p["entailed_direction"] == absorb_dir else opposite_facts).append(e)
    for e in ungrounded_facts:
        p = e.get("payload") or {}
        plan_record_unresolved(
            plan, mechanism=f"scheduled_fact:{str(p.get('kind', 'fact'))[:24]}",
            why="outcome-entailing fact is model-knowledge (no evidence citation); its "
                "occurrence cannot absorb deterministically and its LLM confidence may not "
                "serve as an event probability (§NAP)",
            missing="evidence-cited dated fact or observed institutional schedule")

    # ---- absorbing institution: a declared decision procedure that decides THIS outcome ----
    inst_events = [e for e in plan.scheduled_events
                   if e.get("etype") == "institutional_decision"
                   and str((e.get("payload") or {}).get("outcome_var", "")) == resolve_var]
    inst_in_window = [e for e in inst_events if float(e.get("ts", 0.0)) <= deadline_ts]

    # ---- residual outcome process: ONLY the evidence-updated posterior parameterizes it ----
    posterior_parts = [[float(r), float(w)] for r, w in
                       (getattr(plan, "posterior_rate_particles", None) or [])]
    from swm.world_model_v2.family_hazards import family_base_rate
    fbr, fam, fbr_src = family_base_rate(question)
    # the family rate is registered as a REJECTED input — auditable, never consumed (§NAP)
    led.reject(name="family_fallback_rate", value=fbr, units="probability",
               causal_role="keyword-selected reference-class outcome rate",
               source_class="keyword_family_rate", consumer="convert_binary_to_event_time",
               why=f"family pack ({fam}, {fbr_src}) is a keyword reference class without "
                   f"held-out validation/transport eligibility — diagnostic only")
    led.reject(name="lean_beta_target", value=None, units="beta_params",
               causal_role="qualitative lean → broad Beta outcome mass",
               source_class="lean_beta", consumer="convert_binary_to_event_time",
               why="broad-prior outcome manufacturing is removed from production (§NAP)")

    # ---------------- mutations (everything above computed first so a failure leaves the plan intact) ----
    n_fact_ev = 0
    for e in absorbing_facts:
        p = e.get("payload") or {}
        plan.scheduled_events.append({
            "etype": "hazard_round", "ts": max(plan.as_of + 1.0, float(e.get("ts", plan.as_of + 1.0))),
            "participants": [],
            "payload": {"mode": f"entailed_fact:{str(p.get('kind', 'fact'))[:24]}",
                        "success_prob": 1.0,
                        "numeric_provenance": {
                            "source_class": "observed_measurement",
                            "evidence_id": str(p.get("claim_id", "") or p.get("source", "evidence")),
                            "extraction_confidence_label": p.get("confidence"),
                            "quote": str(p.get("evidence_quote", ""))[:160]},
                        "as_of": plan.as_of, "span_s": span,
                        "fact": str(p.get("fact", ""))[:160], "source": p.get("source")}})
        n_fact_ev += 1
    for e in inst_events:
        pl = e.setdefault("payload", {})
        pl["absorbing"] = True
        pl["absorbing_mode"] = f"institutional:{str(pl.get('institution_id', ''))[:28]}"
    n_resid = 0
    residual_unresolved_reason = None
    if not inst_in_window and not absorbing_facts:
        if posterior_parts:
            if not hasattr(plan, "first_passage_processes") or plan.first_passage_processes is None:
                plan.first_passage_processes = []
            plan.first_passage_processes.append({
                "process_id": "resolution", "kind": "calibrated_resolution", "mode": "resolution",
                "calibration": {
                    "absorb_from": ("rate" if absorb_dir == "yes" else "one_minus_rate"),
                    "fact_floor": 0.0,
                    "posterior_rate_particles": posterior_parts,
                    "posterior_evidence_id": "phase3_posterior",
                    # the SPECIFIC prior behind the posterior (grounded estimate / recurrence /
                    # reference class / generic lean), so the §NAP ledger row acknowledges the
                    # actual remaining assumption rather than a generic one
                    "prior_provenance": dict(getattr(plan, "_outcome_prior_provenance", None) or {})},
                "deadline_ts": deadline_ts, "as_of": plan.as_of, "span_s": span})
            n_resid = 1
        else:
            residual_unresolved_reason = (
                "no evidence-updated posterior, no in-window institutional decision, no "
                "evidence-cited absorbing fact — the residual outcome mechanism is unresolved; "
                "family-rate and lean-Beta manufacturing are removed (§NAP)")
            plan_record_unresolved(
                plan, mechanism="residual_outcome_process",
                why=residual_unresolved_reason,
                missing="evidence-updated posterior, scenario-generated resolving mechanism, "
                        "or production-eligible fitted rate")
    removed = [e for e in plan.scheduled_events
               if e.get("etype") in ("resolve_outcome", "aggregate_outcome_resolution")]
    plan.scheduled_events = [e for e in plan.scheduled_events
                             if e.get("etype") not in ("resolve_outcome", "aggregate_outcome_resolution")]
    _ensure_event_time_mechanisms(
        plan, "provenance_gated" if n_resid else "structural (facts/institutions absorb)")
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
           "n_absorbing_fact_events": n_fact_ev,
           "n_ungrounded_facts_unresolved": len(ungrounded_facts),
           "n_opposite_direction_facts": len(opposite_facts),
           "absorbing_institutions": [str((e.get("payload") or {}).get("institution_id", ""))
                                      for e in inst_events],
           "n_residual_processes": n_resid, "scheduling": "provenance_gated_event_time",
           "residual_skipped_reason": ("institutional decision or evidence-cited fact is the "
                                       "resolution path" if (inst_in_window or absorbing_facts)
                                       else residual_unresolved_reason),
           "posterior_calibrated": bool(posterior_parts),
           "family_rate_rejected": {"family": fam, "provenance": fbr_src, "value": fbr},
           "frequency_semantics": FREQUENCY_SEMANTICS,
           "n_resolver_events_removed": len(removed),
           "numeric_provenance_manifest": led.manifest()}
    if lineage is not None:
        lineage["event_time"] = rep
    return rep
