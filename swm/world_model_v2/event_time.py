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
    Three parameterizations: (a) a fitted family hazard curve h(t) × structural-mode share × grounded
    intention factor ("when" questions); (b) CALIBRATED — the per-particle target absorbed-mass is drawn
    from the evidence-updated POSTERIOR rate particles (fallback: fitted family base rate, then the broad
    lean-Beta prior) and spread over the rounds as exponents shaped by the family curve, so the same
    calibrated information that used to bias the terminal coin now parameterizes a causal process whose
    outcome is observed; (c) `success_prob` — an outcome-entailing DATED FACT absorbs at its real date
    with probability = fact confidence (the residual chain is budgeted down by the fact's mass).
 4. Fitted time-to-event families — `fit_survival_pack` (calibration split ONLY) fits discrete hazards
    over lifetime-fraction buckets from archived price paths. `family_hazard_curve` serves h(t).
 5. Unification — `convert_to_event_time` rewires "when" questions; `convert_binary_to_event_time`
    rewires binary deadline questions through the SAME machinery: the terminal resolver events
    (resolve_outcome / aggregate_outcome_resolution) are removed, outcome-entailing facts and
    institutional decisions become absorbing writers at their real dates, and the answer is read out
    of the trajectories. Universal: routing is linguistic + structural, never scenario-specific.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)

SURV_PACK = Path("experiments/replay_vault_v3/family_survival_pack.json")
_BUCKETS = 5                                                  # lifetime-fraction hazard buckets


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

      curve      payload = {mode, base_hazard | hazard_bucket_curve, intention_factor, consume}: fitted
                 family curve(t) × intention factor, shifted by consumed causal state ("when" questions).
      calibrated payload["calibration"] = {exponent, absorb_from, fact_floor, posterior_rate_particles?,
                 fallback_rate?, fallback_provenance?, lean}: ONE target absorbed-mass per particle is
                 drawn from the posterior rate particles (fallback: fitted family rate, then broad
                 lean-Beta), polarity-mapped, budgeted down by absorbing-fact mass, and spread across the
                 chain as h_k = 1-(1-target)^exponent_k (Σ exponent = 1 ⇒ the chain's total absorbed mass
                 reproduces the target exactly before structural modulation).
      fact       payload["success_prob"]: an outcome-entailing dated fact fires at ITS date with
                 probability = confidence — a calendar fact is not intention- or crowd-modulated.

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
        from swm.world_model_v2.phase_consumers import consume_state_rate, _branch_rng
        a = proposal.action
        frac = min(0.999, max(0.0, (world.clock.now - float(a.get("as_of", world.clock.now)))
                              / max(1.0, float(a.get("span_s", 1.0)))))
        unc = {"lifetime_fraction": round(frac, 3)}
        used = []
        if a.get("success_prob") is not None:                 # dated entailing fact — fires at ITS date
            h = max(0.0, min(0.999, float(a["success_prob"])))
            unc["rate_source"] = "entailed_fact_confidence"
        elif isinstance(a.get("calibration"), dict):          # posterior-calibrated residual chain
            t, src = self._calibrated_target(world, a["calibration"])
            exp = max(0.0, float(a["calibration"].get("exponent", 0.0) or 0.0))
            h = 1.0 - (1.0 - t) ** exp
            h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
            h, used = consume_state_rate(world, h, a.get("consume") or [])
            h = max(0.0, min(0.95, h))
            unc.update({"rate_source": src, "target_mass": round(t, 4),
                        "consumed": used, "intention_factor": a.get("intention_factor")})
        else:                                                 # fitted family curve ("when" questions)
            curve = a.get("hazard_bucket_curve") or []
            h = float(curve[min(int(frac * _BUCKETS), _BUCKETS - 1)]) if curve \
                else float(a.get("base_hazard", 0.05))
            h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
            h, used = consume_state_rate(world, h, a.get("consume") or [])
            h = max(0.0, min(0.95, h))
            unc.update({"consumed": used, "intention_factor": a.get("intention_factor")})
        unc["hazard"] = round(h, 4)
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
                  parameter_source="fitted family hazard curve × grounded intention factor × consumed "
                                   "causal state (bounded, inside the mechanism)", validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("hazard_round", "absorption"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("quantities",), deltas=("quantities",),
                            parameter_source="event-time architecture", validated=True)


# ---------------------------------------------------------------- 4. fitted survival pack
def fit_survival_pack(worlds_with_paths: list, *, pool_strength: float = 6.0) -> dict:
    """worlds_with_paths: [{question, lifetime_fraction_resolved | None}] — calibration split ONLY.
    `lifetime_fraction_resolved` = first time the archived price crossed 0.9 (effective-resolution proxy,
    labeled), as a fraction of market lifetime; None = never (censored). Fits discrete per-bucket hazards
    with partial pooling toward the global curve."""
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
# modes whose absorbing state requires the parties to AGREE — only these are modulated by the grounded
# intention factor (a stated refusal to negotiate suppresses deal hazards; it does NOT suppress a
# military-collapse or frozen-conflict hazard)
_AGREEMENT_TOKENS = ("ceasefire", "treaty", "agreement", "deal", "settlement", "negotiat", "accord",
                     "truce", "pact", "compromise")


def is_when_question(question: str) -> bool:
    q = str(question).lower()
    return any(t in q for t in _WHEN_TOKENS)


def _mode_requires_agreement(mode_id: str) -> bool:
    m = str(mode_id).lower()
    return any(t in m for t in _AGREEMENT_TOKENS)


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


def convert_to_event_time(plan, criterion: dict, *, lineage: dict = None, llm=None) -> dict:
    """Replace the compiled point contract with an EventTimeContract and schedule the timing machinery:
    per-mode hazard_round chains at the trajectory cadence (modes = the compiler's structural hypotheses /
    categorical options — 'how it ends' becomes the absorbed_by marginal), the absorption monitor, and the
    intention factor from grounded intentions. Universal: built only from the plan's own structure."""
    modes = []
    for h in (getattr(plan, "structural_hypotheses", []) or []):
        if isinstance(h, dict) and h.get("id"):
            modes.append({"id": str(h["id"])[:40], "prior": float(h.get("prior", 1.0) or 1.0)})
    if not modes and len(getattr(plan.outcome_contract, "options", []) or []) > 2:
        modes = [{"id": str(o)[:40], "prior": 1.0} for o in plan.outcome_contract.options[:6]]
    modes = modes or [{"id": "resolution", "prior": 1.0}]
    modes, rejected_modes = _filter_absorbing_modes(modes, criterion, llm)
    z = sum(m["prior"] for m in modes) or 1.0
    curve, fam, curve_src = family_hazard_curve(getattr(plan, "question", ""))
    # grounded-intention factor: intention_yes_share 0 → crush hazards; 1 → boost (bounded 0.1..1.6)
    share = None
    for q in (getattr(plan, "quantities", []) or []):
        if isinstance(q, dict) and q.get("name") == "actor_intentions":
            share = q.get("value")
    ifac = 1.0 if share is None else max(0.1, min(1.6, 0.2 + 1.4 * float(share)))
    span = plan.horizon_ts - plan.as_of
    horizon_days = max(1.0, span / 86400.0)
    n_rounds = max(4, min(20, int(horizon_days / max(1.0, horizon_days / 10.0))))
    consumed = list(getattr(plan, "_consumed_state", []) or [])
    n_ev = 0
    mode_ifac = {}
    for m in modes[:6]:
        share = m["prior"] / z
        # grounded intentions modulate only AGREEMENT modes: refusal to negotiate crushes deal hazards,
        # never a unilateral end-state's hazard
        ifac_m = ifac if _mode_requires_agreement(m["id"]) else 1.0
        mode_ifac[m["id"]] = round(ifac_m, 3)
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
                            "intention_factor": ifac_m, "as_of": plan.as_of, "span_s": span,
                            "consume": consumed}})
            n_ev += 1
    _ensure_event_time_mechanisms(plan, curve_src)
    plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") != "resolve_outcome"]
    _declare_readout_quantities(plan)
    plan.outcome_contract = EventTimeContract(as_of=plan.as_of, horizon_ts=plan.horizon_ts,
                                              resolves_iff=str((criterion or {}).get("resolves_yes_iff",
                                                                                     ""))[:300],
                                              modes=[m["id"] for m in modes]).validate()
    rep = {"modes": [m["id"] for m in modes], "rejected_non_absorbing_modes": rejected_modes,
           "n_hazard_rounds": n_ev, "rounds_per_mode": n_rounds,
           "intention_factor": round(ifac, 3), "intention_factor_by_mode": mode_ifac,
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
        the terminal coin now parameterizes causal dynamics whose first passage is OBSERVED;
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
           "consumed_state": consumed, "family": fam, "hazard_curve_source": curve_src,
           "fallback_rate_provenance": fbr_src,
           "posterior_calibrated": bool(calibration_base["posterior_rate_particles"]),
           "n_resolver_events_removed": len(removed)}
    if lineage is not None:
        lineage["event_time"] = rep
    return rep
