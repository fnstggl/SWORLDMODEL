"""EVENT-TIME contract architecture — "when" questions as first-passage problems (universal).

The five components, on the existing event/StateDelta plane:
 1. `EventTimeContract` — terminal readout = per-particle FIRST-PASSAGE times into the absorbing state;
    projects a censoring-aware survival curve, CDF, quantiles, P(censored), and the JOINT mode×time
    distribution. `cdf_at(deadline)` is the binary view: P(X by D) = F(D) (monotone across cutoffs by
    construction).
 2. `AbsorptionMonitorOperator` — runs on EVERY event; when the absorbing predicate first holds, stamps
    `absorbed_at`/`absorbed_by` and freezes them (first passage only). Mechanisms never "resolve the
    outcome"; they change the world and absorption is OBSERVED.
 3. `HazardRoundOperator` — every scheduled round (cadence layer) carries a per-round success hazard
    composed from: the fitted family hazard curve h(t) × the structural mode's prior × the grounded
    intention factor (a 0.9-strength refusal crushes near-term hazard) × consumed causal state. Success
    WRITES the absorbing state at that round's date — timing emerges from causal dynamics.
 4. Fitted time-to-event families — `fit_survival_pack` (calibration split ONLY) fits discrete hazards
    over lifetime-fraction buckets from archived price paths (first time price crossed 0.9 = effective
    resolution proxy; labeled). `family_hazard_curve` serves h(t).
 5. Unification — binary deadline questions are F(deadline); "how does it end" is the absorbed_by
    marginal of the same trajectories.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)

SURV_PACK = Path("experiments/replay_vault_v3/family_survival_pack.json")
INTENTION_HR_PACK = Path("experiments/replay_vault_v3/intention_hr_pack.json")
_BUCKETS = 5                                                  # lifetime-fraction hazard buckets
_Z80 = 1.2816                                                 # 80% two-sided normal quantile

# DOCUMENTED PRIORS — NOT FITTED. Hazard ratio a stated stance implies for AGREEMENT-mode hazards,
# as (median, lo80, hi80). Deliberately conservative (centered nearer 1.0 than any crushed point
# value); each PARTICLE samples its own ratio from the lognormal these bounds define, so the
# uncertainty over the effect size survives into the terminal CDF instead of being collapsed to a
# point coefficient. Replaced wholesale when intention_hr_pack.json exists (fit from resolved cases
# via fit_intention_hazard_ratios — statement-class → observed subsequent hazard change).
INTENTION_HR_PRIORS = {
    "categorical_refusal": (0.55, 0.30, 0.90),
    "conditional_refusal": (0.78, 0.50, 1.10),
    "weak_opposition": (0.90, 0.65, 1.15),
    "neutral": (1.00, 0.80, 1.25),
    "openness_to_agreement": (1.35, 1.00, 1.90),
    "formal_commitment_toward_agreement": (2.10, 1.30, 3.20),
}
# reliability shrinks the LOG-effect toward 1.0 (an inferred leaning moves hazards less than a law)
_RELIABILITY_SHRINK = {"high": 1.0, "medium": 0.6, "low": 0.3}

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


def _shrink_hr(med, lo, hi, reliability):
    s = _RELIABILITY_SHRINK.get(reliability, 0.6)
    return (math.exp(s * math.log(med)), math.exp(s * math.log(lo)), math.exp(s * math.log(hi)))


def agreement_hazard_ratio(stances: list) -> dict:
    """Combine per-actor qualitative stances into ONE agreement-mode hazard-ratio distribution.
    Agreement requires the consent of every veto player, so the BINDING actor is the most opposed
    one (minimum shrunk median) — documented structural choice, not a fitted parameter."""
    table = _hr_table()
    best = None
    for st in stances or []:
        tup = table.get(str(st.get("commitment_level", "")).lower())
        if not tup:
            continue
        med, lo, hi = _shrink_hr(*tup, str(st.get("reliability", "medium")).lower())
        if best is None or med < best["median"]:
            best = {"median": round(med, 4), "lo80": round(lo, 4), "hi80": round(hi, 4),
                    "binding_actor": st.get("actor"), "binding_level": st.get("commitment_level"),
                    "binding_reliability": st.get("reliability")}
    return best or {"median": 1.0, "lo80": 0.8, "hi80": 1.25, "binding_actor": None,
                    "binding_level": "no_grounded_stance", "binding_reliability": None}


def fit_intention_hazard_ratios(rows: list, *, pool_strength: float = 8.0) -> dict:
    """Fit statement-class hazard ratios from RESOLVED historical cases. Each row:
    {commitment_level, hazard_ratio} where hazard_ratio = (agreement hazard after the statement) /
    (agreement hazard before), measured on archived paths of resolved cases. Partial pooling toward
    1.0 (no effect). Writes nothing — caller persists to INTENTION_HR_PACK. Until such labeled data
    exists the documented INTENTION_HR_PRIORS above serve, and are reported as priors."""
    import statistics
    out = {}
    by = {}
    for r in rows:
        lvl = str(r.get("commitment_level", "")).lower()
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
    """First-passage terminal contract. project(branches) reads absorbed_at/absorbed_by from each terminal
    world. Provides the binary view via cdf_at(deadline_ts)."""
    family = "event_time"

    def __init__(self, *, as_of: float, horizon_ts: float, resolves_iff: str = "",
                 modes: list = None, readout_var: str = "absorbed_at"):
        self.as_of, self.horizon_ts = float(as_of), float(horizon_ts)
        self.resolution_rule = resolves_iff[:300]
        self.modes = list(modes or [])
        self.readout_var = readout_var
        self.options = ["absorbed_by_horizon", "censored_beyond_horizon"]

    def validate(self):
        if self.horizon_ts <= self.as_of:
            raise ValueError("event_time contract needs horizon > as_of")
        return self

    def readout(self, world):
        q = world.quantities.get("absorbed_at")
        return getattr(q, "value", None)

    def project(self, branches) -> dict:
        times, modes = [], {}
        n = max(1, len(branches))
        for b in branches:
            t = self.readout(b.world)
            if isinstance(t, (int, float)) and t > 0:
                times.append(float(t))
                m = b.world.quantities.get("absorbed_by")
                mid = str(getattr(m, "value", None) or "unspecified")
                modes[mid] = modes.get(mid, 0) + 1
        times.sort()
        p_absorbed = len(times) / n
        span = self.horizon_ts - self.as_of
        grid = [self.as_of + k / 10.0 * span for k in range(1, 11)]
        cdf = [round(sum(1 for t in times if t <= g) / n, 4) for g in grid]
        qtl = {}
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            k = int(q * n)
            qtl[str(q)] = round(times[k], 1) if k < len(times) else None    # None = beyond horizon
        return {"distribution": {"absorbed_by_horizon": round(p_absorbed, 4),
                                 "censored_beyond_horizon": round(1 - p_absorbed, 4)},
                "family": "event_time", "n_deltas": sum(len(b.log) for b in branches),
                "readout": "terminal_states",
                "event_time": {"n_particles": n, "n_absorbed": len(times),
                               "p_censored": round(1 - p_absorbed, 4),
                               "cdf_grid_ts": [round(g, 0) for g in grid], "cdf": cdf,
                               "survival": [round(1 - c, 4) for c in cdf],
                               "first_passage_quantiles_ts": qtl,
                               "mode_distribution": {k: round(v / n, 4) for k, v in modes.items()}}}

    def cdf_at(self, deadline_ts: float, branches) -> float:
        n = max(1, len(branches))
        return sum(1 for b in branches
                   if isinstance(self.readout(b.world), (int, float))
                   and self.readout(b.world) <= deadline_ts) / n


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
    """One causal round of a process that may enter the absorbing state. payload = {mode, base_hazard,
    hr {median,lo80,hi80} (or legacy intention_factor), hazard_bucket_curve (per lifetime-fraction),
    consume:[{var,weight}]}. The realized per-round hazard = curve(t) × per-branch SAMPLED hazard
    ratio, shifted by consumed causal state (bounded, inside this
    mechanism). On success: writes absorbing_state_reached + absorbing_mode at THIS round's date."""
    name = "hazard_round"

    def applicable(self, world, event):
        q = world.quantities.get("absorbed_at")
        return event.etype == "hazard_round" and getattr(q, "value", None) in (None, 0)

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

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import consume_state_rate, _branch_rng
        a = proposal.action
        frac = min(0.999, max(0.0, (world.clock.now - float(a.get("as_of", world.clock.now)))
                              / max(1.0, float(a.get("span_s", 1.0)))))
        curve = a.get("hazard_bucket_curve") or []
        h = float(curve[min(int(frac * _BUCKETS), _BUCKETS - 1)]) if curve \
            else float(a.get("base_hazard", 0.05))
        hr_used = None
        if isinstance(a.get("hr"), dict):                     # distributional hazard ratio (sampled)
            hr_used = self._sampled_hr(world, str(a.get("mode")), a["hr"])
            h *= hr_used
        else:                                                 # legacy point factor
            h *= max(0.0, min(2.0, float(a.get("intention_factor", 1.0))))
        h, used = consume_state_rate(world, h, a.get("consume") or [])
        h = max(0.0, min(0.95, h))
        rng2 = _branch_rng(world, f"hz:{a.get('mode')}:{world.clock.now}")
        d = StateDelta(at=world.clock.now, event_type="hazard_round", operator=self.name,
                       reason_codes=proposal.reason_codes + [f"h={round(h, 4)}"],
                       uncertainty={"hazard": round(h, 4), "lifetime_fraction": round(frac, 3),
                                    "consumed": used,
                                    "sampled_hazard_ratio": (round(hr_used, 4) if hr_used is not None
                                                             else None),
                                    "intention_factor": a.get("intention_factor")})
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


def _mode_requires_agreement(mode) -> bool:
    """Prefer the mode's own semantic flag (from elicitation); fall back to keyword matching for
    compiler hypotheses that carry no flag."""
    if isinstance(mode, dict) and "requires_agreement" in mode:
        return bool(mode["requires_agreement"])
    mid = str(mode["id"] if isinstance(mode, dict) else mode).lower()
    return any(t in mid for t in _AGREEMENT_TOKENS)


_MODES_PROMPT = """Through which mutually exclusive END-STATES can this question's outcome be REACHED?
List 2-6, each an end-state that SATISFIES the resolution criterion when it holds (not an intermediate
or escalation state). Mark whether reaching it REQUIRES the principal parties to agree (a treaty/deal
does; a unilateral collapse/victory/decision does not).
QUESTION: {q}
RESOLUTION CRITERION: {crit}
Return ONLY JSON:
{{"modes": [{{"id": "<snake_case>", "prior": <0..1 relative weight>,
   "requires_agreement": true|false, "describe": "<one sentence>"}}]}}"""


def _elicit_modes(question, criterion, llm) -> list:
    """When the compiler declared no structural hypotheses, elicit the end-state decomposition
    directly (criterion-anchored) — a single amalgam 'resolution' mode loses the mode structure a
    when-question needs. Fails to [] (caller falls back), never blocks."""
    if llm is None:
        return []
    try:
        from swm.engine.grounding import parse_json
        raw = parse_json(llm(_MODES_PROMPT.format(
            q=question, crit=(criterion or {}).get("resolves_yes_iff", "(as stated)")))) or {}
        out = []
        for m in (raw.get("modes") or []):
            if isinstance(m, dict) and m.get("id"):
                out.append({"id": str(m["id"])[:40], "prior": float(m.get("prior", 1.0) or 1.0),
                            "requires_agreement": bool(m.get("requires_agreement"))})
        return out[:6]
    except Exception:  # noqa: BLE001
        return []


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
    if not modes:
        modes = _elicit_modes(getattr(plan, "question", ""), criterion, llm)
    modes = modes or [{"id": "resolution", "prior": 1.0}]
    modes, rejected_modes = _filter_absorbing_modes(modes, criterion, llm)
    z = sum(m["prior"] for m in modes) or 1.0
    curve, fam, curve_src = family_hazard_curve(getattr(plan, "question", ""))
    # grounded-intention effect on AGREEMENT modes: a hazard-ratio DISTRIBUTION built from the
    # qualitative stance record (binding = most-opposed veto actor), sampled per particle — never a
    # point coefficient. Non-agreement (unilateral) modes are unaffected by stances toward agreement.
    stances = list(getattr(plan, "_intention_stances", []) or [])
    agr_hr = agreement_hazard_ratio(stances)
    if AGREEMENT_HR_OVERRIDE is not None:                     # sensitivity harness only
        v = float(AGREEMENT_HR_OVERRIDE)
        agr_hr = {"median": v, "lo80": v, "hi80": v, "binding_actor": "OVERRIDE",
                  "binding_level": "sensitivity_override", "binding_reliability": None}
    vic_hr = {"median": 1.0, "lo80": 1.0, "hi80": 1.0}
    if VICTORY_HR_OVERRIDE is not None:                       # sensitivity harness only
        v = float(VICTORY_HR_OVERRIDE)
        vic_hr = {"median": v, "lo80": v, "hi80": v}
    span = plan.horizon_ts - plan.as_of
    horizon_days = max(1.0, span / 86400.0)
    n_rounds = max(4, min(20, int(horizon_days / max(1.0, horizon_days / 10.0))))
    consumed = list(getattr(plan, "_consumed_state", []) or [])
    n_ev = 0
    mode_hr = {}
    for m in modes[:6]:
        share = m["prior"] / z
        # grounded stances modulate only AGREEMENT modes: a refusal to negotiate suppresses deal
        # hazards (by a sampled, uncertain ratio), never a unilateral end-state's hazard
        agreement = _mode_requires_agreement(m)
        hr_m = agr_hr if agreement else vic_hr
        mode_hr[m["id"]] = {k: hr_m.get(k) for k in ("median", "lo80", "hi80")}
        mode_hr[m["id"]]["requires_agreement"] = agreement
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
                            "hr": {k: hr_m.get(k) for k in ("median", "lo80", "hi80")},
                            "requires_agreement": agreement,
                            "as_of": plan.as_of, "span_s": span,
                            "consume": consumed}})
            n_ev += 1
    for mech_id, op in (("absorption_monitor", "absorption_monitor"), ("hazard_rounds", "hazard_round")):
        if not any(x.get("operator") == op for x in plan.accepted_mechanisms if isinstance(x, dict)):
            plan.accepted_mechanisms.append({
                "mech_id": mech_id, "ontology_type": "event_time", "operator": op,
                "causal_role": "first-passage timing machinery",
                "parameter_source": curve_src if op == "hazard_round" else "observation",
                "temporal_scale": "event", "calibration_status": curve_src, "sensitivity": 1.0})
    plan.scheduled_events = [e for e in plan.scheduled_events if e.get("etype") != "resolve_outcome"]
    # declare the readout quantities on the plan so build_world registers them and the contract's
    # readout binds at materialization (value None until first passage stamps it)
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    for name in ("absorbed_at", "absorbing_state_reached"):
        if name not in declared:
            plan.quantities.append({"name": name, "qtype": name, "value": None, "sd": None})
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
           "n_particles": n_particles,
           "n_hazard_rounds": n_ev, "rounds_per_mode": n_rounds,
           "agreement_hazard_ratio": agr_hr, "hazard_ratio_by_mode": mode_hr,
           "hazard_ratio_source": ("fitted_pack" if INTENTION_HR_PACK.exists()
                                   else "documented_priors_unfitted"),
           "n_grounded_stances": len(stances),
           "family": fam, "hazard_curve_source": curve_src}
    if lineage is not None:
        lineage["event_time"] = rep
    return rep
