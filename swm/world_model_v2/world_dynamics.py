"""In-run WORLD DYNAMICS — the layers that make simulated worlds move the way real ones do
(universal; no scenario branching anywhere):

 1. STANCE DYNAMICS (`StanceReviewOperator`): grounded stances are INITIAL conditions, not
    constants. At the trajectory cadence each actor's stances are reviewed against the causal state
    their own (and their rivals') actions produced, and may move ONE commitment level per review,
    with hysteresis, each change a StateDelta with an explicit reason code:
      * RIPENESS — an actor facing imminent defeat (a rival-targeted prevent-stance whose target
        mode's process has advanced past the ripeness threshold) softens their refusal on the
        SHARED pathway: a collapsing battlefield makes the loser more open to talks;
      * WINNING — an actor whose own pursued mode is near completion hardens against concessions
        (why settle when you are winning);
      * EXHAUSTION — an actor whose capacity resource has drained below the exhaustion threshold
        softens their pursue-stances: attrition ends wars, delays launches, kills bills;
      * BANDWAGON — when the shared process itself is succeeding, weak/conditional opponents
        drift toward acceptance.
    The Phase-4 policy reads stances live from entity fields, so behavior changes the next
    decision; hazard rounds re-derive their stance hazard-ratio from CURRENT stances
    (stance-hash-keyed re-sampling), so h(t) genuinely shifts mid-trajectory.
 2. PERSISTENCE SEMANTICS (`PersistenceCheckOperator`): a resolution criterion that requires the
    end-state to HOLD ("no active hostilities for >=30 consecutive days") makes near-miss states
    REAL events: hazard success writes a PROVISIONAL absorption and schedules a persistence check;
    the state either confirms (absorption stamps when the criterion is actually satisfied) or
    COLLAPSES (the temporary ceasefire that fails — the criterion's named near-miss — now actually
    happens in trajectories, knocking the process back down).
 3. CAPACITY: `capability` stops being a static 3-level label — each stance-carrying actor gets a
    capacity resource (initialized from the grounded capability), DEPLETED by effortful actions
    (escalate/mobilize/strike/launch), read live by the stance combiner and the exhaustion rule.
 4. SAMPLED COUPLING CONSTANTS: the behavior→state→hazard coupling magnitudes (action step size,
    endogenous stance split, consume weights, persistence survival, contested suppression) are
    documented PRIOR DISTRIBUTIONS sampled once per branch — structural-constant uncertainty
    propagates into the terminal CDF instead of being a hidden point choice — and are replaced
    wholesale by `coupling_pack.json` when `fit_coupling_pack` has been run against scored
    trajectories (the event-time vault). Provenance always names which one is serving.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)
from swm.world_model_v2.mode_graph import (STANCE_ORIENTATION, canon_level, pathway_of)

COUPLING_PACK = Path("experiments/replay_vault_v3/coupling_pack.json")
_Z80 = 1.2816

# DOCUMENTED PRIORS — NOT FITTED (median, lo80, hi80; lognormal on the positive ones). Each branch
# samples its own value once; fit_coupling_pack replaces the table when scored-trajectory data
# exists. Clamps keep even tail draws physically sensible.
COUPLING_PRIORS = {
    "pathway_step":            (0.04, 0.02, 0.08),   # per-action process step × ontology effect
    "endogenous_stance_split": (0.60, 0.45, 0.80),   # residual share of the direct stance→hazard channel
    "own_pathway_weight":      (1.00, 0.70, 1.40),   # hazard consumption of the mode's own process
    "cross_pathway_weight":    (0.25, 0.12, 0.50),   # spillover consumption of other processes
    "world_state_weight":      (0.35, 0.20, 0.60),   # world-driven couplings (nonlinear/population)
    "contested_suppression":   (0.50, 0.30, 0.80),   # rival-mode suppression on contested pathways
    "nonprincipal_step_share": (0.50, 0.30, 0.80),   # non-principals move a shared process at this share
    "attrition_rate_per_day":  (0.0007, 0.0004, 0.0015),  # capacity drain PER REAL ELAPSED DAY
                                                     # while an actor pursues a CONTESTED pathway
                                                     # (≥2 rivals) — attrition accrues over exact
                                                     # elapsed time (advance_interval), never per
                                                     # review round (§13/§14)
    "persistence_survival_shared":   (0.75, 0.55, 0.90),  # a provisional agreement holds its window
    "persistence_survival_default":  (0.85, 0.70, 0.95),  # a provisional unilateral end-state holds
}
_CLAMPS = {"pathway_step": (0.005, 0.15), "endogenous_stance_split": (0.2, 1.0),
           "own_pathway_weight": (0.3, 2.0), "cross_pathway_weight": (0.0, 1.0),
           "world_state_weight": (0.0, 1.0), "contested_suppression": (0.1, 1.0),
           "nonprincipal_step_share": (0.1, 1.0), "attrition_rate_per_day": (0.0, 0.003),
           "persistence_survival_shared": (0.2, 0.99), "persistence_survival_default": (0.3, 0.99)}
#: fitted-pack backward compatibility: an old pack fitted on the per-review coupling converts
#: to per-day at the legacy ~21-day review cadence (recorded conversion, not a hidden default)
_LEGACY_COUPLING_CONVERSIONS = {"attrition_per_review": ("attrition_rate_per_day", 1.0 / 21.0)}

#: capability grounding → initial capacity resource
CAPACITY_INIT = {"high": 0.85, "medium": 0.6, "low": 0.35}
#: capacity an effortful action burns (escalate/mobilize/strike/launch class)
EFFORTFUL_ACTION_COST = 0.02
EXHAUSTION_THRESHOLD = 0.30
RIPENESS_THRESHOLD = 0.70
BANDWAGON_THRESHOLD = 0.70
#: EVENT-DEPENDENT hysteresis (§13): after a stance change, the same actor's stances move again
#: only when the relevant process state has moved MATERIALLY since that change (never a
#: review-count cooldown — there are no reviews to count).
STANCE_MATERIAL_HYSTERESIS = 0.08

_LEVEL_ORDER = ("committed_to_prevent", "conditionally_opposed", "weakly_opposed", "neutral",
                "inclined_toward", "actively_pursuing", "formally_committed")


def coupling_pack_info() -> dict:
    if COUPLING_PACK.exists():
        try:
            p = json.loads(COUPLING_PACK.read_text())
            return {"source": "fitted_pack", "fitted_at": p.get("fitted_at"),
                    "n_trajectories": p.get("n_trajectories")}
        except Exception:  # noqa: BLE001
            pass
    return {"source": "documented_priors_unfitted", "fitted_at": None, "n_trajectories": None}


def _coupling_table() -> dict:
    if COUPLING_PACK.exists():
        try:
            pack = json.loads(COUPLING_PACK.read_text())
            fitted = {}
            for k, v in (pack.get("couplings") or {}).items():
                if k in _LEGACY_COUPLING_CONVERSIONS:
                    new_key, factor = _LEGACY_COUPLING_CONVERSIONS[k]
                    fitted[new_key] = tuple(x * factor for x in v)
                else:
                    fitted[k] = tuple(v)
            return {**COUPLING_PRIORS, **fitted}
        except Exception:  # noqa: BLE001
            return dict(COUPLING_PRIORS)
    return dict(COUPLING_PRIORS)


def sampled_coupling(world, name: str) -> float:
    """One draw of a structural coupling constant PER BRANCH (persisted on the world) from its
    documented prior (lognormal via the 80% interval) or the fitted pack. The same pattern as the
    stance hazard ratios: coefficient uncertainty becomes cross-particle spread, never a hidden
    point choice."""
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    from swm.world_model_v2.phase_consumers import _branch_rng
    qname = f"sampled_coupling:{name}"
    q = world.quantities.get(qname)
    if q is not None and isinstance(getattr(q, "value", None), (int, float)):
        return float(q.value)
    med, lo, hi = _coupling_table().get(name, (1.0, 1.0, 1.0))
    sigma = (math.log(max(hi, 1e-9)) - math.log(max(lo, 1e-9))) / (2 * _Z80) if hi > lo else 0.0
    rng = _branch_rng(world, f"coupling:{name}")
    val = med * math.exp(sigma * rng.gauss(0.0, 1.0)) if sigma > 0 else float(med)
    c_lo, c_hi = _CLAMPS.get(name, (med * 0.25, med * 4.0))
    val = round(max(c_lo, min(c_hi, val)), 5)
    register_quantity_type("sampled_coupling", units="coefficient")
    world.quantities[qname] = Quantity(name=qname, qtype="sampled_coupling", value=val,
                                       timestamp=world.clock.now)
    return val


def fit_coupling_pack(rows: list, *, pool_strength: float = 6.0) -> dict:
    """Fit coupling constants from SCORED trajectories (the event-time vault): each row is one
    scored world {coupling_draws: {name: value}, crps: float}. For each constant, reweight the
    draws by exp(-crps/scale) (better-scoring worlds' draws count more) and pool toward the prior
    median. This is deliberately a simple importance-reweighting estimator — the honest first
    fit, replaced by a likelihood fit when the corpus grows. Caller persists to COUPLING_PACK."""
    import statistics
    if not rows:
        return {"version": "coupling-1.0", "n_trajectories": 0, "couplings": {}}
    scale = statistics.median([r.get("crps", 0.5) for r in rows]) or 0.5
    out = {}
    for name, (med0, lo0, hi0) in COUPLING_PRIORS.items():
        draws = [(r["coupling_draws"][name], math.exp(-float(r.get("crps", scale)) / scale))
                 for r in rows if isinstance((r.get("coupling_draws") or {}).get(name), (int, float))]
        if not draws:
            continue
        z = sum(w for _, w in draws)
        mean_log = sum(w * math.log(max(1e-9, v)) for v, w in draws) / z
        k = pool_strength / (len(draws) + pool_strength)
        med = math.exp((1 - k) * mean_log + k * math.log(med0))
        spread = (math.log(hi0) - math.log(lo0)) / 2 * (0.5 + 0.5 * k)   # narrow as data grows
        out[name] = (round(med, 5), round(med * math.exp(-spread), 5),
                     round(med * math.exp(spread), 5))
    return {"version": "coupling-1.0", "fit_on": "scored event-time vault trajectories",
            "n_trajectories": len(rows), "couplings": {k: list(v) for k, v in out.items()}}


# ---------------------------------------------------------------- live stance access
def live_stances(world) -> list:
    """Every actor's CURRENT stance records, read from entity fields (the same source the Phase-4
    ActorView projects) — the stance-review operator mutates these, so hazards and policies see one
    consistent, evolving stance state."""
    out = []
    for ent in (world.entities or {}).values():
        recs = ent.value("stances", default=None)
        if isinstance(recs, list):
            out.extend(r for r in recs if isinstance(r, dict))
    return out


def stance_state_hash(stances: list) -> str:
    key = sorted((str(s.get("actor")), canon_level(s.get("commitment_level")),
                  str(s.get("pathway")), str(s.get("target_mode"))) for s in (stances or []))
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]


def live_capacity(world) -> dict:
    """{actor_id: capacity 0..1} from the capacity resource, when declared. Resources are KEYED
    entity state — read with the key, never by unwrapping the dict."""
    out = {}
    for aid, ent in (world.entities or {}).items():
        cap = ent.value("resources", key="capacity", default=None)
        if isinstance(cap, (int, float)):
            out[aid] = max(0.05, min(1.0, float(cap)))
    return out


def _progress(world, var: str):
    q = world.quantities.get(var)
    v = getattr(q, "value", None)
    return float(v) if isinstance(v, (int, float)) else None


# ---------------------------------------------------------------- 1. stance dynamics
class StanceReviewOperator(TransitionOperator):
    """EVENT-DRIVEN stance updating (§13): fires when the temporal runtime observes a MATERIAL
    state change (a watched process var crossed a stance-rule threshold or moved materially —
    stance_relevant_change events), never because a review interval passed. At most ONE stance
    moves ONE level per actor per triggering change, with material-change hysteresis (the
    relevant process must move again before the same actor changes again), each change logged
    with its rule. Universal rules over the mode graph — ripeness / winning / exhaustion /
    bandwagon — never scenario keywords. This is how a simulated leader who is losing becomes
    more open to talks: the stance record itself changes, the policy behaves differently at the
    next decision, and the first-passage hazards re-derive their stance ratio from the new
    record (accumulated hazard preserved). The legacy `stance_review` etype remains accepted
    for explicit ablation runs only."""
    name = "stance_review"

    def applicable(self, world, event):
        stamped = world.quantities.get("absorbed_at")
        return event.etype in ("stance_relevant_change", "stance_review") \
            and getattr(stamped, "value", None) in (None, 0)

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=["stance_review"])

    @staticmethod
    def _shift(level: str, delta: int) -> str:
        i = _LEVEL_ORDER.index(canon_level(level)) if canon_level(level) in _LEVEL_ORDER else 3
        return _LEVEL_ORDER[max(0, min(len(_LEVEL_ORDER) - 1, i + delta))]

    def _mode_or_pathway_progress(self, world, target_mode, pathway):
        if target_mode:
            v = _progress(world, f"mode_progress:{pathway}:{target_mode}")
            if v is not None:
                return v
        return _progress(world, f"pathway_progress:{pathway}")

    def _candidate_update(self, world, actor_id, stances, caps):
        """The highest-priority triggered rule for this actor, or None. One change per
        triggering event. Returns (stance, delta_level, why, driver_value) — the driver value
        anchors material-change hysteresis."""
        my = [s for s in stances if str(s.get("actor")) == actor_id]
        if not my:
            return None
        # RIPENESS: a rival-targeted prevent-stance whose target mode is near completion → the
        # actor's prevent-stance on a SHARED pathway softens one level (imminent defeat → openness)
        losing, losing_v = None, None
        for st in my:
            lvl = canon_level(st.get("commitment_level"))
            if STANCE_ORIENTATION.get(lvl, 0.0) < 0 and st.get("target_mode") \
                    and not pathway_of(str(st.get("pathway", ""))).shared_process:
                v = self._mode_or_pathway_progress(world, st["target_mode"], str(st.get("pathway")))
                if v is not None and v >= RIPENESS_THRESHOLD:
                    losing, losing_v = st["target_mode"], v
                    break
        if losing:
            for st in my:
                lvl = canon_level(st.get("commitment_level"))
                if pathway_of(str(st.get("pathway", ""))).shared_process \
                        and STANCE_ORIENTATION.get(lvl, 0.0) < 0:
                    return (st, +1, f"ripeness:rival_mode_{losing}_at_threshold", losing_v)
        # WINNING: own pursued mode near completion → shared-pathway openness hardens one level
        winning, winning_v = None, None
        for st in my:
            lvl = canon_level(st.get("commitment_level"))
            if STANCE_ORIENTATION.get(lvl, 0.0) > 0 and st.get("target_mode") \
                    and not pathway_of(str(st.get("pathway", ""))).shared_process:
                v = self._mode_or_pathway_progress(world, st["target_mode"], str(st.get("pathway")))
                if v is not None and v >= RIPENESS_THRESHOLD:
                    winning, winning_v = st["target_mode"], v
                    break
        if winning:
            for st in my:
                lvl = canon_level(st.get("commitment_level"))
                if pathway_of(str(st.get("pathway", ""))).shared_process \
                        and STANCE_ORIENTATION.get(lvl, 0.0) > -0.9:
                    return (st, -1, f"winning:own_mode_{winning}_at_threshold", winning_v)
        # EXHAUSTION: drained capacity → pursue-stances on per-actor pathways soften one level
        cap = caps.get(actor_id)
        if cap is not None and cap < EXHAUSTION_THRESHOLD:
            for st in my:
                lvl = canon_level(st.get("commitment_level"))
                if STANCE_ORIENTATION.get(lvl, 0.0) > 0 \
                        and not pathway_of(str(st.get("pathway", ""))).shared_process:
                    return (st, -1, f"exhaustion:capacity_{cap:.2f}", cap)
        # BANDWAGON: the shared process itself is succeeding → weak/conditional opposition drifts
        for st in my:
            lvl = canon_level(st.get("commitment_level"))
            pw = str(st.get("pathway", ""))
            if pathway_of(pw).shared_process and lvl in ("conditionally_opposed", "weakly_opposed"):
                v = _progress(world, f"pathway_progress:{pw}")
                if v is not None and v >= BANDWAGON_THRESHOLD:
                    return (st, +1, f"bandwagon:{pw}_at_{v:.2f}", v)
        return None

    def apply(self, world, proposal):
        from swm.world_model_v2.state import F
        trigger_changes = proposal.action.get("changes") or []
        stances = live_stances(world)
        d = StateDelta(at=world.clock.now, event_type="stance_relevant_change",
                       operator=self.name, reason_codes=["stance_update"],
                       uncertainty={"triggering_changes": trigger_changes[:6],
                                    "provenance": proposal.action.get(
                                        "provenance", "material_state_change")})
        caps = live_capacity(world)
        n_changed = 0
        for aid, ent in sorted((world.entities or {}).items()):
            recs = ent.value("stances", default=None)
            if not isinstance(recs, list) or not recs:
                continue
            upd = self._candidate_update(world, aid, stances, caps)
            if not upd:
                continue
            st, delta_lvl, why, driver_v = upd
            # MATERIAL-CHANGE HYSTERESIS (§13): the same actor changes again only when the
            # rule's driving state moved materially since their LAST change — event-dependent,
            # never a review-count cooldown, and immune to duplicate cadence artifacts.
            last_driver = ent.value("latent_state", key="stance_last_change_driver",
                                    default=None)
            if isinstance(last_driver, (int, float)) and isinstance(driver_v, (int, float)) \
                    and abs(float(driver_v) - float(last_driver)) < STANCE_MATERIAL_HYSTERESIS:
                continue
            new_recs = []
            for r in recs:
                if r is st or (r.get("actor") == st.get("actor")
                               and r.get("pathway") == st.get("pathway")
                               and r.get("target_mode") == st.get("target_mode")
                               and r.get("commitment_level") == st.get("commitment_level")):
                    before = canon_level(r.get("commitment_level"))
                    after = self._shift(before, delta_lvl)
                    if after == before:
                        new_recs.append(r)
                        continue
                    nr = dict(r, commitment_level=after)
                    nr.setdefault("updates", []).append(
                        {"at": world.clock.now, "from": before, "to": after, "rule": why})
                    new_recs.append(nr)
                    d.change(f"{aid}.stances[{st.get('pathway')}"
                             f"{':' + str(st.get('target_mode')) if st.get('target_mode') else ''}]"
                             f".commitment_level", before, after)
                    d.reason_codes.append(f"{aid}:{why}")
                    n_changed += 1
                else:
                    new_recs.append(r)
            if n_changed:
                ent.set("stances", F(new_recs, status="derived", method="stance_update",
                                     updated_at=world.clock.now))
                if isinstance(driver_v, (int, float)):
                    ent.set("latent_state", F(float(driver_v), status="derived",
                                              method="stance_update",
                                              updated_at=world.clock.now),
                            key="stance_last_change_driver")
        if not n_changed:
            return None                                       # honest no-op — nothing triggered
        d.uncertainty["n_stances_changed"] = n_changed
        return d


def contested_attrition_interval(world, elapsed_days: float, *, branch_log=None,
                                 at_ts=None) -> int:
    """CONTINUOUS attrition over REAL ELAPSED TIME (§13/§14): while an actor PURSUES a contested
    (non-shared, actor-driven) pathway that at least one RIVAL also pursues, their capacity
    drains at the sampled per-day attrition rate × the exact elapsed interval — wars of
    attrition exhaust by DURATION, not by how many review rounds a scheduler happened to run.
    Called from temporal_runtime.advance_interval between events. Only actors with a declared
    capacity resource drain. Returns the number of capacity writes."""
    if elapsed_days <= 0:
        return 0
    from swm.world_model_v2.state import F
    stances = live_stances(world)
    pursuers = {}
    for st in stances:
        lvl = canon_level(st.get("commitment_level"))
        pw = str(st.get("pathway", ""))
        pwo = pathway_of(pw)
        if STANCE_ORIENTATION.get(lvl, 0.0) > 0 and pwo.actor_driven and not pwo.shared_process:
            pursuers.setdefault(pw, set()).add(str(st.get("actor")))
    contested = {pw for pw, actors in pursuers.items() if len(actors) >= 2}
    if not contested:
        return 0
    rate = sampled_coupling(world, "attrition_rate_per_day")
    drain = rate * float(elapsed_days)
    if drain <= 0:
        return 0
    now = at_ts if at_ts is not None else world.clock.now
    n = 0
    for pw in sorted(contested):
        for aid in sorted(pursuers[pw]):
            ent = (world.entities or {}).get(aid)
            if ent is None:
                continue
            cap = ent.value("resources", key="capacity", default=None)
            if not isinstance(cap, (int, float)):
                continue
            before = float(cap)
            after = max(0.05, before - drain)
            if abs(after - before) < 1e-9:
                continue
            ent.set("resources", F(round(after, 5), status="derived",
                                   method="contested_attrition_elapsed", updated_at=now),
                    key="capacity")
            n += 1
            if branch_log is not None:
                dd = StateDelta(at=now, event_type="interval_evolution",
                                operator="contested_attrition",
                                reason_codes=[f"elapsed_days={round(elapsed_days, 4)}",
                                              f"contested:{pw}"],
                                uncertainty={"rate_per_day": rate})
                branch_log.append(dd.change(f"{aid}.resources[capacity]",
                                            round(before, 5), round(after, 5)))
    return n


# ---------------------------------------------------------------- 2. persistence semantics
class PersistenceCheckOperator(TransitionOperator):
    """A provisional end-state either HOLDS its required window (the absorbing predicate is then
    genuinely satisfied — the monitor stamps first passage at the check, which IS when the
    criterion completes) or COLLAPSES: the named near-miss ("temporary ceasefire that collapses")
    happens as a real event, the mode's process is knocked back, and the world keeps running."""
    name = "persistence_check"

    def applicable(self, world, event):
        stamped = world.quantities.get("absorbed_at")
        prov = world.quantities.get("provisional_absorbing_mode")
        return event.etype == "persistence_check" and getattr(stamped, "value", None) in (None, 0) \
            and bool(getattr(prov, "value", None))

    def validate(self, world, proposal):
        return ValidationResult(ok=True)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name, action=dict(event.payload),
                                  reason_codes=["persistence_check"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.phase_consumers import _branch_rng
        a = proposal.action
        prov_mode = str(getattr(world.quantities.get("provisional_absorbing_mode"), "value", ""))
        if a.get("mode") and str(a["mode"]) != prov_mode:
            return None                                       # a stale check for a collapsed episode
        pw = str(a.get("pathway", ""))
        key = ("persistence_survival_shared" if pathway_of(pw).shared_process
               else "persistence_survival_default")
        p_hold = sampled_coupling(world, key)
        rng = _branch_rng(world, f"persist:{prov_mode}:{world.clock.now}")
        d = StateDelta(at=world.clock.now, event_type="persistence_check", operator=self.name,
                       reason_codes=[f"mode={prov_mode}"],
                       uncertainty={"p_hold": round(p_hold, 4), "coupling_key": key})
        if rng.random() < p_hold:                             # HOLDS → the criterion is satisfied NOW
            register_quantity_type("absorbing_state_reached", units="bool")
            register_quantity_type("absorbing_mode", units="mode")
            world.quantities["absorbing_state_reached"] = Quantity(
                name="absorbing_state_reached", qtype="absorbing_state_reached", value=True,
                timestamp=world.clock.now)
            world.quantities["absorbing_mode"] = Quantity(
                name="absorbing_mode", qtype="absorbing_mode", value=prov_mode,
                timestamp=world.clock.now)
            d.reason_codes.append("persisted_criterion_satisfied")
            return d.change("quantities[absorbing_state_reached]", None, True)
        # COLLAPSES — the near-miss realized: clear the provisional state, knock the process back
        world.quantities["provisional_absorbing_mode"] = Quantity(
            name="provisional_absorbing_mode", qtype="provisional_absorbing_mode", value=None,
            timestamp=world.clock.now)
        d.reason_codes.append("near_miss_realized_collapse")
        for var in (f"mode_progress:{pw}:{prov_mode}", f"pathway_progress:{pw}"):
            v = world.quantities.get(var)
            if v is not None and isinstance(v.value, (int, float)):
                before = float(v.value)
                after = max(0.05, before - 0.15)
                world.quantities[var] = Quantity(name=var, qtype=v.qtype, value=round(after, 4),
                                                 timestamp=world.clock.now)
                d.change(f"quantities[{var}]", round(before, 4), round(after, 4))
        # the mode's FIRST-PASSAGE process resumes: accumulated exposure preserved, fresh
        # threshold segment above it (memoryless continuation), crossing rescheduled as a
        # real follow-up event
        fp_pid = str(a.get("first_passage_process_id", ""))
        if fp_pid:
            from swm.world_model_v2.event_time import resume_first_passage_after_collapse
            st_fp, next_ts = resume_first_passage_after_collapse(world, fp_pid)
            if st_fp is not None and next_ts is not None and next_ts <= st_fp.horizon_ts:
                d.follow_up_events.append({
                    "etype": "first_passage", "ts": max(float(next_ts), world.clock.now),
                    "participants": [],
                    "payload": {**{k: v for k, v in st_fp.payload.items() if k != "spec"},
                                "spec": st_fp.payload.get("spec"),
                                "mode": prov_mode,
                                "hazard_process_id": fp_pid,
                                "hazard_generation": st_fp.generation}})
                d.reason_codes.append("first_passage_resumed_after_collapse")
        return d.change("quantities[provisional_absorbing_mode]", prov_mode, None)


register_operator("stance_review", StanceReviewOperator(), requires=("entities", "quantities"),
                  modifies=("entities",), temporal_scale="scheduled",
                  parameter_source="universal stance-update rules (ripeness/winning/exhaustion/"
                                   "bandwagon) over the mode graph's process state; one level per "
                                   "review with cooldown; every change logged", validated=True)
register_operator("persistence_check", PersistenceCheckOperator(),
                  requires=("quantities",), modifies=("quantities",), temporal_scale="scheduled",
                  parameter_source="documented persistence-survival priors (sampled per branch; "
                                   "fittable via coupling pack)", validated=True)

from swm.world_model_v2.events import event_type_registered, register_event_type  # noqa: E402
for _et in ("stance_review", "persistence_check"):
    if not event_type_registered(_et):
        register_event_type(_et, scheduling="scheduled", reads=("entities", "quantities"),
                            deltas=("entities", "quantities"),
                            parameter_source="world-dynamics layer", validated=True)


# ---------------------------------------------------------------- 3. capacity declaration
def declare_actor_capacity(plan) -> dict:
    """Initialize each stance-carrying actor's capacity resource from their grounded capability —
    capability becomes a live, depletable quantity instead of a static label. Idempotent; only
    touches entities that carry stances."""
    by_actor = {}
    for s in (getattr(plan, "_intention_stances", None) or []):
        caps = by_actor.setdefault(str(s.get("actor")), [])
        caps.append(str(s.get("capability", "high")).lower())
    declared = {}
    for e in (plan.entities or []):
        if not isinstance(e, dict) or str(e.get("id")) not in by_actor:
            continue
        caps = by_actor[str(e["id"])]
        best = max(caps, key=lambda c: CAPACITY_INIT.get(c, 0.6))
        fields = e.setdefault("fields", {})
        res = fields.setdefault("resources", {})
        if "capacity" not in res:
            res["capacity"] = CAPACITY_INIT.get(best, 0.6)
            declared[str(e["id"])] = res["capacity"]
    return {"initialized": declared, "cost_per_effortful_action": EFFORTFUL_ACTION_COST}
