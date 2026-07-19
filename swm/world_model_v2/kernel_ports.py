"""§23/§24 — validated legacy kernels ported as V2 TransitionOperators. Port, don't re-derive.

The mechanism audit found ~10 VALIDATED v1 kernels orphaned outside world_model_v2 while their
V2 registry entries sat dead (``poll_error_aggregation`` and ``whipcount_binomial`` are
experimental entries with an empty operator, rejected loudly by the compiler and pinned by
``tests/test_wmv2_tier_a_fixes.py``). This module ports FOUR of them behind the canonical
``applicable → propose → validate → apply → StateDelta`` contract, following the template the
repo already proved with ``poisson_arrival`` (RareEventArrivalOperator): the closed-form
probability is NEVER hardcoded — one draw per branch, the Monte Carlo emerges across particles.

The ports (each verified against its source module + original tests before porting):

  poll_error_aggregation   swm/api/mechanisms.py::sim_aggregation — latent share ~ Normal
                           (grounded share, empirical poll-error sd) vs threshold. kind =
                           measurement. A 52% lead with 4pt error is a lean, not a lock.
  whipcount_binomial       swm/api/mechanisms.py::sim_whipcount — committed yes plus the
                           undecideds that break yes vs the needed threshold; deterministic
                           when arithmetic already decides; Binomial otherwise. kind =
                           institution. NO INVENTED PROBABILITIES: the legacy kernel defaulted
                           ``lean=0.5``; the port REFUSES to run without grounded member
                           yes-probabilities (a rejection ValidationResult, never a default).
  outside_world_hazard     swm/transition/future_events.py FutureEvent calendar +
                           SurpriseHazard, re-expressed on the §5 residual contract: consumes
                           ``OutsideWorldProcess`` families (scheduled_exact = the dated-event
                           calendar; observed_base_rate / documented_broad_prior = the Poisson
                           surprise floor) and emits TYPED ENTRY EVENTS through the queue. kind
                           = exogenous. It NEVER writes terminal/readout paths — enforced
                           against ``outside_world.FORBIDDEN_WRITES``; the v1 belief-jump
                           impacts are deliberately NOT ported (a direct terminal write is
                           exactly what §5.1 forbids); outcome marks ride the entry payload.
  population_segment_exposure  swm/simulation/mean_field.py MeanFieldRollout — the repo's only
                           validated COUPLED population dynamic (conformity toward an
                           influence-weighted aggregate + social proof; non-separable,
                           EXP-053). kind = population. GROUNDED PARAMETERS ONLY: segments and
                           every coupling constant must arrive with provenance or the operator
                           rejects — the legacy dataclass defaults are never silently used.

Registered through ``transitions.register_operator`` and described by hand-authored
``MechanismSpec`` records (``mechanism_spec.register_ported_spec``) so fitted-vs-prior status,
event I/O, units and write sets are visible at the execution layer. The lean compiler entries
for the two dead mechanisms are NOT mutated here — the compiler keeps rejecting them until the
wiring decision is made deliberately (that rejection is pinned by tests); these operators are
reachable by name for engines that opt in, and the migration manifest records the state.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.world_model_v2 import mechanism_spec as _spec
from swm.world_model_v2.events import event_type_registered, register_event_type
from swm.world_model_v2.outside_world import (ENTRY_MECHANISMS, FORBIDDEN_WRITES,
                                              entry_event_payload, sample_arrivals)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            ValidationResult, register_operator)

# ---------------------------------------------------------------- event types for the ports
for _n, _kw in (("poll_error_aggregation", {"scheduling": "scheduled", "reads": ("quantities",),
                                            "deltas": ("quantities",)}),
                ("whipcount_binomial", {"scheduling": "scheduled",
                                        "reads": ("institutions", "quantities"),
                                        "deltas": ("quantities",)}),
                ("outside_world_window", {"scheduling": "scheduled", "reads": ("quantities",),
                                          "deltas": ("quantities", "event_queue")}),
                ("outside_world_arrival", {"scheduling": "hazard"}),
                ("population_segment_exposure", {"scheduling": "scheduled",
                                                 "reads": ("populations", "quantities"),
                                                 "deltas": ("quantities",)})):
    if not event_type_registered(_n):
        register_event_type(_n, validated=True, parameter_source="kernel_ports (§23)", **_kw)


def _write_bool_quantity(world, var: str, value, *, delta: StateDelta):
    register_quantity_type(var, units="bool")
    before = world.quantities[var].value if var in world.quantities else None
    world.quantities[var] = Quantity(name=var, qtype=var, value=value,
                                     timestamp=world.clock.now)
    delta.change(f"quantities[{var}]", before, value)


# ================================================================ (a) poll_error_aggregation
class PollErrorAggregationOperator(TransitionOperator):
    """v1 ``sim_aggregation`` as a V2 measurement mechanism (fills the dead registry entry's
    semantics). The validated kernel: an aggregate SHARE is latent Normal centered on the
    GROUNDED current/poll share with empirical poll-error sd (~3–6pt); YES iff the latent share
    beats the threshold. Ported per-branch: ONE latent draw per particle, outcome = pure
    threshold readout — across particles the legacy Monte Carlo (and the honesty property that
    a 52% lead with 4pt error is ~0.7, not 1.0) emerges without the closed form ever being
    hardcoded. The latent draw is deliberately UNCLAMPED, exactly as the validated kernel.

    Payload contract: ``aggregation_spec = {share, share_sd?, threshold=0.5, direction='>',
    outcome_var, provenance}``. The share must be grounded WITH provenance; a missing share or
    missing provenance is a rejection ValidationResult (the fallback hierarchy owns base rates
    — this operator never substitutes one). ``share_sd`` may be omitted: the 0.06 default is
    the kernel's DOCUMENTED empirical poll-error prior, labeled on the delta."""
    name = "poll_error_aggregation"
    DEFAULT_SHARE_SD = 0.06                     # empirical US poll error band (documented prior)

    def applicable(self, world, event):
        return event.etype == "poll_error_aggregation" and \
            isinstance(event.payload.get("aggregation_spec"), dict)

    def propose(self, world, event, rng):
        spec = event.payload["aggregation_spec"]
        reject = ""
        share = spec.get("share")
        provenance = str(spec.get("provenance", "")).strip()
        outcome_var = str(spec.get("outcome_var", "")).strip()
        if not isinstance(share, (int, float)):
            reject = "no grounded share — the operator never substitutes a base rate"
        elif not 0.0 <= float(share) <= 1.0:
            reject = f"share {share!r} outside [0,1]"
        elif not provenance:
            reject = "share has no provenance — a grounded poll/current share must name its source"
        elif not outcome_var:
            reject = "no outcome_var"
        sd = spec.get("share_sd")
        sd_defaulted = sd is None
        sd = self.DEFAULT_SHARE_SD if sd_defaulted else float(sd)
        if not reject and sd <= 0.0:
            reject = f"share_sd {sd!r} must be positive (zero error is a fabricated certainty)"
        action = {"outcome_var": outcome_var, "share": share, "share_sd": sd,
                  "threshold": float(spec.get("threshold", 0.5)),
                  "direction": str(spec.get("direction", ">")),
                  "provenance": provenance, "param_rejection": reject}
        reasons = ["poll_error_aggregation"]
        if sd_defaulted:
            reasons.append("share_sd_default_empirical_poll_error_prior_0.06")
        if not reject:                          # one latent draw per branch (kernel semantics)
            action["latent_share"] = rng.gauss(float(share), sd)
        return TransitionProposal(operator=self.name, action=action, reason_codes=reasons)

    def validate(self, world, proposal):
        if proposal.action.get("param_rejection"):
            return ValidationResult(ok=False, reasons=[proposal.action["param_rejection"]])
        return super().validate(world, proposal)

    def apply(self, world, proposal):
        a = proposal.action
        s = a["latent_share"]
        outcome = (s > a["threshold"]) if a["direction"] == ">" else (s < a["threshold"])
        d = StateDelta(at=world.clock.now, event_type="poll_error_aggregation",
                       operator=self.name,
                       reason_codes=list(proposal.reason_codes) + [f"share_source={a['provenance']}"[:80]],
                       uncertainty={"share": a["share"], "share_sd": a["share_sd"],
                                    "latent_draw": round(s, 6), "observation_model":
                                    "latent_share ~ Normal(grounded_share, poll_error_sd)"})
        _write_bool_quantity(world, a["outcome_var"], bool(outcome), delta=d)
        var = f"poll_latent_share:{a['outcome_var']}"
        register_quantity_type(var, units="share")
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=float(s),
                                         timestamp=world.clock.now)
        d.change(f"quantities[{var}]", before, round(float(s), 6))
        return d


# ================================================================ (b) whipcount_binomial
class WhipcountBinomialOperator(TransitionOperator):
    """v1 ``sim_whipcount`` as a V2 institution mechanism (fills the second dead entry's
    semantics): YES iff committed-yes plus the undecideds that break yes reach the needed
    threshold. Preserved semantics, verified against tests/test_mechanisms.py:
      * ``needed`` defaults to a bare majority of ``total`` (total/2 + 0.5);
      * SHORT-CIRCUIT when arithmetic already decides — committed_yes >= needed is a
        deterministic YES, committed_yes + undecided < needed a deterministic NO (the legacy
        0.98/0.02 were output clipping, not mechanism content; per-branch the outcome is the
        deterministic fact and the probability emerges across particles);
      * otherwise each undecided breaks yes independently — per-member Bernoulli, i.e. the
        legacy Binomial(undecided, lean), or heterogeneous ``member_yes_probabilities``.

    THE GUARDRAIL THE LEGACY KERNEL LACKED: ``lean`` defaulted to 0.5 there. Here, when the
    outcome actually depends on how undecideds break, the plan/pack MUST supply ``lean`` or
    ``member_yes_probabilities`` WITH provenance — otherwise the operator returns a rejection
    ValidationResult. An invented break probability is minted human behavior; refusing is the
    condition under which a count mechanism over member decisions is admissible at all.
    Conservation: when ``total`` is declared, committed_yes + committed_no + undecided may not
    exceed it. Payload: ``whipcount_spec = {committed_yes, committed_no?, undecided, needed?,
    total?, lean?, member_yes_probabilities?, outcome_var, provenance: {counts, lean?}}``."""
    name = "whipcount_binomial"

    def applicable(self, world, event):
        return event.etype == "whipcount_binomial" and \
            isinstance(event.payload.get("whipcount_spec"), dict)

    @staticmethod
    def _params(spec: dict):
        cy = int(spec.get("committed_yes", 0))
        cn = int(spec.get("committed_no", 0))
        und = int(spec.get("undecided", 0))
        needed, total = spec.get("needed"), spec.get("total")
        if needed is None and total is not None:
            needed = float(total) / 2.0 + 0.5              # bare majority (legacy semantics)
        return cy, cn, und, needed, total

    def propose(self, world, event, rng):
        spec = event.payload["whipcount_spec"]
        prov = spec.get("provenance") if isinstance(spec.get("provenance"), dict) else {}
        cy, cn, und, needed, total = self._params(spec)
        outcome_var = str(spec.get("outcome_var", "")).strip()
        reject, mode, breaks = "", "", None
        member_ps = spec.get("member_yes_probabilities")
        lean = spec.get("lean")
        if not outcome_var:
            reject = "no outcome_var"
        elif needed is None:
            reject = "no needed threshold and no total to derive a bare majority from — " \
                     "the operator never invents an institutional rule"
        elif not str(prov.get("counts", "")).strip():
            reject = "whip counts have no provenance — grounded counts must name their source"
        elif total is not None and cy + cn + und > int(total):
            reject = (f"conservation violated: committed_yes({cy}) + committed_no({cn}) + "
                      f"undecided({und}) > total({total})")
        elif cy >= needed:
            mode = "arithmetic_decides_yes"                # short-circuit (legacy semantics)
        elif cy + und < needed:
            mode = "arithmetic_decides_no"
        elif isinstance(member_ps, (list, tuple)) and len(member_ps) == und:
            if not str(prov.get("lean", "")).strip():
                reject = "member_yes_probabilities have no provenance — a break probability " \
                         "without a named source is an invented probability; REFUSED"
            elif not all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in member_ps):
                reject = "member_yes_probabilities must be probabilities in [0,1]"
            else:
                mode = "binomial_member_breaks"
                breaks = sum(1 for p in member_ps if rng.random() < float(p))
        elif isinstance(lean, (int, float)):
            if not str(prov.get("lean", "")).strip():
                reject = "lean has no provenance — the legacy default lean=0.5 is exactly the " \
                         "invented probability this port refuses; supply a grounded lean or " \
                         "member_yes_probabilities from the plan/pack"
            elif not 0.0 <= float(lean) <= 1.0:
                reject = f"lean {lean!r} outside [0,1]"
            else:
                mode = "binomial_lean_breaks"
                breaks = sum(1 for _ in range(und) if rng.random() < float(lean))
        else:
            reject = "outcome depends on how undecideds break, but the plan supplies no " \
                     "lean and no member_yes_probabilities — REFUSED (no invented " \
                     "probabilities, never a default)"
        action = {"outcome_var": outcome_var, "committed_yes": cy, "committed_no": cn,
                  "undecided": und, "needed": needed, "total": total, "mode": mode,
                  "breaks": breaks, "provenance": dict(prov), "param_rejection": reject}
        return TransitionProposal(operator=self.name, action=action,
                                  reason_codes=["whipcount_binomial"] + ([mode] if mode else []))

    def validate(self, world, proposal):
        if proposal.action.get("param_rejection"):
            return ValidationResult(ok=False, reasons=[proposal.action["param_rejection"]])
        return super().validate(world, proposal)

    def apply(self, world, proposal):
        a = proposal.action
        if a["mode"] == "arithmetic_decides_yes":
            outcome, tally_yes = True, a["committed_yes"]
        elif a["mode"] == "arithmetic_decides_no":
            outcome, tally_yes = False, a["committed_yes"]
        else:
            tally_yes = a["committed_yes"] + int(a["breaks"])
            outcome = tally_yes >= a["needed"]
        d = StateDelta(at=world.clock.now, event_type="whipcount_binomial", operator=self.name,
                       reason_codes=list(proposal.reason_codes) +
                       [f"counts_source={a['provenance'].get('counts', '')}"[:80]],
                       uncertainty={"committed_yes": a["committed_yes"],
                                    "undecided": a["undecided"], "needed": a["needed"],
                                    "breaks": a["breaks"], "tally_yes": tally_yes,
                                    "mode": a["mode"]})
        _write_bool_quantity(world, a["outcome_var"], bool(outcome), delta=d)
        return d


# ================================================================ (c) outside_world_hazard
#: the arrival kinds THIS operator consumes: the dated calendar (v1 FutureEvent) and the
#: base-rate surprise floor (v1 SurpriseHazard). fitted_hazard / state_dependent /
#: grounded_scenario_data families belong to the temporal-hazard machinery, not this operator.
CONSUMED_ARRIVAL_KINDS = ("scheduled_exact", "observed_base_rate", "documented_broad_prior")


def _forbidden_write_hit(family) -> str:
    """Pure (non-mutating) re-check of outside_world.validate_entry's §5.1 scan: the forbidden
    token a family's declared impact would write, or ''. Defense in depth — families are
    expected to arrive validated, but this operator refuses to trust that."""
    lowered = " ".join(str(c).lower() for c in family.affected_boundary_components) + " " + \
        str(family.impact_description).lower()
    for bad in FORBIDDEN_WRITES:
        if bad in lowered.replace(" ", "_") or bad.replace("_", " ") in lowered:
            return bad
    return ""


def assert_no_forbidden_paths(delta: StateDelta):
    """Final §5.1 guard on the operator's own writes: no change path may name a terminal/
    readout target. Raises — an outside-world write to the answer is a contract violation,
    never a degradable warning."""
    for ch in delta.changes:
        p = str(ch.get("path", "")).lower()
        for bad in FORBIDDEN_WRITES:
            if bad in p:
                raise ValueError(f"outside_world_hazard wrote forbidden path {p!r} ({bad})")
    return delta


class OutsideWorldHazardOperator(TransitionOperator):
    """The residual outside world as a V2 exogenous mechanism — v1's FutureEvent calendar +
    SurpriseHazard re-expressed on the §5 ``OutsideWorldProcess`` contract. On an
    ``outside_world_window`` event carrying the process and a window ``[now/window_start_ts,
    window_end_ts)``, it samples arrivals for every samplable family whose kind this operator
    consumes (``CONSUMED_ARRIVAL_KINDS``) with the branch's own rng:

      * ``scheduled_exact``       → the dated-event calendar (deterministic inclusion of the
                                    scheduled times in-window; v1 ``EventCalendar.scheduled_in``);
      * ``observed_base_rate`` /
        ``documented_broad_prior``→ Poisson arrivals via exponential inter-arrival sampling
                                    (``outside_world.sample_arrivals``; a documented rate BAND
                                    becomes between-branch spread — v1 ``SurpriseHazard`` whose
                                    rate now requires provenance instead of a hand-set float).

    Each arrival becomes ONE typed follow-up event (``outside_world_arrival``) whose payload is
    ``outside_world.entry_event_payload`` — entry mechanism, mark from the family's mark space,
    provenance — queued by the engine and routed through the generated world's control plane.
    The operator's OWN writes are bookkeeping counts only (``outside_world_arrivals:<family>``);
    it NEVER writes terminal/readout paths (checked three ways: family-level §5.1 scan at
    validate, entry-mechanism whitelist, and ``assert_no_forbidden_paths`` on the final delta).
    Unresolved families are NEVER sampled — they surface as reason codes and uncertainty, per
    §5.2. The v1 belief-jump ``impact`` term is intentionally NOT ported: a jump applied
    directly to the tracked belief is a terminal write."""
    name = "outside_world_hazard"

    def applicable(self, world, event):
        proc = event.payload.get("outside_world")
        return event.etype == "outside_world_window" and hasattr(proc, "families") \
            and isinstance(event.payload.get("window_end_ts"), (int, float))

    def propose(self, world, event, rng):
        proc = event.payload["outside_world"]
        t0 = float(event.payload.get("window_start_ts", world.clock.now))
        t0 = max(t0, world.clock.now)
        t1 = float(event.payload["window_end_ts"])
        reasons, follow_ups, sampled, rejected = ["outside_world_hazard"], [], {}, []
        for fam in proc.families:
            if fam.validation_error or fam.arrival.kind == "unresolved":
                reasons.append(f"unresolved_family_never_sampled:{fam.family_id}"[:80])
                continue
            if fam.arrival.kind not in CONSUMED_ARRIVAL_KINDS:
                reasons.append(f"family_kind_delegated:{fam.family_id}:{fam.arrival.kind}"[:80])
                continue
            if fam.impact_mechanism not in ENTRY_MECHANISMS:
                rejected.append(f"family {fam.family_id}: impact_mechanism "
                                f"{fam.impact_mechanism!r} is not a typed entry mechanism")
                continue
            bad = _forbidden_write_hit(fam)
            if bad:
                rejected.append(f"family {fam.family_id} targets forbidden write {bad!r} — "
                                f"an outside-world event may not write the answer (§5.1)")
                continue
            times = sample_arrivals(fam, t0=t0, t1=t1, rng=rng)
            sampled[fam.family_id] = {"n": len(times), "kind": fam.arrival.kind,
                                      "provenance": fam.arrival.provenance}
            for i, ts in enumerate(times):
                follow_ups.append({"etype": "outside_world_arrival", "ts": float(ts),
                                   "participants": [],
                                   "payload": entry_event_payload(
                                       fam, at=float(ts), branch_id=world.branch_id,
                                       arrival_index=i, rng=rng)})
        return TransitionProposal(
            operator=self.name,
            action={"sampled": sampled, "t0": t0, "t1": t1, "rejected_families": rejected},
            reason_codes=reasons, follow_up_events=follow_ups,
            uncertainty={"unresolved_families": [f.family_id for f in proc.unresolved()]})

    def validate(self, world, proposal):
        if proposal.action.get("rejected_families"):
            return ValidationResult(ok=False, reasons=proposal.action["rejected_families"])
        return super().validate(world, proposal)

    def apply(self, world, proposal):
        a = proposal.action
        d = StateDelta(at=world.clock.now, event_type="outside_world_window",
                       operator=self.name, reason_codes=list(proposal.reason_codes),
                       uncertainty={"window": [a["t0"], a["t1"]], "families": a["sampled"],
                                    **proposal.uncertainty})
        for fid, rec in a["sampled"].items():
            var = f"outside_world_arrivals:{fid}"
            register_quantity_type(var, units="count", lo=0)
            before = world.quantities[var].value if var in world.quantities else 0
            after = int(before or 0) + int(rec["n"])
            world.quantities[var] = Quantity(name=var, qtype=var, value=after,
                                             timestamp=world.clock.now)
            d.change(f"quantities[{var}]", before, after)
        return assert_no_forbidden_paths(d)


# ============================================================ (d) population_segment_exposure
_REQUIRED_COUPLINGS = ("k_social", "k_event", "k_proof", "proof_center")


# --- verbatim numerical port of swm/simulation/mean_field.py (EXP-053). The V2 import
# boundary (pinned by tests/test_world_model_v2.py::test_import_boundary_ast_v2_never_
# imports_legacy) forbids world_model_v2 from importing swm.simulation — which is exactly why
# the audit found the kernel orphaned. §23 therefore PORTS the numerics line-for-line;
# tests/test_mechanism_spec.py imports the legacy module (tests are outside the boundary) and
# pins numerical identity on a fixed case, so the two implementations cannot drift silently.
def _mf_clamp(x, lo=1e-4, hi=1 - 1e-4):
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class _MeanFieldAgent:
    belief: float                 # current p in [0,1]
    responsiveness: float = 0.3   # how much they move per step
    influence: float = 1.0        # how hard they pull the aggregate (opinion-leader weight)


def _mf_agents_from_cells(cells) -> list:
    return [_MeanFieldAgent(belief=_mf_clamp(b), responsiveness=max(0.0, min(1.0, r)),
                            influence=max(1e-3, w)) for b, r, w in cells]


def _mf_aggregate(agents) -> float:
    w = sum(a.influence for a in agents) or 1.0
    return sum(a.influence * a.belief for a in agents) / w


def _mf_step(agents, event_impact, *, k_social, k_event, k_proof, proof_center) -> float:
    agg = _mf_aggregate(agents)                     # depends on ALL agents (the coupling)
    proof = (agg - proof_center)                    # bandwagon toward adoption / majority
    for a in agents:
        pull = (k_social * (agg - a.belief)         # conformity toward the evolving aggregate
                + k_event * event_impact            # exogenous shock
                + k_proof * proof)                  # social proof: prevalence begets adoption
        a.belief = _mf_clamp(a.belief + a.responsiveness * pull)
    return agg


def _mf_roll(agents, steps, events, *, k_social, k_event, k_proof, proof_center):
    traj = []
    for t in range(steps):
        ev = 0.0 if not events else (events[t] if t < len(events) else events[-1])
        traj.append(_mf_step(agents, ev, k_social=k_social, k_event=k_event,
                             k_proof=k_proof, proof_center=proof_center))
    traj.append(_mf_aggregate(agents))
    return traj, traj[-1]


class PopulationSegmentExposureOperator(TransitionOperator):
    """v1 ``mean_field.MeanFieldRollout`` as a V2 population mechanism: explicit segments'
    beliefs evolve TOWARD the influence-weighted aggregate (conformity), driven by exogenous
    event impacts and a social-proof self-reinforcement term — non-separable by construction
    (∂pᵢ/∂pⱼ ≠ 0; EXP-053), which neither phase9's compositional posterior nor the stance-level
    bandwagon provides within-horizon. The V2 import boundary forbids importing
    ``swm.simulation`` (that ban is why the kernel was orphaned), so the port carries the
    numerics VERBATIM (``_mf_*`` above, a line-for-line port of ``MeanFieldRollout``) and a
    test imports the legacy module from tests/ to pin numerical identity on a fixed case.

    GROUNDED PARAMETERS ONLY: segments must carry provenance (survey/census cells, a
    demographic pack — a named source), and EVERY coupling constant (k_social, k_event,
    k_proof, proof_center) must arrive as ``{"value": v, "source": <named provenance>}`` — the
    legacy dataclass defaults were priors baked into code; here an unsourced coupling is a
    rejection ValidationResult, never a silent default (the sampled_coupling / fitted-pack
    doctrine of world_dynamics applies). Deterministic given parameters (the kernel has no
    randomness); parameter uncertainty enters as between-branch spread when the caller samples
    couplings per branch. Payload: ``mean_field_spec = {segments: [{belief, responsiveness,
    influence}], couplings: {name: {value, source}}, steps, event_impacts?, outcome_var,
    provenance: {segments}}``. Writes the final aggregate plus per-segment evolved beliefs as
    typed share quantities — wiring evolved beliefs back into a ``Population`` object is the
    caller's explicit join, not a hidden side effect."""
    name = "population_segment_exposure"

    def applicable(self, world, event):
        return event.etype == "population_segment_exposure" and \
            isinstance(event.payload.get("mean_field_spec"), dict)

    @staticmethod
    def _check(spec: dict) -> str:
        segments = spec.get("segments")
        prov = spec.get("provenance") if isinstance(spec.get("provenance"), dict) else {}
        if not isinstance(segments, (list, tuple)) or not segments:
            return "no segments — a population mechanism needs explicit segments"
        if not str(prov.get("segments", "")).strip():
            return "segments have no provenance — grounded cells must name their source " \
                   "(survey/census rows, a demographic pack); REFUSED"
        for seg in segments:
            if not isinstance(seg, dict) or not isinstance(seg.get("belief"), (int, float)):
                return "each segment needs a numeric belief (plus responsiveness, influence)"
        couplings = spec.get("couplings") if isinstance(spec.get("couplings"), dict) else {}
        for name in _REQUIRED_COUPLINGS:
            c = couplings.get(name)
            if not isinstance(c, dict) or not isinstance(c.get("value"), (int, float)):
                return f"coupling {name!r} missing — the legacy default is a prior baked in " \
                       f"code; every coupling must be supplied explicitly"
            if not str(c.get("source", "")).strip():
                return f"coupling {name!r} has no provenance — an unsourced coupling " \
                       f"constant is an invented parameter; REFUSED"
        unknown = sorted(set(couplings) - set(_REQUIRED_COUPLINGS))
        if unknown:
            return f"unknown couplings {unknown} — the validated kernel has exactly " \
                   f"{list(_REQUIRED_COUPLINGS)}"
        if not str(spec.get("outcome_var", "")).strip():
            return "no outcome_var"
        if int(spec.get("steps", 0)) < 1:
            return "steps must be >= 1"
        return ""

    def propose(self, world, event, rng):
        spec = event.payload["mean_field_spec"]
        reject = self._check(spec)
        action = {"param_rejection": reject}
        if not reject:
            action = {"param_rejection": "",
                      "segments": [dict(s) for s in spec["segments"]],
                      "couplings": {k: dict(v) for k, v in spec["couplings"].items()},
                      "steps": int(spec["steps"]),
                      "event_impacts": [float(x) for x in spec.get("event_impacts") or []],
                      "outcome_var": str(spec["outcome_var"]),
                      "provenance": dict(spec.get("provenance") or {})}
        return TransitionProposal(operator=self.name, action=action,
                                  reason_codes=["population_segment_exposure",
                                                "coupled_mean_field_nonseparable"])

    def validate(self, world, proposal):
        if proposal.action.get("param_rejection"):
            return ValidationResult(ok=False, reasons=[proposal.action["param_rejection"]])
        return super().validate(world, proposal)

    def apply(self, world, proposal):
        a = proposal.action
        cells = [(float(s["belief"]), float(s.get("responsiveness", 0.3)),
                  float(s.get("influence", 1.0))) for s in a["segments"]]
        agents = _mf_agents_from_cells(cells)
        before_beliefs = [ag.belief for ag in agents]
        cp = {k: float(v["value"]) for k, v in a["couplings"].items()}
        traj, final = _mf_roll(agents, a["steps"], a["event_impacts"] or None,
                               k_social=cp["k_social"], k_event=cp["k_event"],
                               k_proof=cp["k_proof"], proof_center=cp["proof_center"])
        var = a["outcome_var"]
        register_quantity_type(var, units="share", lo=0.0, hi=1.0)
        before = world.quantities[var].value if var in world.quantities else None
        world.quantities[var] = Quantity(name=var, qtype=var, value=float(final),
                                         timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="population_segment_exposure",
                       operator=self.name,
                       reason_codes=list(proposal.reason_codes) +
                       [f"segments_source={a['provenance'].get('segments', '')}"[:80]] +
                       [f"coupling:{k}={v['value']}({v['source']})"[:80]
                        for k, v in sorted(a["couplings"].items())],
                       uncertainty={"trajectory": [round(x, 6) for x in traj],
                                    "kernel": "verbatim port of swm.simulation.mean_field."
                                              "MeanFieldRollout (EXP-053; identity pinned "
                                              "by test)"})
        d.change(f"quantities[{var}]", before, round(float(final), 6))
        for i, ag in enumerate(agents):
            svar = f"segment_belief:{var}:{i}"
            register_quantity_type(svar, units="share", lo=0.0, hi=1.0)
            sbefore = world.quantities[svar].value if svar in world.quantities \
                else round(before_beliefs[i], 6)
            world.quantities[svar] = Quantity(name=svar, qtype=svar, value=float(ag.belief),
                                              timestamp=world.clock.now)
            d.change(f"quantities[{svar}]", sbefore, round(float(ag.belief), 6))
        return d


# ---------------------------------------------------------------- registration (operators)
register_operator("poll_error_aggregation", PollErrorAggregationOperator(),
                  requires=("quantities",), modifies=("quantities",),
                  temporal_scale="scheduled",
                  parameter_source="grounded poll/current share (provenance required) + "
                                   "empirical poll-error sd (documented prior 3-6pt, "
                                   "default 0.06; v1 sim_aggregation)",
                  invariants=("share in [0,1] with named provenance",
                              "share_sd > 0 (zero error is fabricated certainty)",
                              "one latent draw per branch — closed form never hardcoded"),
                  validated=True)
register_operator("whipcount_binomial", WhipcountBinomialOperator(),
                  requires=("institutions", "quantities"), modifies=("quantities",),
                  temporal_scale="scheduled",
                  parameter_source="grounded whip counts + grounded lean/member probabilities "
                                   "from plan/pack (v1 sim_whipcount); NO default lean — "
                                   "missing probabilities are a rejection, never 0.5",
                  invariants=("committed_yes + committed_no + undecided <= total when total "
                              "declared",
                              "short-circuit only when arithmetic decides",
                              "no invented break probabilities"),
                  validated=True)
register_operator("outside_world_hazard", OutsideWorldHazardOperator(),
                  requires=("quantities",), modifies=("quantities", "event_queue"),
                  temporal_scale="interval",
                  parameter_source="OutsideWorldProcess families: scheduled_exact calendar / "
                                   "observed_base_rate / documented_broad_prior rates with "
                                   "named provenance (v1 FutureEvent + SurpriseHazard); "
                                   "unresolved families never sampled",
                  invariants=("never writes terminal/readout paths (FORBIDDEN_WRITES)",
                              "arrivals enter only through typed entry mechanisms",
                              "unresolved families are surfaced, never sampled"),
                  validated=True)
register_operator("population_segment_exposure", PopulationSegmentExposureOperator(),
                  requires=("populations", "quantities"), modifies=("quantities",),
                  temporal_scale="interval",
                  parameter_source="grounded segment cells + explicitly sourced couplings "
                                   "(documented priors / sampled_coupling / fitted pack; v1 "
                                   "mean_field EXP-053); unsourced params rejected",
                  invariants=("beliefs clamped to [1e-4, 1-1e-4] exactly as the validated "
                              "kernel",
                              "influence weights read-only (aggregate is influence-weighted)",
                              "every coupling carries provenance"),
                  validated=True)

# ------------------------------------------------------- spec-layer metadata for the ports
_spec.OPERATOR_KINDS.update({"poll_error_aggregation": "measurement",
                             "whipcount_binomial": "institution",
                             "outside_world_hazard": "exogenous",
                             "population_segment_exposure": "population"})
_spec.OPERATOR_EVENT_INPUTS.update({"poll_error_aggregation": ("poll_error_aggregation",),
                                    "whipcount_binomial": ("whipcount_binomial",),
                                    "outside_world_hazard": ("outside_world_window",),
                                    "population_segment_exposure":
                                        ("population_segment_exposure",)})
_spec.OPERATOR_EVENT_OUTPUTS.update({"outside_world_hazard": ("outside_world_arrival",)})

# ---------------------------------------------------------------- registration (specs, §22)
_spec.register_ported_spec(_spec.MechanismSpec(
    mechanism_id="poll_error_aggregation", version="1.0.0", mechanism_kind="measurement",
    causal_role="a latent aggregate share, measured with empirical poll error, decides a "
                "threshold outcome — measurement error integrated, never assumed away",
    required_state=("quantities",),
    read_set=("quantities",), write_set=("quantities[",),
    event_inputs=("poll_error_aggregation",), event_outputs=(),
    temporal_behavior={"scale": "scheduled",
                       "semantics": "one latent-share draw per branch; the legacy Monte Carlo "
                                    "emerges across particles"},
    parameter_schema=({"name": "share", "description": "grounded current/poll share",
                       "lo": 0.0, "hi": 1.0, "source": "observed"},
                      {"name": "share_sd", "description": "empirical poll-error sd "
                       "(~0.03-0.06; default 0.06 documented prior)", "lo": 0.0, "hi": 0.2,
                       "source": "published_research"},
                      {"name": "threshold", "description": "decision threshold (default 0.5)",
                       "lo": 0.0, "hi": 1.0, "source": "observed"}),
    parameter_sources=("observed", "published_research"),
    units={"<outcome_var>": "bool", "poll_latent_share:<outcome_var>": "share"},
    conservation_rules=(),
    validation_rules=("share requires named provenance", "share_sd > 0",
                      "latent draw unclamped exactly as the validated kernel"),
    calibration_status="documented_prior", domains=("*",),
    known_limits=("share must be a grounded poll/current share — never an LLM-minted number",
                  "0.06 default sd is a US-poll empirical band; recalibrate per domain",
                  "single latent Normal — no herding/correlated-poll-error model"),
    operator="poll_error_aggregation"))

_spec.register_ported_spec(_spec.MechanismSpec(
    mechanism_id="whipcount_binomial", version="1.0.0", mechanism_kind="institution",
    causal_role="committed yes votes plus undecideds breaking at a grounded lean, against a "
                "declared threshold — a count-based institutional aggregate for members not "
                "individually modeled",
    required_state=("institutions", "quantities"),
    read_set=("institutions", "quantities"), write_set=("quantities[",),
    event_inputs=("whipcount_binomial",), event_outputs=(),
    temporal_behavior={"scale": "scheduled",
                       "semantics": "deterministic when arithmetic decides; per-member "
                                    "Bernoulli draw per branch otherwise"},
    parameter_schema=({"name": "committed_yes", "description": "grounded committed yes count",
                       "lo": 0, "hi": None, "source": "observed"},
                      {"name": "undecided", "description": "grounded undecided count",
                       "lo": 0, "hi": None, "source": "observed"},
                      {"name": "needed", "description": "votes needed (or bare majority of "
                       "total)", "lo": 0, "hi": None, "source": "observed"},
                      {"name": "lean", "description": "P(an undecided breaks yes) — REQUIRED "
                       "with provenance when the outcome depends on it; never defaulted",
                       "lo": 0.0, "hi": 1.0, "source": "inferred_from_data"}),
    parameter_sources=("observed", "inferred_from_data"),
    units={"<outcome_var>": "bool"},
    conservation_rules=("committed_yes + committed_no + undecided <= total when total "
                        "declared",),
    validation_rules=("missing lean/member probabilities => rejection ValidationResult",
                      "short-circuit only when arithmetic decides",
                      "provenance required for counts and lean"),
    calibration_status="grounded_scenario", domains=("*",),
    known_limits=("undecideds exchangeable within the lean unless member_yes_probabilities "
                  "supplied",
                  "no vote-trading/whip-pressure dynamics — a count mechanism, not bargaining",
                  "complements institutional_vote (which needs individually simulated "
                  "members)"),
    operator="whipcount_binomial"))

_spec.register_ported_spec(_spec.MechanismSpec(
    mechanism_id="outside_world_hazard", version="1.0.0", mechanism_kind="exogenous",
    causal_role="the residual outside world: dated calendar arrivals plus base-rate surprise "
                "arrivals entering the boundary ONLY through typed entry mechanisms",
    required_state=("quantities",),
    read_set=("quantities",), write_set=("quantities[outside_world_arrivals:",),
    event_inputs=("outside_world_window",), event_outputs=("outside_world_arrival",),
    temporal_behavior={"scale": "interval",
                       "semantics": "scheduled_exact = deterministic dated calendar (v1 "
                                    "FutureEvent); observed_base_rate/documented_broad_prior "
                                    "= exponential inter-arrival Poisson sampling (v1 "
                                    "SurpriseHazard); documented rate bands become "
                                    "between-branch spread"},
    parameter_schema=({"name": "rate_per_day", "description": "Poisson intensity — "
                       "defensible kinds with named provenance only", "lo": 0.0, "hi": None,
                       "source": "observed"},
                      {"name": "scheduled_times", "description": "exact arrival timestamps "
                       "for scheduled_exact families", "lo": None, "hi": None,
                       "source": "observed"},
                      {"name": "uncertainty_band", "description": "documented [lo, hi] rate "
                       "band (log-uniform per branch when wide)", "lo": 0.0, "hi": None,
                       "source": "published_research"}),
    parameter_sources=("observed", "published_research", "reference_class_prior"),
    units={"outside_world_arrivals:<family_id>": "count"},
    conservation_rules=("never writes terminal/readout paths (outside_world."
                        "FORBIDDEN_WRITES) — arrivals enter only through the nine typed "
                        "entry mechanisms",),
    validation_rules=("unresolved families are never sampled (§5.2)",
                      "family-level §5.1 forbidden-write scan at validate",
                      "assert_no_forbidden_paths on the final delta"),
    calibration_status="documented_prior", domains=("*",),
    known_limits=("v1 belief-jump impacts NOT ported — a direct jump on the tracked belief "
                  "is a forbidden terminal write; outcome marks ride the entry payload",
                  "fitted_hazard/state_dependent/grounded_scenario_data families are "
                  "delegated to the temporal-hazard machinery",
                  "EXP-077 calibrated jump-magnitude model not yet joined (manifest "
                  "follow-up)"),
    operator="outside_world_hazard"))

_spec.register_ported_spec(_spec.MechanismSpec(
    mechanism_id="population_segment_exposure", version="1.0.0", mechanism_kind="population",
    causal_role="explicit population segments' beliefs coupled through an influence-weighted "
                "aggregate (conformity + social proof + exogenous shocks) — the non-separable "
                "within-horizon population dynamic",
    required_state=("populations", "quantities"),
    read_set=("populations", "quantities"), write_set=("quantities[",),
    event_inputs=("population_segment_exposure",), event_outputs=(),
    temporal_behavior={"scale": "interval",
                       "semantics": "deterministic coupled roll-forward over declared steps; "
                                    "parameter uncertainty enters as between-branch spread "
                                    "via caller-sampled couplings"},
    parameter_schema=({"name": "k_social", "description": "conformity strength toward the "
                       "aggregate", "lo": 0.0, "hi": 1.0, "source": "reference_class_prior"},
                      {"name": "k_event", "description": "exogenous event gain", "lo": 0.0,
                       "hi": None, "source": "reference_class_prior"},
                      {"name": "k_proof", "description": "social-proof self-reinforcement "
                       "(nonlinear; 0 = off)", "lo": 0.0, "hi": 1.0,
                       "source": "reference_class_prior"},
                      {"name": "proof_center", "description": "0.5 = conformity-to-majority; "
                       "0.0 = bandwagon adoption", "lo": 0.0, "hi": 1.0,
                       "source": "reference_class_prior"}),
    parameter_sources=("observed", "reference_class_prior"),
    units={"<outcome_var>": "share", "segment_belief:<outcome_var>:<i>": "share"},
    conservation_rules=("beliefs clamped to [1e-4, 1-1e-4] exactly as the validated kernel",
                        "influence weights read-only — never rescaled"),
    validation_rules=("segments require named provenance",
                      "every coupling requires {value, source} — unsourced params rejected",
                      "verbatim-port numerics identical to swm/simulation/mean_field.py "
                      "(pinned by numerical-equality test)"),
    calibration_status="documented_prior", domains=("*",),
    known_limits=("validated against beating the independent mean (EXP-053), not as a fitted "
                  "opinion model",
                  "couplings are documented priors until a fitted pack exists",
                  "population-level belief coupling — not individual cognition (Phase-4 "
                  "policy owns that)"),
    operator="population_segment_exposure"))
