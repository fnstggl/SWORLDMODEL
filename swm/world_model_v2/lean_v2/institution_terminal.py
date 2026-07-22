"""The institution-vote terminal law: compose D7 (faithful representation) + D8 (grounded initial
positions) + D14 (deliberative convergence + seat-weighted tally) into P(YES).

This REPLACES "count independent per-member votes against a (rescaled) threshold" — the source of
the EXP-113 anti-consensus failure and the roster-collapse rescaling — with a deliberative
sub-simulation run PER SHARED WORLD:

    for each shared-world combo (weighted by its counted probability):
        faithful roster (D7)  ->  each voter's initial support for the target from the counted
        action baseline conditional on that world (D8)  ->  bounded grounded deliberation and a
        seat-weighted tally against the REAL threshold (D14/D7)  ->  P(YES | world)
    P(YES) = Σ  weight(world) · P(YES | world)

Nothing is rescaled; nothing is invented. A world with no grounded convergence force deliberates
as independent voters (the honest baseline). Universal — no question-specific logic."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import norm_key
from swm.world_model_v2.lean_v2.institution_deliberation import (classify_institution,
                                                                 resolve_institution_vote)

INSTITUTION_TERMINAL_VERSION = "lean_v2.institution_terminal.v1"


def _target_support_from_states(states, weights_mid: dict, target_option: str,
                                feasible_options: list) -> float:
    """Support for the target option = the counted (D8) weight mass on this actor's states whose
    action tendency IS the target option. Typed via canonical options; no prose guessing."""
    if not states:
        return None
    from swm.world_model_v2.lean_v2.canonical_options import normalize_option
    tgt = normalize_option(target_option, feasible_options) if (target_option and feasible_options) \
        else None
    support, total = 0.0, 0.0
    for h in states:
        w = float(weights_mid.get(h.state_id, 0.0))
        total += w
        tend = (getattr(h, "expected_action_tendency", "")
                or getattr(h, "action_if_state", "")).strip()
        matches = False
        if tgt is not None and feasible_options:
            c = normalize_option(tend, feasible_options)
            matches = c is not None and c.canonical_option_id == tgt.canonical_option_id
        elif target_option:
            matches = norm_key(tend) == norm_key(target_option)
        if matches:
            support += w
    if total <= 0:
        return None
    return support / total


def resolve_institution_terminal(bp, representation, grounding: dict, *, states_by_actor: dict,
                                 gw_by_combo: dict, shared_combos: list, feasible_options_by_actor:
                                 dict = None, target_option: str = "", max_rounds: int = 8) -> dict:
    """Deliberative institution-vote resolution composing D7+D8+D14 over the counted shared-world
    combos. Returns {p_yes, band, per_combo, institution_type, threshold, total_seats,
    provenance}. Deterministic given the frozen grounding + weights — no LLM, no rescaling."""
    import json as _json

    feasible_options_by_actor = feasible_options_by_actor or {}
    target = target_option or getattr(representation, "target_option", "") \
        or str((bp.terminal.get("rule_params") or {}).get("option") or "")
    model = classify_institution(representation, bp, grounding)     # forces are world-independent
    ref = model.forces.reference_prior
    voters = representation.voter_units()

    per_combo, num, den = [], 0.0, 0.0
    band_lo, band_hi = 1.0, 0.0
    resolved_threshold = None
    combos = shared_combos or [({}, 1.0)]
    for combo, w in combos:
        ck = _json.dumps(combo, sort_keys=True)
        table = (gw_by_combo or {}).get(ck, {})
        init = {}
        for u in voters:
            aid = None
            for cand in list(getattr(u, "member_ids", None) or []) + [u.unit_id]:
                if cand in states_by_actor:
                    aid = cand
                    break
            s = None
            if aid is not None:
                mids = (table.get(aid) or {}).get("mid", {})
                s = _target_support_from_states(
                    states_by_actor.get(aid), mids, target,
                    feasible_options_by_actor.get(aid) or [])
            # a repair unit (or an unmapped voter) starts at the grounded settling rate, never 0.5
            init[u.unit_id] = s if s is not None else (ref if ref is not None else 0.5)
        res = resolve_institution_vote(representation, init, model, max_rounds=max_rounds)
        resolved_threshold = res.threshold                     # the ABSOLUTE threshold actually used
        lo, hi = res.convergence_band()
        band_lo, band_hi = min(band_lo, lo), max(band_hi, hi)
        num += float(w) * res.p_yes
        den += float(w)
        per_combo.append({"combo": combo, "weight": round(float(w), 4),
                          "p_yes": round(res.p_yes, 4), "resolution": res.as_dict()})
    p_yes = (num / den) if den > 0 else None
    return {"p_yes": round(p_yes, 4) if p_yes is not None else None,
            "band": [round(band_lo, 4), round(band_hi, 4)] if den > 0 else None,
            "institution_type": model.institution_type,
            "threshold": resolved_threshold,
            "declared_threshold": representation.threshold,
            "total_seats": representation.total_voting_power(),
            "consensus_forces": model.forces.as_dict(),
            "per_combo": per_combo, "version": INSTITUTION_TERMINAL_VERSION,
            "provenance": {"law": "deliberative institution vote (D7 roster + D8 initial positions "
                                  "+ D14 convergence + seat-weighted absolute-threshold tally); "
                                  "no rescaling, no independent product, no invented convergence",
                           "n_voter_units": len(voters), "n_combos": len(combos),
                           "target_option": target}}
