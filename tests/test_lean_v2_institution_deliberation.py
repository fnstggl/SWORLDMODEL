"""D14 — deliberative convergence. Universal machinery only.

Locks the dominant EXP-113 defect: an institution vote must be resolved by simulating the body's
grounded decision process, never as a product of independent per-member draws. Convergence comes
from grounded, institution-specific forces; absent grounding the body is independent (honest
baseline); a strong consensus/unanimity body reaches a high threshold that independent voting
almost never would — with no fixed numeric consensus bonus."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.institution_deliberation import (
    CONSENSUS_BODY, COALITION_BODY, HIERARCHICAL_BODY, INDEPENDENT_BODY, ConvergenceForces,
    ConvergenceModel, classify_institution, resolve_institution_vote,
    run_institution_deliberation, seat_weighted_yes_pmf, seat_weighted_yes_prob)
from swm.world_model_v2.lean_v2.representation import (
    CANDIDATE, VOTER, DecisionUnit, WorldRepresentationSpec, ensure_faithful_representation)
from swm.world_model_v2.lean_v2.resolution_spec import INSTITUTION_VOTE, ResolutionSpec


def _rep(n_individuals, threshold, *, rule="majority", blocs=None, institution_id="b"):
    r = WorldRepresentationSpec(institution_id=institution_id, rule=rule, threshold=threshold)
    for i in range(n_individuals):
        r.decision_units.append(DecisionUnit(unit_id=f"m{i}", kind="individual", seat_count=1,
                                             roles=[VOTER], is_voter=True))
    for bid, seats in (blocs or []):
        r.decision_units.append(DecisionUnit(unit_id=bid, kind="bloc", seat_count=seats,
                                             roles=[VOTER], is_voter=True))
    r.represented_voting_power = r.total_voting_power()
    return r


# ============================================================ seat-weighted tally (D7 threshold)
def test_seat_weighted_convolution_individuals():
    units = [DecisionUnit(unit_id=f"m{i}", seat_count=1, is_voter=True) for i in range(9)]
    pos = {f"m{i}": 0.5 for i in range(9)}
    pmf = seat_weighted_yes_pmf(units, pos)
    assert abs(sum(pmf) - 1.0) < 1e-9 and len(pmf) == 10        # 0..9 yes-seats
    assert abs(seat_weighted_yes_prob(units, pos, 5) - 0.5) < 1e-9   # majority of 9 at p=0.5


def test_seat_weighted_unanimity_is_product():
    units = [DecisionUnit(unit_id=f"m{i}", seat_count=1, is_voter=True) for i in range(5)]
    pos = {f"m{i}": 0.8 for i in range(5)}
    assert abs(seat_weighted_yes_prob(units, pos, 5) - 0.8 ** 5) < 1e-9    # all 5 must agree


def test_seat_weighted_bloc_is_binomial_not_one_vote():
    # a 5-seat bloc at p=0.6 contributes Binomial(5,0.6) yes-seats — a distribution, not one vote
    bloc = DecisionUnit(unit_id="bloc", kind="bloc", seat_count=5, is_voter=True)
    pmf = seat_weighted_yes_pmf([bloc], {"bloc": 0.6})
    assert len(pmf) == 6                                        # 0..5 yes-seats
    from math import comb
    assert abs(pmf[3] - comb(5, 3) * 0.6 ** 3 * 0.4 ** 2) < 1e-9
    assert 0.0 < pmf[5] < 1.0 and pmf[5] != 0.6                # NOT "one vote for 5 seats"


def test_seat_weighted_threshold_is_absolute_never_rescaled():
    # 50 seats, 26 threshold: at p=0.5 that is P(>=26 of 50) ~ just under half, and the threshold
    # is the REAL 26 — never rescaled to "majority of a handful"
    units = [DecisionUnit(unit_id=f"m{i}", seat_count=1, is_voter=True) for i in range(50)]
    pos = {f"m{i}": 0.5 for i in range(50)}
    p = seat_weighted_yes_prob(units, pos, 26)
    assert 0.30 < p < 0.50                                      # 26 of 50 is above the median seat


# ============================================================ 36 — no invented convergence
def test_36_independent_body_does_not_converge():
    r = _rep(9, 5)
    init = {f"m{i}": 0.6 for i in range(9)}
    res = resolve_institution_vote(r, init, ConvergenceModel(INDEPENDENT_BODY, ConvergenceForces()))
    # final == initial; p_yes == the plain independent convolution (no invented convergence)
    assert all(abs(res.transcript["final_positions"][f"m{i}"] - 0.6) < 1e-9 for i in range(9))
    assert abs(res.p_yes - res.p_yes_predeliberation) < 1e-9
    assert res.consensus_strength == 0.0


# ============================================================ 37 — anti-consensus (the core fix)
def test_37_consensus_body_reaches_unanimity_independent_voting_misses():
    # 9 members each lean YES 0.7. Independent unanimity ~ 0.7^9 = 0.04 (drastic under-forecast).
    r = _rep(9, 9, rule="unanimity")
    init = {f"m{i}": 0.7 for i in range(9)}
    indep = resolve_institution_vote(r, init,
                                     ConvergenceModel(INDEPENDENT_BODY, ConvergenceForces()))
    consensus = resolve_institution_vote(
        r, init, ConvergenceModel(CONSENSUS_BODY,
                                  ConvergenceForces(consensus_norm=0.8, reference_prior=0.7)))
    assert indep.p_yes < 0.06                                   # the bug: almost never unanimous
    assert consensus.p_yes > 0.4                                # the fix: consensus reaches it
    assert consensus.p_yes > indep.p_yes


# ============================================================ 38 — calibrated, not over-sharpened
def test_38_full_consensus_is_the_collective_lean_not_an_extreme():
    # a body whose members sit at a 0.7 lean, fully consolidated, lands YES ~0.7 — NEVER 0.99
    r = _rep(9, 5)
    init = {f"m{i}": 0.7 for i in range(9)}
    res = resolve_institution_vote(
        r, init, ConvergenceModel(CONSENSUS_BODY, ConvergenceForces(consensus_norm=1.0)))
    assert abs(res.p_yes_consolidated - 0.7) < 0.05
    assert res.p_yes < 0.85                                     # not blown up to a near-certainty


# ============================================================ 39 — strength scales, no fixed bonus
def test_39_convergence_scales_with_grounded_strength():
    r = _rep(9, 9, rule="unanimity")
    init = {f"m{i}": 0.7 for i in range(9)}
    ps = []
    for cn in (0.0, 0.3, 0.6, 0.9):
        itype = CONSENSUS_BODY if cn > 0 else INDEPENDENT_BODY
        res = resolve_institution_vote(r, init,
                                       ConvergenceModel(itype, ConvergenceForces(consensus_norm=cn)))
        ps.append(res.p_yes)
    assert ps[0] < 0.06                                         # strength 0 → independent baseline
    assert ps[0] < ps[1] < ps[2] < ps[3]                       # monotone in grounded strength


# ============================================================ 40 — coalition body uses bloc rule
def test_40_coalition_body_converges_within_blocs():
    # two 3-member blocs: bloc A leans YES, bloc B leans NO; discipline pulls each toward its bloc
    r = _rep(6, 4)
    coalitions = {"m0": "A", "m1": "A", "m2": "A", "m3": "B", "m4": "B", "m5": "B"}
    init = {"m0": 0.9, "m1": 0.5, "m2": 0.6, "m3": 0.1, "m4": 0.4, "m5": 0.2}
    model = ConvergenceModel(COALITION_BODY,
                             ConvergenceForces(coalition_discipline=0.8, coalitions=coalitions))
    res = resolve_institution_vote(r, init, model)
    fp = res.transcript["final_positions"]
    # within bloc A the members move toward the bloc's YES-leaning mean; within B toward NO
    assert fp["m1"] > init["m1"] and fp["m4"] < init["m4"]
    assert res.consensus_strength == 0.8


# ============================================================ 41 — bounded rounds + material gate
def test_41_bounded_rounds_and_material_change_gate():
    r = _rep(5, 3)
    init = {f"m{i}": 0.5 for i in range(5)}
    # a body already at its target makes no material change → converges immediately, no churn
    res = resolve_institution_vote(
        r, init, ConvergenceModel(CONSENSUS_BODY,
                                  ConvergenceForces(consensus_norm=0.8, reference_prior=0.5)))
    assert res.transcript["rounds_run"] <= 8                    # bounded
    assert res.transcript["converged"]
    # everyone starts and the target is 0.5 → immaterial → no revise messages beyond round 0
    assert res.transcript["material_changes"] == 0


# ============================================================ 42 — classifier grounds the forces
def _bp_board(members, roles=None, rule="majority", procedure=None):
    roles = roles or {}
    actors = [{"id": m, "name": m, "role": roles.get(m, "member")} for m in members]
    inst = {"id": "board", "name": "Board", "members": members, "decision_rule": rule,
            "rule_params": {}, "procedure": procedure or []}
    return ConsumerWorldBlueprint(actors=actors, institutions=[inst],
                                  terminal={"kind": "institution_vote", "institution_id": "board",
                                            "decision_rule": rule},
                                  resolution={"interpretation": "Will the board approve?"})


def test_42_classifier_defaults_to_independent_without_grounding():
    rep = _rep(5, 3, institution_id="board")
    bp = _bp_board([f"m{i}" for i in range(5)])
    model = classify_institution(rep, bp, {})
    assert model.institution_type == INDEPENDENT_BODY          # no counted force → honest baseline
    assert model.forces.consensus_norm == 0.0


def test_43_classifier_types_consensus_from_unanimity_rule():
    rep = _rep(5, 5, rule="unanimity", institution_id="board")
    bp = _bp_board([f"m{i}" for i in range(5)], rule="unanimity")
    model = classify_institution(rep, bp, {})
    assert model.institution_type == CONSENSUS_BODY
    assert model.forces.consensus_norm > 0                     # typed from the unanimity rule
    assert "unanimity" in model.forces.consensus_norm_source


def test_44_classifier_reads_counted_reference_prior_and_leader():
    rep = _rep(5, 3, institution_id="board")
    bp = _bp_board([f"m{i}" for i in range(5)], roles={"m0": "Governor and chair"})
    grounding = {"outcome_reference_class": {"quantity": "the board approves",
                 "provenance": {"rate_mean": 0.72, "denominator": 8, "numerator": 6,
                                "hierarchy_level": "same_institution"}}}
    model = classify_institution(rep, bp, grounding)
    assert abs(model.forces.reference_prior - 0.72) < 1e-9     # counted, grounded
    assert model.forces.leader_unit_id == "m0"                 # presiding office detected
    assert model.institution_type == CONSENSUS_BODY            # a leader with authority


def test_45_classifier_detects_coalitions():
    members = [f"m{i}" for i in range(6)]
    actors = [{"id": m, "name": m, "role": "mp", "party": ("gov" if i < 3 else "opp")}
              for i, m in enumerate(members)]
    inst = {"id": "board", "name": "House", "members": members, "decision_rule": "majority",
            "rule_params": {}, "procedure": []}
    bp = ConsumerWorldBlueprint(actors=actors, institutions=[inst],
                                terminal={"kind": "institution_vote", "institution_id": "board"},
                                resolution={"interpretation": "Will the House pass it?"})
    rep = _rep(6, 4, institution_id="board")
    model = classify_institution(rep, bp, {})
    assert model.institution_type == COALITION_BODY
    assert len(set(model.forces.coalitions.values())) == 2


# ============================================================ end-to-end through D7 representation
def test_end_to_end_faithful_representation_plus_deliberation():
    # a 9-member board modeled as 5 → D7 repairs to 9 seats → D14 resolves the 5-of-9 vote
    actors = [{"id": f"g{i}", "name": f"Gov {i}", "role": "member"} for i in range(5)]
    inst = {"id": "policy_board", "name": "Policy Board", "members": [a["id"] for a in actors],
            "decision_rule": "majority", "rule_params": {"option": "Raise"}}
    bp = ConsumerWorldBlueprint(actors=actors, institutions=[inst],
                                terminal={"kind": "institution_vote",
                                          "institution_id": "policy_board",
                                          "decision_rule": "majority",
                                          "rule_params": {"option": "Raise"}},
                                resolution={"interpretation": "Will the board raise?"})
    rep = ensure_faithful_representation(
        bp, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5),
        evidence_text="the nine-member policy board")
    assert rep.total_voting_power() == 9 and rep.threshold == 5
    init = {u.unit_id: 0.6 for u in rep.voter_units()}
    model = classify_institution(rep, bp, {})
    res = resolve_institution_vote(rep, init, model)
    assert res.threshold == 5 and res.total_seats == 9         # real threshold, never rescaled
    assert 0.0 <= res.p_yes <= 1.0
