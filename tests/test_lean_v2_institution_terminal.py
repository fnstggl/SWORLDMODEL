"""The composed institution-vote terminal law (D7 + D8 + D14). Universal machinery only.

Locks the end-to-end replacement of "count independent votes against a rescaled threshold" with a
deliberative sub-simulation over the faithful roster and the REAL threshold, per shared world."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.institution_terminal import resolve_institution_terminal
from swm.world_model_v2.lean_v2.representation import ensure_faithful_representation
from swm.world_model_v2.lean_v2.resolution_spec import INSTITUTION_VOTE, ResolutionSpec
from swm.world_model_v2.lean_v2.states import ActorStateHypothesis

OPTS = ["Maintain at 0.75%", "Raise to 1.0%"]


def _board_bp(members, *, rule="majority", target="Raise to 1.0%", roles=None):
    roles = roles or {}
    actors = [{"id": m, "name": m, "role": roles.get(m, "member")} for m in members]
    inst = {"id": "board", "name": "Board", "members": members, "decision_rule": rule,
            "rule_params": {"option": target}}
    return ConsumerWorldBlueprint(
        actors=actors, institutions=[inst],
        terminal={"kind": "institution_vote", "institution_id": "board",
                  "decision_rule": rule, "rule_params": {"option": target}},
        resolution={"interpretation": "Will the board raise?", "options": OPTS})


def _two_state_actors(members, raise_w):
    """each member has a raise-state and a hold-state; grounded weight raise_w on raise."""
    sba, mids = {}, {}
    for m in members:
        sba[m] = [ActorStateHypothesis(actor_id=m, state_id="raise", action_if_state=OPTS[1],
                                       expected_action_tendency=OPTS[1]),
                  ActorStateHypothesis(actor_id=m, state_id="hold", action_if_state=OPTS[0],
                                       expected_action_tendency=OPTS[0])]
        mids[m] = {"mid": {"raise": raise_w, "hold": 1 - raise_w}}
    return sba, {"{}": mids}


def _oc(rate, n):
    return {"outcome_reference_class": {"quantity": "board raises",
            "provenance": {"rate_mean": rate, "denominator": n, "numerator": round(rate * n),
                           "hierarchy_level": "same_institution"}}}


# ============================================================ faithful roster + real threshold
def test_composition_preserves_real_threshold_and_roster():
    members = [f"m{i}" for i in range(5)]                       # 5 modeled of a 9-member board
    bp = _board_bp(members)
    rep = ensure_faithful_representation(
        bp, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=9, vote_threshold=5),
        evidence_text="the nine-member policy board")
    sba, gw = _two_state_actors(members, 0.6)
    out = resolve_institution_terminal(bp, rep, _oc(0.55, 10), states_by_actor=sba, gw_by_combo=gw,
                                       shared_combos=[({}, 1.0)],
                                       feasible_options_by_actor={m: OPTS for m in members},
                                       target_option=OPTS[1])
    assert out["threshold"] == 5 and out["total_seats"] == 9    # never rescaled to majority-of-5
    assert 0.0 <= out["p_yes"] <= 1.0
    assert out["provenance"]["n_voter_units"] == 9              # repaired to the real body


# ============================================================ anti-consensus: type changes result
def test_consensus_body_differs_from_independent():
    members = [f"m{i}" for i in range(5)]
    sba, gw = _two_state_actors(members, 0.7)
    feas = {m: OPTS for m in members}
    # independent board (majority, no leader)
    bp_i = _board_bp(members, rule="majority")
    rep_i = ensure_faithful_representation(
        bp_i, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=5, vote_threshold=3),
        evidence_text="a five-member board")
    out_i = resolve_institution_terminal(bp_i, rep_i, {}, states_by_actor=sba, gw_by_combo=gw,
                                         shared_combos=[({}, 1.0)], feasible_options_by_actor=feas,
                                         target_option=OPTS[1])
    # unanimity board (typed consensus) — a high threshold independent voting rarely reaches
    bp_u = _board_bp(members, rule="unanimity")
    rep_u = ensure_faithful_representation(
        bp_u, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=5, vote_threshold=5,
                             vote_rule="unanimity"), evidence_text="a five-member board")
    out_u = resolve_institution_terminal(bp_u, rep_u, {}, states_by_actor=sba, gw_by_combo=gw,
                                         shared_combos=[({}, 1.0)], feasible_options_by_actor=feas,
                                         target_option=OPTS[1])
    assert out_i["institution_type"] == "independent_body"
    assert out_u["institution_type"] == "consensus_body"
    # the consensus body reaches its unanimity threshold far more often than independent voting
    indep_unanimity = 0.7 ** 5
    assert out_u["p_yes"] > indep_unanimity + 0.1


# ============================================================ combo aggregation is weighted
def test_p_yes_aggregates_over_shared_world_combos():
    members = [f"m{i}" for i in range(5)]
    bp = _board_bp(members)
    rep = ensure_faithful_representation(
        bp, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=5, vote_threshold=3),
        evidence_text="a five-member board")
    sba = {m: [ActorStateHypothesis(actor_id=m, state_id="raise", action_if_state=OPTS[1],
                                    expected_action_tendency=OPTS[1]),
               ActorStateHypothesis(actor_id=m, state_id="hold", action_if_state=OPTS[0],
                                    expected_action_tendency=OPTS[0])] for m in members}
    # two combos: a hawkish world (raise 0.9) at 0.3, a dovish world (raise 0.2) at 0.7
    import json
    gw = {json.dumps({"c": "hawk"}, sort_keys=True): {m: {"mid": {"raise": 0.9, "hold": 0.1}}
                                                      for m in members},
          json.dumps({"c": "dove"}, sort_keys=True): {m: {"mid": {"raise": 0.2, "hold": 0.8}}
                                                      for m in members}}
    combos = [({"c": "hawk"}, 0.3), ({"c": "dove"}, 0.7)]
    out = resolve_institution_terminal(bp, rep, {}, states_by_actor=sba, gw_by_combo=gw,
                                       shared_combos=combos,
                                       feasible_options_by_actor={m: OPTS for m in members},
                                       target_option=OPTS[1])
    assert len(out["per_combo"]) == 2
    hawk = next(c for c in out["per_combo"] if c["combo"] == {"c": "hawk"})
    dove = next(c for c in out["per_combo"] if c["combo"] == {"c": "dove"})
    assert hawk["p_yes"] > dove["p_yes"]                        # hawkish world → more YES
    # the aggregate is the weighted mean (dovish-heavy → below the hawkish value)
    assert abs(out["p_yes"] - (0.3 * hawk["p_yes"] + 0.7 * dove["p_yes"])) < 1e-3
    assert out["p_yes"] < hawk["p_yes"]


# ============================================================ repair units use the grounded prior
def test_repair_units_start_at_the_reference_prior():
    members = [f"m{i}" for i in range(5)]                       # 5 modeled of a 20-member body
    bp = _board_bp(members)
    rep = ensure_faithful_representation(
        bp, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=20, vote_threshold=11),
        evidence_text="the 20-member council")
    # modeled members strongly YES, but the 15 repair seats are anchored to a low settling rate
    sba, gw = _two_state_actors(members, 0.95)
    out = resolve_institution_terminal(bp, rep, _oc(0.2, 20), states_by_actor=sba, gw_by_combo=gw,
                                       shared_combos=[({}, 1.0)],
                                       feasible_options_by_actor={m: OPTS for m in members},
                                       target_option=OPTS[1])
    assert out["total_seats"] == 20 and out["threshold"] == 11
    # the 15 grounded-low repair seats keep the body from a near-certain YES despite 5 hawks
    assert out["p_yes"] < 0.6


# ============================================================ Wale: candidates ≠ electorate
def test_candidates_are_not_the_electorate_in_composition():
    # the "members" are the rival candidates; D7 adds the real electorate, D14 resolves it
    rivals = [f"cand_{i}" for i in range(5)]
    bp = _board_bp(rivals, roles={r: "leadership candidate" for r in rivals}, target="cand_0 wins")
    rep = ensure_faithful_representation(
        bp, ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_of_total=50, vote_threshold=26),
        evidence_text="the 50-seat parliament elects a leader")
    out = resolve_institution_terminal(bp, rep, {}, states_by_actor={}, gw_by_combo={},
                                       shared_combos=[({}, 1.0)], target_option="cand_0 wins")
    assert out["total_seats"] == 50 and out["threshold"] == 26  # electorate seats, real threshold
    assert out["p_yes"] is not None
