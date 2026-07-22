"""D7 — faithful world representation. Universal machinery only (no question-specific logic).

These lock the EXP-113 representation failures:
  * BoJ: a 9-member board modeled as 5 units, the real "5 of 9" majority rescaled to "≥3 of 5".
  * Wale: a 50-seat parliament modeled as its 5 rival candidates, "26 of 50" rescaled to "3 of 5".

The invariant under test:  real voting power == represented voting power == threshold denominator,
reconciled by expanding the roster (individual seats for small bodies, seat-weighted blocs for
large ones) — NEVER by rescaling the real threshold, and NEVER by capping decisive actors."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint
from swm.world_model_v2.lean_v2.representation import (
    CANDIDATE, VOTER, DecisionUnit, INDIVIDUAL_BODY_MAX, WorldRepresentationSpec,
    build_representation, ensure_faithful_representation, infer_real_body_size,
    repair_representation, validate_representation)
from swm.world_model_v2.lean_v2.resolution_spec import INSTITUTION_VOTE, ResolutionSpec


# ------------------------------------------------------------------ fixtures
def _vote_bp(*, institution_id, members, target_option="", decision_rule="majority",
            interpretation="Will the body decide YES?"):
    """A minimal institution_vote blueprint with `members` modeled actors."""
    actors = []
    for m in members:
        if isinstance(m, dict):
            actors.append(m)
        else:
            actors.append({"id": m, "name": m.replace("_", " ").title(), "role": "member"})
    inst = {"id": institution_id, "name": institution_id.replace("_", " ").title(),
            "members": [a["id"] for a in actors], "decision_rule": decision_rule,
            "rule_params": {"option": target_option, "threshold": ""}}
    return ConsumerWorldBlueprint(
        actors=actors, institutions=[inst],
        terminal={"kind": "institution_vote", "institution_id": institution_id,
                  "decision_rule": decision_rule, "rule_params": {"option": target_option}},
        resolution={"interpretation": interpretation})


def _res(*, vote_of_total=None, vote_threshold=None, threshold_units="votes", vote_rule="majority"):
    return ResolutionSpec(terminal_kind=INSTITUTION_VOTE, vote_rule=vote_rule,
                          vote_threshold=vote_threshold, vote_of_total=vote_of_total,
                          threshold_units=threshold_units)


# ============================================================ 19 — nine-member board keeps power 9
def test_19_nine_member_board_has_total_voting_power_nine():
    # BoJ collapse: only 5 of the 9 board members modeled.
    bp = _vote_bp(institution_id="policy_board",
                  members=["governor", "deputy_a", "deputy_b", "member_c", "member_d"],
                  target_option="Raise to 1.0%",
                  interpretation="Will the policy board raise the rate?")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=9, vote_threshold=5),
                                          evidence_text="the nine-member policy board")
    assert spec.verdict == "ready"
    assert spec.real_member_count == 9
    assert spec.total_voting_power() == 9           # real body size preserved
    assert len(spec.voter_units()) == 9             # nine seats participate
    # the four unmodeled members were added as individual seats, not collapsed into one bloc vote
    assert all(u.kind == "individual" for u in spec.voter_units())


# ============================================================ 20 — a grouped bloc ≠ one ordinary vote
def test_20_five_member_bloc_cannot_cast_one_ordinary_vote():
    # A grouped bloc's voting power is its seat count, never 1.
    bloc = DecisionUnit(unit_id="b", kind="bloc", seat_count=5, roles=[VOTER], is_voter=True)
    assert bloc.voting_power == 5                    # five seats — NOT one vote
    ind = DecisionUnit(unit_id="i", kind="individual", seat_count=1, roles=[VOTER], is_voter=True)
    assert ind.voting_power == 1


# ============================================================ 21 — a bloc emits a distribution
def test_21_grouped_bloc_emits_distribution_over_member_votes():
    # Large body: 6 modeled, real 50 → the remaining 44 seats become a seat-weighted bloc that
    # carries all 44 seats (a distribution over member votes), never a single vote.
    bp = _vote_bp(institution_id="parliament",
                  members=[f"mp_{i}" for i in range(6)],
                  interpretation="Will parliament pass the motion?")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=50, vote_threshold=26),
                                          evidence_text="the 50-seat parliament")
    blocs = [u for u in spec.voter_units() if u.kind == "bloc"]
    assert len(blocs) == 1
    assert blocs[0].seat_count == 44                 # 50 − 6 modeled
    assert blocs[0].voting_power == 44               # distribution over 44 seats, not one vote
    assert spec.total_voting_power() == 50


# ============================================================ 22 — 50-seat parliament preserves 50
def test_22_fifty_seat_parliament_preserves_fifty_seats():
    bp = _vote_bp(institution_id="parliament",
                  members=[f"mp_{i}" for i in range(5)],
                  interpretation="Will parliament pass the bill?")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=50, vote_threshold=26),
                                          evidence_text="a 50-seat parliament")
    assert spec.verdict == "ready"
    assert spec.real_member_count == 50
    assert spec.total_voting_power() == 50           # not collapsed to 5


# ============================================================ 23 — candidates ≠ the electorate
def test_23_candidates_are_not_treated_as_the_electorate():
    # Wale collapse: the five rivals were modeled as the "members" of the 50-seat body.
    rivals = [{"id": f"cand_{i}", "name": f"Candidate {i}", "role": "leadership candidate"}
              for i in range(5)]
    bp = _vote_bp(institution_id="parliament", members=rivals,
                  interpretation="Will Candidate 0 become prime minister?")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=50, vote_threshold=26),
                                          evidence_text="the 50-seat parliament elects a PM")
    # candidates are typed non-voting; a real electorate carries the seats
    cand_units = [u for u in spec.decision_units if u.is_candidate]
    assert len(cand_units) == 5
    assert all(not u.is_voter for u in cand_units)   # a candidate is not an elector
    assert spec.total_voting_power() == 50           # the electorate, not the candidates, votes
    assert any(u.is_voter and not u.is_candidate for u in spec.decision_units)


def test_23b_candidate_detected_by_name_in_target_option():
    actor = {"id": "jane_doe", "name": "Jane Doe", "role": "politician"}
    bp = _vote_bp(institution_id="parliament", members=[actor, "mp_1", "mp_2"],
                  target_option="Jane Doe elected leader",
                  interpretation="Will Jane Doe be elected leader?")
    spec = build_representation(bp, _res(vote_of_total=50),
                               evidence_text="a 50-seat parliament")
    jane = next(u for u in spec.decision_units if u.unit_id == "jane_doe")
    assert jane.is_candidate and not jane.is_voter


# ============================================================ 24 — a 26-seat threshold stays 26
def test_24_twenty_six_seat_threshold_remains_twenty_six():
    bp = _vote_bp(institution_id="parliament", members=[f"mp_{i}" for i in range(5)])
    spec = ensure_faithful_representation(bp, _res(vote_of_total=50, vote_threshold=26,
                                                   threshold_units="seats"),
                                          evidence_text="a 50-seat parliament")
    assert spec.threshold == 26                       # never rescaled to majority-of-5
    assert spec.threshold_units == "seats"


# ============================================================ 25 — a 5-of-9 threshold stays 5
def test_25_five_of_nine_threshold_remains_five_votes():
    bp = _vote_bp(institution_id="policy_board",
                  members=["governor", "deputy_a", "deputy_b", "member_c", "member_d"],
                  target_option="Raise to 1.0%")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=9, vote_threshold=5),
                                          evidence_text="the nine-member policy board")
    assert spec.threshold == 5                        # never rescaled to ≥3 of 5
    assert spec.total_voting_power() == 9


# ============================================================ 26 — readiness fails on wrong power
def test_26_readiness_fails_when_represented_power_is_wrong():
    # Raw collapse, BEFORE repair: represented power (5) != real (9) → NOT ready.
    bp = _vote_bp(institution_id="policy_board",
                  members=["governor", "deputy_a", "deputy_b", "member_c", "member_d"])
    spec = validate_representation(
        build_representation(bp, _res(vote_of_total=9, vote_threshold=5),
                             evidence_text="the nine-member policy board"))
    assert spec.verdict == "repairable"               # readiness gate must not pass this as-is
    assert spec.verdict != "ready"


def test_26b_threshold_exceeding_real_body_is_not_ready():
    bp = _vote_bp(institution_id="policy_board",
                  members=["governor", "deputy_a", "deputy_b", "member_c", "member_d"])
    spec = ensure_faithful_representation(
        bp, _res(vote_of_total=9, vote_threshold=12),   # 12 of 9 is impossible
        evidence_text="the nine-member policy board")
    assert spec.verdict == "not_ready"                # cannot reconcile — never rescaled to fit


# ============================================================ 27 — no decisive-actor cap
def test_27_no_fixed_decisive_actor_cap_removes_a_voter():
    # A large body keeps EVERY seat even though it exceeds the individual-modeling boundary.
    assert INDIVIDUAL_BODY_MAX < 50
    bp = _vote_bp(institution_id="parliament", members=[f"mp_{i}" for i in range(5)])
    spec = ensure_faithful_representation(bp, _res(vote_of_total=50, vote_threshold=26),
                                          evidence_text="a 50-seat parliament")
    assert spec.total_voting_power() == 50            # not capped to 5, 9, or 15
    assert spec.total_voting_power() > INDIVIDUAL_BODY_MAX
    # a 9-body likewise keeps all 9, never capped to a handful of "decisive" actors
    bp9 = _vote_bp(institution_id="board", members=["a", "b", "c", "d", "e"])
    spec9 = ensure_faithful_representation(bp9, _res(vote_of_total=9, vote_threshold=5),
                                           evidence_text="a nine-member board")
    assert spec9.total_voting_power() == 9


# ============================================================ real-size inference
def test_infer_real_body_size_prefers_resolution_then_evidence():
    # resolution vote_of_total is authoritative
    n, src = infer_real_body_size(_res(vote_of_total=9), "a nine-member board")
    assert (n, src) == (9, "resolution:vote_of_total")
    # else deterministic N-member / N-seat phrasing from sealed evidence, largest wins
    n2, src2 = infer_real_body_size(_res(), "a 9-member board within the 50-seat parliament")
    assert n2 == 50 and src2 == "evidence:body_size_phrase"
    # nothing stated → unavailable, never guessed
    n3, src3 = infer_real_body_size(_res(), "the committee will decide")
    assert n3 is None and src3 == "unavailable"


# ============================================================ non-vote terminal is trivially ready
def test_non_vote_terminal_representation_is_ready():
    bp = ConsumerWorldBlueprint(
        terminal={"kind": "event_occurs"},
        resolution={"interpretation": "Will the event happen?"})
    spec = ensure_faithful_representation(bp, _res(), evidence_text="")
    assert spec.verdict == "ready" and spec.faithful


# ============================================================ threshold NEVER rescaled to modeled
def test_threshold_is_never_rescaled_to_the_modeled_roster():
    # The whole point of D7: with 5 modeled and real 9, the threshold stays 5 (of 9),
    # it does NOT become "majority of 5" (=3). Reconciliation expands the roster instead.
    bp = _vote_bp(institution_id="policy_board",
                  members=["governor", "deputy_a", "deputy_b", "member_c", "member_d"],
                  target_option="Raise to 1.0%")
    spec = ensure_faithful_representation(bp, _res(vote_of_total=9, vote_threshold=5),
                                          evidence_text="the nine-member policy board")
    assert spec.threshold == 5
    assert spec.threshold != 3                         # NOT rescaled to majority-of-5
    assert spec.total_voting_power() == spec.real_member_count == 9
    assert any("expanded the roster" in r for r in spec.repairs)
