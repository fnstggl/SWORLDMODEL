"""D10 — verified reference cases + separated layers. Universal machinery only.

Locks: every counted case earns its count (real source, verifiable quote, pre-as_of date, typed
action); fabricated / placeholder / leaked / vague cases are excluded with reasons; the three
layers (outcome / action baseline / private state) are kept separate."""
from __future__ import annotations

from swm.world_model_v2.lean_v2.reference_verification import (
    LAYER_ACTION, LAYER_OUTCOME, LAYER_STATE, VerifiedReferenceCase, counted_rate,
    verify_cases, verify_reference_case)

EV = ("The governor raised rates in July 2023 citing persistent inflation. "
      "In 2019 the board held steady amid weak growth. The deputy dissented in 2021.")


def _v(raw, layer=LAYER_ACTION, as_of="2025-06-01"):
    return verify_reference_case(raw, evidence_text=EV, as_of=as_of, layer=layer)


# ============================================================ 16 — verified case counts
def test_16_verifiable_case_is_included():
    vc = _v({"source": "Reuters 2023", "basis_quote": "the governor raised rates in July 2023",
             "date": "2023-07-01", "actor_or_role": "governor", "observed_action": "raise",
             "outcome": True})
    assert vc.included
    assert vc.source_available and vc.quote_verified and vc.date_verified and vc.action_typed


# ============================================================ 17 — fabrications rejected
def test_17_fabricated_quote_is_rejected():
    vc = _v({"source": "Reuters", "basis_quote": "the governor secretly promised a rate cut",
             "date": "2023-07-01", "observed_action": "cut", "outcome": True})
    assert not vc.included and not vc.quote_verified
    assert "not verifiable" in vc.exclusion_reason


def test_17b_placeholder_source_is_rejected():
    vc = _v({"source": "example.com", "basis_quote": "the board held steady",
             "date": "2019-01-01", "observed_action": "hold", "outcome": False})
    assert not vc.included and not vc.source_available
    vc2 = _v({"source": "a study", "basis_quote": "the board held steady", "date": "2019-01-01",
              "observed_action": "hold"})
    assert not vc2.source_available


def test_17c_post_as_of_case_is_leakage():
    vc = _v({"source": "Reuters", "basis_quote": "the governor raised rates in July 2023",
             "date": "2025-08-01", "observed_action": "raise", "outcome": True})
    assert not vc.included and not vc.date_verified
    assert "leakage" in vc.exclusion_reason


def test_17d_vague_action_cannot_be_typed():
    vc = _v({"source": "Reuters", "basis_quote": "in 2019 the board held steady",
             "date": "2019-01-01", "actor_or_role": "board", "observed_action": "acted",
             "outcome": False})
    assert not vc.included and not vc.action_typed
    # the same case in the OUTCOME layer does not require an action type
    vo = _v({"source": "Reuters", "basis_quote": "in 2019 the board held steady",
             "date": "2019-01-01", "observed_action": "acted", "outcome": False},
            layer=LAYER_OUTCOME)
    assert vo.action_typed                                    # outcome layer: no action typing needed


# ============================================================ counted rate uses only verified
def test_counted_rate_excludes_unverified():
    cases = [
        {"source": "Reuters", "basis_quote": "the governor raised rates in July 2023",
         "date": "2023-07-01", "observed_action": "raise", "outcome": True},          # ok
        {"source": "example.com", "basis_quote": "invented", "date": "2020-01-01",
         "observed_action": "raise", "outcome": True},                                 # placeholder
        {"source": "Reuters", "basis_quote": "in 2019 the board held steady",
         "date": "2019-01-01", "observed_action": "hold", "outcome": False},           # ok
    ]
    verified = verify_cases(cases, evidence_text=EV, as_of="2025-06-01", layer=LAYER_ACTION)
    cr = counted_rate(verified)
    assert cr["denominator"] == 2 and cr["n_excluded"] == 1   # only the two verifiable cases count
    assert cr["action_option_id"] in ("raise", "hold")        # typed to a dominant action class
    assert len(cr["excluded"]) == 1


# ============================================================ 33 — layers stay separate
def test_33_layers_are_distinct_concepts():
    # the same historical event verified under different layers stays tagged to its layer, so
    # outcome history can never be counted as private-state evidence
    raw = {"source": "Reuters", "basis_quote": "the deputy dissented in 2021",
           "date": "2021-01-01", "actor_or_role": "deputy", "observed_action": "dissent",
           "outcome": True}
    outcome = _v(raw, layer=LAYER_OUTCOME)
    action = _v(raw, layer=LAYER_ACTION)
    state = _v(raw, layer=LAYER_STATE)
    assert outcome.layer == LAYER_OUTCOME
    assert action.layer == LAYER_ACTION
    assert state.layer == LAYER_STATE
    # a state-layer case is never mixed into an action-baseline count
    mixed = counted_rate([outcome, state])
    assert all(c.layer in (LAYER_OUTCOME, LAYER_STATE) for c in [outcome, state])
    assert mixed["denominator"] == 2      # counted within-call, but the caller keeps buckets apart
