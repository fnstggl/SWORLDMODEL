"""The causal chain after §NAP — stances condition the actor's OWN cognition qualitatively;
binding prohibitions come from literal instruments; the mode graph is qualitative structure with
support counts; process grounding is typed state, never a progress bar."""
import types

import pytest

from swm.world_model_v2.mode_graph import (PATHWAYS, PROCESS_STATE_PREFIX, canonical_modes,
                                           declare_typed_processes, ground_process_states,
                                           mode_pathway)
from swm.world_model_v2.phase4_policy import (ACTION_ONTOLOGY, ActionTarget, ActorViewBuilder,
                                              FeasibilityEngine, TypedAction)
from swm.world_model_v2.quantities import Quantity, register_quantity_type
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

T0 = 1_700_000_000.0


def _world(**quants):
    w = WorldState("w", "b1:x", SimulationClock(now=T0, as_of=T0))
    for k, v in quants.items():
        register_quantity_type(k, units="unit")
        w.quantities[k] = Quantity(name=k, qtype=k, value=v, timestamp=T0)
    return w


def _actor(world, aid="leader_a", stances=None, commitments=None):
    e = Entity(aid)
    e.set("roles", F(["principal"], status="observed"))
    e.set("goals", F(["prevail"], status="inferred"))
    e.set("past_actions", F([], status="observed"))
    if stances is not None:
        e.set("stances", F(stances, status="observed", method="grounded_stances"))
    e.set("commitments", F(commitments or [], status="observed"))
    world.entities[aid] = e
    return e


def _action(name, family=None, aid="leader_a"):
    fam = family or next((f for f, names in ACTION_ONTOLOGY.items() if name in names), "generic")
    return TypedAction(action_id=f"a:{name}", actor_id=aid, actor_role="principal",
                       action_family=fam, action_name=name, target=ActionTarget(),
                       provenance={"source": "test"})


# ================================================================ §NAP: no numeric stance layer
def test_phase4_policy_exposes_no_action_magnitude_table():
    import swm.world_model_v2.phase4_policy as pp
    for name in ("ACTION_PATHWAY_EFFECTS", "action_pathway_effects",
                 "actions_advancing_pathway", "stance_action_alignment"):
        assert not hasattr(pp, name), f"{name} must not exist in production phase4_policy"


def test_mode_graph_exposes_no_numeric_tables():
    import swm.world_model_v2.mode_graph as mg
    for name in ("STANCE_ORIENTATION", "RELIABILITY_SHRINK", "CAPABILITY_SHRINK",
                 "CONTROL_WEIGHTS", "ENDOGENOUS_STANCE_SPLIT", "PROCESS_STATE_LEVELS",
                 "combine_stances", "pathway_orientation", "declare_pathway_processes",
                 "progress_var"):
        assert not hasattr(mg, name), f"{name} must not exist in production mode_graph"


# ================================================================ binding commitments (literal)
def test_public_statement_never_binds_feasibility():
    from swm.world_model_v2.resolution_criteria import _binding_prohibitions
    stance = {"actor": "leader_a", "commitment_level": "committed_to_prevent",
              "reliability": "high", "basis_kind": "public_statement",
              "pathway": "cooperative_agreement",
              "explicit_prohibitions": ["accept the proposal"]}
    assert _binding_prohibitions(stance) == []


def test_literal_instrument_binds_with_its_own_prohibitions():
    from swm.world_model_v2.resolution_criteria import _binding_prohibitions
    stance = {"actor": "leader_a", "commitment_level": "committed_to_prevent",
              "reliability": "high", "basis_kind": "contract",
              "pathway": "cooperative_agreement",
              "explicit_prohibitions": ["accept", "sign_agreement"]}
    assert _binding_prohibitions(stance) == ["accept", "sign_agreement"]
    # and ambiguity (no literal prohibition named) → no block
    assert _binding_prohibitions({**stance, "explicit_prohibitions": []}) == []


def test_binding_commitment_blocks_only_the_literal_action_names():
    w = _world()
    _actor(w, commitments=[{"id": "c1", "binding": True, "prohibits": ["accept"],
                            "basis_kind": "contract", "kind": "stated_intention"}])
    view = ActorViewBuilder().build(w, "leader_a")
    eng = FeasibilityEngine()
    blocked = eng.classify(_action("accept"), view, w)
    assert blocked.perceived_status == "binding_commitment_conflict"
    ok = eng.classify(_action("counteroffer"), view, w)
    assert ok.perceived_status != "binding_commitment_conflict"


def test_actor_view_wraps_legacy_string_commitments_without_char_splitting():
    w = _world()
    e = _actor(w)
    e.set("commitments", F("no frozen conflict", status="inferred"))
    view = ActorViewBuilder().build(w, "leader_a")
    assert view.commitments == [{"statement": "no frozen conflict"}]


# ================================================================ canonical modes (support votes)
def test_canonical_modes_majority_vote_without_numeric_priors():
    def llm(prompt):
        assert '"prior"' not in prompt      # the elicitation prompt requests NO numeric weight
        return ('{"modes": [{"id": "peace_treaty", "pathway": "cooperative_agreement",'
                '"decision_structure": {"rule": "unanimity", "approvers": ["A", "B"]}},'
                '{"id": "military_victory_2026", "pathway": "unilateral_action"}]}')
    modes, rep = canonical_modes(question="Will the war end?", criterion={},
                                 hypotheses=[{"id": "peace_treaty",
                                              "pathway": "cooperative_agreement"}],
                                 options=[], llm=llm, k_passes=2)
    ids = {m["id"] for m in modes}
    assert "peace_treaty" in ids
    for m in modes:
        assert "prior" not in m                                # §NAP: no LLM-minted mode priors
        assert isinstance(m["support"], int) and m["support"] >= 1
    treaty = next(m for m in modes if m["id"] == "peace_treaty")
    assert treaty["support"] == 3                              # hypotheses + both passes
    assert rep["n_sources"] == 3


def test_canonical_modes_time_indexed_duplicates_merge():
    def llm(prompt):
        return '{"modes": [{"id": "military_victory_2026"}, {"id": "military_victory"}]}'
    modes, _ = canonical_modes(question="q", criterion={}, hypotheses=[], options=[],
                               llm=llm, k_passes=1)
    assert [m["id"] for m in modes] == ["military_victory"]


def test_mode_pathway_classification():
    assert mode_pathway({"id": "x", "pathway": "institutional_procedure"}) \
        == "institutional_procedure"
    assert mode_pathway({"id": "ceasefire_deal"}) == "cooperative_agreement"
    assert "cooperative_agreement" in PATHWAYS


# ================================================================ typed process state (§NAP)
def test_ground_process_states_is_qualitative_only():
    def llm(prompt):
        return ('{"process_states": [{"pathway": "cooperative_agreement", "state": "active",'
                '"waiting_on": "counterparty response to the draft", "basis": "talks resumed"}]}')
    out = ground_process_states("q", {}, ["cooperative_agreement"], llm=llm)
    rec = out["cooperative_agreement"]
    assert rec["state"] == "active" and rec["basis"] == "talks resumed"
    assert "value" not in rec                                  # NO label→number map exists
    assert rec["waiting_on"].startswith("counterparty")


def test_declare_typed_processes_writes_string_state_not_progress_bars():
    plan = types.SimpleNamespace(quantities=[])
    modes = [{"id": "deal", "pathway": "cooperative_agreement"},
             {"id": "walkaway", "pathway": "unilateral_action"}]
    rep = declare_typed_processes(plan, modes,
                                  grounding={"cooperative_agreement": {"state": "exploratory",
                                                                       "basis": "feelers"}})
    assert rep["qualitative"] is True
    names = {q["name"]: q for q in plan.quantities}
    coop = names[f"{PROCESS_STATE_PREFIX}cooperative_agreement"]
    assert coop["value"] == "exploratory" and coop["sd"] is None
    uni = names[f"{PROCESS_STATE_PREFIX}unilateral_action"]
    assert uni["value"] == "ungrounded"                        # honest unknown, never 0.5
    assert not any(str(q["name"]).startswith(("pathway_progress:", "mode_progress:"))
                   for q in plan.quantities)
    for q in plan.quantities:
        assert not isinstance(q["value"], (int, float))        # nothing numeric declared
    assert plan._process_records["unilateral_action"]["state"] == "ungrounded"


def test_actor_view_projects_process_state_as_qualitative_belief():
    w = _world()
    register_quantity_type("process_state", units="state")
    w.quantities["process_state:cooperative_agreement"] = Quantity(
        name="process_state:cooperative_agreement", qtype="process_state", value="active",
        timestamp=T0)
    _actor(w)
    view = ActorViewBuilder().build(w, "leader_a")
    assert view.beliefs.get("process:process_state:cooperative_agreement") == "active"


# ================================================================ legacy scalar writer quarantine
def test_legacy_scalar_writer_refuses_outside_legacy_mode():
    from swm.world_model_v2.phase4_execution import ActorPolicyRuntime
    w = _world()
    _actor(w)
    with pytest.raises(RuntimeError):
        ActorPolicyRuntime._apply_pathway_effects(
            w, _action("accept"), types.SimpleNamespace(changes=[]),
            consequence_mode="generated_actor_mediated_world")


def test_derive_pathway_summaries_requires_ablation_token():
    from swm.world_model_v2.semantic_consequences import derive_pathway_summaries
    w = _world()
    w.objects = {}
    with pytest.raises(PermissionError):
        derive_pathway_summaries(w)
