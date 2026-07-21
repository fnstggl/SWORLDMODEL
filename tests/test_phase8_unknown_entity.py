"""Regression (EXP-108 Wale crash): a persistence update whose event participant is an entity
this world does not hold must be a RECORDED skip (or a normalized-variant rescue), never a
process-killing KeyError mid-rollout. Compiled worlds can carry mixed naming variants — the
compiler itself emitted both 'Lowy Institute' and 'Lowy_Institute' in one plan."""
from swm.world_model_v2.phase8_transitions import PersistenceUpdateOperator
from swm.world_model_v2.state import Entity, SimulationClock, WorldState
from swm.world_model_v2.transitions import TransitionProposal


def _world():
    w = WorldState(world_id="w", branch_id="root",
                   clock=SimulationClock(now=1000.0, as_of=1000.0))
    w.entities["Jeremiah Manele"] = Entity(identity="Jeremiah Manele")
    return w


def _proposal(actor_id):
    return TransitionProposal(operator="persistence_update",
                              action={"etype": "actor_action", "payload": {"outcome": "engaged"},
                                      "participants": [actor_id]})


def test_underscore_variant_is_rescued_to_the_spaced_entity():
    d = PersistenceUpdateOperator().apply(_world(), _proposal("Jeremiah_Manele"))
    assert d is not None
    assert not any("skipped_unknown_entity" in c for c in (d.reason_codes or [])), \
        "the pure underscore/space variant must rescue to the registered entity"


def test_truly_unknown_entity_is_a_recorded_skip_not_a_crash():
    d = PersistenceUpdateOperator().apply(_world(), _proposal("someone_never_compiled"))
    assert d is not None and "skipped_unknown_entity" in (d.reason_codes or [])
