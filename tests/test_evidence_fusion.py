"""Tests for the general per-person evidence-fusion primitive."""
from swm.variables.evidence import EvidenceFusion, PersonEvidence


def _training_people():
    # two archetypes: item 'q1' answer 0 <-> low first value dim, answer 1 <-> high first value dim
    people = []
    for _ in range(20):
        people.append(PersonEvidence("lo", None, [("q1", 0), ("q2", 0)]))
        people.append(PersonEvidence("hi", None, [("q1", 1), ("q2", 1)]))
    return people


def test_fit_builds_answer_centroids():
    fus = EvidenceFusion(attr_value_fn=lambda a: [0.5] * 10).fit(_training_people())
    assert "q1" in fus._centroid and 0 in fus._centroid["q1"] and 1 in fus._centroid["q1"]
    assert fus._centroid["q1"][0][1] == 20                 # 20 people answered q1 -> 0


def test_responses_move_the_value_profile():
    """A person with attributes A but answers like the 'hi' archetype should be pulled toward it."""
    # attribute map: distinct value per archetype so we can see movement
    def attr_fn(a):
        return [0.9 if a == "hi" else 0.1] + [0.5] * 9
    people = [PersonEvidence("hi", "hi", [("q1", 1), ("q2", 1)]) for _ in range(20)] + \
             [PersonEvidence("lo", "lo", [("q1", 0), ("q2", 0)]) for _ in range(20)]
    fus = EvidenceFusion(attr_value_fn=attr_fn).fit(people)
    # a person whose attributes say 'lo' (0.1) but who answered like 'hi'
    ev = PersonEvidence("x", "lo", [("q1", 1), ("q2", 1)])
    attr_only = attr_fn("lo")
    fused, meta = fus.value_profile(ev)
    assert meta["depth"] == 2 and meta["response_weight"] > 0
    assert fused[0] > attr_only[0]                          # pulled up toward the 'hi' centroid


def test_more_responses_raise_the_fusion_weight():
    fus = EvidenceFusion(attr_value_fn=lambda a: [0.5] * 10).fit(_training_people())
    _, m1 = fus.value_profile(PersonEvidence("x", "a", [("q1", 1)]))
    _, m2 = fus.value_profile(PersonEvidence("x", "a", [("q1", 1), ("q2", 1)]))
    assert m2["response_weight"] > m1["response_weight"]    # evidence depth increases the weight


def test_exclude_item_prevents_target_leakage():
    fus = EvidenceFusion(attr_value_fn=lambda a: [0.5] * 10).fit(_training_people())
    ev = PersonEvidence("x", "a", [("q1", 1), ("q2", 1)])
    _, m = fus.value_profile(ev, exclude_item="q1")
    assert m["depth"] == 1                                  # q1 dropped from the person's own context


def test_to_variable_map_carries_value_vector_and_evidence():
    fus = EvidenceFusion(attr_value_fn=lambda a: [0.3] * 10).fit(_training_people())
    vm = fus.to_variable_map(PersonEvidence("x", "a", [("q1", 1)]))
    assert "value_vector" in vm.meta and len(vm.meta["value_vector"]) == 10
    assert vm.meta["evidence"]["n_responses"] == 1
