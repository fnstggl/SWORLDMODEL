import math

import pytest

from swm.world_model_v2.phase4_completion import (
    ACTIONS, B7FamilyStacker, DecisionExample, EmpiricalFrequencyModel,
    HierarchicalPolicyModel, SparseSoftmaxModel, clustered_difference,
    conformal_summary, phase3_action_particles, transparent_heuristic,
)


def ipd_row(index, label, *, split="train", previous="none", opponent="none", actor="p1"):
    games = index
    cooperate = sum(1 for i in range(index) if i % 2 == 0)
    return DecisionExample(
        record_key=f"ipd:s:{actor}:{index}", dataset="ipd_long", actor_key=actor,
        actor_role="participant", relationship_key=f"s:{actor}:p2",
        sequence_key=f"s:{actor}", cluster_key="s", decision_time=float(index),
        decision_time_label=f"round={index}", label=label, actions=ACTIONS["ipd_long"],
        visible_state={
            "context": {"round": index, "treatment": "fixed_partner", "first_round": index == 0},
            "history": {"own_previous_action": previous, "previous_payoff": 1.0,
                        "cumulative_payoff": float(index), "games_played": games,
                        "own_cooperation_rate": cooperate / max(1, games),
                        "current_opponent_previous_action": opponent},
            "relationships": {"fixed_partner": True, "prior_meetings": games},
            "resources": {"cumulative_payoff": float(index)},
            "commitments": {"simultaneous_choice": True},
        },
        numeric_features={
            "round_scaled": index / 10.0, "fixed_partner": 1.0, "first_round": float(index == 0),
            "prior_own_cooperate": float(previous == "cooperate"),
            "prior_own_defect": float(previous == "defect"),
            "prior_opponent_cooperate": float(opponent == "cooperate"),
            "prior_opponent_defect": float(opponent == "defect"),
            "previous_payoff_scaled": 0.25, "cumulative_payoff_per_game": 0.25,
            "own_cooperation_rate": cooperate / max(1, games),
            "prior_meetings_scaled": games / 20.0,
        }, split=split,
    ).validate()


def test_llm_projection_is_label_blind_and_id_blind():
    row = ipd_row(2, "cooperate", previous="defect", opponent="cooperate")
    packet = row.llm_packet()
    rendered = str(packet)
    assert row.record_key not in rendered and row.actor_key not in rendered
    row.label = "defect"
    assert row.llm_packet() == packet


def test_phase3_particles_are_typed_deterministic_and_noncollapsed():
    row = ipd_row(8, "cooperate", previous="cooperate", opponent="defect")
    first, diagnostics = phase3_action_particles(row, [2.0, 2.0], n_particles=64)
    second, again = phase3_action_particles(row, [2.0, 2.0], n_particles=64)
    assert first == second and diagnostics == again
    assert len(first) == 64 and all(set(p) == set(row.actions) for p, _ in first)
    assert sum(weight for _, weight in first) == pytest.approx(1.0)
    assert diagnostics["ess_fraction"] == 1.0 and not diagnostics["collapsed"]


def test_softmax_hierarchy_and_b7_are_distinct_executable_arms():
    train = [ipd_row(i, "cooperate" if i % 3 == 0 else "defect",
                     previous="cooperate" if i % 2 else "defect",
                     opponent="cooperate" if i % 4 else "defect") for i in range(20)]
    validation = [ipd_row(20 + i, "cooperate" if i % 2 else "defect", split="validation",
                          previous="cooperate", opponent="defect") for i in range(4)]
    features = sorted(train[0].numeric_features)
    base = SparseSoftmaxModel.fit(train, ACTIONS["ipd_long"], features,
                                  regularization=0.1, epochs=2)
    hierarchical = HierarchicalPolicyModel(base, ACTIONS["ipd_long"], strength=3).fit_groups(train)
    frequency = EmpiricalFrequencyModel(ACTIONS["ipd_long"]).fit(train)
    total = sum(frequency.global_counts.values())
    alpha = [1 + 10 * frequency.global_counts[action] / total for action in ACTIONS["ipd_long"]]
    b7 = B7FamilyStacker.fit(validation, hierarchical, ACTIONS["ipd_long"], alpha)
    prediction, diagnostics = b7.predict(validation[0], hierarchical)
    assert sum(prediction.values()) == pytest.approx(1.0)
    assert diagnostics["b7_predict_sha256"] == diagnostics["b7_execute_input_sha256"]
    assert diagnostics["predict_execute_byte_identical"]
    assert transparent_heuristic(validation[0]) != hierarchical.predict(validation[0])


def test_cluster_bootstrap_uses_cluster_not_rows_and_conformal_is_finite_sample():
    rows = [ipd_row(i, "cooperate" if i % 2 else "defect") for i in range(8)]
    for i, row in enumerate(rows):
        row.cluster_key = "a" if i < 4 else "b"
    good = [{row.label: 0.8, ("defect" if row.label == "cooperate" else "cooperate"): 0.2}
            for row in rows]
    bad = [{row.label: 0.4, ("defect" if row.label == "cooperate" else "cooperate"): 0.6}
           for row in rows]
    bootstrap = clustered_difference(good, bad, rows, n_boot=200)
    assert bootstrap["clusters"] == 2 and bootstrap["mean"] < 0
    conformal = conformal_summary(good[:4], rows[:4], good[4:], rows[4:])
    assert conformal["finite_sample_rank"] == 4
    assert 0.0 <= conformal["coverage"] <= 1.0
