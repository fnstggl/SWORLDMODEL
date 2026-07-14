"""Independent question-level relevance gates on the authored 200-row corpus."""
from experiments.activation_corpus_200 import QUESTIONS
from swm.world_model_v2.causal_relevance import adjudicate_question


PHASES = {
    "p4": "phase4_actor_policy", "p6": "phase6_registry", "p7": "phase7_nonlinear",
    "p9pop": "phase9_populations", "p9net": "phase9_networks",
    "p10": "phase10_institutions", "p11": "phase11_recompilation",
}


def test_question_level_relevance_meets_development_gates():
    for label, phase in PHASES.items():
        positive = [(q, labels) for _, q, *_rest, labels in QUESTIONS if label in labels]
        negative = [(q, labels) for _, q, *_rest, labels in QUESTIONS if label not in labels]
        recall = sum(adjudicate_question(q)[phase]["required"] for q, _ in positive) / len(positive)
        false_activation = sum(adjudicate_question(q)[phase]["required"]
                               for q, _ in negative) / len(negative)
        assert recall >= 0.95, (label, recall)
        assert false_activation <= 0.10, (label, false_activation)


def test_suspect_sport_label_is_not_tuned_into_runtime():
    row = next(row for row in QUESTIONS if row[0] == "sport_6")
    assert "p11" in row[-1]  # preserve the authored source label
    assert not adjudicate_question(row[1])["phase11_recompilation"]["required"]
