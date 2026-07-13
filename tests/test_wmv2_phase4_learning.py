"""Phase 4 real-trajectory learning, calibration, and artifact tests."""
import json

import pytest

from swm.world_model_v2.phase4_learning import (
    DatasetManifest, HierarchicalPolicyFitter, HierarchicalPolicyPredictor, ResumableEvaluation,
    TrajectoryRecord, apply_calibration, evaluate_predictions, fit_temperature, paired_bootstrap,
    read_artifact, strict_split, write_artifact,
)


def records(n=80):
    out = []
    for i in range(n):
        actor = f"u{i // 4}"
        action = "support" if i % 3 else "oppose"
        out.append(TrajectoryRecord(
            record_id=f"r{i}", dataset_id="real_actions", actor_id=actor,
            actor_role="member" if i % 2 else "moderator", decision_time=1000 + i,
            context_id=f"q{i // 5}", institution_id=f"inst{i // 20}",
            relationship_id=f"rel{i // 3}", sequence_id=f"seq{i // 4}",
            observed_action=action, candidate_actions=["support", "oppose", "abstain"],
            actor_view_features={"history_depth": i % 4, "belief": (i % 10) / 10},
            outcome={"reward": 1 if action == "support" else 0}, source_ids=[f"source:{i}"],
            provenance={"as_of": 1000 + i, "post_action_features": False},
        ))
    return out


def manifest(rows):
    return DatasetManifest(
        dataset_id="real_actions", source="committed fixture derived from public records",
        license="test fixture", population="members", time_period="ordered",
        action_mapping={"yes": "support", "no": "oppose", "missing": "abstain"},
        available_actor_state=["role", "history_depth", "belief"], missing_actor_state=["private intent"],
        action_set_method="known institutional ballot options", split_methods=["person_disjoint", "time_forward"],
        leakage_risks=["repeated actor", "post-decision outcome"],
    ).seal(rows)


def test_trajectory_requires_observed_action_in_historical_action_set():
    row = records(1)[0]
    row.candidate_actions = ["abstain"]
    with pytest.raises(ValueError):
        row.validate()


@pytest.mark.parametrize("method,attr", [
    ("person_disjoint", "actor_id"), ("relationship_disjoint", "relationship_id"),
    ("context_disjoint", "context_id"), ("institution_disjoint", "institution_id"),
    ("sequence_disjoint", "sequence_id"),
])
def test_group_disjoint_splits_have_no_identity_leakage(method, attr):
    rows = records()
    split = strict_split(rows, method=method, seed=7)
    by = {r.record_id: r for r in rows}
    sets = [{getattr(by[x], attr) for x in ids} for ids in
            (split.train_ids, split.calibration_ids, split.validation_ids, split.test_ids)]
    assert split.verify() and all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4))


def test_time_forward_split_never_trains_on_future():
    rows = records()
    split = strict_split(rows, method="time_forward", seed=3)
    by = {r.record_id: r for r in rows}
    assert max(by[x].decision_time for x in split.train_ids) <= min(
        by[x].decision_time for x in split.calibration_ids)


def test_split_refuses_label_or_post_action_leakage():
    rows = records()
    rows[3].provenance["post_action_features"] = True
    with pytest.raises(ValueError):
        strict_split(rows, method="time_forward")


def test_hierarchical_fit_uses_training_only_and_cold_start_predicts():
    rows = records()
    split = strict_split(rows, method="person_disjoint", seed=4)
    artifact = HierarchicalPolicyFitter(seed=4).fit(rows, split, [manifest(rows)], code_commit="abc")
    assert artifact.verify() and artifact.fit_diagnostics["test_labels_touched"] is False
    predictor = HierarchicalPolicyPredictor(artifact)
    by = {r.record_id: r for r in rows}
    cold = by[split.test_ids[0]]
    p = predictor.predict(cold)
    assert set(p) == set(cold.candidate_actions) and abs(sum(p.values()) - 1) < 1e-12


def test_person_shrinkage_ablation_changes_repeated_actor_prediction():
    rows = records()
    split = strict_split(rows, method="time_forward")
    artifact = HierarchicalPolicyFitter(actor_pool_strength=1.0).fit(rows, split, [manifest(rows)])
    predictor = HierarchicalPolicyPredictor(artifact)
    # A training actor with four repeated choices gets a person-level posterior.
    row = next(r for r in rows if r.actor_id in artifact.actor_counts)
    full = predictor.predict(row)
    ablated = predictor.predict(row, ablations={"no_person_shrinkage"})
    assert full != ablated


def test_calibration_uses_separate_rows_and_preserves_normalization():
    preds = [{"support": 0.95, "oppose": 0.05}, {"support": 0.9, "oppose": 0.1},
             {"support": 0.8, "oppose": 0.2}, {"support": 0.7, "oppose": 0.3}]
    labels = ["oppose", "support", "support", "oppose"]
    artifact = fit_temperature(preds, labels, "calibration-split")
    out = apply_calibration(preds[0], artifact)
    assert artifact.fit_split_id == "calibration-split"
    assert abs(sum(out.values()) - 1) < 1e-12
    assert artifact.post_log_loss <= artifact.pre_log_loss


def test_calibration_and_bootstrap_honor_aggregate_frequency_weights():
    preds = [{"click": 0.8, "ignore": 0.2}, {"click": 0.2, "ignore": 0.8}]
    labels = ["ignore", "click"]
    weights = [100.0, 1.0]
    artifact = fit_temperature(preds, labels, "weighted-calibration", weights)
    assert artifact.post_log_loss <= artifact.pre_log_loss
    result = paired_bootstrap(preds, list(reversed(preds)), labels, weights=weights, n_boot=100, seed=3)
    assert result["effective_weight"] == 101.0
    assert result["resampling_unit"] == "trajectory_row_with_frequency_weight"


def test_metrics_keep_invalid_actions_in_denominator():
    predictions = [{"support": 0.8, "oppose": 0.2}, {"invalid": 0.9, "support": 0.1}]
    labels = ["support", "oppose"]
    metrics = evaluate_predictions(predictions, labels, [["support", "oppose"], ["support", "oppose"]])
    assert metrics["n"] == 2 and metrics["invalid_action_rate"] == 0.5
    assert metrics["log_loss"] > 1


def test_paired_bootstrap_reports_failure_direction():
    labels = ["support"] * 30
    good = [{"support": 0.8, "oppose": 0.2}] * 30
    bad = [{"support": 0.2, "oppose": 0.8}] * 30
    result = paired_bootstrap(good, bad, labels, n_boot=200)
    assert result["mean"] < 0 and result["ci95"][1] < 0


def test_artifact_atomic_round_trip_and_corruption(tmp_path):
    path = tmp_path / "policy.json"
    write_artifact(path, {"artifact_id": "a", "metrics": {"loss": 0.2}})
    assert read_artifact(path)["artifact_id"] == "a"
    body = json.loads(path.read_text()); body["metrics"]["loss"] = 0.1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        read_artifact(path)


def test_resumable_evaluation_and_corrupted_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = records(8)
    job = ResumableEvaluation(path, timeout_s=10)
    done = job.run(rows[:4], lambda r: {"prediction": r.observed_action})
    assert len(done) == 4
    with path.open("a") as handle:
        handle.write('{"record_id":"corrupt"}\n')
    done2 = job.run(rows, lambda r: {"prediction": r.observed_action})
    assert len(done2) == 8 and "corrupt" not in done2


def test_resumable_evaluation_timeout_preserves_completed_rows(tmp_path):
    path = tmp_path / "timeout.jsonl"
    job = ResumableEvaluation(path, timeout_s=-1)
    with pytest.raises(TimeoutError):
        job.run(records(3), lambda r: {})
    assert path.exists()
