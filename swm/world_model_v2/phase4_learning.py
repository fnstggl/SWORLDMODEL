"""Real-trajectory learning, calibration, evaluation, and artifact governance for Phase 4.

The core library remains dependency-free.  The fitter is a hierarchical count/logit
estimator with empirical-Bayes shrinkage; richer numerical models may implement the
same serialized contract.  Candidate action sets are row-specific and are never
silently replaced by a global label vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


LEARNING_SCHEMA_VERSION = "4.0.0"


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass
class DatasetManifest:
    dataset_id: str
    source: str
    license: str
    population: str
    time_period: str
    action_mapping: dict
    available_actor_state: list
    missing_actor_state: list
    action_set_method: str
    split_methods: list
    leakage_risks: list
    network_information: str = ""
    institution_information: str = ""
    limitations: list = field(default_factory=list)
    content_hash: str = ""
    schema_version: str = LEARNING_SCHEMA_VERSION

    def seal(self, records: list) -> "DatasetManifest":
        self.content_hash = digest([r.as_dict() for r in records])
        return self


@dataclass
class TrajectoryRecord:
    record_id: str
    dataset_id: str
    actor_id: str
    actor_role: str
    decision_time: float
    context_id: str
    institution_id: str
    relationship_id: str
    sequence_id: str
    observed_action: str
    candidate_actions: list
    actor_view_features: dict
    outcome: dict = field(default_factory=dict)
    action_set_hypotheses: list = field(default_factory=list)
    source_ids: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    sample_weight: float = 1.0

    def validate(self):
        if not self.record_id or not self.actor_id or not self.observed_action:
            raise ValueError("record_id, actor_id, and observed_action are required")
        if self.observed_action not in self.candidate_actions:
            raise ValueError(f"observed action {self.observed_action!r} missing from reconstructed action set")
        if len(set(self.candidate_actions)) != len(self.candidate_actions):
            raise ValueError("candidate action set contains duplicates")
        if not math.isfinite(float(self.sample_weight)) or float(self.sample_weight) <= 0:
            raise ValueError("sample_weight must be finite and positive")
        if self.provenance.get("post_action_features"):
            raise ValueError("post-action feature leakage is prohibited")
        return self

    def as_dict(self):
        return asdict(self)


@dataclass
class SplitManifest:
    split_id: str
    method: str
    seed: int
    train_ids: list
    calibration_ids: list
    validation_ids: list
    test_ids: list
    frozen_at: str
    leakage_checks: dict
    checksum: str = ""

    def seal(self):
        self.checksum = digest({k: v for k, v in asdict(self).items() if k != "checksum"})
        return self

    def verify(self):
        return self.checksum == digest({k: v for k, v in asdict(self).items() if k != "checksum"})


def strict_split(records: list[TrajectoryRecord], *, method: str, seed: int = 0,
                 fractions=(0.6, 0.15, 0.1, 0.15)) -> SplitManifest:
    """Create immutable train/calibration/validation/test splits.

    Group-disjoint methods split groups, not rows.  ``time_forward`` sorts all
    decisions and therefore never moves a later example into training.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    for record in records:
        record.validate()
    if method == "time_forward":
        ordered = sorted(records, key=lambda r: (r.decision_time, r.record_id))
        buckets = _cut([r.record_id for r in ordered], fractions)
    else:
        attrs = {
            "person_disjoint": "actor_id", "actor_disjoint": "actor_id",
            "relationship_disjoint": "relationship_id", "context_disjoint": "context_id",
            "institution_disjoint": "institution_id", "sequence_disjoint": "sequence_id",
        }
        attr = attrs.get(method)
        if attr is None:
            raise ValueError(f"unsupported strict split method {method!r}")
        groups = {}
        for record in records:
            key = str(getattr(record, attr) or f"missing:{record.record_id}")
            groups.setdefault(key, []).append(record.record_id)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        group_buckets = _cut(keys, fractions)
        buckets = [[rid for key in bucket for rid in groups[key]] for bucket in group_buckets]
    checks = split_leakage_audit(records, buckets, method)
    if not checks["passed"]:
        raise ValueError(f"split leakage detected: {checks['violations']}")
    return SplitManifest(
        split_id=digest({"method": method, "seed": seed, "ids": buckets})[:20],
        method=method, seed=seed, train_ids=buckets[0], calibration_ids=buckets[1],
        validation_ids=buckets[2], test_ids=buckets[3],
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        leakage_checks=checks,
    ).seal()


def _cut(items, fractions):
    n = len(items)
    cuts = [0]
    cumulative = 0.0
    for frac in fractions[:-1]:
        cumulative += frac
        cuts.append(round(n * cumulative))
    cuts.append(n)
    return [list(items[cuts[i]:cuts[i + 1]]) for i in range(4)]


def split_leakage_audit(records, buckets, method):
    lookup = {r.record_id: r for r in records}
    violations = []
    flat = [rid for bucket in buckets for rid in bucket]
    if len(flat) != len(set(flat)):
        violations.append("record appears in more than one split")
    if set(flat) != set(lookup):
        violations.append("split does not preserve every row")
    attr = {
        "person_disjoint": "actor_id", "actor_disjoint": "actor_id",
        "relationship_disjoint": "relationship_id", "context_disjoint": "context_id",
        "institution_disjoint": "institution_id", "sequence_disjoint": "sequence_id",
    }.get(method)
    if attr:
        sets = [{getattr(lookup[rid], attr) for rid in bucket} for bucket in buckets]
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = (sets[i] & sets[j]) - {""}
                if overlap:
                    violations.append(f"{attr} overlap split {i}/{j}: {sorted(overlap)[:3]}")
    if method == "time_forward":
        bounds = [(min((lookup[x].decision_time for x in b), default=math.inf),
                   max((lookup[x].decision_time for x in b), default=-math.inf)) for b in buckets]
        for i in range(3):
            if bounds[i][1] > bounds[i + 1][0]:
                violations.append(f"time overlap split {i}/{i+1}")
    for record in records:
        if record.provenance.get("label_in_features") or record.provenance.get("post_action_features"):
            violations.append(f"feature leakage in {record.record_id}")
    return {"passed": not violations, "violations": violations, "n_rows": len(records)}


@dataclass
class HierarchicalPolicyArtifact:
    artifact_id: str
    schema_version: str
    dataset_hashes: dict
    split_id: str
    seed: int
    config: dict
    global_counts: dict
    domain_counts: dict
    institution_counts: dict
    role_counts: dict
    actor_counts: dict
    feature_effects: dict
    policy_family_weights: dict
    uncertainty: dict
    fit_diagnostics: dict
    created_at: str
    code_commit: str
    checksum: str = ""

    def seal(self):
        self.checksum = digest({k: v for k, v in asdict(self).items() if k != "checksum"})
        return self

    def verify(self):
        return self.checksum == digest({k: v for k, v in asdict(self).items() if k != "checksum"})

    def as_dict(self):
        return asdict(self)


class HierarchicalPolicyFitter:
    """Empirical-Bayes partial pooling over row-specific action sets.

    Counts are used as conjugate sufficient statistics.  Optional numeric features
    learn bounded action-specific mean differences on training rows only.  This is
    intentionally transparent and safe for sparse/cold-start actors.
    """

    def __init__(self, *, alpha: float = 1.0, actor_pool_strength: float = 10.0,
                 role_pool_strength: float = 20.0, institution_pool_strength: float = 30.0,
                 seed: int = 0):
        self.config = {"alpha": alpha, "actor_pool_strength": actor_pool_strength,
                       "role_pool_strength": role_pool_strength,
                       "institution_pool_strength": institution_pool_strength}
        self.seed = seed

    def fit(self, records: list[TrajectoryRecord], split: SplitManifest,
            manifests: list[DatasetManifest], *, code_commit: str = "") -> HierarchicalPolicyArtifact:
        if not split.verify():
            raise ValueError("corrupt or unsealed split manifest")
        by_id = {r.record_id: r for r in records}
        train = [by_id[rid] for rid in split.train_ids]
        train_ids = set(split.train_ids)
        if not train:
            raise ValueError("training split is empty")
        global_counts, domain, institutions, roles, actors = {}, {}, {}, {}, {}
        feature_sums, feature_ns = {}, {}
        invalid_rows = []
        for row in train:
            try:
                row.validate()
            except ValueError as exc:
                invalid_rows.append({"record_id": row.record_id, "reason": str(exc)})
                continue
            weight = float(row.sample_weight)
            _inc(global_counts, row.observed_action, weight)
            _inc_nested(domain, row.dataset_id, row.observed_action, weight)
            _inc_nested(institutions, row.institution_id or "__missing__", row.observed_action, weight)
            _inc_nested(roles, row.actor_role or "__missing__", row.observed_action, weight)
            _inc_nested(actors, row.actor_id, row.observed_action, weight)
            for feature, value in row.actor_view_features.items():
                if not isinstance(value, (int, float)):
                    continue
                key = f"{row.observed_action}:{feature}"
                feature_sums[key] = feature_sums.get(key, 0.0) + float(value)
                feature_ns[key] = feature_ns.get(key, 0) + 1
        effects = {key: feature_sums[key] / feature_ns[key] for key in feature_sums}
        config = dict(self.config)
        config["frozen_before_test"] = True
        payload = {"split": split.split_id, "seed": self.seed, "config": config,
                   "global_counts": global_counts, "train_ids": sorted(train_ids)}
        artifact = HierarchicalPolicyArtifact(
            artifact_id=digest(payload)[:24], schema_version=LEARNING_SCHEMA_VERSION,
            dataset_hashes={m.dataset_id: m.content_hash for m in manifests}, split_id=split.split_id,
            seed=self.seed, config=config, global_counts=global_counts, domain_counts=domain,
            institution_counts=institutions, role_counts=roles, actor_counts=actors,
            feature_effects=effects,
            policy_family_weights={"random_utility": 0.3, "quantal_response": 0.25,
                                   "habit": 0.15, "obligation": 0.1,
                                   "risk_sensitive": 0.1, "limited_depth_reasoning": 0.1},
            uncertainty={"counts": "Dirichlet posterior", "individual": "partial pooling",
                         "action_sets": "row-specific; hypotheses retained when supplied"},
            fit_diagnostics={"n_train": len(train), "n_invalid_preserved": len(invalid_rows),
                             "invalid_rows": invalid_rows, "test_labels_touched": False},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            code_commit=code_commit,
        )
        return artifact.seal()


def _inc(mapping, action, amount=1.0):
    mapping[action] = mapping.get(action, 0) + amount


def _inc_nested(mapping, group, action, amount=1.0):
    mapping.setdefault(group, {})[action] = mapping.setdefault(group, {}).get(action, 0) + amount


def artifact_parameter_pack(artifact: HierarchicalPolicyArtifact, *, dataset_id: str = "") -> dict:
    """Bind a fitted artifact to the universal ActorPolicyModel parameter-pack contract."""
    if not artifact.verify():
        raise ValueError("corrupt policy artifact checksum")

    def intercepts(counts):
        actions = sorted(artifact.global_counts)
        probs = _posterior_mean(counts, actions, float(artifact.config["alpha"]))
        return {a: math.log(max(1e-12, p)) for a, p in probs.items()}

    domain_counts = artifact.domain_counts.get(dataset_id, {}) if dataset_id else artifact.global_counts
    return {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "pack_id": f"phase4:{artifact.artifact_id}",
        "family_id": "actor_policy:regime_mixture", "domain": dataset_id or "multi_domain",
        "population": "training_split_reference_class",
        "source": "fitted_hierarchical_trajectory_model",
        # Fitting alone is not validation.  A promotion job may replace this only
        # after its held-out and transfer gates pass.
        "support_grade": "experimental_fitted", "precision": 1.0,
        "partial_pool_strength": artifact.config["actor_pool_strength"],
        "global": {"success": {"mean": 1.0, "sd": 0.3}},
        "action_intercepts": intercepts(domain_counts or artifact.global_counts),
        "role_action_intercepts": {role: intercepts(counts) for role, counts in artifact.role_counts.items()},
        "actor_action_intercepts": {actor: intercepts(counts) for actor, counts in artifact.actor_counts.items()},
        "policy_family_weights": dict(artifact.policy_family_weights),
        "uncertainty": dict(artifact.uncertainty),
        "fitted_on": artifact.split_id + ":train_only",
        "fit_method": "hierarchical_empirical_bayes_partial_pooling",
        "transport_note": "unvalidated outside the dataset/domain encoded by this pack",
        "fallbacks": [{"tier": 5, "reason": "fitted reference-class pack pending empirical promotion",
                       "uncertainty_widening": 1.0}],
        "artifact_checksum": artifact.checksum,
    }


class HierarchicalPolicyPredictor:
    def __init__(self, artifact: HierarchicalPolicyArtifact):
        if not artifact.verify():
            raise ValueError("corrupt policy artifact checksum")
        self.artifact = artifact

    def predict(self, row: TrajectoryRecord, *, ablations: set | None = None) -> dict:
        ablations = ablations or set()
        actions = list(row.candidate_actions)
        if not actions:
            raise ValueError("candidate action set cannot be empty")
        alpha = float(self.artifact.config["alpha"])
        global_dist = _posterior_mean(self.artifact.global_counts, actions, alpha)
        dist = global_dist
        levels = [
            (self.artifact.domain_counts.get(row.dataset_id, {}), 40.0),
            (self.artifact.institution_counts.get(row.institution_id or "__missing__", {}),
             self.artifact.config["institution_pool_strength"]),
            (self.artifact.role_counts.get(row.actor_role or "__missing__", {}),
             self.artifact.config["role_pool_strength"]),
        ]
        if "no_person_shrinkage" not in ablations and "no_actor_history" not in ablations:
            levels.append((self.artifact.actor_counts.get(row.actor_id, {}),
                           self.artifact.config["actor_pool_strength"]))
        for counts, strength in levels:
            n = sum(counts.get(a, 0) for a in actions)
            if n <= 0:
                continue
            local = _posterior_mean(counts, actions, alpha)
            w = n / (n + float(strength))
            dist = {a: w * local[a] + (1.0 - w) * dist[a] for a in actions}
        if "no_features" not in ablations:
            scores = {}
            for action in actions:
                shift = 0.0
                for feature, value in row.actor_view_features.items():
                    if not isinstance(value, (int, float)):
                        continue
                    mean = self.artifact.feature_effects.get(f"{action}:{feature}")
                    if mean is not None:
                        shift += 0.1 * (float(value) - float(mean))
                scores[action] = math.log(max(1e-12, dist[action])) + max(-2.0, min(2.0, shift))
            dist = _softmax(scores)
        z = sum(dist.values()) or 1.0
        return {a: max(0.0, dist[a]) / z for a in actions}


def _posterior_mean(counts, actions, alpha):
    z = sum(counts.get(a, 0) + alpha for a in actions)
    return {a: (counts.get(a, 0) + alpha) / z for a in actions}


def _softmax(scores):
    m = max(scores.values())
    weights = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(weights.values()) or 1.0
    return {k: v / z for k, v in weights.items()}


@dataclass
class CalibrationArtifact:
    method: str
    temperature: float
    fit_split_id: str
    n_calibration: int
    pre_log_loss: float
    post_log_loss: float
    checksum: str = ""

    def seal(self):
        self.checksum = digest({k: v for k, v in asdict(self).items() if k != "checksum"})
        return self


def fit_temperature(predictions: list[dict], labels: list[str], split_id: str,
                    weights: list[float] | None = None) -> CalibrationArtifact:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("calibration predictions/labels must be non-empty and aligned")
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    if len(weights) != len(labels) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("calibration weights must be aligned, finite, and positive")
    grid = [0.4 + 0.05 * i for i in range(33)]
    losses = {}
    for temp in grid:
        calibrated = [_temperature(p, temp) for p in predictions]
        losses[temp] = log_loss(calibrated, labels, weights)
    best = min(losses, key=losses.get)
    return CalibrationArtifact("temperature_scaling", best, split_id, len(labels),
                               log_loss(predictions, labels, weights), losses[best]).seal()


def apply_calibration(probabilities: dict, artifact: CalibrationArtifact) -> dict:
    expected = digest({k: v for k, v in asdict(artifact).items() if k != "checksum"})
    if expected != artifact.checksum:
        raise ValueError("corrupt calibration artifact checksum")
    return _temperature(probabilities, artifact.temperature)


def _temperature(probabilities, temp):
    logits = {a: math.log(max(1e-12, p)) / temp for a, p in probabilities.items()}
    return _softmax(logits)


def log_loss(predictions, labels, weights=None):
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    z = sum(weights) or 1.0
    return -sum(w * math.log(max(1e-12, p.get(y, 0.0)))
                for p, y, w in zip(predictions, labels, weights)) / z


def multiclass_brier(predictions, labels, weights=None):
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    rows = []
    for probabilities, label, weight in zip(predictions, labels, weights):
        actions = set(probabilities) | {label}
        rows.append(weight * sum((probabilities.get(a, 0.0) - (1.0 if a == label else 0.0)) ** 2
                                 for a in actions))
    return sum(rows) / (sum(weights) or 1.0)


def expected_calibration_error(predictions, labels, bins=10, weights=None):
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    bucket = [[] for _ in range(bins)]
    for p, y, weight in zip(predictions, labels, weights):
        action, confidence = max(p.items(), key=lambda item: item[1])
        bucket[min(bins - 1, int(confidence * bins))].append((confidence, action == y, weight))
    total = sum(weights) or 1.0
    return sum(sum(w for _, _, w in b) / total *
               abs(sum(c * w for c, _, w in b) / sum(w for _, _, w in b) -
                   sum(float(ok) * w for _, ok, w in b) / sum(w for _, _, w in b))
               for b in bucket if b)


def reliability_data(predictions, labels, bins=10, weights=None):
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        rows = []
        for p, y, weight in zip(predictions, labels, weights):
            action, confidence = max(p.items(), key=lambda item: item[1])
            if lo <= confidence < hi or (i == bins - 1 and confidence == 1.0):
                rows.append((confidence, action == y, weight))
        total = sum(w for _, _, w in rows)
        out.append({"lo": lo, "hi": hi, "n": len(rows), "weight": total,
                    "confidence": sum(x * w for x, _, w in rows) / total if rows else None,
                    "accuracy": sum(float(x) * w for _, x, w in rows) / total if rows else None})
    return out


def evaluate_predictions(predictions, labels, candidate_sets=None, weights=None) -> dict:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("evaluation inputs must be non-empty and aligned")
    top = [max(p, key=p.get) for p in predictions]
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    total_weight = sum(weights) or 1.0
    invalid = 0
    if candidate_sets is not None:
        invalid = sum(w for t, actions, w in zip(top, candidate_sets, weights) if t not in actions)
    classes = sorted(set(labels) | {a for p in predictions for a in p})
    confusion = {a: {b: 0 for b in classes} for a in classes}
    for actual, predicted in zip(labels, top):
        confusion[actual][predicted] += 1
    entropy = [-sum(x * math.log(max(1e-12, x)) for x in p.values()) for p in predictions]
    return {
        "n": len(labels), "effective_weight": total_weight,
        "log_loss": log_loss(predictions, labels, weights),
        "multiclass_brier": multiclass_brier(predictions, labels, weights),
        "ece": expected_calibration_error(predictions, labels, weights=weights),
        "top1_accuracy": sum(w for a, b, w in zip(labels, top, weights) if a == b) / total_weight,
        "invalid_action_rate": invalid / total_weight, "confusion_matrix": confusion,
        "mean_entropy": sum(e * w for e, w in zip(entropy, weights)) / total_weight,
        "reliability": reliability_data(predictions, labels, weights=weights),
    }


def paired_bootstrap(pred_a, pred_b, labels, *, metric="log_loss", n_boot=1000, seed=0,
                     weights=None):
    if len(labels) < 2:
        return {"mean": None, "ci95": [None, None], "n": len(labels)}
    weights = list(weights) if weights is not None else [1.0] * len(labels)
    if len(weights) != len(labels) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("bootstrap weights must be aligned, finite, and positive")
    rng = random.Random(seed)

    def row_loss(p, y):
        if metric == "brier":
            return sum((p.get(a, 0.0) - (1.0 if a == y else 0.0)) ** 2 for a in set(p) | {y})
        return -math.log(max(1e-12, p.get(y, 0.0)))

    diffs = [row_loss(a, y) - row_loss(b, y) for a, b, y in zip(pred_a, pred_b, labels)]
    total_weight = sum(weights) or 1.0
    samples = []
    for _ in range(n_boot):
        # Resample observational rows (or aggregate exposure arms) as clusters,
        # then retain their frequency weights inside the replicate.
        chosen = [rng.randrange(len(diffs)) for _ in diffs]
        z = sum(weights[i] for i in chosen) or 1.0
        samples.append(sum(diffs[i] * weights[i] for i in chosen) / z)
    samples.sort()
    return {"mean": sum(d * w for d, w in zip(diffs, weights)) / total_weight,
            "ci95": [samples[int(0.025 * n_boot)], samples[min(n_boot - 1, int(0.975 * n_boot))]],
            "n": len(diffs), "effective_weight": total_weight,
            "resampling_unit": "trajectory_row_with_frequency_weight"}


_ARTIFACT_LOCK = threading.RLock()


def write_artifact(path, payload: dict):
    """Atomic, checksum-sealed artifact write safe under concurrent evaluators."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["artifact_checksum"] = digest(payload)
    with _ARTIFACT_LOCK:
        fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(body, handle, indent=2, sort_keys=True, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def read_artifact(path):
    body = json.loads(Path(path).read_text())
    checksum = body.pop("artifact_checksum", "")
    if checksum != digest(body):
        raise ValueError(f"corrupt artifact: {path}")
    return body


class ResumableEvaluation:
    """Per-row JSONL evaluator with bounded wall-clock handling and resume."""

    def __init__(self, path, *, timeout_s=300.0):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed_ids(self):
        if not self.path.exists():
            return set()
        out = set()
        for line in self.path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("checksum") == digest({k: v for k, v in row.items() if k != "checksum"}):
                out.add(row.get("record_id"))
        return out

    def run(self, records, fn):
        done = self.completed_ids()
        started = time.monotonic()
        with self.path.open("a") as handle:
            for record in records:
                if record.record_id in done:
                    continue
                if time.monotonic() - started > self.timeout_s:
                    raise TimeoutError(f"evaluation exceeded {self.timeout_s}s after preserving progress")
                result = {"record_id": record.record_id, "result": fn(record)}
                result["checksum"] = digest(result)
                handle.write(json.dumps(result, sort_keys=True, default=str) + "\n")
                handle.flush()
        return self.completed_ids()
