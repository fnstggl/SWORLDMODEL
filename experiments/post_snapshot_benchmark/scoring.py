"""Governed Phase-12 fitting and single-open locked benchmark scoring.

The three commands in this module deliberately mirror the benchmark's
chronological governance boundary:

``fit-calibration`` reads calibration outcomes only after all calibration
forecasts are frozen; ``select-validation`` reads validation outcomes only to
choose among the already-fitted candidates; and ``score-locked`` validates all
forecast and baseline hashes before exclusively creating the locked-outcome
access ledger and reading that store once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.post_snapshot_benchmark.forecast import PHASES, _freeze_hash
from swm.world_model_v2.calibration import (
    _brier,
    _logloss,
    ece,
    fit_beta,
    fit_isotonic,
    fit_platt,
    reliability_table,
)


SCHEMA_VERSION = 1
CLIP = 1e-6
GLOBAL_METHODS = ("identity", "platt", "beta", "isotonic")
CONDITIONED_METHODS = (
    "hierarchical_conditioned",
    "task_family_conditioned",
    "horizon_conditioned",
    "support_conditioned",
)
BASELINE_ARMS = (
    "constant_0_50",
    "direct_single",
    "call_matched_direct_ensemble",
    "observer_panel",
    "analogical_retrieval",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verify_forecasts(path: Path, split: str, expected: int) -> list[dict]:
    rows = _load_jsonl(path)
    errors = []
    seen = set()
    for index, row in enumerate(rows):
        key = (row.get("event_id"), row.get("forecast_cutoff"))
        frozen = {k: v for k, v in row.items() if k != "forecast_sha256"}
        if row.get("forecast_sha256") != _freeze_hash(frozen):
            errors.append(f"row {index}: forecast hash mismatch")
        if key in seen:
            errors.append(f"row {index}: duplicate forecast key {key}")
        seen.add(key)
        if row.get("split") != split:
            errors.append(f"row {index}: split={row.get('split')!r}")
        if not row.get("full_system_qualified"):
            errors.append(f"row {index}: full_system_qualified is not true")
        records = row.get("phase_execution_records") or {}
        if set(records) != set(PHASES):
            errors.append(f"row {index}: phase record coverage {len(records)}/11")
        if row.get("blocked_relevant_phases"):
            errors.append(f"row {index}: blocked relevant phase")
        if row.get("terminal_source") != "terminal_world_states":
            errors.append(f"row {index}: invalid terminal source")
        if row.get("resolution_inaccessible") is not True:
            errors.append(f"row {index}: resolution isolation not asserted")
        if not isinstance(row.get("p_yes"), (int, float)):
            errors.append(f"row {index}: missing p_yes")
    if len(rows) != expected:
        errors.append(f"expected {expected} forecast rows, found {len(rows)}")
    if errors:
        raise RuntimeError("forecast freeze verification failed: " + "; ".join(errors[:20]))
    return rows


def _verify_baselines(path: Path, split: str, expected: int, forecasts: list[dict]) -> list[dict]:
    rows = _load_jsonl(path)
    errors = []
    expected_capsules = {
        (row["event_id"], row["forecast_cutoff"]): row["evidence_capsule_sha256"]
        for row in forecasts
    }
    seen = set()
    for index, row in enumerate(rows):
        key = (row.get("event_id"), row.get("forecast_cutoff"))
        frozen = {k: v for k, v in row.items() if k != "baseline_sha256"}
        if row.get("baseline_sha256") != _freeze_hash(frozen):
            errors.append(f"row {index}: baseline hash mismatch")
        if key in seen:
            errors.append(f"row {index}: duplicate baseline key {key}")
        seen.add(key)
        if row.get("split") != split:
            errors.append(f"row {index}: split mismatch")
        if not row.get("all_required_model_arms_complete"):
            errors.append(f"row {index}: incomplete required model arm")
        if not row.get("identical_evidence_for_all_model_arms"):
            errors.append(f"row {index}: evidence parity false")
        if not row.get("call_matched_ensemble_within_v2_budget"):
            errors.append(f"row {index}: ensemble exceeds V2 budget")
        if row.get("evidence_capsule_sha256") != expected_capsules.get(key):
            errors.append(f"row {index}: evidence capsule mismatch")
        for arm in BASELINE_ARMS:
            if not isinstance((row.get("arms") or {}).get(arm, {}).get("p_yes"), (int, float)):
                errors.append(f"row {index}: missing {arm}")
    if len(rows) != expected:
        errors.append(f"expected {expected} baseline rows, found {len(rows)}")
    if errors:
        raise RuntimeError("baseline freeze verification failed: " + "; ".join(errors[:20]))
    return rows


def _read_resolution_store(path: Path, expected_split: str) -> tuple[dict, bytes]:
    data = path.read_bytes()
    payload = json.loads(data)
    if payload.get("split") != expected_split or not isinstance(payload.get("resolutions"), dict):
        raise RuntimeError(f"invalid {expected_split} resolution store")
    return payload["resolutions"], data


def _append_access_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = []
    if path.exists():
        prior = json.loads(path.read_text()).get("accesses", [])
    prior.append(record)
    _atomic_json(path, {"schema_version": SCHEMA_VERSION, "accesses": prior})


def _join(rows: list[dict], resolutions: dict) -> list[tuple[dict, int]]:
    joined = []
    missing = []
    for row in rows:
        resolution = resolutions.get(row["event_id"])
        outcome = None if resolution is None else resolution.get("outcome")
        if outcome not in (0, 1):
            missing.append(row["event_id"])
        else:
            joined.append((row, int(outcome)))
    if missing:
        raise RuntimeError(f"missing binary outcomes for {len(set(missing))} worlds")
    return joined


def _fit_base(method: str, pairs: list[tuple[float, int]], fitted_on: str) -> dict:
    if method == "identity":
        return {"method": "identity", "n_fit": len(pairs), "fitted_on": fitted_on}
    if method == "platt":
        return {"method": method, **asdict(fit_platt(pairs, fitted_on=fitted_on))}
    if method == "beta":
        return {"method": method, **asdict(fit_beta(pairs, fitted_on=fitted_on))}
    if method == "isotonic":
        return {"method": method, **asdict(fit_isotonic(pairs, fitted_on=fitted_on))}
    raise ValueError(method)


def _apply_base(spec: dict, p: float) -> float:
    p = min(1.0 - CLIP, max(CLIP, float(p)))
    method = spec["method"]
    if method == "identity":
        return p
    if method == "platt":
        z = float(spec["a"]) * math.log(p / (1.0 - p)) + float(spec["b"])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    if method == "beta":
        z = (float(spec["a"]) * math.log(p) - float(spec["b"]) * math.log(1.0 - p)
             + float(spec["c"]))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    if method == "isotonic":
        xs, ys = spec.get("xs") or [], spec.get("ys") or []
        if not xs:
            return p
        if p <= xs[0]:
            return float(ys[0])
        if p >= xs[-1]:
            return float(ys[-1])
        for index in range(1, len(xs)):
            if p <= xs[index]:
                x0, x1 = float(xs[index - 1]), float(xs[index])
                y0, y1 = float(ys[index - 1]), float(ys[index])
                weight = (p - x0) / max(CLIP, x1 - x0)
                return y0 + weight * (y1 - y0)
    raise ValueError(method)


def _horizon_key(row: dict) -> str:
    index = int(row.get("cutoff_index", 0))
    return ("early" if index == 0 else "middle" if index in (1, 2) else "late")


def _support_key(row: dict) -> str:
    return str(row.get("support_grade") or "unknown")


def _candidate_key(row: dict, keying: str) -> str:
    if keying == "task_family":
        return str(row.get("domain") or "unknown")
    if keying == "horizon":
        return _horizon_key(row)
    if keying == "support":
        return _support_key(row)
    if keying == "hierarchical":
        return f"{row.get('domain', 'unknown')}|{_horizon_key(row)}|{_support_key(row)}"
    raise ValueError(keying)


def _fit_conditioned(rows_with_y: list[tuple[dict, int]], *, keying: str,
                     name: str, min_cell: int = 40) -> dict:
    pairs = [(float(row["p_yes"]), y) for row, y in rows_with_y]
    by_key = defaultdict(list)
    for row, y in rows_with_y:
        by_key[_candidate_key(row, keying)].append((float(row["p_yes"]), y))
    return {
        "method": "conditioned_platt",
        "name": name,
        "keying": keying,
        "min_cell": min_cell,
        "n_fit": len(pairs),
        "global": _fit_base("platt", pairs, "calibration:global"),
        "cells": {key: _fit_base("platt", values, f"calibration:{name}:{key}")
                  for key, values in sorted(by_key.items())},
        "cell_counts": {key: len(values) for key, values in sorted(by_key.items())},
        "partial_pooling": "linear cell/global blend below min_cell",
    }


def _apply_candidate(spec: dict, row: dict) -> float:
    if spec.get("method") != "conditioned_platt":
        return _apply_base(spec, float(row["p_yes"]))
    key = _candidate_key(row, spec["keying"])
    global_p = _apply_base(spec["global"], float(row["p_yes"]))
    cell = spec.get("cells", {}).get(key)
    count = int(spec.get("cell_counts", {}).get(key, 0))
    if not cell:
        return global_p
    weight = min(1.0, count / max(1, int(spec["min_cell"])))
    return weight * _apply_base(cell, float(row["p_yes"])) + (1.0 - weight) * global_p


def _binary_entropy(p: float) -> float:
    p = min(1.0 - CLIP, max(CLIP, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def _support_features(row: dict) -> list[float]:
    records = row.get("phase_execution_records") or {}
    active = sum(record.get("execution_status") == "causally_active" for record in records.values())
    cutoff = datetime.fromisoformat(row["forecast_cutoff"].replace("Z", "+00:00"))
    resolution = datetime.fromisoformat(row["resolution_time"].replace("Z", "+00:00"))
    horizon_days = max(0.0, (resolution - cutoff).total_seconds() / 86400.0)
    return [
        1.0,
        min(1.0, len(row.get("evidence_byte_hashes") or []) / 8.0),
        min(1.0, horizon_days / 30.0),
        min(1.0, float(row.get("model_calls") or 0) / 10.0),
        min(1.0, len(row.get("limitations") or []) / 10.0),
        active / max(1, len(PHASES)),
        _binary_entropy(float(row["p_yes"])),
    ]


def _fit_support_model(rows_with_y: list[tuple[dict, int]], *, iters: int = 800,
                       lr: float = 0.15, l2: float = 0.01) -> dict:
    # Target is forecast support, not event outcome: error <= 0.25.
    examples = [(_support_features(row), int(abs(float(row["p_yes"]) - y) <= 0.25))
                for row, y in rows_with_y]
    coefficients = [0.0] * 7
    for _ in range(iters):
        gradient = [0.0] * len(coefficients)
        for features, target in examples:
            z = sum(a * b for a, b in zip(coefficients, features))
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            for index, value in enumerate(features):
                gradient[index] += (probability - target) * value
        for index in range(len(coefficients)):
            penalty = 0.0 if index == 0 else l2 * coefficients[index]
            coefficients[index] -= lr * (gradient[index] / max(1, len(examples)) + penalty)
    return {
        "method": "logistic_probability_of_absolute_error_at_most_0.25",
        "feature_names": ["intercept", "evidence_density", "horizon_days_30",
                          "model_calls_10", "limitation_density", "active_phase_fraction",
                          "raw_binary_entropy"],
        "coefficients": [round(value, 8) for value in coefficients],
        "fit_split": "calibration",
        "n_rows": len(examples),
        "n_worlds": len({row["event_world_cluster"] for row, _ in rows_with_y}),
        "grade_thresholds": {"empirically_supported": 0.75, "transfer_supported": 0.55,
                             "exploratory": 0.35, "highly_speculative": 0.0},
    }


def _support_probability(model: dict, row: dict) -> float:
    z = sum(float(a) * b for a, b in zip(model["coefficients"], _support_features(row)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _metrics(pairs: list[tuple[float, int]]) -> dict:
    if not pairs:
        return {"n_rows": 0}
    probabilities = [float(p) for p, _ in pairs]
    positives = [p for p, y in pairs if y == 1]
    negatives = [p for p, y in pairs if y == 0]
    if positives and negatives:
        wins = sum(a > b for a in positives for b in negatives)
        ties = sum(a == b for a in positives for b in negatives)
        auroc = (wins + 0.5 * ties) / (len(positives) * len(negatives))
    else:
        auroc = None
    return {
        "n_rows": len(pairs),
        "brier": round(float(_brier(pairs)), 8),
        "log_loss": round(float(_logloss(pairs)), 8),
        "auroc": None if auroc is None else round(auroc, 8),
        "directional_accuracy": round(sum((p >= 0.5) == bool(y) for p, y in pairs) / len(pairs), 8),
        "ece": ece(pairs),
        "calibration_slope": _calibration_slope(pairs),
        "sharpness_mean_abs_from_half": round(sum(abs(p - 0.5) for p in probabilities) / len(pairs), 8),
        "reliability": reliability_table(pairs),
    }


def _calibration_slope(pairs: list[tuple[float, int]]) -> float | None:
    if len({y for _, y in pairs}) < 2:
        return None
    intercept, slope = 0.0, 1.0
    xs = [math.log(min(1 - CLIP, max(CLIP, p)) /
                   (1.0 - min(1 - CLIP, max(CLIP, p)))) for p, _ in pairs]
    for _ in range(800):
        gi = gs = 0.0
        for x, (_, y) in zip(xs, pairs):
            q = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, intercept + slope * x))))
            gi += q - y
            gs += (q - y) * x
        intercept -= 0.05 * gi / len(pairs)
        slope -= 0.05 * gs / len(pairs)
    return round(slope, 8)


def _cluster_bootstrap(differences: list[tuple[str, float]], *, draws: int = 2000,
                       seed: int = 12012) -> dict:
    by_cluster = defaultdict(list)
    for cluster, value in differences:
        by_cluster[cluster].append(float(value))
    clusters = sorted(by_cluster)
    if not clusters:
        return {"paired_mean": None, "ci95": [None, None], "n_worlds": 0, "n_rows": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(clusters) for _ in clusters]
        values = [value for cluster in chosen for value in by_cluster[cluster]]
        samples.append(sum(values) / len(values))
    samples.sort()
    observed = sum(value for _, value in differences) / len(differences)
    return {
        "paired_mean": round(observed, 8),
        "ci95": [round(samples[int(0.025 * draws)], 8),
                 round(samples[min(draws - 1, int(0.975 * draws))], 8)],
        "n_worlds": len(clusters),
        "n_rows": len(differences),
        "bootstrap_draws": draws,
        "cluster_unit": "event_world_cluster",
    }


def _comparison(rows: list[dict], left: str, right: str) -> dict:
    eligible = [row for row in rows if isinstance(row["probabilities"].get(left), (int, float))
                and isinstance(row["probabilities"].get(right), (int, float))]
    brier = []
    logloss = []
    for row in eligible:
        y = row["outcome"]
        a = min(1 - CLIP, max(CLIP, row["probabilities"][left]))
        b = min(1 - CLIP, max(CLIP, row["probabilities"][right]))
        brier.append((row["event_world_cluster"], (a - y) ** 2 - (b - y) ** 2))
        la = -(y * math.log(a) + (1 - y) * math.log(1 - a))
        lb = -(y * math.log(b) + (1 - y) * math.log(1 - b))
        logloss.append((row["event_world_cluster"], la - lb))
    return {
        "left": left,
        "right": right,
        "difference_sign": "negative_favors_left",
        "brier_difference": _cluster_bootstrap(brier),
        "log_loss_difference": _cluster_bootstrap(logloss, seed=12013),
    }


def fit_calibration(args) -> dict:
    forecasts = _verify_forecasts(args.forecasts, "calibration", 160)
    resolutions, resolution_bytes = _read_resolution_store(args.outcomes, "calibration")
    joined = _join(forecasts, resolutions)
    candidates = {method: _fit_base(method, [(float(row["p_yes"]), y) for row, y in joined],
                                            "calibration")
                  for method in GLOBAL_METHODS}
    keyings = {
        "hierarchical_conditioned": "hierarchical",
        "task_family_conditioned": "task_family",
        "horizon_conditioned": "horizon",
        "support_conditioned": "support",
    }
    for name, keying in keyings.items():
        candidates[name] = _fit_conditioned(joined, keying=keying, name=name)
    world_outcomes = {}
    world_domains = {}
    for row, y in joined:
        world_outcomes[row["event_id"]] = y
        world_domains[row["event_id"]] = row.get("domain", "unknown")
    global_rate = sum(world_outcomes.values()) / len(world_outcomes)
    domains = defaultdict(list)
    for event_id, y in world_outcomes.items():
        domains[world_domains[event_id]].append(y)
    domain_rates = {
        domain: {"n_worlds": len(values), "raw_rate": sum(values) / len(values),
                 "partial_pooled_rate": (sum(values) + 5 * global_rate) / (len(values) + 5)}
        for domain, values in sorted(domains.items())
    }
    support_model = _fit_support_model(joined)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "stage": "phase12_calibration_fit",
        "created_at": _now(),
        "governance": {"fit_split": "calibration", "validation_outcomes_accessed": False,
                       "locked_outcomes_accessed": False},
        "inputs": {"forecast_path": str(args.forecasts), "forecast_sha256": _sha_file(args.forecasts),
                   "resolution_store_sha256": _sha_bytes(resolution_bytes)},
        "counts": {"forecast_rows": len(forecasts), "independent_worlds": len(world_outcomes)},
        "candidates": candidates,
        "base_rates": {"global": global_rate, "domains": domain_rates,
                       "fit_unit": "independent_event_world"},
        "support_model_fit": support_model,
    }
    artifact["artifact_sha256"] = _freeze_hash(artifact)
    _atomic_json(args.output, artifact)
    _append_access_log(args.access_log, {
        "accessed_at": _now(), "split": "calibration", "purpose": "phase12_candidate_fit",
        "resolution_store_sha256": _sha_bytes(resolution_bytes), "forecast_rows_verified_before_access": 160,
    })
    return artifact


def select_validation(args) -> dict:
    fit = json.loads(args.fit.read_text())
    if fit.get("artifact_sha256") != _freeze_hash({k: v for k, v in fit.items()
                                                    if k != "artifact_sha256"}):
        raise RuntimeError("calibration-fit artifact hash mismatch")
    forecasts = _verify_forecasts(args.forecasts, "validation", 80)
    resolutions, resolution_bytes = _read_resolution_store(args.outcomes, "validation")
    joined = _join(forecasts, resolutions)
    comparison = []
    for name, candidate in fit["candidates"].items():
        pairs = [(_apply_candidate(candidate, row), y) for row, y in joined]
        comparison.append({"name": name, **_metrics(pairs)})
    identity = next(row for row in comparison if row["name"] == "identity")
    eligible = [row for row in comparison if row["name"] == "identity" or
                (row["brier"] < identity["brier"] and row["log_loss"] < identity["log_loss"])]
    selected = min(eligible, key=lambda row: (row["log_loss"], row["brier"], row["name"]))
    support_model = dict(fit["support_model_fit"])
    support_predictions = [(_support_probability(support_model, row),
                            int(abs(float(row["p_yes"]) - y) <= 0.25)) for row, y in joined]
    support_model.update({
        "validation_split_used_for_evaluation_only": True,
        "validation_n_rows": len(joined),
        "validation_brier": _brier(support_predictions),
        "validation_log_loss": _logloss(support_predictions),
        "frozen_at": _now(),
    })
    support_model["artifact_sha256"] = _freeze_hash(support_model)
    _atomic_json(args.support_output, support_model)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "stage": "phase12_validation_selection",
        "created_at": _now(),
        "governance": {"candidate_fit_split": "calibration", "selection_split": "validation",
                       "locked_outcomes_accessed": False,
                       "promotion_rule": "strictly beat identity on validation Brier and log loss"},
        "inputs": {"calibration_fit_sha256": fit["artifact_sha256"],
                   "forecast_sha256": _sha_file(args.forecasts),
                   "resolution_store_sha256": _sha_bytes(resolution_bytes)},
        "counts": {"forecast_rows": len(forecasts),
                   "independent_worlds": len({row["event_world_cluster"] for row, _ in joined})},
        "selected": selected["name"],
        "selected_candidate": fit["candidates"][selected["name"]],
        "identity_fallback_selected": selected["name"] == "identity",
        "comparison": comparison,
        "support_model_sha256": support_model["artifact_sha256"],
    }
    artifact["artifact_sha256"] = _freeze_hash(artifact)
    _atomic_json(args.output, artifact)
    _append_access_log(args.access_log, {
        "accessed_at": _now(), "split": "validation", "purpose": "phase12_method_selection",
        "resolution_store_sha256": _sha_bytes(resolution_bytes), "forecast_rows_verified_before_access": 80,
    })
    return artifact


def _exclusive_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def score_locked(args) -> dict:
    forecasts = _verify_forecasts(args.forecasts, "locked_test", 160)
    baselines = _verify_baselines(args.baselines, "locked_test", 160, forecasts)
    fit = json.loads(args.fit.read_text())
    selection = json.loads(args.selection.read_text())
    support_model = json.loads(args.support_model.read_text())
    if fit.get("artifact_sha256") != _freeze_hash({k: v for k, v in fit.items() if k != "artifact_sha256"}):
        raise RuntimeError("calibration-fit artifact hash mismatch")
    if selection.get("artifact_sha256") != _freeze_hash({k: v for k, v in selection.items()
                                                          if k != "artifact_sha256"}):
        raise RuntimeError("validation-selection artifact hash mismatch")
    if support_model.get("artifact_sha256") != _freeze_hash({k: v for k, v in support_model.items()
                                                              if k != "artifact_sha256"}):
        raise RuntimeError("support-model artifact hash mismatch")
    if selection.get("inputs", {}).get("calibration_fit_sha256") != fit["artifact_sha256"]:
        raise RuntimeError("selection is not bound to calibration fit")
    if selection.get("support_model_sha256") != support_model["artifact_sha256"]:
        raise RuntimeError("selection is not bound to support model")
    baseline_by_key = {(row["event_id"], row["forecast_cutoff"]): row for row in baselines}
    market = json.loads(args.market.read_text())
    preopen = {
        "schema_version": SCHEMA_VERSION,
        "state": "open_started",
        "opened_at": _now(),
        "purpose": "single final locked-test scoring",
        "outcome_path": str(args.outcomes),
        "all_forecast_hashes_verified_before_open": True,
        "all_baseline_hashes_verified_before_open": True,
        "forecast_rows": 160,
        "baseline_rows": 160,
        "forecast_file_sha256": _sha_file(args.forecasts),
        "baseline_file_sha256": _sha_file(args.baselines),
        "calibration_fit_sha256": fit["artifact_sha256"],
        "validation_selection_sha256": selection["artifact_sha256"],
        "support_model_sha256": support_model["artifact_sha256"],
    }
    _exclusive_ledger(args.ledger, preopen)
    # This is the only locked resolution-store read in the benchmark scorer.
    resolutions, resolution_bytes = _read_resolution_store(args.outcomes, "locked_test")
    joined = _join(forecasts, resolutions)
    selected = selection["selected_candidate"]
    scored_rows = []
    for forecast, outcome in joined:
        key = (forecast["event_id"], forecast["forecast_cutoff"])
        baseline = baseline_by_key[key]
        domain_record = fit["base_rates"]["domains"].get(forecast.get("domain"), {})
        domain_rate = domain_record.get("partial_pooled_rate", fit["base_rates"]["global"])
        market_record = market.get("snapshots", {}).get(forecast["event_id"], {}).get(
            forecast["forecast_cutoff"], {})
        probabilities = {
            "v2_raw": float(forecast["p_yes"]),
            "v2_calibrated": _apply_candidate(selected, forecast),
            "domain_base_rate": float(domain_rate),
            **{name: float(baseline["arms"][name]["p_yes"]) for name in BASELINE_ARMS},
            "market_midpoint": (float(market_record["midpoint"])
                                if isinstance(market_record.get("midpoint"), (int, float)) else None),
            "v2_market_informed": None,
        }
        support_probability = _support_probability(support_model, forecast)
        grade = "highly_speculative"
        for candidate_grade in ("empirically_supported", "transfer_supported", "exploratory"):
            if support_probability >= support_model["grade_thresholds"][candidate_grade]:
                grade = candidate_grade
                break
        scored_rows.append({
            "event_id": forecast["event_id"],
            "event_world_cluster": forecast["event_world_cluster"],
            "domain": forecast.get("domain"),
            "forecast_cutoff": forecast["forecast_cutoff"],
            "outcome": outcome,
            "probabilities": probabilities,
            "support_probability": support_probability,
            "frozen_support_grade": grade,
            "forecast_sha256": forecast["forecast_sha256"],
            "baseline_sha256": baseline["baseline_sha256"],
            "model_calls": forecast.get("model_calls"),
            "latency_s": forecast.get("latency_s"),
            "market_snapshot_available": probabilities["market_midpoint"] is not None,
        })
    arms = list(scored_rows[0]["probabilities"])
    metrics = {}
    for arm in arms:
        pairs = [(row["probabilities"][arm], row["outcome"]) for row in scored_rows
                 if isinstance(row["probabilities"].get(arm), (int, float))]
        metrics[arm] = {
            **_metrics(pairs),
            "n_worlds": len({row["event_world_cluster"] for row in scored_rows
                             if isinstance(row["probabilities"].get(arm), (int, float))}),
        }
    comparisons = [
        _comparison(scored_rows, "v2_raw", "constant_0_50"),
        _comparison(scored_rows, "v2_calibrated", "constant_0_50"),
        _comparison(scored_rows, "v2_raw", "domain_base_rate"),
        _comparison(scored_rows, "v2_raw", "direct_single"),
        _comparison(scored_rows, "v2_raw", "call_matched_direct_ensemble"),
        _comparison(scored_rows, "v2_raw", "observer_panel"),
        _comparison(scored_rows, "v2_raw", "analogical_retrieval"),
        _comparison(scored_rows, "v2_raw", "market_midpoint"),
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "stage": "locked_test_final_score",
        "scored_at": _now(),
        "temporal_governance": {"locked_store_open_count": 1,
                                "locked_outcomes_used_for_tuning": False,
                                "calibrator_fit_split": "calibration",
                                "calibrator_selection_split": "validation"},
        "counts": {"forecast_rows": len(scored_rows),
                   "independent_worlds": len({row["event_world_cluster"] for row in scored_rows}),
                   "market_rows": sum(row["market_snapshot_available"] for row in scored_rows)},
        "selected_calibrator": selection["selected"],
        "metrics": metrics,
        "comparisons": comparisons,
        "market_informed_comparison": {
            "status": "not_run",
            "reason": "representative V2 arm was prospectively frozen as market-blind; no post-hoc probability blend",
        },
        "cost_and_latency": {
            "v2_model_calls": sum(int(row.get("model_calls") or 0) for row in scored_rows),
            "v2_latency_s": sum(float(row.get("latency_s") or 0.0) for row in scored_rows),
            "usd_cost": None,
            "cost_status": "unavailable: mutable API alias response did not expose auditable billed usage",
        },
        "input_hashes": {"forecasts": preopen["forecast_file_sha256"],
                         "baselines": preopen["baseline_file_sha256"],
                         "market": _sha_file(args.market),
                         "resolution_store": _sha_bytes(resolution_bytes)},
        "negative_results_preserved": True,
    }
    output["artifact_sha256"] = _freeze_hash(output)
    _atomic_jsonl(args.row_output, scored_rows)
    _atomic_json(args.output, output)
    completed_ledger = {
        **preopen,
        "state": "open_completed",
        "completed_at": _now(),
        "resolution_store_sha256": _sha_bytes(resolution_bytes),
        "score_artifact_sha256": output["artifact_sha256"],
        "scored_row_file_sha256": _sha_file(args.row_output),
        "read_count": 1,
    }
    _atomic_json(args.ledger, completed_ledger)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-calibration")
    fit.add_argument("--forecasts", type=Path, required=True)
    fit.add_argument("--outcomes", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--access-log", type=Path, required=True)
    select = subparsers.add_parser("select-validation")
    select.add_argument("--forecasts", type=Path, required=True)
    select.add_argument("--outcomes", type=Path, required=True)
    select.add_argument("--fit", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--support-output", type=Path, required=True)
    select.add_argument("--access-log", type=Path, required=True)
    score = subparsers.add_parser("score-locked")
    score.add_argument("--forecasts", type=Path, required=True)
    score.add_argument("--baselines", type=Path, required=True)
    score.add_argument("--fit", type=Path, required=True)
    score.add_argument("--selection", type=Path, required=True)
    score.add_argument("--support-model", type=Path, required=True)
    score.add_argument("--market", type=Path, required=True)
    score.add_argument("--outcomes", type=Path, required=True)
    score.add_argument("--ledger", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--row-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fit-calibration":
        result = fit_calibration(args)
    elif args.command == "select-validation":
        result = select_validation(args)
    else:
        result = score_locked(args)
    print(json.dumps({"stage": result["stage"], "artifact_sha256": result["artifact_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
