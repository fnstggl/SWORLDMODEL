#!/usr/bin/env python3
"""Reproducible Phase 4 empirical-completion runner.

Examples:
  python -m experiments.wmv2_phase4_completion numeric --enron-archive ... --ipd-root ... --voteview-root ...
  python -m experiments.wmv2_phase4_completion collect-llm --api-key-stdin
  python -m experiments.wmv2_phase4_completion score-llm
  python -m experiments.wmv2_phase4_completion finalize
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import sys
import termios
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.world_model_v2.phase4_completion import (
    ACTIONS, DatasetBuild, build_enron_repaired, build_ipd_long, build_voteview_senate,
    fit_dataset_models, score_numeric_arms, stable_hash,
)
from swm.world_model_v2.phase4_learning import (
    apply_calibration, digest, evaluate_predictions, fit_temperature, read_artifact, write_artifact,
)
from swm.world_model_v2.phase4_llm_baselines import (
    LENSES, DeepSeekEnvelopeClient, assert_complete_collection, collect_one,
    collection_manifest, logarithmic_pool, request_identity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "phase4_completion"


def _select_stratified(build: DatasetBuild, split: str, count: int) -> list[dict]:
    candidates = [row for row in build.examples if row.split == split]
    strata = defaultdict(list)
    for row in candidates:
        strata[row.label].append(row)
    for label in strata:
        strata[label].sort(key=lambda row: stable_hash("404", build.dataset, row.record_key))
    selected, offsets = [], Counter()
    action_order = list(ACTIONS[build.dataset])
    while len(selected) < min(count, len(candidates)):
        progressed = False
        for action in action_order:
            if offsets[action] < len(strata[action]) and len(selected) < count:
                selected.append(strata[action][offsets[action]])
                offsets[action] += 1
                progressed = True
        if not progressed:
            break
    selected_counts = Counter(row.label for row in selected)
    population_counts = Counter(row.label for row in candidates)
    return [{
        "dataset": build.dataset, "split": split, "record_key": row.record_key,
        "label": row.label, "actions": list(row.actions), "packet": row.llm_packet(),
        "selection_hash": stable_hash("404", build.dataset, row.record_key),
        "inverse_sampling_weight": population_counts[row.label] / selected_counts[row.label],
        "population_stratum_n": population_counts[row.label],
        "sample_stratum_n": selected_counts[row.label],
    } for row in selected]


def _prepare_payloads(build: DatasetBuild) -> tuple[dict, dict, dict, list[dict]]:
    manifest = dict(build.manifest)
    split = dict(build.split_manifest)
    actions = dict(build.action_diagnostics)
    llm_rows = _select_stratified(build, "calibration", 32) + _select_stratified(build, "test", 64)
    return manifest, split, actions, llm_rows


def run_numeric(args) -> None:
    output = Path(args.output)
    builders = [
        ("ipd_long", lambda: build_ipd_long(args.ipd_root)),
        ("voteview_senate", lambda: build_voteview_senate(args.voteview_root)),
        ("enron_repaired", lambda: build_enron_repaired(args.enron_archive)),
    ]
    manifests, splits, action_diagnostics, llm_rows = {}, {}, {}, []
    model_packs, summaries, particles, families = {}, {}, {}, {}
    calibrations, uncertainty, reliability, risk, cold, sequences, downstream = (
        {}, {}, {}, {}, {}, {}, {})
    confidence, ablations, baselines, traces, costs = {}, {}, {}, {}, {}
    started_all = time.monotonic()
    for dataset, builder in builders:
        started = time.monotonic()
        print(f"building {dataset}", flush=True)
        build = builder()
        manifest, split, action_diag, requests = _prepare_payloads(build)
        manifests[dataset], splits[dataset], action_diagnostics[dataset] = manifest, split, action_diag
        llm_rows.extend(requests)
        print(f"fitting {dataset}: {len(build.examples)} decisions", flush=True)
        models = fit_dataset_models(build)
        result = score_numeric_arms(build, models)
        write_artifact(output / f"{dataset}_results.json", result)
        model_packs[dataset] = models.artifact()
        summaries[dataset] = {k: v for k, v in result.items()
                              if k not in ("prediction_table", "conformal", "sequence", "downstream")}
        particles[dataset] = result["particle_diagnostics"]
        families[dataset] = models.b7.as_dict()
        calibrations[dataset] = result["calibrators"]
        uncertainty[dataset] = {"conformal": result["conformal"],
                                "particle_diagnostics": result["particle_diagnostics"]}
        reliability[dataset] = {arm: values["calibrated"]["reliability"]
                                for arm, values in result["metrics"].items()}
        risk[dataset] = result["risk_coverage"]
        cold[dataset] = result["cold_start_slices"]
        sequences[dataset] = result["sequence"]
        downstream[dataset] = result["downstream"]
        confidence[dataset] = result["clustered_comparisons"]
        ablations[dataset] = {arm: result["metrics"][arm]
                              for arm in result["metrics"] if arm.startswith("A_")}
        baselines[dataset] = {arm: result["metrics"][arm]
                              for arm in result["metrics"] if not arm.startswith("A_")}
        traces[dataset] = {
            "particle_traces": result["particle_diagnostics"]["test"]["selected_traces"],
            "prediction_traces": result["prediction_table"][:5],
            "execution_invariance": result["execution_invariance"],
        }
        costs[dataset] = {"numeric_elapsed_seconds": result["elapsed_seconds"],
                          "build_fit_score_seconds": time.monotonic() - started,
                          "rows": len(build.examples)}
        del result, models, build
        gc.collect()
    if len(llm_rows) != 288:
        raise RuntimeError(f"frozen LLM sampling produced {len(llm_rows)} packets, expected 288")
    write_artifact(output / "dataset_manifests.json", {
        "schema_version": "wmv2.phase4-completion.dataset-manifests.v1", "datasets": manifests})
    write_artifact(output / "source_hashes_and_citations.json", {
        "schema_version": "wmv2.phase4-completion.sources.v1",
        "sources": {name: value["source"] for name, value in manifests.items()}})
    write_artifact(output / "frozen_split_manifests.json", {
        "schema_version": "wmv2.phase4-completion.splits.v1", "datasets": splits})
    write_artifact(output / "action_set_reconstruction_diagnostics.json", {
        "schema_version": "wmv2.phase4-completion.action-sets.v1", "datasets": action_diagnostics})
    write_artifact(output / "llm_request_manifest.json", {
        "schema_version": "wmv2.phase4-completion.llm-requests.v1", "seed": 404,
        "sampling_rule": "action-stratified round-robin by sha256(404|dataset|record_key)",
        "rows": llm_rows, "n_packets": len(llm_rows),
        "packet_payload_sha256": digest([row["packet"] for row in llm_rows]),
        "labels_excluded_from_packets": all(row["label"] not in json.dumps(row["packet"])
                                             or row["label"] in json.dumps(row["packet"]["candidate_actions"])
                                             for row in llm_rows),
    })
    artifacts = {
        "parameter_packs.json": ("parameter-packs.v1", model_packs),
        "particle_diagnostics.json": ("particle-diagnostics.v1", particles),
        "policy_family_results.json": ("policy-family-results.v1", families),
        "calibration_results.json": ("calibration-results.v1", calibrations),
        "uncertainty_results.json": ("uncertainty-results.v1", uncertainty),
        "reliability_curves.json": ("reliability-curves.v1", reliability),
        "risk_coverage_curves.json": ("risk-coverage.v1", risk),
        "cold_start_results.json": ("cold-start.v1", cold),
        "sequence_results.json": ("sequence-results.v1", sequences),
        "downstream_results.json": ("downstream-results.v1", downstream),
        "ablation_results.json": ("ablation-results.v1", ablations),
        "baseline_results.json": ("baseline-results.v1", baselines),
        "confidence_intervals.json": ("clustered-confidence-intervals.v1", confidence),
        "forensic_traces.json": ("forensic-traces.v1", traces),
        "cost_latency_memory.json": ("cost-latency-memory.v1", costs),
    }
    for filename, (version, data) in artifacts.items():
        write_artifact(output / filename, {"schema_version": "wmv2.phase4-completion." + version,
                                           "datasets": data})
    write_artifact(output / "numeric_summary.json", {
        "schema_version": "wmv2.phase4-completion.numeric-summary.v1", "datasets": summaries,
        "total_elapsed_seconds": time.monotonic() - started_all,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    print(f"numeric completion written to {output}; packets={len(llm_rows)}", flush=True)


def _read_stdin_secret() -> str:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    fd = sys.stdin.fileno()
    old = None
    if os.isatty(fd):
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        print("DeepSeek API credential: ", end="", flush=True)
    try:
        raw = sys.stdin.buffer.readline(514)
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print(flush=True)
    if len(raw) > 513 or not raw.endswith(b"\n"):
        raise ValueError("credential input must be one bounded line")
    raw = raw.rstrip(b"\r\n")
    try:
        secret = raw.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("credential input must be ASCII") from None
    if not secret:
        raise ValueError("credential input is empty")
    return secret


def run_collect_llm(args) -> None:
    if not args.api_key_stdin:
        raise ValueError("only --api-key-stdin is supported")
    output = Path(args.output)
    request_manifest = read_artifact(output / "llm_request_manifest.json")
    split_manifest = read_artifact(output / "frozen_split_manifests.json")
    dataset_manifest = read_artifact(output / "dataset_manifests.json")
    secret = _read_stdin_secret()
    client = DeepSeekEnvelopeClient(secret)
    secret = ""
    collection_rows, expected, jobs = [], [], []
    code_commit = args.code_commit or _git_head()
    for item in request_manifest["rows"]:
        dataset = item["dataset"]
        manifest_hash = dataset_manifest["datasets"][dataset]["checksum"]
        split_checksum = split_manifest["datasets"][dataset]["checksum"]
        for lens in LENSES:
            request_hash, _ = request_identity(
                item["packet"], lens, code_commit=code_commit,
                dataset_manifest_hash=manifest_hash, split_checksum=split_checksum)
            expected.append(request_hash)
            jobs.append((item, lens, manifest_hash, split_checksum))

    def execute_job(job):
        item, lens, manifest_hash, split_checksum = job
        dataset = item["dataset"]
        result = collect_one(
            client=client, packet=item["packet"], lens=lens,
            raw_root=output / "raw_llm", code_commit=code_commit,
            dataset_manifest_hash=manifest_hash, split_checksum=split_checksum,
            retries=2,
        )
        return {
            "dataset": dataset, "split": item["split"], "record_key": item["record_key"],
            "label": item["label"], "actions": item["actions"],
            "inverse_sampling_weight": item["inverse_sampling_weight"], **result,
        }

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="phase4-llm") as pool:
        futures = [pool.submit(execute_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            collection_rows.append(future.result())
            if index % 40 == 0:
                partial = collection_manifest(collection_rows, expected_request_hashes=expected)
                partial["complete"] = False
                partial["planned_total_requests"] = len(expected)
                partial["transport_workers"] = args.workers
                write_artifact(output / "llm_collection_progress.json", partial)
                print(f"collected {len(collection_rows)}/{len(expected)}", flush=True)
    manifest = collection_manifest(collection_rows, expected_request_hashes=expected)
    manifest["planned_total_requests"] = len(expected)
    write_artifact(output / "llm_collection_manifest.json", manifest)
    client._api_key = ""
    if manifest["invalid"]:
        print(f"sealed collection requests={len(expected)} valid={manifest['valid']} "
              f"invalid={manifest['invalid']}; confirmatory scoring blocked", flush=True)
        return
    assert_complete_collection(manifest)
    print(f"sealed collection requests={len(expected)} valid={manifest['valid']}", flush=True)


def _git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_score_llm(args) -> None:
    output = Path(args.output)
    manifest = read_artifact(output / "llm_collection_manifest.json")
    if not args.allow_partial_diagnostic:
        assert_complete_collection(manifest)
    elif not manifest.get("complete"):
        raise ValueError("even diagnostic scoring requires every planned request to have completed attempts")
    grouped = defaultdict(lambda: defaultdict(dict))
    metadata = {}
    for row in manifest["rows"]:
        key = (row["dataset"], row["split"], row["record_key"])
        if row.get("valid"):
            grouped[key][row["lens"]] = row["probabilities"]
        metadata[key] = row
    assembled = defaultdict(lambda: defaultdict(list))
    request_manifest = read_artifact(output / "llm_request_manifest.json")
    expected_keys = [(row["dataset"], row["split"], row["record_key"])
                     for row in request_manifest["rows"]]
    for key in expected_keys:
        lenses = grouped[key]
        dataset, split, record_key = key
        row = metadata[key]
        lens_index = int(stable_hash("cost-matched", dataset, record_key)[:8], 16) % len(LENSES)
        panel = (logarithmic_pool([lenses[lens] for lens in LENSES], row["actions"])
                 if set(lenses) == set(LENSES) else None)
        assembled[dataset][split].append({
            "record_key": record_key, "label": row["label"], "actions": row["actions"],
            "weight": row["inverse_sampling_weight"], "B2": lenses.get(LENSES[0]), "B3": panel,
            "B3_COST_MATCHED": lenses.get(LENSES[lens_index]),
            "cost_matched_lens": LENSES[lens_index],
        })
    results = {}
    for dataset, partitions in assembled.items():
        calibration = sorted(partitions["calibration"], key=lambda row: row["record_key"])
        test = sorted(partitions["test"], key=lambda row: row["record_key"])
        dataset_result = {"calibration_n": len(calibration), "test_n": len(test), "arms": {}}
        for arm in ("B2", "B3", "B3_COST_MATCHED"):
            available_calibration = [row for row in calibration if row[arm] is not None]
            available_test = [row for row in test if row[arm] is not None]
            if not available_calibration or not available_test:
                dataset_result["arms"][arm] = {
                    "coverage": {"calibration": len(available_calibration) / max(1, len(calibration)),
                                 "test": len(available_test) / max(1, len(test))},
                    "status": "unscorable_no_covered_rows"}
                continue
            artifact = fit_temperature([row[arm] for row in available_calibration],
                                       [row["label"] for row in available_calibration],
                                       f"llm-sample:{dataset}:calibration",
                                       [row["weight"] for row in available_calibration])
            calibrated = [apply_calibration(row[arm], artifact) for row in available_test]
            dataset_result["arms"][arm] = {
                "coverage": {"calibration": len(available_calibration) / len(calibration),
                             "test": len(available_test) / len(test),
                             "calibration_n": len(available_calibration), "test_n": len(available_test)},
                "calibrator": artifact.__dict__,
                "uncalibrated": evaluate_predictions([row[arm] for row in available_test],
                                                       [row["label"] for row in available_test],
                                                       [row["actions"] for row in available_test],
                                                       [row["weight"] for row in available_test]),
                "calibrated": evaluate_predictions(calibrated,
                                                    [row["label"] for row in available_test],
                                                    [row["actions"] for row in available_test],
                                                    [row["weight"] for row in available_test]),
            }
        dataset_result["predictions"] = test
        dataset_result["confirmatory_coverage_gate"] = all(
            values.get("coverage", {}).get("test") == 1.0 for values in dataset_result["arms"].values())
        results[dataset] = dataset_result
    write_artifact(output / "llm_baseline_results.json", {
        "schema_version": "wmv2.phase4-completion.llm-baselines.v1", "datasets": results,
        "coverage_complete": manifest["invalid"] == 0,
        "confirmatory_valid": manifest["invalid"] == 0,
        "partial_metrics_status": ("confirmatory" if manifest["invalid"] == 0 else
                                   "diagnostic_only_selection_biased_by_provider_schema_compliance"),
        "raw_attempts_preserved_before_parse": True,
    })
    usage = defaultdict(Counter)
    for row in manifest["rows"]:
        for attempt in row["attempts"]:
            path = attempt.get("path")
            if not path:
                continue
            raw = read_artifact(path)
            for key, value in raw.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[row["dataset"]][key] += value
            usage[row["dataset"]]["latency_ms"] += raw.get("latency_ms", 0.0)
            usage[row["dataset"]]["requests"] += 1
    write_artifact(output / "llm_cost_latency.json", {
        "schema_version": "wmv2.phase4-completion.llm-cost-latency.v1",
        "provider_reported_usage": {key: dict(value) for key, value in usage.items()},
        "price_claim": "not computed; provider billing schedule was not frozen in the protocol",
    })
    print("LLM baselines scored from sealed raw collection", flush=True)


def run_finalize(args) -> None:
    output = Path(args.output)
    numeric = read_artifact(output / "numeric_summary.json")
    llm = read_artifact(output / "llm_baseline_results.json")
    gates = {}
    for dataset, result in numeric["datasets"].items():
        metrics = result["metrics"]
        b7 = metrics["B7"]["calibrated"]
        b6 = metrics["B6"]["calibrated"]
        comparison = result["clustered_comparisons"]["B7_minus_B6"]
        gates[dataset] = {
            "b7_log_loss": b7["log_loss"], "b6_log_loss": b6["log_loss"],
            "b7_minus_b6": comparison,
            "predictive_noninferiority_margin_0_01": comparison["ci95"][1] is not None
            and comparison["ci95"][1] < 0.01,
            "predictive_superiority": comparison["ci95"][1] is not None and comparison["ci95"][1] < 0.0,
            "invalid_action_gate": b7["invalid_action_rate"] == 0.0,
            "particle_collapse_gate": result["particle_diagnostics"]["test"]["collapse_count"] == 0,
        }
    equal_domain_delta = sum(value["b7_minus_b6"]["mean"] for value in gates.values()) / len(gates)
    summary = {
        "schema_version": "wmv2.phase4-completion.summary.v1", "datasets": gates,
        "equal_domain_b7_minus_b6_log_loss": equal_domain_delta,
        "all_domain_noninferiority": all(value["predictive_noninferiority_margin_0_01"]
                                          for value in gates.values()),
        "all_domain_superiority": all(value["predictive_superiority"] for value in gates.values()),
        "llm_complete": llm["coverage_complete"],
        "production_promotion": "withheld_pending_gate_review",
    }
    write_artifact(output / "aggregate_results.json", summary)
    write_artifact(output / "summary.json", summary)
    write_artifact(output / "failure_analysis.json", {
        "schema_version": "wmv2.phase4-completion.failure-analysis.v1",
        "failed_or_unmet_gates": {dataset: [key for key, value in row.items()
                                              if isinstance(value, bool) and not value]
                                  for dataset, row in gates.items()},
        "interpretation": "Failures are retained as empirical results; no arm, row, or domain is removed post hoc.",
    })
    write_artifact(output / "quarantine_status.json", {
        "schema_version": "wmv2.phase4-completion.quarantine.v1",
        "prior_phase4_quarantine_unchanged": True,
        "new_status": "production promotion withheld until every preregistered gate is reviewed",
        "original_results_namespace_untouched": True,
    })
    _write_checksums(output)
    print("final summary and checksum manifest sealed", flush=True)


def _write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*.json")):
        if path.name == "checksums_manifest.json" or "raw_llm" in path.parts:
            continue
        rows.append({"path": str(path.relative_to(output)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bytes": path.stat().st_size})
    write_artifact(output / "checksums_manifest.json", {
        "schema_version": "wmv2.phase4-completion.checksums.v1", "files": rows,
        "raw_llm_excluded_from_flat_manifest": True,
        "raw_llm_integrity": "each raw attempt carries its own verified artifact_checksum",
    })


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--output", default=str(DEFAULT_OUTPUT))
    commands = out.add_subparsers(dest="command", required=True)
    numeric = commands.add_parser("numeric")
    numeric.add_argument("--enron-archive", required=True)
    numeric.add_argument("--ipd-root", required=True)
    numeric.add_argument("--voteview-root", required=True)
    collect = commands.add_parser("collect-llm")
    collect.add_argument("--api-key-stdin", action="store_true")
    collect.add_argument("--code-commit", default="")
    collect.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    score = commands.add_parser("score-llm")
    score.add_argument("--allow-partial-diagnostic", action="store_true")
    commands.add_parser("finalize")
    return out


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    {"numeric": run_numeric, "collect-llm": run_collect_llm,
     "score-llm": run_score_llm, "finalize": run_finalize}[args.command](args)


if __name__ == "__main__":
    main()
