"""Separate 60-world controlled causal-coverage benchmark.

This diagnostic never contributes to the representative accuracy score.  It
uses independently authored scenario labels from ``activation_corpus_200``,
two cutoffs per world, and matched common-randomness ablations.  Each required
phase is scored on three effects: terminal distribution, StateDelta count, and
StateDelta sequence structure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiments.activation200 import _make_llm, _run_supervised
from experiments.activation_corpus_200 import QUESTIONS
from experiments.post_snapshot_benchmark.credentials import read_deepseek_key


RESULTS = Path("experiments/results/post_snapshot_benchmark")
DEFAULT_OUTPUT = RESULTS / "causal_coverage_results.json"
PHASE_BY_CATEGORY = {
    "p4": "phase4_actor_policy",
    "p6": "phase6_registry",
    "p7": "phase7_nonlinear",
    "p8": "phase8_persistence",
    "p9pop": "phase9_populations",
    "p9net": "phase9_networks",
    "p10": "phase10_institutions",
    "p11": "phase11_recompilation",
}
CATEGORY_NAMES = {
    "p4": "strategic_actor_policy",
    "p6": "nontrivial_mechanisms",
    "p7": "nonlinear_dynamics",
    "p8": "persistence",
    "p9pop": "heterogeneous_populations",
    "p9net": "multilayer_networks",
    "p10": "institutions",
    "p11": "natural_structural_change",
}
_WRITE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _categories(question_row) -> set[str]:
    # Every scenario spans three months and therefore genuinely requires the
    # default-on persistence-aware rollout; other labels are independently
    # authored in activation_corpus_200.
    return set(question_row[6]) | {"p8"}


def select_worlds(n: int = 60) -> list[tuple]:
    """Deterministic coverage-first selection, independent of runtime output."""
    remaining = list(QUESTIONS)
    selected = []
    counts = Counter()
    while len(selected) < n:
        def score(row):
            categories = _categories(row)
            uncovered = sum(counts[category] < 10 for category in categories)
            deficit = sum(max(0, 10 - counts[category]) for category in categories)
            return uncovered * 100 + deficit

        best = max(range(len(remaining)), key=lambda index: (score(remaining[index]), -index))
        row = remaining.pop(best)
        selected.append(row)
        counts.update(_categories(row))
    missing = {category: count for category, count in counts.items() if count < 10}
    if missing or any(counts[category] < 10 for category in PHASE_BY_CATEGORY):
        raise RuntimeError(f"causal category selection failed: {missing or counts}")
    return selected


def _cutoffs(as_of: str, horizon: str) -> list[str]:
    start = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    end = datetime.fromisoformat(horizon.replace("Z", "+00:00"))
    midpoint = start + (end - start) / 2
    return [start.isoformat(), midpoint.isoformat()]


def _total_variation(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _one(question_row, cutoff_index: int, llm) -> dict:
    qid, question, as_of, horizon, domain, family, _ = question_row
    cutoff = _cutoffs(as_of, horizon)[cutoff_index]
    seed = int(hashlib.sha256(f"{qid}:{cutoff_index}".encode()).hexdigest()[:8], 16)
    from swm.world_model_v2.compiler import compile_world

    row = {
        "schema_version": 1,
        "world_id": qid,
        "question": question,
        "domain": domain,
        "scenario_family": family,
        "source_type": "controlled_simulated_world",
        "excluded_from_representative_accuracy": True,
        "cutoff_index": cutoff_index,
        "forecast_cutoff": cutoff,
        "horizon": horizon,
        "seed": seed,
        "required_categories": sorted(_categories(question_row)),
        "common_randomness": True,
        "outcomes_accessed": False,
    }
    try:
        base = compile_world(question, llm=llm, evidence="", as_of=cutoff, horizon=horizon, seed=seed)
        full_records, p_full, terminal_full, trajectory_full = _run_supervised(
            copy.deepcopy(base), seed=seed)
        row["full_phase_records"] = {
            phase: {"execution_status": record.execution_status, "relevant": record.relevant,
                    "n_state_deltas": record.n_state_deltas,
                    "terminal_influence": record.terminal_influence}
            for phase, record in full_records.items()
        }
        row["full_terminal_distribution"] = terminal_full
        row["full_state_trajectory"] = trajectory_full
        ablations = {}
        for category in sorted(_categories(question_row)):
            phase = PHASE_BY_CATEGORY[category]
            ablated_records, p_ablated, terminal_ablated, trajectory_ablated = _run_supervised(
                copy.deepcopy(base), seed=seed, force_off=phase)
            terminal_tv = _total_variation(terminal_full, terminal_ablated)
            delta_signed = int(trajectory_full["n_state_deltas"]) - int(
                trajectory_ablated["n_state_deltas"])
            structural_changed = trajectory_full["sha256"] != trajectory_ablated["sha256"]
            targets = {
                "terminal_distribution": {
                    "full_affirmative_probability": p_full,
                    "ablated_affirmative_probability": p_ablated,
                    "signed_full_minus_ablated": p_full - p_ablated,
                    "magnitude_total_variation": terminal_tv,
                    "direction": ("higher_with_phase" if p_full > p_ablated else
                                  "lower_with_phase" if p_full < p_ablated else "unchanged"),
                    "effect_detected": terminal_tv >= 0.02,
                },
                "state_delta_count": {
                    "full": trajectory_full["n_state_deltas"],
                    "ablated": trajectory_ablated["n_state_deltas"],
                    "signed_full_minus_ablated": delta_signed,
                    "magnitude": abs(delta_signed),
                    "direction": ("more_with_phase" if delta_signed > 0 else
                                  "fewer_with_phase" if delta_signed < 0 else "unchanged"),
                    "effect_detected": delta_signed != 0,
                },
                "state_delta_sequence": {
                    "full_sha256": trajectory_full["sha256"],
                    "ablated_sha256": trajectory_ablated["sha256"],
                    "magnitude_binary_structural_change": int(structural_changed),
                    "direction": "changed" if structural_changed else "unchanged",
                    "effect_detected": structural_changed,
                },
            }
            ablations[category] = {
                "category_name": CATEGORY_NAMES[category],
                "phase": phase,
                "common_randomness_seed": seed,
                "full_execution_status": full_records[phase].execution_status,
                "ablated_execution_status": ablated_records[phase].execution_status,
                "trajectory_targets": targets,
                "target_count": len(targets),
                "any_meaningful_effect": any(target["effect_detected"] for target in targets.values()),
            }
        row["matched_ablations"] = ablations
        row["error"] = None
    except Exception as exc:  # noqa: BLE001 - preserve every diagnostic failure
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    row["row_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
    return row


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _aggregate(selection: list[tuple], rows: list[dict]) -> dict:
    scored = [row for row in rows if not row.get("error")]
    category_worlds = Counter()
    for question_row in selection:
        category_worlds.update(_categories(question_row))
    per_category = {}
    for category, phase in PHASE_BY_CATEGORY.items():
        relevant = [row for row in scored if category in row["required_categories"]]
        ablations = [row["matched_ablations"][category] for row in relevant]
        per_category[category] = {
            "category_name": CATEGORY_NAMES[category],
            "phase": phase,
            "independent_worlds": category_worlds[category],
            "cutoff_rows": len(relevant),
            "full_causally_active_rate": (sum(item["full_execution_status"] == "causally_active"
                                                for item in ablations) / len(ablations) if ablations else None),
            "ablated_explicit_noop_rate": (sum(item["ablated_execution_status"] ==
                                                "no_op_causally_irrelevant" for item in ablations) /
                                              len(ablations) if ablations else None),
            "any_meaningful_effect_rate": (sum(item["any_meaningful_effect"] for item in ablations) /
                                            len(ablations) if ablations else None),
            "target_effect_rates": {
                target: (sum(item["trajectory_targets"][target]["effect_detected"] for item in ablations) /
                         len(ablations) if ablations else None)
                for target in ("terminal_distribution", "state_delta_count", "state_delta_sequence")
            },
        }
    completion_gates = {
        "exactly_60_worlds": len(selection) == 60,
        "exactly_two_cutoffs_each": len(rows) == 120 and
            all(sum(row["world_id"] == question_row[0] for row in rows) == 2 for question_row in selection),
        "at_least_10_worlds_each_category": all(category_worlds[category] >= 10
                                                for category in PHASE_BY_CATEGORY),
        "zero_execution_errors": len(scored) == len(rows),
        "three_targets_per_ablation": all(
            ablation["target_count"] >= 3 for row in scored
            for ablation in row["matched_ablations"].values()),
        "common_randomness_all_ablations": all(
            row["common_randomness"] and all(
                ablation["common_randomness_seed"] == row["seed"]
                for ablation in row["matched_ablations"].values()) for row in scored),
        "outcomes_never_accessed": all(row["outcomes_accessed"] is False for row in rows),
        "reported_separately_from_accuracy": all(row["excluded_from_representative_accuracy"] for row in rows),
    }
    return {
        "schema_version": 1,
        "benchmark_role": "controlled causal-behavior diagnostic; never representative accuracy",
        "real_world_worlds": 0,
        "controlled_simulated_worlds": len(selection),
        "real_world_limitation": "source corpus lacks independently archived intermediate trajectory labels",
        "worlds": len(selection),
        "cutoff_rows": len(rows),
        "scored_rows": len(scored),
        "errors": len(rows) - len(scored),
        "category_world_counts": dict(sorted(category_worlds.items())),
        "per_category": per_category,
        "completion_gates": completion_gates,
        "all_completion_gates_pass": all(completion_gates.values()),
    }


def run(*, credential_source: Path, output: Path, workers: int = 6) -> dict:
    selection = select_worlds(60)
    key = read_deepseek_key(credential_source)
    llm = _make_llm(api_key=key)
    existing = {}
    if output.exists():
        prior = json.loads(output.read_text())
        existing = {(row["world_id"], row["cutoff_index"]): row for row in prior.get("rows", [])
                    if not row.get("error")}
    jobs = [(question_row, cutoff_index) for question_row in selection for cutoff_index in (0, 1)
            if (question_row[0], cutoff_index) not in existing]
    rows = list(existing.values())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_one, question_row, cutoff_index, llm):
                   (question_row[0], cutoff_index) for question_row, cutoff_index in jobs}
        for future in as_completed(futures):
            row = future.result()
            with _WRITE_LOCK:
                rows = [old for old in rows if (old["world_id"], old["cutoff_index"]) !=
                        (row["world_id"], row["cutoff_index"])]
                rows.append(row)
                rows.sort(key=lambda item: (item["world_id"], item["cutoff_index"]))
                _atomic(output, {
                    "schema_version": 1,
                    "created_at": _now(),
                    "selection": [{"world_id": q[0], "domain": q[4], "scenario_family": q[5],
                                   "required_categories": sorted(_categories(q))} for q in selection],
                    "rows": rows,
                    "aggregate": _aggregate(selection, rows),
                })
            print(f"{row['world_id']} cutoff={row['cutoff_index']} error={row['error']}", flush=True)
    final = json.loads(output.read_text())
    final["aggregate"] = _aggregate(selection, final["rows"])
    final["completed_at"] = _now()
    final["artifact_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in final.items() if key != "artifact_sha256"},
                   sort_keys=True, default=str).encode()).hexdigest()
    _atomic(output, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = run(credential_source=args.credential_source, output=args.output, workers=args.workers)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
