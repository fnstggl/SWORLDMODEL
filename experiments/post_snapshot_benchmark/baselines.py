"""Evidence-parity model baselines for the blinded historical benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.post_snapshot_benchmark.forecast import _atomic_write_attempt, _freeze_hash, _read_key
from swm.api.deepseek_backend import deepseek_chat_fn
from swm.engine.grounding import parse_json


def _capsule_path(root: Path, event_id: str, cutoff: str) -> Path:
    safe = "".join(ch for ch in cutoff if ch.isalnum())
    return root / f"{event_id}__{safe}.json"


def _prompt(world: dict, capsule: dict, instruction: str) -> str:
    evidence = "\n".join(str(item.get("blinded_text") or "") for item in capsule["items"])
    return (f"{instruction}\nForecast cutoff: {capsule['cutoff']}\n"
            f"Binary question: {world['question']}\nAdmissible evidence:\n{evidence[:5000]}\n"
            'Return ONLY JSON: {"p_yes": <number from 0 to 1>, "rationale": "<brief>"}')


def _ask(llm, prompt: str) -> tuple[float | None, dict]:
    try:
        parsed = parse_json(llm(prompt)) or {}
        value = parsed.get("p_yes")
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value))), parsed
        return None, parsed
    except Exception as exc:  # noqa: BLE001 - preserved baseline failure
        return None, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def _v2_calls(path: Path) -> dict:
    calls = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("full_system_qualified"):
            calls[(row["event_id"], row["forecast_cutoff"])] = int(row.get("model_calls") or 1)
    return calls


def _load_completed(path: Path) -> set[tuple[str, str]]:
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        row = json.loads(line)
        frozen = {key: value for key, value in row.items() if key != "baseline_sha256"}
        if (row.get("baseline_sha256") == _freeze_hash(frozen)
                and row.get("all_required_model_arms_complete")):
            completed.add((row["event_id"], row["forecast_cutoff"]))
    return completed


def run(*, credential_fd: int, forecast_input: Path, capsule_root: Path, v2_forecasts: Path,
        output: Path, split: str, world_limit: int | None = None) -> dict:
    key = _read_key(credential_fd)
    llm = deepseek_chat_fn(model="deepseek-v4-flash", api_key=key,
                           system="Reply ONLY with valid JSON.", max_tokens=700,
                           temperature=0.0, thinking="disabled")
    worlds = [world for world in json.loads(forecast_input.read_text())["worlds"]
              if world["split"] == split]
    if world_limit is not None:
        worlds = worlds[:world_limit]
    call_budgets = _v2_calls(v2_forecasts)
    completed = _load_completed(output)
    attempted = complete = skipped = 0
    for world in worlds:
        for cutoff in world["forecast_cutoffs"]:
            key = (world["event_id"], cutoff)
            if key in completed:
                skipped += 1
                continue
            attempted += 1
            started = time.time()
            capsule = json.loads(_capsule_path(capsule_root, world["event_id"], cutoff).read_text())
            budget = max(1, call_budgets.get(key, 1))
            direct_p, direct_trace = _ask(llm, _prompt(
                world, capsule, "Act as a direct calibrated forecaster. Use only the supplied evidence."))
            ensemble = []
            for index in range(budget):
                p, trace = _ask(llm, _prompt(
                    world, capsule, f"Independent direct forecast replicate {index + 1}; do not simulate a world."))
                ensemble.append({"p_yes": p, "trace": trace})
            panel = []
            roles = ("base-rate analyst", "causal skeptic", "domain generalist")
            for role in roles[:min(3, budget)]:
                p, trace = _ask(llm, _prompt(world, capsule, f"You are the observer panel's {role}."))
                panel.append({"role": role, "p_yes": p, "trace": trace})
            analog_p, analog_trace = _ask(llm, _prompt(
                world, capsule, "Use analogical retrieval from general pre-cutoff patterns, then forecast."))
            mean = lambda values: (sum(values) / len(values) if values else None)
            ensemble_values = [entry["p_yes"] for entry in ensemble if entry["p_yes"] is not None]
            panel_values = [entry["p_yes"] for entry in panel if entry["p_yes"] is not None]
            row = {
                "schema_version": 1, "event_id": world["event_id"],
                "forecast_cutoff": cutoff, "split": split,
                "evidence_capsule_sha256": capsule["capsule_sha256"],
                "evidence_byte_hashes": [item["source_raw_sha256"] for item in capsule["items"]],
                "model_identifier": "deepseek-v4-flash", "v2_call_budget": budget,
                "arms": {
                    "constant_0_50": {"p_yes": 0.5, "model_calls": 0},
                    "direct_single": {"p_yes": direct_p, "model_calls": 1, "trace": direct_trace},
                    "call_matched_direct_ensemble": {"p_yes": mean(ensemble_values),
                                                     "model_calls": len(ensemble), "members": ensemble},
                    "observer_panel": {"p_yes": mean(panel_values),
                                       "model_calls": len(panel), "members": panel},
                    "analogical_retrieval": {"p_yes": analog_p, "model_calls": 1,
                                             "trace": analog_trace},
                },
                "identical_evidence_for_all_model_arms": True,
                "call_matched_ensemble_within_v2_budget": len(ensemble) <= budget,
                "call_matched_ensemble_exactly_v2_budget": len(ensemble) == budget,
                "model_calls": 2 + len(ensemble) + len(panel),
                "latency_s": round(time.time() - started, 3),
                "cost_usd": None,
                "cost_status": "unavailable_model_alias_usage_not_exposed",
            }
            row["all_required_model_arms_complete"] = all(
                isinstance(arm.get("p_yes"), (int, float)) for arm in row["arms"].values())
            row["baseline_frozen_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            row["baseline_sha256"] = _freeze_hash(row)
            _atomic_write_attempt(output, row, canonical=True)
            complete += int(row["all_required_model_arms_complete"])
            print(f"baseline row {attempted}: complete={row['all_required_model_arms_complete']}", flush=True)
    return {"split": split, "attempted": attempted, "complete_this_run": complete,
            "skipped_frozen": skipped, "expected": len(worlds) * 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-fd", type=int, required=True)
    parser.add_argument("--forecast-input", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--v2-forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation", "locked_test"), required=True)
    parser.add_argument("--world-limit", type=int)
    args = parser.parse_args()
    result = run(credential_fd=args.credential_fd, forecast_input=args.forecast_input,
                 capsule_root=args.capsule_root, v2_forecasts=args.v2_forecasts,
                 output=args.output, split=args.split, world_limit=args.world_limit)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
