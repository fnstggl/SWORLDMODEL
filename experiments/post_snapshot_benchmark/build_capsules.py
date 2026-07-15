"""Construct the Tier-C blinded forecast packets and evidence capsules.

This is an evidence-construction role, not the forecaster.  It can see public
source wording and the pseudonym mapping, but it never reads any resolution
store.  Original canonical source bytes and mappings are written outside the
forecast mount.  Forecast-visible files contain only consistently blinded text,
relative date labels, hashes, and cutoff-safe availability metadata.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiments.post_snapshot_benchmark.credentials import read_deepseek_key
from swm.api.deepseek_backend import deepseek_chat_fn
from swm.replay.blinding import build_mapping


ROOT = Path("experiments/results/post_snapshot_benchmark")
VAULT = ROOT / "representative_vault.json"
CAPSULES = ROOT / "capsules"
FORECAST_INPUT = ROOT / "blinded_forecast_input.json"
MANIFEST = ROOT / "evidence_capsule_manifest.json"

_FULL_DATE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?\s*,?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b", re.I)
_ISO_DATE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|et|utc|gmt)\b", re.I)
_QUOTED = re.compile(r"[\"“][^\"”]{12,}[\"”]")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(world: dict) -> bytes:
    source = {
        "event_id": world.get("source_event_id"), "market_id": world.get("source_market_id"),
        "condition_id": world.get("source_condition_id"), "created_at": world.get("question_open_time"),
        "resolution_time": world.get("resolution_time"), "question": world.get("question"),
        "description": world.get("description"), "resolution_rule": world.get("resolution_rule"),
    }
    # Match the pool builder's canonical JSON byte encoding exactly, including
    # JSON escapes for non-ASCII source punctuation.
    return json.dumps(source, sort_keys=True, ensure_ascii=True).encode()


def reduce_fingerprint(text: str, mapping: dict) -> tuple[str, list[str]]:
    """Pseudonymize identities, abstract dates/times, and remove long quotes."""
    transformed = str(text or "")
    for name in sorted(mapping, key=len, reverse=True):
        transformed = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", str(mapping[name]),
                             transformed, flags=re.I)
    steps = ["stable_entity_pseudonymization"]
    date_counter = 0

    def date_alias(_):
        nonlocal date_counter
        date_counter += 1
        return f" relative-date-{date_counter} "

    transformed, n1 = _FULL_DATE.subn(date_alias, transformed)
    transformed, n2 = _ISO_DATE.subn(date_alias, transformed)
    transformed, n3 = _YEAR.subn("specified-year", transformed)
    transformed, n4 = _CLOCK.subn("specified-time", transformed)
    transformed, n5 = _QUOTED.subn("[identifying quotation removed]", transformed)
    if n1 + n2 + n3 + n4:
        steps.append("absolute_date_and_clock_abstraction")
    if n5:
        steps.append("identifying_quotation_removal")
    return re.sub(r"\s+", " ", transformed).strip(), steps


def _build_one(world: dict, llm) -> tuple[dict, dict, dict, list[dict]]:
    public_text = "\n".join(filter(None, (
        world.get("question"), world.get("description"), world.get("resolution_rule"))))
    mapping = build_mapping(public_text, llm)
    blinded_question, q_steps = reduce_fingerprint(world["question"], mapping)
    blinded_description, d_steps = reduce_fingerprint(world.get("description", ""), mapping)
    blinded_rule, r_steps = reduce_fingerprint(world.get("resolution_rule", ""), mapping)
    original = _canonical_bytes(world)
    source_record = {
        "event_id": world["event_id"], "source_raw_sha256": _sha(original),
        "raw_bytes_base64": base64.b64encode(original).decode(),
        "source_record_sha256_matches_pool": _sha(original) == world.get("source_record_sha256"),
    }
    if not source_record["source_record_sha256_matches_pool"]:
        raise RuntimeError(f"canonical source hash mismatch for {world['event_id']}")
    mapping_record = {
        "event_id": world["event_id"], "mapping": mapping,
        "mapping_sha256": _sha(json.dumps(mapping, sort_keys=True).encode()),
    }
    visible = {
        "event_id": world["event_id"], "event_world_cluster": world["event_world_cluster"],
        "domain": world["domain"], "split": world["split"],
        "question": blinded_question, "description": blinded_description,
        "resolution_rule": blinded_rule, "question_open_time": world["question_open_time"],
        "resolution_time": world["resolution_time"], "forecast_cutoffs": world["forecast_cutoffs"],
        "temporal_safety_tier": "causally_blinded_historical",
        "mapping_sha256": mapping_record["mapping_sha256"],
        "source_raw_sha256": source_record["source_raw_sha256"],
        "transformations": sorted(set(q_steps + d_steps + r_steps)),
    }
    visible_text = "\n".join((blinded_question, blinded_description, blinded_rule))
    pseudonym_text = "\n".join(str(value) for value in mapping.values())
    residual = [name for name, pseudonym in mapping.items()
                if len(name) >= 3 and name.lower() != str(pseudonym).lower() and
                not re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", pseudonym_text, re.I) and
                re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", visible_text, re.I)]
    if residual:
        raise RuntimeError(f"entity blinding residual for {world['event_id']}")
    capsules = []
    for cutoff in world["forecast_cutoffs"]:
        payload = {
            "event_id": world["event_id"], "cutoff": cutoff,
            "question": blinded_question,
            "items": [{
                "item_id": f"{world['event_id']}:source-record",
                "source_type": "official_market_question_record",
                "archive_source": "polymarket_gamma_canonical_source_record",
                "archive_retrieval_id": f"event:{world['source_event_id']}/market:{world['source_market_id']}",
                "source_raw_sha256": source_record["source_raw_sha256"],
                "blinded_text": "\n".join(filter(None, (blinded_description, blinded_rule))),
                "first_proven_available_at": world["question_open_time"],
                "claimed_publication_time": world["question_open_time"],
                "forecast_cutoff": cutoff,
                "temporal_verification_status": "provider_creation_timestamp",
                "transformation_history": visible["transformations"],
                "claim_mapping": ["event_question", "event_description", "resolution_rule"],
            }],
        }
        payload["capsule_sha256"] = _sha(json.dumps(payload, sort_keys=True).encode())
        capsules.append(payload)
    return visible, mapping_record, source_record, capsules


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(*, api_key: str, scorer_root: Path, archive_root: Path, workers: int = 4) -> dict:
    vault = json.loads(VAULT.read_text())
    worlds = vault["worlds"]
    cache_dir = scorer_root / "blinding_build_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm = deepseek_chat_fn(model="deepseek-v4-flash", api_key=api_key,
                           system="Reply ONLY with valid JSON.", max_tokens=900,
                           temperature=0.0, thinking="disabled")
    results = {}
    pending = []
    for world in worlds:
        path = cache_dir / f"{world['event_id']}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            results[world["event_id"]] = (cached["visible"], cached["mapping"],
                                           cached["source"], cached["capsules"])
        else:
            pending.append(world)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, world, llm): world["event_id"] for world in pending}
        for future in as_completed(futures):
            event_id = futures[future]
            result = future.result()
            results[event_id] = result
            v, m, s, c = result
            cache_path = cache_dir / f"{event_id}.json"
            _write_json(cache_path, {"visible": v, "mapping": m, "source": s, "capsules": c})
            cache_path.chmod(0o600)
            print(f"blinded capsule source {len(results)}/{len(worlds)}", flush=True)
    visible, mappings, sources, capsules = [], [], [], []
    for world in worlds:  # restore frozen chronological order
        v, m, s, c = results[world["event_id"]]
        visible.append(v); mappings.append(m); sources.append(s); capsules.extend(c)
    CAPSULES.mkdir(parents=True, exist_ok=True)
    for capsule in capsules:
        safe_cutoff = re.sub(r"[^0-9A-Za-z]", "", capsule["cutoff"])
        _write_json(CAPSULES / f"{capsule['event_id']}__{safe_cutoff}.json", capsule)
    _write_json(FORECAST_INPUT, {
        "schema_version": 1, "n_worlds": len(visible), "n_rows": len(capsules),
        "outcomes_present": False, "pseudonym_mappings_present": False, "worlds": visible,
    })
    scorer_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    _write_json(scorer_root / "pseudonym_mappings.json", {
        "schema_version": 1, "mappings": mappings})
    _write_json(archive_root / "canonical_source_records.json", {
        "schema_version": 1, "records": sources})
    (scorer_root / "pseudonym_mappings.json").chmod(0o600)
    (archive_root / "canonical_source_records.json").chmod(0o600)
    manifest_rows = [{
        "event_id": c["event_id"], "cutoff": c["cutoff"],
        "capsule_sha256": c["capsule_sha256"],
        "item_source_hashes": [item["source_raw_sha256"] for item in c["items"]],
        "all_items_cutoff_safe": all(item["first_proven_available_at"] <= item["forecast_cutoff"]
                                     for item in c["items"]),
    } for c in capsules]
    manifest = {
        "schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_worlds": len(visible), "n_capsules": len(capsules),
        "all_capsules_cutoff_safe": all(row["all_items_cutoff_safe"] for row in manifest_rows),
        "original_source_bytes_outside_forecast_mount": str(archive_root.resolve()),
        "pseudonym_mappings_outside_forecast_mount": str(scorer_root.resolve()),
        "rows": manifest_rows,
    }
    _write_json(MANIFEST, manifest)
    return {key: manifest[key] for key in ("n_worlds", "n_capsules", "all_capsules_cutoff_safe")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-source", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = read_deepseek_key(args.credential_source)
    report = build(api_key=key, scorer_root=args.scorer_root.resolve(),
                   archive_root=args.archive_root.resolve(), workers=args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
