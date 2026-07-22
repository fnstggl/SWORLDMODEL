"""Temporal Replay v2 — FORECASTER (Part 15/16/18). Sealed: never reads resolutions.

Arms per (world, cutoff):
  * pre_cutoff_checkpoint      — BLOCKED_EXTERNAL in this environment (see model_registry.json): no
                                 verifiable pre-cutoff checkpoint exists. Rows are emitted as explicit
                                 failures (failure_reason=arm_a_blocked_external), never silently skipped —
                                 the 800-clean-forecast gate therefore CANNOT pass until unblocked.
  * blinded_current_llm        — clean Arm B: frozen archived evidence (capsule), all LLM-visible text
                                 pseudonymized, six leakage probes, full supervised runtime.
  * cutoff_prompted_unblinded  — diagnostic only; never counts toward clean totals.

Every attempted row produces ONE machine-readable audit row (Part 18) with a freeze hash stamped before any
scoring. Resumable: rows are keyed (event_id, cutoff, arm) and reruns skip completed rows.

Isolation boundary (Part 16, documented honestly): evidence capsules are constructed and FROZEN to disk by
the evidence process (build_capsules step) before forecasting; the forecaster reads only frozen capsule
files + the public vault; the sealed resolution store requires REPLAY_SCORER=1 and is verified untouched by
freeze-hash checks. The forecaster's only network dependency is the LLM API. OS/container-level network
whitelisting (LLM endpoint only) is a deployment requirement recorded in the audit rows as
`open_internet_disabled=false_process_level_only` — reported as a PARTIAL isolation gate, not claimed.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path

from swm.replay.vault import freeze_hash
from swm.replay.blinding import build_mapping, blind_question, blind_bundle
from swm.replay.probes2 import run_probes_v2
from swm.replay.archive_evidence import build_capsule, ReplayBundle

VAULT = Path("experiments/replay_vault_v2")
OUT = Path("experiments/results/replay_v2")
CAPS = VAULT / "capsules"
ART = OUT / "audit_rows.jsonl"
MAPS = VAULT / "blinding_mappings.json"

CLEAN_ARMS = ("pre_cutoff_checkpoint", "blinded_current_llm")
ALL_ARMS = CLEAN_ARMS + ("cutoff_prompted_unblinded",)


def _make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    return default_chat_fn(system="Reply ONLY JSON.", max_tokens=2400, temperature=0.2)


def _worlds():
    return json.loads((VAULT / "events.json").read_text())


def _p_yes(res):
    for attr in ("calibrated_probability", "raw_probability"):
        v = getattr(res, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    d = getattr(res, "raw_distribution", None) or {}
    if d:
        keys = list(d)
        for k in keys:
            if str(k).lower() in ("true", "yes"):
                return float(d[k])
        return float(d[keys[0]])
    return None


def build_capsules(limit=None, llm=None):
    """Evidence-construction process (run FIRST, separately): freeze one capsule per (world, cutoff)."""
    CAPS.mkdir(parents=True, exist_ok=True)
    llm = llm or _make_llm()
    v = _worlds()
    done = 0
    for w in (v["worlds"][:limit] if limit else v["worlds"]):
        for cutoff in w["forecast_cutoffs"]:
            fn = CAPS / f"{w['event_id']}__{cutoff}.json"
            if fn.exists():
                continue
            cap = build_capsule(w["event_id"], w["question"], cutoff, llm=llm)
            fn.write_text(json.dumps(cap.as_dict(), indent=1))
            done += 1
            print(f"capsule {fn.name}: {len(cap.as_dict()['items'])} archived items", flush=True)
    print(f"{done} new capsules")


def _audit_row(w, cutoff, arm, fp):
    return {
        "event_id": w["event_id"], "event_family": w["event_family"], "domain": w["domain"],
        "cutoff": cutoff, "arm": arm, "cluster": w["event_family"],
        "split": _worlds()["splits"].get(w["event_id"], ""),
        "evidence_source_ids": [], "raw_archived_byte_hashes": [],
        "first_proven_availability": [], "future_content_access_impossible": None,
        "question_blinded": None, "evidence_blinded": None,
        "resolution_inaccessible_to_forecaster": True,       # PermissionError-guarded sealed store
        "open_internet_disabled": "false_process_level_only (LLM API required; OS whitelist is a "
                                  "deployment requirement — PARTIAL isolation, not claimed)",
        "model_checkpoint": None, "model_cutoff": None, "model_hash": None,
        "leakage_probes": None, "leakage_class_pending_scorer": True,
        "phase_relevance": None, "phase_execution_records": None,
        "active_relevant_phases": None, "explicit_noop_phases": None, "blocked_phases": None,
        "statedelta_by_phase": None, "terminal_influence_by_phase": None,
        "terminal_source": None, "raw_terminal_distribution": None,
        "market_snapshot": (w.get("market_snapshots") or {}).get(cutoff),
        "runtime_fingerprint": fp, "clean_headline_eligible": None,
        "cost": None, "latency_s": None, "failure_reason": None, "scorer_identity": None,
    }


def run(limit=None, arms=ALL_ARMS, seed=0, smoke=False):
    from swm.world_model_v2.unified_runtime import simulate_world
    from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
    OUT.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    fp = runtime_fingerprint()["fingerprint_hash"]
    done = set()
    if ART.exists():
        for line in ART.read_text().splitlines():
            r = json.loads(line)
            if not r.get("failure_reason") or r["failure_reason"] == "arm_a_blocked_external":
                done.add((r["event_id"], r["cutoff"], r["arm"]))
    mappings = json.loads(MAPS.read_text())["mappings"] if MAPS.exists() else {}
    v = _worlds()
    worlds = v["worlds"][:limit] if limit else v["worlds"]
    for w in worlds:
        cutoffs = w["forecast_cutoffs"][:1] if smoke else w["forecast_cutoffs"]
        for cutoff in cutoffs:
            capfile = CAPS / f"{w['event_id']}__{cutoff}.json"
            for arm in arms:
                if (w["event_id"], cutoff, arm) in done:
                    continue
                t0 = time.time()
                row = _audit_row(w, cutoff, arm, fp)
                try:
                    if arm == "pre_cutoff_checkpoint":
                        reg = json.loads((VAULT / "model_registry.json").read_text())
                        row["failure_reason"] = "arm_a_blocked_external"
                        row["model_checkpoint"] = reg["models"][0]["model"]
                        row["model_cutoff"] = reg["models"][0]["documented_cutoff"]
                        row["model_hash"] = reg["models"][0]["model_hash"]
                        _emit(row, t0)
                        continue
                    if not capfile.exists():
                        row["failure_reason"] = "capsule_missing (run build_capsules first)"
                        _emit(row, t0)
                        continue
                    cap = json.loads(capfile.read_text())
                    row["evidence_source_ids"] = [i["archive_retrieval_id"] for i in cap["items"]]
                    row["raw_archived_byte_hashes"] = [i["raw_sha256"] for i in cap["items"]]
                    row["first_proven_availability"] = [i["first_proven_available_at"]
                                                        for i in cap["items"]]
                    row["future_content_access_impossible"] = True   # capsule enforces the cutoff rule
                    bundle = ReplayBundle(cap, w["question"])
                    row["model_checkpoint"] = "deepseek-chat (current)"
                    if arm == "blinded_current_llm":
                        mapping = mappings.get(w["event_id"]) or build_mapping(w["question"], llm)
                        mappings[w["event_id"]] = mapping
                        MAPS.write_text(json.dumps({"note": "pseudonym mappings (no outcomes)",
                                                    "mappings": mappings}, indent=1))
                        bq = blind_question(w["question"], mapping)
                        bb = blind_bundle(copy.deepcopy(bundle), mapping)
                        row["question_blinded"] = True
                        row["evidence_blinded"] = True
                        row["leakage_probes"] = run_probes_v2(
                            llm, real_question=w["question"], blinded_question=bq, mapping=mapping,
                            cutoff=cutoff, evidence_text=bb.render(max_chars=1800))
                        res = simulate_world(bq, as_of=cutoff, horizon=w["horizon"], llm=llm,
                                             seed=seed, prebuilt_bundle=bb)
                    else:
                        row["question_blinded"] = row["evidence_blinded"] = False
                        res = simulate_world(w["question"], as_of=cutoff, horizon=w["horizon"],
                                             llm=llm, seed=seed, prebuilt_bundle=bundle)
                    prov = getattr(res, "provenance", {}) or {}
                    pers = prov.get("phase_execution_records") or {}
                    row["phase_execution_records"] = pers
                    row["phase_relevance"] = {p: r.get("relevant") for p, r in pers.items()}
                    row["active_relevant_phases"] = [p for p, r in pers.items()
                                                     if r.get("execution_status") == "causally_active"]
                    row["explicit_noop_phases"] = [p for p, r in pers.items()
                                                   if r.get("execution_status") == "no_op_causally_irrelevant"]
                    row["blocked_phases"] = [p for p, r in pers.items()
                                             if str(r.get("execution_status", "")).startswith("blocked")]
                    row["statedelta_by_phase"] = {p: r.get("n_state_deltas") for p, r in pers.items()}
                    row["terminal_influence_by_phase"] = {p: r.get("terminal_influence")
                                                          for p, r in pers.items()}
                    row["terminal_source"] = "terminal_world_states"
                    row["raw_terminal_distribution"] = getattr(res, "raw_distribution", None)
                    row["p_yes"] = _p_yes(res)
                    row["simulation_status"] = getattr(res, "simulation_status", "")
                    if row["blocked_phases"]:
                        row["failure_reason"] = f"blocked_relevant_phases:{row['blocked_phases']}"
                except Exception as e:  # noqa: BLE001
                    row["failure_reason"] = f"{type(e).__name__}: {e}"[:180]
                _emit(row, t0)
                print(f"{w['event_id']:12s} {cutoff} {arm:26s} p={row.get('p_yes')} "
                      f"fail={str(row.get('failure_reason'))[:40]}", flush=True)


def _emit(row, t0):
    row["latency_s"] = round(time.time() - t0, 2)
    row["freeze_hash"] = freeze_hash({k: v for k, v in row.items() if k != "freeze_hash"})
    with ART.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capsules", action="store_true", help="run the evidence-construction step")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 cutoff/world infra validation (NOT benchmark)")
    a = ap.parse_args()
    if a.capsules:
        build_capsules(limit=a.limit)
    else:
        run(limit=a.limit, smoke=a.smoke)
