"""Post-snapshot benchmark runner (v3). Primary arm: TIER B provider_attested_post_cutoff, ADDITIONALLY
causally blinded + six-probed (belt and braces: the HF repo shows post-release activity, so serving-weight
mutation is a live risk; blinding + probes measure it per row instead of assuming it away).

Per (world, cutoff) row:
  frozen capsule (archived bytes, cutoff-enforced) → blinding mapping → six leakage probes →
  simulate_world(blinded question, prebuilt_bundle=blinded capsule) with deepseek-v4-flash →
  per-row FULL-SYSTEM QUALIFICATION (all 11 PhaseExecutionRecords, zero blocked relevant phases,
  terminal from world states) → fair baselines on the SAME blinded question + capsule text
  (direct single call; call-matched ensemble of 3; observer panel of 3; analogical retrieval) →
  frozen audit row (hash stamped).

Execution order (Part 19) is enforced by --split: calibration → validation → (fit/select happens in the
scorer) → locked_test. Resumable: rows keyed (event_id, cutoff); reruns skip completed rows. Parallel
across rows (thread pool; LLM-bound).
"""
from __future__ import annotations
import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.replay.vault import freeze_hash
from swm.replay.blinding import build_mapping, blind_question, blind_bundle, apply_mapping
from swm.replay.probes2 import run_probes_v2
from swm.replay.archive_evidence import build_capsule, ReplayBundle

VAULT = Path("experiments/replay_vault_v3")
OUT = Path("experiments/results/replay_v3")
CAPS = VAULT / "capsules"
MAPS = VAULT / "blinding_mappings.json"
ARM = "provider_attested_post_cutoff_blinded"
MODEL = "deepseek-v4-flash"
_LOCK = threading.Lock()


def _llm(max_tokens=2400, temperature=0.2):
    from swm.api.deepseek_backend import deepseek_chat_fn
    return deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=max_tokens,
                            temperature=temperature)


def _worlds(vault_file="events.json"):
    return json.loads((VAULT / vault_file).read_text())


def _p_yes(res):
    for attr in ("calibrated_probability", "raw_probability"):
        v = getattr(res, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    d = getattr(res, "raw_distribution", None) or {}
    if d:
        for k in d:
            if str(k).lower() in ("true", "yes"):
                return float(d[k])
        return float(d[list(d)[0]])
    return None


def _parse_p(txt):
    from swm.engine.grounding import parse_json
    v = (parse_json(txt) or {}).get("p_yes")
    return float(v) if isinstance(v, (int, float)) else None


_BASE_PROMPT = """Evidence (all archived strictly before {cutoff}):
{ev}
Question: {q}
Estimate the probability the answer is YES by {horizon}. Return ONLY JSON: {{"p_yes": <0..1>}}"""

_PANEL_ROLES = ("a careful superforecaster focused on base rates",
                "a domain analyst focused on causal mechanisms",
                "a skeptical auditor focused on what the evidence does NOT show")

_ANALOG_PROMPT = """Question: {q}
List 3 analogous historical situations (pseudonymous is fine) and the base rate of YES-like outcomes among
them, then give a probability. Return ONLY JSON: {{"analogs": ["..."], "p_yes": <0..1>}}"""


def _baselines(llm_fast, bq, ev_text, cutoff, horizon):
    base = _BASE_PROMPT.format(cutoff=cutoff, ev=ev_text[:2400], q=bq, horizon=horizon)
    out = {"direct": None, "ensemble": [], "panel": [], "analogical": None}
    try:
        out["direct"] = _parse_p(llm_fast(base))
    except Exception:  # noqa: BLE001
        pass
    for i in range(3):                                       # call-matched ensemble (temp 0.7)
        try:
            out["ensemble"].append(_parse_p(_llm(400, 0.7)(base)))
        except Exception:  # noqa: BLE001
            out["ensemble"].append(None)
    for role in _PANEL_ROLES:
        try:
            out["panel"].append(_parse_p(llm_fast(f"You are {role}.\n" + base)))
        except Exception:  # noqa: BLE001
            out["panel"].append(None)
    try:
        out["analogical"] = _parse_p(llm_fast(_ANALOG_PROMPT.format(q=bq)))
    except Exception:  # noqa: BLE001
        pass
    return out


def build_capsules(vault_file="events.json", workers=8):
    CAPS.mkdir(parents=True, exist_ok=True)
    llm = _llm(600)
    v = _worlds(vault_file)
    jobs = [(w, c) for w in v["worlds"] for c in w["forecast_cutoffs"]
            if not (CAPS / f"{w['event_id']}__{c}.json").exists()]
    print(f"{len(jobs)} capsules to build")

    def one(w, c):
        try:
            cap = build_capsule(w["event_id"], w["question"], c, llm=llm)
            (CAPS / f"{w['event_id']}__{c}.json").write_text(json.dumps(cap.as_dict(), indent=1))
            return f"{w['event_id']}__{c}: {len(cap.as_dict()['items'])} items"
        except Exception as e:  # noqa: BLE001
            return f"{w['event_id']}__{c}: ERROR {e}"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, w, c) for w, c in jobs]):
            print(fut.result(), flush=True)


def _row(w, cutoff, fp, split):
    from swm.world_model_v2.unified_runtime import simulate_world
    t0 = time.time()
    row = {"event_id": w["event_id"], "cluster": w["event_family"], "domain": w["domain"],
           "question_open": w.get("question_open"), "cutoff": cutoff, "horizon": w["horizon"],
           "arm": ARM, "tier": "B_provider_attested_post_cutoff", "split": split,
           "model": MODEL, "model_revision_reference": "hf:deepseek-ai/DeepSeek-V4-Flash@60d8d707",
           "model_release": "2026-04-24", "runtime_fingerprint": fp,
           "market_snapshot": (w.get("market_snapshots") or {}).get(cutoff),
           "open_internet": "process_level_partial (LLM API only network dependency; documented)",
           "resolution_inaccessible_to_forecaster": True, "failure_reason": None}
    try:
        capfile = CAPS / f"{w['event_id']}__{cutoff}.json"
        if not capfile.exists():
            row["failure_reason"] = "capsule_missing"
            return _fin(row, t0)
        cap = json.loads(capfile.read_text())
        row["evidence_capsule_hash"] = cap.get("capsule_hash")
        row["evidence_byte_hashes"] = [i["raw_sha256"] for i in cap["items"]]
        row["first_proven_availability"] = [i["first_proven_available_at"] for i in cap["items"]]
        llm = _llm()
        with _LOCK:
            mappings = json.loads(MAPS.read_text())["mappings"] if MAPS.exists() else {}
        mapping = mappings.get(w["event_id"]) or build_mapping(w["question"], llm)
        with _LOCK:
            mappings[w["event_id"]] = mapping
            MAPS.write_text(json.dumps({"note": "pseudonym mappings (no outcomes)",
                                        "mappings": mappings}, indent=1))
        bundle = ReplayBundle(cap, w["question"])
        bq = blind_question(w["question"], mapping)
        bb = blind_bundle(copy.deepcopy(bundle), mapping)
        ev_text = bb.render(max_chars=2400)
        row["question_blinded"], row["evidence_blinded"] = True, True
        row["blinded_question"] = bq
        row["leakage_probes"] = run_probes_v2(llm, real_question=w["question"], blinded_question=bq,
                                              mapping=mapping, cutoff=cutoff, evidence_text=ev_text)
        res = simulate_world(bq, as_of=cutoff, horizon=w["horizon"], llm=llm, seed=0,
                             prebuilt_bundle=bb)
        prov = getattr(res, "provenance", {}) or {}
        pers = prov.get("phase_execution_records") or {}
        row["phase_execution_records"] = pers
        row["active_relevant_phases"] = [p for p, r in pers.items()
                                         if r.get("execution_status") == "causally_active"]
        row["explicit_noop_phases"] = [p for p, r in pers.items()
                                       if r.get("execution_status") == "no_op_causally_irrelevant"]
        row["blocked_phases"] = [p for p, r in pers.items()
                                 if str(r.get("execution_status", "")).startswith("blocked")]
        row["statedelta_by_phase"] = {p: r.get("n_state_deltas") for p, r in pers.items()}
        row["terminal_source"] = "terminal_world_states"
        row["raw_terminal_distribution"] = getattr(res, "raw_distribution", None)
        row["p_yes"] = _p_yes(res)
        row["simulation_status"] = getattr(res, "simulation_status", "")
        if len(pers) != 11:
            row["failure_reason"] = f"phase_record_coverage_{len(pers)}_of_11"
        elif row["blocked_phases"]:
            row["failure_reason"] = f"blocked_relevant_phases:{row['blocked_phases']}"
        elif row["p_yes"] is None:
            row["failure_reason"] = "no_terminal_probability"
        row["baselines"] = _baselines(_llm(400), bq, ev_text, cutoff, w["horizon"])
    except Exception as e:  # noqa: BLE001
        row["failure_reason"] = f"{type(e).__name__}: {e}"[:180]
    return _fin(row, t0)


def _fin(row, t0):
    row["latency_s"] = round(time.time() - t0, 2)
    row["freeze_hash"] = freeze_hash({k: v for k, v in row.items() if k != "freeze_hash"})
    return row


def run(split, vault_file="events.json", workers=6, art_name=None):
    from swm.world_model_v2.runtime_fingerprint import runtime_fingerprint
    OUT.mkdir(parents=True, exist_ok=True)
    art = OUT / (art_name or f"forecasts_{split}.jsonl")
    fp = runtime_fingerprint()["fingerprint_hash"]
    v = _worlds(vault_file)
    splits = v.get("splits", {})
    done = set()
    if art.exists():
        for line in art.read_text().splitlines():
            r = json.loads(line)
            if not r.get("failure_reason"):
                done.add((r["event_id"], r["cutoff"]))
    jobs = []
    for w in v["worlds"]:
        wsplit = splits.get(w["event_id"], "coverage")
        if split != "all" and wsplit != split:
            continue
        for c in w["forecast_cutoffs"]:
            if (w["event_id"], c) not in done:
                jobs.append((w, c, wsplit))
    print(f"{len(jobs)} rows to run for split={split}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_row, w, c, fp, s) for w, c, s in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            with _LOCK:
                with art.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            print(f"{row['event_id']:12s} {row['cutoff']} p={row.get('p_yes')} "
                  f"active={len(row.get('active_relevant_phases') or [])} "
                  f"fail={str(row.get('failure_reason'))[:50]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capsules", action="store_true")
    ap.add_argument("--vault", default="events.json")
    ap.add_argument("--split", default="calibration",
                    choices=["calibration", "validation", "locked_test", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.capsules:
        build_capsules(vault_file=a.vault, workers=8)
    else:
        run(a.split, vault_file=a.vault, workers=a.workers)
