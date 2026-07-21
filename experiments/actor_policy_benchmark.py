"""Five-arm actor-policy benchmark on frozen historical decisions.

Runs the SAME frozen decision cases (experiments/frozen_decision_cases.json — evidence stops at
each case's as_of; the actual action never enters any prompt) under:

    A numeric_policy                     — the untouched Phase-4 numeric pipeline
    B persona_blended_numeric_policy     — cc17199, honestly renamed (LLM rates, blend decides)
    C stateless_llm_policy               — qualitative choice, no persistent hidden state
    D persistent_qualitative_llm_policy  — the hypothesis: K hidden-state particles × samples
    E hybrid_relevant_actor_policy       — D for Tier-1 actors via the causal selector

and reports measured next-action accuracy, top-2 accuracy, log loss, Brier score, a crude
confidence-vs-accuracy calibration gap, novel-action rate, LLM calls, and latency. Counted
distributions (C/D/E) come from branch-selection frequencies; A/B report their model
posteriors. This is a PILOT (n=9 hand-curated cases, single decision point per case): it
measures the decision layer under frozen evidence, not long-horizon trajectory fidelity.

Usage:
    DEEPSEEK_API_KEY=… python experiments/actor_policy_benchmark.py            # all arms
    python experiments/actor_policy_benchmark.py --backend scripted --arms A,C # offline check
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import time as _time
from pathlib import Path

from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.phase4_execution import ActorPolicyRuntime
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

CASES_PATH = Path("experiments/frozen_decision_cases.json")
RESULTS_DIR = Path("experiments/results")
ARMS = {"A": "numeric_policy", "B": "persona_blended_numeric_policy",
        "C": "stateless_llm_policy", "D": "persistent_qualitative_llm_policy",
        "E": "hybrid_relevant_actor_policy"}


def _ts(day: str) -> float:
    return float(calendar.timegm(_time.strptime(day, "%Y-%m-%d"))) + 12 * 3600.0


def leakage_check(case: dict) -> None:
    """The benchmark's hard leakage rule: every evidence line predates as_of; the decision
    postdates it; the label never appears in actor-facing fields."""
    as_of = _ts(case["as_of"])
    for ev in case["evidence"]:
        if _ts(ev["date"]) > as_of:
            raise ValueError(f"{case['case_id']}: evidence dated {ev['date']} after as_of")
    if _ts(case["actual_action_date"]) < as_of:
        raise ValueError(f"{case['case_id']}: actual action predates as_of")
    rendered = json.dumps({k: case[k] for k in ("situation", "goals", "commitments", "evidence",
                                                "candidate_actions")})
    if case["source_note"][:40] in rendered:
        raise ValueError(f"{case['case_id']}: label leakage into actor-facing fields")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text())
    for case in data["cases"]:
        leakage_check(case)
    return data["cases"]


def build_case_world(case: dict, branch_id: str = "b000") -> tuple[WorldState, dict]:
    """Frozen actor-local world: the actor, their goals/commitments, and ONLY pre-as_of
    evidence exposed as their observed information. The label never enters."""
    now = _ts(case["as_of"])
    w = WorldState(case["case_id"], branch_id, SimulationClock(now, now),
                   network=RelationGraph(), information=InformationLedger())
    actor = Entity(case["actor_id"])
    actor.set("roles", F([case["role"]], status="observed"))
    actor.set("goals", F(list(case.get("goals") or []), status="inferred"))
    actor.set("commitments", F(list(case.get("commitments") or []), status="observed"))
    actor.set("past_actions", F([], status="observed"))
    w.entities = {case["actor_id"]: actor}
    for i, ev in enumerate(case["evidence"]):
        iid = f"ev_{i}"
        w.information.publish(InformationItem(iid, ev["text"], source="public_record",
                                              created_at=_ts(ev["date"])))
        w.information.expose(case["actor_id"], iid, _ts(ev["date"]))
    # candidate order is shuffled deterministically per case: the file lists the label first,
    # and option-position bias must never correlate with the label
    import random as _random
    candidates = list(case["candidate_actions"])
    _random.Random(case["case_id"]).shuffle(candidates)
    decision = {"situation": case["situation"], "candidate_actions": candidates}
    return w, decision


def _name_distribution_from_posterior(posterior, trace) -> dict:
    name_of = {a["action_id"]: a["action_name"] for a in trace.candidate_actions}
    dist: dict = {}
    for aid, p in posterior.action_probabilities.items():
        name = name_of.get(aid, aid)
        dist[name] = dist.get(name, 0.0) + float(p)
    return dist


def _scripted_backend(case_actual_by_prompt_marker):
    """Deterministic offline backend for harness self-tests: chooses the FIRST candidate."""
    def fn(prompt):
        if "ALTERNATIVE HYPOTHESES" in prompt:
            return json.dumps([{"hypothesis_label": f"h{i}",
                                "core_worldview": f"variant {i}",
                                "current_private_beliefs": [f"belief {i}"]} for i in range(3)])
        first = None
        for line in prompt.splitlines():
            if line.startswith("- ") and ":" in line and "/" in line:
                first = line[2:].split(":", 1)[0].strip()
                break
        return json.dumps({"decision": {"act_or_wait": "act", "chosen_action": first or "wait",
                                        "observability": "public"},
                           "actor_state_update": {}, "decision_summary": "scripted"})
    return fn


# ---------------------------------------------------------------------------- arms
def run_numeric(case, *, seed=0, **_):
    w, decision = build_case_world(case)
    t0 = _time.monotonic()
    _, posterior, trace = ActorPolicyRuntime().decide(None, [w], case["actor_id"],
                                                      decision=decision, seed=seed)
    return {"distribution": _name_distribution_from_posterior(posterior, trace),
            "llm_calls": 0, "latency_s": _time.monotonic() - t0, "n_samples": 0,
            "floor_kind": "model_posterior"}


def run_persona(case, *, llm, seed=0, **_):
    from swm.world_model_v2.llm_actor import (PersonaActorPolicyRuntime, PersonaConfig,
                                              PersonaEngine)
    w, decision = build_case_world(case)
    rt = PersonaActorPolicyRuntime(PersonaEngine(PersonaConfig(llm=llm, scope="all",
                                                               max_llm_calls=6)))
    t0 = _time.monotonic()
    _, posterior, trace = rt.decide(None, [w], case["actor_id"], decision=decision, seed=seed)
    return {"distribution": _name_distribution_from_posterior(posterior, trace),
            "llm_calls": int(trace.cost.get("llm_calls", 0)),
            "latency_s": _time.monotonic() - t0, "n_samples": 0,
            "floor_kind": "model_posterior"}


def run_qualitative(case, *, llm, hypothesis_llm=None, mapper_llm=None,
                    mode="persistent_qualitative_llm_policy", hypotheses=3, samples=2, seed=0):
    from swm.world_model_v2.actor_selection import RelevantActorSelector
    from swm.world_model_v2.qualitative_actor import (ActionClusterer,
                                                      QualitativeActorPolicyRuntime,
                                                      QualitativeConfig,
                                                      QualitativeDecisionEngine,
                                                      aggregate_actor_decisions)
    n = max(1, hypotheses) * max(1, samples)
    cfg = QualitativeConfig(llm=llm, hypothesis_llm=hypothesis_llm, n_hypotheses=hypotheses,
                            max_llm_calls=4 * n + 2)
    tiers = None
    if mode == "hybrid_relevant_actor_policy":
        # the case actor holds the decision by construction — the selector sees it that way
        from types import SimpleNamespace
        plan = SimpleNamespace(entities=[{"id": case["actor_id"]}], institutions=[],
                               scheduled_events=[{"etype": "decision_opportunity",
                                                  "participants": [case["actor_id"]]}],
                               actor_decisions=[], relations=[], quantities=[],
                               _intention_stances=[], question=case["situation"][:120])
        tiers = RelevantActorSelector().select(plan, plan.question)
    rt = QualitativeActorPolicyRuntime(QualitativeDecisionEngine(cfg), mode=mode, tiers=tiers,
                                       selector=None)
    t0 = _time.monotonic()
    novel = 0
    for i in range(n):
        w, decision = build_case_world(case, branch_id=f"b{i:03d}")
        _, posterior, _tr = rt.decide(None, [w], case["actor_id"], decision=dict(decision),
                                      seed=seed * 7919 + i)
        if (posterior.provenance.get("qualitative") or {}).get("resolution") == "novel_compiled":
            novel += 1
    # cluster-2.0 scoring: novel phrasings map onto candidates via validated ontology anchors
    # and (when a mapper backend is supplied) conservative auditable LLM equivalence — the raw
    # wording, method, and explanation are preserved on every row
    clusterer = ActionClusterer(candidates=list(case["candidate_actions"]),
                                known_entities=[case["actor_id"]], llm=mapper_llm)
    agg = aggregate_actor_decisions(rt.decision_records, clusterer=clusterer
                                    ).get(case["actor_id"], {})
    rows = agg.get("rows", [])
    dist: dict = {}
    for row in rows:
        dist[row["cluster_base"]] = dist.get(row["cluster_base"], 0.0) + 1.0
    z = sum(dist.values()) or 1.0
    return {"distribution": {k: v / z for k, v in dist.items()},
            "llm_calls": rt.engine.calls_used(), "latency_s": _time.monotonic() - t0,
            "n_samples": len(rows), "novel_rate": novel / max(1, n),
            "n_fallbacks": agg.get("n_excluded_numeric_fallbacks", 0),
            "floor_kind": "counted_frequency",
            "raw_rows": [{k: r.get(k, "") for k in ("hypothesis_id", "action_name",
                                                    "cluster_base", "cluster_method",
                                                    "cluster_explanation", "decision_source")}
                         for r in rows]}


# ---------------------------------------------------------------------------- scoring
def score_case(result: dict, case: dict) -> dict:
    dist = result["distribution"]
    actual = case["actual_action"]
    n = result.get("n_samples") or 0
    floor = 1.0 / (2 * n) if (result["floor_kind"] == "counted_frequency" and n) else 1e-4
    p_actual = float(dist.get(actual, 0.0))
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    top1 = ranked[0][0] if ranked else ""
    top2 = {name for name, _ in ranked[:2]}
    keys = set(dist) | {actual}
    brier = sum((float(dist.get(k, 0.0)) - (1.0 if k == actual else 0.0)) ** 2 for k in keys)
    return {"case_id": case["case_id"], "actual": actual, "top1": top1,
            "correct": top1 == actual, "top2_correct": actual in top2,
            "p_actual": round(p_actual, 4),
            "log_loss": round(-math.log(max(p_actual, floor)), 4),
            "brier": round(brier, 4), "confidence": round(ranked[0][1], 4) if ranked else 0.0,
            "distribution": {k: round(v, 4) for k, v in ranked},
            "novel_rate": result.get("novel_rate", 0.0),
            "n_fallbacks": result.get("n_fallbacks", 0),
            "llm_calls": result["llm_calls"], "latency_s": round(result["latency_s"], 2)}


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {"n_cases": len(rows),
            "next_action_accuracy": round(sum(r["correct"] for r in rows) / n, 4),
            "top2_accuracy": round(sum(r["top2_correct"] for r in rows) / n, 4),
            "mean_log_loss": round(sum(r["log_loss"] for r in rows) / n, 4),
            "mean_brier": round(sum(r["brier"] for r in rows) / n, 4),
            "mean_confidence": round(sum(r["confidence"] for r in rows) / n, 4),
            "confidence_accuracy_gap": round(abs(sum(r["confidence"] for r in rows) / n -
                                                 sum(r["correct"] for r in rows) / n), 4),
            "mean_novel_rate": round(sum(r.get("novel_rate", 0.0) for r in rows) / n, 4),
            "total_fallbacks": sum(r.get("n_fallbacks", 0) for r in rows),
            "total_llm_calls": sum(r["llm_calls"] for r in rows),
            "total_latency_s": round(sum(r["latency_s"] for r in rows), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--hypotheses", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=5,
                    help="cases per saved checkpoint; a killed run resumes from completed batches")
    ap.add_argument("--run-id", default="",
                    help="checkpoint namespace; reuse the same id to resume a killed run")
    ap.add_argument("--backend", default="deepseek", choices=("deepseek", "scripted"))
    args = ap.parse_args()
    cases = load_cases(Path(args.cases))
    if args.backend == "deepseek":
        from swm.api.deepseek_backend import deepseek_chat_fn
        decide_llm = deepseek_chat_fn(temperature=0.9, max_tokens=2000)
        hypo_llm = deepseek_chat_fn(temperature=0.8, max_tokens=3600)
        persona_llm = deepseek_chat_fn(temperature=0.3, max_tokens=700)
        mapper_llm = deepseek_chat_fn(temperature=0.1, max_tokens=300)
    else:
        decide_llm = hypo_llm = persona_llm = _scripted_backend(None)
        mapper_llm = None
    out = {"schema_version": "actor.policy.benchmark.v2", "backend": args.backend,
           "n_cases": len(cases), "hypotheses": args.hypotheses, "samples": args.samples,
           "seed": args.seed, "cluster_version": "cluster-2.0", "arms": {}}

    def one(arm, case):
        try:
            if arm == "A":
                res = run_numeric(case, seed=args.seed)
            elif arm == "B":
                res = run_persona(case, llm=persona_llm, seed=args.seed)
            else:
                res = run_qualitative(case, llm=decide_llm, hypothesis_llm=hypo_llm,
                                      mapper_llm=mapper_llm, mode=ARMS[arm],
                                      hypotheses=args.hypotheses, samples=args.samples,
                                      seed=args.seed)
            row = score_case(res, case)
        except Exception as e:  # noqa: BLE001 — one case must never kill the arm; scored as empty
            row = score_case({"distribution": {}, "llm_calls": 0, "latency_s": 0.0,
                              "n_samples": 0, "floor_kind": "counted_frequency"}, case)
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        print(f"[{arm}:{ARMS[arm]}] {case['case_id']}: top1={row['top1']} "
              f"actual={case['actual_action']} p={row['p_actual']} "
              f"calls={row['llm_calls']} {row['latency_s']}s"
              + (f" ERROR={row['error']}" if row.get("error") else ""), flush=True)
        return row

    # RESUMABLE BATCHES: never one long process that loses everything. Each batch of
    # --batch-size cases saves a checkpoint the moment it completes; relaunching with the
    # same --run-id skips completed batches and combines everything at the end.
    from concurrent.futures import ThreadPoolExecutor
    run_id = args.run_id or f"run{args.seed}_{Path(args.cases).stem}"
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        mode = ARMS[arm]
        rows = []
        for b0 in range(0, len(cases), max(1, args.batch_size)):
            batch = cases[b0:b0 + args.batch_size]
            ckpt = ckpt_dir / f"{run_id}_{arm}_batch{b0:03d}.json"
            if ckpt.exists():
                saved = json.loads(ckpt.read_text())
                if [r["case_id"] for r in saved] == [c["case_id"] for c in batch]:
                    rows.extend(saved)
                    print(f"[{arm}] batch {b0}-{b0 + len(batch) - 1}: resumed from checkpoint",
                          flush=True)
                    continue
            with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
                batch_rows = list(pool.map(lambda c: one(arm, c), batch))
            ckpt.write_text(json.dumps(batch_rows, indent=1, default=str))
            rows.extend(batch_rows)
            print(f"[{arm}] batch {b0}-{b0 + len(batch) - 1}: saved {ckpt.name}", flush=True)
        out["arms"][mode] = {"summary": summarize(rows), "cases": rows}
        print(f"== {mode}: {json.dumps(out['arms'][mode]['summary'])}", flush=True)
    out["run_id"] = run_id
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"actor_policy_benchmark_{int(_time.time())}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
