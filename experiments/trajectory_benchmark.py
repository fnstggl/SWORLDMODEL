"""Closed-loop trajectory benchmark — sequences, reactions, persistence, error compounding.

Each case is a REAL multi-step decision sequence (experiments/frozen_trajectory_cases.json).
Per branch, ONE persistent world runs the whole sequence: each step exposes only that step's
pre-as_of public record, the step's actor decides through the qualitative runtime and the
action EXECUTES (state, memories, expectations persist), then the next step continues in the
same evolved world. Scored against reality:

  * per-step next-action accuracy (counted branch distribution, cluster-2.0 onto candidates);
  * full-sequence accuracy (branches whose entire action sequence matches history);
  * accuracy-by-depth (does error compound?);
  * anticipated-reaction accuracy: when actor A's decision anticipates a named actor B, the
    anticipation text is resolved to a candidate action (deterministic token match; ambiguous
    → unscored, counted) and compared to B's REAL next action;
  * persistence checks (state revisions grow; the branch's hypothesis never switches).

Arms: D persistent_qualitative_llm_policy vs C stateless_llm_policy — the trajectory-level
persistence ablation. Contamination caveat identical to the single-decision corpus.

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/trajectory_benchmark.py
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swm.world_model_v2.information import InformationItem, InformationLedger
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.qualitative_actor import (
    ActionClusterer, QualitativeActorPolicyRuntime, QualitativeConfig,
    QualitativeDecisionEngine, load_actor_state,
)
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState

CASES = Path("experiments/frozen_trajectory_cases.json")
RESULTS = Path("experiments/results")


def _ts(day: str) -> float:
    return float(calendar.timegm(_time.strptime(day, "%Y-%m-%d"))) + 12 * 3600.0


def leakage_check(case: dict) -> None:
    prev = float("-inf")                                    # pre-1970 cases have negative epochs
    for ev in case["base_evidence"]:
        if _ts(ev["date"]) > _ts(case["steps"][0]["as_of"]):
            raise ValueError(f"{case['case_id']}: base evidence postdates step 1")
    for step in case["steps"]:
        at = _ts(step["as_of"])
        if at < prev:
            raise ValueError(f"{case['case_id']} step {step['step']}: time went backwards")
        prev = at
        for ev in step["evidence_additions"]:
            if _ts(ev["date"]) > at:
                raise ValueError(f"{case['case_id']} step {step['step']}: future evidence")
        if step["actual_action"] not in step["candidate_actions"]:
            raise ValueError(f"{case['case_id']} step {step['step']}: label not in candidates")


def load_cases(path: Path = CASES) -> list[dict]:
    data = json.loads(Path(path).read_text())
    for case in data["cases"]:
        leakage_check(case)
    return data["cases"]


def build_world(case: dict, branch_id: str) -> WorldState:
    now = _ts(case["steps"][0]["as_of"])
    w = WorldState(case["case_id"], branch_id, SimulationClock(now, now),
                   network=RelationGraph(), information=InformationLedger())
    for aid, spec in case["actors"].items():
        e = Entity(aid)
        e.set("roles", F([spec["role"]], status="observed"))
        e.set("goals", F(list(spec.get("goals") or []), status="inferred"))
        e.set("commitments", F(list(spec.get("commitments") or []), status="observed"))
        e.set("past_actions", F([], status="observed"))
        w.entities[aid] = e
    for rel in case.get("relations") or []:
        w.network.add(rel["src"], rel["rel"], rel["dst"])
    for i, ev in enumerate(case["base_evidence"]):
        _expose(w, f"base_{i}", ev)
    return w


def _expose(w, iid, ev):
    w.information.publish(InformationItem(iid, ev["text"], source="public_record",
                                          created_at=_ts(ev["date"])))
    for aid in w.entities:
        w.information.expose(aid, iid, _ts(ev["date"]))


def _resolve_anticipation(text: str, candidates: list[str]) -> str | None:
    """Deterministic anticipation→candidate resolution: exactly one candidate's tokens appear
    in the anticipation text ⇒ that candidate; zero or several ⇒ unscored (None)."""
    hits = []
    low = (text or "").lower()
    for cand in candidates:
        tokens = cand.split("_")
        if all(re.search(rf"\b{re.escape(t)}", low) for t in tokens):
            hits.append(cand)
    return hits[0] if len(hits) == 1 else None


def run_case(case: dict, *, mode: str, llm, hypo_llm, branches: int = 6, seed: int = 0) -> dict:
    cfg = QualitativeConfig(llm=llm, hypothesis_llm=hypo_llm, n_hypotheses=3,
                            max_llm_calls=4 * branches * len(case["steps"]) + 8)
    rt = QualitativeActorPolicyRuntime(QualitativeDecisionEngine(cfg), mode=mode)
    worlds = [build_world(case, f"b{i:03d}") for i in range(branches)]
    step_rows, anticipations = [], []
    t0 = _time.monotonic()
    for si, step in enumerate(case["steps"]):
        at = _ts(step["as_of"])
        for w in worlds:
            if at > w.clock.now:
                w.clock.advance_to(at)
            for j, ev in enumerate(step["evidence_additions"]):
                _expose(w, f"s{si}_{j}", ev)
        clusterer = ActionClusterer(candidates=list(step["candidate_actions"]),
                                    known_entities=list(case["actors"]))
        # label-position debias: the file lists the actual action first; shuffle per step
        import random as _random
        candidates = list(step["candidate_actions"])
        _random.Random(f"{case['case_id']}:{step['step']}").shuffle(candidates)
        for bi, w in enumerate(worlds):
            sel, post, tr = rt.decide(
                None, [w], step["actor"],
                decision={"situation": step["situation"],
                          "candidate_actions": candidates},
                seed=seed * 7919 + 100 * si + bi)
            rt.execute(w, sel, post, tr, seed=seed * 7919 + 100 * si + bi)
            q = (post.provenance or {}).get("qualitative") or {}
            crow = clusterer.cluster_row({
                "action_name": sel.action_name,
                "target": sel.target.target_id,
                "ontology_anchor": (sel.parameters or {}).get("ontology_anchor"),
                "intended_effect": (sel.parameters or {}).get("intended_effect", ""),
                "observability_intent": (sel.parameters or {}).get("observability_intent", ""),
                "timing": (sel.parameters or {}).get("timing", "")})
            step_rows.append({"step": step["step"], "actor": step["actor"], "branch": w.branch_id,
                              "chosen": sel.action_name, "cluster_base": crow["cluster_base"],
                              "cluster_method": crow["cluster_method"],
                              "actual": step["actual_action"],
                              "decision_source": q.get("decision_source", "numeric_fallback"),
                              "hypothesis_id": q.get("hypothesis_id", "")})
            for r in q.get("anticipated_reactions_subjective") or []:
                other = clusterer.normalize_target(str(r.get("actor_or_group", "")))
                anticipations.append({"step": step["step"], "branch": w.branch_id,
                                      "by": step["actor"], "about": other,
                                      "text": str(r.get("expected_reaction", ""))})
    # ---- scoring -------------------------------------------------------------------
    per_step, correct_by_depth = [], []
    for si, step in enumerate(case["steps"]):
        rows = [r for r in step_rows if r["step"] == step["step"] and r["actor"] == step["actor"]
                and r["decision_source"] != "numeric_fallback"]
        counts: dict = {}
        for r in rows:
            counts[r["cluster_base"]] = counts.get(r["cluster_base"], 0) + 1
        total = sum(counts.values()) or 1
        dist = {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
        top1 = next(iter(dist), "")
        per_step.append({"step": step["step"], "actor": step["actor"],
                         "actual": step["actual_action"], "top1": top1,
                         "correct": top1 == step["actual_action"],
                         "p_actual": dist.get(step["actual_action"], 0.0),
                         "distribution": dist, "n_branches_scored": len(rows)})
        correct_by_depth.append(int(top1 == step["actual_action"]))
    # full-sequence accuracy per branch
    seq_ok = 0
    for bi in range(branches):
        bid = f"b{bi:03d}"
        ok = all(any(r["branch"] == bid and r["step"] == s["step"]
                     and r["cluster_base"] == s["actual_action"] for r in step_rows)
                 for s in case["steps"])
        seq_ok += int(ok)
    # anticipated-reaction accuracy vs the OTHER actor's REAL next step
    scored, correct, unscored = 0, 0, 0
    for a in anticipations:
        nxt = next((s for s in case["steps"] if s["actor"] == a["about"]
                    and s["step"] > a["step"]), None)
        if nxt is None:
            continue
        predicted = _resolve_anticipation(a["text"], nxt["candidate_actions"])
        if predicted is None:
            unscored += 1
            continue
        scored += 1
        correct += int(predicted == nxt["actual_action"])
        a.update(predicted=predicted, actual=nxt["actual_action"],
                 correct=predicted == nxt["actual_action"])
    # persistence checks
    persistence = {"hypothesis_switches": 0, "final_revision_counts": []}
    for bi in range(branches):
        bid = f"b{bi:03d}"
        # persistence is per (branch, ACTOR): each actor holds its own hypothesis in a branch
        for aid in case["actors"]:
            hyps = {r["hypothesis_id"] for r in step_rows
                    if r["branch"] == bid and r["actor"] == aid and r["hypothesis_id"]}
            if len(hyps) > 1:
                persistence["hypothesis_switches"] += 1
        state = None
        for aid in case["actors"]:
            state = load_actor_state(worlds[bi], aid)
            if state is not None:
                persistence["final_revision_counts"].append(len(state.revision_log))
    n_steps = len(case["steps"]) or 1
    return {"case_id": case["case_id"], "mode": mode,
            "per_step": per_step,
            "step_accuracy": round(sum(s["correct"] for s in per_step) / n_steps, 4),
            "accuracy_by_depth": correct_by_depth,
            "sequence_accuracy": round(seq_ok / branches, 4),
            "reaction_prediction": {"scored": scored, "correct": correct,
                                    "unscored_ambiguous": unscored,
                                    "accuracy": round(correct / scored, 4) if scored else None},
            "anticipations": anticipations[:24],
            "persistence": persistence,
            "llm_calls": rt.engine.calls_used(),
            "wall_s": round(_time.monotonic() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=int, default=6)
    ap.add_argument("--arms", default="D,C")
    ap.add_argument("--backend", default="deepseek", choices=("deepseek", "scripted"))
    args = ap.parse_args()
    cases = load_cases()
    if args.backend == "deepseek":
        from swm.api.deepseek_backend import deepseek_chat_fn
        llm = deepseek_chat_fn(temperature=0.9, max_tokens=2000)
        hypo = deepseek_chat_fn(temperature=0.8, max_tokens=3600)
    else:
        def llm(prompt):
            first = re.search(r"^- ([a-z_@]+): ", prompt, flags=re.M)
            return json.dumps({"decision": {"act_or_wait": "act",
                                            "chosen_action": first.group(1) if first else "wait"},
                               "actor_state_update": {}, "decision_summary": "scripted"})
        hypo = None
    modes = {"D": "persistent_qualitative_llm_policy", "C": "stateless_llm_policy"}
    out = {"schema_version": "trajectory.benchmark.v1", "branches": args.branches, "arms": {}}
    for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        mode = modes[arm]
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(lambda c: run_case(c, mode=mode, llm=llm, hypo_llm=hypo,
                                                    branches=args.branches), cases))
        n = len(rows) or 1
        out["arms"][mode] = {
            "cases": rows,
            "summary": {
                "mean_step_accuracy": round(sum(r["step_accuracy"] for r in rows) / n, 4),
                "mean_sequence_accuracy": round(sum(r["sequence_accuracy"] for r in rows) / n, 4),
                "accuracy_by_depth_pooled": [
                    round(sum(r["accuracy_by_depth"][d] for r in rows
                              if len(r["accuracy_by_depth"]) > d) /
                          max(1, sum(1 for r in rows if len(r["accuracy_by_depth"]) > d)), 4)
                    for d in range(max(len(r["accuracy_by_depth"]) for r in rows))],
                "reaction_scored": sum(r["reaction_prediction"]["scored"] for r in rows),
                "reaction_correct": sum(r["reaction_prediction"]["correct"] for r in rows),
                "reaction_unscored": sum(r["reaction_prediction"]["unscored_ambiguous"]
                                         for r in rows),
                "hypothesis_switches": sum(r["persistence"]["hypothesis_switches"] for r in rows),
                "total_llm_calls": sum(r["llm_calls"] for r in rows),
                "total_wall_s": round(sum(r["wall_s"] for r in rows), 1)}}
        print(f"== {mode}: {json.dumps(out['arms'][mode]['summary'])}", flush=True)
        for r in rows:
            print(f"  {r['case_id']}: steps={[s['correct'] for s in r['per_step']]} "
                  f"seq={r['sequence_accuracy']} reactions={r['reaction_prediction']}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"trajectory_benchmark_{int(_time.time())}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
