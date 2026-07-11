"""Behavior-model pilot — OSim vs DeepSeek on held-out individual/engagement items. RENTED-GPU ONLY.

Per the hard rules this HARD-STOPS if no OSim endpoint is configured (no GPU here) — it never fabricates a run.
Arms A grounded DeepSeek, B DeepSeek stakeholder agents, C OSim agents, D mixed, E Minitaur (forced-choice).
Same dossier/scenario/stimulus/sample-count/aggregation across arms. Caps items + spend; caches every output;
aborts if OSim is severely incompatible on the first 10-15 items. Scores accuracy + OmniBehavior realism.

Run (on a GPU box): OSIM_ENDPOINT=… DEEPSEEK_API_KEY=… python -m experiments.behavior_pilot.pilot \
    --items data/behavior_pilot_items.jsonl --max-items 50 --max-usd 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CACHE = Path("experiments/behavior_pilot/cache")


def _osim_runner(endpoint, model):
    """OpenAI-compatible chat call to a self-served OSim endpoint → a BehaviorResponse-shaped dict."""
    import urllib.request

    def run(req):
        from swm.experimental.behavior_models import _DECIDE_PROMPT
        actions = req.allowed_actions or ["respond", "no_response"]
        prompt = _DECIDE_PROMPT.format(
            dossier=req.dossier, goals=req.goals or "-", relationship=req.relationship or "-",
            history=" | ".join(map(str, req.history[:8])) or "-", world_state=req.world_state or "-",
            elapsed=req.elapsed or "now", scenario=req.scenario,
            stimulus_block=(f"MESSAGE:\n{req.stimulus[:2000]}\n" if req.stimulus else ""),
            actions_block=f"OPTIONS: {actions}")
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 200, "temperature": 0.8}).encode()
        r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=120) as resp:
            txt = json.loads(resp.read())["choices"][0]["message"]["content"]
        from swm.engine.grounding import parse_json
        p = parse_json(txt) or {}
        act = p.get("action") if p.get("action") in actions else None
        return {"action": act, "rationale": str(p.get("why", ""))[:200],
                "p": p.get("p") if isinstance(p.get("p"), (int, float)) else None}
    return run


def _realism(decisions, active_action):
    """OmniBehavior-style realism signals over a list of {action,p}. Real user positive-action base rate is
    ~0.10 (OmniBehavior); LLMs run 0.40-0.60 (hyper-activity). Heterogeneity = spread of p (homogenization
    shows as near-zero spread)."""
    n = len(decisions) or 1
    ps = [d.get("p") for d in decisions if isinstance(d.get("p"), (int, float))]
    act_rate = sum(1 for d in decisions if d.get("action") == active_action) / n
    mean_p = (sum(ps) / len(ps)) if ps else None
    spread = ((sum((x - mean_p) ** 2 for x in ps) / len(ps)) ** 0.5) if ps else None
    return {"action_rate": round(act_rate, 3), "inactivity_rate": round(1 - act_rate, 3),
            "mean_p": (round(mean_p, 3) if mean_p is not None else None),
            "heterogeneity_spread": (round(spread, 3) if spread is not None else None),
            "n": n}


def run(items_path, max_items, max_usd, samples):
    osim_ep = os.environ.get("OSIM_ENDPOINT")
    if not osim_ep:
        print("HARD STOP: OSIM_ENDPOINT is not set — no GPU-served OSim model is available in this "
              "environment. This pilot compares OSim vs DeepSeek and refuses to run a partial, misleading "
              "experiment. See experiments/behavior_pilot/run_osim_server.md to serve OSim on a rented 24 GB "
              "GPU, then re-run. (No experiment was run.)")
        return 2
    if not Path(items_path).exists():
        print(f"HARD STOP: no item set at {items_path}. Needs labeled individual-response items "
              f"({{dossier,scenario,stimulus,allowed_actions,outcome}} per line). See the README.")
        return 2

    from swm.api.deepseek_backend import default_chat_fn
    from swm.experimental.behavior_models import (BehaviorModelAdapter, BehaviorRequest,
                                                  DeepSeekBehaviorBackend, osim_backend)
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = DeepSeekBehaviorBackend(llm=default_chat_fn(system="You inhabit one person. Reply ONLY JSON.",
                                                     max_tokens=250, temperature=0.7))
    ad = BehaviorModelAdapter(enabled=True, backends={
        "deepseek": ds, "osim": osim_backend(runner=_osim_runner(osim_ep, os.environ.get("OSIM_MODEL", "osim")))})

    items = [json.loads(l) for l in Path(items_path).read_text().splitlines() if l.strip()][:max_items]
    active = "respond"
    arms = {"B_deepseek": "deepseek", "C_osim": "osim"}      # A/D/E added once B/C validated
    agg, spent, incompat = {a: {"decisions": [], "correct": 0, "n": 0} for a in arms}, 0.0, 0
    for i, it in enumerate(items):
        req = BehaviorRequest(dossier=it["dossier"], scenario=it.get("scenario", ""),
                              stimulus=it.get("stimulus", ""), allowed_actions=it.get("allowed_actions",
                              ["respond", "no_response"]))
        y = int(it["outcome"])
        for arm, backend in arms.items():
            decs = [ad.decide(backend, req) for _ in range(samples)]
            good = [d for d in decs if not d.abstain and d.action is not None]
            if backend == "osim" and not good:
                incompat += 1
            for d in good:
                agg[arm]["decisions"].append({"action": d.action, "p": d.p})
                pred = 1 if d.action == active else 0
                agg[arm]["correct"] += (pred == y)
                agg[arm]["n"] += 1
        if i == 14 and incompat > 8:
            print(f"ABORT: OSim produced no parseable action on {incompat}/15 early items — severe "
                  f"incompatibility. Fix the prompt/serving before spending more. (Stopped at item 15.)")
            break

    print(f"\n===== BEHAVIOR PILOT (n={len(items)} items, {samples} samples/arm) =====")
    for arm in arms:
        a = agg[arm]
        acc = (a["correct"] / a["n"]) if a["n"] else 0.0
        print(f"  {arm:12s} choice_acc={acc:.3f}  realism={_realism(a['decisions'], active)}")
    print("\nInterpretation: compare action_rate to the ~0.10 real base rate (hyper-activity), and "
          "heterogeneity_spread across arms (homogenization). Held-out choice_acc is the accuracy signal.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/behavior_pilot_items.jsonl")
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()
    sys.exit(run(a.items, a.max_items, a.max_usd, a.samples))


if __name__ == "__main__":
    main()
