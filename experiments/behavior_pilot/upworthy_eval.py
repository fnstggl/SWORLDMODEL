"""Labeled headline-CLICK eval (Upworthy randomized A/B) — DeepSeek here, OSim on the pod, same prompts.

The causal engagement benchmark OSim should be good at: given the competing headlines of one randomized
Upworthy test, rank them by predicted click appeal; score precision@1 (picked the empirical CTR winner) and
pairwise accuracy vs random. Downloads the Upworthy exploratory CSV (CC-BY) on first run.

Run:  DEEPSEEK_API_KEY=… [OSIM_ENDPOINT=… OSIM_MODEL=…] \
        python -m experiments.behavior_pilot.upworthy_eval --tests 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RESULT = "experiments/results/upworthy_eval.json"
_RANK = """You are a typical Upworthy reader in 2015 scrolling a social feed. Here are competing headlines
for the SAME story. Rank them from the one YOU are most likely to click to least likely — react as a real
reader, not an editor. Headlines:
{items}
Return ONLY JSON: {{"order": [<headline numbers best-first, e.g. 2,1,3>]}}"""


def _rank_fn(chat):
    from swm.engine.grounding import parse_json

    def rank(headlines):
        items = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
        r = parse_json(chat(_RANK.format(items=items))) or {}
        order = r.get("order")
        if not isinstance(order, list):
            return None
        out = []
        for x in order:
            try:
                idx = int(x) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(headlines) and headlines[idx] not in out:
                out.append(headlines[idx])
        return out or None
    return rank


def _osim_chat(endpoint, model):
    import urllib.request

    def chat(prompt):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 60, "temperature": 0.7}).encode()
        r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    return chat


def run(n_tests):
    from swm.eval.response_datasets import download_upworthy, load_upworthy_tests, score_headline_ranking
    path = download_upworthy()
    tests = load_upworthy_tests(path, min_impressions=1000, limit=n_tests)
    print(f"loaded {len(tests)} randomized headline tests (>=1000 impressions/variant)")
    arms = {}
    from swm.api.deepseek_backend import default_chat_fn
    ds = default_chat_fn(system="Reply ONLY compact JSON.", max_tokens=60, temperature=0.7)
    if ds is not None:
        arms["deepseek"] = _rank_fn(ds)
    ep = os.environ.get("OSIM_ENDPOINT")
    if ep:
        arms["osim"] = _rank_fn(_osim_chat(ep, os.environ.get("OSIM_MODEL", "osim")))
    if not arms:
        print("HARD STOP: no arm (set DEEPSEEK_API_KEY and/or OSIM_ENDPOINT)."); return 2

    out = {}
    for name, rank_fn in arms.items():
        sc = score_headline_ranking(tests, rank_fn)
        out[name] = sc
        print(f"  {name:10s} precision@1={sc['precision_at_1']} (random {sc['random_p1']})  "
              f"pairwise_acc={sc['pairwise_accuracy']}  n={sc['n_tests']}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps({"n_tests": len(tests), "arms": out}, indent=1))
    print(f"wrote {RESULT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, default=60)
    sys.exit(run(ap.parse_args().tests))


if __name__ == "__main__":
    main()
