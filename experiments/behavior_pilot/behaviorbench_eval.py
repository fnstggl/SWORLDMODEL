"""Labeled forced-choice/economic-game eval — DeepSeek here, OSim on the GPU pod, same prompts.

Grades each arm on BehaviorBench economic games by DISTRIBUTIONAL alignment to real human choices
(Wasserstein-1, lower = more human-like). This is the LABELED predictive half the pilot needs — it cannot be
faked by a realism probe. Runs whichever arms are configured (DeepSeek if DEEPSEEK_API_KEY; OSim if
OSIM_ENDPOINT), so on the pod it produces the OSim column beside DeepSeek's committed baseline.

Run:  DEEPSEEK_API_KEY=… [OSIM_ENDPOINT=… OSIM_MODEL=…] \
        python -m experiments.behavior_pilot.behaviorbench_eval --games bomb dictator ultimatum_responder \
          --reps 6 --limit 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RESULT = "experiments/results/behaviorbench_eval.json"


def _osim_sample(endpoint, model):
    import urllib.request

    def sample(prompt):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 80, "temperature": 0.9}).encode()
        r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    return sample


def run(games, reps, limit):
    from swm.eval.behavior_eval import eval_arm
    arms = {}
    from swm.api.deepseek_backend import default_chat_fn
    ds = default_chat_fn(system="", max_tokens=80, temperature=0.9)
    if ds is not None:
        arms["deepseek"] = ds
    ep = os.environ.get("OSIM_ENDPOINT")
    if ep:
        arms["osim"] = _osim_sample(ep, os.environ.get("OSIM_MODEL", "osim"))
    if not arms:
        print("HARD STOP: no arm available (set DEEPSEEK_API_KEY and/or OSIM_ENDPOINT)."); return 2

    out = {}
    for name, fn in arms.items():
        print(f"\n--- arm: {name} ---")
        res = eval_arm(fn, games=games, reps=reps, limit=limit)
        out[name] = res
        for r in res["per_game"]:
            if r.get("wasserstein_norm") is not None:
                ci = r.get("wasserstein_ci95")
                cis = f" CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
                print(f"  {r['game']:20s} human_mean={r['human_mean']:>6} model_mean={str(r['model_mean']):>6} "
                      f"W1_norm={r['wasserstein_norm']:.4f}{cis} (lower=more human)  unparsed={r['n_unparsed']}")
            else:
                print(f"  {r['game']:20s} (no parseable samples)")
        print(f"  >> {name} mean_wasserstein_norm = {res['mean_wasserstein_norm']} over {res['n_games']} games")

    if len(out) == 2:
        d, o = out.get("deepseek", {}).get("mean_wasserstein_norm"), out.get("osim", {}).get("mean_wasserstein_norm")
        if d is not None and o is not None:
            print(f"\n===== VERDICT: OSim {'MORE' if o < d else 'NOT more'} human-aligned than DeepSeek "
                  f"(W1_norm {o} vs {d}) =====")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps({"games": games, "reps": reps, "limit": limit, "arms": out}, indent=1))
    print(f"wrote {RESULT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=["bomb", "dictator", "ultimatum_responder"])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    sys.exit(run(a.games, a.reps, a.limit))


if __name__ == "__main__":
    main()
