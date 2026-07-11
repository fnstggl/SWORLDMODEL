"""Run the OmniBehavior slice — DeepSeek here, OSim on the pod (same items). See swm/eval/omnibehavior_eval.

Run: DEEPSEEK_API_KEY=… [OSIM_ENDPOINT=… OSIM_MODEL=…] \
       python -m experiments.behavior_pilot.omnibehavior_run --users 10 --per-user 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RESULT = "experiments/results/omnibehavior_eval.json"


def run(n_users, per_user):
    from swm.eval.omnibehavior_eval import build_items, download_users, eval_arm
    paths = download_users(n_users=n_users)
    items = build_items(paths, per_user=per_user)
    real_rate = sum(it["y"] for it in items) / len(items) if items else 0
    print(f"users={len(paths)} items={len(items)} real_action_rate={real_rate:.2f}")
    arms = {}
    from swm.api.deepseek_backend import default_chat_fn
    ds = default_chat_fn(system="Reply ONLY compact JSON.", max_tokens=80, temperature=0.4)
    if ds is not None:
        arms["deepseek"] = ds
    ep = os.environ.get("OSIM_ENDPOINT")
    if ep:
        import urllib.request

        def osim(prompt):
            body = json.dumps({"model": os.environ.get("OSIM_MODEL", "osim"),
                               "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 80, "temperature": 0.4}).encode()
            r = urllib.request.Request(ep.rstrip("/") + "/chat/completions", data=body,
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        arms["osim"] = osim
    if not arms:
        print("HARD STOP: no arm (set DEEPSEEK_API_KEY and/or OSIM_ENDPOINT)."); return 2
    out = {}
    for name, fn in arms.items():
        sc = eval_arm(fn, items)
        out[name] = sc
        print(f"  {name:10s} {json.dumps(sc)}")
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps({"n_items": len(items), "arms": out}, indent=1))
    print(f"wrote {RESULT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--per-user", type=int, default=8)
    a = ap.parse_args()
    sys.exit(run(a.users, a.per_user))


if __name__ == "__main__":
    main()
