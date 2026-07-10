"""Label-free realism probe — the cheapest decisive OSim-vs-DeepSeek test. No labeled dataset needed.

OmniBehavior's core finding: general LLMs are HYPER-ACTIVE — they predict 40-60% action rates where real
humans act ~10% of the time on cold outreach / engagement decisions, and they HOMOGENIZE personas. This probe
runs a bundled set of low-base-rate engagement scenarios through both backends N times and reports:
  - action_rate  (real cold-outreach reply / ad-click / like base rates are ~2-15%; closer = more human)
  - heterogeneity_spread (personas should DIFFER; near-zero spread = homogenization)
No ground-truth labels required — the human base rate IS the yardstick. If OSim lands near ~10% and DeepSeek
near ~50%, OSim is worth a labeled accuracy pilot; if both are hyper-active, OSim adds nothing here.

Run on the GPU pod (after serving OSim, see run_osim_server.md):
    OSIM_ENDPOINT=http://127.0.0.1:8000/v1 OSIM_MODEL=cmu-lti/osim-8b DEEPSEEK_API_KEY=… \
      python -m experiments.behavior_pilot.realism_probe --samples 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Low-base-rate engagement scenarios (a human would mostly NOT act). active action = the ENGAGE choice.
SCENARIOS = [
    {"dossier": "Marc, a busy Series-A VC, ~300 unread emails, only opens warm intros.",
     "scenario": "A cold email arrives pitching a seed-stage dev-tools startup, no mutual connection.",
     "stimulus": "Subject: 10x faster CI — 2 min? Hi Marc, we cut build times 90%. Worth a quick call?",
     "allowed_actions": ["respond", "no_response"]},
    {"dossier": "Dana, a mid-level engineer, scrolls X during lunch, rarely engages with ads.",
     "scenario": "A promoted post for an AI coding tool appears in the feed.",
     "stimulus": "Ship features 3x faster with our AI pair-programmer. Free trial.",
     "allowed_actions": ["click", "scroll_past"]},
    {"dossier": "Priya, a marketing manager, gets ~40 LinkedIn pitches/week, ignores most.",
     "scenario": "A recruiter cold-DMs about a lateral role at an unknown startup.",
     "stimulus": "Hi Priya — impressed by your work. Open to a Head of Growth role? 15 min?",
     "allowed_actions": ["reply", "ignore"]},
    {"dossier": "Tom, a retiree, checks email twice a day, cautious about anything unsolicited.",
     "scenario": "A newsletter he half-remembers subscribing to sends its weekly issue.",
     "stimulus": "This week: 5 dividend stocks for a safe retirement. Read more →",
     "allowed_actions": ["open_and_read", "delete"]},
    {"dossier": "Aisha, a PhD student, active on Reddit, upvotes sparingly.",
     "scenario": "A decent-but-unremarkable post appears in a subreddit she follows.",
     "stimulus": "[Post] 'A clean way to structure PyTorch training loops' (mildly useful, 200 words).",
     "allowed_actions": ["upvote", "keep_scrolling"]},
    {"dossier": "Carlos, a small-business owner, wary of sales calls, values his time.",
     "scenario": "A SaaS SDR cold-emails offering a demo.",
     "stimulus": "Hi Carlos, businesses like yours save 20 hrs/mo with us. Book a demo?",
     "allowed_actions": ["book_demo", "no_response"]},
    {"dossier": "Nina, a designer, follows 800 accounts, likes ~1 in 50 posts.",
     "scenario": "A design-tip carousel from an account she follows shows up.",
     "stimulus": "5 spacing rules that instantly improve any UI (swipe).",
     "allowed_actions": ["like", "scroll_past"]},
    {"dossier": "Sam, an overwhelmed founder, inbox-zero is a fantasy, replies only to investors/customers.",
     "scenario": "A partnership cold-email from a complementary startup.",
     "stimulus": "Hi Sam, our users need what you build — co-marketing idea inside?",
     "allowed_actions": ["reply", "archive"]},
]


def _osim_runner(endpoint, model):
    import urllib.request
    from swm.engine.grounding import parse_json
    from swm.experimental.behavior_models import _DECIDE_PROMPT

    def run(req):
        actions = req.allowed_actions
        prompt = _DECIDE_PROMPT.format(
            dossier=req.dossier, goals="-", relationship="-", history="-", world_state="-", elapsed="now",
            scenario=req.scenario,
            stimulus_block=(f"MESSAGE:\n{req.stimulus[:2000]}\n" if req.stimulus else ""),
            actions_block=f"OPTIONS: {actions}")
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 200, "temperature": 0.9}).encode()
        r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=120) as resp:
            txt = json.loads(resp.read())["choices"][0]["message"]["content"]
        p = parse_json(txt) or {}
        return {"action": p.get("action") if p.get("action") in actions else None,
                "p": p.get("p") if isinstance(p.get("p"), (int, float)) else None,
                "rationale": str(p.get("why", ""))[:120]}
    return run


def _stats(decisions):
    """decisions: [{engaged: bool, p: float|None}]. action_rate = fraction that chose the ENGAGE option."""
    n = len(decisions) or 1
    ps = [d["p"] for d in decisions if isinstance(d.get("p"), (int, float))]
    rate = sum(1 for d in decisions if d.get("engaged")) / n
    mean = (sum(ps) / len(ps)) if ps else None
    spread = ((sum((x - mean) ** 2 for x in ps) / len(ps)) ** 0.5) if ps else None
    return rate, (round(spread, 3) if spread is not None else None), len(decisions)


def run(samples):
    osim_ep = os.environ.get("OSIM_ENDPOINT")
    from swm.api.deepseek_backend import default_chat_fn
    from swm.experimental.behavior_models import (BehaviorModelAdapter, BehaviorRequest,
                                                  DeepSeekBehaviorBackend, osim_backend)
    backends = {}
    ds_llm = default_chat_fn(system="You inhabit one person. Reply ONLY JSON.", max_tokens=220, temperature=0.9)
    if ds_llm is not None:
        backends["deepseek"] = DeepSeekBehaviorBackend(llm=ds_llm)
    else:
        print("(no DEEPSEEK_API_KEY — skipping DeepSeek arm; our committed baseline is action_rate≈18.8%)")
    if osim_ep:
        backends["osim"] = osim_backend(runner=_osim_runner(osim_ep, os.environ.get("OSIM_MODEL", "osim")))
    else:
        print("(no OSIM_ENDPOINT — DeepSeek-only baseline; serve OSim on the pod to get the comparison)\n")
    if not backends:
        print("HARD STOP: no backend available (set DEEPSEEK_API_KEY and/or OSIM_ENDPOINT)."); return 2
    ad = BehaviorModelAdapter(enabled=True, backends=backends)

    per_arm = {a: [] for a in backends}
    for sc in SCENARIOS:
        engage = sc["allowed_actions"][0]                       # index 0 is always the ENGAGE/active choice
        req = BehaviorRequest(dossier=sc["dossier"], scenario=sc["scenario"], stimulus=sc["stimulus"],
                              allowed_actions=sc["allowed_actions"])
        for arm in backends:
            for _ in range(samples):
                d = ad.decide(arm, req)
                if not d.abstain and d.action is not None:
                    per_arm[arm].append({"engaged": d.action == engage, "p": d.p})

    print(f"===== REALISM PROBE ({len(SCENARIOS)} scenarios x {samples} samples) =====")
    print("  human cold-outreach/engagement base rate ≈ 0.02–0.15 (closer = more human-like)\n")
    print(f"  {'arm':10s} {'action_rate':>12s} {'inactivity':>11s} {'p_spread':>9s} {'n':>4s}")
    for arm in backends:
        rate, spread, n = _stats(per_arm[arm])
        print(f"  {arm:10s} {rate:>12.1%} {1-rate:>11.1%} {str(spread):>9s} {n:>4d}")
    print("\n  READ: if OSim action_rate ≈ 10-20% and DeepSeek ≈ 40-60%, OSim fixes the hyper-activity bias "
          "→ worth a labeled accuracy pilot. If both are ≈50%, OSim adds nothing on realism.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    sys.exit(run(ap.parse_args().samples))


if __name__ == "__main__":
    main()
