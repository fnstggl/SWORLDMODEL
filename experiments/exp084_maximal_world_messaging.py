"""EXP-084 — The Maximal World, Part B: does modeling the TARGET as a full agent beat the flat shortcut on
"best launch messaging"?

Part A (cascade) showed agent structure doesn't beat the mean-field on an AGGREGATE observable — individual
detail is unidentifiable when you only see the population number. This is the complementary test, in the
regime where agent modeling SHOULD pay: the target is a single INDIVIDUAL (a CMV original poster), and the
commercial question is "which of several candidate messages will change THIS person's mind?" — exactly your
"what's the best text / launch to send them."

Two approaches rank the candidate messages; precision@1 on the SAME 200 held-out OPs as EXP-076:

  SHORTCUT (flat readout)   — score each message by text features, ignore who the target is. EXP-076 measured
                              this on this exact test set: lexical 0.555, DeepSeek-per-feature 0.615, and the
                              LLM's one-shot holistic judgment 0.640 (the strongest shortcut).
  AGENT WORLD (this file)   — model the TARGET as an agent first: one DeepSeek call per OP builds a structured
                              persona (their core values, the CRUX of why they hold the view, what kind of
                              argument moves them vs bounces off, their openness), THEN — conditioned on that
                              explicit model of the person — rates how much each candidate message would move
                              THEM. Ranking by the agent-simulated response is the "rebuild the person and
                              roll their reaction forward" approach, at the individual scale where it's
                              identifiable.

The question: does explicitly modeling the person as an agent beat the one-shot judgment (0.640) and the
readout — or does it land at the same ~0.64 irreducible ceiling, meaning even a perfect model of the person
can't recover information that isn't there (their mood the day they read it)? Either way it's a real answer.

Cache: experiments/results/exp084/agent_sim.json (resumable; DEEPSEEK_API_KEY from env only).
Run: DEEPSEEK_API_KEY=... python -m experiments.exp084_maximal_world_messaging
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

PEROP = "experiments/results/exp075/cmv_perop.json"
CACHE = "experiments/results/exp084/agent_sim.json"
CTRL_CACHE = "experiments/results/exp084/joint_control.json"
RESULT = "experiments/results/exp084_maximal_world_messaging.json"

# EXP-076 shortcut numbers on the IDENTICAL 200-test-OP split (ops[2288:2488]) — cited for a fair contrast.
SHORTCUT = {"random_pick": 0.5123, "lexical_readout": 0.555, "deepseek_features_readout": 0.615,
            "deepseek_oneshot_judgment": 0.640}


def _test_ops(ops):
    return ops[2288:2488]                                  # same held-out OPs as EXP-076


def _agent_simulate(test):
    """One DeepSeek call per OP: model the person, THEN rate each candidate message conditioned on that model.
    Returns {op_id: [score per arg]}. Cached & resumable."""
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(CACHE).read_text()) if Path(CACHE).exists() else {}
    todo = [o for o in test if o["op_id"] not in cache]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You simulate a specific person's mind to predict what would change it. "
                                     "First model the person, then judge each message AS that person. Return ONLY JSON.",
                              max_tokens=700)
        for j, o in enumerate(todo):
            args = o["args"]
            listing = "\n".join(f"[{i}] {a['text'][:700]}" for i, a in enumerate(args))
            prompt = (f"A person holds this view:\n{o['op_text'][:800]}\n\n"
                      f"STEP 1 — model this person as an agent: their core values, the CRUX of why they hold "
                      f"this view, what KIND of argument would actually move them vs bounce off, and how open "
                      f"they seem.\nSTEP 2 — as THAT person, for each candidate reply below, rate 0.0-1.0 how "
                      f"much it would move YOUR view.\n\nCandidate replies:\n{listing}\n\n"
                      f'Return ONLY JSON: {{"persona":"<one line>","scores":[<one number per reply, in order>]}}')
            for attempt in range(5):
                try:
                    raw = fn(prompt)
                    obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    scores = [float(x) for x in obj.get("scores", [])]
                    if len(scores) != len(args):
                        scores = (scores + [0.5] * len(args))[:len(args)]     # be robust to length drift
                    cache[o["op_id"]] = {"persona": str(obj.get("persona", ""))[:300], "scores": scores}
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  deepseek gave up at {j}/{len(todo)}: {str(e)[:70]}")
                    else:
                        time.sleep(2 ** attempt)
            else:
                continue
            if j % 20 == 0:
                Path(CACHE).write_text(json.dumps(cache))
                print(f"  simulated {j}/{len(todo)} OPs")
        Path(CACHE).write_text(json.dumps(cache))
    return cache


def _joint_control(test):
    """CONTROL arm: the SAME joint call, but WITHOUT the persona-modeling step — rate each reply for how much
    it changes this person's mind, all in one call. Isolates whether the drop is the persona step or just the
    joint-comparative FRAMING (vs the one-shot judgment, which scored each reply in a SEPARATE call)."""
    Path(CTRL_CACHE).parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(Path(CTRL_CACHE).read_text()) if Path(CTRL_CACHE).exists() else {}
    todo = [o for o in test if o["op_id"] not in cache]
    if todo and os.environ.get("DEEPSEEK_API_KEY"):
        from swm.api.deepseek_backend import deepseek_chat_fn
        fn = deepseek_chat_fn(system="You judge which replies would change a person's mind. Return ONLY JSON.",
                              max_tokens=400)
        for j, o in enumerate(todo):
            args = o["args"]
            listing = "\n".join(f"[{i}] {a['text'][:700]}" for i, a in enumerate(args))
            prompt = (f"A person holds this view:\n{o['op_text'][:800]}\n\nFor each candidate reply below, rate "
                      f"0.0-1.0 how much it would change this person's mind.\n\nReplies:\n{listing}\n\n"
                      f'Return ONLY JSON: {{"scores":[<one number per reply, in order>]}}')
            for attempt in range(5):
                try:
                    raw = fn(prompt)
                    obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    scores = [float(x) for x in obj.get("scores", [])]
                    if len(scores) != len(args):
                        scores = (scores + [0.5] * len(args))[:len(args)]
                    cache[o["op_id"]] = {"scores": scores}
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  control gave up at {j}/{len(todo)}: {str(e)[:60]}")
                    else:
                        time.sleep(2 ** attempt)
            else:
                continue
            if j % 20 == 0:
                Path(CTRL_CACHE).write_text(json.dumps(cache))
        Path(CTRL_CACHE).write_text(json.dumps(cache))
    return cache


def _p1_from_scores(test, scoremap):
    ok = [o for o in test if o["op_id"] in scoremap and len(scoremap[o["op_id"]]["scores"]) == len(o["args"])]
    if len(ok) < len(test):
        return None, len(ok)
    hits = 0
    for o in ok:
        s = scoremap[o["op_id"]]["scores"]
        top = max(range(len(o["args"])), key=lambda i: s[i])
        hits += o["args"][top]["success"]
    return hits / len(ok), len(ok)


def run():
    ops = json.loads(Path(PEROP).read_text())
    test = _test_ops(ops)
    sim = _agent_simulate(test)
    ctrl = _joint_control(test)

    scored = [o for o in test if o["op_id"] in sim and len(sim[o["op_id"]]["scores"]) == len(o["args"])]
    missing = len(test) - len(scored)
    if missing:
        print(f"  INCOMPLETE: {missing}/{len(test)} OPs unsimulated — re-run to finish (resumes from cache).")
        return {"status": "incomplete", "n_simulated": len(scored), "n_needed": len(test)}

    hits = rand = 0
    for o in scored:
        s = sim[o["op_id"]]["scores"]
        top = max(range(len(o["args"])), key=lambda i: s[i])
        hits += o["args"][top]["success"]
        rand += sum(a["success"] for a in o["args"]) / len(o["args"])
    agent_p1 = hits / len(scored)
    rand_p1 = rand / len(scored)

    ctrl_p1, n_ctrl = _p1_from_scores(test, ctrl)
    best_shortcut = max(SHORTCUT["lexical_readout"], SHORTCUT["deepseek_features_readout"],
                        SHORTCUT["deepseek_oneshot_judgment"])
    out = {"experiment": "Maximal World Part B — target-as-agent messaging vs the flat shortcut",
           "data": "CMV, 200 held-out OPs (same split as EXP-076); precision@1 = top-ranked reply is a delta winner",
           "n_ops": len(scored),
           "random_pick": round(rand_p1, 4),
           "shortcut_readouts_exp076": SHORTCUT,
           "joint_control_no_persona": round(ctrl_p1, 4) if ctrl_p1 is not None else None,
           "AGENT_WORLD_target_as_agent": round(agent_p1, 4),
           "agent_vs_best_shortcut": round(agent_p1 - best_shortcut, 4),
           "agent_vs_oneshot_judgment": round(agent_p1 - SHORTCUT["deepseek_oneshot_judgment"], 4),
           "persona_effect_vs_joint_control": round(agent_p1 - ctrl_p1, 4) if ctrl_p1 is not None else None,
           "agent_vs_random": round(agent_p1 - rand_p1, 4)}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-084  MAXIMAL WORLD, Part B: model the TARGET as an agent vs the flat shortcut (CMV, 200 test OPs)")
    print(f"  random pick:                              {rand_p1:.4f}")
    print(f"  SHORTCUT lexical readout (EXP-076):       {SHORTCUT['lexical_readout']:.4f}")
    print(f"  SHORTCUT DeepSeek features (EXP-076):     {SHORTCUT['deepseek_features_readout']:.4f}")
    print(f"  SHORTCUT DeepSeek one-shot (independent): {SHORTCUT['deepseek_oneshot_judgment']:.4f}")
    print(f"  CONTROL joint scoring, NO persona:        {ctrl_p1 if ctrl_p1 is None else round(ctrl_p1,4)}  (isolates the framing)")
    print(f"  AGENT WORLD (model the person, then sim): {agent_p1:.4f}")
    print(f"  -> agent vs one-shot {out['agent_vs_oneshot_judgment']:+.4f} | persona effect vs control "
          f"{out['persona_effect_vs_joint_control']} | vs random {out['agent_vs_random']:+.4f}")
    verdict = ("AGENT WORLD beats the shortcut — modeling the person pays" if agent_p1 > best_shortcut + 0.02
               else "AGENT WORLD does NOT beat the shortcut — modeling the target as an agent adds no "
                    "recoverable signal at the individual scale either")
    print(f"  VERDICT: {verdict}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
