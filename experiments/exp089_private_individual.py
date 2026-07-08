"""EXP-089 — The private-individual path: dossier + user-supplied context, end to end.

EXP-086 proved the mechanism on ground truth (senators): reference-class-only inference is ~23% of the way
to a measurement; give the model real EVIDENCE and it jumps to ~87%. Senators are public — the LLM already
knows them. This exercises the SAME pillar stack for the case the product actually faces: a PRIVATE
individual (someone you're messaging) the model cannot look up, where the evidence must come from the message
history and FROM THE USER (the relationship, how they met, their read on the person).

It demonstrates, end to end with the live LLM:
  1. THIN context -> the honest 'ask the user' gate fires (we don't guess from nothing).
  2. RICH user-supplied context -> assemble a dossier -> infer the variables through the three pillars
     (evidence shrunk toward the reference-class base rate by calibrated uncertainty).
  3. The dossier SHARPENS the estimate vs a cold reference-class guess (tighter sd) AND moves the value off
     the base rate toward what the evidence says — the 23%->87% mechanism, now for a private person.

This is a demonstration of the product path (not a ground-truth backtest — EXP-086 is the quantitative proof
on measurable truth). Run: DEEPSEEK_API_KEY=... python -m experiments.exp089_private_individual
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from swm.api.retrieval_grounding import CalibratedExtractor
from swm.api.anchored_extractor import AnchoredExtractor
from swm.variables.dossier import DossierAssembler, needs_user_context, context_questions, infer_variables

RESULT = "experiments/results/exp089_private_individual.json"

# The scenario: you want to send a specific ask to someone you know. The variables the outcome turns on.
ENTITY = "Jordan (a founder you met twice at conferences)"
QUESTION = "Will Jordan agree to a 30-minute intro call if I email a warm, specific ask?"
VARIABLES = ["Jordan's openness to a cold-ish intro call (0=never, 1=eagerly)",
             "strength of your existing relationship with Jordan (0=stranger, 1=close)",
             "Jordan's current bandwidth / how busy they are right now (0=slammed, 1=wide open)"]

# What the user knows that no dataset does — the highest-signal evidence for a private individual.
USER_CONTEXT = [
    "I met Jordan twice at fintech conferences; we had two good 20-minute hallway chats about payments infra.",
    "Jordan runs a seed-stage startup, seemed genuinely curious and generous with time, replied to my LinkedIn "
    "comment within a day.",
    "Last exchange was 3 weeks ago; they said 'let's find time to go deeper' but we never scheduled.",
]


def _num_llm():
    from swm.api.deepseek_backend import deepseek_chat_fn
    return deepseek_chat_fn(system="You estimate a numeric value with a calibrated interval. Return ONLY JSON.",
                            temperature=0.0, max_tokens=120)


def run():
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("EXP-089 needs DEEPSEEK_API_KEY (live demonstration).")
        return {"status": "no_key"}

    extractor = AnchoredExtractor(CalibratedExtractor(_num_llm()))
    assembler = DossierAssembler(search_fn=None)          # a private person: no public lookup, user context only

    # 1. THIN: nothing supplied -> the gate says ASK, and gives the exact questions.
    thin = assembler.assemble(ENTITY, question=QUESTION)
    ask = needs_user_context(thin)
    questions = context_questions(ENTITY, QUESTION)

    # 2. COLD (reference-class only) vs 3. DOSSIER (user context) — same pillar stack, different evidence.
    cold = infer_variables(assembler.assemble(ENTITY, question=QUESTION), VARIABLES, extractor, question=QUESTION)
    rich_dossier = assembler.assemble(ENTITY, user_context=USER_CONTEXT, question=QUESTION)
    dossier_inf = infer_variables(rich_dossier, VARIABLES, extractor, question=QUESTION)

    # the dossier should (a) fire OFF the ask-gate and (b) tighten the estimates vs cold
    tighter = sum(1 for v in VARIABLES
                  if v in dossier_inf and v in cold and dossier_inf[v]["sd"] <= cold[v]["sd"] + 1e-9)
    moved = sum(1 for v in VARIABLES
                if v in dossier_inf and v in cold and abs(dossier_inf[v]["value"] - cold[v]["value"]) > 0.02)

    out = {"scenario": QUESTION, "entity": ENTITY,
           "thin_context_needs_user": ask, "context_questions": questions,
           "dossier_strength": {"cold": 0.0, "with_user_context": round(rich_dossier.strength, 3)},
           "cold_reference_class_inference": cold,
           "dossier_inference": dossier_inf,
           "variables_tightened_by_dossier": f"{tighter}/{len(VARIABLES)}",
           "variables_moved_off_base_rate": f"{moved}/{len(VARIABLES)}",
           "anchor_to_EXP086": "reference-class ~23% of the measurement gap; evidence (a real dossier) ~87%"}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-089  private-individual path: dossier + user-supplied context, end to end")
    print(f"  scenario: {QUESTION}")
    print(f"  1. thin context -> needs_user_context = {ask}  (would ask: '{questions[0]}')")
    print(f"  dossier strength: cold 0.0 -> with user context {rich_dossier.strength:.2f}")
    print(f"  2/3. inference per variable (cold reference-class  vs  dossier):")
    for v in VARIABLES:
        c, d = cold.get(v), dossier_inf.get(v)
        if c and d:
            print(f"     - {v[:52]:52s}  cold {c['value']:+.2f}±{c['sd']:.2f}   dossier {d['value']:+.2f}±{d['sd']:.2f}")
    print(f"  dossier tightened {tighter}/{len(VARIABLES)} estimates, moved {moved}/{len(VARIABLES)} off the base rate")
    print(f"  -> the EXP-086 23%->87% mechanism, now for a private person: evidence from the USER does the work.")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
