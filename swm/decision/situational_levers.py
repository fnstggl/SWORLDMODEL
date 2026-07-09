"""Per-recipient SITUATIONAL levers — the recipient-specific message qualities the universal set can't
enumerate ahead of time.

The universal message levers (personalization, relevance, clarity, credibility_proof, responder_incentive,
ask_directness, low_effort_ask, pushiness, warmth, length_fit, credential_signaling) are the physics of any
inbound ask. But some of what moves a reply is specific to the recipient: a genuinely CONTRARIAN thesis
moves Peter Thiel; TRACTION/hustle moves Mark Cuban; METHODOLOGICAL RIGOR moves an academic. Hardcoding
"contrarian_pitch" into the universal set was overfitting to one person.

So instead the LLM PROPOSES the situational levers for THIS recipient — each a name, a description, and its
recipient-conditioned elasticity (signed per-unit logit effect) — exactly the "general set + per-question
generated set" pattern the repo already uses for priors (`llm_prior.prior_from_llm`, `prior_registry`). The
message encoder then scores a message on each lever, and the scorer adds `elasticity × lever_score`.

Pluggable and degrading: with a live `chat_fn` the LLM generates levers; offline it returns none, so the
model falls back to the pure universal set (still fully functional, just less recipient-specific).
"""
from __future__ import annotations

import json
import re

from swm.decision.strategy_scorer import MESSAGE_VARS, Lever

_UNIVERSAL_NOTE = ("The message is ALREADY scored on these universal levers, so do NOT propose anything "
                   "equivalent to them: " + ", ".join(MESSAGE_VARS) + ".")


def generate_levers(chat_fn, recipient_name: str, recipient_vars: dict | None = None, *,
                    evidence: str = "", k: int = 4) -> list:
    """Ask the LLM for up to k situational levers for this recipient. Each is a Lever(name, elasticity_mean,
    elasticity_sd, description, evidence). Returns [] if no chat_fn or on any error (pure universal model)."""
    if chat_fn is None:
        return []
    traits = ", ".join(f"{k2}={v:.2f}" for k2, v in (recipient_vars or {}).items())
    prompt = "\n".join([
        f"You are calibrating a model of whether {recipient_name} replies to a cold message.",
        "Propose the SITUATIONAL levers specific to this recipient: message qualities that would move "
        f"{recipient_name}'s reply decision MORE than they'd move an average person's.",
        _UNIVERSAL_NOTE,
        f"Inferred disposition: {traits}" if traits else "",
        f"Evidence:\n{evidence}" if evidence else "",
        f"Give up to {k} levers. For each: a short snake_case name, a one-line description, and "
        "'elasticity' = the signed per-unit effect on the log-odds of a reply (roughly -3..+3; positive "
        "if more of this quality makes a reply MORE likely for this person, negative if it backfires), "
        "and 'confidence' 0..1.",
        'Return ONLY a JSON array of {"name","description","elasticity","confidence"}. No prose.',
    ])
    try:
        raw = chat_fn(prompt)
        m = re.search(r"\[.*\]", raw, re.S)
        arr = json.loads(m.group(0)) if m else []
    except Exception:
        return []
    levers = []
    for d in arr[:k]:
        try:
            name = re.sub(r"[^a-z0-9_]", "", str(d["name"]).strip().lower().replace(" ", "_"))
            if not name or name in MESSAGE_VARS:
                continue
            elas = max(-3.0, min(3.0, float(d.get("elasticity", 0.0))))
            conf = max(0.05, min(1.0, float(d.get("confidence", 0.5))))
            # confidence -> prior sd: a confident lever has a tighter elasticity CI
            sd = 0.4 + 1.2 * (1.0 - conf)
            levers.append(Lever(name=name, elasticity_mean=elas, elasticity_sd=sd,
                                description=str(d.get("description", ""))[:160], evidence="llm situational lever"))
        except Exception:
            continue
    return levers


def levers_summary(levers: list) -> list:
    return [{"name": lv.name, "elasticity": round(lv.elasticity_mean, 2), "description": lv.description}
            for lv in levers]
