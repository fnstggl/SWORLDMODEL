"""The PR #115 composition seam — general planner decides, reply-first writes the words.

`make_message_realizer` returns the callable the planner injects: when a plan step is a
consequential person-to-person message, the general system has already chosen the target,
route, goal, timing, and desired response; this bridge hands ONLY the wording to the
reply-first planner (`swm/decision/reply_first.py`), whose truth / human-language / blind
outcome gates run unchanged, and returns the exact realized text plus its honesty label and
gate verdicts for embedding into the step. Message optimization never bypasses the general
action comparison — the realized message rides inside the candidate that must still win the
matched-world evaluation against non-message strategies.

`reply_first.py` is imported from its home in `swm/decision/`; nothing is duplicated into
the Phase 13 package.
"""
from __future__ import annotations


def make_message_realizer(chat_fn, *, sender_brief, dossier, hypotheses=None,
                          recipient_notes: str = "", persona_draws: int = 2,
                          trace_path: str = None):
    """Build realizer(target_id, intent, draft, candidate_id) -> {text, label, gates}.

    Fails CLOSED: any error inside realization returns None and the planner keeps the
    general step draft (recorded on the step's provenance) — a broken message subsystem
    never silently blocks or rewrites the general plan."""
    from swm.decision.reply_first import ReplyFirstPlanner

    def realize(*, target_id: str, intent: str, draft: str = "", candidate_id: str = ""):
        planner = ReplyFirstPlanner(
            chat_fn, sender_brief=sender_brief, dossier=dossier,
            hypotheses=list(hypotheses or []), recipient_notes=recipient_notes,
            trace_path=trace_path, persona_draws=persona_draws)
        pr = planner.run()
        winner_gates = next((f.get("gates") for f in pr.finalists
                             if f.get("ordinal_note") == "selected"), {})
        return {"text": pr.winner_text, "label": pr.label, "gates": winner_gates,
                "origin": pr.winner_origin, "n_llm_calls": pr.n_llm_calls,
                "system": "reply_first(PR#115)", "for_candidate": candidate_id,
                "target": target_id}

    return realize
