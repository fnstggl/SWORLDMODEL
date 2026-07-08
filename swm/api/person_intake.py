"""Person intake — the ask-the-user flow, wired into the front door.

When a question turns on a SPECIFIC INDIVIDUAL (will Jordan take the call? will this recruiter say yes? how
will my ex react?), the outcome is driven by that person's disposition/state — variables no dataset holds and
the LLM can't look up for a private person. EXP-086/089 established the honest move: don't guess from nothing.
Assemble a dossier (Pillar 1); if it's too thin, ASK THE USER for what only they know (the relationship, how
they met, their read on the person), then infer the person's variables from that.

`PersonIntake.preflight(question, user_context)` runs at the top of `WorldModel.simulate`:
  - not a specific-person question -> {'mode': 'proceed'} (unchanged behavior).
  - a person question with THIN evidence and no user context -> {'mode': 'ask', questions: [...]}: the
    simulation short-circuits and returns the questions instead of a fabricated forecast.
  - a person question WITH enough evidence (user context and/or public footprint) -> {'mode': 'proceed',
    enriched_context, inferred_person_variables}: the dossier is folded into the context and the person's
    high-leverage variables are grounded through the three-pillar stack before the sim runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from swm.api.retrieval_grounding import parse_json_lenient
from swm.variables.dossier import (DossierAssembler, needs_user_context, context_questions, infer_variables)


@dataclass
class PersonIntake:
    llm: object                            # identifies the person + the person-variables the outcome turns on
    search_fn: object = None               # public footprint (web); None for private-only
    extractor: object = None               # AnchoredExtractor — grounds the person-variables from the dossier
    ask_threshold: float = 0.35

    def identify(self, question):
        """One LLM call: does this question turn on a SPECIFIC individual, and which of their personal
        variables decide the outcome? Returns {'entity', 'variables'} or None for a non-person question."""
        if self.llm is None:
            return None
        prompt = (
            "Does the answer to this question depend mainly on ONE SPECIFIC individual person's disposition, "
            "relationship, or state (not an institution, market, population, or public official acting in "
            f'office)?\n\nQuestion: "{question}"\n\n'
            'If yes, return JSON {"is_person": true, "entity": "<the person, as named/described>", '
            '"variables": ["<personal variable the outcome turns on>", "..."]} (2-4 variables like their '
            "openness, the strength of your relationship, their current mood/bandwidth, their stance on the "
            'topic). If no, return {"is_person": false}.')
        r = parse_json_lenient(self.llm(prompt))
        if not r or not r.get("is_person") or not r.get("entity"):
            return None
        variables = [v for v in (r.get("variables") or []) if isinstance(v, str)][:4]
        return {"entity": r["entity"], "variables": variables or [f"{r['entity']}'s disposition toward the ask"]}

    def preflight(self, question, *, user_context="", as_of=None):
        ident = self.identify(question)
        if ident is None:
            return {"mode": "proceed", "person": None}
        entity, variables = ident["entity"], ident["variables"]
        dossier = DossierAssembler(self.search_fn).assemble(
            entity, user_context=(user_context or None), question=question, as_of=as_of)
        if needs_user_context(dossier, threshold=self.ask_threshold):
            return {"mode": "ask", "person": entity, "variables": variables,
                    "questions": context_questions(entity, question),
                    "reason": f"the outcome turns on {entity}, and there isn't enough about them to infer "
                              f"honestly — tell me what you know rather than have me guess"}
        inferred = (infer_variables(dossier, variables, self.extractor, question=question)
                    if self.extractor is not None else {})
        lines = "\n".join(f"- {p}" for p in dossier.passages)
        enriched = ((user_context + "\n" if user_context else "")
                    + f"What is known about {entity}:\n{lines}")
        return {"mode": "proceed", "person": entity, "dossier_strength": round(dossier.strength, 3),
                "inferred_person_variables": inferred, "enriched_context": enriched}


def build_person_intake(*, llm=None, search_fn=None):
    """Assemble a PersonIntake with the live DeepSeek + web backends and the three-pillar extractor. None if no
    LLM key (the front door then skips person intake and behaves exactly as before)."""
    if llm is None:
        from swm.api.live_grounding import json_llm
        llm = json_llm()
    if llm is None:
        return None
    if search_fn is None:
        from swm.api.live_search import web_search_fn
        search_fn = web_search_fn()
    from swm.api.retrieval_grounding import CalibratedExtractor
    from swm.api.anchored_extractor import AnchoredExtractor
    return PersonIntake(llm=llm, search_fn=search_fn,
                        extractor=AnchoredExtractor(CalibratedExtractor(llm)))
