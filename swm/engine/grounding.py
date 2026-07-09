"""Scene grounding — retrieve the facts that DEFINE the situation, and abstain loudly when we can't.

Stage 0 of the vision. Before anything is simulated, the engine must know what the scene actually is: for
"who wins the NY-10 primary" that means the actual candidates, the incumbent, polls, money, endorsements,
the election date. The old path shredded the question into latent constructs ("name_recognition = 0.5 ±
0.23, source=retrieval") and threw away the one hard fact the web returned. This module does the opposite:

  1. The LLM writes a CHECKLIST of the deciding facts (what would a domain expert need to know?) and the
     targeted queries to find them.
  2. The retrieval stack fetches real, dated passages (swm/engine/retrieval.py).
  3. The LLM DISTILLS passages → facts, each with a citation. It is instructed to mark a checklist item
     MISSING rather than fill it from its own knowledge — a fact without a passage is not a fact here.
  4. The result is a SceneDossier with a coverage score and an explicit `missing` list. If the deciding
     facts are missing, `abstain=True` with a human-readable reason — LOUD, never a neutral prior dressed
     as evidence.

The distiller also checks whether the question is ALREADY RESOLVED by the evidence (an election that
already happened, a decision already announced). A world model that doesn't notice the world already
answered the question is theater; `resolved` short-circuits the simulation with provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.engine.retrieval import multi_search


def parse_json(txt):
    """Lenient JSON from an LLM reply (dict as-is; strip fences; first {...} block)."""
    import re
    if isinstance(txt, dict):
        return txt
    if not isinstance(txt, str):
        return None
    s = re.sub(r"```(?:json)?|```", "", txt).strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
        return None


@dataclass
class SceneDossier:
    question: str
    facts: list = field(default_factory=list)        # [{"fact", "detail", "source", "date"}]
    actors_hint: list = field(default_factory=list)  # real actor names/segments surfaced by the evidence
    missing: list = field(default_factory=list)      # checklist items NO passage established
    checklist: list = field(default_factory=list)    # what a domain expert said was needed
    resolved: dict = None                            # {"answer","evidence","source"} if already decided
    standing: str = ""                               # rendered deciding-signal string (for prompts)
    standing_struct: dict = None                     # {favored, margin, basis, confidence} — the DIRECTIONAL
    #                                                  deciding signal (Rank-1 lever); None if not established
    n_passages: int = 0
    n_rounds: int = 1                                # retrieval rounds run (deepening)
    coverage: float = 0.0                            # grounded checklist items / all checklist items
    abstain: bool = False
    abstain_reason: str = ""

    @property
    def standing_directional(self) -> bool:
        """True iff the evidence established a DIRECTIONAL favorite with real confidence — the signal that
        decides direction. A blank or hedged standing means the forecasters would revert to a coin flip."""
        s = self.standing_struct or {}
        try:
            conf = float(s.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        return bool(s.get("favored")) and str(s.get("favored")).lower() not in ("", "unknown", "unclear",
                                                                                "toss-up", "tossup", "even") \
            and conf >= 0.35

    def brief(self, max_facts=16) -> str:
        """The grounded scene as prompt context — every line cited, the DIRECTIONAL STANDING first."""
        lines = []
        if self.standing:
            lines.append(f"- CURRENT STANDING (the deciding signal): {self.standing}")
        lines += [f"- {f['fact']}: {f.get('detail', '')}  [{f.get('source', '?')}"
                  f"{', ' + f['date'] if f.get('date') else ''}]" for f in self.facts[:max_facts]]
        if self.missing:
            lines.append(f"- NOT ESTABLISHED (no evidence found): {'; '.join(self.missing[:6])}")
        return "\n".join(lines)

    def as_report(self) -> dict:
        return {"n_passages": self.n_passages, "coverage": round(self.coverage, 3),
                "abstain": self.abstain, "abstain_reason": self.abstain_reason,
                "resolved": self.resolved, "missing": self.missing,
                "detail": [{"fact": f["fact"], "grounded": True, "source": f.get("source"),
                            "date": f.get("date")} for f in self.facts] +
                          [{"fact": m, "grounded": False, "source": None} for m in self.missing]}


_PLAN_PROMPT = """A forecasting engine must ground this question in real current facts before simulating it.
QUESTION: {q}
TODAY: {today}

List (a) the DECIDING FACTS a domain expert would need (the facts the outcome actually turns on — who the
real actors are, current standings/polls/money/endorsements, key dates, the state of play), and (b) 3-6
targeted web-search queries to find them. Return ONLY JSON:
{{"checklist": ["<deciding fact needed>", ...], "queries": ["<search query>", ...]}}"""

_STANDING_QUERY_PROMPT = """A forecaster needs the DIRECTIONAL deciding signal for this question — who or
which outcome is currently FAVORED, and by how much (polls, odds, front-runner, incumbency, money).
QUESTION: {q}
TODAY: {today}
Give 3-4 web-search queries most likely to surface that favored-side signal as of the question date (e.g.
'<race> latest poll', '<race> odds favorite', '<subject> frontrunner', '<event> likely to pass').
Return ONLY JSON: {{"queries": ["...", ...]}}"""

_DISTILL_PROMPT = """You are grounding a forecasting question in evidence. Use ONLY the passages below —
if a checklist item is not established by a passage, put it in "missing"; do NOT fill it from memory.
QUESTION: {q}
TODAY: {today}
CHECKLIST (deciding facts needed): {checklist}

PASSAGES:
{passages}

Return ONLY JSON:
{{"facts": [{{"fact": "<checklist item or other key fact>", "detail": "<the grounded specifics>",
             "source": "<passage source>", "date": "<passage date if any>"}}],
  "standing": {{"favored": "<which OUTCOME the evidence favors — the front-runner's name, or YES / NO for a
               yes-no event; 'unclear' ONLY if the evidence genuinely gives no directional signal>",
               "margin": "<how big the edge is: the poll lead, cash ratio, incumbency, base-rate strength —
               specific (e.g. 'leads only poll 52-38, 4:1 cash, incumbent')>",
               "basis": "<the cited evidence for it>",
               "confidence": <0..1 how strongly the evidence points to 'favored'; a clear front-runner is
               0.8+, a genuine toss-up 0.3, no signal 0.0>}},
  "actors": ["<real named actors or concrete population segments this question turns on>"],
  "missing": ["<checklist items no passage established>"],
  "resolved": <null, or {{"answer": "<the outcome>", "evidence": "<passage text>", "source": "<source>"}}
              if the passages show the question's outcome has ALREADY been decided/announced>}}"""


@dataclass
class SceneGrounder:
    """question → SceneDossier via checklist-planned retrieval + evidence-only distillation."""
    llm: object                                  # callable(prompt) -> text (DeepSeek/Anthropic/...)
    search_fn: object = None                     # callable(queries:list, k_each) -> [Passage]; default stack
    min_coverage: float = 0.4                    # below this, the deciding facts are not grounded → abstain
    min_passages: int = 3
    today: str = ""

    def ground(self, question: str, evidence=None) -> SceneDossier:
        """`evidence` (list of Passage|str), when given, REPLACES live retrieval — the leak-free /
        caller-supplied path (as-of backtests feed the frozen context here; nothing is fetched from the
        live web, so a resolved-in-the-past question cannot leak its answer through today's news)."""
        today = self.today or __import__("time").strftime("%Y-%m-%d")
        plan = parse_json(self.llm(_PLAN_PROMPT.format(q=question, today=today))) or {}
        checklist = [str(c) for c in plan.get("checklist", [])][:10]
        queries = [str(q) for q in plan.get("queries", [])][:6] or [question]

        if evidence is not None:
            from swm.engine.retrieval import Passage
            passages = [e if isinstance(e, Passage) else Passage(str(e), "provided") for e in evidence]
        else:
            passages = (self.search_fn or multi_search)(queries, 8)

        d = SceneDossier(question=question, checklist=checklist, n_passages=len(passages))
        # The passage-COUNT floor is a LIVE-RETRIEVAL starvation signal (a dead search returns nothing).
        # It does not apply to caller-injected evidence — that path is gated on CONTENT (coverage) below,
        # since the caller vouched for the passages and a curated as-of context is legitimately compact.
        if evidence is None and len(passages) < self.min_passages:
            d.abstain = True
            d.missing = checklist
            d.abstain_reason = (f"GROUNDING STARVED: retrieval returned {len(passages)} passages for "
                                f"{len(queries)} queries — cannot establish the deciding facts. "
                                f"Refusing to simulate on the LLM's imagination.")
            return d

        self._distill(d, question, today, passages, checklist)

        # ROUND 2 (Rank-1 deepening): if the DIRECTIONAL deciding signal is still missing, run a targeted
        # 'who is favored' retrieval round and re-distill on the merged evidence. This is the highest-leverage
        # fix — without a directional standing every forecaster reverts to base rate / a coin flip. Only when
        # we can actually retrieve more (search_fn/live or as-of search_fn), never for fixed injected evidence.
        can_retrieve = evidence is None or callable(self.search_fn)
        if can_retrieve and d.resolved is None and not d.standing_directional:
            tq = parse_json(self.llm(_STANDING_QUERY_PROMPT.format(q=question, today=today))) or {}
            follow = [str(x) for x in tq.get("queries", [])][:4]
            if follow:
                more = (self.search_fn or multi_search)(follow, 8)
                seen = {p.text[:80].lower() for p in passages}
                passages = passages + [p for p in more if p.text[:80].lower() not in seen]
                d.n_passages = len(passages)
                d.n_rounds = 2
                self._distill(d, question, today, passages, checklist)

        n_check = max(1, len(checklist))
        established = sum(1 for c in checklist
                          if any(c.lower()[:24] in (f["fact"] + " " + f.get("detail", "")).lower()
                                 for f in d.facts))
        d.coverage = max(established, min(len(d.facts), n_check) - len(d.missing) * 0) / n_check
        d.coverage = min(1.0, max(d.coverage, (n_check - len(d.missing)) / n_check if d.missing else
                                  min(1.0, len(d.facts) / n_check)))
        if d.coverage < self.min_coverage and d.resolved is None:
            d.abstain = True
            d.abstain_reason = (f"DECIDING FACTS NOT GROUNDED (coverage {d.coverage:.0%}; missing: "
                                f"{'; '.join(d.missing[:4]) or 'most of the checklist'}). A simulation run "
                                f"on ungrounded facts would be theater — abstaining.")
        return d

    def _distill(self, d, question, today, passages, checklist):
        """Distill passages → facts + STRUCTURED directional standing (mutates d)."""
        ptxt = "\n".join(p.cite() for p in passages[:40])
        dist = parse_json(self.llm(_DISTILL_PROMPT.format(
            q=question, today=today, checklist=json.dumps(checklist), passages=ptxt))) or {}
        d.facts = [f for f in dist.get("facts", []) if isinstance(f, dict) and f.get("fact")]
        d.actors_hint = [str(a) for a in dist.get("actors", [])][:12]
        d.missing = [str(m) for m in dist.get("missing", [])]
        st = dist.get("standing")
        if isinstance(st, dict) and st.get("favored"):
            d.standing_struct = st
            d.standing = (f"{st.get('favored')} favored ({st.get('margin', '')}); {st.get('basis', '')} "
                          f"[confidence {st.get('confidence', '?')}]")[:400]
        elif isinstance(st, str) and st.strip():           # tolerate a free-text standing
            d.standing = st[:400]
            d.standing_struct = None
        r = dist.get("resolved")
        d.resolved = r if isinstance(r, dict) and r.get("answer") else None
