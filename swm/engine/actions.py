"""Best action — intervention search over the SAME engine. The output is the actual artifact, ranked.

"What's the best landing-page headline to maximize AirPods Max sales" is answered by DOING it in
simulation: generate REAL candidate headlines grounded in the product facts, put each one in front of the
same cast of audience personas (drawn once — paired comparison, so persona luck can't pick the winner),
have each persona reason and DECIDE (engage or ignore, at temperature, R times), and rank the artifacts
by simulated engagement. The answer is ranked actual texts — never a scalar about "headline_clarity=7.5"
of a headline that was never written.

The same loop is the general do-operator: any candidate action (an endorsement, a price change, a
different email) is an artifact; evaluate ranked candidates under identical personas and report the
CONTRAST, which is more trustworthy than any absolute level (shared persona bias cancels in the ranking).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from swm.engine.agents import decide, draw_variants, slice_private_facts
from swm.engine.grounding import parse_json

_GEN_PROMPT = """Generate {k} genuinely DIFFERENT candidate {kind} for this goal — different angles
(feature-led, emotion-led, price/value, social-proof, minimalist), not paraphrases. Ground every claim in
the evidence; invent no specs.
GOAL: {q}
GROUNDED FACTS:
{scene}
Return ONLY JSON: {{"candidates": [{{"text": "<the actual {kind_singular}>", "angle": "<one word>"}}]}}"""


def generate_artifacts(llm, question, scene_brief, *, k=5, kind="headlines") -> list:
    raw = parse_json(llm(_GEN_PROMPT.format(k=k, kind=kind, q=question, scene=scene_brief,
                                            kind_singular=kind.rstrip("s")))) or {}
    out = []
    for c in raw.get("candidates", []) or []:
        if isinstance(c, dict) and c.get("text"):
            out.append({"text": str(c["text"])[:300], "angle": str(c.get("angle", ""))[:40]})
    return out[:k]


@dataclass
class ArtifactOptimizer:
    llm_hot: object
    llm: object = None
    reps: int = 2                          # temperature resamples per persona per artifact
    max_workers: int = 8

    def run(self, question, cast, dossier, *, artifacts=None, k=5, kind="headlines",
            today="") -> dict:
        llm_cold = self.llm or self.llm_hot
        artifacts = artifacts or generate_artifacts(llm_cold, question, dossier.brief(), k=k, kind=kind)
        if not artifacts:
            return {"ranked": [], "error": "artifact generation failed"}

        # ONE shared cast of audience personas — paired comparison across all artifacts
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            drawn = list(ex.map(lambda a: draw_variants(llm_cold, a, question, dossier.brief()),
                                cast.actors))
        personas = [p for group in drawn for p in group]
        for i, p in enumerate(personas):
            p.private_facts = slice_private_facts(dossier.facts, i)

        options = ["engage", "ignore"]
        jobs = [(ai, p) for ai in range(len(artifacts)) for p in personas for _ in range(self.reps)]

        def one(job):
            ai, p = job
            stim = (f"{question}\nYou encounter this {kind.rstrip('s')}:\n\"{artifacts[ai]['text']}\"\n"
                    f"'engage' means it works on you (you click / read on / consider buying); "
                    f"'ignore' means it doesn't.")
            return decide(self.llm_hot, p, stim, options, date=today,
                          public="(you are just browsing; no one is watching)")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(one, jobs))

        ranked = []
        for ai, art in enumerate(artifacts):
            got = [(p, r) for (aj, p), r in zip(jobs, results) if aj == ai and r is not None]
            if not got:
                continue
            wz = sum(p.weight for p, _ in got) or 1.0
            p_eng = sum(p.weight * r["probs"].get("engage", 0.0) for p, r in got) / wz
            why = [r["why"] for _, r in got if r.get("why")][:3]
            ranked.append({**art, "p_engage": round(p_eng, 4), "n_reads": len(got), "reasons": why})
        ranked.sort(key=lambda a: -a["p_engage"])
        return {"ranked": ranked, "n_personas": len(personas),
                "n_calls": len(jobs) + len(cast.actors),
                "note": ("paired comparison under identical grounded personas — trust the RANKING and the "
                         "contrast between candidates; the absolute engagement level is ungraded until "
                         "this question-class is backtested on real CTR outcomes")}
