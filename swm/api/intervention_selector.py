"""Semantic intervention selector — pick the causally-better action with an LLM (SIMULATION_AUDIT KPI-A).

EXP-054 built the interventional KPI (choose a headline = a real do(x) on randomized A/B data) and showed
lexical features capture only ~9.5% of the achievable uplift and rank arms at chance — the frontier is
semantic, exactly like EXP-044/047. This is the semantic model: an LLM judge reads the candidate actions
(headlines) for a goal (maximize clicks) and ranks them, the same pluggable-backend pattern as
`semantic_stance` (production = Anthropic API; here = committed judgments replayed).

The judge is scored on the causal scoreboard from EXP-054 — policy value / regret + CATE-sign — never on
its own confidence, and blind to the realized CTRs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


def build_selection_prompt(goal: str, options: list) -> str:
    lines = [f"GOAL: {goal}", "Candidate options (headlines) — rank by how well each achieves the goal:"]
    for i, o in enumerate(options):
        lines.append(f"  [{i}] {o}")
    lines.append('Return ONLY compact JSON: {"best": <index>, "scores": [<score per option, higher=better>]}. '
                 "Judge only from the text; do not invent outcomes you cannot know.")
    return "\n".join(lines)


def _coerce(raw, n):
    obj = raw if isinstance(raw, dict) else json.loads(str(raw)[str(raw).find("{"):str(raw).rfind("}") + 1])
    best = int(obj.get("best", 0)) if obj else 0
    scores = obj.get("scores") if isinstance(obj, dict) else None
    if not (isinstance(scores, list) and len(scores) == n):
        scores = [1.0 if i == best else 0.0 for i in range(n)]
    return {"best": max(0, min(n - 1, best)), "scores": [float(s) for s in scores]}


@dataclass
class InterventionSelector:
    """Ranks candidate interventions with an LLM judge. `judge_fn(prompt) -> {best, scores}` (or raw JSON)."""
    judge_fn: object

    def select(self, goal: str, options: list) -> dict:
        if not options:
            return {"best": None, "scores": []}
        return _coerce(self.judge_fn(build_selection_prompt(goal, options)), len(options))


def cached_selector(cache: dict):
    """Replay committed selections keyed by a stable id; raises on a miss so a run never silently guesses."""
    def fn(key):
        if key not in cache:
            raise KeyError(key)
        return cache[key]
    return fn
