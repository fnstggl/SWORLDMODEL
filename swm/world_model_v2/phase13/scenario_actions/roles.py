"""Separated LLM roles with a complete trace — generation never certifies itself.

Every call records: stage, role, prompt, response, parsed result, acceptance, reasons,
candidate ancestry, call index, and model label. Comparative roles (the adjudicator, the
omission critic when comparing) receive BLIND labels — candidate provenance and generator
identity never enter their prompts. Critics produce typed findings that map to structural
gates or surfaced flags; no critic can eliminate a candidate merely by disliking it, and no
critic can select the final action — simulation evidence does that.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

ROLES = ("goal_backward_strategist", "forward_affordance_discoverer",
         "orthogonal_strategy_generator", "adversarial_omission_critic",
         "feasibility_authority_critic", "mechanism_critic", "domain_reality_critic",
         "goal_gaming_critic", "implementation_critic", "final_adjudicator")


@dataclass
class RoleTrace:
    """Append-only record of every role call; optionally mirrored to a JSONL file."""
    path: str = ""
    rows: list = field(default_factory=list)
    model_label: str = ""

    def record(self, *, stage: str, role: str, prompt: str, response: str, parsed=None,
               accepted: bool = True, reasons: str = "", ancestry: str = ""):
        row = {"call": len(self.rows) + 1, "stage": stage, "role": role,
               "prompt": str(prompt), "response": str(response),
               "parsed": parsed if isinstance(parsed, (dict, list, str, int, float, bool,
                                                       type(None))) else str(parsed)[:500],
               "accepted": bool(accepted), "reasons": str(reasons)[:300],
               "ancestry": str(ancestry)[:200], "model": self.model_label,
               "token_estimate": (len(str(prompt)) + len(str(response))) // 4}
        self.rows.append(row)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def n_calls(self) -> int:
        return len(self.rows)

    def by_role(self) -> dict:
        out: dict = {}
        for r in self.rows:
            out[r["role"]] = out.get(r["role"], 0) + 1
        return out


class RoleRunner:
    """One seat per role; a shared budget; every call traced. `chat` is the raw backend."""

    def __init__(self, chat=None, *, trace: RoleTrace = None, max_calls: int = 200):
        self.chat = chat
        self.trace = trace or RoleTrace()
        self.max_calls = max_calls
        self.calls = 0

    def available(self) -> bool:
        return self.chat is not None and self.calls < self.max_calls

    def ask(self, role: str, stage: str, prompt: str, *, ancestry: str = "",
            expect: str = "json"):
        """Returns (parsed_or_text, ok). Failures are traced and returned as (None, False) —
        callers degrade loudly, never silently."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} (valid: {ROLES})")
        if not self.available():
            self.trace.record(stage=stage, role=role, prompt="<budget exhausted>",
                              response="", parsed=None, accepted=False,
                              reasons="llm budget exhausted or no backend",
                              ancestry=ancestry)
            return None, False
        self.calls += 1
        try:
            raw = self.chat(prompt)
        except Exception as e:  # noqa: BLE001
            self.trace.record(stage=stage, role=role, prompt=prompt,
                              response=f"<error {type(e).__name__}>", parsed=None,
                              accepted=False, reasons=f"llm error {type(e).__name__}",
                              ancestry=ancestry)
            return None, False
        if expect == "text":
            self.trace.record(stage=stage, role=role, prompt=prompt, response=str(raw)[:4000],
                              parsed=None, accepted=True, ancestry=ancestry)
            return str(raw), True
        from swm.engine.grounding import parse_json
        parsed = parse_json(raw)
        ok = isinstance(parsed, (dict, list))
        self.trace.record(stage=stage, role=role, prompt=prompt, response=str(raw)[:4000],
                          parsed=parsed if ok else None, accepted=ok,
                          reasons="" if ok else "unparseable json", ancestry=ancestry)
        return (parsed if ok else None), ok


def blind_labels(candidates: list, seed: int = 0) -> tuple:
    """Shuffled anonymous labels for comparative roles: returns (ordered list of
    (label, candidate), label->candidate_id map kept OUTSIDE any prompt)."""
    rng = random.Random(seed)
    order = list(candidates)
    rng.shuffle(order)
    labeled = [(f"OPTION_{chr(65 + i)}", c) for i, c in enumerate(order)]
    return labeled, {lab: c.candidate_id for lab, c in labeled}


def blind_candidate_view(candidate) -> dict:
    """What a comparative critic may see: the intervention itself — never source, generator,
    ancestry, or revision history."""
    return {"title": candidate.title,
            "causal_theory": candidate.causal_theory,
            "steps": [{"intent": s.intent, "targets": list(s.target_ids),
                       "channel": s.channel, "content": s.exact_content[:400],
                       "terms": s.terms, "timing_ts": s.timing_ts,
                       "conditions": [getattr(c, "description", "") for c in s.conditions],
                       "visibility": s.visibility}
                      for s in candidate.steps],
            "stop_conditions": [getattr(c, "description", "")
                                for c in candidate.stop_conditions],
            "assumptions": candidate.assumptions[:6]}
