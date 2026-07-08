"""SelectingCompiler — the candidate-and-verify loop (the compiler's missing keystone).

EXP-068's `ValidatingCompiler` fixes ONE compiled spec's numeric bugs (validate → repair). But it never asks
the deeper question: *is this the right STRUCTURE at all?* A single LLM call can pick the wrong mechanism
(a deliberation for a competition — the NBA bug) and no amount of equation-repair fixes a wrong mechanism.
This layer closes that gap by applying the project's founding discipline — "believable ≠ accurate; build the
evaluator that embarrasses the model before the model does" — to the COMPILER itself:

  1. PROPOSE several candidate structural models (diverse mechanisms/framings), each validated + repaired.
  2. SCORE each candidate on structural quality WITHOUT the answer — validity (it simulates cleanly, no
     degeneracy: reuses `spec_validator.validate`) × an optional LLM critic (is this the right mechanism and
     a sane structure for this question?). A broken candidate cannot win on critic charm (validity gates).
  3. SELECT the best, and report cross-candidate AGREEMENT — if the candidates converge (same mechanism,
     close forecasts) we are confident in the structure; if they diverge, that is surfaced honestly rather
     than hidden behind one confident number.

`SelectingCompiler.compile(...)` returns a `SelectedModel` (a drop-in `CompiledModel` carrying the
verification report), so `WorldModel(compiler=SelectingCompiler(...))` gets self-selecting compilation for
free. Backends are pluggable exactly like the rest of the system: `cached_critic_fn` (dev/test),
`anthropic_critic_fn` (prod); with no critic the loop still runs on validity + agreement alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.api.compiler import CompiledModel
from swm.api.spec_validator import ValidatingCompiler, validate


# ============================ scoring a single candidate ============================
def validity_score(issues) -> float:
    """1.0 = clean; errors hurt hard (a degenerate/mis-structured spec must not win), warns mildly."""
    errs = sum(1 for i in issues if i.severity == "error")
    warns = sum(1 for i in issues if i.severity == "warn")
    return max(0.0, 1.0 - 0.5 * errs - 0.1 * warns)


def score_candidate(spec, question, *, critic_fn=None, issues=None, n=1500) -> tuple:
    """Score a candidate spec in [0,1] WITHOUT the outcome. `validity` gates (a broken spec scores ~0 no
    matter how plausible it reads); an optional critic ranks among the valid ones. Returns (score, detail)."""
    issues = validate(spec, n=n) if issues is None else issues
    vscore = validity_score(issues)
    detail = {"validity": round(vscore, 3),
              "issues": [i.as_dict() for i in issues if i.severity == "error"][:6],
              "mechanism": spec.mechanism}
    critic = None
    if critic_fn is not None:
        try:
            c = _coerce_critic(critic_fn(build_critic_prompt(question, spec)))
            critic = c["score"]
            detail["critic"] = {"score": round(critic, 3), "critique": c.get("critique", "")[:200]}
        except Exception as e:                       # a critic failure must never crash selection
            detail["critic_error"] = str(e)[:80]
    score = vscore if critic is None else vscore * (0.4 + 0.6 * critic)
    return score, detail


def build_critic_prompt(question: str, spec) -> str:
    from swm.api.spec_validator import _spec_to_json
    return ("You are a STRUCTURAL-MODEL CRITIC for a social world model. Judge whether this compiled "
            "simulation matches the REAL generative process of the question — the right MECHANISM "
            "(competition vs deliberation vs population vs single-person vs coupled SCM), the right drivers, "
            "and sanely calibrated timescales. Do NOT answer the question; judge the MODEL.\n\n"
            f"QUESTION: {question}\n\nSPEC:\n{json.dumps(_spec_to_json(spec), indent=1)}\n\n"
            'Return ONLY JSON: {"score": <0..1, higher = better-matched structure>, "critique": "<one line>"}.')


def _coerce_critic(raw) -> dict:
    obj = raw if isinstance(raw, dict) else json.loads(str(raw)[str(raw).find("{"):str(raw).rfind("}") + 1])
    return {"score": max(0.0, min(1.0, float(obj.get("score", 0.5)))), "critique": str(obj.get("critique", ""))}


def cached_critic_fn(cache: dict):
    """Replay committed critic verdicts keyed by a stable id; raises on miss so a run never silently guesses."""
    def fn(key):
        if key not in cache:
            raise KeyError(key)
        return cache[key]
    return fn


def anthropic_critic_fn(api_key: str, model: str = "claude-sonnet-5", max_tokens: int = 400):
    """PRODUCTION critic backend (pure-stdlib urllib), same shape as anthropic_compile_fn."""
    import urllib.request

    def fn(prompt):
        body = json.dumps({"model": model, "max_tokens": max_tokens,
                           "system": "You critique compiled structural models for mechanism-fit. Emit ONLY JSON.",
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                     headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return fn


# ============================ cross-candidate agreement ============================
def _point(forecast: dict):
    for k in ("p_event", "p_target", "mean", "mean_vote_share", "mean_share", "p_respond_mean"):
        if forecast.get(k) is not None:
            return forecast[k]
    return None


def agreement(models, *, n=1500) -> dict:
    """How much the candidates converge — the honest confidence-in-structure signal. Reports the mechanism
    vote and (where comparable) the spread of the candidates' point forecasts. Divergence is surfaced, not
    hidden: candidates that pick different mechanisms, or forecasts that spread widely, mean low confidence."""
    from collections import Counter
    mechs = Counter(m.spec.mechanism for m in models)
    points = []
    for m in models:
        try:
            points.append(_point(m.run(n=n)))
        except Exception:
            points.append(None)
    pts = [p for p in points if p is not None]
    spread = (max(pts) - min(pts)) if len(pts) >= 2 else 0.0
    mech_agree = mechs.most_common(1)[0][1] / len(models) if models else 1.0
    point_agree = max(0.0, 1.0 - spread) if pts else None                # spread on a [0,1] scale
    return {"mechanism_vote": dict(mechs), "mechanism_agreement": round(mech_agree, 3),
            "point_spread": round(spread, 4), "point_agreement": round(point_agree, 3) if point_agree is not None else None,
            "points": [round(p, 4) if p is not None else None for p in points]}


# ============================ the selecting compiler ============================
@dataclass
class SelectedModel(CompiledModel):
    """The chosen compiled model — a drop-in CompiledModel carrying the verification report (candidate scores,
    the winner's validity/critic detail, and cross-candidate agreement)."""
    verification: dict = None
    candidates: list = field(default_factory=list)


@dataclass
class Candidate:
    model: CompiledModel
    score: float
    detail: dict
    validation: dict

    def brief(self):
        return {"mechanism": self.model.spec.mechanism, "score": round(self.score, 4),
                "validity": self.detail.get("validity"),
                "critic": self.detail.get("critic", {}).get("score") if "critic" in self.detail else None,
                "n_errors": len(self.detail.get("issues", []))}


@dataclass
class SelectingCompiler:
    """Wrap a base compiler with propose-k → validate/repair each → score → select, + an agreement report.
    Pass explicit `keys` (one per candidate) for deterministic/cached runs; otherwise the context is
    diversified per candidate to induce alternative mechanisms from the LLM."""
    compiler: object
    critic_fn: object = None
    repair_fn: object = None
    k: int = 3
    keys: list = None
    score_n: int = 1500

    def _validating(self):
        return ValidatingCompiler(self.compiler, repair_fn=self.repair_fn)

    def _candidate_inputs(self, context, key):
        if self.keys:                                            # explicit per-candidate keys (cached/testable)
            return [(context, k) for k in self.keys]
        out = [(context, key)]                                   # candidate 0: the obvious framing
        for i in range(1, self.k):                               # 1..k-1: nudge toward an alternative mechanism
            alt = context + (f"\n\n[ALTERNATIVE {i}: if a DIFFERENT mechanism or framing fits the real "
                             "process better than the obvious one, use it instead.]")
            out.append((alt, key))
        return out

    def compile(self, question: str, context: str = "", *, key: str = None) -> SelectedModel:
        vc = self._validating()
        cands = []
        for ctx, ck in self._candidate_inputs(context, key):
            compiled = vc.compile(question, ctx, key=ck)
            report = dict(vc.last_report or {})
            score, detail = score_candidate(compiled.spec, question, critic_fn=self.critic_fn, n=self.score_n)
            cands.append(Candidate(compiled, score, detail, report))
        cands.sort(key=lambda c: -c.score)
        best = cands[0]
        agr = agreement([c.model for c in cands], n=self.score_n)
        sm = SelectedModel(best.model.spec,
                           verification={"selected_score": round(best.score, 4), "n_candidates": len(cands),
                                         "selected_detail": best.detail, "candidates": [c.brief() for c in cands],
                                         "agreement": agr},
                           candidates=cands)
        return sm
