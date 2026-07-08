"""The central benchmark (spec Phase 5 / EXP-009): raw LLM vs the world model, same items, same metrics.

Compares up to six tiers on ONE dataset:

  1. raw_llm              : LLM, input only                          [pluggable predictor]
  2. raw_llm_context      : LLM + as-of retrieved context           [pluggable predictor]
  3. structured          : a structured statistical model (segment / logistic) — no LLM
  4. calibrated          : the current calibrated system (structured + Platt / grade)
  5. aggregate_world     : explicit AGGREGATE state-transition model
  6. individual_world    : explicit INDIVIDUAL state-transition model (where entity data exists)

Tiers 3/5/6 run live and need no API key. Tiers 1/2/4 accept a `predictor` callable or a table of
precomputed predictions (so an agent swarm — the only LLM available here — can supply them without
this module importing an SDK). Every tier is scored with the same proper-scoring + calibration +
decision metrics, and the verdict states plainly whether the world model beats raw LLM + context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k


def score_tier(y: list[int], p: list[float]) -> dict:
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4),
            "uplift@10": round(uplift_at_k(y, p, 0.1), 4),
            "uplift@20": round(uplift_at_k(y, p, 0.2), 4)}


@dataclass
class BenchmarkResult:
    target: str
    n: int
    base_rate: float
    tiers: dict[str, dict] = field(default_factory=dict)
    verdict: str = ""

    def to_dict(self) -> dict:
        return {"target": self.target, "n": self.n, "base_rate": round(self.base_rate, 4),
                "tiers": self.tiers, "verdict": self.verdict}


def run_benchmark(y: list[int], tier_predictions: dict[str, list[float] | None], *,
                  target: str = "outcome") -> BenchmarkResult:
    """tier_predictions: {tier_name -> list-of-p aligned with y, or None if blocked}.
    Scores every present tier and writes the world-model-vs-raw-LLM verdict."""
    n = len(y)
    res = BenchmarkResult(target=target, n=n, base_rate=sum(y) / n)
    for name, preds in tier_predictions.items():
        if preds is None:
            res.tiers[name] = {"status": "BLOCKED (no predictions supplied)"}
        elif len(preds) != n:
            res.tiers[name] = {"status": f"SKIP (len {len(preds)} != n {n})"}
        else:
            res.tiers[name] = score_tier(y, preds)

    def ll(name):
        t = res.tiers.get(name, {})
        return t.get("log_loss") if isinstance(t, dict) else None

    # verdict: does an explicit state model beat raw LLM + context on log loss?
    world = min([x for x in (ll("aggregate_world"), ll("individual_world")) if x is not None],
                default=None)
    rawc = ll("raw_llm_context")
    raw = ll("raw_llm")
    parts = []
    if world is not None and rawc is not None:
        parts.append("world model BEATS raw LLM + context on log loss"
                     if world < rawc else
                     "world model does NOT beat raw LLM + context on log loss")
    if world is not None and raw is not None:
        parts.append(f"vs raw LLM (input-only): world {'better' if world < raw else 'worse'}")
    if rawc is not None and raw is not None:
        parts.append(f"retrieval {'helped' if rawc < raw else 'did not help'} the raw LLM")
    res.verdict = "; ".join(parts) if parts else "insufficient tiers to render a verdict"
    return res
