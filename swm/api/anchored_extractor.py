"""AnchoredExtractor — make the DEFAULT value-grounding a full three-pillar estimate (EXP-086, wired in).

The universal `CalibratedExtractor` already does Pillar 1 (extract a value from as-of WEB EVIDENCE) and the
CI-calibration half of Pillar 3 (a trustworthy sd). EXP-086 measured the two pillars it was missing — and
showed they are what turn a *guess* into ~87% of a *measurement*:

  Pillar 2 — REFERENCE-CLASS ANCHOR (the outside view). Estimate the variable's TYPICAL / base-rate value for
             a case like this, WITHOUT the specific evidence — the reference class. The number starts here.
  Pillar 3 — ENSEMBLE + SHRINK. Sample the evidence extractor (spread = extra uncertainty), then shrink the
             evidence-based "inside view" toward the reference-class "outside view" by its CALIBRATED
             uncertainty (`grounded_inference.shrink`). This is empirical Bayes: strong, precise web evidence
             (small sd) barely moves; a vague estimate (large sd) is pulled to the base rate — the antidote to
             the LLM's over-individuation, and SELF-REGULATING (it can only help a confident value, never
             override it).

This wraps any `CalibratedExtractor` and exposes the same `extract(variable, question, evidence) -> {value,
sd}` seam, so it drops into `RetrievalGrounder` unchanged and composes with the web search + structured feeds.
`build_retrieval_grounder(..., anchor=True)` (the default) makes every general-question grounding run this.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from swm.api.retrieval_grounding import CalibratedExtractor, parse_json_lenient
from swm.variables.grounded_inference import shrink


@dataclass
class AnchoredExtractor:
    """Pillars 2+3 over a `CalibratedExtractor` (Pillar 1). `ensemble_k` samples of the evidence extractor
    (>1 only helps a stochastic LLM); `anchor` toggles the reference-class shrink (off = raw evidence, for
    leakage-free backtests that isolate a single pillar)."""
    base: CalibratedExtractor
    ensemble_k: int = 1
    anchor: bool = True

    def _reference(self, variable, question):
        """Pillar 2: the OUTSIDE VIEW — the reference-class base rate, WITHOUT the specific evidence."""
        prompt = (f"Question context: {question or ''}\n"
                  f'What is the TYPICAL, base-rate current value of "{variable}" for a case like this — the '
                  f'reference-class average, ignoring specifics you cannot verify? Answer from population '
                  f'knowledge, not this specific case.\n'
                  f'Return JSON: {{"value": <number>, "ci95": <95% half-width>, "confidence": <0..1>}}.')
        r = parse_json_lenient(self.base.llm(prompt))
        if r is None or r.get("value") is None:
            return None
        val = float(r["value"])
        ci = r.get("ci95")
        if ci is None or float(ci) <= 0:
            conf = float(r.get("confidence", 0.4))
            ci = self.base.base_ci_frac * (abs(val) + 1e-6) * (1.0 - conf) + 1e-6
        return val, max(float(ci) / 1.96, 1e-6)

    def extract(self, variable, question, evidence):
        samples = []
        for _ in range(max(1, self.ensemble_k)):
            s = self.base.extract(variable, question, evidence)
            if s is not None:
                samples.append(s)
        if not samples:                                   # no evidence-based value -> fall back to the outside view
            ref = self._reference(variable, question) if self.anchor else None
            return {"value": ref[0], "sd": ref[1]} if ref else None
        vals = [s["value"] for s in samples]
        ev_mean = fmean(vals)
        within = fmean([s["sd"] ** 2 for s in samples])
        between = pstdev(vals) ** 2 if len(vals) > 1 else 0.0
        ev_sd = (within + between) ** 0.5                 # hierarchical: within-sample + ensemble spread
        if not self.anchor:
            return {"value": ev_mean, "sd": ev_sd}
        ref = self._reference(variable, question)
        if ref is None:
            return {"value": ev_mean, "sd": ev_sd}
        post, post_sd = shrink(ev_mean, ev_sd, ref[0], ref[1])
        return {"value": post, "sd": post_sd}
