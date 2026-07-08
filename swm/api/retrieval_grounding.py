"""The universal, CALIBRATED RetrievalGrounder — grounds any text-derivable variable, in any domain, from
as-of evidence, with a confidence interval you can trust.

Structured sources (swm.api.grounding_sources) cover the domains with precise numeric series. Everything else
— the long tail of every domain — has to be grounded from unstructured evidence: retrieve as-of passages, then
have an LLM extract the numeric value. The danger is a confidently-wrong value: an extractor that reports
"4.3% ± 0.1" from a stale or misread passage is worse than an honest prior. So the extractor must return a
CALIBRATED interval — and we VALIDATE that calibration on a labeled set and widen it until its stated 90% CIs
actually cover truth 90% of the time. A grounded value we cannot trust the CI of is not grounding, it's guessing
with extra steps.

  - `WebRetriever(search_fn)`  — as-of retrieval; `search_fn(query, as_of)` returns evidence passages (a live
    web/search client in production, a fixture offline). As-of is enforced by the backend (no future passages).
  - `CalibratedExtractor(llm, ci_multiplier)` — prompts the LLM for {value, ci95, confidence}, converts to a
    value + sd, and WIDENS the sd by the fitted `ci_multiplier`.
  - `calibrate_extractor(...)` — fits that multiplier so empirical CI coverage matches nominal on labeled
    (variable, question, evidence, truth) examples. This is the honest core: the RetrievalGrounder's CIs are
    measured, not asserted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from swm.api.state_grounding import RetrievalGrounder

_Z = {0.9: 1.645, 0.95: 1.96, 0.8: 1.2816}


def parse_json_lenient(txt):
    """Parse an LLM's JSON reply robustly: accept a dict as-is, strip ```json fences, and fall back to the
    first {...} block. Real models wrap JSON in prose/fences; a brittle json.loads would drop good answers."""
    if isinstance(txt, dict):
        return txt
    if not isinstance(txt, str):
        return None
    s = re.sub(r"```(?:json)?|```", "", txt).strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None


@dataclass
class WebRetriever:
    """As-of retrieval. `search_fn(query, as_of) -> [passage, ...]`; the backend must not return evidence
    dated after `as_of` (leakage-free). Pluggable: a live search client in production, a fixture in tests."""
    search_fn: object
    k: int = 6

    def retrieve(self, query, as_of=None):
        res = self.search_fn(query, as_of) or []
        return list(res)[: self.k]


@dataclass
class CalibratedExtractor:
    """Extract a numeric value + a CALIBRATED CI from evidence. `llm(prompt) -> {value, ci95, confidence}`
    (dict or JSON string; value None if not derivable). The reported 95% half-width is converted to an sd and
    WIDENED by `ci_multiplier` (fit by `calibrate_extractor`). `base_ci_frac` supplies a fallback half-width
    (as a fraction of |value|) when the model omits ci95, scaled by (1 − confidence)."""
    llm: object
    ci_multiplier: float = 1.0
    base_ci_frac: float = 0.25
    evidence_only: bool = False           # True = leakage-free backtests (evidence only); False = present-grounding

    def _prompt(self, variable, question, evidence):
        ev = "\n".join(f"- {p}" for p in (evidence or [])[:6]) or "- (no evidence retrieved)"
        rule = ("Use ONLY the evidence below (as-of the question date); do NOT use any other knowledge. value "
                "null if the evidence does not determine it."
                if self.evidence_only else
                "Prefer the evidence below. If it does not contain the number, DO NOT return null — give your "
                "best current estimate from your own knowledge and LOWER the confidence (widen ci95) to reflect "
                "the added uncertainty. Return value null ONLY if the variable is not a measurable current "
                "quantity at all (e.g. a latent index with no real-world scale).")
        return (f"Question context: {question or ''}\n"
                f'Extract the CURRENT numeric value of this variable: "{variable}".\n{rule}\n{ev}\n\n'
                f'Return JSON: {{"value": <number or null>, "ci95": <95% half-width>, '
                f'"confidence": <0..1>}}.')

    def extract(self, variable, question, evidence):
        r = parse_json_lenient(self.llm(self._prompt(variable, question, evidence)))
        if r is None or r.get("value") is None:
            return None
        val = float(r["value"])
        ci = r.get("ci95")
        if ci is None or float(ci) <= 0:                 # missing/zero CI would fake perfect certainty
            conf = float(r.get("confidence", 0.5))
            ci = self.base_ci_frac * (abs(val) + 1e-6) * (1.0 - conf) + 1e-6
        sd = max(float(ci) / 1.96, 0.002 * (abs(val) + 1e-6)) * self.ci_multiplier   # floor: never 0
        return {"value": val, "sd": sd}


def calibrate_extractor(extractor: CalibratedExtractor, labeled,
                        *, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0), nominal=0.9) -> dict:
    """Fit `ci_multiplier` so the extractor's `nominal` CIs actually cover truth at that rate on labeled
    (variable, question, evidence, truth) examples. Returns the chosen multiplier + before/after coverage —
    the RetrievalGrounder's honesty check. Mutates `extractor.ci_multiplier` to the fitted value."""
    z = _Z.get(nominal, 1.645)
    base = extractor.ci_multiplier
    rows = []
    for ex in labeled:
        r = CalibratedExtractor(extractor.llm, ci_multiplier=1.0, base_ci_frac=extractor.base_ci_frac).extract(
            ex["variable"], ex.get("question"), ex.get("evidence"))
        if r is not None:
            rows.append((r["value"], r["sd"], float(ex["truth"])))

    def coverage(mult):
        if not rows:
            return 0.0
        return sum(1 for v, sd, t in rows if abs(v - t) <= z * sd * mult) / len(rows)

    before = coverage(base)
    best_m = min(grid, key=lambda m: abs(coverage(m) - nominal))
    extractor.ci_multiplier = best_m
    return {"n_labeled": len(rows), "ci_multiplier": best_m,
            "coverage_before": round(before, 3), "coverage_after": round(coverage(best_m), 3),
            "nominal": nominal}


def build_retrieval_grounder(search_fn, llm, *, ci_multiplier=1.0, k=6, name="retrieval",
                             anchor=True, ensemble_k=1) -> RetrievalGrounder:
    """Assemble a universal RetrievalGrounder: as-of retrieval + a calibrated LLM value-extractor. Plugs in as
    the `GroundingRouter` fallback (the long-tail, any-domain grounder).

    `anchor=True` (the DEFAULT) wraps the extractor in the three-pillar `AnchoredExtractor` (EXP-086): the
    evidence-based value is shrunk toward a reference-class base rate by its calibrated uncertainty, so weak
    grounding degrades gracefully to the outside view instead of a confident guess — and strong, precise web
    evidence is left essentially untouched (the shrink self-regulates by sd). `anchor=False` gives the raw
    evidence extractor (leakage-free backtests isolating a single pillar)."""
    ext = CalibratedExtractor(llm, ci_multiplier=ci_multiplier)
    if anchor:
        from swm.api.anchored_extractor import AnchoredExtractor
        extract_fn = AnchoredExtractor(ext, ensemble_k=ensemble_k, anchor=True).extract
    else:
        extract_fn = ext.extract
    return RetrievalGrounder(WebRetriever(search_fn, k=k), extract_fn, name=name)
