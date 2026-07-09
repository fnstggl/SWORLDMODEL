"""ObserverPanel — a panel of diverse informed forecasters for a binary event, base-rate-anchored.

The voter-segment society is the right model for "who do people CHOOSE" (an election's native distribution).
But most market questions are "will EVENT X happen" — and there the accurate, literature-backed engine
(Halawi et al 2024 "Approaching Human-Level Forecasting"; Schoenegger LLM crowds; Tetlock superforecasting)
is: a diverse ensemble of informed forecasters who each READ the grounded as-of evidence, state the
reference-class BASE RATE, then ADJUST for the specific standing — aggregated in log-odds.

This directly targets our two failure modes, both of which are base-rate failures:
  - "p=0.57 on a party the market priced at 0.01" — an observer who starts from "fringe parties almost never
    win" (base rate ~2%) and adjusts only on real evidence cannot land at 0.57.
  - "p=0.54 on a clear favorite (market 0.90)" — an observer who sees "leads the only poll 52-38, 4:1 cash"
    ADJUSTS UP hard from the base rate instead of hovering at a coin flip.

Diversity is over REASONING LENSES (base-rate / insider / skeptic / momentum / market-aware), not just
demographics — that is what decorrelates errors in a monoculture of one base model. Each forecaster is still
an agent reasoning from grounded context (the engine's philosophy); it just reasons as a forecaster, not a
voter. Aggregation: weighted log-linear pool (calibrate.pool_distribution) → out-of-sample temperature.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from swm.engine.calibrate import pool_distribution
from swm.engine.grounding import parse_json

# distinct forecasting LENSES — each a genuinely different prior/process, to decorrelate a single model's errors
LENSES = [
    ("outside_view", "a base-rate purist: you start from the reference-class frequency for THIS kind of event "
     "and move slowly; you distrust narratives and vivid recent news."),
    ("insider", "a domain insider who weighs the specific current standing heavily — polls, money, "
     "endorsements, incumbency, who actually controls the decision."),
    ("skeptic", "a contrarian skeptic: the status quo usually holds, most proposed changes DON'T happen by "
     "their deadline, and hype/longshots are overpriced; you fade them toward their low base rate."),
    ("momentum", "a momentum/trend reader: you extrapolate the direction of recent movement and late shifts."),
    ("market_aware", "a market-savvy forecaster who asks what a sharp bettor with this same evidence would "
     "price, and avoids rounding to 0/1 without decisive proof."),
]

_PROMPT = """You are a professional SUPERFORECASTER — {lens}
QUESTION (resolves YES or NO): {q}
TODAY: {today}
GROUNDED AS-OF EVIDENCE (only what was known by the question date; nothing after):
{scene}

Forecast in the disciplined way:
1. REFERENCE CLASS: what is the base rate for this kind of event? (e.g. incumbents usually win; fringe
   parties almost never win a seat; most bills do NOT pass by a given deadline.)
2. ADJUST from the base rate ONLY for what the grounded standing/evidence actually says — and adjust
   PROPORTIONALLY: a clear front-runner with a real lead should move you FAR toward YES; thin or absent
   evidence should keep you near the base rate, not at a coin flip.
3. Do not output 0 or 1 without decisive, resolved evidence.

Return ONLY JSON: {{"base_rate": <0..1>, "p": <0..1 your final probability of YES>,
"why": "<one sentence: base rate X, adjusted to Y because Z>"}}"""


@dataclass
class PanelForecast:
    p_event: float
    n_forecasters: int
    audit: list = field(default_factory=list)     # per-forecaster {lens, base_rate, p, why}
    n_calls: int = 0


@dataclass
class ObserverPanel:
    llm_hot: object                                # temperature>0 so repeated lenses differ
    reps_per_lens: int = 2                         # resample each lens (denoise) — scale up for more agents
    max_workers: int = 8

    def forecast(self, question, dossier, *, today="") -> PanelForecast:
        jobs = [lens for lens in LENSES for _ in range(self.reps_per_lens)]
        scene = dossier.brief()

        def one(lens):
            r = parse_json(self.llm_hot(_PROMPT.format(lens=lens[1], q=question, today=today, scene=scene)))
            if not r:
                return None
            try:
                p = min(1.0, max(0.0, float(r["p"])))
            except (KeyError, TypeError, ValueError):
                return None
            return {"lens": lens[0], "base_rate": r.get("base_rate"), "p": p, "why": str(r.get("why", ""))[:200]}

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = [r for r in ex.map(one, jobs) if r is not None]
        if not results:
            return PanelForecast(p_event=None, n_forecasters=0, n_calls=len(jobs))
        pooled = pool_distribution([{"yes": r["p"], "no": 1 - r["p"]} for r in results])
        return PanelForecast(p_event=pooled["yes"], n_forecasters=len(results),
                             audit=results[:12], n_calls=len(jobs))
