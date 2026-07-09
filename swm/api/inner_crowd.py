"""Inner-crowd ensemble — simulate the crowd, not one agent.

A prediction market beats an individual because it AGGREGATES many diverse views, cancelling each one's bias. A
single LLM pass is the opposite: one agent with one framing and its biases. This runs the latent-state
simulation through a PANEL of genuinely diverse forecaster archetypes — outside-view base-rater, inside-view
bull, inside-view bear, mechanism domain-expert, balanced superforecaster, contrarian red-teamer, historian,
and quant — each producing its own honest simulation spec (base rate + state + drivers), then AGGREGATES their
calibrated forecasts the way a crowd does: mean of log-odds, then EXTREMIZED (Satopää et al. — independent
partial-signal forecasts, once pooled, justify sharpening; the factor is tuned, not assumed). This is
wisdom-of-crowds computed INSIDE the model — the direct implementation of "compute via many views, not one
instinct."
"""
from __future__ import annotations

import math

from swm.api.latent_forecast import latent_forecast

# Each persona is a distinct COGNITIVE STANCE — the diversity is the whole point. They disagree on purpose.
PERSONAS = {
    "base_rater": "You are a disciplined OUTSIDE-VIEW forecaster. You anchor hard on the reference-class base "
                  "rate and are deeply skeptical of story-driven adjustments — most specific narratives are "
                  "noise and the base rate dominates. Your drivers are few and weak unless truly decisive.",
    "bull": "You are an INSIDE-VIEW OPTIMIST. You actively hunt for concrete reasons the event WILL happen — "
            "momentum, incentives, capability, precedent, who benefits — and weight the evidence that points "
            "toward YES. You believe things happen more often than skeptics expect.",
    "bear": "You are an INSIDE-VIEW SKEPTIC. You actively hunt for reasons the event will NOT happen — the "
            "obstacles, the inertia, everything that must go right, the ways it quietly fails — and weight the "
            "evidence that points toward NO. Most bold things don't happen on schedule.",
    "domain_expert": "You are a DOMAIN EXPERT. You reason from the concrete causal MECHANISM of how this exact "
                     "kind of event resolves — the process, the decision gates, the few variables that truly "
                     "control the outcome — and ignore surface narrative.",
    "superforecaster": "You are an elite SUPERFORECASTER (Tetlock GJP style). You fuse the outside and inside "
                       "views, enumerate 2-4 concrete scenarios and weight them, avoid overconfidence, and make "
                       "granular, well-reasoned probability adjustments off the base rate.",
    "contrarian": "You are a CONTRARIAN RED-TEAMER. You ask what the consensus is missing, where the crowd is "
                  "biased (recency, salience, wishful thinking), and whether the obvious answer is a trap. You "
                  "surface the overlooked factor that flips the call.",
    "historian": "You are a HISTORIAN of analogous cases. You judge by the empirical frequency of the closest "
                 "reference class of past situations — how often did things like this actually resolve YES — "
                 "and let that dominate your base rate.",
    "quant": "You are a QUANTITATIVE forecaster. For anything measurable you insist on the current value, the "
             "distance to the threshold, and the realistic volatility over the horizon; you distrust vibes and "
             "reason in numbers, probabilities, and rates.",
}


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-max(-35, min(35, z))))


def aggregate(probs, *, method="logodds", extremize=1.0):
    """Pool a panel of forecasts. `logodds` = mean of log-odds (the standard forecast-combination); `median`
    = robust; then EXTREMIZE by the factor (>1 sharpens toward the shared signal, 1.0 = plain average)."""
    ps = [p for p in probs if p is not None]
    if not ps:
        return None
    if method == "median":
        s = sorted(ps)
        agg = s[len(s) // 2]
        z = _logit(agg)
    else:
        z = sum(_logit(p) for p in ps) / len(ps)
    return _sig(extremize * z)


def inner_crowd(question, as_of_ts, resolve_ts, persona_llms, *, n=2500, metric_grounder=None, news=None):
    """Run the latent simulation through every persona; return {persona: p}. Each persona is an independent
    forecaster with its own framing, so the panel genuinely disagrees — the raw material for aggregation."""
    out = {}
    for name, llm in persona_llms.items():
        p, _ = latent_forecast(question, as_of_ts, resolve_ts, llm, n=n,
                               metric_grounder=metric_grounder, news=news)
        out[name] = p
    return out
