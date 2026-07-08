"""The OBJECTIVE FUNCTION of the message optimizer: P(reply | recipient, message-strategy).

The whole point of the three-layer optimizer is to make the world model — not an LLM's taste — decide
which message is best. This module is that world model, in the form the search needs: a cheap, analytic,
UNCERTAINTY-AWARE map from a *strategy* (a vector of message-controllable variables) to a distribution of
reply probabilities, for a FIXED recipient state.

Why not just reuse a fitted readout? Two reasons, and they are the crux of the architecture:

  1. RECIPIENT-CONDITIONED INTERACTIONS. A plain logistic over the variables is linear and cannot express
     "credential-signaling helps a status-seeker but HURTS a prestige-skeptic." That sign-flip is an
     INTERACTION (message_var × recipient_trait). We add those terms explicitly — this is what makes the
     optimizer recommend credential_signaling→0 for Peter Thiel and →high for a status-driven recipient,
     from the SAME objective.
  2. HONEST UNCERTAINTY FOR THE GUARDRAIL. Each elasticity is a prior with a CI (a `WeightPrior`), never a
     point. `score_dist` samples the weights from that prior (the ensemble), so the objective returns a
     DISTRIBUTION of P(reply). The optimizer then maximizes a pessimistic LOWER BOUND (a low percentile),
     not the mean — so it cannot win by exploiting a single high-variance weight (anti-Goodhart).

Elasticities here are WORLD-KNOWLEDGE PRIORS (the cached analog of `llm_prior.prior_from_llm`), coarse and
signed, NOT fitted to reply data. Everything this scorer produces is therefore `unvalidated` until a real
backtest calibrates the magnitudes. It is an honest prior objective — good enough to SEARCH against, and
labelled as a claim to check, not a truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.variables.calibrated_weights import WeightPrior
from swm.variables.schema import BY_CATEGORY, spec

# The variables the SENDER controls — the search space of the optimizer (everything else is the recipient).
MESSAGE_VARS = [
    "personalization", "clarity", "pushiness", "ask_directness", "length_fit",
    "credential_signaling", "contrarian_pitch", "secret_density",
]


def _sigmoid(z: float) -> float:
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _neutral(name: str) -> float:
    """The 'no signal' point for a variable: 0 for a signed axis, else its population default."""
    s = spec(name)
    return 0.0 if s.signed else s.default


# Global magnitude on the (coarse, world-knowledge) elasticities. Their SIGNS and RELATIVE RANKING are
# the trustworthy signal; their absolute scale is NOT calibrated. This factor keeps a fully-optimized
# cold message from producing an absurd P(reply); it is set conservatively and everything downstream is
# stamped `unvalidated` until a real reply-outcome backtest fits these magnitudes.
ELASTICITY_SCALE = 0.5


# --- the elasticity priors (signed per-unit logit effects; world knowledge, NOT fitted) -----------
# MAIN effects: how a message variable moves P(reply) on its own, holding the recipient neutral.
_MAIN: list[WeightPrior] = [
    WeightPrior("personalization", 1.3, 0.6, "llm"),
    WeightPrior("clarity", 0.9, 0.5, "llm"),
    WeightPrior("ask_directness", 0.7, 0.5, "llm"),
    WeightPrior("length_fit", 1.1, 0.5, "llm"),      # length_fit is already a [0,1] fit-quality
    WeightPrior("pushiness", -1.8, 0.6, "llm"),
    WeightPrior("credential_signaling", -0.35, 0.6, "llm"),   # mild main cost (busy readers); the action is the interaction
    WeightPrior("contrarian_pitch", 0.20, 0.6, "llm"),        # risky on its own
    WeightPrior("secret_density", 0.45, 0.5, "llm"),
]

# INTERACTION effects: (message_var × recipient_var), signed. This is where the recipient's inferred
# disposition changes what the optimal message IS. Each is a WeightPrior over the product of the two
# centered variables. These encode the non-obvious, recipient-specific truths.
_INTERACTIONS: list[tuple[str, str, WeightPrior]] = [
    # a prestige-skeptic PUNISHES credential parading — the Thiel sign-flip, made mechanical.
    ("credential_signaling", "status_orientation", WeightPrior("cred×statusorient", -2.4, 0.8, "llm")),
    # ...but a status-driven recipient rewards it.
    ("credential_signaling", "status", WeightPrior("cred×status", 0.7, 0.7, "llm")),
    # a contrarian/skeptic rewards a genuinely contrarian pitch; a consensus-minded one is unmoved/put off.
    ("contrarian_pitch", "skepticism", WeightPrior("contra×skeptic", 2.0, 0.8, "llm")),
    # a secret lands harder with someone who values non-obvious truths (skeptics/high-openness).
    ("secret_density", "skepticism", WeightPrior("secret×skeptic", 0.9, 0.6, "llm")),
    ("secret_density", "openness_to_outreach", WeightPrior("secret×open", 0.6, 0.6, "llm")),
    # personalization compounds with any existing relationship.
    ("personalization", "relationship_strength", WeightPrior("pers×rel", 0.6, 0.5, "llm")),
    # the higher a person's status, the more pushiness costs (they don't chase).
    ("pushiness", "status", WeightPrior("push×status", -1.0, 0.6, "llm")),
    # an unsolicited-outreach-friendly recipient forgives a direct ask.
    ("ask_directness", "openness_to_outreach", WeightPrior("ask×open", 0.6, 0.5, "llm")),
    # a busy / low-attention recipient punishes effort (long, multi-ask) messages harder — length_fit
    # already carries the sign; this sharpens it when attention is scarce.
    ("length_fit", "attention_availability", WeightPrior("len×attn", 0.5, 0.5, "llm")),
]


@dataclass
class ScoreDist:
    """A distribution of P(reply) for one strategy, plus the driver attribution at the mean weights."""
    mean: float
    samples: list = field(default_factory=list)
    drivers: list = field(default_factory=list)     # [{"term":.., "contribution":..}] at mean weights

    def lower_bound(self, q: float = 0.2) -> float:
        """The q-quantile of P(reply) across weight samples — the pessimistic objective (anti-Goodhart)."""
        if not self.samples:
            return self.mean
        s = sorted(self.samples)
        i = min(len(s) - 1, max(0, int(q * len(s))))
        return s[i]

    def interval(self, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
        if not self.samples:
            return (self.mean, self.mean)
        s = sorted(self.samples)
        return (s[max(0, int(lo * len(s)))], s[min(len(s) - 1, int(hi * len(s)))])


@dataclass
class StrategyScorer:
    """P(reply | recipient, strategy) as an elasticity-prior logistic with recipient-conditioned
    interactions and weight-posterior sampling. Construct one per recipient; call `score_dist(strategy)`."""
    recipient: dict = field(default_factory=dict)       # {var: value} the FIXED recipient state
    base_responsiveness: float = 0.28                   # recipient's inferred base reply rate
    n_weight_samples: int = 160
    seed: int = 0

    # cached term list so main + interaction terms share one loop
    def _terms(self, strategy: dict):
        terms = []
        for wp in _MAIN:
            f = strategy.get(wp.name, _neutral(wp.name)) - _neutral(wp.name)
            terms.append((wp, f))
        for mvar, rvar, wp in _INTERACTIONS:
            mf = strategy.get(mvar, _neutral(mvar)) - _neutral(mvar)
            rf = self.recipient.get(rvar, _neutral(rvar)) - _neutral(rvar)
            terms.append((wp, mf * rf))
        return terms

    def _base_logit(self) -> float:
        # start from the recipient's inferred base reply rate; a platform norm nudge if present.
        z = _logit(self.base_responsiveness)
        norm = self.recipient.get("platform_response_norm")
        if norm is not None:
            z += 0.4 * (_logit(min(0.95, max(0.02, norm))) - _logit(0.3))
        return z

    def score_dist(self, strategy: dict) -> ScoreDist:
        import random
        rng = random.Random(self.seed)
        base = self._base_logit()
        terms = self._terms(strategy)
        sc = ELASTICITY_SCALE
        # mean-weight prediction + driver attribution
        mean_logit = base + sum(sc * wp.mean * f for wp, f in terms)
        drivers = sorted(
            [{"term": wp.name, "contribution": round(sc * wp.mean * f, 4)} for wp, f in terms if abs(f) > 1e-9],
            key=lambda d: abs(d["contribution"]), reverse=True)
        # ensemble: sample each elasticity from its prior (mean, sd) -> a distribution of P(reply)
        samples = []
        for _ in range(self.n_weight_samples):
            z = base
            for wp, f in terms:
                if f != 0.0:
                    z += sc * rng.gauss(wp.mean, wp.sd) * f
            samples.append(_sigmoid(z))
        return ScoreDist(mean=_sigmoid(mean_logit), samples=samples, drivers=drivers[:12])

    # convenience
    def mean(self, strategy: dict) -> float:
        return self.score_dist(strategy).mean

    def lower_bound(self, strategy: dict, q: float = 0.2) -> float:
        return self.score_dist(strategy).lower_bound(q)


def scorer_from_recipient(recipient_vars: dict, base_responsiveness: float, *,
                          n_weight_samples: int = 160, seed: int = 0) -> StrategyScorer:
    """Build a scorer from a recipient's inferred variable values (persona + web/public-figure evidence)."""
    return StrategyScorer(recipient=dict(recipient_vars), base_responsiveness=base_responsiveness,
                          n_weight_samples=n_weight_samples, seed=seed)
