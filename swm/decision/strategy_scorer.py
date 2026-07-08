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


# unified term table: (name, message_var, recipient_var|None, prior_mean, prior_sd). Shared by the
# scorer AND the elasticity-fitting harness so both compute identical features.
TERMS = ([(wp.name, wp.name, None, wp.mean, wp.sd) for wp in _MAIN]
         + [(wp.name, mvar, rvar, wp.mean, wp.sd) for mvar, rvar, wp in _INTERACTIONS])
TERM_NAMES = [t[0] for t in TERMS]


def term_features(strategy: dict, recipient: dict) -> list:
    """Feature value for every term given (strategy, recipient): main = centered message var; interaction
    = centered message var × centered recipient var. Returns [(name, feature_value, prior_mean, prior_sd)]."""
    out = []
    for name, mvar, rvar, mean, sd in TERMS:
        mf = strategy.get(mvar, _neutral(mvar)) - _neutral(mvar)
        f = mf if rvar is None else mf * (recipient.get(rvar, _neutral(rvar)) - _neutral(rvar))
        out.append((name, f, mean, sd))
    return out


def base_logit(base_responsiveness: float, recipient: dict) -> float:
    """The per-recipient offset: their inferred base reply rate, nudged by the platform response norm."""
    z = _logit(base_responsiveness)
    norm = recipient.get("platform_response_norm")
    if norm is not None:
        z += 0.4 * (_logit(min(0.95, max(0.02, norm))) - _logit(0.3))
    return z


@dataclass
class StrategyScorer:
    """P(reply | recipient, strategy) as an elasticity logistic with recipient-conditioned interactions
    and weight-posterior sampling. Construct one per recipient; call `score_dist(strategy)`.

    By default the elasticities are the coarse world-knowledge PRIORS (× ELASTICITY_SCALE) and any
    prediction is `unvalidated`. Pass `weights` (a fitted {term: (mean, sd)} from the elasticity-fitting
    harness, ABSOLUTE — no scale) and `grade` to turn it into a data-calibrated scorer that carries a
    real grade."""
    recipient: dict = field(default_factory=dict)       # {var: value} the FIXED recipient state
    base_responsiveness: float = 0.28                   # recipient's inferred base reply rate
    n_weight_samples: int = 160
    seed: int = 0
    weights: dict | None = None                         # fitted {term: (mean, sd)}; overrides priors
    grade: dict | None = None                           # calibration grade of the fitted weights

    def _weight_for(self, name: str, prior_mean: float, prior_sd: float):
        """(mean, sd, scale) for a term: the fitted weight (scale 1) if present, else the prior × scale."""
        if self.weights and name in self.weights:
            m, s = self.weights[name]
            return m, s, 1.0
        return prior_mean, prior_sd, ELASTICITY_SCALE

    def _base_logit(self) -> float:
        return base_logit(self.base_responsiveness, self.recipient)

    def score_dist(self, strategy: dict) -> ScoreDist:
        import random
        rng = random.Random(self.seed)
        base = self._base_logit()
        terms = term_features(strategy, self.recipient)
        mean_logit = base
        drivers = []
        for name, f, pmean, psd in terms:
            m, _s, sc = self._weight_for(name, pmean, psd)
            mean_logit += sc * m * f
            if abs(f) > 1e-9:
                drivers.append({"term": name, "contribution": round(sc * m * f, 4)})
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        # ensemble: sample each elasticity from its (fitted or prior) mean/sd -> a distribution of P(reply)
        samples = []
        for _ in range(self.n_weight_samples):
            z = base
            for name, f, pmean, psd in terms:
                if f != 0.0:
                    m, s, sc = self._weight_for(name, pmean, psd)
                    z += sc * rng.gauss(m, s) * f
            samples.append(_sigmoid(z))
        return ScoreDist(mean=_sigmoid(mean_logit), samples=samples, drivers=drivers[:12])

    # convenience
    def mean(self, strategy: dict) -> float:
        return self.score_dist(strategy).mean

    def lower_bound(self, strategy: dict, q: float = 0.2) -> float:
        return self.score_dist(strategy).lower_bound(q)


def scorer_from_recipient(recipient_vars: dict, base_responsiveness: float, *,
                          n_weight_samples: int = 160, seed: int = 0,
                          weights: dict | None = None, grade: dict | None = None) -> StrategyScorer:
    """Build a scorer from a recipient's inferred variable values (persona + web/public-figure evidence).
    Pass fitted `weights`/`grade` from the elasticity-fitting harness to use a data-calibrated objective."""
    return StrategyScorer(recipient=dict(recipient_vars), base_responsiveness=base_responsiveness,
                          n_weight_samples=n_weight_samples, seed=seed, weights=weights, grade=grade)
