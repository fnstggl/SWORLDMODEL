"""The RESPONSE FUNNEL — a conjunctive model of whether a busy stranger replies, and how.

The additive logit scorer has a structural blind spot for cold outreach: it lets one maxed lever
arithmetically compensate for a failed gate. A message can score 1.0 on contrarian specificity and
still die because the recipient never figured out who was writing. Real cold-outreach response is a
CHAIN OF GATES, every one of which must pass:

    open  ×  understand  ×  believe  ×  relevant  ×  worth-my-time  ×  easy-to-act

so the model is a product of stage probabilities (a noisy-AND), not a sum of merits:

    P(positive reply) = P(open) · Π_stage σ(z_stage)
    P(negative reply) = P(open) · σ(z_annoy)          # irritated correction / "remove me" / rep damage

Stage inputs are the funnel levers (identity_legibility, claim_believability, cognitive_effort,
adversarial_framing, next_step_clarity) plus the classic levers routed to the stage they actually
gate. Situational (per-recipient) levers enter the worth stage. The outcome is VALENCED: the
objective downstream is P(positive) − λ·P(negative), because "any reply" is a misspecified goal —
an irritated correction counts as a reply and should not count as success.

Weights are world-knowledge priors with uncertainty (WeightPrior mean/sd, sampled as an ensemble for
the pessimistic lower bound, same discipline as the additive scorer). HONESTY: this funnel is a
STRUCTURAL PRIOR, not a fitted model — there is no labeled cold-email corpus behind these magnitudes
(the CMV grade applies to the additive persuasion model only). Trust the ranking and the gate
structure; treat absolute levels as claims. The API is duck-compatible with StrategyScorer
(mean / lower_bound / score_dist / optimizable_vars) so L1 search and the constructors reuse it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.decision.strategy_scorer import WeightPrior, _logit, _sigmoid

NEGATIVE_REPLY_WEIGHT = 0.25            # λ: utility cost of an irritated reply vs a positive one

# ---------------------------------------------------------------- stage definitions
# stage -> (intercept prior, [(lever, weight prior, recipient_moderator|None)]).
# A moderator multiplies the lever's weight by the recipient var (centered at 0.5): e.g. credential
# parading hurts MORE for a status-skeptic; adversarial framing is excused BY an existing relationship.
_STAGES: dict = {
    # knowing WHO is writing and WHY is not the same gate as parsing the sentence: clarity gets a
    # small weight so it cannot buy back missing identity ("begins in the middle of a conversation
    # that never happened" fails here no matter how crisp the prose is)
    "understand": (WeightPrior("understand_b", -0.9, 0.4, "llm"), [
        ("identity_legibility", WeightPrior("u_identity", 3.0, 0.8, "llm"), None),
        ("clarity", WeightPrior("u_clarity", 0.8, 0.4, "llm"), None),
    ]),
    "believe": (WeightPrior("believe_b", 0.1, 0.4, "llm"), [
        ("claim_believability", WeightPrior("b_believable", 2.2, 0.7, "llm"), None),
        ("credibility_proof", WeightPrior("b_proof", 0.5, 0.4, "llm"), None),
        # a skeptic discounts unanchored claims hardest: believability matters MORE for them
        ("claim_believability", WeightPrior("b_believe_x_skeptic", 0.8, 0.5, "llm"), "skepticism"),
    ]),
    "relevant": (WeightPrior("relevant_b", 0.0, 0.4, "llm"), [
        ("relevance_fit", WeightPrior("r_relevance", 1.8, 0.6, "llm"), None),
        ("personalization", WeightPrior("r_personal", 0.8, 0.5, "llm"), None),
    ]),
    "worth": (WeightPrior("worth_b", 0.5, 0.4, "llm"), [
        ("responder_incentive", WeightPrior("w_incentive", 0.9, 0.5, "llm"), None),
        ("warmth", WeightPrior("w_warmth", 0.5, 0.4, "llm"), None),
        ("cognitive_effort", WeightPrior("w_effort", -1.8, 0.6, "llm"), None),
        ("adversarial_framing", WeightPrior("w_adversarial", -1.6, 0.6, "llm"), None),
        # an existing relationship excuses challenge framing (positive moderator on a negative lever)
        ("adversarial_framing", WeightPrior("w_advers_x_rel", 0.9, 0.5, "llm"), "relationship_strength"),
        ("pushiness", WeightPrior("w_pushy", -1.8, 0.6, "llm"), None),
        ("convenience_selling", WeightPrior("w_convenience", -0.8, 0.5, "llm"), None),
        ("convenience_selling", WeightPrior("w_conv_x_status", -1.6, 0.7, "llm"), "status_orientation"),
        ("credential_signaling", WeightPrior("w_cred", -0.3, 0.4, "llm"), None),
        ("credential_signaling", WeightPrior("w_cred_x_status", -2.2, 0.8, "llm"), "status_orientation"),
    ]),
    "easy": (WeightPrior("easy_b", 0.4, 0.4, "llm"), [
        ("next_step_clarity", WeightPrior("e_next_step", 2.0, 0.6, "llm"), None),
        ("low_effort_ask", WeightPrior("e_low_effort", 0.9, 0.5, "llm"), None),
        ("cognitive_effort", WeightPrior("e_effort", -1.2, 0.5, "llm"), None),
    ]),
}

# annoyance model (negative-valence reply / silent reputational damage)
_ANNOY = (WeightPrior("annoy_b", -1.6, 0.4, "llm"), [
    ("adversarial_framing", WeightPrior("a_adversarial", 1.6, 0.6, "llm"), None),
    ("pushiness", WeightPrior("a_pushy", 1.6, 0.6, "llm"), None),
    ("convenience_selling", WeightPrior("a_convenience", 1.0, 0.5, "llm"), None),
    ("credential_signaling", WeightPrior("a_cred_x_status", 1.0, 0.6, "llm"), "status_orientation"),
])


@dataclass
class FunnelDist:
    """Ensemble distribution of (p_positive, p_negative) plus the mean stage trace (the WHY)."""
    p_pos: list = field(default_factory=list)
    p_neg: list = field(default_factory=list)
    stage_trace: dict = field(default_factory=dict)      # stage -> mean pass probability

    @property
    def mean(self) -> float:
        return sum(self.p_pos) / len(self.p_pos) if self.p_pos else 0.0

    @property
    def mean_neg(self) -> float:
        return sum(self.p_neg) / len(self.p_neg) if self.p_neg else 0.0

    def objective_samples(self, lam: float = NEGATIVE_REPLY_WEIGHT) -> list:
        return [p - lam * n for p, n in zip(self.p_pos, self.p_neg)]

    def lower_bound(self, q: float = 0.2, lam: float = NEGATIVE_REPLY_WEIGHT) -> float:
        s = sorted(self.objective_samples(lam))
        return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0

    @property
    def drivers(self) -> list:
        """StrategyScorer-compatible driver attribution: the funnel's 'why' is the stage pass
        probabilities (the gate that limits the product is the driver that matters)."""
        return [{"term": f"stage:{k}", "contribution": round(v, 4)}
                for k, v in sorted(self.stage_trace.items(), key=lambda kv: kv[1])]


@dataclass
class FunnelScorer:
    """Duck-compatible with StrategyScorer for L1 search and construction: `mean(strategy)`,
    `lower_bound(strategy, q)`, `score_dist(strategy)`, `optimizable_vars()`. The objective every
    caller sees is VALENCED: P(positive) − λ·P(negative)."""
    recipient: dict
    base_responsiveness: float = 0.28
    n_weight_samples: int = 120
    seed: int = 0
    levers: list = field(default_factory=list)           # situational levers -> worth stage
    grade: object = None                                 # provenance passthrough (never invented)

    def optimizable_vars(self) -> list:
        from swm.decision.strategy_scorer import MESSAGE_VARS
        return list(MESSAGE_VARS) + [lv.name for lv in self.levers]

    # ---- internals ----
    def _moderator(self, name: str) -> float:
        return float(self.recipient.get(name, 0.5)) - 0.5

    def _stage_z(self, stage_key: str, strategy: dict, draw) -> float:
        intercept, terms = _STAGES[stage_key]
        z = draw(intercept)
        for lever, wp, moderator in terms:
            v = float(strategy.get(lever, 0.0) or 0.0)
            w = draw(wp)
            if moderator is not None:
                w = w * 2.0 * self._moderator(moderator)   # centered moderator in [-1, 1]
            z += w * v
        if stage_key == "worth":
            for lv in self.levers:
                z += draw(WeightPrior(lv.name, lv.elasticity_mean, lv.elasticity_sd, "llm")) * \
                    float(strategy.get(lv.name, 0.0) or 0.0)
        return z

    def _p_open(self) -> float:
        z = _logit(min(0.95, max(0.005, self.base_responsiveness)))
        norm = self.recipient.get("platform_response_norm")
        if norm is not None:
            z += 0.4 * (_logit(min(0.95, max(0.02, norm))) - _logit(0.3))
        return _sigmoid(z)

    def score_dist(self, strategy: dict) -> FunnelDist:
        rng = random.Random(self.seed)
        out = FunnelDist()
        stage_acc = {k: 0.0 for k in _STAGES}
        p_open = self._p_open()
        for _ in range(max(1, self.n_weight_samples)):
            def draw(wp: WeightPrior) -> float:
                return rng.gauss(wp.mean, wp.sd)
            p = p_open
            for k in _STAGES:
                sp = _sigmoid(self._stage_z(k, strategy, draw))
                stage_acc[k] += sp
                p *= sp
            zb, terms = _ANNOY
            za = draw(zb)
            for lever, wp, moderator in terms:
                w = draw(wp)
                if moderator is not None:
                    w = w * 2.0 * self._moderator(moderator)
                za += w * float(strategy.get(lever, 0.0) or 0.0)
            out.p_pos.append(p)
            out.p_neg.append(p_open * _sigmoid(za))
        n = max(1, self.n_weight_samples)
        out.stage_trace = {k: round(v / n, 4) for k, v in stage_acc.items()}
        out.stage_trace["open"] = round(p_open, 4)
        return out

    # ---- the StrategyScorer-compatible surface ----
    def mean(self, strategy: dict) -> float:
        d = self.score_dist(strategy)
        return d.mean - NEGATIVE_REPLY_WEIGHT * d.mean_neg

    def lower_bound(self, strategy: dict, q: float = 0.2) -> float:
        return self.score_dist(strategy).lower_bound(q)


def funnel_scorer_from_recipient(recipient_vars: dict, base_mean: float, *, seed: int = 0,
                                 levers=None, grade=None) -> FunnelScorer:
    return FunnelScorer(recipient=dict(recipient_vars), base_responsiveness=base_mean, seed=seed,
                        levers=list(levers or []), grade=grade)
