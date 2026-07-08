"""The message optimizer, end to end: recipient → L1 strategy → L2 construction → L3 evaluation.

Orchestrates the three layers into one call and connects them to the rest of the system: the recipient
state comes from a `World` persona + public-figure profile (the inference-by-default path), the objective
is the `StrategyScorer`, and the result carries the constructed email, the optimal strategy spec, the
Monte-Carlo reply distribution, and an honesty stamp. For contrast it also runs a couple of naive drafts
(a credential-parade cover letter, a pushy follow-up) through the SAME evaluator, so the lift is measured,
not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.decision.compositional_search import (ConstructedEmail, construct_email,
                                                default_proposer, encode_text_to_strategy)
from swm.decision.mc_evaluation import MCResult, mc_evaluate
from swm.decision.message_optimizer import StrategySpec, optimize_strategy
from swm.decision.strategy_scorer import scorer_from_recipient

# recipient variables we derive/assume beyond what the profile provides (email cold-outreach defaults)
_DEFAULT_RECIPIENT = {"platform_response_norm": 0.30, "attention_availability": 0.6,
                      "relationship_strength": 0.0}


@dataclass
class RecipientState:
    vars: dict
    base_mean: float
    base_n_effective: float
    confidences: dict = field(default_factory=dict)
    label: str = ""


def recipient_from_world(world, contact_id: str, *, name: str | None = None, domain: str = "",
                         ask: str = "") -> RecipientState:
    """Build the recipient state from the World's inferred persona + public-figure web evidence."""
    p = world.persona(contact_id, name=name, domain=domain, ask=ask)
    prof = world.profile(contact_id)
    rvars = dict(_DEFAULT_RECIPIENT)
    confs = {}
    if prof:
        for k, meta in prof.get("inferred_variables", {}).items():
            if k == "base_responsiveness":
                continue
            rvars[k] = meta["value"]
            confs[k] = meta.get("confidence", 0.4)
    # relationship strength from any private history
    rvars["relationship_strength"] = min(1.0, p.n_sends / 8.0)
    return RecipientState(vars=rvars, base_mean=p.responsiveness.mean,
                          base_n_effective=p.responsiveness.n_effective, confidences=confs,
                          label=name or contact_id)


# baseline drafts (the kind of thing the old pipeline / a human would send) — evaluated for contrast only
_BASELINES = {
    "credential_cover_letter":
        "Dear Mr. Thiel, I am a 17-year-old Princeton admit, recently featured in the New York Times for "
        "my affordable-housing startup. I have a 4.0 and several prestigious awards. I would love to set "
        "up a 30-minute call at your earliest convenience to discuss my AI infrastructure company. "
        "Looking forward to hearing back from you soon.",
    "pushy_followup":
        "Hi Peter, just following up and circling back per my last email. Please respond ASAP about my AI "
        "infrastructure startup — I'd really love to get on your calendar this week.",
}


@dataclass
class OptimizationResult:
    recipient: str
    spec: StrategySpec
    email: ConstructedEmail
    evaluation: MCResult
    baselines: dict = field(default_factory=dict)     # label -> {"text":.., "mc": MCResult}

    def summary(self) -> dict:
        return {
            "report_type": "prediction",
            "recipient": self.recipient,
            "optimal_strategy_spec": self.spec.summary(),
            "constructed_email": self.email.summary(),
            "evaluation": self.evaluation.summary(),
            "baselines_for_contrast": {
                k: {"text": v["text"], "reply_mean": round(v["mc"].p_mean, 4),
                    "interval80": [round(v["mc"].interval80[0], 4), round(v["mc"].interval80[1], 4)]}
                for k, v in self.baselines.items()},
            "honesty": "UNVALIDATED. The objective uses coarse world-knowledge elasticity priors, not a "
                       "reply-outcome backtest. Trust the RANKING and the DIRECTION of the levers; treat "
                       "the absolute P(reply) as a claim to check. Import labeled reply outcomes to fit "
                       "the elasticities and earn a calibration grade.",
        }


def optimize_message(recipient: RecipientState, *, proposer=default_proposer, q: float = 0.2,
                     restarts: int = 12, beam: int = 6, n_mc: int = 2000, seed: int = 0,
                     baselines: dict | None = None) -> OptimizationResult:
    """Run L1 → L2 → L3 for a recipient and return the constructed email + its evaluated distribution."""
    scorer = scorer_from_recipient(recipient.vars, recipient.base_mean, seed=seed)

    # L1 — optimal strategy in variable space (no text)
    spec = optimize_strategy(scorer, q=q, restarts=restarts, seed=seed)

    # L2 — assemble the email move-by-move to realize the spec
    email = construct_email(scorer, spec.strategy, proposer=proposer, beam=beam, q=q,
                            context={"recipient": recipient.label})

    # L3 — Monte-Carlo evaluate the finalist under recipient hidden state
    evaluation = mc_evaluate(recipient.vars, recipient.base_mean, email.strategy,
                             base_n_effective=recipient.base_n_effective,
                             confidences=recipient.confidences, n_samples=n_mc, seed=seed)

    # contrast: run naive drafts through the SAME evaluator
    result_baselines = {}
    for label, text in (baselines or _BASELINES).items():
        mc = mc_evaluate(recipient.vars, recipient.base_mean, encode_text_to_strategy(text),
                         base_n_effective=recipient.base_n_effective, confidences=recipient.confidences,
                         n_samples=n_mc, seed=seed)
        result_baselines[label] = {"text": text, "mc": mc}

    return OptimizationResult(recipient=recipient.label, spec=spec, email=email, evaluation=evaluation,
                              baselines=result_baselines)


def optimize_for_world(world, contact_id: str, *, name: str | None = None, domain: str = "",
                       ask: str = "", **kw) -> OptimizationResult:
    """Convenience: build the recipient from a World and optimize in one call."""
    rs = recipient_from_world(world, contact_id, name=name, domain=domain, ask=ask)
    return optimize_message(rs, **kw)
