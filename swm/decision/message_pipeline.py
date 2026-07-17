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
                                                default_proposer, encode_text_to_strategy, polish_email)
from swm.decision.mc_evaluation import MCResult, mc_evaluate
from swm.decision.message_optimizer import StrategySpec, optimize_strategy
from swm.decision.semantic_critic import SemanticCritic
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


def build_contrast_baselines(sender_brief=None, recipient_label: str = "") -> dict:
    """Naive contrast drafts BUILT FROM THE ACTUAL BRIEF — the instinctive credential-parade cover
    letter and the pushy follow-up a person would actually send about THIS idea to THIS recipient.
    (These replace an old module-level constant that hardcoded one sender's stale facts — NYT feature,
    housing startup — into every run regardless of the brief.) Evaluated for contrast only; they never
    feed construction."""
    first = (recipient_label or "there").split()[0]
    last = (recipient_label or "").split()[-1] if recipient_label else "there"
    facts = list(getattr(sender_brief, "facts", []) or [])
    thesis = getattr(sender_brief, "thesis", "") or "my startup"
    sender = getattr(sender_brief, "sender", "") or "a founder"
    fact_clause = ("; ".join(f.rstrip(".") for f in facts[:3])) if facts else "an impressive background"
    return {
        "credential_cover_letter":
            f"Dear Mr. {last}, I hope this email finds you well. My name is {sender} and I am reaching "
            f"out because I have {fact_clause}. I am building something I truly believe is "
            f"revolutionary: {thesis.rstrip('.')}. I would love to set up a 30-minute call at your "
            "earliest convenience to tell you more. Looking forward to hearing back from you soon.",
        "pushy_followup":
            f"Hi {first}, just following up and circling back per my last email. Please respond ASAP "
            f"about {thesis.rstrip('.')} — I'd really love to get on your calendar this week.",
    }


@dataclass
class OptimizationResult:
    recipient: str
    spec: StrategySpec
    email: ConstructedEmail
    evaluation: MCResult
    baselines: dict = field(default_factory=dict)     # label -> {"text":.., "mc": MCResult}
    grade: dict | None = None                         # calibration grade if a fitted model was used

    def summary(self) -> dict:
        graded = self.grade and self.grade.get("grade") not in (None, "unvalidated", "F")
        honesty = (
            f"CALIBRATED (grade {self.grade['grade']}, ECE {self.grade.get('ece')}). The elasticities were "
            "fit to reply outcomes and graded on a held-out split; the P(reply) carries that grade."
            if graded else
            "UNVALIDATED. The objective uses coarse world-knowledge elasticity priors, not a reply-outcome "
            "backtest. Trust the RANKING and the DIRECTION of the levers; treat the absolute P(reply) as a "
            "claim to check. Fit the elasticities on labeled reply outcomes (swm/decision/elasticity_fit.py) "
            "to earn a calibration grade.")
        return {
            "report_type": "prediction",
            "recipient": self.recipient,
            "calibration_grade": (self.grade or {}).get("grade", "unvalidated"),
            "optimal_strategy_spec": self.spec.summary(),
            "constructed_email": self.email.summary(),
            "evaluation": self.evaluation.summary(),
            "baselines_for_contrast": {
                k: {"text": v["text"], "reply_mean": round(v["mc"].p_mean, 4),
                    "interval80": [round(v["mc"].interval80[0], 4), round(v["mc"].interval80[1], 4)]}
                for k, v in self.baselines.items()},
            "honesty": honesty,
        }


def optimize_message(recipient: RecipientState, *, proposer=default_proposer, q: float = 0.2,
                     restarts: int = 12, beam: int = 6, n_mc: int = 2000, seed: int = 0,
                     baselines: dict | None = None, judge_fn=None, chat_fn=None,
                     sender_brief=None, recipient_notes: str = "", fit=None) -> OptimizationResult:
    """Run L1 → L2 → (critic gate) → L3 for a recipient and return the constructed email + distribution.

    chat_fn — optional live LLM `fn(prompt)->text`. When given, it becomes BOTH the move proposer (the LLM
    writes candidate sentences per slot, constrained by the L1 strategy + recipient evidence + sender
    facts) AND the sentence judge for the critic — turning the offline "no slop" into real, well-written
    text. Without it, the offline sentence bank + lexical critic run. judge_fn/proposer can be set
    explicitly to override. Everything degrades gracefully."""
    # a fitted elasticity model (FittedElasticities from elasticity_fit.grade_fit) makes the objective
    # data-calibrated and the reported grade real instead of 'unvalidated'.
    fit_weights = fit.weights if fit is not None else None
    fit_grade = fit.grade if fit is not None else None
    grade_letter = fit_grade.get("grade") if fit_grade else "unvalidated"

    # per-recipient SITUATIONAL levers + the LLM message ENCODER (both live only when a chat_fn is given)
    levers = []
    encode_fn = encode_text_to_strategy               # lexical fallback
    rewrite_fn = None
    if chat_fn is None and sender_brief is not None and proposer is default_proposer:
        # offline with a real brief: the bank derives from the brief, not the fixture scenario
        from swm.decision.compositional_search import bank_proposer
        proposer = bank_proposer(sender_brief, recipient.label)
    if chat_fn is not None:
        from swm.decision.llm_moves import (llm_message_encoder, llm_proposer, llm_rewriter,
                                            llm_sentence_judge)
        from swm.decision.situational_levers import generate_levers
        levers = generate_levers(chat_fn, recipient.label, recipient.vars,
                                 evidence=recipient_notes)
        encode_fn = llm_message_encoder(chat_fn, levers=levers)
        if proposer is default_proposer:
            proposer = llm_proposer(chat_fn, recipient_notes=recipient_notes, sender=sender_brief,
                                    levers=levers)
        if judge_fn is None:
            # the judge sees the sender's REAL facts so it can flag fabricated specifics (a number,
            # dataset, client, or result the facts don't contain) alongside annoying/AI-sounding lines
            judge_fn = llm_sentence_judge(
                chat_fn, facts_text=(sender_brief.to_prompt() if sender_brief is not None else ""))
        rewrite_fn = llm_rewriter(chat_fn, recipient_notes=recipient_notes, sender=sender_brief)

    scorer = scorer_from_recipient(recipient.vars, recipient.base_mean, seed=seed,
                                   weights=fit_weights, grade=fit_grade, levers=levers)
    # the deterministic numeric-factuality floor: distinctive numbers must come from the sender's
    # real facts (or the recipient notes) — enforced in BOTH critics, LLM or lexical
    fact_numbers = None
    if sender_brief is not None:
        from swm.decision.llm_moves import allowed_numbers
        fact_numbers = allowed_numbers(sender_brief.to_prompt(), recipient_notes)
    fast_critic = SemanticCritic(allowed_numbers=fact_numbers)   # cheap lexical — safe inside the beam
    final_critic = SemanticCritic(judge_fn=judge_fn,             # LLM if provided — the final gate
                                  allowed_numbers=fact_numbers)

    # L1 — optimal strategy in variable space (no text), over the general + situational levers
    spec = optimize_strategy(scorer, q=q, restarts=restarts, seed=seed)

    # L2 — assemble the email move-by-move to realize the spec (slop pruned as it builds)
    email = construct_email(scorer, spec.strategy, proposer=proposer, beam=beam, q=q,
                            critic=fast_critic, context={"recipient": recipient.label}, encode_fn=encode_fn)

    # CRITIC GATE — the critical evaluator at the end: flag/repair incoherent or annoying lines. With a
    # live LLM, a targeted rewrite (critic reason -> writer) fixes lines whose whole sample-register is
    # slop, and we RANK repairs with the same (LLM) critic that flags — so the ranker isn't blind to the
    # exact issue the gate found.
    email = polish_email(email, scorer, spec.strategy, proposer=proposer, critic=final_critic, q=q,
                         rewrite_fn=rewrite_fn, rank_critic=(final_critic if chat_fn is not None else None),
                         encode_fn=encode_fn)

    # L3 — Monte-Carlo evaluate the finalist under recipient hidden state (fitted weights + grade if any)
    evaluation = mc_evaluate(recipient.vars, recipient.base_mean, email.strategy,
                             base_n_effective=recipient.base_n_effective,
                             confidences=recipient.confidences, n_samples=n_mc, seed=seed,
                             weights=fit_weights, grade=grade_letter, levers=levers)

    # contrast: run naive drafts through the SAME evaluator (built from the ACTUAL brief + recipient)
    result_baselines = {}
    for label, text in (baselines or build_contrast_baselines(sender_brief, recipient.label)).items():
        mc = mc_evaluate(recipient.vars, recipient.base_mean, encode_fn(text),
                         base_n_effective=recipient.base_n_effective, confidences=recipient.confidences,
                         n_samples=n_mc, seed=seed, weights=fit_weights, grade=grade_letter, levers=levers)
        result_baselines[label] = {"text": text, "mc": mc}

    return OptimizationResult(recipient=recipient.label, spec=spec, email=email, evaluation=evaluation,
                              baselines=result_baselines, grade=fit_grade)


def _recipient_notes(world, contact_id: str, name: str | None) -> str:
    """Compact notes for the LLM proposer: who the recipient is + the web evidence + inferred traits."""
    prof = world.profile(contact_id)
    lines = [f"Recipient: {name or contact_id}"]
    if prof:
        iv = prof.get("inferred_variables", {})
        traits = ", ".join(f"{k}={v['value']:.2f}" for k, v in iv.items() if k != "base_responsiveness")
        if traits:
            lines.append(f"Inferred disposition (0-1): {traits}")
        ev = prof.get("evidence")
        # evidence snippets live on the resolver; summarize what we have
        lines.append("Note: high status_orientation means they are put off by credential/prestige "
                     "signaling; high skepticism means they reward a genuinely contrarian, specific claim.")
    return "\n".join(lines)


def optimize_for_world(world, contact_id: str, *, name: str | None = None, domain: str = "",
                       ask: str = "", sender_brief=None, chat_fn=None, **kw) -> OptimizationResult:
    """Convenience: build the recipient from a World and optimize in one call. Pass a live `chat_fn`
    (e.g. swm.api.deepseek_backend.default_chat_fn()) and a `sender_brief` to have the LLM write the moves."""
    rs = recipient_from_world(world, contact_id, name=name, domain=domain, ask=ask)
    notes = _recipient_notes(world, contact_id, name)
    return optimize_message(rs, sender_brief=sender_brief, chat_fn=chat_fn, recipient_notes=notes, **kw)
