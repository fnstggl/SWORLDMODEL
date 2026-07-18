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


@dataclass
class ColdOutreachResult:
    """The corrected cold-outreach optimizer's output: a RANKED slate with uncertainty, never one
    'best possible' declaration. `candidates` carries every evaluated arm (persona-ensemble verdict,
    funnel MC + stage trace, contract verdict, critic verdicts, cold-read verdict); `winner` is
    'best-supported among tested' — the top arm among gate-passers under the persona ensemble when
    live (funnel lower bound offline) — with hypothesis-fragility and within-noise arms reported."""
    recipient: str
    spec: StrategySpec
    candidates: dict
    winner: str
    within_noise: list
    honesty: str
    fragility: dict = field(default_factory=dict)

    def summary(self) -> dict:
        def key(kv):
            e = kv[1]
            if "persona" in e:
                return -e["persona"]["expected_utility"]
            return -(e.get("funnel") or {}).get("objective_mean", 0.0)
        ranked = sorted(self.candidates.items(), key=key)
        return {"report_type": "cold_outreach_slate", "recipient": self.recipient,
                "winner_best_supported_among_tested": self.winner,
                "within_noise_of_winner": self.within_noise,
                "hypothesis_fragility": self.fragility,
                "ranked": [{"arm": k, "text": v["text"],
                            "persona": v.get("persona"),
                            "funnel_objective": (v.get("funnel") or {}).get("objective_mean"),
                            "funnel_interval80": (v.get("funnel") or {}).get("interval80"),
                            "stage_trace": (v.get("funnel") or {}).get("stage_trace"),
                            "contract": v.get("contract"), "cold_read": v.get("cold_read"),
                            "critic": v.get("critic"), "gates": v.get("gates"),
                            "origin": v.get("origin")} for k, v in ranked],
                "optimal_strategy_spec": (self.spec.summary() if self.spec is not None else None),
                "honesty": self.honesty}


def optimize_cold_outreach(recipient: RecipientState, *, sender_brief, chat_fn=None,
                           recipient_notes: str = "", k_drafts: int = 8, n_mc: int = 400,
                           seed: int = 0, include_legacy_beam: bool = False,
                           dossier=None, hypotheses=None, persona_draws: int = 3,
                           persona_top_k: int = 5, arrival_context: str = "",
                           outcome_utilities=None, method: str = "reply_first",
                           trace_path: str = None) -> ColdOutreachResult:
    """The corrected cold-outreach path (the failed Thiel output is this function's regression case).

    What changed vs optimize_message and why:
      OBJECTIVE  — the conjunctive response FUNNEL with valenced outcomes (P(positive) − λ·P(negative)),
                   not an additive logit where one maxed lever buys back a failed gate, and not
                   'any reply counts'.
      CONTENT    — a deterministic cold-outreach CONTRACT (identity / thesis / evidence-with-
                   provenance / relevance / tiny next step): drafts missing an element are rejected
                   before scoring. Style gates cannot supply missing content.
      GENERATION — contract-constrained WHOLE drafts (global coherence; the slot-beam is myopic and
                   produced mid-conversation openers), still selected by the world model, never by
                   the LLM. The deterministic plain-human baseline is always in the slate: if the
                   machinery cannot beat the plain draft under its own evaluator, the plain draft
                   wins and that is reported.
      GATES      — numeric fact guard + four-axis register critic + the COLD-READ critic (a busy
                   stranger's five-second read: who/why/believable/bait/effort/next-step). Critics
                   gate and diagnose; the funnel ranks. An uncalibrated judge is never the objective.
      BEHAVIOR   — when live, the primary evaluator is the QUALITATIVE PERSONA ENSEMBLE
                   (persona_response.py): the recipient — rendered as a qualitative dossier, never
                   invented numeric traits — reads each finalist under COMPETING inbox-reality
                   hypotheses and chooses a categorical outcome; probabilities come from counting
                   those choices. This replaces the circular loop of scoring messages against
                   trait numbers an LLM invented. The funnel remains the structural prior/offline
                   ranking and the stage diagnosis.
      OUTPUT     — a ranked slate with intervals, per-hypothesis fragility, and 'within noise'
                   honesty: 'best-supported among tested', never 'best possible'."""
    from swm.decision.llm_moves import (allowed_numbers, llm_cold_read_critic, llm_draft_proposer,
                                        llm_message_encoder, llm_rewriter, llm_sentence_judge)
    from swm.decision.mc_evaluation import mc_evaluate_funnel
    from swm.decision.outreach_contract import plain_baseline_draft, validate
    from swm.decision.response_funnel import funnel_scorer_from_recipient
    from swm.decision.situational_levers import generate_levers

    # DEFAULT PATH — the reply-first beat planner (method="reply_first"): design the message around
    # the exact reply wanted, search beats before sentences, certify with three SEPARATED judges
    # (truth / human-language / blind outcome), return ONE message with no simulated percentages.
    # method="slate" keeps the previous generate-and-rank path for comparisons.
    if method == "reply_first" and dossier is not None:
        from swm.decision.reply_first import ReplyFirstPlanner
        planner = ReplyFirstPlanner(chat_fn, sender_brief=sender_brief, dossier=dossier,
                                    hypotheses=hypotheses, recipient_notes=recipient_notes,
                                    seed=seed, trace_path=trace_path,
                                    persona_draws=persona_draws)
        pr = planner.run()
        cands = {"reply_first_winner": {"text": pr.winner_text,
                                        "origin": pr.winner_origin,
                                        "gates": next((f.get("gates") for f in pr.finalists
                                                       if f.get("ordinal_note") == "selected"), {}),
                                        "funnel": {}}}
        for f in pr.finalists:
            if f.get("ordinal_note") != "selected":
                cands[f"finalist_{f['label']}"] = {"text": f["text"], "gates": f.get("gates"),
                                                   "funnel": {}}
        return ColdOutreachResult(recipient=recipient.label, spec=None, candidates=cands,
                                  winner="reply_first_winner",
                                  within_noise=[k for k in cands if k != "reply_first_winner"],
                                  honesty=pr.label, fragility={"n_llm_calls": pr.n_llm_calls})

    levers = []
    encode_fn = encode_text_to_strategy
    judge_fn = rewrite_fn = cold_read = None
    if chat_fn is not None:
        levers = generate_levers(chat_fn, recipient.label, recipient.vars, evidence=recipient_notes)
        encode_fn = llm_message_encoder(chat_fn, levers=levers)
        judge_fn = llm_sentence_judge(chat_fn, facts_text=sender_brief.to_prompt())
        rewrite_fn = llm_rewriter(chat_fn, recipient_notes=recipient_notes, sender=sender_brief)
        cold_read = llm_cold_read_critic(chat_fn, recipient_notes=recipient_notes,
                                         facts_text=sender_brief.to_prompt())

    fact_numbers = allowed_numbers(sender_brief.to_prompt(), recipient_notes)
    final_critic = SemanticCritic(judge_fn=judge_fn, allowed_numbers=fact_numbers)
    scorer = funnel_scorer_from_recipient(recipient.vars, recipient.base_mean, seed=seed,
                                          levers=levers)

    # L1 — optimal strategy under the FUNNEL objective (drives instruction translation)
    spec = optimize_strategy(scorer, q=0.2, restarts=10, seed=seed)

    # candidate slate: plain baseline (always) + contract-constrained LLM drafts + optional legacy beam
    texts = {"plain_baseline": plain_baseline_draft(sender_brief, recipient.label)}
    if chat_fn is not None:
        proposer = llm_draft_proposer(chat_fn, recipient_notes=recipient_notes, sender=sender_brief,
                                      levers=levers)
        for i, d in enumerate(proposer(spec.strategy, k=k_drafts)):
            texts[f"draft_{i}"] = d
        texts["_draft_rejects"] = proposer.last_rejected
    if include_legacy_beam:
        from swm.decision.compositional_search import bank_proposer, construct_email
        legacy = construct_email(scorer, spec.strategy,
                                 proposer=bank_proposer(sender_brief, recipient.label),
                                 beam=4, critic=SemanticCritic(allowed_numbers=fact_numbers),
                                 encode_fn=encode_text_to_strategy)
        texts["legacy_slot_beam"] = legacy.text

    rejects = texts.pop("_draft_rejects", [])
    candidates = {}
    from swm.decision.compositional_search import _prune_flagged_sentences
    for name, text in texts.items():
        # repair: targeted rewrite of flagged sentences, then sentence-level prune (never ship a flag)
        crit = final_critic.critique(text)
        if rewrite_fn is not None and crit.flags():
            for fl in crit.flags():
                fixed = rewrite_fn(fl["sentence"], fl["reasons"], spec.strategy)
                if fixed != fl["sentence"]:
                    text = text.replace(fl["sentence"], fixed).strip()
            text = " ".join(text.split())
        text = _prune_flagged_sentences(text, final_critic)
        cv = validate(text, sender_brief)
        if not cv.ok and name != "plain_baseline":
            continue                                     # repair broke the contract -> drop the arm
        strat = encode_fn(text)
        fmc = mc_evaluate_funnel(recipient.vars, recipient.base_mean, strat,
                                 base_n_effective=recipient.base_n_effective,
                                 confidences=recipient.confidences, n_samples=n_mc,
                                 seed=seed, levers=levers)
        entry = {"text": text, "strategy": {k: round(v, 3) for k, v in strat.items()},
                 "funnel": fmc.summary(), "contract": cv.as_dict(),
                 "critic": final_critic.critique(text).summary()}
        if cold_read is not None:
            entry["cold_read"] = cold_read(text)
        candidates[name] = entry

    # rank pass 1 (cheap): cold-read gate failures demoted below every gate-passer; ties by funnel q20
    def sort_key(kv):
        e = kv[1]
        gate_ok = e.get("cold_read", {}).get("gates_ok", True)
        return (0 if gate_ok else 1, -e["funnel"]["interval80"][0], -e["funnel"]["objective_mean"])

    ranked = sorted(candidates.items(), key=sort_key)
    fragility = {}

    # rank pass 2 (behavioral, live only): the persona ensemble reads the top gate-passing finalists
    # under COMPETING inbox-reality hypotheses and CHOOSES an outcome per draw — the primary ranking
    # signal, with per-hypothesis fragility surfaced (a one-hypothesis winner is never confident)
    if chat_fn is not None and dossier is not None and ranked:
        from swm.decision.persona_response import ensemble_evaluate, fragility_report
        hyps = hypotheses or []
        if not hyps:
            from swm.decision.persona_response import specialize_hypotheses
            hyps = specialize_hypotheses(chat_fn, dossier)
        finalists = [k for k, _ in ranked[:persona_top_k]]
        persona_results = {}
        for name in finalists:
            pr = ensemble_evaluate(chat_fn, dossier, hyps, candidates[name]["text"],
                                   arrival_context=arrival_context,
                                   draws_per_hypothesis=persona_draws)
            persona_results[name] = pr
            candidates[name]["persona"] = pr.summary(outcome_utilities)
        fragility = fragility_report(persona_results, outcome_utilities)
        gate_ok = {k for k, e in candidates.items()
                   if e.get("cold_read", {}).get("gates_ok", True)}
        order = [a for a in fragility.get("overall_utility", {}) if a in gate_ok] or finalists
        winner = order[0] if order else (ranked[0][0] if ranked else None)
        within = list(fragility.get("within_noise_of_winner", []))
    else:
        winner = ranked[0][0] if ranked else None
        win_lo = candidates[winner]["funnel"]["interval80"][0] if winner else 0
        within = [k for k, e in ranked[1:]
                  if e["funnel"]["interval80"][1] >= win_lo
                  and e.get("cold_read", {}).get("gates_ok", True)]

    honesty = ("Winner = BEST-SUPPORTED AMONG TESTED, not 'best possible'. Live ranking is the "
               "qualitative persona ensemble (LLM role-play under competing inbox-reality "
               "hypotheses; outcomes are counted choices) — a model-based judgment, UNCALIBRATED "
               "against real outreach outcomes. The funnel is a structural prior; its absolute "
               "levels are claims. Arms in within_noise_of_winner are indistinguishable at this "
               "draw count, and a hypothesis-fragile winner is flagged, not trusted. "
               f"{len(rejects)} generated drafts were rejected by the contract/fact guard.")
    return ColdOutreachResult(recipient=recipient.label, spec=spec, candidates=candidates,
                              winner=winner, within_noise=within, honesty=honesty,
                              fragility=fragility)


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
