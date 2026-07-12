"""WorldModelV2 best-action layer, run on a real question — no hardcoded recipient variables.

    QUESTION (natural language): what should REALER ESTATE email ambassadors@hpd.nyc.gov to become a
    Housing Ambassador, given what the Housing Connect program actually wants?

This drives the UNIVERSAL v2 machinery only (swm/world_model_v2/*), the interventional twin of a forecast:

  1. ACTION GENERATION      the LLM proposes N candidate emails (diverse strategies) from the brief + the
                            program facts. The system generates the action set; I do not hand-write options.
  2. INTERPRETATION         for each candidate the LLM PLAYS THE RECIPIENT (actor_cognition.interpret) and
                            returns a TYPED reading (urgency/obligation/relevance/benefit/effort/…). The
                            simulated agent's variables are INFERRED BY THE LLM FROM THE EMAIL TEXT — nothing
                            is hardcoded. This is "the LLM gives the simulated actor its context."
  3. HIDDEN STATE           the recipient's attention / responsiveness / obligation-sensitivity are sampled
                            as DISTRIBUTIONS per particle (actor_cognition.hidden_state_latents) — never a
                            fabricated certainty.
  4. MATCHED COUNTERFACTUAL each candidate email is a typed Intervention; WorldModelV2Run.evaluate_interventions
                            clones the SAME sampled worlds per candidate, injects the email at t0, rolls the
                            event-driven reply decision forward at real calendar time to a 14-day horizon, and
                            reads the reply from TERMINAL STATE. Same exogenous seeds isolate the email's effect
                            from world luck -> expected P(reply), P(best), expected regret, vs do-nothing.

Honesty: the engagement policy that maps the interpretation to a reply propensity is a LABELED WORLD-KNOWLEDGE
PRIOR (v2 'typed_action_policy' in uncalibrated/prior mode — there is no fitted HPD reply corpus). Trust the
RANKING and the lever directions; the absolute P(reply) is a claim to check, exactly as the repo stamps its
unvalidated priors.
"""
from __future__ import annotations

import json
import math
import random

from swm.api.deepseek_backend import default_chat_fn
from swm.engine.grounding import parse_json
from swm.world_model_v2.actor_cognition import (FEATURE_DIMS, Interpretation, hidden_state_latents,
                                                interpret)
from swm.world_model_v2.contracts import ActionSpace, Intervention, OutcomeContract, UtilityFunction
from swm.world_model_v2.events import (Event, EventQueue, StochasticHazard, register_event_type)
from swm.world_model_v2.init_state import InitialStateModel
from swm.world_model_v2.rollout import WorldModelV2Run
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta, TransitionOperator, TransitionProposal

DAY = 86400.0
T0 = 1_752_000_000.0            # a fixed as-of epoch (calendar time is real; the absolute value is arbitrary)
HORIZON_DAYS = 14.0

# ---- the natural-language brief (what the user told me — the sender's REAL, true facts) ----------------
SENDER_BRIEF = """SENDER: Beckett, founder of REALER ESTATE (www.realerestate.org), an affordable-housing
organization. TRUE FACTS the email may use and must not exceed:
- REALER ESTATE has a physical, in-person location with office hours where people can come for help.
- It is very tech-forward and can run online webinars for applicants (reach many people at once).
- It helps people apply for and navigate Housing Connect.
- Beckett submitted the Housing Ambassador questionnaire about 1-2 weeks ago and has had no reply yet.
GOAL: become an enrolled Housing Ambassador; the email should prompt a reply with next steps."""

# ---- the RECIPIENT context: public facts about the program (from HPD's own page/questionnaire) ---------
# This is context the LLM reads AS the recipient; it is public program description, not invented psychology.
RECIPIENT_CONTEXT = """You are the team at NYC HPD that runs the Housing Ambassador program
(ambassadors@hpd.nyc.gov). Housing Ambassadors are community-based partners who help people prepare and apply
for Housing Connect affordable-housing lotteries, primarily on a VOLUNTEER basis. The program is supported by
Citi Community Development. You actively RECRUIT new partner orgs ("Become a Housing Ambassador — email us").
Your goal is to expand reach: more applicants helped, across more boroughs and languages.

Before you enroll an org you need to know: (a) do they have a physical location / office hours / events where
people can get help; (b) is it wheelchair accessible and what disability-access features exist; (c) what
languages other than English they serve; (d) will they provide up-to-date HPD/HDC lottery materials and relay
questions/feedback to HPD; (e) can they send >=1 representative per year to HPD's affordable-housing lottery &
Marketing training; (f) are they willing to be publicized on nyc.gov/housing-ambassadors and in 311's
directory. Ambassadors do NOT provide housing directly, cannot guarantee units, and cannot charge fees.
You are a mission-driven public office, likely thinly staffed. What earns a reply: clear mission-fit, concrete
answers to those enrollment requirements, low-friction next steps, and evidence of a legitimate free service."""

N_CANDIDATES = 5
N_PARTICLES = 800

# LABELED WORLD-KNOWLEDGE PRIOR policy: interpretation dims -> engagement log-odds shift (centered at neutral).
# Signs are world knowledge (higher relevance/benefit/obligation/ownership -> more reply; more effort or
# needing-clarification -> less immediate reply). Magnitudes are a coarse prior (unvalidated), NOT fitted.
_PRIOR_W = {"relevance_to_goals": 1.4, "benefit_of_action": 1.2, "obligation": 0.9, "task_ownership": 0.7,
            "risk_of_inaction": 0.5, "effort_required": -1.0, "needs_clarification": -0.7,
            "urgency": 0.3, "thread_continuity": 0.4}
_NEUTRAL = {d: (0.0 if d in ("needs_clarification", "needs_delegation", "thread_continuity") else 0.5)
            for d in FEATURE_DIMS}


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def p_engage_prior(interp: Interpretation, base_rate: float) -> float:
    """v2 typed_action_policy in PRIOR mode: logit(base_rate) + Σ prior_w · (dim − neutral)."""
    z = _logit(base_rate)
    for d, w in _PRIOR_W.items():
        z += w * (getattr(interp, d) - _NEUTRAL[d])
    return min(0.97, max(0.01, _sigmoid(z)))


# ---- LLM step 1: infer the program's base reply rate to a well-formed enrollment email (labeled prior) --
def infer_base_rate(chat_fn) -> tuple[float, str]:
    prompt = (RECIPIENT_CONTEXT + "\n\nQUESTION: of well-formed, complete enrollment emails from legitimate "
              "orgs, what fraction do you realistically reply to within 14 days? This is a PRIOR estimate. "
              'Return ONLY JSON: {"base_rate": <0..1>, "why": "<8 words>"}')
    r = parse_json(chat_fn(prompt)) or {}
    try:
        return min(0.9, max(0.05, float(r.get("base_rate", 0.35)))), str(r.get("why", ""))[:80]
    except (TypeError, ValueError):
        return 0.35, "fallback prior"


# ---- LLM step 2: GENERATE candidate emails (the action set) --------------------------------------------
_STRATEGIES = [
    ("minimal_followup", "short polite status-check follow-up"),
    ("answers_every_requirement", "proactively answer every enrollment requirement"),
    ("tech_reach_lead", "lead with tech-forward webinar reach"),
    ("warm_mission_fit", "warm mission-fit, why we help"),
    ("low_friction_ask", "very short low-friction single ask"),
]


def propose_emails(gen_fn, n: int) -> list[dict]:
    """Generate the action set ONE email per call (robust) — each a distinct strategy the engine will score."""
    out = []
    for slug, strat in _STRATEGIES[:n]:
        prompt = f"""Write ONE real, sendable email. Do not invent facts beyond those given.

{SENDER_BRIEF}

RECIPIENT you are writing TO:
{RECIPIENT_CONTEXT}

STRATEGY for THIS email: {strat}.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""
        r = parse_json(gen_fn(prompt)) or {}
        if r.get("body"):
            out.append({"id": slug, "strategy": strat, "subject": str(r.get("subject", "")),
                        "body": str(r.get("body", ""))})
    return out


# ---- the v2 reply-decision operator (event-driven, hazard-integrated, prior-parameterized) -------------
register_event_type("message_delivered", scheduling="scheduled", validated=False)
register_event_type("reply_check", scheduling="hazard", validated=False)


class ReplyDecision(TransitionOperator):
    """At each inbox-check opportunity the recipient replies with a per-opportunity hazard integrated so that
    P(reply by horizon) ≈ p_target, modulated by the PARTICLE'S sampled attention × responsiveness. p_target
    (this email's engagement propensity from the LLM interpretation) is carried on the world as a quantity."""
    name = "reply_decision"
    check_rate_per_day = 1.0                      # prior inbox-check rate (labeled prior)

    def applicable(self, world, event):
        return (event.etype == "reply_check" and "msg_p_target" in world.quantities
                and world.entity("recipient").value("current_action") is None)

    def propose(self, world, event, rng):
        p_target = world.quantities["msg_p_target"].value
        att = float(world.entity("recipient").value("attention") or 0.7)
        resp = float(world.entity("recipient").value("responsiveness") or 1.0)
        p_eff = min(0.97, max(0.005, p_target * (0.4 + 0.85 * att) * resp))
        n_opp = max(1.0, self.check_rate_per_day * HORIZON_DAYS)
        h = 1.0 - (1.0 - p_eff) ** (1.0 / n_opp)          # hazard integration: 1-(1-h)^n_opp = p_eff
        act = "reply" if rng.random() < h else "wait"
        return TransitionProposal(operator=self.name, action={"actor": "recipient", "type": act},
                                  p_dist={"reply": round(h, 4), "wait": round(1 - h, 4)},
                                  reason_codes=[f"att={att:.2f}", f"resp={resp:.2f}", "hazard_integrated"])

    def apply(self, world, proposal):
        d = StateDelta(at=world.clock.now, event_type="reply_decision", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty={"p_dist": proposal.p_dist})
        if proposal.action["type"] == "reply":
            before = world.entity("recipient").value("current_action")
            world.entity("recipient").set("current_action", F("reply", status="derived", method=self.name,
                                                               updated_at=world.clock.now))
            d.change("recipient.current_action", before, "reply")
        return d


class _Q:                                          # a minimal terminal-quantity carrier (as in enron.py)
    def __init__(self, v):
        self.value = v


def build_run(base_rate: float) -> WorldModelV2Run:
    clock = SimulationClock(now=T0, as_of=T0)
    base = WorldState(world_id="hpd", branch_id="root", clock=clock)
    rcpt = Entity(identity="recipient", entity_type="person")
    rcpt.set("attention", F(0.7, status="assumed"))
    rcpt.set("current_action", F(None, status="assumed"))
    rcpt.set("responsiveness", F(1.0, status="assumed"))
    base.entities["recipient"] = rcpt
    # hidden state sampled per particle (attention, responsiveness, obligation_sensitivity) — distributions
    latents, correlations = hidden_state_latents("recipient", workload_norm=0.5, hetero_sd=0.3)
    init = InitialStateModel(base_world=base, latents=latents, correlations=correlations)
    contract = OutcomeContract(
        family="response_occurrence", options=["reply", "no_reply"],
        resolution_rule=f"program replies within {HORIZON_DAYS:.0f}d",
        readout=lambda w: "reply" if w.entity("recipient").value("current_action") == "reply" else "no_reply",
        horizon_ts=T0 + HORIZON_DAYS * DAY).validate()

    def queue_builder(world):
        return EventQueue(horizon_ts=T0 + HORIZON_DAYS * DAY)     # empty; interventions inject the email

    return WorldModelV2Run(initial=init, queue_builder=queue_builder, operators=[ReplyDecision()],
                           contract=contract, n_particles=N_PARTICLES)


def make_intervention(cand: dict, p_target: float) -> Intervention:
    def apply(world, queue):
        world.quantities["msg_p_target"] = _Q(p_target)          # this email's engagement propensity
        queue.schedule(Event(ts=T0 + 60.0, etype="message_delivered", participants=["recipient"]))
        rng = random.Random(hash(world.branch_id) & 0xFFFF)
        queue.add_hazard(StochasticHazard(etype="reply_check", rate_per_day=ReplyDecision.check_rate_per_day,
                                          participants=["recipient"]), now=T0, rng=rng, world=world)
    return Intervention(intervention_id=cand["id"], description=cand["strategy"], apply=apply, kind="artifact")


def main():
    chat_fn = default_chat_fn(max_tokens=400, temperature=0.2)          # interpretation / base rate
    gen_fn = default_chat_fn(max_tokens=900, temperature=0.4)           # email generation (longer output)
    if chat_fn is None:
        raise SystemExit("no LLM backend key set — this run requires the live model for interpretation")
    print("backend: DeepSeek (live)\n")

    base_rate, br_why = infer_base_rate(chat_fn)
    print(f"[0] recipient base reply rate (LLM prior): {base_rate:.2f}  — \"{br_why}\"")

    cands = propose_emails(gen_fn, N_CANDIDATES)
    print(f"[1] ACTION SET: LLM generated {len(cands)} candidate emails")
    for c in cands:
        print(f"      - {c['id']:<22} strategy: {c['strategy']}")

    print("\n[2] INTERPRETATION: the LLM reads each email AS the HPD program and returns typed dims")
    interps, targets = {}, {}
    for c in cands:
        it = interpret(chat_fn, actor="the NYC HPD Housing Ambassador program team",
                       channel="program enrollment inbox (ambassadors@hpd.nyc.gov)",
                       context=RECIPIENT_CONTEXT, content=f"SUBJECT: {c['subject']}\n\n{c['body']}")
        if it is None:
            it = Interpretation()          # abstain -> neutral reading (falls back to base rate)
        interps[c["id"]] = it
        targets[c["id"]] = p_engage_prior(it, base_rate)
        print(f"      - {c['id']:<22} intent={it.intent:<18} relevance={it.relevance_to_goals:.2f} "
              f"benefit={it.benefit_of_action:.2f} effort={it.effort_required:.2f} "
              f"clarify={it.needs_clarification:.2f}  -> p_target={targets[c['id']]:.3f}")

    print("\n[3+4] MATCHED-COUNTERFACTUAL ROLLOUT (v2 evaluate_interventions, same worlds per candidate)")
    run = build_run(base_rate)
    space = ActionSpace(interventions=[make_intervention(c, targets[c["id"]]) for c in cands])
    utility = UtilityFunction(name="reply",
                              fn=lambda w: 1.0 if w.entity("recipient").value("current_action") == "reply"
                              else 0.0)
    report = run.evaluate_interventions(space, utility, seed=1)

    print(f"      rolled {report['n_matched_worlds']} matched worlds x {len(space.interventions)} arms "
          f"(incl. do-nothing); readout={report['readout']}\n")
    print(f"      {'candidate':<24}{'P(reply)':>10}{'downside_p10':>14}{'P(best)':>10}{'regret':>10}")
    for row in report["ranking"]:
        print(f"      {row['intervention']:<24}{row['expected_utility']:>10.3f}"
              f"{row['downside_p10']:>14.3f}{row['p_best']:>10.3f}{row['expected_regret']:>10.4f}")

    best_id = report["best"]
    best = next(c for c in cands if c["id"] == best_id)
    none_row = next(r for r in report["ranking"] if r["intervention"] == "none")
    best_row = next(r for r in report["ranking"] if r["intervention"] == best_id)
    print(f"\n[BEST ACTION] {best_id}  (strategy: {best['strategy']})")
    print(f"   P(reply) {best_row['expected_utility']:.3f}  vs do-nothing {none_row['expected_utility']:.3f}  "
          f"(+{best_row['expected_utility'] - none_row['expected_utility']:.3f})   P(best among candidates) "
          f"{best_row['p_best']:.3f}")
    print("\n" + "=" * 90)
    print(f"SUBJECT: {best['subject']}")
    print("=" * 90)
    print(best["body"])
    print("=" * 90)


if __name__ == "__main__":
    main()
