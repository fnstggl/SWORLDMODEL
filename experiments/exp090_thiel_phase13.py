"""EXP-090 — the Thiel cold-email decision, end to end through the UPGRADED stack.

Two layers compose, and both are the production path:

  A. MESSAGE OPTIMIZER (L1→L4): public-figure inference → L1 optimal strategy (variable space)
     → L2 beam-constructed email (LLM proposes single MOVES; deterministic numeric fact-guard
     rejects fabricated/reused numbers BEFORE scoring) → L4 four-axis LLM critic gate
     (coherent / annoying / AI-sounding / fabricated-vs-facts; strict fail-closed parsing) with
     targeted rewrite repair → L3 Monte-Carlo reply evaluation under the recipient's hidden state,
     using elasticities FITTED AND GRADED on 19,714 real CMV persuasion outcomes (grade A held-out
     calibration; persuasion→cold-email transport is an assumption and is stamped).

  B. PHASE 13 CANONICAL DECISION LAYER: the constructed email and its naive contrasts become typed
     ACTIONS in a DecisionProblem; each action is a canonical intervention (decision_action event →
     message_delivered follow-up); a ReplyMechanismOperator turns the graded scorer into the world's
     reply mechanism; MatchedEvaluator rolls all arms over the SAME posterior particles (Thiel's
     hidden disposition + a gatekeeper/base/engaged structural hypothesis over the inbox process)
     with stream-partitioned CRN; the DecisionResult reports paired lift vs do_nothing, CVaR,
     regret, by-hypothesis fragility; the frozen decision enters the prospective ledger.

Sender facts are VERBATIM from https://www.runaurelius.com (fetched 2026-07-17) — the fact guard
holds every number in the email to this list. Recipient evidence: recorded public-profile snippets
+ a live-fetched Thiel Fellowship line (provenance stamped below).

Run:  PYTHONPATH=. python experiments/exp090_thiel_phase13.py            (live LLM)
      PYTHONPATH=. python experiments/exp090_thiel_phase13.py --offline  (bank + lexical critic)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.elasticity_fit import load_fitted
from swm.decision.llm_moves import SenderBrief
from swm.decision.message_pipeline import optimize_for_world
from swm.decision.strategy_scorer import scorer_from_recipient
from swm.entities.public_figure import PublicFigureResolver
from swm.ingestion.store import EventStore
from swm.worlds.world import World

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13", "thiel_run")
FIT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "phase13",
                        "message_calibration", "cmv_fit.json")

# ---------------------------------------------------------------- sender facts (verbatim, sourced)
FACTS_SOURCE = {"url": "https://www.runaurelius.com", "fetched": "2026-07-17",
                "note": "every number below appears on the page; the fact guard rejects any other "
                        "number the proposer tries to write"}

AURELIUS_SENDER = SenderBrief(
    sender="Beckett",
    thesis="AI scheduling treats electricity as a fixed cost; forecasting power constraints and "
           "scheduling against them is the cheapest capacity there is",
    ask="a one-line reaction on whether the thesis is wrong",
    facts=[
        "17 years old, starting Princeton in the fall",
        "building Aurelius (runaurelius.com): constraint-aware orchestration for AI infrastructure",
        "a predictive world model forecasts power constraints, simulates candidate scheduling "
        "decisions, and ranks them by economic outcome before execution",
        "+724% average SLA-safe goodput per dollar vs the production scheduler, in simulated replay "
        "of ~1.5M requests of public production traces",
        "per-grid mean improvements: +698% on PJM, +718% on ERCOT, +755% on CAISO",
        "-84% GPU-hours in the same replay",
        "results are simulated replay of public traces, not a production deployment",
        "offering read-only shadow mode to a small batch of infrastructure operators",
    ],
)

# recorded public-profile evidence; the last snippet is live-fetched (provenance in title)
_EVIDENCE = [
    {"title": "The Thiel Fellowship pays people to skip college",
     "snippet": "backs young founders who drop out; discovered founders via cold email; took a meeting "
                "with a teenager who had an unusual, contrarian thesis"},
    {"title": "Peter Thiel, the contrarian",
     "snippet": "provocative, heterodox, iconoclast; a skeptic who challenges consensus; famously "
                "skeptical of elite university prestige and the higher-ed status game"},
    {"title": "How to actually reach Peter Thiel",
     "snippet": "hard to reach, screens heavily, rarely responds to cold outreach unless the pitch is "
                "genuinely contrarian and specific"},
    {"title": "Founders Fund and definite optimism",
     "snippet": "billionaire investor and chairman; looks for definite, specific plans and secrets; "
                "dislikes generic status-seeking outreach"},
    {"title": "thielfellowship.org (live fetch 2026-07-17)",
     "snippet": "gives $250,000 and mentorship to young people to skip or stop out of college; "
                "'College can be good for learning about what's been done before, but it can also "
                "discourage you from doing something new'"},
]


# ---------------------------------------------------------------- traced chat_fn (full audit trail)
class TracingChat:
    """Wraps the backend chat_fn: records EVERY call (classified seam, full prompt, full response,
    latency, per-call kwargs) to a JSONL trail — the exp090 under-the-hood artifact."""
    _SEAMS = (("Write ", "proposer"), ("You are a ruthless editor", "judge"),
              ("You score a cold message", "encoder"), ("You are calibrating", "levers"),
              ("Rewrite this ONE line", "rewriter"))

    def __init__(self, fn, path: str):
        self.fn = fn
        self.path = path
        self.n = 0
        self.default_max_tokens = getattr(fn, "default_max_tokens", None)
        self.default_temperature = getattr(fn, "default_temperature", None)

    def __call__(self, prompt: str, **kw) -> str:
        t0 = time.time()
        out = self.fn(prompt, **kw) if kw else self.fn(prompt)
        seam = next((s for pre, s in self._SEAMS if prompt.lstrip().startswith(pre)), "other")
        self.n += 1
        with open(self.path, "a") as f:
            f.write(json.dumps({"call": self.n, "seam": seam, "kwargs": kw,
                                "wall_s": round(time.time() - t0, 2),
                                "prompt_sha16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                                "prompt": prompt, "response": out}) + "\n")
        return out


# ---------------------------------------------------------------- Phase 13 decision world
def build_decision_context(profile_vars: dict, confidences: dict, base_mean: float,
                           base_n_eff: float, fitted, arm_strategies: dict, levers=None):
    """The canonical runtime pieces for the send-decision world: Beckett + Thiel entities, Thiel's
    hidden disposition as DECLARED LATENTS (sampled per particle), the reply mechanism as a
    registered operator, and a replied/... quantity the outcome contract reads."""
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.events import EventQueue, StochasticHazard, register_event_type
    from swm.world_model_v2.init_state import InitialStateModel, LatentVariableRecord
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
    from swm.world_model_v2.transitions import (StateDelta, TransitionOperator,
                                                TransitionProposal, ValidationResult)

    from swm.world_model_v2.state import parse_time
    T0 = parse_time("2026-09-01T09:00:00Z")           # the contract's as_of — times must AGREE with
    DAY = 86400.0                                     # the DecisionProblem or feasibility rejects
    register_quantity_type("replied", units="indicator")
    register_event_type("news_cycle", scheduling="hazard", validated=True,
                        participants="recipient — an exogenous attention shock (no direct effect; "
                                     "it exists so arms share a nontrivial exogenous trace for the "
                                     "CRN pairing check)")

    base = WorldState(world_id="thiel_outreach", branch_id="root",
                      clock=SimulationClock(now=T0, as_of=T0))
    beckett = Entity(identity="beckett")
    beckett.set("resources", F(1.0, status="observed"), key="cold_shot")
    thiel = Entity(identity="peter_thiel")
    base.entities["beckett"] = beckett
    base.entities["peter_thiel"] = thiel
    base.quantities["replied"] = Quantity(name="replied", qtype="replied", value=0.0, timestamp=T0)

    latents = [LatentVariableRecord(
        path="peter_thiel.latent_state[base_responsiveness]",
        candidates={"mean": base_mean, "sd": max(0.03, base_mean * (1 - base_mean) /
                                                 max(1.0, base_n_eff) ** 0.5), "lo": 0.005, "hi": 0.6},
        method="dataset", confidence=0.5)]
    for var, val in profile_vars.items():
        sd = 0.18 * (1.0 - float(confidences.get(var, 0.5)))
        latents.append(LatentVariableRecord(
            path=f"peter_thiel.latent_state[{var}]",
            candidates={"mean": float(val), "sd": max(0.02, sd), "lo": 0.0, "hi": 1.0},
            method="llm", confidence=confidences.get(var, 0.5)))

    class ReplyMechanismOperator(TransitionOperator):
        """The graded reply model AS the world's mechanism: on message_delivered, read the particle's
        sampled disposition, score the delivered message's encoded strategy with the FITTED weights,
        apply the structural-hypothesis inbox multiplier, add the CRN mood shock, draw the reply."""
        name = "reply_mechanism"

        def applicable(self, world, event):
            return event.etype == "message_delivered" and \
                "peter_thiel" in (event.participants or [])

        def propose(self, world, event, rng):
            return TransitionProposal(operator=self.name, action={}, reason_codes=["reply_draw"])

        def apply(self, world, proposal):
            return None

        def run(self, world, event, rng):
            import math
            aid = (event.payload or {}).get("from_decision_action", "")
            strat = arm_strategies.get(aid)
            d = StateDelta(at=world.clock.now, event_type="message_delivered", operator=self.name)
            if strat is None:
                return d, ValidationResult(ok=True)
            ent = world.entities["peter_thiel"]

            def lat(name, default):
                f = ent.get("latent_state", key=name)
                return float(f.value) if f is not None and f.value is not None else default

            rvars = {v: lat(v, profile_vars.get(v, 0.5)) for v in profile_vars}
            rvars.setdefault("platform_response_norm", 0.30)
            rvars.setdefault("relationship_strength", 0.0)
            base_resp = lat("base_responsiveness", base_mean)
            scorer = scorer_from_recipient(rvars, base_resp, seed=0,
                                           weights=fitted.weights if fitted else None,
                                           grade=fitted.grade if fitted else None, levers=levers)
            p = scorer.mean(strat)
            hyp = str((world.uncertainty_meta.get("model") or {}).get("hypothesis", "H_base"))
            mult = {"H_gatekeeper": 0.4, "H_base": 1.0, "H_engaged": 1.6}.get(hyp, 1.0)
            # mood: an aleatory day-quality shock shared across arms through the CRN op stream
            mood = rng.gauss(0.0, 0.25)
            logit = math.log(max(p, 1e-6) / max(1 - p, 1e-6)) + mood + math.log(mult)
            p_final = 1.0 / (1.0 + math.exp(-logit))
            replied = 1.0 if rng.random() < p_final else 0.0
            q = world.quantities["replied"]
            before = float(q.value or 0.0)
            if replied:
                q.value = 1.0
                q.timestamp = world.clock.now
                d.change("quantities.replied", before, 1.0)
            d.uncertainty["p_reply"] = round(p_final, 4)
            d.uncertainty["hypothesis"] = hyp
            return d, ValidationResult(ok=True)

    init = InitialStateModel(base_world=base, latents=latents)

    def qb(world):
        q = EventQueue(horizon_ts=T0 + 14 * DAY)
        q.add_hazard(StochasticHazard(etype="news_cycle", rate_per_day=0.2,
                                      participants=["peter_thiel"]),
                     now=world.clock.now, rng=__import__("random").Random(0), world=world)
        return q

    contract = OutcomeContract(family="binary",
                               readout=lambda w: float(w.quantities["replied"].value or 0.0),
                               horizon_ts=T0 + 14 * DAY)
    hypotheses = [{"id": "H_gatekeeper", "prior": 0.35,
                   "note": "assistant screens the inbox; only forwarded mail is read (x0.4)"},
                  {"id": "H_base", "prior": 0.50, "note": "profile-inferred responsiveness as-is"},
                  {"id": "H_engaged", "prior": 0.15,
                   "note": "reads own inbound in bursts; contrarian hooks over-perform (x1.6)"}]
    return {"initial": init, "queue_builder": qb, "operators": [ReplyMechanismOperator()],
            "contract": contract, "n_particles": 400, "hypotheses": hypotheses, "T0": T0}


def run_phase13_decision(result, encode_fn, fitted, profile_vars, confidences, base_mean,
                         base_n_eff, levers, seed: int = 90):
    """Wrap the optimizer's finalist + contrasts as typed actions and evaluate them through the
    canonical Phase 13 funnel with matched CRN. Returns (DecisionResult, arm texts)."""
    from swm.world_model_v2.phase13.api import evaluate_actions
    from swm.world_model_v2.phase13.contracts import (DecisionProblem, RiskSpec, Stakeholder,
                                                      UtilitySpec)
    from swm.world_model_v2.phase13.ledger import DecisionLedger
    from swm.world_model_v2.phase13.ontology import ActionSchema

    arms = {"send_optimized": result.email.text}
    for label, b in result.baselines.items():
        arms[f"send_{label}"] = b["text"]
    arm_strategies = {aid: encode_fn(text) for aid, text in arms.items()}

    ctx = build_decision_context(profile_vars, confidences, base_mean, base_n_eff, fitted,
                                 arm_strategies, levers=levers)
    T0 = ctx.pop("T0")

    # utility: a reply is worth 1.0; sending costs 0.03 (the burned cold-shot / annoyance risk of a
    # bad first impression with this recipient); do_nothing = 0. Explicit, decomposed, documented.
    send_cost = 0.03

    def send_action(aid):
        return ActionSchema(action_id=aid, actor="beckett", operation="communicate",
                            object="peter_thiel", recipients=["peter_thiel"], timing_ts=T0 + 3600.0,
                            direct_cost=send_cost, content={"variant": aid},
                            authority_basis="communicate")

    actions = [send_action(aid) for aid in arms]
    problem = DecisionProblem(
        decision_id="thiel_cold_email_2026",
        decision_maker="beckett", role="founder",
        authority=["communicate"],
        controllable_resources={"cold_shot": 1.0},
        context="17-year-old founder of Aurelius decides what (if anything) to cold-email "
                "Peter Thiel to get a reply",
        as_of="2026-09-01", horizon="2026-09-15",
        utility=UtilitySpec(stakeholders=[
            Stakeholder("beckett", utility_fn=lambda o: float(o.get("readout") or 0.0))],
            aggregation="weighted_sum", provenance="user_supplied"),
        risk=RiskSpec(tolerance="neutral", robustness="expected"),
        information_gathering_allowed=False,
        human_approval_required=True)

    r = evaluate_actions(problem, actions, ctx, budget="production", seed=seed, n_particles=400)
    os.makedirs(ART, exist_ok=True)
    DecisionLedger(os.path.join(ART, "ledger.jsonl")).freeze(problem, r)
    return r, arms


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=90)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()

    world = World(store=EventStore(":memory:"),
                  resolver=PublicFigureResolver(search_fn=lambda q: _EVIDENCE))
    chat = None
    if not args.offline:
        backend = default_chat_fn(max_tokens=800, temperature=0.2)
        if backend is not None:
            chat = TracingChat(backend, os.path.join(ART, "llm_trace.jsonl"))
    fitted = load_fitted(FIT_PATH) if os.path.exists(FIT_PATH) else None

    result = optimize_for_world(
        world, "peter_thiel", name="Peter Thiel", domain="AI infrastructure",
        ask="cold outreach from a 17-year-old founder",
        sender_brief=AURELIUS_SENDER, chat_fn=chat, beam=3 if chat else 6, n_mc=2000,
        fit=fitted)

    prof = world.profile("peter_thiel") or {}
    iv = prof.get("inferred_variables", {})
    profile_vars = {k: v["value"] for k, v in iv.items() if k != "base_responsiveness"}
    confidences = {k: v.get("confidence", 0.5) for k, v in iv.items()}
    persona = world.persona("peter_thiel", name="Peter Thiel")

    # the encoder the pipeline used (LLM if live, lexical otherwise) re-derived for the arm encoding
    from swm.decision.compositional_search import encode_text_to_strategy
    levers = []
    encode_fn = encode_text_to_strategy
    if chat is not None:
        from swm.decision.llm_moves import llm_message_encoder
        from swm.decision.situational_levers import generate_levers
        levers = generate_levers(chat, "Peter Thiel", profile_vars, evidence="")
        encode_fn = llm_message_encoder(chat, levers=levers)
    dec, arms = run_phase13_decision(result, encode_fn, fitted, profile_vars, confidences,
                                     persona.responsiveness.mean,
                                     persona.responsiveness.n_effective, levers, seed=args.seed)

    payload = {
        "experiment": "exp090_thiel_phase13",
        "facts_source": FACTS_SOURCE,
        "sender_brief": {"sender": AURELIUS_SENDER.sender, "thesis": AURELIUS_SENDER.thesis,
                         "ask": AURELIUS_SENDER.ask, "facts": AURELIUS_SENDER.facts},
        "mode": "live_llm" if chat is not None else "offline",
        "n_llm_calls": chat.n if chat is not None else 0,
        "optimizer": result.summary(),
        "phase13_decision": dec.as_dict(),
        "arm_texts": arms,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ART, "result.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)

    print("=" * 88)
    print("EXP-090  Thiel cold email — upgraded optimizer + Phase 13 canonical decision layer")
    print("=" * 88)
    print("\n[EMAIL] (constructed, gated, repaired)")
    print("  " + result.email.text.replace(". ", ".\n  "))
    crit = result.email.critique
    print("\n[CRITIC] coherence=%.2f naturalness=%.2f humanness=%.2f factuality=%.2f (source=%s)"
          % (crit.coherence, crit.naturalness, crit.humanness, crit.factuality, crit.source))
    for fl in crit.flags():
        print("   flag:", fl["issue"], "—", fl["sentence"][:64])
    ev = result.evaluation
    print("\n[L3] p(reply) mean=%.3f 80%%=[%.3f,%.3f] grade=%s"
          % (ev.p_mean, ev.interval80[0], ev.interval80[1], ev.grade))
    print("\n[PHASE 13] matched decision vs do_nothing (400 particles, CRN-paired, 3 hypotheses)")
    for e in dec.evaluated:
        pr = e.get("paired_vs_reference", {})
        print("  %-28s EU=%+.4f  P(improve)=%s  CVaR=%+.4f  byH=%s" % (
            e["action_id"], e["expected_utility"], pr.get("p_improvement"),
            e["cvar"], {h: round(v["mean"], 3) for h, v in e["by_hypothesis"].items()}))
    print("  recommended:", dec.recommended, "| causal claim:", dec.causal_claim)
    print("\nartifacts ->", os.path.relpath(ART), " wall=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
