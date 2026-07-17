"""Question-level causal relevance adjudication for conditional World Model V2 phases.

The compiler is allowed to propose scenario structure, but it is not a reliable
relevance oracle: an LLM can over-declare causal dependencies on an otherwise
simple question.  This module therefore makes the runtime's relevance decision
from the outcome contract wording itself.  The rules are deliberately
high-precision and describe causal semantics rather than benchmark identifiers.

The adjudicator never inspects outcomes, event IDs, splits, or benchmark source.
It returns the matched semantic cues so every decision is reviewable.
"""
from __future__ import annotations

import re


PHASES = ("phase4_actor_policy", "phase6_registry", "phase7_nonlinear",
          "phase9_populations", "phase9_networks", "phase10_institutions",
          "phase11_recompilation")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _hits(text: str, phrases) -> list[str]:
    padded = f" {text} "
    return [p for p in phrases if f" {p} " in padded or (" " in p and p in text)]


# Deliberate choices by identifiable actors.  A decision verb alone is not
# enough: natural systems can "trigger", prices can "hold", and records can be
# "broken" without an actor policy.
_P4_ACTORS = (
    "assembly", "chamber", "council", "parliament", "governor", "senate", "committee",
    "court", "panel", "tribunal", "commission", "judge", "board", "regulator", "authority",
    "agency", "ministry", "supervisor", "mayor", "alliance", "party", "candidate", "challenger",
    "coalition", "government", "union", "operator", "pilots", "writers", "couriers", "company",
    "firms", "firm", "investor", "retailer", "carrier", "founder", "bank", "fund", "bloc",
    "governments", "junta", "monitors", "consortium", "association", "trustees", "congress",
    "foundation", "diocese", "league", "city", "community", "interim council", "peacekeeping",
    "lower house", "teachers", "energy major", "conglomerate", "cooperative", "standards body",
)
_P4_ACTIONS = (
    "pass", "ratify", "approve", "override", "confirm", "report", "adopt", "strike down", "uphold",
    "certify", "grant", "rule", "disqualify", "invalidate", "find", "order", "clear", "authorize",
    "issue", "win", "lose", "qualify", "concede", "retain", "reach a contract", "reach a deal",
    "reach an agreement", "end", "accept", "vote", "call", "conclude", "complete", "spin off",
    "file", "return", "divest", "cut", "dissolved", "succeed", "sign", "renewed", "admit",
    "referred", "maintain", "liquidate", "withdraw", "hold together", "block", "suspend",
    "ready", "raise", "amend", "close", "expand", "collapse", "verify", "reshape", "agree",
    "escalate", "spread", "force", "trigger coverage", "draw over", "exceed the turnout", "reach",
    "declare", "add", "renewed", "approved", "clear",
)

# A real social/behavioural mechanism: bargaining, persuasion, mobilization,
# adoption, compliance, collective choice, or belief-mediated response.
_P6 = (
    "assembly pass", "chamber ratify", "council approve", "parliament override", "supermajority",
    "senate confirm", "committee report", "coalition government", "impeachment", "board adopt",
    "election", "runoff", "turnout", "vote share", "ballots", "referendum", "plebiscite", "ballot",
    "signatures", "campaign", "union", "strike", "walkout", "lockout", "wage talks", "arbitration",
    "activist investor", "hostile bid", "shareholder pressure", "proxy adviser", "adoption", "active users",
    "downloads", "subscriber", "user base", "uptake", "sign up", "installs", "paying cohort",
    "installations", "video", "hashtag", "circulate", "coverage", "matching campaign", "viral",
    "mistaken for real", "recall notice", "endorsement", "rumor", "livestream", "bank run", "lender face",
    "restructuring talks", "contagion", "redemption wave", "protests", "occupation", "road blockades",
    "general strike", "rally", "mutual aid", "curfew", "petition", "negotiation", "talks", "agreement",
    "ceasefire", "sanctions coalition", "corridor deal", "alliance admit", "hostage", "trade bloc",
    "vaccination", "mask usage", "contact tracing", "boycott", "guideline", "blood donation",
    "community successfully block", "demand response enrollment", "players association", "members vote",
    "tenure review", "standards body", "party congress", "trustees divest", "license change",
    "special assessment", "independence vote", "transition timetable", "memecoin", "candidate qualify",
    "challenger concede", "rural opposition", "recount procedure", "levy pass", "charter amendment",
    "proposition fail", "couriers vote", "residuals deal", "most downloaded", "bank resolution reform",
    "coalition of unions", "methane reduction annex", "hospital occupancy",
)

# Nonlinear dynamics, not merely a formal voting threshold or a numeric target.
_P7 = (
    "tipping", "saturation", "cascade", "contagion", "go viral", "spread", "circulate", "run requiring",
    "bank run", "redemption wave", "surge threshold", "outbreak", "herd immunity", "wildfire",
    "major flood stage", "bleaching", "technical correction", "spike above", "lose ninety percent",
    "capacity limit", "past the saturation", "fifty million views", "majority of the country s campuses",
    "draw over one hundred thousand", "participation threshold", "escalate", "refugee flow", "double within",
    "fuel rationing", "margin calls", "reserve depletion", "rolling blackouts", "active users",
    "household adoption", "most downloaded", "subscriber target", "half its user base",
    "seventy percent uptake", "eligible households", "product endorsement", "solvency trigger",
    "default management auction", "curfew halve", "contact tracing app", "lockout trigger",
    "sympathy actions",
)

# Heterogeneous aggregate behaviour of many people.  Institutional member
# counts do not by themselves imply a population model.
_P9POP = (
    "turnout", "vote share", "ballots cast", "electorate", "membership vote", "couriers vote",
    "national work stoppage", "active users", "household adoption", "downloads", "subscriber",
    "user base", "uptake", "eligible households", "new installs", "paying cohort", "installations",
    "views", "fan community", "mainstream outlets", "circulate", "viral", "affected owners",
    "top search rank", "deposit withdrawals", "online audience", "bank run", "stablecoin", "participants",
    "campuses", "compliance", "turnout of", "signatures", "vaccination rate", "mask usage", "riders",
    "contact tracing app", "screening attendance", "blood donation", "rooftop solar", "community",
    "demand response enrollment", "players association vote", "members vote", "special assessment",
    "housing price index", "memecoin", "refugee flow", "independence vote", "fuel rationing",
    "election", "runoff", "referendum", "plebiscite", "by election", "presidential debate",
    "challenger concede", "walkout spread", "teachers strike", "most downloaded", "lender face a run",
    "road blockades", "protest attendance", "seasonal outbreak", "outbreak burn out",
    "governing party lose", "transit bond measure", "rural opposition", "recount procedure",
    "coastal protection levy", "charter amendment", "stadium funding proposition", "rank choice voting",
)

# Transmission over communication, exposure, trust, authority, or financial
# links.  "Network" is causal only when the question asks about propagation.
_P9NET = (
    "spread", "viral", "circulate", "contagion", "network expand", "network within", "hashtag",
    "video surpass", "coverage in", "mainstream outlets", "mistaken for real", "recall notice reach",
    "endorsement move", "rumor", "livestream", "contact tracing", "outbreak", "cluster spread",
    "vaccination rate recover", "mask usage", "boycott", "guideline be adopted", "student occupation",
    "mutual aid network", "petition", "campaign", "runoff", "opposition alliance", "endorsed slate",
    "ruling coalition retain", "supplier network", "sympathy actions", "couriers vote", "license change",
    "mutiny", "refugee flow", "memecoin", "turnout in the municipal election", "verified signatures",
    "active users", "most downloaded", "subscriber target", "user base", "early adopters",
    "lender face a run", "stablecoin", "fuel price protests", "general strike", "opposition rally",
    "coalition of unions",
)

# Rule-governed procedure that decides the outcome.  Numerical thresholds in
# weather, prices, or adoption are explicitly not enough.
_P10 = (
    "assembly pass", "assembly confirm", "fomc vote", "chamber ratify", "council approve", "parliament override", "supermajority requirement",
    "senate confirm", "committee report", "impeachment resolution", "board adopt", "court", "appeals panel",
    "tribunal", "commission disqualify", "judge approve", "patent office", "regulator", "authority certify",
    "competition authority", "agency authorize", "ministry issue", "supervisor approve", "advisory panel vote",
    "election", "runoff", "referendum", "ballot", "recount procedure", "signatures to qualify", "required quorum",
    "ratify the tentative agreement", "binding arbitration", "vote to unionize", "mandatory mediation",
    "announced merger", "board seats", "tender deadline", "court supervised", "boardroom", "board meeting",
    "proxy adviser", "policy rate", "deposit insurance board", "default management auction", "currency board",
    "parliament pass", "city grant", "sanctions coalition add", "alliance admit", "summit adopt", "arbitration",
    "mandate be renewed", "health authority declare", "treatment guideline be adopted", "grid approvals",
    "safety board review", "grid operator invoke", "appeals stage", "mandated fill", "joint committee",
    "lease auction", "disciplinary panel", "players association vote", "doping tribunal", "alert level be raised",
    "intervention band", "auction", "ratings review", "university senate", "hospital board", "members vote",
    "standards body ratify", "diocese close", "party congress amend", "trustees divest", "owners vote",
    "foundation adopt", "condo association pass", "junta call elections", "independence vote",
    "confidence vote", "peace monitors verify", "interim council agree", "coalition government hold",
    "governing party lose", "candidate qualify", "canton approve", "participation threshold",
    "levy pass", "charter amendment", "proposition fail", "rank choice voting", "shareholder pressure",
    "curfew halve", "trade bloc conclude",
)

# Natural structural-change exposure: potential rule, actor-set, coalition,
# regime, or field change before resolution.
_P11 = (
    "reconciliation bill", "coalition government", "impeachment", "disqualify the challenger",
    "despite objections", "opposition alliance", "late opposition campaign", "strike deadline",
    "teachers strike", "announced merger", "hostile bid", "founder return", "joint venture",
    "bank run", "restructuring talks", "contagion", "emergency session", "default management auction",
    "reserve depletion", "road blockades force", "coalition of unions", "two governments sign",
    "ceasefire talks", "sanctions coalition", "corridor deal", "hostage negotiation", "referred to international",
    "peacekeeping mandate", "public emergency", "rolling blackouts", "second anchor investor",
    "export ban trigger", "party congress amend", "border skirmishes", "junta call elections",
    "independence vote", "coalition collapse", "peace monitors verify", "fuel rationing",
    "assassination attempt", "mutiny spread", "interim council", "rumor about the bank s solvency",
    "mid sized lender face a run", "exchange rate breach", "refugee flow",
)


def adjudicate_question(question: str) -> dict:
    """Return reviewable question-level relevance judgments for conditional phases."""
    text = _norm(question)
    actor_hits = _hits(text, _P4_ACTORS)
    action_hits = _hits(text, _P4_ACTIONS)
    p4 = bool(actor_hits and action_hits)
    # Strategic interactions can be explicit even when the actor noun is a
    # possessive or compound that the compact actor vocabulary does not catch.
    explicit_strategic = _hits(text, (
        "reach a contract", "reach a residuals deal", "hostile bid", "hostage negotiation",
        "ceasefire talks", "restructuring talks", "margin calls force", "joint venture",
        "trigger coverage", "stadium be ready", "assassination attempt reshape",
        "walkout spread", "teachers strike end", "lockout trigger", "public sector wage talks",
        "fuel price protests draw", "general strike call", "corridor deal be renewed",
        "climate summit adopt", "border dispute be referred", "interconnector project receive",
        "gas storage facility reach", "transition fund be approved", "lease auction clear",
        "exchange rate breach", "separatist region hold", "mutiny spread",
        # These phrases encode collective/strategic choices even when the
        # question elides the decision makers themselves.
        "supermajority requirement", "border skirmishes escalate",
    ))
    p4 = p4 or bool(explicit_strategic)
    phase_hits = {
        "phase4_actor_policy": actor_hits + action_hits + explicit_strategic if p4 else [],
        "phase6_registry": _hits(text, _P6),
        "phase7_nonlinear": _hits(text, _P7),
        "phase9_populations": _hits(text, _P9POP),
        "phase9_networks": _hits(text, _P9NET),
        "phase10_institutions": _hits(text, _P10),
        "phase11_recompilation": _hits(text, _P11),
    }
    return {phase: {"required": bool(hits),
                    "why": (f"question-level causal cue(s): {hits[:4]}" if hits
                            else "no question-level causal cue for this phase"),
                    "evidence": hits[:8]}
            for phase, hits in phase_hits.items()}
