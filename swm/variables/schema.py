"""The behavioral variable taxonomy — the CORE of the social world model.

The central thesis: to simulate how a person reacts to a message/event, you must map EVERY variable
acting on that behavior — the known ones (from data/user context) and the inferred ones (from context
clues). This module is that taxonomy: a grounded, explicit registry of the variables that influence a
social response, organized by the behavioral-science-standard determinants of action.

Design commitments (so this is a real model, not a vibe):
- Every variable has a CATEGORY, a KIND (stable trait / relational / incentive / transient state /
  platform / message-fit), a RANGE, and the PROVENANCE it is *allowed* to come from.
- A variable is a value + provenance + confidence — never a bare number. `data`/`user` provenance is
  high-confidence (known); `llm`/`heuristic` is inferred (carries the inferer's confidence).
- The LLM may INFER these variables from context clues; it never predicts the outcome. The variables
  feed a calibrated simulation/readout, and each must earn its place on held-out backtests.

Categories (the determinants of a social response):
  DISPOSITION   — stable traits of the person (who they are)
  RELATIONAL    — their stance toward the source/topic (the tie)
  INCENTIVE     — what they gain/lose by responding (the stakes)
  STATE         — transient condition right now (attention/mood/timing)
  PLATFORM      — norms/mechanics of the setting
  MESSAGE_FIT   — how well the specific action fits them
"""
from __future__ import annotations

from dataclasses import dataclass

DISPOSITION = "disposition"
RELATIONAL = "relational"
INCENTIVE = "incentive"
STATE = "state"
PLATFORM = "platform"
MESSAGE_FIT = "message_fit"
PERSONA = "persona"        # deep stable traits inferred from a person's writing history (the "interview")

# provenance ranks (higher = more trusted); used to resolve conflicts and weight the readout.
# `web` = observed public behavior/statements about the entity, gathered online (public figures). It
# ranks BELOW your own interaction logs (`data`) and provided facts (`user`) — it is real evidence,
# not a private log — but ABOVE a bare `llm` prior, because it is grounded in cited external signal.
PROVENANCE_RANK = {"user": 4, "data": 4, "web": 3, "llm": 2, "heuristic": 1, "prior": 0}


@dataclass(frozen=True)
class VariableSpec:
    name: str
    category: str
    description: str
    signed: bool = False          # False: [0,1]; True: [-1,1] (a stance/valence axis)
    allowed_provenance: tuple = ("data", "user", "web", "llm", "heuristic", "prior")
    default: float = 0.5
    prior_confidence: float = 0.15   # confidence of the population prior when nothing is known


# --- the registry: the variables acting on a social response -------------------------------------
_SPECS: list[VariableSpec] = [
    # DISPOSITION — who they are (stable)
    VariableSpec("base_responsiveness", DISPOSITION,
                 "general propensity to respond/engage at all", default=0.3),
    VariableSpec("expertise", DISPOSITION, "domain expertise / seniority relevant to the ask"),
    VariableSpec("conscientiousness", DISPOSITION, "tendency to follow through vs let things drop"),
    VariableSpec("openness_to_outreach", DISPOSITION, "receptivity to unsolicited/new contact"),
    VariableSpec("status", DISPOSITION, "their standing/authority in this setting (maintainer, exec)"),
    VariableSpec("skepticism", DISPOSITION, "disposition to distrust/scrutinize claims", signed=False),
    # RELATIONAL — the tie to source/topic
    VariableSpec("relationship_strength", RELATIONAL, "closeness/history with the sender/source",
                 default=0.0, signed=False),
    VariableSpec("trust_in_source", RELATIONAL, "trust in the sender/source", signed=True, default=0.0),
    VariableSpec("prior_stance", RELATIONAL, "existing attitude toward the topic/ask", signed=True,
                 default=0.0),
    VariableSpec("reciprocity_debt", RELATIONAL, "sense of owing/being owed a response", signed=True,
                 default=0.0),
    # INCENTIVE — the stakes
    VariableSpec("goal_alignment", INCENTIVE, "how well the ask aligns with their goals/interests",
                 signed=True, default=0.0),
    VariableSpec("stakes", INCENTIVE, "how consequential responding (or not) is for them"),
    VariableSpec("effort_cost", INCENTIVE, "effort required to respond (suppresses response)"),
    VariableSpec("reputational_incentive", INCENTIVE, "visibility/reputation payoff for engaging"),
    # STATE — transient condition
    VariableSpec("attention_availability", STATE, "current bandwidth / not-too-busy", default=0.6),
    VariableSpec("recency_of_contact", STATE, "how recently they last engaged with this source/topic"),
    VariableSpec("mood_valence", STATE, "inferred current affect", signed=True, default=0.0),
    VariableSpec("urgency_fit", STATE, "match of the ask's urgency to their situation"),
    # PLATFORM — setting mechanics
    VariableSpec("platform_response_norm", PLATFORM, "baseline response norm of this channel/setting",
                 default=0.3),
    VariableSpec("visibility", PLATFORM, "public/audience visibility of the exchange"),
    VariableSpec("formality", PLATFORM, "formality expected in this setting"),
    # MESSAGE_FIT — how the specific action fits them
    VariableSpec("personalization", MESSAGE_FIT, "how tailored the message is to this person"),
    VariableSpec("clarity", MESSAGE_FIT, "how clear/actionable the ask is"),
    VariableSpec("pushiness", MESSAGE_FIT, "how aggressive/salesy (suppresses response)"),
    VariableSpec("ask_directness", MESSAGE_FIT, "explicit specific ask vs vague"),
    VariableSpec("length_fit", MESSAGE_FIT, "message length vs their inferred preference"),
    # GENERAL message-fit levers the SENDER controls — the universal "physics" of any inbound ask,
    # scored for every message. Their EFFECT is recipient-conditioned via interactions
    # (see swm/decision/strategy_scorer.py); recipient-SPECIFIC levers (e.g. a contrarian pitch for
    # Thiel) are generated per recipient as situational levers (swm/decision/situational_levers.py).
    VariableSpec("relevance_fit", MESSAGE_FIT, "how well the ask matches the recipient's interests/mandate",
                 default=0.4),
    VariableSpec("credibility_proof", MESSAGE_FIT, "concrete evidence/traction backing the claims",
                 default=0.3),
    VariableSpec("responder_incentive", MESSAGE_FIT, "what's in it for the responder (payoff to them for engaging)",
                 default=0.3),
    VariableSpec("low_effort_ask", MESSAGE_FIT, "how easy/low-friction it is for them to respond",
                 default=0.4),
    # the register axis the user's feedback isolated: PERFORMING easiness/benefit is DISTINCT from the
    # ask genuinely being brief. A message that announces 'no follow-up required', pre-chews a
    # 'you could test this yourself' step, or assures the reader what they'll get is convenience-
    # selling — it reads as pushy/AI to a high-status skeptic even though it looks helpful.
    VariableSpec("convenience_selling", MESSAGE_FIT,
                 "how much the message PERFORMS easiness / assures the reader a payoff / pre-chews "
                 "their next step ('no follow-up required', 'you could just test this yourself') — "
                 "salesmanship dressed as politeness, distinct from the ask genuinely being short",
                 default=0.2),
    VariableSpec("warmth", MESSAGE_FIT, "warm, respectful, human tone (vs cold/transactional)", default=0.5),
    VariableSpec("credential_signaling", MESSAGE_FIT, "how much the message parades status/credentials",
                 default=0.3),
    # retained in the schema but no longer in the UNIVERSAL set — now emitted as situational levers when
    # a recipient's inferred values call for them (kept for back-compat / registry keys).
    VariableSpec("contrarian_pitch", MESSAGE_FIT, "how non-consensus / against-the-grain the thesis is",
                 default=0.3),
    VariableSpec("secret_density", MESSAGE_FIT, "presence of a specific, non-obvious claim ('a secret')",
                 default=0.3),
    # PERSONA — the deep, stable traits a person's WRITING HISTORY reveals (our scalable analog of the
    # 2-hour interview in Generative-Agent SOTA). Inferred multi-pass over the as-of corpus; confidence
    # grows with corpus depth + internal consistency. These are the "everything we model about them".
    # Big Five
    VariableSpec("trait_openness", PERSONA, "openness to experience / intellectual curiosity"),
    VariableSpec("trait_conscientiousness", PERSONA, "diligence, follow-through, thoroughness"),
    VariableSpec("trait_extraversion", PERSONA, "social energy / assertive engagement"),
    VariableSpec("trait_agreeableness", PERSONA, "warmth, cooperativeness vs antagonism"),
    VariableSpec("trait_emotional_stability", PERSONA, "calm/even vs reactive/volatile"),
    # epistemic / cognitive style
    VariableSpec("epistemic_rigor", PERSONA, "reliance on evidence, logic, sourcing vs assertion"),
    VariableSpec("intellectual_humility", PERSONA, "willingness to concede / update / say 'I might be wrong'"),
    VariableSpec("analytical_style", PERSONA, "analytical/systematic vs intuitive/associative"),
    VariableSpec("certainty_disposition", PERSONA, "how confidently/absolutely they assert claims"),
    # communication style
    VariableSpec("verbosity", PERSONA, "characteristic length/elaboration of their writing"),
    VariableSpec("politeness_disposition", PERSONA, "respectful/civil vs combative/dismissive tone"),
    VariableSpec("emotional_expressiveness", PERSONA, "affect and emotion shown in writing", signed=False),
    VariableSpec("humor_disposition", PERSONA, "use of humor/irony/levity"),
    # social / interpersonal orientation
    VariableSpec("combativeness", PERSONA, "eagerness to confront/argue vs seek common ground"),
    VariableSpec("empathy_display", PERSONA, "perspective-taking / acknowledging others' views"),
    VariableSpec("status_orientation", PERSONA, "concern with standing/being-right vs truth-seeking"),
    # values / worldview axes (drive opinion & stance outcomes)
    VariableSpec("value_individualism", PERSONA, "individualist vs collectivist leaning", signed=True,
                 default=0.0),
    VariableSpec("value_traditionalism", PERSONA, "traditional vs progressive/novel leaning", signed=True,
                 default=0.0),
    VariableSpec("risk_tolerance", PERSONA, "comfort with risk/uncertainty vs caution"),
    VariableSpec("moral_absolutism", PERSONA, "rule/principle-driven vs pragmatic/consequentialist"),
    # domain footprint
    VariableSpec("expertise_breadth", PERSONA, "range of domains they engage competently"),
    VariableSpec("topical_focus", PERSONA, "specialist (narrow) vs generalist (broad) engagement"),
    VariableSpec("persistence", PERSONA, "tenacity in sustained back-and-forth"),
]

SPECS: dict[str, VariableSpec] = {s.name: s for s in _SPECS}
NAMES: list[str] = [s.name for s in _SPECS]
BY_CATEGORY: dict[str, list[str]] = {}
for _s in _SPECS:
    BY_CATEGORY.setdefault(_s.category, []).append(_s.name)


def spec(name: str) -> VariableSpec:
    return SPECS[name]
