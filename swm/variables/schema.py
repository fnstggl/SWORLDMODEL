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
    # content-stance choices the SENDER controls (the message optimizer searches over these). Their
    # effect is recipient-conditioned: credential-signaling flips sign against a prestige-skeptic,
    # a contrarian pitch pays off with a contrarian recipient (see swm/decision/strategy_scorer.py).
    VariableSpec("credential_signaling", MESSAGE_FIT, "how much the message parades status/credentials",
                 default=0.3),
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
