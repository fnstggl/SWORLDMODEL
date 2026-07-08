"""Response model — P(this person responds favorably to this message, in this state).

The function the IndividualAgent reads its response through. Two backends, same signature
`(variables: VariableMap, state: dict, message: dict) -> {"p", "drivers", ...}`:

  - `StructuredResponseModel` (VALIDATED): a grounded, fittable logistic over four interpretable
    quantities — the person's RECEPTIVITY, the message's QUALITY/fit, their INTERACTION (the crux: a
    strong argument moves an open mind far more than an entrenched one), and the STATE FRICTION (a busy,
    depleted, sour person responds to less). Fit on real persuasion data in EXP-060; because it is the
    same object used to score and to simulate, the validation calibrates the real thing.
  - `llm_response_fn(judge_fn)` (PRODUCTION): the LLM reasons AS the person given who they are, how they
    are right now, and the message, and returns P(respond). Pluggable backend, identical to
    semantic_stance / intervention_selector.

The four quantities are computed from NAMED schema variables, normalized to a common "supportive" 0..1
scale (so signed and unsigned variables compose), defaulting missing variables to neutral — so the model
is general across question types, not wired to one dataset.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from swm.transition.readout import LogisticReadout
from swm.variables.schema import SPECS, spec

# which named variables push RECEPTIVITY (the person's susceptibility to responding / being moved)
RECEPTIVITY_POS = ("trait_openness", "openness_to_outreach", "goal_alignment", "trust_in_source",
                   "relationship_strength", "intellectual_humility", "base_responsiveness",
                   "reciprocity_debt", "conscientiousness")
RECEPTIVITY_NEG = ("skepticism", "combativeness", "certainty_disposition", "entrenchment")
# which named variables constitute message QUALITY / fit (persuasive/response-eliciting force)
QUALITY_POS = ("clarity", "ask_directness", "personalization", "expertise", "epistemic_rigor",
               "politeness_disposition", "addresses_crux", "evidence", "reputational_incentive")
QUALITY_NEG = ("pushiness",)
# state variables that create FRICTION (suppress response when depleted / loaded / sour / not-owed)
FRICTION = ("cognitive_load",)                    # + derived: low attention, negative mood


def _supportive(name: str, v: float, negate: bool = False) -> float:
    """Normalize any variable to a 0..1 'supports responding' scale (signed vars centered at 0.5)."""
    signed = spec(name).signed if name in SPECS else False
    x = (0.5 + 0.5 * v) if signed else v
    return (1.0 - x) if negate else x


def _agg(source: dict, pos, neg) -> float:
    """Mean supportive value over the named variables present in `source` (neutral 0.5 if none)."""
    vals = []
    for n in pos:
        if n in source:
            vals.append(_supportive(n, float(source[n])))
    for n in neg:
        if n in source:
            vals.append(_supportive(n, float(source[n]), negate=True))
    return sum(vals) / len(vals) if vals else 0.5


def _as_dict(variables) -> dict:
    """Accept a VariableMap or a plain dict of {name: value}."""
    if hasattr(variables, "vars"):
        return {n: var.value for n, var in variables.vars.items()}
    return dict(variables or {})


def quantities(variables, state: dict, message: dict) -> dict:
    """The four interpretable drivers of a response, from named variables (general across questions)."""
    pv = _as_dict(variables)
    receptivity = _agg(pv, RECEPTIVITY_POS, RECEPTIVITY_NEG)
    quality = _agg(message, QUALITY_POS, QUALITY_NEG)
    # friction: high cognitive load, low attention, and negative mood all suppress the response
    load = float(state.get("cognitive_load", 0.2))
    low_attention = 1.0 - float(state.get("attention_availability", 0.6))
    sour = 0.5 - 0.5 * float(state.get("mood_valence", 0.0))
    friction = (load + low_attention + sour) / 3.0
    return {"receptivity": receptivity, "quality": quality, "friction": friction}


FEATURES = ("receptivity", "quality", "interaction", "friction", "quality_when_fresh")


def _features(q: dict, which=FEATURES) -> list:
    full = {"receptivity": q["receptivity"], "quality": q["quality"],
            "interaction": q["receptivity"] * q["quality"], "friction": q["friction"],
            "quality_when_fresh": q["quality"] * (1.0 - q["friction"])}
    return [full[k] for k in which]


FIT_FEATURES = ("receptivity", "quality", "interaction")   # the person x message signal identifiable in data
# friction at the neutral resting state: load 0.2, low-attention (1-0.6)=0.4, sour (0.5-0.5*0)=0.5
FRICTION_BASELINE = (0.2 + 0.4 + 0.5) / 3.0


@dataclass
class StructuredResponseModel:
    """Grounded logistic response model. Two clearly-separated parts:
      - the FITTED person x message coefficients (receptivity, quality, their interaction), learned from
        data — this is what EXP-060 validates;
      - an optional grounded STATE GATE applied in log-odds: a worse transient state (busy/sour/loaded)
        lowers the response odds, a better one raises them, and it is ZERO at the resting state so it never
        disturbs the fit. It is first-principles (not fit) because the validation data has no state
        variation to identify it — an honest separation, not a hidden knob."""
    features: tuple = FIT_FEATURES
    l2: float = 0.3
    state_gate: bool = False
    gate_strength: float = 1.5
    readout: LogisticReadout = None
    base_rate: float = 0.5

    def featurize(self, variables, state: dict, message: dict) -> list:
        return _features(quantities(variables, state, message), self.features)

    def fit(self, rows: list) -> "StructuredResponseModel":
        """rows: list of (variables, state, message, y). Fits the logistic on the selected features."""
        X = [self.featurize(v, s, m) for v, s, m, _ in rows]
        y = [int(o) for *_, o in rows]
        self.base_rate = sum(y) / len(y) if y else 0.5
        self.readout = LogisticReadout(l2=self.l2).fit(X, y)
        return self

    def __call__(self, variables, state: dict, message: dict) -> dict:
        q = quantities(variables, state, message)
        x = _features(q, self.features)
        p = self.readout.predict_proba(x) if self.readout is not None else self.base_rate
        if self.state_gate:                                # grounded, bounded, zero at the resting state
            p = min(1 - 1e-6, max(1e-6, p))
            logit = math.log(p / (1 - p)) - self.gate_strength * (q["friction"] - FRICTION_BASELINE)
            p = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
        return {"p": p, "drivers": {k: round(v, 3) for k, v in q.items()}}


# ---- production LLM backend -------------------------------------------------------------------------
def build_response_prompt(variables, state: dict, message: dict, goal: str = "respond") -> str:
    pv = _as_dict(variables)
    return (
        "You are simulating ONE specific person's reaction to a message. Reason AS this person.\n"
        f"WHO THEY ARE (stable variables 0..1, signed −1..1): {json.dumps({k: round(v, 2) for k, v in pv.items()})}\n"
        f"HOW THEY ARE RIGHT NOW (transient state): {json.dumps({k: round(float(v), 2) for k, v in state.items()})}\n"
        f"THE MESSAGE (its properties 0..1): {json.dumps({k: round(float(v), 2) for k, v in message.items()})}\n"
        f"Given who they are, their current state (busyness, mood, load, whether they owe a reply), and "
        f"this specific message, what is the probability they {goal}? Judge from the person and situation; "
        "do not assume an outcome. Return ONLY JSON: {\"p\": <0..1>, \"reason\": \"<=15 words\"}")


def _coerce_p(raw) -> dict:
    obj = raw if isinstance(raw, dict) else json.loads(str(raw)[str(raw).find("{"):str(raw).rfind("}") + 1])
    p = float(obj.get("p", 0.5)) if obj else 0.5
    return {"p": max(0.0, min(1.0, p)), "reason": (obj.get("reason", "") if obj else "")[:120]}


def llm_response_fn(judge_fn, goal: str = "respond"):
    """PRODUCTION response_fn: the LLM reasons as the person. `judge_fn(prompt) -> {p} | raw JSON`."""
    def fn(variables, state, message):
        out = _coerce_p(judge_fn(build_response_prompt(variables, state, message, goal)))
        return {"p": out["p"], "drivers": {"reason": out["reason"]},
                "quantities": quantities(variables, state, message)}
    return fn
