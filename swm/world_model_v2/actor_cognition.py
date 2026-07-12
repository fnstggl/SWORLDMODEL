"""Universal actor cognition — the structured decision boundary (Phase 2C).

The universal principle, for EVERY actor decision in EVERY domain:

    typed shared WorldState
    → actor-specific observable view
    → structured INTERPRETATION of the incoming item (typed, clamped, abstaining)
    → actor-local state update
    → distribution over TYPED actions (fitted policy over interpretation × hidden state × anchors)
    → simulator validation → StateDelta → shared-world consequences

Nothing in this module is messaging-specific. "Interpretation" is *an actor making sense of an incoming
information item* — an email, a negotiation offer, a campaign ad, a platform post, a committee motion all
fit. Domains contribute an ACTION PACK (typed action vocabulary) and a parameter pack; the interpretation
schema, the fitted policy, the hidden-state sampler and the temporal processes are shared.

Why the fitted policy layer exists: the Enron max-capacity round proved a raw LLM scalar is badly
miscalibrated (worse than bag-of-words). Here the LLM produces STRUCTURED SEMANTIC FEATURES only; a small
calibration layer FITTED ON TRAIN maps those features (+ the metadata anchor) to behavior. The LLM never
mints the probability; it reads meaning. Provenance: features "inferred (LLM interpretation)"; mapping
"fitted (train)"; hidden-state priors labeled; no unsupported precise constants.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.mechanisms import MechanismEntry, register_mechanism
from swm.world_model_v2.state import register_entity_extension

# ---------------------------------------------------------------- interpretation schema (universal)
INTENTS = ("request_action", "request_information", "directive", "share_information",
           "social", "transactional", "other")

#: fixed feature order — the fitted policy's contract. 11 continuous dims + 7 intent one-hots.
FEATURE_DIMS = ("urgency", "obligation", "task_ownership", "effort_required", "relevance_to_goals",
                "risk_of_inaction", "benefit_of_action", "relationship_salience",
                "needs_clarification", "needs_delegation", "thread_continuity")


@dataclass
class Interpretation:
    """One actor's structured reading of one incoming information item. Every dim ∈ [0,1]; intent is one
    of INTENTS. Status: inferred (LLM interpretation) — semantic evidence, never a probability."""
    intent: str = "other"
    urgency: float = 0.5
    obligation: float = 0.5
    task_ownership: float = 0.5
    effort_required: float = 0.5
    relevance_to_goals: float = 0.5
    risk_of_inaction: float = 0.5
    benefit_of_action: float = 0.5
    relationship_salience: float = 0.5
    needs_clarification: float = 0.0
    needs_delegation: float = 0.0
    thread_continuity: float = 0.0
    why: str = ""
    raw: dict = field(default_factory=dict)

    def features(self) -> list:
        x = [getattr(self, d) for d in FEATURE_DIMS]
        x += [1.0 if self.intent == it else 0.0 for it in INTENTS]
        return x

    def as_dict(self):
        return {"intent": self.intent, **{d: round(getattr(self, d), 3) for d in FEATURE_DIMS},
                "why": self.why[:80]}


_INTERPRET_PROMPT = """You ARE the actor described below, making sense of something you just observed.
Judge ONLY from what you can see now — never from the future.
YOU ARE: {actor}
CHANNEL: {channel}
YOUR OBSERVABLE CONTEXT (what you know right now):
{context}
THE ITEM YOU JUST OBSERVED:
---
{content}
---
Interpret this item FROM YOUR PERSPECTIVE. Rate each dimension 0..1:
- urgency: how time-critical a response/action is
- obligation: how obligated YOU are to act (hierarchy, ownership, explicit ask)
- task_ownership: is the implied task YOURS (1) shared (0.5) or someone else's (0)
- effort_required: how costly a meaningful response/action would be
- relevance_to_goals: relevance to YOUR current work and goals
- risk_of_inaction: bad consequences for you if you do nothing
- benefit_of_action: what you gain by acting
- relationship_salience: how much this specific relationship matters to you
- needs_clarification: whether you'd need to ask something back before acting
- needs_delegation: whether the right move is handing this to someone else
- thread_continuity: is this a continuation of an exchange you are actively part of
- intent: what the source wants — one of {intents}
Return ONLY compact JSON:
{{"intent": "<one>", "urgency": <0..1>, "obligation": <0..1>, "task_ownership": <0..1>,
"effort_required": <0..1>, "relevance_to_goals": <0..1>, "risk_of_inaction": <0..1>,
"benefit_of_action": <0..1>, "relationship_salience": <0..1>, "needs_clarification": <0..1>,
"needs_delegation": <0..1>, "thread_continuity": <0..1>, "why": "<8 words>"}}"""


def _clamp01(v, default=0.5):
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return default


def interpret(chat_fn, *, actor: str, channel: str, context: str, content: str,
              meter=None) -> Interpretation | None:
    """The universal interpretation boundary: ONE LLM call over the actor-observable view + the exact item
    → a typed Interpretation. Returns None on parse failure (the arm ABSTAINS from semantic evidence and
    falls back to its metadata anchor — never a fabricated middle value passed off as a reading)."""
    from swm.engine.grounding import parse_json
    prompt = _INTERPRET_PROMPT.format(actor=actor, channel=channel, context=context,
                                      content=content, intents=list(INTENTS))
    txt = chat_fn(prompt)
    if meter is not None:
        meter["calls"] = meter.get("calls", 0) + 1
        meter["tokens"] = meter.get("tokens", 0) + (len(prompt) + len(txt or "")) // 4
    r = parse_json(txt) or {}
    if "intent" not in r and not any(d in r for d in FEATURE_DIMS):
        return None
    it = str(r.get("intent", "other")).strip().lower()
    return Interpretation(intent=it if it in INTENTS else "other",
                          **{d: _clamp01(r.get(d), 0.5 if d not in
                             ("needs_clarification", "needs_delegation", "thread_continuity") else 0.0)
                             for d in FEATURE_DIMS},
                          why=str(r.get("why", ""))[:120], raw=r)


# ---------------------------------------------------------------- fitted action policy (train-only)
def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


@dataclass
class ActionPolicy:
    """The calibration layer: logistic over interpretation features + the metadata-anchor logit.
    p_engage = σ(w·features + w_anchor·logit(base_p) + b). Fitted on TRAIN ONLY. With w=0, w_anchor=1,
    b=0 it reduces exactly to the metadata model — semantic features must EARN weight from data.
    status: fitted (train) over LLM-inferred semantic features."""
    w: list = field(default_factory=list)
    w_anchor: float = 1.0
    b: float = 0.0
    n_train: int = 0
    status: str = "fitted (train) over LLM-inferred semantic features"

    def p_engage(self, feats: list, base_p: float) -> float:
        z = sum(wi * xi for wi, xi in zip(self.w, feats)) + self.w_anchor * _logit(base_p) + self.b
        return min(0.97, max(0.01, _sigmoid(z)))


def fit_action_policy(samples, *, l2=0.03, iters=600, lr=0.15) -> ActionPolicy:
    """samples: [(features, base_p, engaged 0/1)]. Gradient-descent logistic with L2 on feature weights
    and (w_anchor−1)² shrinkage — the fit must beat the anchor to move off it. Small-n safe."""
    if not samples:
        return ActionPolicy()
    k = len(samples[0][0])
    w, wa, b = [0.0] * k, 1.0, 0.0
    n = len(samples)
    for _ in range(iters):
        gw, gwa, gb = [0.0] * k, 0.0, 0.0
        for x, bp, y in samples:
            q = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + wa * _logit(bp) + b)
            e = q - y
            for i in range(k):
                gw[i] += e * x[i]
            gwa += e * _logit(bp)
            gb += e
        w = [wi - lr * (gi / n + l2 * wi) for wi, gi in zip(w, gw)]
        wa -= lr * (gwa / n + l2 * (wa - 1.0))
        b -= lr * gb / n
    return ActionPolicy(w=[round(x, 5) for x in w], w_anchor=round(wa, 5), b=round(b, 5), n_train=n)


# ---------------------------------------------------------------- typed action packs (domain vocabulary)
ACTION_PACKS = {
    "messaging":   ("reply_now", "reply_later", "ask_clarification", "delegate", "ignore"),
    "negotiation": ("accept", "counteroffer", "ask_clarification", "delay", "reject"),
    "election":    ("act_support", "act_oppose", "ask_clarification", "delegate", "abstain"),
    "platform":    ("engage_now", "engage_later", "ask_clarification", "reshare", "ignore"),
    "institution": ("approve", "amend", "ask_clarification", "refer", "defer"),
}
#: actions that constitute ENGAGEMENT (produce an observable response) per pack, in preference order:
#: (act_immediately, act_deferred, clarify) — the remaining two are (hand_off, no_action).
ENGAGE_SLOTS = {p: (a[0], a[1], a[2]) for p, a in ACTION_PACKS.items()}
PASSIVE_SLOTS = {p: (a[3], a[4]) for p, a in ACTION_PACKS.items()}


def action_distribution(pack: str, interp: Interpretation, p_engage: float) -> dict:
    """Universal composition: the FITTED p_engage fixes the total engagement mass (calibration is never
    re-decided here); the interpretation shapes HOW that mass splits into typed actions. Splits are bounded,
    documented priors (prior_backed) pending per-domain fits:
      clarify   ← needs_clarification (≤35% of engage mass)
      immediate ← urgency (35–85% of the remaining engage mass)
      hand_off  ← needs_delegation (≤50% of the passive mass)
    Always returns a full normalized distribution over the pack."""
    acts = ACTION_PACKS[pack]
    now_a, later_a, clar_a = ENGAGE_SLOTS[pack]
    hand_a, none_a = PASSIVE_SLOTS[pack]
    pe = min(0.97, max(0.01, p_engage))
    clar = pe * 0.35 * interp.needs_clarification
    rest = pe - clar
    now_frac = 0.35 + 0.5 * interp.urgency
    dist = {now_a: rest * now_frac, later_a: rest * (1 - now_frac), clar_a: clar}
    passive = 1.0 - pe
    hand = passive * 0.5 * interp.needs_delegation
    dist[hand_a] = hand
    dist[none_a] = passive - hand
    z = sum(dist.values())
    return {a: dist.get(a, 0.0) / z for a in acts}


# ---------------------------------------------------------------- hidden actor state (per particle)
def hidden_state_latents(actor_id: str, *, workload_norm: float, hetero_sd: float,
                         attention_mean: float = 0.7):
    """PART 3: the per-particle coherent hidden actor state, as LatentVariableRecords + CorrelationRules
    for InitialStateModel. Every prior labeled; hetero_sd must be FITTED (e.g. dispersion of per-actor
    response rates) — never an invented constant.
      attention       ~ N(mean − 0.2·workload, 0.25) ∈ [0.05, 1]   (prior_backed, workload-coupled)
      responsiveness  ~ N(1, hetero_sd) ∈ [0.5, 1.8]               (fitted dispersion)
      obligation_sensitivity ~ N(1, 0.2) ∈ [0.5, 1.5]              (broad prior)
    Correlations: responsiveness↔attention +0.3 (engaged people are reachable) — declared, bounded."""
    from swm.world_model_v2.init_state import CorrelationRule, LatentVariableRecord
    wl = min(1.0, max(0.0, workload_norm))
    lat = [
        LatentVariableRecord(path=f"{actor_id}.responsiveness",
                             candidates={"mean": 1.0, "sd": hetero_sd, "lo": 0.5, "hi": 1.8},
                             method="dataset", confidence=0.6),
        LatentVariableRecord(path=f"{actor_id}.attention",
                             candidates={"mean": attention_mean - 0.2 * wl, "sd": 0.25,
                                         "lo": 0.05, "hi": 1.0},
                             method="prior", confidence=0.4),
        LatentVariableRecord(path=f"{actor_id}.obligation_sensitivity",
                             candidates={"mean": 1.0, "sd": 0.2, "lo": 0.5, "hi": 1.5},
                             method="prior", confidence=0.4),
    ]
    cors = [CorrelationRule(src=f"{actor_id}.responsiveness", dst=f"{actor_id}.attention", strength=0.3)]
    return lat, cors


# entity extension: typed hidden-state fields (controlled registry — no arbitrary untyped keys)
register_entity_extension("actor_cognition", fields={
    "responsiveness": "per-actor engagement multiplier (fitted dispersion around 1)",
    "obligation_sensitivity": "how strongly felt obligation moves this actor (broad prior)",
    "workload_pressure": "normalized current workload 0..1 (observed where data exists)",
    "last_observation_ts": "when this actor last observed the world (temporal process bookkeeping)",
}, entity_types=("person",))


# ---------------------------------------------------------------- temporal state processes (PART 4)
def attention_transition(att: float, *, dt_days: float, workload_norm: float, hour: int,
                         rng: random.Random) -> float:
    """attention_t ~ transition(attention_{t-1}, workload, time_of_day, noise): bounded mean-reversion
    toward a workload/daytime-set target + diffusion. Parameters are broad labeled priors (prior_backed);
    the PROCESS SHAPE (mean-reverting, bounded, workload-coupled) is the claim, not the constants."""
    wl = min(1.0, max(0.0, workload_norm))
    target = min(0.95, max(0.15, 0.75 - 0.25 * wl + (0.10 if 8 <= (hour % 24) <= 18 else -0.10)))
    tau = 0.5                                                  # mean-reversion timescale, days (prior)
    pull = 1.0 - math.exp(-max(0.0, dt_days) / tau)
    noise = rng.gauss(0.0, 0.08 * math.sqrt(min(1.0, max(1e-6, dt_days))))
    return min(1.0, max(0.05, att + pull * (target - att) + noise))


def relationship_strength(pair_n: float, pair_rate: float, global_rate: float) -> float:
    """Edge strength ∈ [0,1] inferred from interaction history (status: inferred from data):
    familiarity (interaction count, saturating) × reciprocity (pair response rate vs global)."""
    fam = pair_n / (pair_n + 8.0)
    rec = min(1.0, (pair_rate / max(1e-4, global_rate)) / 5.0)
    return min(1.0, max(0.0, 0.5 * fam + 0.5 * rec))


def relationship_modulator(strength: float, salience: float) -> float:
    """Bounded engagement multiplier from relationship state × the interpretation's relationship_salience.
    ∈ [0.75, 1.35], =1 at (strength .5, salience .5) — prior_backed, cannot manufacture certainty."""
    return min(1.35, max(0.75, 1.0 + 0.7 * (strength - 0.5) * (0.5 + salience)))


def relationship_transition(strength: float, *, engaged: bool) -> float:
    """relationship_t ~ transition(relationship_{t-1}, reciprocity): bounded shift — engagement
    strengthens, being ignored decays slightly. |Δ| ≤ 0.05 (registry-bounded like relationship_update)."""
    return min(1.0, max(0.0, strength + (0.05 if engaged else -0.02)))


# ---------------------------------------------------------------- registry entries
for _e in (
    MechanismEntry("information_interpretation", "belief",
                   "an actor forms a STRUCTURED typed reading of an incoming item (intent/urgency/"
                   "obligation/effort/relevance/risk/benefit/salience/clarify/delegate/continuity)",
                   required_state=("entity", "information_set"),
                   parameter_source="LLM over the actor-observable view ONLY; typed+clamped; abstains",
                   operator="typed_action_decision", calibration_status="experimental", experimental=True),
    MechanismEntry("typed_action_policy", "decision",
                   "fitted calibration layer maps interpretation features + metadata anchor → typed "
                   "action distribution; the LLM never mints the probability",
                   required_state=("entity",), parameter_source="logistic fitted on TRAIN only",
                   operator="typed_action_decision", calibration_status="calibrated"),
    MechanismEntry("attention_dynamics", "exogenous",
                   "bounded mean-reverting attention process coupled to workload/time-of-day",
                   required_state=("entity.attention",), parameter_source="broad labeled priors",
                   temporal_scale="continuous", calibration_status="prior"),
    MechanismEntry("relationship_dynamics", "relationship",
                   "interaction history → edge strength; engagement/neglect shifts it, bounded |Δ|≤0.05",
                   required_state=("network",), parameter_source="inferred from data + bounded shifts",
                   calibration_status="prior"),
):
    register_mechanism(_e)
