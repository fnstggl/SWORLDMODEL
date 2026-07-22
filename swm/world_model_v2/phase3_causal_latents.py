"""Phase 3 accuracy — scenario-specific causal latent state (the intended Phase-3 path, focused + real).

    evidence → scenario-specific hidden causal latents → combination mechanism → outcome rate → terminal

Instead of one generic outcome-rate posterior, each question gets a SMALL set of TYPED causal latents (actor
intent, capability, institutional authority, procedural feasibility, coalition support, resource constraint,
operational readiness, event hazard, structural regime). Each latent is a Bernoulli "favorable-state"
probability in [0,1] with:
  * an operational definition (from the LLM proposal, qualitative);
  * an explicit representation (Beta over the favorable-state probability);
  * evidence links (claims mapped to it, with direction + strength — qualitative, from the LLM);
  * a prior (from a REGISTERED type-prior table — NOT LLM-minted);
  * an observation model (the directional model, hand-set or FITTED — NOT LLM-minted);
  * uncertainty (the Beta posterior spread);
  * a mechanism consumer (a REGISTERED combination: necessary-conjunction / sufficient-disjunction /
    single-driver — the LLM proposes WHICH structure qualitatively; the numbers are fixed);
  * a measurable terminal effect (the combined rate is scored against the realized outcome).

The LLM proposes ONLY qualitative structure (which latents, their polarity, which claims inform them, the
combination type). Every number — prior, likelihood, combined rate — comes from a registered/fitted table.
The numeric inference is pure and deterministic given the (captured) proposal, so it reproduces exactly and is
computed OFFLINE from a frozen capture.
"""
from __future__ import annotations
import json
import math

# Registered type-priors: base favorable-state probability for a latent of this TYPE, absent evidence.
# Weakly-informative, chosen to be conservative (near 0.5) and transport-safe. NOT LLM-minted.
LATENT_TYPES = {
    "intent":       {"alpha": 1.2, "beta": 1.2, "desc": "an actor wants/plans the outcome-favoring action"},
    "capability":   {"alpha": 1.3, "beta": 1.1, "desc": "an actor is able to bring about the outcome"},
    "authority":    {"alpha": 1.2, "beta": 1.2, "desc": "an institution has the authority/mandate to act"},
    "feasibility":  {"alpha": 1.1, "beta": 1.3, "desc": "the outcome is procedurally feasible in the window"},
    "coalition":    {"alpha": 1.2, "beta": 1.2, "desc": "sufficient coalition/political support exists"},
    "resources":    {"alpha": 1.2, "beta": 1.2, "desc": "required resources/means are available"},
    "readiness":    {"alpha": 1.2, "beta": 1.2, "desc": "operational readiness to execute in the window"},
    "hazard":       {"alpha": 1.0, "beta": 1.6, "desc": "an event hazard fires within the horizon (base low)"},
    "regime":       {"alpha": 1.2, "beta": 1.2, "desc": "the prevailing structural regime favors the outcome"},
}
DEFAULT_TYPE = {"alpha": 1.2, "beta": 1.2, "desc": "generic favorable latent state"}

# Combination mechanisms (REGISTERED). Each maps latent favorable-state means -> outcome rate.
#   necessary_conjunction: the outcome needs ALL latents favorable (noisy-AND) -> product (bounded)
#   sufficient_disjunction: ANY latent favorable suffices (noisy-OR) -> 1 - prod(1-m)
#   single_driver: one dominant latent carries the outcome -> that latent's mean
#   weighted_mean: geometric-ish blend when structure is unclear
COMBINATIONS = ("necessary_conjunction", "sufficient_disjunction", "single_driver", "weighted_mean")

_EPS = 1e-6

_PROPOSE_PROMPT = """You are decomposing a forecasting question into a SMALL set of hidden CAUSAL LATENT
states that jointly determine the answer. Do NOT give any probabilities or numbers.

Pick 1 to 4 latents, each of a TYPE from this list: intent, capability, authority, feasibility, coalition,
resources, readiness, hazard, regime. For each latent give:
  - "id": short snake_case name
  - "type": one of the types above
  - "definition": one operational sentence (what 'favorable state' means, concretely, for THIS question)
  - "favorable_supports_yes": true if the latent being in its favorable state pushes the answer toward YES

Then choose ONE "combination" describing how the latents jointly produce the outcome:
  - "necessary_conjunction": the YES outcome needs ALL these latents favorable
  - "sufficient_disjunction": any ONE favorable latent is enough for YES
  - "single_driver": one latent dominates
  - "weighted_mean": unclear / they trade off

Reply ONLY JSON:
{{"latents": [{{"id": "...", "type": "...", "definition": "...", "favorable_supports_yes": true}}],
  "combination": "necessary_conjunction"}}

QUESTION: {question}
HORIZON: {horizon}"""

_MAP_PROMPT = """For each CLAIM below, say which latent(s) it is evidence about and in which direction. Do NOT
give numbers. A claim can inform 0, 1 or more latents. Direction is whether the claim pushes the latent toward
its FAVORABLE state ("favors") or away ("against"); "neutral" if it bears on the latent but is ambiguous.

LATENTS: {latents}

Reply ONLY JSON: {{"map": [{{"claim_id": "...", "links": [{{"latent_id": "...",
  "direction": "favors|against|neutral"}}]}}]}}

CLAIMS:
{claims}"""


def propose_latents(question, *, horizon, llm):
    """LLM proposes the qualitative latent structure. Returns {latents:[...], combination:...}. Safe fallback."""
    if llm is None:
        return {"latents": [], "combination": "weighted_mean"}
    from swm.engine.grounding import parse_json
    try:
        raw = parse_json(llm(_PROPOSE_PROMPT.format(question=question, horizon=horizon))) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    lats = []
    for L in (raw.get("latents") or [])[:4]:
        if not isinstance(L, dict):
            continue
        t = str(L.get("type", "")).strip()
        if t not in LATENT_TYPES:
            t = "regime"
        lats.append({"id": str(L.get("id", f"lat{len(lats)}"))[:40], "type": t,
                     "definition": str(L.get("definition", ""))[:200],
                     "favorable_supports_yes": bool(L.get("favorable_supports_yes", True))})
    comb = str(raw.get("combination", "weighted_mean"))
    if comb not in COMBINATIONS:
        comb = "weighted_mean"
    return {"latents": lats, "combination": comb}


def map_claims_to_latents(latents, claims_brief, *, llm):
    """LLM maps each claim to latent(s) + direction (qualitative). Returns {claim_id: [{latent_id,direction}]}."""
    if llm is None or not latents:
        return {}
    from swm.engine.grounding import parse_json
    lat_txt = "; ".join(f"{L['id']} ({L['type']}): {L['definition']}" for L in latents)
    claims_txt = "\n".join(f"- {c['claim_id']}: {c['text'][:180]}" for c in claims_brief)
    try:
        raw = parse_json(llm(_MAP_PROMPT.format(latents=lat_txt, claims=claims_txt))) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    out = {}
    for m in (raw.get("map") or []):
        if not isinstance(m, dict):
            continue
        cid = str(m.get("claim_id", ""))
        links = [{"latent_id": str(l.get("latent_id", "")), "direction": str(l.get("direction", "neutral"))}
                 for l in (m.get("links") or []) if isinstance(l, dict)]
        if cid:
            out[cid] = links
    return out


# --------------------------------------------------------------------------- numeric inference (offline)
def _lr_for(direction, strength, reliability, obs_lr=None):
    """Likelihood ratio a 'favors' observation contributes for the favorable state. Uses a fitted LR when
    provided (from phase3_fitted_obs), else a hand-set strength/reliability table (mirrors DirectionalRateModel
    discrimination). 'against' inverts, 'neutral' -> 1.0."""
    if direction == "neutral":
        return 1.0
    if obs_lr is not None:
        lr = float(obs_lr)
    else:
        sens = {"weak": 0.58, "moderate": 0.72, "strong": 0.85}.get(strength, 0.72)
        rel = max(0.0, min(1.0, reliability))
        sens = 0.5 + rel * (sens - 0.5)
        lr = sens / max(_EPS, 1.0 - sens)                 # LR of a favorable-vote for the favorable state
    return lr if direction == "favors" else 1.0 / lr


def infer_latent_posteriors(latents, claim_links, tag_by_claim, *, lr_lookup=None):
    """Per-latent Beta posterior over the favorable-state probability, from mapped claims. Deterministic.
    `tag_by_claim[cid]` gives strength/reliability; `lr_lookup(claim)` optionally returns a fitted LR."""
    post = {}
    for L in latents:
        a, b = LATENT_TYPES.get(L["type"], DEFAULT_TYPE)["alpha"], LATENT_TYPES.get(L["type"], DEFAULT_TYPE)["beta"]
        log_lr = 0.0
        n_obs = 0
        for cid, links in claim_links.items():
            tag = tag_by_claim.get(cid, {})
            for link in links:
                if link.get("latent_id") != L["id"]:
                    continue
                d = link.get("direction", "neutral")
                if d == "neutral":
                    continue
                obs_lr = lr_lookup(tag) if lr_lookup else None
                lr = _lr_for(d, tag.get("strength", "moderate"), tag.get("reliability", 0.8), obs_lr=obs_lr)
                log_lr += math.log(max(_EPS, lr))
                n_obs += 1
        # convert accumulated log-LR into a pseudo-count update on the Beta (favorable vs unfavorable weight)
        # w = exp(log_lr) is the odds multiplier; fold into the Beta mean via a bounded pseudo-observation.
        mean0 = a / (a + b)
        odds = mean0 / (1 - mean0) * math.exp(log_lr)
        mean = odds / (1 + odds)
        strength = (a + b) + n_obs                          # posterior gets tighter with more obs
        post[L["id"]] = {"type": L["type"], "mean": max(_EPS, min(1 - _EPS, mean)),
                         "favorable_supports_yes": L["favorable_supports_yes"],
                         "n_obs": n_obs, "pseudo_count": strength}
    return post


def combine_to_rate(latents, posteriors, combination):
    """REGISTERED mechanism: combine per-latent favorable-state means into P(yes). Each latent's contribution
    is oriented by favorable_supports_yes (flip the mean if favorable pushes toward NO)."""
    oriented = []
    for L in latents:
        p = posteriors.get(L["id"])
        if not p:
            continue
        m = p["mean"] if L["favorable_supports_yes"] else (1 - p["mean"])
        oriented.append(m)
    if not oriented:
        return 0.5
    if combination == "necessary_conjunction":
        r = 1.0
        for m in oriented:
            r *= m
        return max(_EPS, min(1 - _EPS, r))
    if combination == "sufficient_disjunction":
        r = 1.0
        for m in oriented:
            r *= (1 - m)
        return max(_EPS, min(1 - _EPS, 1 - r))
    if combination == "single_driver":
        # the latent whose mean is furthest from 0.5 (most-informative) drives the outcome
        return max(oriented, key=lambda m: abs(m - 0.5))
    # weighted_mean: geometric mean in logit space (stable, symmetric)
    s = sum(math.log(m / (1 - m)) for m in oriented) / len(oriented)
    return max(_EPS, min(1 - _EPS, 1 / (1 + math.exp(-s))))


def causal_latent_rate(proposal, claim_links, tag_by_claim, *, lr_lookup=None):
    """Full offline path: proposal + mappings + tags -> (rate, per-latent posteriors). Deterministic."""
    latents = proposal.get("latents", [])
    if not latents:
        return None, {}
    posteriors = infer_latent_posteriors(latents, claim_links, tag_by_claim, lr_lookup=lr_lookup)
    rate = combine_to_rate(latents, posteriors, proposal.get("combination", "weighted_mean"))
    return rate, posteriors
