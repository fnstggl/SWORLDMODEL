"""Retrieval-augmented response_fn — situation-conditioned recall on top of the global persona.

The Level-1 response model (`swm/simulation/response_model.py`) scores P(respond) from a person's *global*
persona + transient state + the message. It has no access to **how this specific person reacted to similar
situations before, recently** — the single most predictive signal for a history-driven outcome (reply,
churn, adherence). This wrapper adds exactly that, and nothing else:

  p_final = σ( logit(p_base) + β · memory_signal )

where, at prediction time and strictly as-of the current moment (no leakage):
  - retrieve the person's top-k episodes relevant to THIS message (recency × importance × relevance);
  - `personal_base` = the person's own past response rate — so the signal isolates *this kind of message*
    from *this person's* overall responsiveness (which the base model already sees via their persona);
  - `observed_shrunk` = a **Beta-Binomial posterior** for the response rate on the retrieved *similar* past
    contacts, shrunk toward `personal_base` with strength κ and weighted by each episode's similarity ×
    recency: `(κ·base + Σ wᵢ oᵢ) / (κ + Σ wᵢ)`. Thin or off-topic history → posterior ≈ base → no move;
    plentiful, on-topic, consistent history → posterior moves fully to the situation-specific rate. This
    shrinkage IS the evidence weighting — it is calibrated by construction, not an asserted magnitude;
  - `memory_signal` = logit(observed_shrunk) − logit(personal_base), clamped.

With the default β = 1.0 and the common case p_base = personal_base, p_final collapses to exactly
`observed_shrunk` — the prediction becomes the shrunk situation-conditioned response rate, an interpretable
and honest estimator. The wrapper keeps the `(variables, state, message) -> {"p", ...}` contract, so it
drops into `IndividualSimulator` unchanged. It reads the entity id and the as-of time from framework keys
on the message (`_entity_id`, `_as_of`); absent either, it returns the base prediction untouched. κ and β
are hyperparameters that EARN their place on the held-out backtest (EXP-074) — a mechanism, not a
guaranteed win, exactly like every other coupling in the repo.
"""
from __future__ import annotations

import math

_CLAMP = 4.0            # max |memory_signal| in log-odds (a similar-history nudge, not an override)
_KAPPA = 6.0            # Beta-Binomial shrinkage toward the personal base (pseudo-episodes); tuned in EXP-074:
#                        maximizes held-out history-driven skill and keeps the uninformative case self-limiting


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


def _message_text(message: dict) -> str:
    """A query string for retrieval: prefer explicit text/topic, else stitch the human-readable fields."""
    if message.get("text"):
        return str(message["text"])
    if message.get("topic"):
        return str(message["topic"])
    parts = [str(v) for k, v in message.items()
             if isinstance(v, str) and not k.startswith("_")]
    return " ".join(parts)


def memory_signal(hits: list, stream, as_of: float, *, outcome_key: str = "responded",
                  kappa: float = _KAPPA) -> dict:
    """Compute the situation-conditioned response signal from retrieved episodes via a Beta-Binomial
    posterior shrunk toward the person's own base rate. Returns the signal + diagnostics. `hits` are the
    dicts returned by `MemoryStream.retrieve`."""
    personal_base = stream.personal_response_rate(as_of) if stream is not None else None
    contact_hits = [h for h in hits if outcome_key in h["episode"].meta]
    if personal_base is None or not contact_hits:
        return {"signal": 0.0, "evidence_weight": 0.0, "personal_base": personal_base,
                "n_contact_hits": len(contact_hits), "observed": None}
    # weight each similar past contact by relevance² × recency: relevance² sharpens topic focus (marginally
    # related episodes contribute little), recency favors recent behavior (drift). importance already
    # shaped which episodes surfaced in retrieval.
    num = den = 0.0
    for h in contact_hits:
        w = (max(0.0, h["relevance"]) ** 2) * max(0.0, h["recency"])
        num += w * (1.0 if h["episode"].meta.get(outcome_key) else 0.0)
        den += w
    # Beta-Binomial posterior mean, shrunk toward the personal base by κ pseudo-episodes. Thin/off-topic
    # evidence (small den) → posterior ≈ base → signal ≈ 0. This shrinkage IS the calibration.
    observed_shrunk = (kappa * personal_base + num) / (kappa + den)
    sig = max(-_CLAMP, min(_CLAMP, _logit(observed_shrunk) - _logit(personal_base)))
    ev_weight = den / (kappa + den)                              # fraction of the posterior from real data
    return {"signal": sig, "evidence_weight": max(0.0, min(1.0, ev_weight)), "observed": observed_shrunk,
            "personal_base": personal_base, "n_contact_hits": len(contact_hits), "weight_mass": round(den, 4)}


def retrieval_augmented_response_fn(base_fn, store, *, beta: float = 1.0, k: int = 8, kappa: float = _KAPPA,
                                    outcome_key: str = "responded", include_reflections: bool = True):
    """Wrap a base response_fn with as-of episodic retrieval. `store` is an `EpisodicStore`. The message
    must carry `_entity_id` and `_as_of` for augmentation to fire; otherwise the base fn passes through.
    `beta` scales the nudge and `kappa` sets the shrinkage toward the personal base — both are meant to be
    tuned on a validation split (higher κ / lower β = more conservative)."""
    def fn(variables, state, message):
        base = base_fn(variables, state, message)
        eid = message.get("_entity_id")
        now = message.get("_as_of")
        if eid is None or now is None:
            return base                                    # not an individual-regime call → pass through
        query = _message_text(message)
        hits = store.retrieve(eid, query, as_of=now, k=k, include_reflections=include_reflections)
        stream = store.stream(eid) if eid in store else None    # unseen entity → no personal base → no move
        ms = memory_signal(hits, stream, now, outcome_key=outcome_key, kappa=kappa)
        out = dict(base)
        if ms["signal"] != 0.0:                            # shrinkage already scaled the signal by evidence
            out["p"] = _sigmoid(_logit(base["p"]) + beta * ms["signal"])
        out["memory"] = {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in ms.items()}
        return out
    return fn
