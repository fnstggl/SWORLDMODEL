"""Typed event-history & memory schema — Phase 7, Part 11.

Nonlinear mechanisms condition on the PAST: how many times a receiver was exposed, how long since the last
exposure, how bursty the arrivals were, how much a memory has decayed. This module represents an actor's
event history as a typed, append-only log on WorldState and derives history features from it — with one
inviolable rule: **every feature is computed from events at or before `now`; future events never enter.**
That is what makes history-conditioning leakage-free (the audit in `context.py` covers exogenous context;
this covers endogenous history).

The log lives in the entity's built-in `latent_state` namespace under typed keys, so no new WorldState schema
field is needed (additive, Phase-9/10-safe). `record_event` appends; the feature extractors read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

HISTORY_FEATURES = ("cum_count", "recency_h", "time_since_last_h", "time_since_last_action_h",
                    "mean_spacing_h", "burstiness", "event_age_h", "memory_decay", "novelty_decay",
                    "n_distinct_sources", "source_diversity", "rolling_count_24h", "rolling_count_7d",
                    "short_term_state", "long_term_state")

_EXPOSURE_KEY = "p7_exposure_log"     # [{"at": ts, "source": id, "kind": str}]
_ACTION_KEY = "p7_action_log"         # [{"at": ts, "action": str}]


def _log(entity, key):
    sf = entity.get(key, None) if hasattr(entity, "get") else None
    from swm.world_model_v2.state import StateField
    if isinstance(sf, StateField) and isinstance(sf.value, list):
        return sf.value
    return []


def record_exposure(entity, *, at: float, source: str = "", kind: str = "exposure"):
    """Append one exposure to the actor's typed log (idempotent-safe; ordered by insertion, times monotone
    in a correct rollout). Uses the built-in latent_state extension door — no new schema field."""
    from swm.world_model_v2.state import F
    log = list(_log(entity, _EXPOSURE_KEY))
    log.append({"at": float(at), "source": str(source), "kind": str(kind)})
    _ensure_extension()
    entity.set(_EXPOSURE_KEY, F(log, status="derived", method="p7_history", updated_at=at))
    return log


def record_action(entity, *, at: float, action: str):
    from swm.world_model_v2.state import F
    log = list(_log(entity, _ACTION_KEY))
    log.append({"at": float(at), "action": str(action)})
    _ensure_extension()
    entity.set(_ACTION_KEY, F(log, status="derived", method="p7_history", updated_at=at))
    return log


_EXT_DONE = False


def _ensure_extension():
    global _EXT_DONE
    if _EXT_DONE:
        return
    from swm.world_model_v2.state import register_entity_extension, extension_fields
    if _EXPOSURE_KEY not in extension_fields("person"):
        register_entity_extension("p7_history",
                                  fields={_EXPOSURE_KEY: "Phase 7 typed exposure event log",
                                          _ACTION_KEY: "Phase 7 typed action event log"},
                                  entity_types=("person", "institution"))
    _EXT_DONE = True


@dataclass
class HistoryWindow:
    """Declares HOW history is summarized for a mechanism: which features, decay constants, window sizes.
    Serialized into the mechanism instance so a run is replayable and the history spec is auditable."""
    features: tuple = HISTORY_FEATURES
    memory_tau_h: float = 24.0            # exponential memory decay half-life
    novelty_tau_h: float = 12.0           # novelty decay
    short_window_h: float = 24.0
    long_window_h: float = 168.0
    refractory_h: float = 0.0

    def as_dict(self):
        return {"features": list(self.features), "memory_tau_h": self.memory_tau_h,
                "novelty_tau_h": self.novelty_tau_h, "short_window_h": self.short_window_h,
                "long_window_h": self.long_window_h, "refractory_h": self.refractory_h}


def history_features(entity, *, now: float, window: HistoryWindow | None = None,
                     source_of_current: str = "") -> dict:
    """Compute the typed history feature vector from events STRICTLY at or before `now`.

    Any event with at > now is dropped (a hard leakage guard — even if a buggy caller queued a future event
    into the log, it cannot enter a feature). Returns a dict keyed by HISTORY_FEATURES."""
    w = window or HistoryWindow()
    exp = [e for e in _log(entity, _EXPOSURE_KEY) if float(e["at"]) <= now]      # <= now: no future
    act = [a for a in _log(entity, _ACTION_KEY) if float(a["at"]) <= now]
    times = sorted(float(e["at"]) for e in exp)
    out = {f: 0.0 for f in HISTORY_FEATURES}
    out["cum_count"] = float(len(exp))
    if times:
        last = times[-1]
        out["recency_h"] = out["time_since_last_h"] = max(0.0, (now - last) / 3600.0)
        out["event_age_h"] = max(0.0, (now - times[0]) / 3600.0)
        if len(times) >= 2:
            gaps = [(times[j + 1] - times[j]) / 3600.0 for j in range(len(times) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            out["mean_spacing_h"] = mean_gap
            if mean_gap > 0 and len(gaps) >= 2:
                var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                sd = math.sqrt(var)
                # Goh–Barabási burstiness B = (σ−μ)/(σ+μ) ∈ [−1,1]; +1 bursty, −1 regular
                out["burstiness"] = (sd - mean_gap) / (sd + mean_gap) if (sd + mean_gap) > 0 else 0.0
        # exponential memory: Σ exp(−age/τ) — recency-weighted cumulative pressure
        out["memory_decay"] = sum(math.exp(-max(0.0, (now - t) / 3600.0) / (w.memory_tau_h or 1.0))
                                  for t in times)
        out["novelty_decay"] = math.exp(-out["recency_h"] / (w.novelty_tau_h or 1.0))
        out["rolling_count_24h"] = float(sum(1 for t in times if (now - t) <= w.short_window_h * 3600.0))
        out["rolling_count_7d"] = float(sum(1 for t in times if (now - t) <= w.long_window_h * 3600.0))
        out["short_term_state"] = out["rolling_count_24h"]
        out["long_term_state"] = out["memory_decay"]
    srcs = [str(e.get("source", "")) for e in exp if e.get("source")]
    out["n_distinct_sources"] = float(len(set(srcs)))
    out["source_diversity"] = float(len(set(srcs))) / max(1.0, float(len(srcs))) if srcs else 0.0
    if act:
        last_a = max(float(a["at"]) for a in act)
        out["time_since_last_action_h"] = max(0.0, (now - last_a) / 3600.0)
    # refractory flag (1.0 = within refractory window since last exposure → suppressed)
    out["_refractory_active"] = 1.0 if (times and w.refractory_h > 0
                                        and out["time_since_last_h"] < w.refractory_h) else 0.0
    return out
