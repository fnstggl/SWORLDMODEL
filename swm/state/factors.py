"""Factor registry (audit's core principle + the user's "map every relevant variable").

Reconciliation: be EXPANSIVE about what you *map* (admit every decision-relevant candidate),
be STRICT about what you *keep* (only factors whose ablation worsens held-out prediction /
decision lift survive as KEEP; the rest are EXPERIMENTAL or DROP). The registry is the map;
ablation.py is the filter.

Each factor declares metadata so leakage and provenance are auditable, plus:
  extract(entity, action, context) -> float   (the feature value at prediction time)
  update(entity, context, action, magnitude)   (how state transitions after an outcome; entity/
                                                 context factors only) — this is what makes the
                                                 model a *state-transition* model, not a static one.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from swm.state.state import Action, ContextState, EntityState, Posterior

EMA_W = 0.35  # weight of a new observation in slow state updates


@dataclass
class Factor:
    name: str
    kind: str                    # "entity" | "action" | "context"
    source: str
    timescale: str               # static | fast | slow | stable
    leakage_risk: str            # low | med | high
    extract: Callable[[EntityState, Action, ContextState], float]
    update: Callable[[EntityState, ContextState, Action, float], None] | None = None
    allowed_evidence: str = "pre-as_of only"
    status: str = "EXPERIMENTAL"  # KEEP | EXPERIMENTAL | DROP


class FactorRegistry:
    def __init__(self) -> None:
        self._f: dict[str, Factor] = {}

    def register(self, f: Factor) -> None:
        self._f[f.name] = f

    def all(self) -> list[Factor]:
        return list(self._f.values())

    def active(self, exclude: tuple[str, ...] = ()) -> list[Factor]:
        return [f for f in self._f.values()
                if f.status in ("KEEP", "EXPERIMENTAL") and f.name not in exclude]

    def names(self, exclude: tuple[str, ...] = ()) -> list[str]:
        return [f.name for f in self.active(exclude)]

    def vector(self, entity: EntityState, action: Action, context: ContextState,
               exclude: tuple[str, ...] = ()) -> list[float]:
        return [f.extract(entity, action, context) for f in self.active(exclude)]

    def apply_update(self, entity: EntityState, context: ContextState, action: Action,
                     magnitude: float) -> None:
        """State transition: mutate entity + context factors given the observed/sampled outcome."""
        for f in self._f.values():
            if f.update is not None:
                f.update(entity, context, action, magnitude)

    def set_status(self, name: str, status: str) -> None:
        self._f[name].status = status


# ---------------------------------------------------------------------------
# The HN factor set. Entity = author; action = story submission; magnitude = score.
# Expansive on purpose; ablation decides survival.
# ---------------------------------------------------------------------------

def _ls(mag: float) -> float:      # log-score, the natural scale for heavy-tailed HN points
    return math.log1p(max(0.0, mag))


def build_hn_registry() -> FactorRegistry:
    r = FactorRegistry()

    # --- ENTITY (author) latent state — these UPDATE after each post (true transition) ---
    r.register(Factor(
        "author_quality", "entity", "author_history", "slow", "low",
        extract=lambda e, a, c: e.stable_traits.get("quality", Posterior(_ls(3), 1)).mean,
        update=lambda e, c, a, m: e.stable_traits.setdefault(
            "quality", Posterior(_ls(3), 1)).observe(_ls(m), EMA_W * 3)))
    r.register(Factor(
        "author_ceiling", "entity", "author_history", "stable", "low",
        extract=lambda e, a, c: e.stable_traits.get("ceiling", Posterior(_ls(3), 1)).mean,
        update=lambda e, c, a, m: e.stable_traits.__setitem__(
            "ceiling", Posterior(max(e.stable_traits.get("ceiling", Posterior(0)).mean, _ls(m)),
                                 e.stable_traits.get("ceiling", Posterior(0)).n + 1))))
    r.register(Factor(
        "author_standing", "entity", "author_history", "slow", "low",
        extract=lambda e, a, c: e.relationship_stance.get("standing", Posterior(0.13, 1)).mean,
        update=lambda e, c, a, m: e.relationship_stance.setdefault(
            "standing", Posterior(0.13, 1)).observe(1.0 if m >= 10 else 0.0, EMA_W * 2)))
    r.register(Factor(
        "author_volume", "entity", "author_history", "slow", "low",
        extract=lambda e, a, c: math.log1p(e.history_features.get("n_posts", 0)),
        update=lambda e, c, a, m: e.history_features.__setitem__(
            "n_posts", e.history_features.get("n_posts", 0) + 1)))
    r.register(Factor(
        "author_recency", "entity", "author_history", "fast", "low",
        extract=lambda e, a, c: e.current_attention.get("recency_log", Posterior(12, 1)).mean,
        update=lambda e, c, a, m: e.current_attention.__setitem__(
            "recency_log", Posterior(0.0, 1))))   # reset: they just posted

    # --- ACTION (content/timing) — stateless, derived from the submission ---
    r.register(Factor("title_len", "action", "content", "static", "low",
                      extract=lambda e, a, c: a.content_features.get("title_len", 0.5)))
    r.register(Factor("is_show", "action", "content", "static", "low",
                      extract=lambda e, a, c: a.content_features.get("is_show", 0.0)))
    r.register(Factor("is_ask", "action", "content", "static", "low",
                      extract=lambda e, a, c: a.content_features.get("is_ask", 0.0)))
    r.register(Factor("is_text", "action", "content", "static", "low",
                      extract=lambda e, a, c: a.content_features.get("is_text", 0.0)))
    r.register(Factor("hour_sin", "action", "time", "static", "low",
                      extract=lambda e, a, c: math.sin(2 * math.pi * a.timing.get("hour", 12) / 24)))
    r.register(Factor("hour_cos", "action", "time", "static", "low",
                      extract=lambda e, a, c: math.cos(2 * math.pi * a.timing.get("hour", 12) / 24)))
    r.register(Factor("is_weekend", "action", "time", "static", "low",
                      extract=lambda e, a, c: 1.0 if a.timing.get("weekday", 0) >= 5 else 0.0))

    # --- CONTEXT (domain / topic) — these UPDATE too (shared-state transition) ---
    r.register(Factor(
        "domain_reputation", "context", "domain_history", "slow", "low",
        extract=lambda e, a, c: c.domain_reputation.get(
            a.meta.get("domain", ""), Posterior(_ls(3), 1)).mean,
        update=lambda e, c, a, m: c.domain_reputation.setdefault(
            a.meta.get("domain", ""), Posterior(_ls(3), 1)).observe(_ls(m), EMA_W * 2)))
    r.register(Factor(
        "topic_salience", "context", "topic_history", "fast", "low",
        extract=lambda e, a, c: c.topic_salience.get(
            a.content_features.get("topic", "other"), Posterior(0.13, 1)).mean
        if isinstance(a.content_features.get("topic"), str) else 0.13,
        update=lambda e, c, a, m: c.topic_salience.setdefault(
            str(a.content_features.get("topic", "other")), Posterior(0.13, 1)
        ).observe(1.0 if m >= 10 else 0.0, EMA_W)))

    return r


TOPIC_KEYWORDS = {
    "ai": ["ai", "llm", "gpt", "claude", "gemini", "model", "agent", "openai", "anthropic",
           "neural", "ml "],
    "crypto": ["crypto", "bitcoin", "blockchain", "ethereum", "web3", "nft", "defi"],
    "security": ["security", "vuln", "exploit", "cve", "hack", "breach", "password", "malware"],
    "hardware": ["chip", "gpu", "cpu", "silicon", "hardware", "arm", "risc", "fpga", "nvidia"],
    "science": ["study", "research", "quantum", "physics", "brain", "gene", "climate", "space",
                "nasa", "astronom"],
    "programming": ["rust", "python", "compiler", "kernel", "database", "javascript", "golang",
                    "type", "framework", "linux"],
    "business": ["startup", "vc", "funding", "ipo", "layoff", "acqui", "revenue", "market"],
    "politics": ["trump", "senate", "policy", "government", "election", "regulation", "eu ", "law"],
}


def tag_topic(title: str) -> str:
    t = title.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in t for k in kws):
            return topic
    return "other"
