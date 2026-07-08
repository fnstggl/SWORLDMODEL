"""Embedding-keyed prior registry — let calibrated elasticities transfer across PHRASINGS and DOMAINS.

The string-keyed `PriorRegistry` only reuses an elasticity across EXACT variable/outcome names. At corpus
scale we want a weight learned for "inflation → rate hike" to inform "price growth → policy tightening", and
"mood → reply" to inform "affect → response". This wraps the registry with a pluggable EMBEDDING: on a miss,
it finds the nearest known (variable, outcome-class) by cosine similarity and returns its prior with an
sd WIDENED by the transfer distance (1 − similarity) — an honest "borrowed, less certain" weight.

The default `lexical_embed` is offline (token-bag cosine → transfers phrasing variants). A real sentence-
embedding model plugs into `embed_fn` for true semantic transfer (synonyms/paraphrases) — the peak version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.variables.calibrated_weights import WeightPrior
from swm.variables.prior_registry import PriorRegistry


def lexical_embed(text: str, d: int = 96) -> list:
    """Offline default: a normalized token-bag vector over a hashed space. Captures phrasing overlap
    ('inflation rate' ↔ 'inflation'), not synonymy — swap in a real embedding model for that."""
    v = [0.0] * d
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        v[hash(t) % d] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass
class EmbeddingPriorRegistry:
    """Wrap a `PriorRegistry` with embedding-based nearest-neighbor fallback for cross-phrasing transfer."""
    base: PriorRegistry
    embed_fn: object = lexical_embed
    threshold: float = 0.6           # minimum cosine similarity to transfer a weight
    _index: list = field(default_factory=list)

    def build_index(self):
        """Embed each stored key's '(variable, outcome-class)' text once for nearest-neighbor search. Keys
        whose embedding is unavailable (embed_fn returns None) are skipped — never mix embedding spaces."""
        self._index = []
        for key, rec in self.base.records.items():
            emb = self.embed_fn(key.replace("|", " "))
            if emb is not None:
                self._index.append((key, emb, rec))
        return self

    def get(self, variable, outcome_class, *, min_n=1):
        exact = self.base.get(variable, outcome_class, min_n=min_n)
        if exact is not None:
            return exact
        if not self._index:
            self.build_index()
        q = self.embed_fn(f"{variable} {outcome_class}")
        if q is None:
            return None
        best_sim, best_rec = self.threshold, None
        for _key, emb, rec in self._index:
            if rec.n < min_n:
                continue
            sim = _cos(q, emb)
            if sim > best_sim:
                best_sim, best_rec = sim, rec
        if best_rec is None:
            return None
        widened = best_rec.sd * (1.0 + 2.0 * (1.0 - best_sim))   # transfer distance widens the CI (honest)
        return WeightPrior(variable, best_rec.mean, widened, source=f"transfer(sim={best_sim:.2f},n={best_rec.n})")

    def prior_for(self, variable, outcome_class, *, fallback=None):
        got = self.get(variable, outcome_class)
        if got is not None:
            return got
        if fallback is not None:
            return fallback
        return WeightPrior(variable, 0.0, 3.0, source="uninformative")
