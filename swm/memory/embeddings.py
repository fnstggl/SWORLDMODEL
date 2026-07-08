"""Dependency-free text embedding — shared retrieval infra (audit C.4).

The core `swm/` library is dependency-free on purpose, so the episodic store cannot pull in a sentence-
transformer. This module gives a transparent, deterministic **hashing embedding**: tokenize → hash each
token into a fixed number of buckets with tf weighting → L2-normalize. Cosine similarity of two such
vectors is a cheap bag-of-words relevance that is good enough to rank "which of this person's past
episodes is about the same thing as this message" — the *relevance* channel of Generative-Agents-style
retrieval — with zero dependencies and no network.

It is **pluggable**: `TextEmbedder(embed_fn=...)` swaps in a real embedding backend (Anthropic / a local
sentence model) in production, keeping the exact same interface the store and tests use. The default
hashing backend is what the offline tests and the EXP harness run on, so the mechanism is validated on the
transparent path and the production path is a drop-in upgrade.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
_DEFAULT_DIM = 256


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _bucket(token: str, dim: int) -> int:
    # deterministic, salt-free hash (Python's hash() is salted per-process → not reproducible); FNV-1a.
    h = 0x811C9DC5
    for ch in token:
        h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return h % dim


def hashing_embed(text: str, dim: int = _DEFAULT_DIM) -> list[float]:
    """tf hashing embedding, L2-normalized. Deterministic and dependency-free."""
    vec = [0.0] * dim
    toks = tokenize(text)
    if not toks:
        return vec
    for t in toks:
        vec[_bucket(t, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs are assumed L2-normalized (hashing_embed guarantees it); if not, this is
    still a valid dot product but not bounded to [-1,1]. Returns 0 for an empty/zero vector."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


@dataclass
class TextEmbedder:
    """Pluggable embedder. Default = the transparent hashing backend; inject `embed_fn` for production."""
    dim: int = _DEFAULT_DIM
    embed_fn: object = None                 # optional callable(text) -> list[float]

    def __call__(self, text: str) -> list[float]:
        if self.embed_fn is not None:
            try:
                v = self.embed_fn(text)
                if v:
                    return list(v)
            except Exception:
                pass
        return hashing_embed(text, self.dim)

    def similarity(self, a_text_or_vec, b_text_or_vec) -> float:
        a = a_text_or_vec if isinstance(a_text_or_vec, list) else self(a_text_or_vec)
        b = b_text_or_vec if isinstance(b_text_or_vec, list) else self(b_text_or_vec)
        return cosine(a, b)
