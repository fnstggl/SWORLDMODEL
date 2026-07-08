"""Real embedding backends for the prior registry — true semantic elasticity transfer across the corpus.

The lexical default only transfers phrasing overlap. A real sentence embedding transfers MEANING: a weight
learned for "inflation → rate hike" informs "price growth → policy tightening"; "mood → reply" informs
"affect → response". This provides pluggable production embedders (HuggingFace Inference API, OpenAI) behind
the same `embed_fn(text) -> vector` seam the `EmbeddingPriorRegistry` already uses, plus a disk cache so the
592-key corpus is embedded ONCE and committed — the semantic transfer then works offline and reproducibly.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field


def hf_embed_fn(model: str = "sentence-transformers/all-MiniLM-L6-v2", token: str = None, timeout: int = 45):
    """Production embedder via the HuggingFace Inference API (pure stdlib). Accepts a string or a list;
    returns one vector or a list of vectors. Uses `HF_TOKEN` if `token` is not given."""
    token = token or os.environ.get("HF_TOKEN")
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"

    def fn(texts):
        single = isinstance(texts, str)
        body = json.dumps({"inputs": [texts] if single else list(texts)}).encode()
        req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {token}",
                                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out[0] if single else out
    return fn


def openai_embed_fn(api_key: str = None, model: str = "text-embedding-3-small", timeout: int = 45):
    """Production embedder via the OpenAI embeddings API (alternative backend)."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def fn(texts):
        single = isinstance(texts, str)
        body = json.dumps({"model": model, "input": [texts] if single else list(texts)}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body,
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        vecs = [d["embedding"] for d in data["data"]]
        return vecs[0] if single else vecs
    return fn


@dataclass
class EmbeddingCache:
    """Disk cache of text -> embedding, so the corpus is embedded once and reused offline (committed JSON)."""
    path: str
    vecs: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        try:
            return cls(path, json.loads(open(path).read()))
        except (FileNotFoundError, ValueError):
            return cls(path, {})

    def save(self, ndigits=4):
        with open(self.path, "w") as f:
            json.dump({k: [round(x, ndigits) for x in v] for k, v in self.vecs.items()}, f)
        return self

    def precompute(self, texts, live_fn, batch=64):
        """Batch-embed any missing texts via `live_fn` and cache them."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.vecs]
        for i in range(0, len(missing), batch):
            chunk = missing[i:i + batch]
            for t, v in zip(chunk, live_fn(chunk)):
                self.vecs[t] = v
        return self

    def embed_fn(self, live_fn=None):
        """A `text -> vector | None` function: cached first, then live (and cache it), else None (skip transfer
        — never mix embedding spaces of different dimensions)."""
        def fn(text):
            if text in self.vecs:
                return self.vecs[text]
            if live_fn is not None:
                v = live_fn(text)
                self.vecs[text] = v
                return v
            return None
        return fn
