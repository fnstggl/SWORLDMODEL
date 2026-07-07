"""HuggingFace-router LLM backend — a SEPARATE model to compile specs, for honest spec-quality testing.

The compiler's LLM backend is pluggable. For an honest spec-quality benchmark we must NOT grade the same
model that authored the spec, and we must not let outcome knowledge leak in. This backend runs an external
open model (via the HF router's OpenAI-compatible endpoint) so the specs are produced by a model that is
not the grader and has no privileged view of this session's outcomes. Token comes from the environment
only (never stored); calls are resilient and meant to be cached to committed files for reproducibility.
"""
from __future__ import annotations

import json
import os
import urllib.request

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"


def hf_chat_fn(model: str = "Qwen/Qwen2.5-72B-Instruct", *, system: str = "", max_tokens: int = 400,
               temperature: float = 0.0):
    """Return a callable(prompt) -> text using the HF router. Reads HF_TOKEN from the env (never stored)."""
    tok = os.environ.get("HF_TOKEN", "")

    def fn(prompt: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                           "temperature": temperature}).encode()
        req = urllib.request.Request(HF_ROUTER, data=body,
                                     headers={"Authorization": f"Bearer {tok}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]
    return fn
