"""DeepSeek LLM backend — the default production model for the compiler and the LLM judges.

Stronger than the free Qwen path used in the earlier experiments (it picks mechanisms and estimates
variables more reliably). Same pluggable `fn(prompt) -> text` contract as `hf_backend.hf_chat_fn` and the
`anthropic_*` backends, so it drops into `StructuralCompiler`, the semantic judges, the persona inference,
etc. The API key is read from the environment ONLY (never stored, never logged); calls are OpenAI-compatible.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def deepseek_chat_fn(model: str = "deepseek-chat", *, system: str = "", max_tokens: int = 800,
                     temperature: float = 0.0):
    """Return a callable(prompt) -> text using the DeepSeek API. Reads DEEPSEEK_API_KEY from env only.
    model: 'deepseek-chat' (flagship V3 chat) or 'deepseek-reasoner' (R1)."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    # a pasted key can pick up non-ASCII junk (smart quotes, bullets, a trailing comment) → the HTTP header
    # crashes with an opaque latin-1 UnicodeEncodeError. Fail LOUDLY and clearly instead.
    if key and not key.isascii():
        raise ValueError("DEEPSEEK_API_KEY contains non-ASCII characters — it was likely corrupted on paste. "
                         "Re-set it by hand (no smart quotes / trailing comment): export DEEPSEEK_API_KEY=sk-…")

    def fn(prompt: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                           "temperature": temperature}).encode()
        # bounded retry on TRANSIENT failures (connection resets, 429/5xx, timeouts) so a single network
        # blip cannot kill a long batch. Deterministic backoff (no jitter — Math.random is unavailable
        # in some sandboxes and determinism aids replay). Non-transient errors (4xx auth) raise at once.
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(DEEPSEEK_URL, data=body,
                                             headers={"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read())
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last = e
                if attempt == 4:
                    raise
            time.sleep(2 ** attempt)                       # 1, 2, 4, 8s
        raise last                                         # unreachable (loop raises), keeps type-checkers happy
    return fn


def default_chat_fn(*, system: str = "", max_tokens: int = 800, temperature: float = 0.0):
    """The standard backend selector, used going forward: DeepSeek if its key is set, else the HF router
    (Qwen), else None (callers fall back to their cached/committed judgments). One place to change the
    production model."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return deepseek_chat_fn(system=system, max_tokens=max_tokens, temperature=temperature)
    if os.environ.get("HF_TOKEN"):
        from swm.api.hf_backend import hf_chat_fn
        return hf_chat_fn(system=system, max_tokens=max_tokens, temperature=temperature)
    return None
