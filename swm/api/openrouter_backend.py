"""OpenRouter LLM backend — access to model families with OLD training cutoffs for pastcasting.

Same pluggable `fn(prompt) -> text` contract as `deepseek_backend.deepseek_chat_fn`. Exists for one
reason: leak-free pastcasting requires a model whose training cutoff PREDATES the question window
(EXP-101 lesson: DeepSeek-V4's real cutoff is Apr 2026 — fine for BTF-3's May-Jul 2026 resolutions,
contaminated for BTF-2's Oct-Dec 2025 ones; self-reported cutoffs cannot be trusted either way).
OpenRouter serves dated snapshots (e.g. deepseek/deepseek-chat-v3-0324, cutoff ~mid-2024) that are
clean for both. The API key is read from OPENROUTER_API_KEY env ONLY (never stored, never logged).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_chat_fn(model: str = "deepseek/deepseek-chat-v3-0324", *, system: str = "",
                       max_tokens: int = 800, temperature: float = 0.0):
    """Return a callable(prompt) -> text via OpenRouter. Reads OPENROUTER_API_KEY from env only."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key and not key.isascii():
        raise ValueError("OPENROUTER_API_KEY contains non-ASCII characters — likely corrupted on paste.")

    def fn(prompt: str, *, max_tokens: int = None, temperature: float = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = json.dumps({"model": model, "messages": msgs,
                           "max_tokens": int(max_tokens if max_tokens is not None else fn.default_max_tokens),
                           "temperature": (temperature if temperature is not None
                                           else fn.default_temperature)}).encode()
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(OPENROUTER_URL, data=body,
                                             headers={"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
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
            import time
            time.sleep(2 * (attempt + 1))
        raise last

    fn.default_max_tokens, fn.default_temperature = max_tokens, temperature
    return fn
