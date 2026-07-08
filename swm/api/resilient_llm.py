"""Resilient LLM backend for the large-scale backtest — HF router (DeepSeek-V3) first, DeepSeek direct as
fallback, with an on-disk response cache so the ablation sweeps re-use every call for free.

Priority (per the run config): the HF Inference router serving `deepseek-ai/DeepSeek-V3-0324` is tried first
(the HF token's quota), and the direct DeepSeek API is the fallback when HF errors or is exhausted. A prompt is
cached by (model-tier, prompt) hash on disk, so re-running the harness — and especially the Stage-2 ablations,
which re-simulate the same compiled specs — costs zero new tokens. Keys are read from the environment only,
never stored or logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
HF_MODEL = "deepseek-ai/DeepSeek-V3-0324"
CACHE_DIR = "data/llm_cache"                      # gitignored; persists across runs


def _post(url, key, model, prompt, system, max_tokens, temperature, timeout):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}",
                                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


class ResilientLLM:
    """callable(prompt) -> text. HF DeepSeek-V3 first, DeepSeek direct fallback, on-disk cached."""

    def __init__(self, *, system="", max_tokens=1100, temperature=0.0, timeout=90, cache_dir=CACHE_DIR,
                 retries=2):
        self.system, self.max_tokens, self.temperature, self.timeout = system, max_tokens, temperature, timeout
        self.retries = retries
        self.hf_key = os.environ.get("HF_TOKEN", "")
        self.ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = {"cache": 0, "hf": 0, "deepseek": 0, "fail": 0}

    def _key(self, prompt):
        h = hashlib.sha256(f"{self.system}\x00{self.max_tokens}\x00{prompt}".encode()).hexdigest()[:40]
        return self.cache / f"{h}.json"

    def __call__(self, prompt: str) -> str:
        cp = self._key(prompt)
        if cp.exists():
            self.calls["cache"] += 1
            return json.loads(cp.read_text())["text"]
        text = None
        for attempt in range(self.retries + 1):
            if self.hf_key:
                try:
                    text = _post(HF_ROUTER, self.hf_key, HF_MODEL, prompt, self.system, self.max_tokens,
                                 self.temperature, self.timeout)
                    self.calls["hf"] += 1
                    break
                except Exception:
                    pass
            if self.ds_key:
                try:
                    text = _post(DEEPSEEK_URL, self.ds_key, "deepseek-chat", prompt, self.system,
                                 self.max_tokens, self.temperature, self.timeout)
                    self.calls["deepseek"] += 1
                    break
                except Exception:
                    pass
            time.sleep(1.5 * (attempt + 1))
        if text is None:
            self.calls["fail"] += 1
            return ""
        cp.write_text(json.dumps({"text": text}))
        return text


def resilient_chat_fn(**kw):
    return ResilientLLM(**kw)
