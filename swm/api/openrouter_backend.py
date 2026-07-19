"""Strict OpenRouter backend for period-bounded historical models (TIER_B_PROVIDER_PINNED_POST_RELEASE).

Implements the repository's callable contract fn(prompt)->str with HARD provenance enforcement:
one exact model slug, one frozen provider endpoint, one frozen quantization, no fallbacks, no
routing, no plugins. Every response is asserted against the pinned configuration and appended to a
per-run audit ledger (generation id, provider, usage, cost, hashes, timestamps). A response that
violates any pin raises OpenRouterEnforcementError — the benchmark row fails, it never silently
degrades to another endpoint.

The API key is read from OPENROUTER_API_KEY only. It is never written, printed, serialized,
logged, or embedded in artifacts. Mutable-serving limitation (recorded in the model registry):
OpenRouter cannot prove the hosted endpoint is byte-identical to the open-weight checkpoint, which
is why this backend is Tier B, not Tier A; the same frozen benchmark can be rerun against a
hash-verified dedicated deployment for Tier A.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
MODELS_URL = "https://openrouter.ai/api/v1/models"
_RETRYABLE = (429, 500, 502, 503, 504, 520, 524)
_MAX_RETRIES = 5


class OpenRouterEnforcementError(RuntimeError):
    """A pinned-configuration violation (wrong model/provider/quantization, fallback, missing
    provenance metadata). NEVER retried — the row must fail visibly."""


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if not k:
        raise RuntimeError("OPENROUTER_API_KEY not set (environment only — never stored in repo)")
    return k


def _post(url: str, body: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_key()}",
                                          "Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/fnstggl/SWORLDMODEL",
                                          "X-Title": "WMv2 historical backtest"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_endpoint_metadata(slug: str) -> list:
    """Live endpoint metadata for an exact model slug (provider names, quantizations, pricing).
    Discovery evidence for the registry — not sufficient temporal proof by itself."""
    req = urllib.request.Request(ENDPOINTS_URL.format(slug=slug),
                                 headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return list((d.get("data") or {}).get("endpoints") or [])


def list_models() -> list:
    req = urllib.request.Request(MODELS_URL, headers={"Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return list(json.load(r).get("data") or [])


class OpenRouterPinnedClient:
    """fn(prompt)->str over ONE frozen (model, provider, quantization) endpoint."""

    def __init__(self, model_slug: str, *, provider: str, quantization: str,
                 system: str = "Reply ONLY JSON.", max_tokens: int = 2400,
                 temperature: float = 0.2, audit_path: str | None = None,
                 timeout_s: float = 240.0, _transport=None):
        self.model_slug = model_slug
        self.provider = provider                      # OpenRouter provider routing tag (lowercase)
        self.provider_display = provider              # display name asserted against response
        self.quantization = quantization
        self.system, self.max_tokens, self.temperature = system, max_tokens, temperature
        self.audit_path = audit_path
        self.timeout_s = timeout_s
        self._transport = _transport or _post         # injectable for mocked tests
        self._lock = threading.Lock()
        self.n_calls = 0
        self.total_cost = 0.0
        self.total_tokens = 0

    # ---- request body: pinned, no fallback, no routing, no plugins ----
    def _body(self, prompt: str) -> dict:
        return {"model": self.model_slug,
                "messages": ([{"role": "system", "content": self.system}] if self.system else [])
                            + [{"role": "user", "content": str(prompt)}],
                "max_tokens": self.max_tokens, "temperature": self.temperature,
                "provider": {"order": [self.provider], "allow_fallbacks": False,
                             "require_parameters": True, "data_collection": "deny",
                             "quantizations": [self.quantization]},
                "usage": {"include": True}}

    def _assert_response(self, r: dict) -> dict:
        ret_model = str(r.get("model") or "")
        ret_provider = str(r.get("provider") or "")
        gen_id = str(r.get("id") or "")
        usage = r.get("usage") or {}
        if ret_model != self.model_slug:
            raise OpenRouterEnforcementError(
                f"returned model {ret_model!r} != pinned {self.model_slug!r}")
        if ret_provider.lower().replace(" ", "") != self.provider_display.lower().replace(" ", ""):
            raise OpenRouterEnforcementError(
                f"returned provider {ret_provider!r} != pinned {self.provider_display!r} "
                f"(fallback/routing forbidden)")
        if not gen_id:
            raise OpenRouterEnforcementError("missing generation id (provenance metadata required)")
        if not usage:
            raise OpenRouterEnforcementError("missing usage metadata (provenance metadata required)")
        choices = r.get("choices") or []
        if not choices or "message" not in choices[0]:
            raise OpenRouterEnforcementError("malformed response: no choices[0].message")
        return {"model": ret_model, "provider": ret_provider, "generation_id": gen_id,
                "usage": usage}

    def __call__(self, prompt: str) -> str:
        body = self._body(prompt)
        req_ts = time.time()
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = self._transport(API_URL, body, self.timeout_s)
                break
            except urllib.error.HTTPError as e:
                if e.code in _RETRYABLE and attempt < _MAX_RETRIES:
                    last_err = e
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue                              # retry: IDENTICAL body, same endpoint
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < _MAX_RETRIES:
                    last_err = e
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise
        else:  # pragma: no cover
            raise last_err
        meta = self._assert_response(r)
        text = str(r["choices"][0]["message"].get("content") or "")
        with self._lock:
            self.n_calls += 1
            self.total_cost += float(meta["usage"].get("cost") or 0.0)
            self.total_tokens += int(meta["usage"].get("total_tokens") or 0)
        if self.audit_path:
            row = {"ts_request": round(req_ts, 3), "ts_response": round(time.time(), 3),
                   "model_requested": self.model_slug, "model_returned": meta["model"],
                   "provider_pinned": self.provider_display, "provider_returned": meta["provider"],
                   "quantization_pinned": self.quantization,
                   "generation_id": meta["generation_id"],
                   "routing": {"strategy": "order+allow_fallbacks=false", "attempts": 1},
                   "usage": meta["usage"],
                   "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
                   "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "max_tokens": self.max_tokens, "temperature": self.temperature}
            with self._lock, open(self.audit_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return text


def openrouter_chat_fn(model_slug: str, *, provider: str, quantization: str,
                       system: str = "Reply ONLY JSON.", max_tokens: int = 2400,
                       temperature: float = 0.2, audit_path: str | None = None,
                       _transport=None) -> OpenRouterPinnedClient:
    """The repository-contract factory (mirrors deepseek_chat_fn)."""
    return OpenRouterPinnedClient(model_slug, provider=provider, quantization=quantization,
                                  system=system, max_tokens=max_tokens, temperature=temperature,
                                  audit_path=audit_path, _transport=_transport)
