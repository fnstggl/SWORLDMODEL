"""Tests for the DeepSeek/default backend selector (no live network)."""
import json
import os
from swm.api.deepseek_backend import default_chat_fn, deepseek_chat_fn


def test_selector_routes_by_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert default_chat_fn() is None                       # no keys -> None (callers use cached judgments)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert callable(default_chat_fn())                     # DeepSeek key present -> a callable backend


def test_deepseek_fn_accepts_process_memory_credential():
    fn = deepseek_chat_fn(api_key="test-only")              # constructs without a network call
    assert callable(fn)
    assert fn.provider_metadata["api_alias"] == "deepseek-v4-flash"
    assert fn.provider_metadata["credential_transport"] == "process_memory"


def test_deepseek_request_pins_model_and_thinking_without_serializing_key(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("swm.api.deepseek_backend.urllib.request.urlopen", fake_urlopen)
    assert deepseek_chat_fn(api_key="test-only")("prompt") == "ok"
    request = captured["request"]
    body = json.loads(request.data)
    assert body["model"] == "deepseek-v4-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert "test-only" not in request.data.decode()
    assert request.headers["Authorization"] == "Bearer test-only"
