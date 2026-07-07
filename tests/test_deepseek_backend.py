"""Tests for the DeepSeek/default backend selector (no network — env-var routing only)."""
import os
from swm.api.deepseek_backend import default_chat_fn, deepseek_chat_fn


def test_selector_routes_by_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert default_chat_fn() is None                       # no keys -> None (callers use cached judgments)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert callable(default_chat_fn())                     # DeepSeek key present -> a callable backend


def test_deepseek_fn_is_callable_and_reads_env_only():
    fn = deepseek_chat_fn()                                 # constructs without a network call
    assert callable(fn)
