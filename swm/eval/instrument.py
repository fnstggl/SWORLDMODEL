"""Compute & evidence parity instrumentation for the ablation harness.

Every fair arm comparison the program demands ("do not claim simulation superiority unless it beats the
strongest fair compute-matched alternative") requires knowing exactly how much each arm spent. This wraps a
chat fn to count CALLS exactly and INPUT/OUTPUT tokens approximately (chars/4 — the DeepSeek backend discards
the API usage field, so this is the honest proxy), and stamps every ablation row with the evidence hash,
code commit and model version so no two arms can be silently compared on different inputs.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, field

# DeepSeek pricing (chat, cache-miss) as of 2026-01; adjust in one place. USD per 1M tokens.
_PRICE_IN = 0.27 / 1_000_000
_PRICE_OUT = 1.10 / 1_000_000


@dataclass
class Meter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    def cost_usd(self) -> float:
        return round(self.tokens_in * _PRICE_IN + self.tokens_out * _PRICE_OUT, 6)

    def snapshot(self) -> dict:
        return {"calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "seconds": round(self.seconds, 2), "cost_usd": self.cost_usd()}


class CountingLLM:
    """Wrap a callable(prompt)->text, accumulating call/token/latency into a shared Meter. Token counts are
    chars/4 estimates (no usage field from the backend). Reset the meter between arms to attribute spend."""

    def __init__(self, fn, meter: Meter = None):
        self._fn = fn
        self.meter = meter or Meter()

    def __call__(self, prompt: str) -> str:
        t0 = time.time()
        out = self._fn(prompt)
        self.meter.calls += 1
        self.meter.tokens_in += len(prompt) // 4
        self.meter.tokens_out += len(out or "") // 4
        self.meter.seconds += time.time() - t0
        return out


def evidence_hash(text: str) -> str:
    """Stable short hash of the frozen dossier — two arms MUST share this to be a fair comparison."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def prompt_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def model_version(default="deepseek-chat") -> str:
    import os
    if os.environ.get("DEEPSEEK_API_KEY"):
        return default
    if os.environ.get("HF_TOKEN"):
        return "hf-router"
    return "none"
