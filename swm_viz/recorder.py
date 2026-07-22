"""Non-invasive capture layer for World Model V2 — Lean V2.

This module records EXACTLY the LLM inputs/outputs and the compiled world of a *real*
`lean_v2` run. It does so by transparently wrapping the two natural observation points of
the pipeline, WITHOUT modifying anything under `swm/`:

  * ``LLMGateway.call(stage, prompt) -> reply`` — the ONE gateway every external provider
    call in a Lean V2 run flows through (blueprint, grounding, state-generation, every actor
    decision, every deliberation, schema repair, the challenger). We capture the full prompt
    and full reply verbatim, plus the tier/latency/retry the gateway itself recorded.

  * ``lean_v2.runtime.compile_blueprint(...)`` — returns the ``ConsumerWorldBlueprint`` (the
    compiled causal world: actors, institutions, decision rules, vote options, terminal
    logic). We keep the reference and serialize it AFTER the run, so it includes the grounded
    private-state variants the orchestrator installs later.

Both wrappers invoke the original and return its result **unchanged** — the simulation runs
identically; nothing about its behavior, control flow, prompts, or replies is altered. This is
pure observation ("read/take in the same LLM api inputs and outputs and states created, state
transitions"), never a behavioral change. All patches are restored on exit.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class Capture:
    """Accumulates the verbatim LLM I/O and the compiled world of one lean_v2 run."""

    def __init__(self) -> None:
        self.calls: list[dict] = []          # ordered, one row per logical gateway call
        self.blueprints: list = []           # captured ConsumerWorldBlueprint references
        self.challenger_blueprints: list = []
        self._lock = threading.Lock()
        self._t0: float = time.time()
        self._seq = 0

    # -- gateway observation -------------------------------------------------------------
    def _record_call(self, *, stage, prompt, reply, latency, tier, retried, t_start,
                     error=False) -> None:
        with self._lock:
            seq = self._seq
            self._seq += 1
            self.calls.append({
                "seq": seq,
                "stage": str(stage),
                "prompt": str(prompt or ""),
                "reply": reply if isinstance(reply, str) else "",
                "prompt_chars": len(prompt or ""),
                "reply_chars": len(reply or "") if isinstance(reply, str) else 0,
                "latency_s": round(float(latency or 0.0), 3),
                "tier": str(tier or ""),
                "retried": bool(retried),
                "t_start": round(float(t_start or 0.0), 3),
                "error": bool(error),
            })

    def elapsed(self) -> float:
        return time.time() - self._t0


@contextmanager
def capture_lean_v2():
    """Context manager that installs the transparent observers and yields a :class:`Capture`.

    Usage::

        with capture_lean_v2() as cap:
            res = simulate_world(..., execution_profile="lean_v2")
        # cap.calls  -> verbatim ordered LLM I/O
        # cap.blueprints[0] -> the compiled ConsumerWorldBlueprint (serialize via .as_dict())
    """
    from swm.world_model_v2.lean_v2 import gateway as gw_mod
    from swm.world_model_v2.lean_v2 import runtime as rt_mod

    cap = Capture()
    cap._t0 = time.time()

    orig_call = gw_mod.LLMGateway.call
    orig_compile = rt_mod.compile_blueprint
    orig_challenger = getattr(rt_mod, "build_challenger_blueprint", None)

    def wrapped_call(gw_self, stage, prompt):
        t_start = time.time() - cap._t0
        n_before = len(gw_self.rows)
        t = time.time()
        try:
            reply = orig_call(gw_self, stage, prompt)
        except Exception:
            # a refused/failed call (e.g. BudgetExhausted) — record the attempt, then re-raise
            try:
                tier = gw_self.tier_for(stage)
            except Exception:  # noqa: BLE001
                tier = ""
            cap._record_call(stage=stage, prompt=prompt, reply=None,
                             latency=time.time() - t, tier=tier, retried=False,
                             t_start=t_start, error=True)
            raise
        # the original appended an authoritative ledger row (tier/latency/retried) — pair it
        row = gw_self.rows[n_before] if len(gw_self.rows) > n_before else {}
        cap._record_call(
            stage=stage, prompt=prompt, reply=reply,
            latency=row.get("latency_s", time.time() - t),
            tier=row.get("tier") or _safe_tier(gw_self, stage),
            retried=bool(row.get("retried")), t_start=t_start)
        return reply

    def wrapped_compile(*a, **k):
        out = orig_compile(*a, **k)
        try:
            bp = out[0] if isinstance(out, tuple) else out
            cap.blueprints.append(bp)
        except Exception:  # noqa: BLE001 — capture must never break the run
            pass
        return out

    def wrapped_challenger(*a, **k):
        out = orig_challenger(*a, **k)
        try:
            cap.challenger_blueprints.append(out)
        except Exception:  # noqa: BLE001
            pass
        return out

    gw_mod.LLMGateway.call = wrapped_call
    rt_mod.compile_blueprint = wrapped_compile
    if orig_challenger is not None:
        rt_mod.build_challenger_blueprint = wrapped_challenger
    try:
        yield cap
    finally:
        gw_mod.LLMGateway.call = orig_call
        rt_mod.compile_blueprint = orig_compile
        if orig_challenger is not None:
            rt_mod.build_challenger_blueprint = orig_challenger


def _safe_tier(gw_self, stage: str) -> str:
    try:
        return gw_self.tier_for(stage)
    except Exception:  # noqa: BLE001
        return ""
