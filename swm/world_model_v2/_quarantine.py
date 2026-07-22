"""Quarantine for deprecated / superseded execution paths.

A quarantined entry point still exists (old experiments and validation harnesses import it), but it is NOT
part of the live world-model-v2 forecast path and must never be used for new work. The canonical path is
`swm.world_model_v2.unified_runtime.simulate_world` (reached in production via `swm.facade.forecast(
architecture="world_model_v2")`), which threads evidence retrieval, Phase-3 posterior reweighting,
Phase-10 institution normalization, the scheduled-reality/calendar layer and the full actor rollout through
one plan/world/queue. The deprecated inner entries skip most of that and silently degrade to a broad prior —
the failure mode that made a real backtest (EXP-102) collapse 4/5 questions to a ~0.5 guess.

`@quarantined(...)` marks such an entry: calling it emits a LOUD DeprecationWarning (once per process, so a
sweep is not drowned out) naming the canonical replacement. It never changes the return value, so the
legitimate validation experiments that pin the old behaviour keep working — the warning is the guardrail,
not a hard block, because a hard raise would break those pinned-science harnesses.
"""
from __future__ import annotations

import functools
import warnings

_WARNED: set[str] = set()

#: the one true forecast entry — everything else in the module map routes here.
CANONICAL_ENTRY = "swm.world_model_v2.unified_runtime.simulate_world"


def quarantined(*, use_instead: str = CANONICAL_ENTRY, reason: str = "", since: str = "EXP-102"):
    """Mark a deprecated/superseded callable. Calling it warns once (per process) and passes through.

    use_instead: the canonical replacement to name in the warning (defaults to simulate_world).
    reason:      one line on why it is quarantined (what it silently skips).
    """
    def deco(fn):
        tag = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn.__name__)}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if tag not in _WARNED:
                _WARNED.add(tag)
                msg = (f"QUARANTINED path {tag} was called — it is NOT the live world-model-v2 forecast "
                       f"path and skips major subsystems (evidence retrieval, posterior reweighting, "
                       f"institution execution, scheduled-reality/calendar). Use {use_instead} instead.")
                if reason:
                    msg += f" ({reason})"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__quarantined__ = True
        wrapper.__use_instead__ = use_instead
        return wrapper
    return deco
