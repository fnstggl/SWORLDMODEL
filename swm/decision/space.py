"""ActionSpace + proposers — where candidate actions come from.

Three regimes, matching how the decision is shaped:
  - ENUMERABLE  — a fixed set (which of 5 emails, 3 features): score all.
  - CONTINUOUS  — a parameter (price, discount, timing): a grid of `set_var` actions, refine around the best.
  - OPEN-ENDED  — what to *say*: an LLM proposes a diverse candidate set (the generation seam, pluggable like
    every other LLM-touching part); the simulator scores each. The LLM proposes; simulation selects.

An ActionSpace optionally carries a `baseline` (do-nothing / status quo) so every recommendation is
contrasted against the honest counterfactual of taking no action.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.decision.action import Action, noop, set_message, set_var


@dataclass
class ActionSpace:
    actions: list                     # list[Action]
    baseline: object = None           # optional Action (do-nothing); None => no baseline contrast

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)

    def with_baseline(self, action: Action) -> "ActionSpace":
        return ActionSpace(self.actions, action)


def enumerate_actions(actions, *, baseline=None) -> ActionSpace:
    """A fixed candidate set. `baseline` (default: do-nothing) is the contrast counterfactual."""
    return ActionSpace(list(actions), baseline if baseline is not None else noop())


def grid(var, lo, hi, steps, *, label=None, baseline=None) -> ActionSpace:
    """Sweep a continuous parameter as a grid of `set_var` interventions (pricing, discount, timing)."""
    if steps < 1:
        steps = 1
    vals = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)] if steps > 1 else [lo]
    acts = [set_var(var, v, label=f"{var}={round(v, 4)}") for v in vals]
    return ActionSpace(acts, baseline)


def message_options(messages, *, labels=None, baseline=None) -> ActionSpace:
    """Candidate messages (the best-message decision). Each becomes a `set_message` do-operator."""
    acts = [set_message(m, label=(labels[i] if labels else f"msg_{i}")) for i, m in enumerate(messages)]
    return ActionSpace(acts, baseline)


def llm_proposed(propose_fn, question, spec, *, k=6, baseline=None) -> ActionSpace:
    """Open-ended generation seam: `propose_fn(question, spec, k) -> list[Action]`. The LLM proposes a
    diverse candidate set (it has read how people negotiate/pitch/price); the simulator then scores them."""
    acts = list(propose_fn(question, spec, k))
    return ActionSpace(acts, baseline if baseline is not None else noop())
