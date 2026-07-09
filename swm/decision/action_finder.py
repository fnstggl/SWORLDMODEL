"""The GENERAL best-action finder — argmax_a E[U(outcome) | do(a), context] over a TYPED action space.

The message optimizer is one specialization (text actions). The general product recognizes that an action
has a TYPE, and the search operator must match it — but the SPINE is shared: a calibrated world-model
objective, and uncertainty-aware best-arm racing (swm/decision/best_action.py) that finds the true argmax
with a confidence statement, not a fixed-N guess.

    ┌ action type        ┌ search operator (tries MANY actions)            ┌ LLM's role
    Continuous  (price…)   grid → local-refine over the response curve       proposes which levers exist
    Discrete    (vendor…)  enumerate the set + best-arm race                  scores/among options
    Generative  (copy…)    propose → score → mutate (the message pattern)     proposer + critic
    Structured  (policy…)  coordinate ascent / combinatorial over the fields  generates text fields; realism critic

Everything reduces to the same `sample_fn(action, rng) -> outcome_value` that best_action races. For a
probabilistic social world model, `world_model(...)` wraps a `score_fn(action) -> P(outcome)` (a point OR
an ensemble of samples) into that sampler, so maximizing the mean outcome IS maximizing P(desired outcome),
with confidence intervals from the racing.

The unlock is NOT the search (cheap, general) — it's the calibrated world model per domain (the hard,
data-hungry part). This module is the search harness; you bring the world model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.decision.best_action import (DecisionResult, best_action, best_action_generative,
                                      best_continuous)
from swm.decision.utility import Mean, Utility


@dataclass
class Action:
    """A generic candidate action: a value (number, option, config dict, or text) + a display label."""
    value: object
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = str(self.value)[:60]


# ---- typed action spaces -------------------------------------------------------------------------

@dataclass
class Continuous:
    """A numeric lever: price, discount, bid, dosage, rate, timing. Searched by grid → local refine."""
    var: str
    lo: float
    hi: float
    steps: int = 15
    rounds: int = 4


@dataclass
class DiscreteChoice:
    """Pick the best of an enumerable set: which candidate, vendor, market, feature, channel."""
    options: list                                  # values or Action objects


@dataclass
class GenerativeText:
    """An open-ended text action: copy, subject line, pitch, script. Propose → score → mutate."""
    propose_fn: object                             # propose_fn(seed) -> list[Action]
    mutate_fn: object = None                       # mutate_fn(list[Action], seed) -> list[Action]
    rounds: int = 3
    k: int = 8


@dataclass
class Structured:
    """A combinatorial config: a policy (bundle of levers), a campaign (channel+budget+creative), a product
    bundle. `fields` maps name -> a list of categorical choices OR a (lo, hi[, steps]) numeric range."""
    fields: dict = field(default_factory=dict)
    sweeps: int = 3


# ---- world-model adapter: score_fn(action) -> P(outcome)  =>  sample_fn(action, rng) -> outcome ---

def world_model(score_fn, value_fn=None):
    """Wrap a calibrated world model into a Monte-Carlo sampler for the racing machinery. `score_fn(action)`
    returns P(desired outcome) — either a point in [0,1] or a LIST of ensemble samples (predictive
    uncertainty). The sampler draws a p (from the ensemble if given) then a Bernoulli. By default the
    outcome value is the 0/1 success, so the racer maximizes P(outcome); pass `value_fn(action, success) ->
    value` to maximize a UTILITY instead (e.g. revenue = price × sale). CI reflects sampling + model
    uncertainty."""
    def sample(action, rng):
        av = action.value if isinstance(action, Action) else action
        p = score_fn(av)
        if isinstance(p, (list, tuple)):
            p = p[rng.randrange(len(p))] if p else 0.5
        success = 1.0 if rng.random() < max(0.0, min(1.0, float(p))) else 0.0
        return value_fn(av, success) if value_fn is not None else success
    return sample


def _to_action(x):
    return x if isinstance(x, Action) else Action(x)


# ---- the dispatcher ------------------------------------------------------------------------------

def find_best_action(space, sample_fn, *, utility=None, objective=None, baseline=None,
                     seed=0, **kw) -> DecisionResult:
    """Find the best action over a typed `space`, scoring with `sample_fn(action, rng) -> outcome_value`
    (use `world_model(score_fn)` to build it). Returns a DecisionResult (winner + ranking + confidence +
    contrast vs baseline). Dispatches to the search operator matched to the action type."""
    # the racer's navigable path needs a Utility object (has .fn/.desc); wrap a bare callable.
    if utility is None:
        utility = Utility(lambda o: float(o), "E[outcome]")
    elif not isinstance(utility, Utility):
        utility = Utility(utility, "utility")
    objective = objective or Mean()

    if isinstance(space, Continuous):
        def outcome_fn_for(v):
            a = Action(v, f"{space.var}={round(v, 4)}")
            return lambda rng: (sample_fn(a, rng), {"value": v})
        return best_continuous(outcome_fn_for, space.var, space.lo, space.hi, utility,
                               objective=objective, steps=space.steps, rounds=space.rounds,
                               baseline=baseline, seed=seed, **kw)

    if isinstance(space, DiscreteChoice):
        actions = [_to_action(o) for o in space.options]
        def outcome_fn(a, rng):
            return (sample_fn(a, rng), {})
        return best_action(outcome_fn, actions, utility, objective=objective, baseline=baseline,
                           seed=seed, **kw)

    if isinstance(space, GenerativeText):
        def outcome_fn(a, rng):
            return (sample_fn(a, rng), {})
        return best_action_generative(outcome_fn, space.propose_fn, utility, mutate_fn=space.mutate_fn,
                                      rounds=space.rounds, k=space.k, objective=objective,
                                      baseline=baseline, seed=seed, **kw)

    if isinstance(space, Structured):
        return _best_structured(space, sample_fn, utility, objective, baseline=baseline, seed=seed, **kw)

    raise TypeError(f"unknown action space type: {type(space).__name__}")


def _field_choices(spec):
    if isinstance(spec, tuple) and len(spec) >= 2:
        lo, hi = float(spec[0]), float(spec[1])
        steps = int(spec[2]) if len(spec) > 2 else 5
        return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)] if steps > 1 else [lo]
    return list(spec)


def _cfg_label(cfg):
    return "{" + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in cfg.items()) + "}"


def _best_structured(space, sample_fn, utility, objective, *, baseline=None, seed=0, n_eval=160,
                     **kw) -> DecisionResult:
    """Combinatorial search: coordinate ascent over the fields (try every choice of each field, holding the
    rest, and keep improvements), sweeping to convergence — then best-arm race the winner against its
    strongest neighbors for a confidence statement. Evaluates many configs (Σ fields × choices × sweeps),
    not 3–4."""
    rng = random.Random(seed)
    fields = space.fields
    choices = {f: _field_choices(sp) for f, sp in fields.items()}
    cfg = {f: choices[f][0] for f in fields}

    def quick(config):
        a = Action(dict(config), _cfg_label(config))
        return sum(utility(sample_fn(a, rng)) for _ in range(n_eval)) / n_eval

    best, best_v = dict(cfg), quick(cfg)
    seen = {_cfg_label(best): (best, best_v)}
    for _ in range(max(1, space.sweeps)):
        improved = False
        for f in fields:
            for c in choices[f]:
                trial = {**best, f: c}
                lab = _cfg_label(trial)
                v = seen[lab][1] if lab in seen else quick(trial)
                seen[lab] = (trial, v)
                if v > best_v + 1e-9:
                    best, best_v, improved = trial, v, True
        if not improved:
            break

    # race the winner against the strongest distinct neighbors found (uncertainty-aware final selection)
    top = sorted(seen.values(), key=lambda kv: kv[1], reverse=True)[:6]
    actions = [Action(dict(c), _cfg_label(c)) for c, _ in top]
    def outcome_fn(a, rng2):
        return (sample_fn(a, rng2), {})
    return best_action(outcome_fn, actions, utility, objective=objective, baseline=baseline,
                       seed=seed + 1, **kw)
