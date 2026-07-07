"""ActionWorldModel — the general best-action front door: question → compile → do(a) over the world → best action.

The interventional twin of `WorldModel.simulate`. Where `WorldModel` answers "what will happen", this answers
"what should I DO": it compiles the question into the right structural model ONCE (the same compiler, the same
mechanism library), then treats each candidate action as a `do`-operator that transforms that compiled model,
rolls each variant forward as a Monte-Carlo ensemble, and returns the argmax action with a navigable object
(distribution + reducible/irreducible + pivotal worlds), a confidence statement, and the contrast vs doing
nothing. It generalizes across every mechanism the compiler emits — a message to a person, a price on a
generic-SCM demand curve, a roster change in a bracket, a mobilization push in an electorate — because the
action is just a transform of the spec and the sampler is mechanism-agnostic.

    awm = ActionWorldModel(compiler, retriever)
    awm.best_action("What should I charge for the pro tier?", grid("price", 20, 200, 19), profit_utility)

This is the top-2 value prop: interventional, off-market, forecastable decisions — the regime where a
simulation is the only instrument and the crowd has no price.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.api.compiler import build_sampler
from swm.decision.best_action import best_action, compare_actions
from swm.decision.utility import Mean


def spec_outcome_fn(spec):
    """Bridge the compiler to the decision core: `outcome_fn(action, rng) -> (outcome, factors)` where the
    action is applied to the spec and the resulting world is sampled. The per-action sampler is built once
    (deterministic transform) and cached, so drawing thousands of samples is cheap."""
    cache = {}

    def f(action, rng):
        tr = cache.get(id(action))
        if tr is None:
            tr = build_sampler(action.apply(spec)).traced
            cache[id(action)] = tr
        return tr(rng)
    return f


@dataclass
class ActionWorldModel:
    compiler: object                       # swm.api.compiler.StructuralCompiler
    retriever: object = None               # optional; context can be passed in directly
    n: int = 4000                          # per-action navigable ensemble size

    def _spec(self, question, context, as_of, key):
        if self.retriever is not None and not context:
            ctx = self.retriever.retrieve(question, as_of=as_of)
            context = ctx.to_prompt()
        return self.compiler.compile(question, context, key=key).spec

    def best_action(self, question, action_space, utility, *, objective=None, context="", as_of="",
                    key=None, baseline=None, **kw):
        """Compile the question, then choose the best action over the compiled world. `action_space` is an
        ActionSpace (or any iterable of Actions); `baseline` defaults to the space's do-nothing action."""
        spec = self._spec(question, context, as_of, key)
        base = baseline if baseline is not None else getattr(action_space, "baseline", None)
        return best_action(spec_outcome_fn(spec), action_space, utility,
                           objective=objective or Mean(), baseline=base, n_navigable=self.n, **kw)

    def decide_over_spec(self, spec, action_space, utility, *, objective=None, baseline=None, **kw):
        """Skip compilation — decide directly over a spec you already have (a hand-built or cached model)."""
        base = baseline if baseline is not None else getattr(action_space, "baseline", None)
        return best_action(spec_outcome_fn(spec), action_space, utility,
                           objective=objective or Mean(), baseline=base, n_navigable=self.n, **kw)

    def compare(self, question, a, b, utility, *, context="", as_of="", key=None, **kw):
        """do(A) vs do(B) on the compiled world with common-random-numbers pairing."""
        spec = self._spec(question, context, as_of, key)
        return compare_actions(spec_outcome_fn(spec), a, b, utility, **kw)
