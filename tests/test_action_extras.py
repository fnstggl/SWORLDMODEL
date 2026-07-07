"""Tests for the remaining action-layer components: constrained risk (C4), continuous refine + generative
propose-mutate (C2), provenance + calibration grade (C5/C7), and single temporal do-operators (C1)."""
import random

from swm.decision.action import Action, inject_event, set_var
from swm.decision.best_action import (best_action, best_action_generative, best_continuous, validated_domain)
from swm.decision.utility import Constrained, Mean, identity, value
from swm.api.compiler import build_sampler
from swm.api.model_spec import parse_spec
from swm.simulation.structural import montecarlo


# ---- C4: constrained risk objective ----
def test_constrained_prefers_safe_arm_when_risky_violates_cap():
    # 'risky' has a higher MEAN but frequently dips into disaster (<0.2); 'safe' never does. With a cap on
    # P(disaster), the safe arm must win despite the lower mean.
    def of(action, rng):
        if action.label == "risky":
            return (0.95 if rng.random() < 0.6 else 0.0), {}   # mean ~0.57 but 40% disasters
        return 0.5, {}                                          # safe: mean 0.5, zero disasters
    obj = Constrained(Mean(), is_disaster=lambda u: u < 0.2, max_disaster_prob=0.1)
    res = best_action(of, [Action("risky"), Action("safe")], identity(), objective=obj, max_per_arm=2000, seed=0)
    assert res.best.label == "safe"


# ---- C2: continuous local-refine (grid -> narrow -> repeat) beats a coarse grid ----
def test_best_continuous_refines_to_profit_max():
    # profit(price) = price*(1 - price/100), true argmax at 50. A coarse grid of 10s could miss; refinement
    # narrows around the winner across rounds and lands close to 50.
    def outcome_fn_for(price):
        def of(rng):
            demand = max(0.0, 1 - price / 100) + rng.gauss(0, 0.01)
            return price * demand, {}
        return of
    res = best_continuous(outcome_fn_for, "price", 10, 90, value(lambda o: o), objective=Mean(),
                          rounds=4, steps=7, max_per_arm=1500, seed=0)
    assert 46 <= res.best.action.value <= 54                    # refined optimum near 50


# ---- C2: generative propose -> score -> mutate -> re-score ----
def test_best_action_generative_improves_with_mutation():
    # the "true best" message has clarity 0.9; the proposer only offers weak options; the mutator nudges the
    # top survivors toward higher clarity -> the final pick beats the best initial proposal.
    def outcome_fn(action, rng):
        return action.meta["clarity"], {}
    def propose(seed):
        return [Action(f"p{c}", meta={"clarity": c}) for c in (0.2, 0.35, 0.5)]
    def mutate(survivors, seed):
        out = []
        for a in survivors:
            c = min(1.0, a.meta["clarity"] + 0.25)
            out.append(Action(f"m{round(c,2)}", meta={"clarity": c}))
        return out
    res = best_action_generative(outcome_fn, propose, identity(), mutate_fn=mutate, rounds=3, keep=2,
                                 objective=Mean(), max_per_arm=400, seed=0)
    assert res.best.value > 0.5                                 # mutation climbed past the best proposal (0.5)


# ---- C5/C7: provenance + calibration grade on the recommendation ----
def test_provenance_and_grade():
    def of(action, rng):
        return (0.9 if action.label == "A" else 0.1), {}
    prov = validated_domain("CMV persuasion", 0.74)
    res = best_action(of, [Action("A"), Action("B")], identity(), max_per_arm=1500, seed=0, provenance=prov)
    assert res.provenance["status"] == "validated"
    assert res.grade().startswith("A")                         # decisive + validated
    d = res.as_dict()
    assert d["grade"].startswith("A") and d["provenance"]["domain"] == "CMV persuasion"
    # a hypothesis-domain recommendation grades lower even when decisive
    res2 = best_action(of, [Action("A"), Action("B")], identity(), max_per_arm=1500, seed=0,
                       provenance=validated_domain("pricing", status="hypothesis"))
    assert res2.grade().startswith("B")


# ---- C1: a single temporal do-operator moves a generic_scm outcome ----
def test_inject_event_raises_generic_scm_outcome():
    spec = parse_spec({"mechanism": "generic_scm",
                       "variables": [{"name": "x", "value": 0.4, "est_sd": 0.0, "volatility": 0.02}],
                       "equations": {"x": "0.1*(0.5 - x)"}, "outcome": {"variable": "x"}, "horizon": 6})
    base = montecarlo(build_sampler(spec).once, n=3000, seed=0)["mean"]
    shocked_spec = inject_event("x", 0.3, time=1.0).apply(spec)
    shocked = montecarlo(build_sampler(shocked_spec).once, n=3000, seed=0)["mean"]
    assert shocked > base + 0.05                                # the injected event carries through the diffusion
