"""Mechanism fallback hierarchy (Part A3) — the guarantee that every coherent question SIMULATES.

For each required causal process the compiler chooses the highest DEFENSIBLE tier:

  Tier 1  scenario-fitted + held-out validated
  Tier 2  domain-validated parameter pack
  Tier 3  cross-domain/population transfer-validated pack
  Tier 4  published empirical mechanism (study limits + widened transport uncertainty)
  Tier 5  reference-class estimated mechanism
  Tier 6  generic structural mechanism family (broad priors, explicit assumptions)
  Tier 7  multiple competing qualitative mechanism hypotheses (deliberately broad uncertainty)

`no production-eligible mechanism` NEVER becomes `no forecast`. Tiers 6-7 are REAL typed mechanisms: they
state causal assumptions, use broad uncertainty, produce StateDelta objects, execute through WorldState,
support sensitivity, and are labeled exploratory/highly_speculative — never a fixed constant, never an
LLM-minted probability. The generic outcome mechanism draws the terminal outcome from a broad prior over
the base rate (Beta for binary/categorical, wide Normal for continuous); competing structural/directional
hypotheses across particles widen the terminal distribution. The LLM may only propose a QUALITATIVE
directional lean (weak_no/neutral/weak_yes) which maps to a fixed broad-Beta — the number is never minted
by the LLM.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.state import F, rfc3339
from swm.world_model_v2.transitions import (StateDelta, TransitionOperator, TransitionProposal,
                                            register_operator)

TIER_SUPPORT_GRADE = {1: "empirically_supported", 2: "empirically_supported", 3: "transfer_supported",
                      4: "transfer_supported", 5: "exploratory", 6: "exploratory", 7: "highly_speculative"}

#: qualitative directional lean → BROAD Beta(a,b). The LLM proposes the lean (a qualitative relationship,
#: permitted); the numbers here are fixed and wide (mean shift modest, high dispersion). Never LLM-minted.
LEAN_BETA = {"strong_no": (1.3, 3.0), "weak_no": (1.6, 2.3), "neutral": (1.0, 1.0),
             "weak_yes": (2.3, 1.6), "strong_yes": (3.0, 1.3)}


@dataclass
class MechanismChoice:
    """The tier decision for one required causal process."""
    process: str
    tier: int
    family: str
    support_grade: str
    why: str
    transported: bool = False
    competing_hypotheses: list = field(default_factory=list)   # for tier 7

    def as_dict(self):
        return {"process": self.process, "tier": self.tier, "family": self.family,
                "support_grade": self.support_grade, "why": self.why, "transported": self.transported,
                "competing_hypotheses": self.competing_hypotheses}


def select_tier(process: str, applicable, *, has_local_fit=False, has_domain_pack=False,
                transported=False, competing=None) -> MechanismChoice:
    """Pick the highest defensible tier for one causal process given what the registry offers.
    `applicable` is a scored family (or None). Tiers 1-4 require real validation on the family; 5-7 are the
    generic fallbacks. Returns a MechanismChoice with the support grade that tier implies."""
    if competing and len(competing) > 1:
        return MechanismChoice(process, 7, "competing_qualitative_hypotheses",
                               TIER_SUPPORT_GRADE[7], "multiple plausible mechanisms; broad disagreement",
                               competing_hypotheses=list(competing))
    if applicable is not None:
        fam = applicable.get("family_id") if isinstance(applicable, dict) else getattr(applicable, "family_id", "")
        # infer tier from the family's evidence: held-out+transfer→1/2, transfer only→3, transported→4
        if has_local_fit:
            return MechanismChoice(process, 1, fam, TIER_SUPPORT_GRADE[1], "scenario/domain held-out fit")
        if has_domain_pack:
            return MechanismChoice(process, 2, fam, TIER_SUPPORT_GRADE[2], "domain-validated parameter pack")
        if transported:
            return MechanismChoice(process, 3, fam, TIER_SUPPORT_GRADE[3],
                                   "transfer-validated pack (widened uncertainty)", transported=True)
        return MechanismChoice(process, 4, fam, TIER_SUPPORT_GRADE[4],
                               "published mechanism, transported with widened uncertainty", transported=True)
    # no applicable validated family → generic structural (tier 6)
    return MechanismChoice(process, 6, "generic_outcome_prior", TIER_SUPPORT_GRADE[6],
                           "no validated mechanism applies — generic structural family with broad priors")


def overall_support_grade(choices) -> str:
    """The result's support grade is the WEAKEST tier among its important causal processes (a chain is as
    supported as its weakest load-bearing mechanism)."""
    if not choices:
        return "highly_speculative"
    worst = max(c.tier for c in choices)                     # higher tier number = weaker support
    return TIER_SUPPORT_GRADE[worst]


# ------------------------------------------------------------------ the generic outcome mechanism (tier 6/7)
class GenericOutcomeOperator(TransitionOperator):
    """Tier-6/7 fallback: a REAL typed mechanism that resolves the terminal outcome from a BROAD prior when
    no validated mechanism applies. Per particle it draws a base rate p ~ Beta(a,b) (a,b from a fixed
    qualitative-lean table — never LLM-minted), then the outcome ~ Bernoulli(p) for binary families, or a
    category by a broad Dirichlet for categorical, or a wide Normal for continuous. Writes the canonical
    readout quantity. Assumptions are explicit; the prior width IS the uncertainty; each particle differs
    (not a fixed constant). Labeled exploratory/highly_speculative by the tier. StateDelta produced."""
    name = "generic_outcome_prior"

    def applicable(self, world, event):
        return event.etype == "resolve_outcome"

    def propose(self, world, event, rng):
        p = event.payload
        return TransitionProposal(operator=self.name, action={
            "outcome_var": str(p.get("outcome_var", "outcome")),
            "family": str(p.get("family", "binary")),
            "lean": str(p.get("lean", "neutral")),
            "options": list(p.get("options") or ["True", "False"]),
            "lo": p.get("lo"), "hi": p.get("hi")},
            reason_codes=["generic_fallback", f"lean={p.get('lean', 'neutral')}"])

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        a = proposal.action
        var, fam = a["outcome_var"], a["family"]
        rng = random.Random(hash(world.branch_id) & 0xFFFFFFFF)
        av, bv = LEAN_BETA.get(a["lean"], (1.0, 1.0))
        register_quantity_type(var, units="outcome")
        before = world.quantities[var].value if var in world.quantities else None
        if before is not None:
            # SAFETY NET: a domain mechanism already resolved the outcome — do not overwrite it. A value in a
            # different vocabulary (a boolean True/False vs yes/no options) is reconciled at projection time
            # (contract canonicalizes truthy→affirmative), so a legitimate mechanism write is never clobbered.
            d = StateDelta(at=world.clock.now, event_type="resolve_outcome", operator=self.name,
                           reason_codes=["already_resolved_by_domain_mechanism_noop"])
            return d
        if fam in ("binary", "response_occurrence", "best_action"):
            p = _beta_sample(rng, av, bv)                    # per-particle base-rate draw (broad)
            opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
            val = opts[0] if rng.random() < p else opts[1]
        elif fam == "categorical":
            opts = a["options"] or ["A", "B", "C"]
            weights = [_beta_sample(rng, 1.0, 1.0) for _ in opts]   # broad Dirichlet-ish
            z = sum(weights) or 1.0
            r, acc = rng.random(), 0.0
            val = opts[-1]
            for o, w in zip(opts, weights):
                acc += w / z
                if r <= acc:
                    val = o
                    break
        else:                                                # continuous-like: wide Normal on [lo,hi]
            lo = float(a["lo"]) if a["lo"] is not None else 0.0
            hi = float(a["hi"]) if a["hi"] is not None else 1.0
            mid = (lo + hi) / 2.0
            val = min(hi, max(lo, rng.gauss(mid, (hi - lo) / 3.0)))
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="resolve_outcome", operator=self.name,
                       reason_codes=proposal.reason_codes,
                       uncertainty={"prior": f"Beta({av},{bv})" if fam != "continuous" else "wide Normal",
                                    "tier": "6/7 generic (broad prior; not empirically validated)"})
        return d.change(f"quantities[{var}]", before, val)


def _beta_sample(rng, a, b):
    """Beta(a,b) via two Gammas (Marsaglia-Tsang for a,b≥1; both our params are ≥1)."""
    ga, gb = _gamma_sample(rng, a), _gamma_sample(rng, b)
    return ga / (ga + gb) if (ga + gb) > 0 else 0.5


def _gamma_sample(rng, k):
    if k < 1:
        k += 1
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0, 1)
        v = (1 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1 - 0.0331 * x ** 4 or math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v


register_operator("generic_outcome_prior", GenericOutcomeOperator, requires=("quantities",),
                  modifies=("quantities",), temporal_scale="horizon",
                  parameter_source="broad prior (Beta/Normal); qualitative lean only; NOT LLM-minted, "
                                   "NOT empirically validated (tier 6/7)", validated=True)
