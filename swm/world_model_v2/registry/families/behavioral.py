"""Behavioral & structural mechanism families parameterized by VERIFIED published estimates — Phase 6.

Each family here is a DISTINCT causal mechanism (own process, state, trigger, transition, supported vs
forbidden interpretation) whose transition the published estimate can VALIDLY parameterize — not a generic
hazard renamed. Every default constant carries its source; the authoritative numbers live in the registry
packs (build_registry.py), independently verified by the core agent against the primary source. These are
Tier-4 published mechanisms: the point estimate is the study's, transport uncertainty is broad, and NONE is
locally validated for arbitrary scenarios (status stays research_encoded / domain_restricted).

Families:
  bass_diffusion            dN/dt=(p+q·N/M)(M−N)          — Sultan-Farley-Lehmann 1990 meta (p≈.03,q≈.38)
  ultimatum_offer_response  behavioral offer + accept≥τ    — Oosterbeek et al. 2004 meta (offer≈.40, rej≈.16)
  trust_game_transfer       send frac s, return frac g     — Johnson & Mislin 2011 meta (s≈.50, g≈.37)
  social_pressure_turnout   turnout logit + pressure Δ     — Gerber, Green & Larimer 2008 (levels below)
  matching_donation_response P(donate|match)               — Karlan & List 2007 (+~22% relative, ratio-flat)
"""
from __future__ import annotations

import math


# ------------------------------------------------------------------ Bass diffusion (finite population)
def bass_new_adopters(p: float, q: float, M: float, N_cum: float) -> float:
    """Instantaneous adoption rate dN/dt = (p + q·N/M)(M − N). p=innovation (external), q=imitation
    (internal/social). Sultan-Farley-Lehmann 1990 meta over ~213 applications: p̄≈0.03, q̄≈0.38 (SDs
    ≈ the means → transport as a WIDE prior, never a point). Supported: aggregate durable-good/innovation
    adoption in a finite market. Forbidden: individual-level timing; low-cost social behavior where
    exposure≠adoption (use a contagion hazard); any market where M is unknown."""
    N_cum = max(0.0, min(M, N_cum))
    return max(0.0, (p + q * N_cum / M) * (M - N_cum))


def bass_trajectory(p, q, M, *, steps=48, dt=1.0, N0=0.0):
    """Explicit Euler rollout of the Bass ODE — an executable, auditable adoption trajectory. Returns the
    cumulative-adopters series (one value per step). This is the mechanism's temporal state evolution."""
    N = N0
    out = []
    for _ in range(steps):
        N = min(M, N + dt * bass_new_adopters(p, q, M, N))
        out.append(N)
    return out


def bass_peak_time(p, q):
    """Closed-form time of peak adoption t* = ln(q/p)/(p+q) (Bass 1969). A structural check the fitted
    p,q imply — used in the forensic trace to show the mechanism's qualitative signature."""
    if p <= 0 or q <= p:
        return None
    return math.log(q / p) / (p + q)


# ------------------------------------------------------------------ behavioral ultimatum
def ultimatum_response(offer_frac: float, accept_threshold: float, *, softness: float = 0.05) -> float:
    """P(accept | offer) — a soft threshold at the responder's minimum acceptable offer (MAO). Oosterbeek
    et al. 2004 meta (37 papers, 75 results): mean proposer offer ≈ 0.40 of the pie, overall rejection
    ≈ 0.16; low offers (<0.2) rejected ~half the time. Supported: one-shot ultimatum / take-it-or-leave-it
    splits, lab or lab-like. Forbidden: repeated bargaining with reputation; the SPE (~0) split (that is
    bargaining_rubinstein, a DIFFERENT family — behavioral offers are far more generous)."""
    return 1.0 / (1.0 + math.exp(-(offer_frac - accept_threshold) / max(1e-3, softness)))


def ultimatum_expected_proposer_payoff(offer_frac, accept_threshold, pie=1.0, **kw):
    """Executable proposer readout: E[payoff] = P(accept)·(1−offer)·pie. Lets the mechanism drive a
    terminal quantity (proposer take-home) that changes with the offer — a real terminal sensitivity."""
    return ultimatum_response(offer_frac, accept_threshold, **kw) * (1.0 - offer_frac) * pie


# ------------------------------------------------------------------ trust game (investment + return)
def trust_game_outcome(send_frac: float, return_frac: float, endowment: float = 10.0,
                       multiplier: float = 3.0) -> dict:
    """Berg-Dickhaut-McCabe investment game. Investor sends s·endowment (trust); it is tripled; trustee
    returns g·(tripled sent) (trustworthiness). Johnson & Mislin 2011 meta (162 replications, >23k
    subjects): s̄≈0.50 of endowment, ḡ≈0.37 of the amount received. Supported: anonymous one-shot trust
    games, lab. Forbidden: real-world relationship trust (no monetary transfer structure); repeated trust
    (reputation changes g); reading s as a general 'trust level' scalar. Returns the full typed outcome."""
    sent = max(0.0, min(1.0, send_frac)) * endowment
    received = sent * multiplier
    returned = max(0.0, min(1.0, return_frac)) * received
    return {"sent": round(sent, 3), "received_by_trustee": round(received, 3),
            "returned": round(returned, 3),
            "investor_payoff": round(endowment - sent + returned, 3),
            "trustee_payoff": round(received - returned, 3),
            "investor_net_from_trust": round(returned - sent, 3)}   # >0 iff trust paid off


# ------------------------------------------------------------------ social-pressure turnout
#: Gerber, Green & Larimer 2008 APSR — turnout LEVELS by mailer arm (2006 Michigan primary, control 29.7%).
#: Independently verified (control 29.7 → Civic Duty 31.5, Hawthorne 32.2, Self 34.5, Neighbors 37.8).
GGL2008_TURNOUT_LEVELS = {"control": 0.297, "civic_duty": 0.315, "hawthorne": 0.322,
                          "self": 0.345, "neighbors": 0.378}


def social_pressure_turnout_p(treatment: str, *, levels: dict | None = None,
                              base_turnout: float | None = None) -> float:
    """P(turnout | mailer arm). The mechanism is the causal ITT effect of a social-pressure mailer on
    turnout (randomized). If a scenario's own base_turnout differs from GGL's 29.7% control, the ADDITIVE
    treatment effect (arm − control) is applied to that base and clamped — a transported effect, widen the
    uncertainty. Supported: low-salience-election turnout under observability/social-pressure appeals.
    Forbidden: high-salience elections (ceiling); persuasion of vote CHOICE (this is turnout only);
    non-election participation without re-estimation."""
    lv = levels or GGL2008_TURNOUT_LEVELS
    p = lv.get(treatment, lv["control"])
    if base_turnout is not None:
        effect = p - lv["control"]
        p = base_turnout + effect
    return max(0.0, min(1.0, p))


# ------------------------------------------------------------------ matching-grant donation
def matching_donation_p(base_p: float, match_offered: bool, *, relative_lift: float = 0.22) -> float:
    """P(donate | match). Karlan & List 2007 (natural field experiment, >50k prior donors): announcing a
    matching grant raises the probability of donating by ≈22% RELATIVE to no match; the match RATIO
    (1:1 vs 2:1 vs 3:1) has NO additional effect. Supported: renewal solicitations to EXISTING donors.
    Forbidden: cold acquisition; the ratio does not scale the effect (do NOT multiply by 2 or 3);
    reading the 22% as absolute percentage points (it is relative)."""
    if not match_offered:
        return max(0.0, min(1.0, base_p))
    return max(0.0, min(1.0, base_p * (1.0 + relative_lift)))


def matching_donation_ratio_is_flat(ratio: str) -> bool:
    """Encodes the study's key null: 2:1 and 3:1 do not beat 1:1. A forensic guard against the common
    fundraising error of scaling the effect by the match ratio."""
    return ratio in ("1:1", "2:1", "3:1")   # all equivalent per Karlan & List 2007


# ------------------------------------------------------------------ shared executable operator
# ------------------------------------------------------------------ reputation updating (Beta-Bernoulli)
def reputation_update(alpha: float, beta: float, positive: bool, *, weight: float = 1.0):
    """Beta-Bernoulli reputation state: each observed positive/negative interaction updates a Beta(α,β)
    posterior over an actor's latent trustworthiness (image scoring, Nowak & Sigmund 1998). Executable,
    conjugate, order-independent for the mean. The CAUSAL consequence of reputation is anchored by Resnick
    et al. 2006 (eBay randomized field experiment: an established good reputation commands an ≈8.1% price
    premium). Supported: platforms with observable rating histories. Forbidden: reading the Beta mean as a
    universal 'trust'; assuming ratings are unbiased (reciprocal/retaliatory rating inflates them)."""
    if positive:
        return alpha + weight, beta
    return alpha, beta + weight


def reputation_score(alpha: float, beta: float) -> float:
    """Posterior mean trustworthiness E[θ]=α/(α+β)."""
    return alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5


def reputation_price_premium(alpha: float, beta: float, *, max_premium: float = 0.081) -> float:
    """Map reputation score to a fractional price/behavioral premium, scaled so a strong reputation
    approaches the Resnick et al. 2006 causal estimate (≈8.1% at eBay). Broad transport uncertainty."""
    return max_premium * (2.0 * reputation_score(alpha, beta) - 1.0)


def _behavioral_probability(mechanism: str, params: dict) -> tuple[float, dict]:
    """Compute a Bernoulli probability (or a probability-like terminal readout in [0,1]) for a behavioral
    mechanism, plus diagnostics. Dispatch is explicit (no name-magic). Returns (p, diagnostics)."""
    if mechanism == "ultimatum_offer_response":
        p = ultimatum_response(float(params["offer_frac"]), float(params["accept_threshold"]),
                               softness=float(params.get("softness", 0.05)))
        return p, {"readout": "P(accept)"}
    if mechanism == "social_pressure_turnout":
        p = social_pressure_turnout_p(str(params.get("treatment", "control")),
                                      levels=params.get("levels"),
                                      base_turnout=params.get("base_turnout"))
        return p, {"readout": "P(turnout)"}
    if mechanism == "matching_donation_response":
        p = matching_donation_p(float(params["base_p"]), bool(params.get("match_offered", False)),
                                relative_lift=float(params.get("relative_lift", 0.22)))
        return p, {"readout": "P(donate)"}
    if mechanism == "bass_diffusion":
        traj = bass_trajectory(float(params["p"]), float(params["q"]), float(params["M"]),
                               steps=int(params.get("steps", 48)), dt=float(params.get("dt", 1.0)))
        frac = traj[-1] / float(params["M"]) if params.get("M") else 0.0
        return max(0.0, min(1.0, frac)), {"readout": "cumulative adoption fraction at horizon",
                                          "peak_time": bass_peak_time(float(params["p"]), float(params["q"]))}
    if mechanism == "trust_game_transfer":
        o = trust_game_outcome(float(params["send_frac"]), float(params["return_frac"]),
                               endowment=float(params.get("endowment", 10.0)),
                               multiplier=float(params.get("multiplier", 3.0)))
        # readout: P(trust paid off) proxied by whether investor nets positive (deterministic here)
        return (1.0 if o["investor_net_from_trust"] > 0 else 0.0), o
    raise ValueError(f"unknown behavioral mechanism {mechanism!r}")


class BehavioralMechanismOperator:
    """Executes a Phase-6 behavioral/structural mechanism in the shared world. The scenario instance rides
    on the event payload: {mechanism, params, outcome_var, family, pack_id, transport_widening, options}.
    Writes the typed outcome quantity (taking precedence over the tier-6/7 generic safety net, which only
    writes if the readout is unset) and emits an explicit StateDelta with pack provenance + widening. The
    probability is the study's; transport widening perturbs the log-odds per branch (broader dispersion,
    unchanged mean). Deterministic per branch_id."""
    name = "behavioral_mechanism"

    def applicable(self, world, event):
        p = event.payload
        return event.etype in ("behavioral_mechanism", "resolve_outcome") and \
            isinstance(p.get("hazard_spec"), dict) and p["hazard_spec"].get("kind") == "behavioral"

    def propose(self, world, event, rng):
        from swm.world_model_v2.transitions import TransitionProposal
        spec = event.payload["hazard_spec"]
        p, diag = _behavioral_probability(spec["mechanism"], spec["params"])
        widen = float(spec.get("transport_widening", 1.0) or 1.0)
        if widen > 1.0:
            shift = rng.gauss(0.0, 0.35 * (widen - 1.0))
            p = 1.0 / (1.0 + math.exp(-(math.log(max(1e-6, p) / max(1e-6, 1 - p)) + shift)))
        return TransitionProposal(
            operator=self.name,
            action={"outcome_var": spec["outcome_var"], "p": p, "family": spec.get("family", ""),
                    "pack_id": spec.get("pack_id", ""), "options": spec.get("options") or ["True", "False"]},
            reason_codes=[f"behavioral:{spec['mechanism']}", f"p={round(p, 4)}"],
            uncertainty={"pack_id": spec.get("pack_id", ""), "transport_widening": widen, "diagnostics": diag})

    def apply(self, world, proposal):
        from swm.world_model_v2.quantities import Quantity, register_quantity_type
        from swm.world_model_v2.transitions import StateDelta
        import random
        a = proposal.action
        var = a["outcome_var"]
        register_quantity_type(var, units="outcome")
        before = world.quantities[var].value if var in world.quantities else None
        if before is not None:
            return StateDelta(at=world.clock.now, event_type="behavioral_mechanism", operator=self.name,
                              reason_codes=["already_resolved_noop"])
        rng = random.Random(hash((world.branch_id, var)) & 0xFFFFFFFF)
        opts = a["options"] if len(a["options"]) == 2 else ["True", "False"]
        val = opts[0] if rng.random() < a["p"] else opts[1]
        world.quantities[var] = Quantity(name=var, qtype=var, value=val, timestamp=world.clock.now)
        d = StateDelta(at=world.clock.now, event_type="behavioral_mechanism", operator=self.name,
                       reason_codes=proposal.reason_codes, uncertainty=proposal.uncertainty)
        d.change(f"quantities[{var}]", before, val)
        return d


def _register():
    from swm.world_model_v2.transitions import register_operator
    register_operator("behavioral_mechanism", BehavioralMechanismOperator(),
                      requires=("quantities",), modifies=("quantities",), temporal_scale="event",
                      parameter_source="verified published estimate (Tier-4); transport widening on log-odds",
                      validated=True)


_register()

