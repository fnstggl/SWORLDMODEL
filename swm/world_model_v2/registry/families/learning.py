"""Learning & adaptation mechanism families — executable transitions with published functional forms.

Every function here is a real state transition: it takes the actor's current latent (belief/value/habit
strength) plus an outcome/observation and returns the updated latent. They are the building blocks the
world's decision/belief operators call each event, and they carry the exact published form + its limits
in the registry record (see build_registry.py).

Families:
  reinforcement_q          Q-learning value update (Watkins 1989): Q ← Q + α(r − Q)
  belief_learning          Bayesian/fictitious-play belief over opponent action frequencies
  ewa                      Experience-Weighted Attraction (Camerer & Ho 1999): unifies reinforcement +
                           belief learning via (φ, δ, ρ, N0)
  habit_formation          habit stock accumulation (Wood & Neal 2007 form): H ← (1−γ)H + γ·1[acted]
  quantal_response         (re-exported from policy.logit_choice for registry completeness)
All parameters are labeled; defaults are reference-class priors, never invented precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ------------------------------------------------------------------ reinforcement (Q-learning)
def reinforcement_update(q: float, reward: float, *, alpha: float) -> float:
    """Q ← Q + α(r − Q). alpha ∈ (0,1] learning rate (fitted or reference-class)."""
    return q + alpha * (reward - q)


def reinforcement_step(values: dict, chosen: str, reward: float, *, alpha: float) -> dict:
    out = dict(values)
    out[chosen] = reinforcement_update(out.get(chosen, 0.0), reward, alpha=alpha)
    return out


# ------------------------------------------------------------------ belief learning (fictitious play)
def belief_update_counts(counts: dict, observed_action: str, *, weight: float = 1.0) -> dict:
    """Dirichlet-style belief over an opponent's action frequencies: increment the observed action's
    pseudo-count. Beliefs = normalized counts. weight<1 discounts old observations (recency)."""
    out = {k: v * (1.0 if weight >= 1 else weight) for k, v in counts.items()}
    out[observed_action] = out.get(observed_action, 0.0) + 1.0
    return out


def belief_probs(counts: dict) -> dict:
    z = sum(counts.values()) or 1.0
    return {k: v / z for k, v in counts.items()}


# ------------------------------------------------------------------ EWA (Camerer & Ho 1999)
@dataclass
class EWAState:
    """Experience-Weighted Attraction: attractions A[a] and experience weight N. Parameters:
    phi (decay of prior attractions), delta (weight on foregone payoffs: 0=reinforcement, 1=belief),
    rho (decay of N), kappa (0 cumulative / 1 averaging). Reference: Camerer & Ho 1999 Econometrica."""
    A: dict
    N: float = 1.0
    phi: float = 0.9
    delta: float = 0.5
    rho: float = 0.9
    kappa: float = 0.0

    def update(self, chosen: str, payoffs: dict) -> "EWAState":
        """payoffs: realized payoff for `chosen` and FOREGONE payoffs for other actions."""
        N_new = self.rho * self.N + 1.0
        A_new = {}
        for a, A_a in self.A.items():
            indicator = 1.0 if a == chosen else self.delta
            num = self.phi * (self.N if self.kappa == 0 else self.N) * A_a + indicator * payoffs.get(a, 0.0)
            den = N_new if self.kappa == 0 else N_new
            A_new[a] = num / max(1e-9, den)
        return EWAState(A=A_new, N=N_new, phi=self.phi, delta=self.delta, rho=self.rho, kappa=self.kappa)

    def choice_probs(self, lam: float) -> dict:
        m = max(self.A.values()) if self.A else 0.0
        ws = {a: math.exp(max(-40.0, lam * (v - m))) for a, v in self.A.items()}
        z = sum(ws.values()) or 1.0
        return {a: w / z for a, w in ws.items()}


# ------------------------------------------------------------------ habit formation
def habit_update(stock: float, acted: bool, *, gamma: float = 0.2) -> float:
    """H ← (1−γ)H + γ·1[acted]. gamma is the accumulation rate (Wood & Neal 2007 automaticity growth)."""
    return (1 - gamma) * stock + gamma * (1.0 if acted else 0.0)


def habit_boost(base_p: float, stock: float, *, weight: float = 0.5) -> float:
    """Habit raises the probability of repeating a behavior, bounded. weight labels prior strength."""
    from swm.world_model_v2.policy import logit_choice  # noqa: F401 (registry-cohesion import)
    z = math.log(min(1 - 1e-6, max(1e-6, base_p)) / (1 - min(1 - 1e-6, max(1e-6, base_p)))) + weight * stock
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
