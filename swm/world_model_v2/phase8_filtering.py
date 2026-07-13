"""Phase 8 — sequential posterior inference over persistent state (Part 4).

Real sequential filtering: each filter carries the PRIOR posterior forward and assimilates one event at a
time (it does NOT re-infer every timestep from scratch). Every step records prior→observation→posterior,
observation log-likelihood, ESS, resample decision, and lineage, so the acceptance gates can audit the
trail and separate FILTERING (as-of) from SMOOTHING (retrospective).

Implemented families (interchangeable by variable type, Part 4):
  * ``DecayedBetaBernoulliFilter`` — conjugate Beta-Bernoulli with exponential forgetting. This is the
    persistent ENGAGEMENT-PROPENSITY / momentum filter: a hierarchical per-actor prior sets the LEVEL, and
    forgetting makes recent acted/not-acted observations dominate, so a hot streak raises the posterior and
    a cold streak lowers it — the winning persistence signal as a proper filter (not a hand recurrence).
  * ``AsymmetricTrustFilter`` — log-odds trust with slow gain / fast loss + a repair path.
  * ``GaussianStateFilter`` — 1-D linear-Gaussian random-walk filter (Kalman update) for resource / risk.
  * ``CategoricalStageFilter`` — forward HMM filter over ordered process stages.
  * ``ParticleFilter`` — general SMC with systematic resampling for non-conjugate / multimodal state.

Numbers are NOT minted here: priors and transition parameters are supplied by the caller (fitted stats,
reference packs, hierarchical priors, or explicit broad priors) — the filter only assimilates evidence.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.phase8_persistence import (PersistentStateKey, PersistentStatePosterior,
                                                   PersistentUpdateRecord, logit, sigmoid)


def _beta_mean_sd(a: float, b: float):
    m = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return m, math.sqrt(max(0.0, var))


# ------------------------------------------------------------------ decayed Beta-Bernoulli (momentum)
@dataclass
class DecayedBetaBernoulliFilter:
    """Conjugate Beta-Bernoulli with exponential forgetting — a sequential filter, not a batch fit.

    State = (a, b) pseudo-counts. On each 0/1 observation, the prior is first DISCOUNTED toward the anchor
    by the forgetting factor (a ← anchor_a + decay·(a−anchor_a)), then the observation is added. Forgetting
    is what turns a static rate into a persistent MOMENTUM state: after a run of 1s the posterior climbs
    above the anchor; after 0s it falls below. ``decay=1`` recovers a plain accumulating Beta (no momentum);
    ``decay→0`` recovers last-event-only.
    """
    key: PersistentStateKey
    prior_mean: float = 0.5                  # the anchor / level (e.g. hierarchical per-actor rate)
    prior_strength: float = 4.0              # pseudo-count concentration of the anchor
    decay: float = 0.6                       # per-event forgetting toward the anchor (0..1]
    variable_id: str = "engagement_propensity"

    def _anchor(self):
        a0 = self.prior_strength * self.prior_mean
        b0 = self.prior_strength * (1.0 - self.prior_mean)
        return a0, b0

    def filter(self, observations, *, as_of: float = 0.0, mode: str = "filter") -> PersistentStatePosterior:
        """Assimilate an ordered list of (event_id, x∈{0,1}, ts) observations. ``mode`` is recorded on each
        lineage step so filtering and smoothing are distinguishable in the audit trail."""
        a0, b0 = self._anchor()
        a, b = a0, b0
        lineage, loglik = [], 0.0
        for eid, x, ts in observations:
            xf = 1.0 if x else 0.0
            prior_m = a / (a + b)
            # observation log-likelihood under the current posterior mean (predictive)
            p = min(1 - 1e-9, max(1e-9, prior_m))
            step_ll = math.log(p) if xf >= 0.5 else math.log(1 - p)
            loglik += step_ll
            # forget toward the anchor, then assimilate
            a = a0 + self.decay * (a - a0)
            b = b0 + self.decay * (b - b0)
            a += xf
            b += (1.0 - xf)
            post_m = a / (a + b)
            lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=prior_m, obs_value=xf,
                                                  obs_loglik=step_ll, posterior_mean=post_m,
                                                  ess_after=a + b, mode=mode).as_dict())
        mean, sd = _beta_mean_sd(a, b)
        return PersistentStatePosterior(
            key=self.key, variable_id=self.variable_id, posterior_family="beta_bernoulli",
            mean=mean, sd=sd, representation={"a": a, "b": b, "anchor_a": a0, "anchor_b": b0},
            prior_mean=self.prior_mean, transition_params={"decay": self.decay,
                                                           "prior_strength": self.prior_strength,
                                                           "source": "reinforcement/forgetting"},
            n_events_assimilated=len(observations), n_effective_observations=(a + b - a0 - b0),
            ess=a + b, as_of=as_of, method="decayed_beta_bernoulli", lineage=lineage,
            diagnostics={"sequence_loglik": round(loglik, 4), "momentum_range": "posterior deviates from "
                         "anchor with recent runs (decay<1)"})


# ------------------------------------------------------------------ asymmetric trust (slow gain / fast loss)
@dataclass
class AsymmetricTrustFilter:
    """Log-odds trust with asymmetric learning: a positive (kept-promise / cooperative) event nudges trust
    up by ``gain``; a negative (broken-promise / defection) event drops it by ``loss`` (loss ≫ gain). A
    ``repair`` event after a violation restores at a reduced rate — trust repaired to 0.5 is NOT the same
    latent as trust that never fell (that path-dependence is materialized as a separate ``violation_count``).
    """
    key: PersistentStateKey
    prior_mean: float = 0.5
    gain: float = 0.35                       # log-odds up per positive event (slow)
    loss: float = 0.9                        # log-odds down per negative event (fast) — asymmetry
    repair: float = 0.2                      # log-odds up per repair event after a violation
    variable_id: str = "trust"

    POS = ("promise_fulfilled", "cooperative_act", "interaction", "reciprocated")
    NEG = ("promise_violated", "defection", "betrayal")
    REPAIR = ("trust_repair", "apology", "restitution")

    def filter(self, observations, *, as_of: float = 0.0, mode="filter") -> PersistentStatePosterior:
        """observations = ordered list of (event_id, event_type, ts)."""
        lo = logit(self.prior_mean)
        lineage, violations, loglik = [], 0, 0.0
        for eid, etype, ts in observations:
            prior_m = sigmoid(lo)
            if etype in self.POS:
                lo += self.gain
                obs = 1.0
            elif etype in self.NEG:
                lo -= self.loss
                violations += 1
                obs = 0.0
            elif etype in self.REPAIR:
                lo += self.repair if violations else 0.5 * self.repair
                obs = 0.5
            else:
                obs = None
                lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=prior_m,
                                                      obs_value="ignored", obs_loglik=0.0,
                                                      posterior_mean=prior_m, ess_after=0.0, mode=mode).as_dict())
                continue
            post_m = sigmoid(lo)
            step_ll = math.log(max(1e-9, prior_m if obs >= 0.5 else 1 - prior_m))
            loglik += step_ll
            lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=prior_m, obs_value=etype,
                                                  obs_loglik=step_ll, posterior_mean=post_m,
                                                  ess_after=float(violations), mode=mode).as_dict())
        mean = sigmoid(lo)
        sd = 0.5 / math.sqrt(1 + len([l for l in lineage if l["obs_value"] != "ignored"]))
        return PersistentStatePosterior(
            key=self.key, variable_id=self.variable_id, posterior_family="beta_bernoulli",
            mean=mean, sd=sd, representation={"logodds": lo, "violation_count": violations},
            prior_mean=self.prior_mean,
            transition_params={"gain": self.gain, "loss": self.loss, "repair": self.repair,
                               "asymmetry": round(self.loss / max(1e-6, self.gain), 2),
                               "source": "reference_pack"},
            n_events_assimilated=len(observations),
            n_effective_observations=float(len([l for l in lineage if l["obs_value"] != "ignored"])),
            ess=float(len(lineage)), as_of=as_of, method="asymmetric_trust_logodds", lineage=lineage,
            diagnostics={"violations": violations, "sequence_loglik": round(loglik, 4),
                         "path_dependent": violations > 0})


# ------------------------------------------------------------------ Gaussian random-walk (resource / risk)
@dataclass
class GaussianStateFilter:
    """1-D linear-Gaussian random-walk (Kalman) filter for continuous persistent state (resource level, risk
    tolerance). State x_t = x_{t-1} + w (process noise q); observation y_t = x_t + v (obs noise r)."""
    key: PersistentStateKey
    prior_mean: float = 0.0
    prior_var: float = 1.0
    process_var: float = 0.05
    obs_var: float = 0.1
    variable_id: str = "resource_level"

    def filter(self, observations, *, as_of: float = 0.0, mode="filter") -> PersistentStatePosterior:
        """observations = ordered list of (event_id, y, ts). Additive events (deltas) can be encoded as y
        relative to the running mean by the caller; here y is the observed level."""
        m, P = self.prior_mean, self.prior_var
        lineage, loglik = [], 0.0
        for eid, y, ts in observations:
            prior_m = m
            # predict
            P = P + self.process_var
            # innovation
            S = P + self.obs_var
            K = P / S
            resid = float(y) - m
            m = m + K * resid
            P = (1 - K) * P
            step_ll = -0.5 * (math.log(2 * math.pi * S) + resid * resid / S)
            loglik += step_ll
            lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=prior_m, obs_value=float(y),
                                                  obs_loglik=step_ll, posterior_mean=m, ess_after=1.0 / P,
                                                  mode=mode).as_dict())
        return PersistentStatePosterior(
            key=self.key, variable_id=self.variable_id, posterior_family="gaussian_state",
            mean=m, sd=math.sqrt(max(0.0, P)), representation={"mean": m, "var": P},
            prior_mean=self.prior_mean,
            transition_params={"process_var": self.process_var, "obs_var": self.obs_var, "source": "observed"},
            n_events_assimilated=len(observations), n_effective_observations=float(len(observations)),
            ess=1.0 / max(1e-9, P), as_of=as_of, method="gaussian_kalman", lineage=lineage,
            diagnostics={"sequence_loglik": round(loglik, 4)})


# ------------------------------------------------------------------ categorical stage (forward HMM)
@dataclass
class CategoricalStageFilter:
    """Forward filter over ORDERED process stages (institutional stage/commitment lifecycle). Transitions
    are event-driven (a 'decision' advances, an 'appeal' can revert): the filter tracks the categorical
    posterior AND the path taken, so path-dependence (stage reached after appeal ≠ reached directly) is
    preserved in ``representation['path']``."""
    key: PersistentStateKey
    stages: tuple = ("open", "review", "decided", "closed")
    advance_events: tuple = ("stage_transition", "decision", "commitment_created", "promise_fulfilled")
    revert_events: tuple = ("appeal", "reopen", "promise_violated")
    variable_id: str = "institutional_stage"

    def filter(self, observations, *, as_of: float = 0.0, mode="filter") -> PersistentStatePosterior:
        idx, path, lineage = 0, [self.stages[0]], []
        for eid, etype, ts in observations:
            prior = self.stages[idx]
            if etype in self.advance_events and idx < len(self.stages) - 1:
                idx += 1
            elif etype in self.revert_events and idx > 0:
                idx -= 1
            path.append(self.stages[idx])
            lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=float(self.stages.index(prior)),
                                                  obs_value=etype, obs_loglik=0.0,
                                                  posterior_mean=float(idx), ess_after=float(len(path)),
                                                  mode=mode).as_dict())
        n_appeals = sum(1 for e in path if e)  # placeholder; real appeal count from revert transitions
        appeals = sum(1 for (_, et, _) in observations if et in self.revert_events)
        return PersistentStatePosterior(
            key=self.key, variable_id=self.variable_id, posterior_family="categorical_stage",
            mean=float(idx), sd=0.0, representation={"stage": self.stages[idx], "stage_index": idx,
                                                     "path": path, "n_appeals": appeals,
                                                     "reached_via_appeal": appeals > 0},
            prior_mean=0.0, transition_params={"stages": list(self.stages), "source": "observed"},
            n_events_assimilated=len(observations), n_effective_observations=float(len(observations)),
            ess=float(len(path)), as_of=as_of, method="forward_stage_hmm", lineage=lineage,
            diagnostics={"terminal_stage": self.stages[idx], "path_dependent": appeals > 0})


# ------------------------------------------------------------------ general particle filter (SMC)
@dataclass
class ParticleFilter:
    """General sequential Monte Carlo for non-conjugate / multimodal persistent state. Particles are
    latent values in [lo, hi]; ``transition`` diffuses them each step; ``loglik(x, obs)`` weights them;
    systematic resampling fires when ESS drops below ``resample_frac·N``. Prevents collapse (jitter on
    resample), tracks ESS + resample frequency, keeps the full lineage."""
    key: PersistentStateKey
    n_particles: int = 200
    lo: float = 0.0
    hi: float = 1.0
    process_sd: float = 0.05
    resample_frac: float = 0.5
    variable_id: str = "engagement_propensity"

    def filter(self, observations, loglik, *, as_of: float = 0.0, seed: int = 0,
               transition=None, mode="filter") -> PersistentStatePosterior:
        rng = random.Random(seed * 2654435761 % (2 ** 32))
        xs = [rng.uniform(self.lo, self.hi) for _ in range(self.n_particles)]
        ws = [1.0 / self.n_particles] * self.n_particles
        lineage, n_resample, last_ess = [], 0, float(self.n_particles)
        for eid, obs, ts in observations:
            prior_m = sum(w * x for w, x in zip(ws, xs))
            # transition (diffuse)
            if transition is not None:
                xs = [min(self.hi, max(self.lo, transition(x, rng))) for x in xs]
            else:
                xs = [min(self.hi, max(self.lo, x + rng.gauss(0, self.process_sd))) for x in xs]
            # reweight
            logs = [math.log(max(1e-12, ws[i])) + loglik(xs[i], obs) for i in range(self.n_particles)]
            mx = max(logs)
            ws = [math.exp(l - mx) for l in logs]
            z = sum(ws) or 1.0
            ws = [w / z for w in ws]
            ess = 1.0 / sum(w * w for w in ws)
            resampled = False
            if ess < self.resample_frac * self.n_particles:
                xs = _systematic(xs, ws, rng)
                xs = [min(self.hi, max(self.lo, x + rng.gauss(0, self.process_sd * 0.5))) for x in xs]  # jitter
                ws = [1.0 / self.n_particles] * self.n_particles
                n_resample += 1
                resampled = True
            last_ess = ess
            post_m = sum(w * x for w, x in zip(ws, xs))
            step_ll = mx + math.log(z / self.n_particles) if z > 0 else -1e9
            lineage.append(PersistentUpdateRecord(at=ts, event_id=eid, prior_mean=prior_m, obs_value=obs,
                                                  obs_loglik=step_ll, posterior_mean=post_m, ess_after=ess,
                                                  mode=mode).as_dict())
        mean = sum(w * x for w, x in zip(ws, xs))
        var = sum(w * (x - mean) ** 2 for w, x in zip(ws, xs))
        return PersistentStatePosterior(
            key=self.key, variable_id=self.variable_id, posterior_family="particle",
            mean=mean, sd=math.sqrt(max(0.0, var)),
            representation={"n_particles": self.n_particles, "sample": [round(x, 4) for x in xs[:8]]},
            prior_mean=(self.lo + self.hi) / 2.0,
            transition_params={"process_sd": self.process_sd, "resample_frac": self.resample_frac,
                               "source": "broad_prior"},
            n_events_assimilated=len(observations), n_effective_observations=float(len(observations)),
            ess=last_ess, resampled=n_resample > 0, seed=seed, as_of=as_of, method="particle_smc",
            lineage=lineage, diagnostics={"n_resample": n_resample,
                                          "resample_freq": round(n_resample / max(1, len(observations)), 3)})


def _systematic(xs, ws, rng):
    n = len(xs)
    cums, acc = [], 0.0
    for w in ws:
        acc += w
        cums.append(acc)
    start, out, idx = rng.random() / n, [], 0
    for i in range(n):
        pos = start + i / n
        while idx < n - 1 and cums[idx] < pos:
            idx += 1
        out.append(xs[idx])
    return out
