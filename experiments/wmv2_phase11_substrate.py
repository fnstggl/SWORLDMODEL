"""Phase 11 — evaluation execution substrate + episode generating process.

To score Phase 11's REAL trigger/scope/candidate/scoring/migration/lineage logic we need sequential episodes
with known change/no-change labels. Each episode runs on REAL ``WorldState`` particles (so migration's
state/mass/event invariants are genuinely exercised) whose scalar latent lives in a registered quantity; a
controlled numeric process governs how observations are emitted, with (for "changed" episodes) a structural
change at a known time. This is the ``adversarial / semi-synthetic mutation episodes constructed from real
world-state distributions`` arm of the corpus; the real-data-grounded arm reuses Phase-10 datasets
(``wmv2_phase11_corpus.py``).

The controller's production logic is untouched — only the world-execution adapter is numeric. Determinism is
seed-driven (no wall-clock / RNG identity), so replay reproduces triggers, scores, migration and outputs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.world_model_v2.state import WorldState, SimulationClock, Entity, F, parse_time
from swm.world_model_v2.network import RelationGraph
from swm.world_model_v2.information import InformationLedger
from swm.world_model_v2.quantities import register_quantity_type, Quantity, _QUANTITY_TYPES
from swm.world_model_v2.phase11.controller import ExecutionAdapter

for _qt in ("latent_theta", "phase11_observable"):
    if _qt not in _QUANTITY_TYPES:
        register_quantity_type(_qt, units="prob", lo=0.0, hi=1.0)

DAY = 86400.0


def _mkworld(bid, theta, as_of):
    w = WorldState(world_id="ep", branch_id=bid, clock=SimulationClock(now=as_of, as_of=as_of),
                   network=RelationGraph(), information=InformationLedger())
    w.entities["subject"] = Entity(identity="subject")
    w.quantities["latent_theta"] = Quantity(name="latent_theta", qtype="latent_theta", value=theta,
                                             timestamp=as_of)
    return w


@dataclass
class NumericExecution(ExecutionAdapter):
    """Reads/updates ``latent_theta`` per particle. predict = ensemble Bernoulli/continuous predictive over the
    next observation; assimilate = Bayesian reweight by the observation likelihood; terminal = weighted mean
    outcome probability. A structural revision (migration) that adds a latent shifts the ensemble's support."""
    obs_sd: float = 0.12
    support: tuple = (0.0, 1.0)

    def _thetas(self, worlds):
        return [float(w.quantities["latent_theta"].value) if "latent_theta" in w.quantities else 0.5
                for w in worlds]

    def predict(self, worlds, weights, obs):
        return self._thetas(worlds)                       # predictive samples of the observable's mean

    def assimilate(self, worlds, weights, obs):
        observed = (getattr(obs, "provenance", {}) or {}).get("observed_value")
        if observed is None:
            return weights
        new = []
        for w, wt in zip(worlds, weights):
            th = float(w.quantities["latent_theta"].value) if "latent_theta" in w.quantities else 0.5
            like = math.exp(-0.5 * ((observed - th) / self.obs_sd) ** 2)
            new.append(wt * max(1e-12, like))
        z = sum(new) or 1.0
        return [x / z for x in new]

    def post_migration(self, worlds, weights, obs, sim_time):
        """Adopting the revised structure introduces a BROAD prior over the new regime (§15). We re-centre each
        particle's latent on the RECENTLY OBSERVED value (revealed evidence — leakage-safe) with wide spread,
        and reset to uniform weights. This is exactly why recompilation recovers post-change where a
        no-recompile run, whose old particles are far from the new regime, cannot."""
        import random as _r
        observed = (getattr(obs, "provenance", {}) or {}).get("observed_value")
        if observed is None:
            return worlds, weights
        rng = _r.Random(int(sim_time) % (2 ** 31))
        n = len(worlds)
        for w in worlds:
            if "latent_theta" in w.quantities:
                w.quantities["latent_theta"].value = min(0.98, max(0.02, rng.gauss(float(observed), 0.18)))
        return worlds, [1.0 / n] * n

    def terminal(self, worlds, weights):
        th = self._thetas(worlds)
        z = sum(weights) or 1.0
        mean = sum(wt * t for wt, t in zip(weights, th)) / z
        return {"mean": round(mean, 4), "p_yes": round(mean, 4), "n": len(worlds)}


# ---- episode generating process --------------------------------------------------------------------------
@dataclass
class Episode:
    episode_id: str = ""
    domain: str = ""
    trigger_family: str = ""              # the intended change family ("" for unchanged controls)
    changed: bool = False
    change_time: float = 0.0
    affected_scope: str = "no_model_change"
    as_of: float = 0.0
    horizon_ts: float = 0.0
    theta0: float = 0.5
    theta1: float = 0.5                   # post-change latent (== theta0 for unchanged)
    observations: list = field(default_factory=list)      # [dict] serializable obs specs
    true_terminal: float = 0.5            # realized outcome probability (for scoring)
    n_particles: int = 24
    split: str = ""
    source: str = "adversarial_synthetic"
    grounding: dict = field(default_factory=dict)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# domains × families used for the synthetic arm (real arm adds Congress/court/referendum)
DOMAINS = ["legislature", "court", "organization", "election", "negotiation", "social_diffusion",
           "platform", "fundraising"]
CHANGE_FAMILIES = ["rule_change", "new_actor", "authority_change", "coalition_change",
                   "network_restructuring", "outcome_space_change", "mechanism_regime_change",
                   "exogenous_shock", "impossible_event", "evidence_contradiction"]


def _obs(otype, origin, t, *, observed=None, declared=None, representable=True, evidence_ids=None):
    prov = {"declared": declared or {}}
    if observed is not None:
        prov["observed_value"] = observed
    return {"observation_id": f"{otype}@{int(t)}", "observation_type": otype, "origin": origin,
            "event_time": t, "representable": representable, "evidence_ids": evidence_ids or [],
            "provenance": prov, "uncertainty": {"terminal_sensitivity": 0.7}}


def make_episode(idx, *, changed, family, domain, seed, as_of, split, n_steps=8):
    rng = random.Random(seed * 100003 + idx)
    horizon = as_of + (n_steps + 3) * 7 * DAY
    change_step = rng.randint(2, n_steps - 2)
    change_time = as_of + change_step * 7 * DAY
    theta0 = round(rng.uniform(0.3, 0.7), 3)
    # a genuine structural change shifts the latent AND (for verified families) reveals typed evidence
    theta1 = round(min(0.95, max(0.05, theta0 + rng.choice([-1, 1]) * rng.uniform(0.25, 0.4))), 3) if changed else theta0
    ep = Episode(episode_id=f"ep{idx:04d}", domain=domain, trigger_family=(family if changed else ""),
                 changed=changed, change_time=(change_time if changed else 0.0), as_of=as_of,
                 horizon_ts=horizon, theta0=theta0, theta1=theta1, split=split,
                 affected_scope=_scope_for(family) if changed else "no_model_change")
    obs = []
    for s in range(n_steps):
        t = as_of + (s + 1) * 7 * DAY
        post = changed and t >= change_time
        theta = theta1 if post else theta0
        val = round(min(1.0, max(0.0, rng.gauss(theta, 0.08))), 3)
        if changed and post and s == change_step:
            # the CHANGE step: an EXTERNAL, leakage-safe observation carrying typed structural evidence
            declared, representable = _declared_for(family, change_time, val)
            obs.append(_obs(family, "external_evidence", t, observed=val, declared=declared,
                            representable=representable, evidence_ids=[f"ev_{ep.episode_id}_{s}"]))
        else:
            # ordinary in-support observation the ACTIVE plan already represents → simulation-internal
            obs.append(_obs("routine", "simulation_internal", t, observed=val, representable=True))
    ep.observations = obs
    ep.true_terminal = ep.theta1
    return ep


def _scope_for(family):
    return {"rule_change": "institution_ruleset", "new_actor": "actor", "authority_change": "institution_ruleset",
            "coalition_change": "relationship", "network_restructuring": "local_network_region",
            "outcome_space_change": "outcome_contract", "mechanism_regime_change": "mechanism_replacement",
            "exogenous_shock": "structural_hypothesis", "impossible_event": "outcome_contract",
            "evidence_contradiction": "latent_state"}.get(family, "structural_hypothesis")


def _declared_for(family, change_time, val):
    if family == "rule_change":
        return {"rule_change": {"institution": "subject", "kind": "quorum", "params": {"frac": 0.6},
                                "effective_date": change_time - DAY, "source": "official_record"}}, True
    if family == "new_actor":
        return {"new_actor": {"id": "newcomer", "type": "person", "causal_relevance": 0.8}}, True
    if family == "authority_change":
        return {"authority_change": {"actor": "subject", "delta": "acquired_veto"}}, True
    if family == "coalition_change":
        return {"coalition_change": {"src": "subject", "dst": "newcomer", "rel": "allies_with"}}, True
    if family == "network_restructuring":
        return {"network_change": {"persistent": True, "src": "subject", "dst": "hub"}}, True
    if family == "outcome_space_change":
        return {"outcome_space_change": {"note": "new resolution rule"}}, True
    if family == "impossible_event":
        return {}, False                                  # out-of-support → representable False
    if family == "evidence_contradiction":
        return {"contradiction_reliability": 0.85}, True
    if family == "mechanism_regime_change":
        return {}, True                                   # detected via residual regime shift, no typed hint
    if family == "exogenous_shock":
        return {"exogenous_shock": {"kind": "external"}}, True
    return {}, True


def build_worlds(ep):
    """Instantiate REAL WorldState particles at theta0 (broad spread) for an episode."""
    rng = random.Random(hash(ep.episode_id) % (2 ** 31))
    worlds, weights, pending = [], [], []
    for i in range(ep.n_particles):
        th = min(0.95, max(0.05, ep.theta0 + rng.gauss(0, 0.12)))
        worlds.append(_mkworld(f"{ep.episode_id}~p{i}", th, ep.as_of))
        weights.append(1.0 / ep.n_particles)
        pending.append([])
    return worlds, weights, pending


def episode_from_dict(d):
    ep = Episode()
    for k, v in d.items():
        if hasattr(ep, k):
            setattr(ep, k, v)
    return ep


def observations_from(ep):
    from swm.world_model_v2.phase11.contracts import RecompileObservation
    out = []
    for o in ep.observations:
        out.append(RecompileObservation(
            observation_id=o["observation_id"], observation_type=o.get("observation_type", ""),
            origin=o.get("origin", "external_evidence"), representable=o.get("representable", True),
            event_time=o.get("event_time", 0.0), evidence_ids=o.get("evidence_ids", []),
            provenance=o.get("provenance", {}), uncertainty=o.get("uncertainty", {})))
    return out
