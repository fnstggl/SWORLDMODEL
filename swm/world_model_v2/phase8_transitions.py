"""Phase 8 — in-world persistent-state transitions (Parts 5, 7D/E).

These operators run DURING a rollout and close the loop: an action or outcome event updates persistent
latent state, which changes later actor views and action distributions. Each transition is typed, registered
through the CANONICAL operator registry (not a competing one), reads the state it needs, and emits BOTH a
``StateDelta`` (for the rollout branch log) AND a ``PersistentStateDelta`` (for the persistence audit) — no
silent in-place mutation.

Families here complement, rather than duplicate, the existing operators: ``transitions.py`` already covers
generic belief/relationship/resource shifts; Phase 8 adds the PATH-DEPENDENT persistent families that the
existing operators do not model — asymmetric trust with an explicit violation count + repair, commitment
lifecycle, reputation accrual, habit reinforcement of the engagement latent, and episodic memory
consolidation/decay. Transition parameters are labeled priors / reference packs; none are LLM-minted.
"""
from __future__ import annotations

import math

from swm.world_model_v2.phase8_persistence import PersistentStateDelta, logit, sigmoid
from swm.world_model_v2.state import F, StateField
from swm.world_model_v2.transitions import StateDelta, TransitionOperator, TransitionProposal, register_operator

# reference-pack transition parameters (labeled; NOT minted) — asymmetric trust learning rates
TRUST_GAIN, TRUST_LOSS, TRUST_REPAIR = 0.35, 0.9, 0.2
HABIT_LR = 0.2
REPUTATION_GAIN, REPUTATION_DECAY = 0.15, 0.05
MEMORY_HALFLIFE_DAYS = 30.0


def _latent(actor, key, default=0.0):
    latent = actor.get("latent_state") or {}
    sf = latent.get(key) if isinstance(latent, dict) else None
    return float(sf.value) if isinstance(sf, StateField) and isinstance(sf.value, (int, float)) else default


def _set_latent(actor, key, value, at, *, method, sources=()):
    actor.set("latent_state", F(value, status="derived", method=method, sources=list(sources),
                                updated_at=at), key=key)


class PersistenceUpdateOperator(TransitionOperator):
    """The unified in-world persistence transition. Dispatches on event type across the persistent families
    and writes the corresponding WorldState field, so the NEXT ActorView/policy sees changed state.

      * reinforcement (engagement latent): a positive engagement outcome raises the actor's
        ``phase4_policy_value:engage`` Q-value the reinforcement family reads;
      * asymmetric trust: cooperative/defection/repair events move a network edge's trust with slow-gain /
        fast-loss + a tracked violation count (path-dependence);
      * reputation accrual: a public outcome moves the actor's reputation belief with slow recovery;
      * commitment lifecycle: create/fulfill/violate move the actor's commitment record.

    Emits a StateDelta (rollout log) with the PersistentStateDelta attached under uncertainty['p8_delta'].
    """
    name = "persistence_update"

    POS_ENGAGE = ("engaged", "converted", "clicked", "responded")
    TRUST_POS = ("promise_fulfilled", "cooperative_act", "reciprocated")
    TRUST_NEG = ("promise_violated", "defection", "betrayal")
    TRUST_REPAIR = ("trust_repair", "apology", "restitution")

    def applicable(self, world, event):
        return event.etype in ("actor_action", "policy_feedback", "actor_reaction", "external_shock",
                               "public_outcome", "collective_vote") and bool(event.participants)

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name,
                                  action={"etype": event.etype, "payload": dict(event.payload or {}),
                                          "participants": list(event.participants)})

    def apply(self, world, proposal):
        a = proposal.action
        payload, parts = a["payload"], a["participants"]
        actor_id = parts[0]
        actor = world.entity(actor_id)
        at = world.clock.now
        d = StateDelta(at=at, event_type="persistence_update", operator=self.name)
        outcome = str(payload.get("outcome", payload.get("kind", "")))
        p8 = None
        # ---- reinforcement of the engagement latent (habit/RL Q-value the policy reads) ----
        if outcome in self.POS_ENGAGE or payload.get("reward") is not None:
            key = "phase4_policy_value:engage"
            before = _latent(actor, key, 0.0)
            reward = float(payload.get("reward", 1.0 if outcome in self.POS_ENGAGE else 0.0))
            after = before + HABIT_LR * (reward - before)
            _set_latent(actor, key, after, at, method="phase8_reinforcement",
                        sources=[str(payload.get("source_event_id", ""))])
            d.change(f"{actor_id}.latent_state[{key}]", round(before, 4), round(after, 4))
            p8 = PersistentStateDelta(at=at, variable_id="engagement_propensity",
                                      key=f"{actor_id}:engagement_propensity",
                                      transition_family="reinforcement", before=round(before, 4),
                                      after=round(after, 4), reason_codes=["in_world_reinforcement"],
                                      uncertainty={"learning_rate": HABIT_LR})
        # ---- asymmetric trust update on a dyadic edge ----
        elif outcome in self.TRUST_POS + self.TRUST_NEG + self.TRUST_REPAIR and len(parts) >= 2 and world.network:
            src, dst = actor_id, parts[1]
            rel = str(payload.get("rel", "trusts"))
            e = world.network.edge(src, rel, dst) if hasattr(world.network, "edge") else None
            if e is None:
                e = world.network.add(src, rel, dst)
            before = e.trust if getattr(e, "trust", None) is not None else 0.5
            lo = logit(before)
            vkey = "p8_violations"
            v = _latent(actor, f"{vkey}:{dst}", 0.0)
            if outcome in self.TRUST_POS:
                lo += TRUST_GAIN
            elif outcome in self.TRUST_NEG:
                lo -= TRUST_LOSS
                v += 1
                _set_latent(actor, f"{vkey}:{dst}", v, at, method="phase8_violation_count")
            else:
                lo += TRUST_REPAIR if v else 0.5 * TRUST_REPAIR
            e.trust = sigmoid(lo)
            d.change(f"edge({src},{rel},{dst}).trust", round(before, 4), round(e.trust, 4))
            p8 = PersistentStateDelta(at=at, variable_id="trust", key=f"{src}|{rel}|{dst}:trust",
                                      transition_family="trust_asymmetric", before=round(before, 4),
                                      after=round(e.trust, 4), reason_codes=["asymmetric_trust", outcome],
                                      uncertainty={"violation_count": v, "path_dependent": v > 0})
        # ---- reputation accrual (public outcome) ----
        elif a["etype"] in ("public_outcome", "collective_vote", "external_shock"):
            beliefs = actor.get("beliefs") or {}
            cur = beliefs.get("reputation") if isinstance(beliefs, dict) else None
            before = float(cur.value) if isinstance(cur, StateField) and isinstance(cur.value, (int, float)) else 0.5
            success = bool(payload.get("success", payload.get("passed", outcome in ("success", "won"))))
            after = min(1.0, max(0.0, before + (REPUTATION_GAIN if success else -REPUTATION_GAIN) * (1 - before if success else before)))
            actor.set("beliefs", F(after, status="derived", method="phase8_reputation", updated_at=at),
                      key="reputation")
            d.change(f"{actor_id}.beliefs[reputation]", round(before, 4), round(after, 4))
            p8 = PersistentStateDelta(at=at, variable_id="reputation", key=f"{actor_id}:reputation",
                                      transition_family="reputation_accrual", before=round(before, 4),
                                      after=round(after, 4), reason_codes=["reputation_accrual"])
        if p8 is not None:
            d.uncertainty["p8_delta"] = p8.as_dict()
        return d if d.changes else None


class MemoryConsolidationOperator(TransitionOperator):
    """Episodic memory consolidation/decay over elapsed time (Part 6). On a background tick, the actor's
    materialized memory traces decay in salience (exp half-life) and low-salience traces below a floor are
    forgotten — so an actor's RECALL changes with time, not just the external state. Emits a StateDelta only
    when recall actually changes."""
    name = "memory_consolidation"

    def applicable(self, world, event):
        return event.etype == "background_tick"

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name,
                                  action={"elapsed_days": float(event.payload.get("elapsed_days", 1.0))})

    def apply(self, world, proposal):
        days = proposal.action["elapsed_days"]
        decay = 0.5 ** (days / MEMORY_HALFLIFE_DAYS)
        at = world.clock.now
        d = StateDelta(at=at, event_type="memory_consolidation", operator=self.name,
                       uncertainty={"decay": round(decay, 4), "half_life_days": MEMORY_HALFLIFE_DAYS})
        for eid, ent in world.entities.items():
            mem = ent.get("memory")
            traces = list(mem.value) if isinstance(mem, StateField) and isinstance(mem.value, list) else []
            if not traces:
                continue
            before_n = len(traces)
            kept = []
            for t in traces:
                if isinstance(t, dict):
                    t = {**t, "salience": float(t.get("salience", 0.5)) * decay}
                    if t["salience"] >= 0.05:                # forgetting floor
                        kept.append(t)
                else:
                    kept.append(t)
            if len(kept) != before_n or before_n:
                ent.set("memory", F(kept, status="derived", method=self.name, updated_at=at))
                if len(kept) != before_n:
                    d.change(f"{eid}.memory[n_recallable]", before_n, len(kept))
        return d if d.changes else None


register_operator("persistence_update", PersistenceUpdateOperator,
                  requires=("entity.latent_state", "network"),
                  modifies=("entity.latent_state", "network.edges", "entity.beliefs"),
                  temporal_scale="event",
                  parameter_source="reference-pack asymmetric-learning params (labeled; not minted)",
                  validated=True)
register_operator("memory_consolidation", MemoryConsolidationOperator, requires=("entity.memory",),
                  modifies=("entity.memory",), temporal_scale="interval",
                  parameter_source="exponential memory half-life (labeled prior)", validated=True)
