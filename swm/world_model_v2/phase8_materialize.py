"""Phase 8 — materialize filtered persistent posteriors into the shared WorldState (Parts 3, 6, 7).

This is the WORLD-STATE plane bridge: a posterior sitting in the persistence store is ornamental until it is
WRITTEN into an actual WorldState field that a mechanism reads. ``materialize_persistent_state`` copies each
posterior into the field its spec declares (``materializes_into``) and emits a ``PersistentStateDelta`` — no
silent in-place mutation. The targets are the exact fields the existing ActorView→policy path consumes:

    engagement_propensity → entity.latent_state[phase4_policy_value:engage]  (Phase-4 reinforcement family)
    habit_strength        → entity.past_actions                             (Phase-4 habit family)
    trust                 → network.edge.trust                              (Phase-4 reciprocity family)
    resource_level        → entity.resources                               (feasibility + resource_update)
    reputation/risk       → entity.beliefs[...]                            (utility / risk_sensitive family)
    institutional_stage   → entity.latent_state[institutional_stage]       (feasibility gate)

Because these are the fields ``ActorViewBuilder`` projects and ``ActorPolicyModel`` scores, removing history
(→ posterior reverts to prior → different field value → different ActorView → different action distribution)
is a genuine causal ablation, not a cosmetic diff.

``HistoryIngestor`` handles the Part-3 realities: partial/delayed/duplicate/out-of-order/conflicting events,
uncertain timestamps, and probabilistic identity linkage — feeding a leakage-safe ``EventLog``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.world_model_v2.phase8_events import EventLog, PersistentEvent
from swm.world_model_v2.phase8_persistence import (MemoryTrace, PersistentStateDelta, PersistentStateKey,
                                                   PersistentStateView, get_persistent_variable, logit)
from swm.world_model_v2.state import F, StateField


# ------------------------------------------------------------------ materialize posterior → WorldState
def _entity(world, eid):
    return world.entities.get(eid)


def materialize_persistent_state(world, posteriors, *, actor_map=None) -> list:
    """Write each ``PersistentStatePosterior`` into the WorldState field its spec declares, returning the
    list of ``PersistentStateDelta`` produced. ``actor_map`` optionally maps a posterior key's entity_id to
    a WorldState entity id (default identity). Idempotent: re-materializing the same posterior writes the
    same value and produces a delta with equal before/after only if unchanged."""
    deltas = []
    for post in posteriors:
        try:
            spec = get_persistent_variable(post.variable_id)
        except KeyError:
            continue
        target = spec.materializes_into
        eid = (actor_map or {}).get(post.key.entity_id, post.key.entity_id)
        delta = _write_target(world, target, post, eid)
        if delta is not None:
            deltas.append(delta)
    if deltas:
        world.uncertainty_meta.setdefault("phase8", {})["materialized"] = [d.as_dict() for d in deltas]
    return deltas


def _write_target(world, target: str, post, eid: str):
    at = world.clock.now
    prov = {"posterior_hash": post.posterior_hash(), "method": post.method,
            "n_events": post.n_events_assimilated, "transition_params": post.transition_params}
    # ---- entity.latent_state[key] ----
    if target.startswith("entity.latent_state["):
        key = target[len("entity.latent_state["):-1]
        ent = _entity(world, eid)
        if ent is None:
            return None
        cur = ent.get("latent_state", key=key)
        before = cur.value if isinstance(cur, StateField) else None
        after = post.mean if post.posterior_family != "categorical_stage" else post.representation.get("stage")
        ent.set("latent_state", F(after, status="derived", method="phase8_materialize",
                                  sources=[post.key.token()], confidence=1.0 - post.sd, updated_at=at), key=key)
        return _delta(post, target, before, after, at, prov)
    # ---- entity.beliefs[key] ----
    if target.startswith("entity.beliefs["):
        key = target[len("entity.beliefs["):-1]
        ent = _entity(world, eid)
        if ent is None:
            return None
        cur = ent.get("beliefs", key=key)
        before = cur.value if isinstance(cur, StateField) else None
        ent.set("beliefs", F(post.mean, status="derived", method="phase8_materialize",
                             sources=[post.key.token()], updated_at=at), key=key)
        return _delta(post, target, before, post.mean, at, prov)
    # ---- entity.resources (variable_id keyed) ----
    if target == "entity.resources":
        ent = _entity(world, eid)
        if ent is None:
            return None
        key = post.variable_id
        cur = ent.get("resources", key=key)
        before = cur.value if isinstance(cur, StateField) else None
        ent.set("resources", F(post.mean, status="derived", method="phase8_materialize",
                               sources=[post.key.token()], updated_at=at), key=key)
        return _delta(post, target, before, post.mean, at, prov)
    # ---- entity.past_actions (habit: append a materialized habit summary) ----
    if target == "entity.past_actions":
        ent = _entity(world, eid)
        if ent is None:
            return None
        cur = ent.get("past_actions")
        before = list(cur.value) if isinstance(cur, StateField) and isinstance(cur.value, list) else []
        # materialize habit strength as repeated action tokens the habit family counts (log1p of strength)
        n_repeat = int(round(math.log1p(max(0.0, post.mean * post.n_events_assimilated))))
        after = before + [{"at": at, "action": post.representation.get("action", "engage"),
                           "materialized_habit": True} for _ in range(n_repeat)]
        ent.set("past_actions", F(after, status="derived", method="phase8_materialize",
                                  sources=[post.key.token()], updated_at=at))
        return _delta(post, target, len(before), len(after), at, prov)
    # ---- entity.commitments ----
    if target == "entity.commitments":
        ent = _entity(world, eid)
        if ent is None:
            return None
        cur = ent.get("commitments")
        before = list(cur.value) if isinstance(cur, StateField) and isinstance(cur.value, list) else []
        stage = post.representation.get("stage", "open")
        after = [c for c in before if c.get("id") != post.key.entity_id]
        after.append({"id": post.key.entity_id, "state": stage, "at": at,
                      "reached_via_appeal": post.representation.get("reached_via_appeal", False)})
        ent.set("commitments", F(after, status="derived", method="phase8_materialize",
                                 sources=[post.key.token()], updated_at=at))
        return _delta(post, target, [c.get("state") for c in before], stage, at, prov)
    # ---- network.edge.trust / .strength ----
    if target in ("network.edge.trust", "network.edge.strength") and world.network is not None:
        parts = post.key.entity_id.split("|")
        if len(parts) >= 2:
            src, dst = parts[0], parts[-1]
            rel = parts[1] if len(parts) == 3 else "relates_to"
            e = world.network.edge(src, rel, dst) if hasattr(world.network, "edge") else None
            if e is None:
                try:
                    e = world.network.add(src, rel, dst)
                except Exception:
                    return None
            if target.endswith("trust"):
                before = getattr(e, "trust", None)
                e.trust = post.mean
                return _delta(post, target, before, post.mean, at, prov)
            else:
                before = float(e.strength.value) if hasattr(e, "strength") and isinstance(e.strength, StateField) else None
                if hasattr(e, "strength") and isinstance(e.strength, StateField):
                    e.strength.value = post.mean
                return _delta(post, target, before, post.mean, at, prov)
    return None


def _delta(post, target, before, after, at, prov):
    return PersistentStateDelta(
        at=at, variable_id=post.variable_id, key=post.key.token(),
        transition_family=get_persistent_variable(post.variable_id).transition_family,
        before=before, after=after, driven_by_event_id="", reason_codes=["phase8_materialize"],
        uncertainty={"sd": round(post.sd, 4), "ess": round(post.ess, 3)},
        evidence_deps=[r.get("event_id") for r in post.lineage[-5:]], provenance=prov)


# ------------------------------------------------------------------ actor-visible persistent view (Part 6/7)
def build_persistent_view(world, actor_id: str, posteriors, *, as_of: float, memory_store=None,
                          max_recall: int = 8) -> PersistentStateView:
    """Project the omniscient persistent posteriors into ONE actor's visible slice (Part 6/7). Only
    variables whose spec ``actor_visibility`` admits this actor surface; episodic memory is retrieved
    leakage-safe (strictly before ``as_of``) and is probabilistic — the actor does not get perfect recall.
    """
    view = PersistentStateView(actor_id=actor_id, as_of=as_of)
    for post in posteriors:
        try:
            spec = get_persistent_variable(post.variable_id)
        except KeyError:
            continue
        vis = spec.actor_visibility
        involves = actor_id in (post.key.entity_id, *post.key.entity_id.split("|"))
        if vis in ("self", "private") and post.key.entity_id != actor_id:
            continue
        if vis == "dyad" and actor_id not in post.key.entity_id.split("|"):
            continue
        if spec.scope == "actor" and post.key.entity_id == actor_id:
            view.beliefs[post.variable_id] = post.mean
            view.uncertainty[post.variable_id] = round(post.sd, 4)
        elif spec.variable_id == "trust" and involves:
            other = [p for p in post.key.entity_id.split("|") if p != actor_id]
            view.trust[other[0] if other else post.key.entity_id] = post.mean
        elif spec.variable_id == "reputation":
            view.reputation[post.key.entity_id] = post.mean
        elif spec.variable_id == "institutional_stage":
            view.institutional_stage[post.key.entity_id] = post.representation.get("stage", post.mean)
        if spec.variable_id == "risk_tolerance" and post.key.entity_id == actor_id:
            view.risk_tolerance = post.mean
    # episodic memory retrieval (leakage-safe, probabilistic)
    if memory_store is not None and hasattr(memory_store, "retrieve"):
        try:
            hits = memory_store.retrieve(actor_id, "", as_of=as_of, k=max_recall)
            for h in hits:
                ep = h.get("episode") if isinstance(h, dict) else h
                view.recalled_events.append(MemoryTrace(entity_id=actor_id, at=getattr(ep, "timestamp", as_of),
                                                        text=getattr(ep, "text", ""),
                                                        salience=getattr(ep, "importance", 0.5),
                                                        retrieval_prob=round(h.get("score", 1.0), 3)
                                                        if isinstance(h, dict) else 1.0).as_dict())
        except Exception:
            pass
    view.provenance = {"as_of": as_of, "n_posteriors": len(posteriors),
                       "leakage_mode": "filter", "memory": bool(memory_store)}
    return view


# ------------------------------------------------------------------ Part 3: robust history ingestion
@dataclass
class HistoryIngestor:
    """Ingests real event histories into a leakage-safe ``EventLog`` while handling the Part-3 realities:
    duplicate / out-of-order / delayed / conflicting events, uncertain timestamps, and PROBABILISTIC
    identity linkage. Uncertain attributions carry ``identity_link_uncertainty`` rather than being forced
    onto one actor with false certainty. Conflicts are appended as ``revised_observation`` events (the log
    never overwrites)."""
    log: EventLog
    alias_map: dict = field(default_factory=dict)          # raw actor token -> (canonical_id, link_uncertainty)
    _seen_conflicts: dict = field(default_factory=dict)     # (actor, event_time, type) -> outcome

    def resolve_identity(self, raw_actor: str) -> tuple:
        """Return (canonical_actor_id, link_uncertainty). Unknown tokens pass through with uncertainty 0.0
        (treated as their own canonical id); ambiguous aliases carry the declared uncertainty."""
        if raw_actor in self.alias_map:
            canon, unc = self.alias_map[raw_actor]
            return canon, float(unc)
        return raw_actor, 0.0

    def ingest(self, *, event_type: str, event_time: float, actor: str, outcome=None, params=None,
               observed_time=None, source_id: str = "", scope: str = "actor", visibility: str = "public",
               confidence: float = 1.0) -> tuple:
        """Ingest one raw record. Returns (event, is_new). Idempotent via the log's content-id dedup.
        Detects an outcome CONFLICT (same actor/time/type, different outcome) and records it as a typed
        ``revised_observation`` rather than silently overwriting."""
        canon, link_unc = self.resolve_identity(actor)
        ev = PersistentEvent(
            world_id=self.log.world_id, scenario_id=self.log.scenario_id, event_type=event_type,
            event_time=float(event_time),
            observed_time=float(observed_time) if observed_time is not None else float(event_time),
            actor_ids=(canon,), scope=scope, source_id=source_id, visibility=visibility,
            params=dict(params or {}), outcome=outcome, confidence=confidence,
            identity_link_uncertainty=link_unc)
        ckey = (canon, float(event_time), event_type)
        prior_outcome = self._seen_conflicts.get(ckey)
        if prior_outcome is not None and prior_outcome != outcome:
            # conflicting observation of the same event → append a typed revision (never overwrite)
            ev = PersistentEvent(
                world_id=self.log.world_id, scenario_id=self.log.scenario_id, event_type=event_type,
                event_time=float(event_time),
                observed_time=float(observed_time) if observed_time is not None else float(event_time),
                actor_ids=(canon,), scope=scope, source_id=source_id, visibility=visibility,
                params={**dict(params or {}), "_conflict_with_outcome": prior_outcome}, outcome=outcome,
                confidence=confidence, identity_link_uncertainty=link_unc, kind="revised_observation")
        self._seen_conflicts[ckey] = outcome
        return self.log.append(ev)
