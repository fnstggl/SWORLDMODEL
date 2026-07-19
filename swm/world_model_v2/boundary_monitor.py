"""Runtime boundary expansion (§7) + outside-world entry — the boundary is not frozen at compile.

Two operators on the canonical event queue plus a per-branch scheduler:

``schedule_outside_arrivals(plan, world, queue)`` — at branch construction, every samplable
``ExternalEventFamily`` of the model's ``OutsideWorldProcess`` draws its arrival times with the
BRANCH-ROOT rng (matched across counterfactual arms: clones share the root stream, so the same
outside shocks hit every arm — §22 CRN) and schedules typed ``outside_world_event``s. Unresolved
families are never sampled; they ride the result as unresolved external risks.

``OutsideWorldEntryOperator`` — converts one arrival into its DECLARED entry mechanism (§5.1):
observation delivery routes through the generated world's control plane (delivery → attention →
bounded cognition); resource/capacity/price changes write ONLY a matching typed quantity (no
match ⇒ the impact is recorded unresolved — nothing is invented); every other mechanism enters
as an observable semantic event PLUS an honest record when its full mechanical effect is not yet
representable. Terminal/readout paths are never written (enforced here and in tests).

``BoundaryMonitorOperator`` — watches semantic events for boundary-expansion triggers: newly
mentioned actors matching promotion rules or critic findings, action targets outside the
boundary, external families whose promotion trigger fired. Promotion happens at the EXACT event
time that made the component relevant (§30):
  * actor promotion (§7.1): entity created; history reconstructed ONLY from what they could have
    observed (public-visibility items up to now — never branch-private history); actor-local
    information state seeded from those items; cognitive state initializes lazily at their first
    real decision trigger (the hypothesizer), so no omniscient snapshot exists;
  * population-member promotion (§7.2): a ``population_promotions`` record marks the segment
    decrement so the person is never double-counted inside the remaining aggregate;
  * external-system promotion (§7.3): the family is marked promoted and subsequent arrivals are
    tagged internal-mechanism candidates (recorded; a validated internal mechanism must exist
    for full interior simulation — absent one, the gap is honest, not hidden).
"""
from __future__ import annotations

import hashlib
import json
import random as _random

from swm.world_model_v2.events import Event, event_type_registered, register_event_type
from swm.world_model_v2.transitions import StateDelta, ValidationResult

OUTSIDE_EVENT_TYPE = "outside_world_event"
if not event_type_registered(OUTSIDE_EVENT_TYPE):
    register_event_type(OUTSIDE_EVENT_TYPE, scheduling="stochastic", validated=True,
                        parameter_source="outside_world residual process (defensible arrivals "
                                         "only; unresolved families never sampled)")


def _hash(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _branch_root(world) -> str:
    return str(getattr(world, "branch_id", "") or "").split(":")[0]


def schedule_outside_arrivals(plan, world, queue) -> int:
    """Sample and schedule this branch's outside-world arrivals (matched across arms via the
    branch-ROOT rng). Returns the number of scheduled events. No process → 0. Records the
    unresolved families on the world so aggregation surfaces them."""
    from swm.world_model_v2.outside_world import entry_event_payload
    proc = getattr(plan, "_outside_world", None)
    if proc is None:
        return 0
    root = _branch_root(world)
    rng = _random.Random(int(_hash(["outside", root]), 16) & 0x7FFFFFFF)
    t0, t1 = float(world.clock.now), float(getattr(plan, "horizon_ts", world.clock.now))
    n = 0
    from swm.world_model_v2.outside_world import sample_arrivals
    for fam in proc.samplable():
        try:
            times = sample_arrivals(fam, t0=t0, t1=t1, rng=rng)
        except ValueError:
            continue
        for k, at in enumerate(times[:24]):
            payload = entry_event_payload(fam, at=at, branch_id=str(world.branch_id),
                                          arrival_index=k, rng=rng)
            queue.schedule(Event(ts=float(at), etype=OUTSIDE_EVENT_TYPE, participants=[],
                                 payload=payload, visibility="world",
                                 source=f"outside_world:{fam.family_id}"))
            n += 1
    if proc.unresolved():
        world.omissions.append({
            "kind": "unresolved_outside_world_families",
            "families": [f.family_id for f in proc.unresolved()][:8],
            "reason": "no defensible arrival model — surfaced as unresolved external risk, "
                      "never sampled (§5.2)"})
    return n


class OutsideWorldEntryOperator:
    """Route one outside-world arrival through its declared typed entry mechanism (§5.1)."""

    name = "outside_world_entry"

    def __init__(self, *, report: dict = None):
        self.report = report if report is not None else {}

    def applicable(self, world, event):
        return event.etype == OUTSIDE_EVENT_TYPE

    def run(self, world, event, rng):
        p = dict(event.payload or {})
        mech = str(p.get("entry_mechanism", "observation_delivery"))
        fam = str(p.get("outside_world_family", "external"))
        mark = str(p.get("mark", ""))[:300]
        delta = StateDelta(at=world.clock.now, event_type=OUTSIDE_EVENT_TYPE,
                           operator=self.name,
                           reason_codes=[f"outside:{fam}", f"entry:{mech}"])
        self.report["outside_events_entered"] = self.report.get("outside_events_entered", 0) + 1
        # ---- resource-like entries write ONLY a matching typed quantity ----
        if mech in ("resource_change", "capacity_change", "price_change"):
            target = self._matching_quantity(world, p.get("affected_boundary_components") or [])
            if target is None:
                self.report.setdefault("outside_unresolved_impacts", []).append(
                    {"family": fam, "mechanism": mech,
                     "reason": "no typed quantity matches the affected component — impact NOT "
                               "invented; entering as observable news only"})
                return self._as_observation(world, event, delta, fam, mark, downgraded=mech)
            q = world.quantities.get(target)
            cur = float(getattr(q, "value", 0.0) or 0.0)
            # the mark's direction is qualitative; magnitude must come from the family's
            # grounded data — absent one, a shock is recorded as an OBSERVED disturbance flag
            # on the quantity's provenance, never an invented magnitude
            delta.change(f"quantities[{target}].outside_disturbance", None,
                         f"{fam}@{world.clock.now}")
            self.report.setdefault("outside_quantity_disturbances", []).append(
                {"family": fam, "quantity": target, "at": world.clock.now})
            return self._as_observation(world, event, delta, fam, mark, downgraded="")
        if mech == "actor_promotion":
            delta.follow_up_events = [{
                "etype": "ctrl_boundary_promotion", "ts": world.clock.now,
                "participants": [], "payload": {"component": mark or fam, "kind":
                                                "individual_actor",
                                                "trigger": f"outside_world:{fam}"}}]
            return delta, ValidationResult(ok=True)
        # ---- everything else enters as an observable semantic event (delivery → attention →
        # cognition); mechanisms whose full mechanical effect is not representable are recorded
        downgraded = "" if mech in ("observation_delivery", "population_exposure") else mech
        return self._as_observation(world, event, delta, fam, mark, downgraded=downgraded)

    @staticmethod
    def _matching_quantity(world, components) -> str | None:
        names = {str(c).lower().replace(" ", "_") for c in components}
        for qname in (world.quantities or {}):
            ql = str(qname).lower()
            if any(n in ql or ql in n for n in names if len(n) > 3):
                return qname
        return None

    def _as_observation(self, world, event, delta, fam, mark, *, downgraded: str):
        if downgraded:
            self.report.setdefault("outside_entry_downgrades", []).append(
                {"family": fam, "declared_mechanism": downgraded,
                 "served_as": "observable event (full mechanical effect not representable — "
                              "recorded, not invented)"})
            delta.reason_codes.append(f"entry_downgraded:{downgraded}->observation")
        if getattr(world, "scenario_schema", None) is not None:
            sev = {"event_id": f"owe_{_hash([fam, world.clock.now])[:12]}",
                   "semantic_type_id": f"outside_{fam}"[:60],
                   "exact_content": mark or f"external development: {fam}",
                   "source_actor_id": "outside_world",
                   "intended_visibility": "public",
                   "observability_paths": list(event.payload.get("observability_paths") or []),
                   "outside_world_family": fam}
            delta.follow_up_events = list(delta.follow_up_events or []) + [{
                "etype": "ctrl_semantic_event", "ts": world.clock.now, "participants": [],
                "payload": {"semantic_event": sev,
                            "reason": "outside-world arrival entering through observation "
                                      "delivery"}}]
        else:
            # fixed-v1 worlds: publish + availability through the temporal attention plane
            try:
                from swm.world_model_v2.information import InformationItem
                from swm.world_model_v2.temporal_runtime import (record_available_observation,
                                                                 schedule_attention,
                                                                 temporal_model_of, get_stats)
                iid = f"owe_{_hash([fam, world.clock.now])[:12]}"
                if world.information is not None:
                    world.information.publish(InformationItem(
                        iid, mark or f"[{fam}]", kind="public", source="outside_world",
                        created_at=world.clock.now, about=fam[:60]))
                model = temporal_model_of(world)
                stats = get_stats(world)
                for actor_id in list(world.entities or {})[:12]:
                    record_available_observation(
                        world, recipient=actor_id,
                        item={"iid": iid, "content": mark, "source": "outside_world",
                              "urgency": 0.0},
                        available_ts=world.clock.now, channel="news", stats=stats)
                    att = schedule_attention(world, model, actor_id=actor_id,
                                             channel_id="news",
                                             available_ts=world.clock.now, stats=stats)
                    if att is not None:
                        delta.follow_up_events = list(delta.follow_up_events or []) + [{
                            "etype": att.etype, "ts": att.ts,
                            "participants": list(att.participants),
                            "payload": dict(att.payload)}]
            except Exception:  # noqa: BLE001 — availability plumbing absent: recorded only
                self.report.setdefault("outside_entry_downgrades", []).append(
                    {"family": fam, "declared_mechanism": "observation_delivery",
                     "served_as": "recorded_only (no information plane on this world)"})
        return delta, ValidationResult(ok=True)


if not event_type_registered("ctrl_boundary_promotion"):
    register_event_type("ctrl_boundary_promotion", scheduling="scheduled", validated=True,
                        parameter_source="boundary monitor (§7 dynamic promotion)")


class BoundaryMonitorOperator:
    """§7: inspect events for boundary-expansion triggers and promote at the exact event time."""

    name = "boundary_monitor"

    def __init__(self, boundary=None, *, report: dict = None, llm=None):
        self.boundary = boundary                      # compile-time WorldBoundary (shared, read-only
        #                                               except promotion records)
        self.report = report if report is not None else {}
        self.llm = llm

    def applicable(self, world, event):
        if self.boundary is None:
            return False
        return event.etype in ("ctrl_semantic_event", "ctrl_boundary_promotion",
                               OUTSIDE_EVENT_TYPE)

    # ------------------------------------------------------------------ trigger detection
    def _promotion_candidates(self, world, event) -> list:
        """Deterministic trigger matching: promotion rules / unresolved components whose name or
        trigger text appears in the event's content/participants/targets, plus explicit
        ctrl_boundary_promotion requests. No LLM minting — the boundary's own rules fire."""
        out = []
        p = dict(event.payload or {})
        if event.etype == "ctrl_boundary_promotion":
            comp = str(p.get("component", "")).strip()
            if comp:
                out.append({"name": comp, "kind": str(p.get("kind", "individual_actor")),
                            "trigger": str(p.get("trigger", "explicit_promotion_event"))})
            return out
        sev = p.get("semantic_event") or {}
        text = " ".join([str(sev.get("exact_content", "")), str(p.get("mark", "")),
                         " ".join(map(str, event.participants or [])),
                         str(sev.get("target_actor_id", ""))]).lower()
        if not text.strip():
            return out
        rules = list(self.boundary.promotion_rules or [])
        for u in self.boundary.unresolved_components:
            if u.get("sensitivity") in ("decisive", "material"):
                rules.append({"component": u.get("name", ""), "trigger": u.get("name", ""),
                              "action": "promote_to_individual"})
        seen = set()
        for r in rules:
            name = str(r.get("component", "")).strip()
            trig = str(r.get("trigger", "")).strip().lower()
            if not name or name.lower() in seen:
                continue
            name_hit = len(name) > 3 and name.lower() in text
            trig_hit = len(trig) > 6 and trig[:60] in text
            if name_hit or trig_hit:
                seen.add(name.lower())
                comp = self.boundary.component(name)
                out.append({"name": name,
                            "kind": getattr(comp, "kind", "individual_actor"),
                            "trigger": (f"event matched promotion rule "
                                        f"({'name' if name_hit else 'trigger'} match)")})
        return out

    # ------------------------------------------------------------------ §7.1 actor promotion
    def _reconstructable_history(self, world, now_ts: float) -> list:
        """ONLY what the promoted actor could have observed: public-visibility information items
        up to now. Never branch-private communications, private states, or simulator variables."""
        out = []
        info = getattr(world, "information", None)
        items = getattr(info, "items", None) if info is not None else None
        if isinstance(items, dict):
            for iid, item in list(items.items())[:200]:
                kind = str(getattr(item, "kind", ""))
                created = float(getattr(item, "created_at", 0.0) or 0.0)
                if kind == "public" and created <= now_ts:
                    out.append({"iid": iid, "content": str(getattr(item, "content", ""))[:200],
                                "created_at": created})
        out.sort(key=lambda r: r["created_at"])
        return out[-40:]

    def run(self, world, event, rng):
        candidates = self._promotion_candidates(world, event)
        if not candidates:
            return None, ValidationResult(ok=True, reasons=["no_boundary_trigger"])
        delta = StateDelta(at=world.clock.now, event_type=event.etype, operator=self.name,
                           reason_codes=[])
        promos = getattr(world, "boundary_promotions", None)
        if promos is None:
            promos = []
            world.boundary_promotions = promos
        for cand in candidates[:3]:
            name = cand["name"]
            actor_id = name.strip()
            if actor_id in (world.entities or {}):
                continue
            history = self._reconstructable_history(world, world.clock.now)
            # §7.1: promotion CREATES the entity (WorldState.entity only looks up and raises
            # on unknown ids — the promoted actor does not exist yet, that is the point)
            from swm.world_model_v2.state import Entity, F
            ent = world.entities.get(actor_id)
            if ent is None:
                ent = Entity(actor_id)
                world.entities[actor_id] = ent
            ent.set("latent_state",
                    F({"promoted_by_boundary_monitor": True,
                       "promoted_at": world.clock.now,
                       "trigger": cand["trigger"],
                       "role_hint": cand.get("kind", "individual_actor")},
                      status="derived", method="boundary_promotion",
                      updated_at=world.clock.now), key="boundary_promotion_record")
            # actor-local information state: expose ONLY the reconstructable public items
            if world.information is not None:
                for h in history[-20:]:
                    try:
                        world.information.expose(actor_id, h["iid"], world.clock.now,
                                                 channel="public_record")
                    except KeyError:
                        continue
            # §7.2: if the person was inside a declared population, record the decrement so
            # aggregation never double-counts them
            seg = self._segment_of(world, name)
            pop_rec = None
            if seg:
                pop_promos = getattr(world, "population_promotions", None)
                if pop_promos is None:
                    pop_promos = []
                    world.population_promotions = pop_promos
                pop_rec = {"segment": seg, "person": actor_id, "at": world.clock.now}
                pop_promos.append(pop_rec)
            rec = {"component": name, "kind": cand.get("kind", "individual_actor"),
                   "at": world.clock.now, "trigger": cand["trigger"],
                   "promoted_to": "individual_actor",
                   "reconstructed_history_events": len(history),
                   "population_decrement": pop_rec,
                   "branch_id": str(world.branch_id)}
            promos.append(rec)
            self.report.setdefault("boundary_promotions", []).append(rec)
            try:
                self.boundary.record_promotion(name=name, kind=cand.get("kind",
                                                                        "individual_actor"),
                                               at=world.clock.now, trigger=cand["trigger"],
                                               promoted_to="individual_actor",
                                               reconstructed_history=len(history))
            except Exception:  # noqa: BLE001 — shared boundary may be frozen; branch record stands
                pass
            delta.reason_codes.append(f"boundary_promotion:{actor_id}")
            delta.change(f"{actor_id}.latent_state[boundary_promotion_record]", None,
                         world.clock.now)
            # §7.1.8: schedule a decision ONLY when a real trigger exists — the promoting event
            # itself is that trigger when it names/addresses the actor; delivery→attention→
            # cognition then govern whether and when they actually engage
            sev = (event.payload or {}).get("semantic_event") or {}
            if getattr(world, "scenario_schema", None) is not None and sev:
                delta.follow_up_events = list(delta.follow_up_events or []) + [{
                    "etype": "ctrl_deliver_observation", "ts": world.clock.now,
                    "participants": [actor_id],
                    "payload": {"recipient": actor_id, "semantic_event": dict(sev),
                                "channel": str(sev.get("channel", "public")),
                                "representation": "complete",
                                "reason": "promotion trigger delivers the promoting event"}}]
        if not delta.reason_codes:
            return None, ValidationResult(ok=True, reasons=["already_inside_boundary"])
        return delta, ValidationResult(ok=True)

    @staticmethod
    def _segment_of(world, name: str) -> str:
        """Best-effort match of a promoted person to a declared population segment."""
        pops = getattr(world, "populations", None) or {}
        try:
            items = pops.items() if isinstance(pops, dict) else []
        except Exception:  # noqa: BLE001
            items = []
        lowered = name.lower()
        for seg_id, seg in items:
            seg_text = f"{seg_id} {getattr(seg, 'description', '')}".lower()
            for tok in lowered.replace("_", " ").split():
                if len(tok) > 4 and tok in seg_text:
                    return str(seg_id)
        return ""
