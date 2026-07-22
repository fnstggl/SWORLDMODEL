"""D13 — actor knowledge packets. What an actor actually KNOWS at the moment they decide,
assembled from the real researched world — not a thin stub of labels and hashes.

A real decision-maker reasons from: who they are and what they may decide; their own private
mindset (D9, latent — never invented external events asserted as fact); the state of the world
they can observe (D12 shared conditions that affect them); the real facts available to them (D11
canonical facts, public and role-private, with credibility and open contradictions); and, inside
an institution, the live process — the proposal on the table, the stage, the deadline, the visible
tally, the substantive messages and commitments, their relationships and the positions they can
see (D14).

`ActorKnowledgePacket` assembles exactly this and RENDERS it as real content. Its leakage guards
are hard: it never carries another actor's private mindset, anything dated on/after as_of, a
future event, or a secret ballot. Universal — the packet is built from the stores, never
hand-authored per question."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm

KNOWLEDGE_PACKET_VERSION = "lean_v2.knowledge_packet.v1"


@dataclass
class ActorKnowledgePacket:
    actor_id: str
    role: str = ""
    authority: list = field(default_factory=list)
    # own private mindset (D9) — LATENT only
    latent_beliefs: list = field(default_factory=list)
    latent_goals: list = field(default_factory=list)
    latent_preferences: list = field(default_factory=list)
    risk_tolerance: str = ""
    hypothetical_assumptions: list = field(default_factory=list)   # simulated possibilities, flagged
    # the observable world (D12) — only conditions that affect this actor
    shared_conditions: dict = field(default_factory=dict)
    # real facts (D11), rendered as content with credibility, split by visibility
    public_facts: list = field(default_factory=list)
    role_private_facts: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    # institution process (D14)
    proposal: str = ""
    stage: str = ""
    deadline: str = ""
    visible_tally: dict = field(default_factory=dict)
    received_messages: list = field(default_factory=list)
    commitments: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    visible_positions: dict = field(default_factory=dict)          # others' PUBLIC positions only
    feasible_actions: list = field(default_factory=list)
    resources: list = field(default_factory=list)
    day: str = ""
    leakage_flags: list = field(default_factory=list)
    version: str = KNOWLEDGE_PACKET_VERSION

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("actor_id", "role", "authority", "latent_beliefs", "latent_goals",
                 "latent_preferences", "risk_tolerance", "hypothetical_assumptions",
                 "shared_conditions", "public_facts", "role_private_facts", "contradictions",
                 "proposal", "stage", "deadline", "visible_tally", "received_messages",
                 "commitments", "relationships", "visible_positions", "feasible_actions",
                 "resources", "day", "leakage_flags", "version")}

    def render(self) -> str:
        """The actor-facing packet: real content, no hashes. This is what the decision prompt sees."""
        L = [f"You are {self.actor_id}" + (f" ({self.role})" if self.role else "") + f". As of {self.day}."]
        if self.authority:
            L.append("Your authority: " + "; ".join(self.authority))
        if self.latent_beliefs:
            L.append("What you believe: " + "; ".join(self.latent_beliefs))
        if self.latent_goals:
            L.append("Your goals: " + "; ".join(self.latent_goals))
        if self.latent_preferences:
            L.append("Your preferences: " + "; ".join(self.latent_preferences))
        if self.risk_tolerance:
            L.append(f"Your risk posture: {self.risk_tolerance}")
        if self.shared_conditions:
            L.append("The situation as you understand it: "
                     + "; ".join(f"{k}={v}" for k, v in self.shared_conditions.items()))
        if self.public_facts:
            L.append("Public facts you know:\n  - " + "\n  - ".join(self.public_facts))
        if self.role_private_facts:
            L.append("What you privately know (your channels):\n  - "
                     + "\n  - ".join(self.role_private_facts))
        if self.contradictions:
            L.append("Conflicting reports (unresolved):\n  - " + "\n  - ".join(self.contradictions))
        if self.proposal:
            L.append(f"On the table: {self.proposal}"
                     + (f" (stage: {self.stage})" if self.stage else ""))
        if self.visible_tally:
            L.append("Visible tally so far: "
                     + "; ".join(f"{k}: {v}" for k, v in self.visible_tally.items()))
        if self.received_messages:
            L.append("Messages you have received:\n  - " + "\n  - ".join(self.received_messages))
        if self.visible_positions:
            L.append("Positions others have stated publicly: "
                     + "; ".join(f"{k}: {v}" for k, v in self.visible_positions.items()))
        if self.relationships:
            L.append("Your relationships: "
                     + "; ".join(f"{k}: {v}" for k, v in self.relationships.items()))
        if self.commitments:
            L.append("Your commitments: " + "; ".join(self.commitments))
        if self.hypothetical_assumptions:
            L.append("Possibilities you are UNSURE of (not established):\n  - "
                     + "\n  - ".join(self.hypothetical_assumptions))
        if self.deadline:
            L.append(f"Decision deadline: {self.deadline}")
        if self.feasible_actions:
            L.append("Your options: " + " | ".join(self.feasible_actions))
        return "\n".join(L)


def build_knowledge_packet(actor: dict, *, evidence_store=None, state=None, shared_world: dict = None,
                           institution_state: dict = None, day: str = "", roles: dict = None,
                           institutions: dict = None, feasible_actions: list = None,
                           received_messages: list = None) -> ActorKnowledgePacket:
    """Assemble the actor's knowledge packet from the real stores, applying every leakage guard.
    `state` is the actor's separated ActorStateHypothesis (D9); `evidence_store` an EvidenceStore
    (D11); `shared_world` the D12 combo; `institution_state` the live D14 process view."""
    aid = actor.get("id") or ""
    pkt = ActorKnowledgePacket(
        actor_id=aid, role=norm(actor.get("role"), 120),
        authority=[norm(a, 80) for a in (actor.get("authority") or [])], day=str(day or "")[:10],
        feasible_actions=list(feasible_actions or []),
        resources=[norm(r, 80) for r in (actor.get("resources") or [])])

    # own LATENT mindset only (D9) — never another actor's private state
    if state is not None:
        pkt.latent_beliefs = list(getattr(state, "latent_beliefs", None) or getattr(state, "beliefs", []))
        pkt.latent_goals = list(getattr(state, "latent_goals", None) or getattr(state, "goals", []))
        pkt.latent_preferences = list(getattr(state, "latent_preferences", None)
                                      or getattr(state, "stances", []))
        pkt.risk_tolerance = getattr(state, "latent_risk_tolerance", "")
        pkt.hypothetical_assumptions = list(getattr(state, "hypothetical_assumptions", []))
        pkt.relationships = dict(getattr(state, "relationships", {}) or {})
        pkt.commitments = list(getattr(state, "known_commitments", None)
                               or getattr(state, "commitments", []))

    # the observable world (D12) — only conditions that affect THIS actor
    for cid, sstate in (shared_world or {}).items():
        node_affects = ((institution_state or {}).get("affects", {}) or {}).get(cid)
        if node_affects is None or aid in node_affects:
            pkt.shared_conditions[cid] = sstate

    # real facts (D11), visibility-guarded and leakage-guarded, rendered as content
    if evidence_store is not None:
        facts = evidence_store.facts_for_actor(aid, day=day or evidence_store.as_of,
                                               roles=roles, institutions=institutions)
        for f in facts:
            if not f.knowable_on(day or evidence_store.as_of):
                pkt.leakage_flags.append(f"blocked non-knowable fact {f.fact_id}")
                continue
            (pkt.public_facts if f.visibility == "public" else pkt.role_private_facts).append(f.render())
        for grp, fs in evidence_store.contradictions().items():
            visible = [f for f in fs if f.visible_to(aid, roles=roles, institutions=institutions)]
            if len(visible) > 1:
                pkt.contradictions.append(f"[{grp}] " + " VS ".join(f.content for f in visible))

    # institution process (D14) — proposal/stage/deadline/visible tally/positions; secret ballots
    # and other actors' private state are NEVER included
    ist = institution_state or {}
    pkt.proposal = norm(ist.get("proposal"), 240)
    pkt.stage = norm(ist.get("stage"), 80)
    pkt.deadline = str(ist.get("deadline") or "")[:10]
    if ist.get("tally_visible"):
        pkt.visible_tally = dict(ist.get("tally") or {})
    pkt.visible_positions = {k: v for k, v in (ist.get("public_positions") or {}).items() if k != aid}
    pkt.received_messages = [norm(m, 240) for m in (received_messages or ist.get("messages_to", {})
                                                    .get(aid, []) if ist else [])]
    return pkt
