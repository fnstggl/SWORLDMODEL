"""Compact actor prompts: a stable per-actor context prefix + a per-decision delta.

Full fidelity resends an actor's complete rendered situation on every call. The lean profile
splits the prompt into

  ActorContextSnapshot — identity, role, authority, organization, persistent goals, stable
      relationships, stable institution rules, cohort private-state template, information
      boundary, public calendar facts: compiled ONCE per (actor, cohort), rendered as one
      BYTE-STABLE block with a content hash. Byte stability makes the block a deterministic
      provider prompt-prefix: backends with automatic prefix caching (DeepSeek exposes
      prompt_cache_hit_tokens) reuse it across calls; where no provider cache exists the compact
      delta still shrinks the prompt.

  ActorDecisionDelta — only what changed for THIS decision: new observations, working-memory and
      retrieved-memory content, changed beliefs/constraints/resources, the trigger, deadline
      state, the current action menu, the prior decision + its invalidation reason.

LOSSLESSNESS: compression must never hide decision-relevant information. `effective_actor_view`
reconstructs the full effective view from (snapshot + accumulated branch state + delta); the debug
check (tests) proves it carries exactly the actor's information boundary — nothing added, nothing
decision-relevant removed."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

PROMPT_VERSION = "lean.actor.prompt.v1"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class ActorContextSnapshot:
    """The stable prefix for one (actor, cohort). Rendering is deterministic: sorted collections,
    day-granularity dates only, no branch/particle identity anywhere."""
    actor_id: str
    cohort_id: str
    rendered: str
    content_hash: str
    n_chars: int

    @classmethod
    def build(cls, *, view, state, public_facts_lines: list, structural_frame: str = ""
              ) -> "ActorContextSnapshot":
        rows = [f"YOU ARE: {view.actor_id}"
                + (f", {view.actor_role}" if view.actor_role and view.actor_role != "unknown"
                   else "")]
        if state is not None:
            # the COMPLETE cohort-baseline private state (every section the staged decision
            # prompt renders) — the delta then carries only branch drift from this baseline, so
            # snapshot + delta always covers the full information boundary (losslessness)
            rows.append(f"HYPOTHESIS OF YOUR HIDDEN REALITY: {state.hypothesis_id}")
            if state.identity_and_role:
                rows.append(f"IDENTITY AND ROLE: {state.identity_and_role}")
            if state.core_worldview:
                rows.append(f"CORE WORLDVIEW: {state.core_worldview}")
            if state.current_goals:
                rows.append("PERSISTENT GOALS: " + "; ".join(state.current_goals[:6]))
            if state.fears_and_failure_conditions:
                rows.append("FEARS: " + "; ".join(state.fears_and_failure_conditions[:5]))
            if state.current_private_beliefs:
                rows.append("CURRENT PRIVATE BELIEFS (at baseline):\n" + "\n".join(
                    f"  - {b}" for b in state.current_private_beliefs[:8]))
            if state.beliefs_about_others:
                rows.append("BELIEFS ABOUT OTHERS (at baseline):\n" + "\n".join(
                    f"  - {k}: {v}" for k, v in
                    sorted(state.beliefs_about_others.items())[:8]))
            if state.personal_condition:
                rows.append(f"PERSONAL CONDITION (at baseline): {state.personal_condition}")
            if state.organizational_pressures:
                rows.append("ORGANIZATIONAL PRESSURES (at baseline): "
                            + state.organizational_pressures)
            if state.commitments_and_identity_constraints:
                rows.append("COMMITMENTS AND IDENTITY CONSTRAINTS: "
                            + "; ".join(state.commitments_and_identity_constraints[:6]))
            if state.relationships:
                rows.append("STABLE RELATIONSHIPS:\n" + "\n".join(
                    f"  - {k}: {v}" for k, v in sorted(state.relationships.items())[:8]))
            if state.important_memories:
                rows.append("IMPORTANT MEMORIES (at baseline):\n" + "\n".join(
                    f"  - {m.get('memory')}" for m in state.important_memories[-8:]
                    if isinstance(m, dict)))
            if state.unresolved_uncertainties:
                rows.append("UNRESOLVED UNCERTAINTIES (at baseline): "
                            + "; ".join(state.unresolved_uncertainties[:6]))
            if state.assumptions:
                rows.append("(Labeled assumptions behind this hypothesis: "
                            + "; ".join(state.assumptions[:4]) + ")")
        if view.authority:
            rows.append("YOUR FORMAL AUTHORITY: " + ", ".join(sorted(map(str, view.authority))))
        rules = sorted(
            f"- rule {r.get('institution_id')}:{r.get('kind')} "
            f"{json.dumps(r.get('params', {}), sort_keys=True, default=str)[:100]}"
            for r in (view.institution_rules or [])[:8] if isinstance(r, dict))
        if rules:
            rows.append("STANDING INSTITUTIONAL RULES:\n" + "\n".join(rules))
        if public_facts_lines:
            rows.append("PUBLIC CALENDAR AND SCHEDULED FACTS (known to everyone in your world):\n"
                        + "\n".join(public_facts_lines))
        if structural_frame:
            rows.append("(Structural frame under evaluation — a conjecture, never evidence: "
                        + str(structural_frame)[:300] + ")")
        rows.append("INFORMATION BOUNDARY: you know only your own private state above, public "
                    "facts, and the observations delivered to you below — never other people's "
                    "private states or undelivered events.")
        rendered = "\n".join(rows)
        return cls(actor_id=view.actor_id,
                   cohort_id=str(getattr(state, "hypothesis_id", "") or ""),
                   rendered=rendered, content_hash=_h(rendered), n_chars=len(rendered))


@dataclass
class ActorDecisionDelta:
    """Everything decision-specific, rendered deterministically. Unchanged stable context is
    NEVER re-rendered here (the losslessness check proves the union still covers the boundary)."""
    rendered: str
    n_chars: int

    @classmethod
    def build(cls, *, day: str, situation: str, observations: list, working_memory: list,
              retrieved: list, changed_state_rows: list, resources: list, action_history: list,
              menu_lines: list, prior_decision_note: str = "", obstacle: str = "") -> \
            "ActorDecisionDelta":
        rows = [f"TODAY IS {day}."]
        if changed_state_rows:
            rows.append("WHAT HAS CHANGED IN YOUR PRIVATE STATE SINCE THE STABLE BRIEF:\n"
                        + "\n".join(f"  - {r}" for r in changed_state_rows[:10]))
        rows.append("OBSERVATIONS DELIVERED TO YOU NOW (refer to them by obs_id):\n" + ("\n".join(
            f"- obs_id={o.get('obs_id', '?')} [{o.get('channel', '?')}|{o.get('source', '?')}] "
            f"{o.get('content', o.get('summary', ''))}"
            for o in observations) or "- (nothing new delivered)"))
        if working_memory:
            rows.append("CURRENTLY ACTIVE IN YOUR MIND:\n" + "\n".join(
                f"- ({i.get('kind', 'item')}) {i.get('content', '')}" for i in working_memory))
        if retrieved:
            rows.append("MEMORIES THAT COME BACK TO YOU:\n" + "\n".join(
                f"- {m.get('content', '')}" for m in retrieved))
        rows.append("RESOURCES YOU HOLD: " + (", ".join(resources) or "unknown"))
        rows.append("YOUR RECENT ACTIONS: " + (", ".join(action_history) or "(none yet)"))
        if prior_decision_note:
            rows.append(f"YOUR STANDING PRIOR DECISION: {prior_decision_note}")
        rows.append(f"THE SITUATION REQUIRING A DECISION:\n{situation}")
        if obstacle:
            rows.append("YOU JUST TRIED TO ACT AND HIT AN OBSTACLE (as you perceive it): "
                        + obstacle + "\nRevise your decision accordingly.")
        rows.append("YOUR OPTIONS (you may also propose a genuinely different act):\n"
                    + "\n".join(menu_lines))
        rendered = "\n\n".join(rows)
        return cls(rendered=rendered, n_chars=len(rendered))


@dataclass
class ActorPromptManifest:
    """Per-run accounting: stable-prefix reuse, delta sizes, and what a full re-render would have
    cost — the §23 'prompts became materially smaller' evidence, measured not projected."""
    snapshots: dict = field(default_factory=dict)   # (actor,cohort) -> {hash, chars, built}
    calls: int = 0
    prefix_reuses: int = 0
    delta_chars: int = 0
    prefix_chars_sent: int = 0
    full_equivalent_chars: int = 0                  # what resending everything would have cost

    def record_snapshot(self, snap: ActorContextSnapshot, *, reused: bool):
        key = f"{snap.actor_id}|{snap.cohort_id}"
        self.snapshots.setdefault(key, {"hash": snap.content_hash, "chars": snap.n_chars,
                                        "built": 0, "reused": 0})
        self.snapshots[key]["reused" if reused else "built"] += 1
        if reused:
            self.prefix_reuses += 1

    def record_call(self, *, prefix_chars: int, delta_chars: int, full_equivalent_chars: int):
        self.calls += 1
        self.prefix_chars_sent += prefix_chars
        self.delta_chars += delta_chars
        self.full_equivalent_chars += full_equivalent_chars

    def as_dict(self) -> dict:
        sent = self.prefix_chars_sent + self.delta_chars
        return {"version": PROMPT_VERSION, "calls": self.calls,
                "prefix_reuses": self.prefix_reuses,
                "distinct_snapshots": len(self.snapshots),
                "prompt_chars_sent": sent, "delta_chars": self.delta_chars,
                "full_equivalent_chars": self.full_equivalent_chars,
                "chars_saved_vs_full_rerender": max(0, self.full_equivalent_chars - sent),
                "snapshots": dict(sorted(self.snapshots.items())[:40])}


def effective_actor_view(snapshot: ActorContextSnapshot, delta: ActorDecisionDelta) -> str:
    """DEBUG/LOSSLESSNESS: the complete effective actor view a lean call presents = stable prefix
    + per-decision delta, in exactly the order the model reads them. Tests reconstruct this and
    verify it covers the actor's intended information boundary (state sections, delivered
    observations, rules, authority, menu) with nothing decision-relevant missing."""
    return f"{snapshot.rendered}\n\n{delta.rendered}"
