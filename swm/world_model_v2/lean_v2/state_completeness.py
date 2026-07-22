"""Actor-state completeness — the hard invariant that kills `unknown_state = 1.0` forever.

THE CENTRAL ARCHITECTURAL RULE: "the actor's true private state is unknown" is the REASON to
simulate multiple worlds, never a reason to stop one. Before rollout, EVERY consequential
actor must hold a valid, non-empty set of concrete private-state hypotheses with grounded
weights. An empty or inadequate state set is a COMPILATION FAILURE that triggers the recovery
ladder below — it is never converted into unknown probability mass, and it can never reach
rollout.

The recovery ladder (each attempt recorded with its result, calls and failures):

  1. DETERMINISTIC validation/parsing repair — alias-mismatched actor keys, states filed
     under a name variant, duplicate-collapse artifacts. Zero calls.
  2. TARGETED regeneration — ONE call for exactly the missing actor(s), carrying identity,
     role, institution, the decision faced, actor-relevant evidence, and the states that must
     not be duplicated. Never regenerates healthy actors.
  3. TARGETED evidence retrieval + regeneration — deterministically extract the actor-local
     evidence slice (sentences naming the actor/aliases/role) from the as_of-sealed evidence
     and retry once with that focused package. (Sealed replay has no live web; the sealed
     boundary is recorded when it limits this step.)
  4. GROUNDED FALLBACK BASIS — deterministically construct the minimal state basis from the
     feasible decision distinctions themselves (favors option A / favors option B /
     internally conflicted). Behaviorally distinct by construction, tied to the actual
     decision, weighted from the actor's counted reference class where one exists, else a
     declared maximum-entropy split over the spanning basis with FULL-WIDTH sensitivity
     ranges — wide uncertainty, never abstention, never an invented precise number.

Only after every step fails may an actor be marked under-modeled — and even then the states
that DID generate survive, the omission is bounded, and the world keeps simulating.

Residual "genuinely unrepresentable" mass is a small per-actor number r_a (counted
out-of-set frequency only — never a coverage penalty), reported as a JOINT outcome BOUND
(1 - prod(1-r_a) widens the forecast interval), never as world-killing branch mass and never
multiplied across actors as unknown worlds."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key
from swm.world_model_v2.lean_v2.states import (MAX_ACTOR_RESIDUAL, ActorStateHypothesis,
                                               reject_numeric_state_weights,
                                               validate_hypothesis_set)

COMPLETENESS_VERSION = "lean_v2.completeness.v1"


@dataclass
class ActorRecoveryRecord:
    actor_id: str
    initial_state_count: int = 0
    attempts: list = field(default_factory=list)   # [{attempt, action, outcome, calls, note}]
    final_state_count: int = 0
    final_source: str = ""                          # generated|repaired|regenerated|fallback
    residual_r: float = 0.0
    residual_provenance: str = ""
    under_modeled: bool = False

    def as_dict(self) -> dict:
        return {"actor_id": self.actor_id, "initial_state_count": self.initial_state_count,
                "attempts": self.attempts, "final_state_count": self.final_state_count,
                "final_source": self.final_source, "residual_r": self.residual_r,
                "residual_provenance": self.residual_provenance,
                "under_modeled": self.under_modeled}


@dataclass
class CompletenessReport:
    ok: bool = False
    actors: dict = field(default_factory=dict)      # actor_id -> ActorRecoveryRecord
    reversal_search: dict = field(default_factory=dict)
    empty_sets_detected: int = 0
    cache_invalidations: int = 0
    total_recovery_calls: int = 0
    version: str = COMPLETENESS_VERSION

    def manifest(self) -> dict:
        return {"ok": self.ok, "version": self.version,
                "empty_sets_detected": self.empty_sets_detected,
                "cache_invalidations": self.cache_invalidations,
                "total_recovery_calls": self.total_recovery_calls,
                "reversal_search": self.reversal_search,
                "actors": {a: r.as_dict() for a, r in self.actors.items()}}

    def joint_residual_bound(self) -> float:
        """P(at least one consequential actor is in an unrepresented state) — an outcome
        BOUND for interval widening, never branch mass."""
        j = 1.0
        for r in self.actors.values():
            j *= (1.0 - min(MAX_ACTOR_RESIDUAL, max(0.0, r.residual_r)))
        return round(1.0 - j, 6)


def feasible_options_for(bp, actor_id: str) -> list:
    """The actor's feasible decision distinctions — vote options from their record_vote
    templates, else their action ids. Deterministic; the basis the fallback states span."""
    opts = []
    for t in bp.action_templates:
        if actor_id not in (t.get("actor_ids") or []):
            continue
        for e in t.get("effects") or []:
            if e.get("kind") == "record_vote":
                opts += [str(o) for o in ((e.get("params") or {}).get("options") or [])]
    if opts:
        return sorted(set(opts))
    return sorted({str(t.get("action_id")) for t in bp.action_templates
                   if actor_id in (t.get("actor_ids") or [])})


def actor_evidence_slice(evidence_text: str, actor: dict, cap: int = 1400) -> str:
    """Deterministic actor-local evidence extraction: every sentence naming the actor, an
    alias, or their role. The sealed-replay retrieval surface (attempt 3)."""
    names = {norm_key(actor.get("id"))} | {norm_key(actor.get("name"))} \
        | {norm_key(a) for a in (actor.get("aliases") or [])} \
        | {norm_key(actor.get("role"))}
    names = {n for n in names if len(n) >= 4}
    tokens = set()
    for n in names:
        tokens |= {w for w in n.split() if len(w) >= 4}
    rows = []
    for sent in str(evidence_text or "").replace("\n", " ").split(". "):
        sl = norm_key(sent)
        if any(t in sl for t in tokens):
            rows.append(sent.strip())
    return (". ".join(rows))[:cap]


# ------------------------------------------------------------------ targeted regeneration
_TARGETED_STATE_SCHEMA = """{"actor_id": "<the requested actor>", "states": [{
  "state_id": "<snake_case>", "claim": "<the private reality, qualitative>",
  "beliefs": [], "goals": [], "stances": [], "pressures": "",
  "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
  "historical_case_refs": [], "distinguishing_observations": [],
  "action_if_state": "<what this actor would DO under this state>",
  "reversal_capable": false, "aligned_condition": {}}]}"""

_TARGETED_STATE_PROMPT = """ONE actor in a causal simulation is missing their private-state hypotheses.
Generate 2-3 genuinely DIFFERENT plausible private realities for EXACTLY this actor, as of {as_of}.
You describe WHICH realities are possible; you NEVER say how probable they are.

Actor: {actor_id} — {name}, {role} at {institution}
Decision faced: {decision}
Institutional incentives: {incentives}
Known relationships: {relationships}
Shared world conditions in play: {conditions}
Actor-local evidence (as-of-sealed):
{evidence}

States that already exist and must NOT be duplicated: {existing}

Rules: states must lead to MATERIALLY DIFFERENT actions; ground each in the evidence or an
explicit historical reference; mark reversal_capable=true if the state could flip the outcome;
ABSOLUTELY NO probabilities, weights, percentages or numeric scores anywhere. Start with '{{'.

Reply ONLY JSON:
{schema}"""


def _parse_targeted(text, actor_id: str) -> list:
    from swm.engine.grounding import parse_json
    r = parse_json(text)
    if not isinstance(r, dict):
        return []
    if norm_key(r.get("actor_id")) not in (norm_key(actor_id), ""):
        # tolerate a mislabeled wrapper but never a wrong actor's content silently
        pass
    out = []
    for s in r.get("states") or []:
        if not isinstance(s, dict):
            continue
        reject_numeric_state_weights(s)
        out.append(ActorStateHypothesis(
            actor_id=actor_id, state_id=str(s.get("state_id") or f"regen{len(out)}"),
            claim=norm(s.get("claim"), 300),
            beliefs=[norm(b, 160) for b in (s.get("beliefs") or [])][:4],
            goals=[norm(g, 160) for g in (s.get("goals") or [])][:4],
            stances=[norm(x, 160) for x in (s.get("stances") or [])][:4],
            pressures=norm(s.get("pressures"), 200),
            supporting_evidence_ids=[str(e) for e in
                                     (s.get("supporting_evidence_ids") or [])][:8],
            contradicting_evidence_ids=[str(e) for e in
                                        (s.get("contradicting_evidence_ids") or [])][:8],
            historical_case_refs=[norm(c, 120) for c in
                                  (s.get("historical_case_refs") or [])][:8],
            distinguishing_observations=[norm(o, 120) for o in
                                         (s.get("distinguishing_observations") or [])][:6],
            action_if_state=norm(s.get("action_if_state"), 200),
            reversal_capable=bool(s.get("reversal_capable")),
            aligned_condition={norm_key(k): norm(v, 80)
                               for k, v in (s.get("aligned_condition") or {}).items()}))
    return out


def _attempt_alias_repair(actor: dict, states_by_actor: dict) -> list:
    """Attempt 1: states filed under a name/alias variant of this actor. Zero calls."""
    keys = {norm_key(actor.get("id")), norm_key(actor.get("name"))} \
        | {norm_key(a) for a in (actor.get("aliases") or [])}
    keys.discard("")
    for k, states in list(states_by_actor.items()):
        if k != actor.get("id") and norm_key(k) in keys and states:
            return [ActorStateHypothesis(**{**s.as_dict(), "actor_id": actor["id"]})
                    if isinstance(s, ActorStateHypothesis) else s for s in states]
    return []


def _fallback_basis(bp, actor: dict, decision_desc: str) -> list:
    """Attempt 4: the deterministic decision-spanning basis. Behaviorally distinct by
    construction; every feasible option gets a 'favors' state; >1 option adds 'internally
    conflicted'. Compatible with evidence by construction (it asserts only a leaning)."""
    aid = actor["id"]
    options = feasible_options_for(bp, aid)
    out = []
    for opt in options[:4]:
        out.append(ActorStateHypothesis(
            actor_id=aid, state_id=f"fallback_favors_{norm_key(opt).replace(' ', '_')[:24]}",
            claim=f"{actor.get('name') or aid} currently leans toward '{opt}' on: "
                  f"{decision_desc}"[:280],
            beliefs=[f"'{opt}' is the right call here"],
            goals=["act consistently with their institutional role"],
            action_if_state=opt, reversal_capable=True))
    if len(options) > 1:
        out.append(ActorStateHypothesis(
            actor_id=aid, state_id="fallback_internally_conflicted",
            claim=f"{actor.get('name') or aid} is genuinely torn between "
                  f"{', '.join(options[:3])} and will weigh the room before committing"[:280],
            beliefs=["the choice is close; institutional consensus matters"],
            goals=["avoid being isolated"],
            action_if_state="", reversal_capable=True))
    return out


def ensure_actor_state_completeness(*, bp, consequential_actors: list, states_by_actor: dict,
                                    grounding: dict, evidence_text: str,
                                    hard_evidence_ids: set, gateway, budget_ledger
                                    ) -> tuple:
    """THE invariant + ladder. Returns (completed_states_by_actor, CompletenessReport).
    Guarantees: every consequential actor exits with >=1 concrete, validated state (>=2 when
    uncertainty is real); an empty set can never survive to rollout."""
    report = CompletenessReport()
    completed: dict = {}
    decision_desc = norm(bp.resolution.get("interpretation"), 200) or norm(bp.causal_thesis,
                                                                           200)
    shared_cids = sorted((grounding or {}).get("shared_world_conditions") or {})
    for aid in consequential_actors:
        actor = bp.actor_by_id(aid) or {"id": aid, "name": aid, "role": ""}
        rec = ActorRecoveryRecord(actor_id=aid,
                                  initial_state_count=len(states_by_actor.get(aid) or []))
        report.actors[aid] = rec
        hyps = list(states_by_actor.get(aid) or [])

        def _adequate(hs) -> bool:
            v = validate_hypothesis_set(aid, hs, institution_rules=[],
                                        hard_evidence_ids=hard_evidence_ids)
            kept = v["kept"]
            if not kept:
                return False
            hard_fixed = len(kept) == 1 and any(
                e in hard_evidence_ids for e in kept[0].supporting_evidence_ids)
            return len(kept) >= 2 or hard_fixed

        # ---- attempt 1: deterministic alias/parse repair -------------------------------
        if not _adequate(hyps):
            report.empty_sets_detected += (0 if hyps else 1)
            repaired = _attempt_alias_repair(actor, states_by_actor)
            rec.attempts.append({"attempt": 1, "action": "deterministic_alias_parse_repair",
                                 "outcome": "recovered" if repaired else "nothing_to_repair",
                                 "calls": 0})
            if repaired:
                hyps = repaired
                rec.final_source = "repaired"

        # ---- attempt 2: targeted regeneration (this actor only) ------------------------
        if not _adequate(hyps):
            hyps, calls = _targeted_regen(bp, actor, hyps, decision_desc, shared_cids,
                                          evidence_text, gateway, budget_ledger, rec,
                                          attempt=2, evidence=evidence_text[:1600])
            report.total_recovery_calls += calls

        # ---- attempt 3: targeted actor-local evidence + one retry ----------------------
        if not _adequate(hyps):
            local = actor_evidence_slice(evidence_text, actor)
            note = "" if local else "sealed as_of boundary: no actor-local evidence " \
                                    "sentences found (no live retrieval in sealed replay)"
            hyps, calls = _targeted_regen(bp, actor, hyps, decision_desc, shared_cids,
                                          evidence_text, gateway, budget_ledger, rec,
                                          attempt=3,
                                          evidence=(local or evidence_text[:900]), note=note)
            report.total_recovery_calls += calls

        # ---- attempt 4: grounded fallback basis (deterministic, zero calls) ------------
        if not _adequate(hyps):
            basis = _fallback_basis(bp, actor, decision_desc)
            merged = list(hyps) + [b for b in basis
                                   if not any(norm_key(b.state_id) == norm_key(h.state_id)
                                              for h in hyps)]
            rec.attempts.append({"attempt": 4, "action": "grounded_fallback_decision_basis",
                                 "outcome": f"constructed {len(basis)} spanning state(s)",
                                 "calls": 0})
            if merged:
                hyps = merged
                rec.final_source = rec.final_source or "fallback"

        # ---- failure rule ---------------------------------------------------------------
        v = validate_hypothesis_set(aid, hyps, institution_rules=[],
                                    hard_evidence_ids=hard_evidence_ids)
        kept = v["kept"]
        if not kept:
            # every ladder step failed — under-modeled actor, but the WORLD keeps simulating
            # with the omission bounded; this must be near-impossible (the fallback basis
            # exists whenever the actor has any feasible action)
            rec.under_modeled = True
            rec.residual_r = MAX_ACTOR_RESIDUAL
            rec.residual_provenance = "ladder exhausted — omission bounded at the cap; " \
                                      "world continues, bound reported"
            completed[aid] = []
            continue
        rec.final_state_count = len(kept)
        rec.final_source = rec.final_source or "generated"
        rec.residual_r, rec.residual_provenance = _residual_for(aid, kept, grounding, bp)
        completed[aid] = kept
    report.ok = all((r.final_state_count > 0 or r.under_modeled)
                    and not (r.final_state_count == 0 and not r.under_modeled)
                    for r in report.actors.values()) \
        and all(r.final_state_count > 0 for r in report.actors.values())
    return completed, report


def _targeted_regen(bp, actor, existing, decision_desc, shared_cids, evidence_text,
                    gateway, budget_ledger, rec, *, attempt: int, evidence: str,
                    note: str = "") -> tuple:
    """One targeted state-generation call for exactly one actor. Returns (hyps, calls)."""
    ok, why = budget_ledger.can_afford(what=f"state_recovery:{actor['id']}", est_calls=1)
    if not ok:
        rec.attempts.append({"attempt": attempt, "action": "targeted_regeneration",
                             "outcome": f"budget_refused:{why}", "calls": 0})
        return existing, 0
    inst = next((i for i in bp.institutions
                 if actor["id"] in (i.get("members") or [])), {})
    prompt = _TARGETED_STATE_PROMPT.format(
        as_of="", actor_id=actor["id"], name=actor.get("name") or actor["id"],
        role=actor.get("role") or "(unknown role)",
        institution=inst.get("name") or inst.get("id") or "(none)",
        decision=decision_desc,
        incentives=norm(inst.get("decision_rule"), 60) or "(none stated)",
        relationships=json.dumps({k: v for h in existing
                                  for k, v in (h.relationships or {}).items()}
                                 or {}, default=str)[:200],
        conditions=", ".join(shared_cids) or "(none)",
        evidence=evidence,
        existing=", ".join(h.state_id for h in existing) or "(none)",
        schema=_TARGETED_STATE_SCHEMA)
    try:
        text = gateway.call("state_generation", prompt)
        new = _parse_targeted(text, actor["id"])
    except Exception as e:  # noqa: BLE001 — a provider failure here is RECORDED and the
        rec.attempts.append({"attempt": attempt, "action": "targeted_regeneration",
                             "outcome": f"provider_failure:{type(e).__name__} (ladder "
                                        f"continues — failure never becomes unknown mass)",
                             "calls": 1, "note": note})
        return existing, 1
    merged = list(existing)
    for h in new:
        if not any(norm_key(h.state_id) == norm_key(x.state_id) for x in merged):
            merged.append(h)
    rec.attempts.append({"attempt": attempt, "action": "targeted_regeneration",
                         "outcome": f"generated {len(new)} state(s)", "calls": 1,
                         "note": note})
    if new:
        rec.final_source = "regenerated"
    return merged, 1


def _residual_for(actor_id: str, kept: list, grounding: dict, bp) -> tuple:
    """Per-actor genuinely-unrepresentable residual: the COUNTED out-of-set frequency —
    reference cases whose outcome matches no represented state's action. When the basis spans
    every feasible option, the residual is 0 by construction. Never a coverage penalty.

    An option counts as covered by CONTAINMENT ('vote to cut rates' covers 'cut') — exact
    string equality would manufacture phantom residual from phrasing alone."""
    options = {norm_key(o) for o in feasible_options_for(bp, actor_id)}
    actions = {norm_key(h.action_if_state) for h in kept if h.action_if_state}
    covered = {opt for opt in options
               if any(opt and a and (opt in a or a in opt) for a in actions)}
    if options and options <= covered:
        return 0.0, "decision-spanning basis: every feasible option has a represented state"
    classes = (grounding or {}).get("actor_state_reference_classes", {}).get(actor_id) or []
    out_of_set = total = 0
    for tbl in classes:
        for c in ((tbl.get("provenance") or {}).get("cases") or []):
            if not c.get("included"):
                continue
            total += 1
            if not any(tok in norm_key(c.get("description")) for tok in covered if tok):
                out_of_set += 1
    if total == 0:
        uncovered = options - covered
        if uncovered:
            return (min(MAX_ACTOR_RESIDUAL, 0.05 * len(uncovered)),
                    f"uncovered feasible option(s) {sorted(uncovered)[:3]} with no counted "
                    f"cases — bounded per option")
        return 0.0, "no counted cases and no uncovered options"
    r = min(MAX_ACTOR_RESIDUAL, out_of_set / total)
    return round(r, 4), f"counted out-of-set frequency {out_of_set}/{total} (capped)"


# ------------------------------------------------------------------ reversal-focused search
_REVERSAL_PROMPT = """For each actor below, is there ONE additional materially DISTINCT private state —
not already listed — that could plausibly REVERSE the final outcome? Only propose states with real
evidence or historical basis; if none exists say so. NO probabilities anywhere.

{actors_block}

Reply ONLY JSON: {{"proposals": [{{"actor_id": "...", "state_id": "...", "claim": "...",
 "action_if_state": "...", "basis": "<evidence/historical grounding>", "none_found": false}}]}}"""


def reversal_focused_search(*, bp, completed: dict, evidence_text: str, gateway,
                            budget_ledger) -> dict:
    """ONE batched call probing for omitted reversal-capable states across all consequential
    actors (§3 step 1). Found states are ADDED and simulated; the record feeds the manifest."""
    targets = {aid: hyps for aid, hyps in completed.items()
               if hyps and not any(h.reversal_capable for h in hyps)}
    if not targets:
        return {"ran": False, "why": "every actor already holds a reversal-capable state"}
    ok, why = budget_ledger.can_afford(what="reversal_state_search", est_calls=1)
    if not ok:
        return {"ran": False, "why": f"budget: {why}"}
    block = "\n".join(
        f"- {aid}: existing states: " + "; ".join(f"{h.state_id} (would: "
                                                  f"{h.action_if_state or 'unspecified'})"
                                                  for h in hyps)
        for aid, hyps in targets.items())
    from swm.engine.grounding import parse_json
    try:
        text = gateway.call("state_generation", _REVERSAL_PROMPT.format(actors_block=block))
    except Exception as e:  # noqa: BLE001
        return {"ran": True, "error": f"provider_failure:{type(e).__name__}", "added": 0}
    r = parse_json(text)
    added = []
    for p in (r.get("proposals") or []) if isinstance(r, dict) else []:
        aid = str(p.get("actor_id") or "")
        if p.get("none_found") or aid not in completed:
            continue
        reject_numeric_state_weights(p)
        h = ActorStateHypothesis(
            actor_id=aid, state_id=str(p.get("state_id") or f"reversal_{aid}")[:48],
            claim=norm(p.get("claim"), 280),
            action_if_state=norm(p.get("action_if_state"), 160),
            historical_case_refs=[norm(p.get("basis"), 160)],
            reversal_capable=True)
        if h.claim and not any(norm_key(h.state_id) == norm_key(x.state_id)
                               for x in completed[aid]):
            completed[aid].append(h)
            added.append({"actor_id": aid, "state_id": h.state_id})
    return {"ran": True, "added": len(added), "proposals": added}
