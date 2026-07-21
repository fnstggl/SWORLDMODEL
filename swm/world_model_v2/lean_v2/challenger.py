"""The genuinely conditional structural challenger + the no-pointless-replicates policy.

ORDER: primary blueprint → answerability → weighted primary run (the pilot IS the run —
waves are cheap) → inspect evidence conflict and terminal sensitivity → challenger ONLY if
materially necessary. "An LLM can imagine a different story" is NOT a trigger; every trigger
below is deterministic over the primary result + the blueprint's own evidence-verified
conflict markers:

    * primary probability materially unstable (weight-sensitive across the grounded ranges
      AND near the decision threshold);
    * near-threshold with an evidence-supported concrete alternative reading (verbatim quote
      verified against the evidence text);
    * one disputed assumption capable of reversing the answer (reversal_capable=true AND a
      verified evidence_conflict quote);
    * two materially conflicting causal interpretations in the evidence;
    * cheap sensitivity: a reversal-capable omitted component while the result sits near the
      threshold.

LOCALIZED execution: when the challenger differs in ONE causal assumption, the fork reuses the
primary's validated compilation and shared history — the challenger engine SHARES the primary's
decision-equivalence cache and compilation cache, so every unchanged decision context is a
cache hit (zero calls) and only the genuinely divergent downstream pathway spends computation.
A fully separate challenger world runs only when the structural difference invalidates primary
history from the beginning (roster/terminal-rule change). The challenger path is never removed:
if it is necessary, it runs, and the run reports that the question required deeper computation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import ConsumerWorldBlueprint, norm

#: |p - 0.5| below this is "near the decision threshold"
NEAR_THRESHOLD = 0.10
#: primary/challenger spread that marks material structural disagreement
DISAGREEMENT_SPREAD = 0.10


@dataclass
class ChallengerDecision:
    triggered: bool = False
    triggers: list = field(default_factory=list)
    mode: str = ""                       # localized_fork | full_world | (empty when skipped)
    divergence: str = ""
    skipped_reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"triggered": self.triggered, "triggers": self.triggers, "mode": self.mode,
                "divergence": self.divergence, "skipped_reasons": self.skipped_reasons}


def _quote_in_evidence(quote: str, evidence_text: str) -> bool:
    q = norm(quote, 300).lower()
    return bool(q) and q[:100] in norm(evidence_text, 200000).lower()


def decide_challenger(bp: ConsumerWorldBlueprint, *, p_mid, weight_sensitive: bool,
                      unresolved_share: float, evidence_text: str) -> ChallengerDecision:
    d = ChallengerDecision()
    near = p_mid is not None and abs(p_mid - 0.5) < NEAR_THRESHOLD
    alt = bp.alternative_causal_reading or {}
    alt_supported = bool(alt.get("exists")) and _quote_in_evidence(
        alt.get("evidence_quote", ""), evidence_text)

    if weight_sensitive and near:
        d.triggers.append("primary_probability_materially_unstable_near_threshold")
    if near and alt_supported:
        d.triggers.append("near_threshold_with_evidence_supported_alternative")
    disputed = [u for u in bp.unresolved_assumptions
                if u.get("reversal_capable")
                and _quote_in_evidence(u.get("evidence_conflict", ""), evidence_text)]
    if disputed:
        d.triggers.append(f"disputed_reversal_assumption:{norm(disputed[0].get('assumption'), 80)}")
    if alt_supported and unresolved_share > 0.5:
        d.triggers.append("conflicting_causal_interpretations_with_majority_unresolved")
    omissions = (bp.world_boundary or {}).get("reversal_capable_omissions") or []
    if omissions and p_mid is not None and abs(p_mid - 0.5) < 0.15:
        d.triggers.append(f"reversal_capable_omission_near_threshold:"
                          f"{norm((omissions[0] or {}).get('component'), 60)}")

    if not d.triggers:
        d.skipped_reasons.append(
            "no deterministic trigger fired: primary stable, no verified evidence conflict, "
            "no reversal-capable disputed assumption (imagination alone is not a trigger)")
        return d
    d.triggered = True
    d.divergence = norm(alt.get("diverges_at") or
                        (disputed[0].get("assumption") if disputed else "") or
                        "alternative_reading", 120)
    structural_from_start = str(alt.get("diverges_at") or "").lower() == "structural"
    d.mode = "full_world" if structural_from_start else "localized_fork"
    return d


def build_challenger_blueprint(bp: ConsumerWorldBlueprint, decision: ChallengerDecision,
                               *, gateway, cache) -> ConsumerWorldBlueprint | None:
    """ONE structural call producing a LOCALIZED DELTA (changed actor variants / assumption),
    applied to a deep copy of the primary blueprint — every unchanged component is reused by
    construction (shared compilation + shared decision cache do the rest)."""
    from swm.engine.grounding import parse_json
    alt = bp.alternative_causal_reading or {}
    prompt = (
        "A primary causal world model may be wrong in ONE localized way. Produce the minimal "
        "delta for the challenger world.\n"
        f"Primary causal thesis: {bp.causal_thesis}\n"
        f"Challenged at: {decision.divergence}\n"
        f"Alternative reading: {norm(alt.get('reading'), 400) or '(from the disputed assumption)'}\n"
        "Actors (id: variants): "
        + "; ".join(f"{a.get('id')}: "
                    + ",".join(str(v.get('variant_id'))
                               for v in a.get('private_state_variants') or [])
                    for a in bp.actors) + "\n"
        'Reply ONLY JSON: {"challenger_thesis": "...", '
        '"changed_actor_variants": {"<actor_id>": [{"variant_id": "...", '
        '"state": {"beliefs": [], "goals": [], "stances": [], "pressures": ""}, '
        '"evidence_basis": "<quote or unstated>", '
        '"support": "well_supported|plausible|speculative"}]}, '
        '"changed_assumption": "...", "unchanged_note": "everything else identical"}')
    deps = {"blueprint": bp.raw_response_hash, "divergence": decision.divergence,
            "backend": gateway.backend_fingerprint}
    cached = cache.get("blueprint_response", deps)
    text = cached if cached is not None else gateway.call("structural_generation", prompt)
    r = parse_json(text)
    if not isinstance(r, dict):
        return None                                 # failure never cached
    if cached is None:
        cache.put("blueprint_response", deps, text)
    ch = copy.deepcopy(bp)
    ch.causal_thesis = norm(r.get("challenger_thesis"), 600) or ("CHALLENGER: "
                                                                + bp.causal_thesis)
    changed = r.get("changed_actor_variants") or {}
    n_changed = 0
    for aid, variants in changed.items():
        a = ch.actor_by_id(str(aid))
        if a is None or not isinstance(variants, list) or not variants:
            continue
        vv = []
        for v in variants[:3]:
            if isinstance(v, dict) and v.get("variant_id"):
                if str(v.get("support")) not in ("well_supported", "plausible", "speculative"):
                    v["support"] = "speculative"
                vv.append(v)
        if vv:
            a["private_state_variants"] = vv
            n_changed += 1
    if n_changed == 0:
        return None
    ch.raw_response_hash = bp.raw_response_hash + "+challenger"
    ch.validation = {**bp.validation, "challenger_delta": {
        "changed_actors": sorted(str(k) for k in changed),
        "changed_assumption": norm(r.get("changed_assumption"), 200),
        "divergence": decision.divergence}}
    return ch


# ------------------------------------------------------------------ replicate policy (§10)
SCOREABLE = ("completed", "completed_with_degradation", "partially_resolved")


def should_replicate(*, status: str, p_mid, unresolved_share: float,
                     requested_behavioral_replicates: int, terminal_mechanism_failed: bool
                     ) -> tuple:
    """(allowed, reason). A second run must test a REAL uncertainty, never repeat a failure:
    under-modeled / unresolved / failed-terminal results NEVER auto-replicate (the replicate
    would re-execute the same missing mechanism); replicates run only when the primary is
    scoreable, execution variation could materially move the number, and the caller actually
    requested behavioral replicates. The full world is never relaunched automatically."""
    if terminal_mechanism_failed:
        return False, "terminal mechanism failed — a replicate repeats the same failure"
    if status not in SCOREABLE:
        return False, f"status '{status}' is not scoreable — replicate would repeat the same " \
                      f"missing mechanism, not test execution uncertainty"
    if p_mid is None:
        return False, "no probability — nothing a replicate could stabilize"
    if unresolved_share >= 0.999:
        return False, "all mass unresolved — no new decision draw can make this scoreable"
    if requested_behavioral_replicates <= 1:
        return False, "behavioral replicates not requested (deterministic execution: " \
                      "content-addressed draws make replicate 0 exhaustive)"
    return True, "scoreable + requested: replicate tests genuine execution variation"
