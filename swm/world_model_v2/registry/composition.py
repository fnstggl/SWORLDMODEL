"""Composition engine — Phase 6, Part 7.

Multiple mechanisms run in ONE world. This module takes the per-process selections the compiler made
(each a `select_for_process` result) and produces a composition plan that:
  * detects DOUBLE-COUNTING — two mechanisms that would write the same effect (e.g. two exposure→adoption
    hazards, two persuasion effects, two social-influence updates on the same edge);
  * preserves COMPETING mechanisms — when several families plausibly answer one process and the evidence
    does not distinguish them, they are kept as competing hypotheses (evidence-weighted), NOT averaged;
  * assigns PRECEDENCE — validated (production/local) mechanisms outrank domain-restricted, which outrank
    research-encoded, which outrank the generic tier-6/7 fallback;
  * flags CONFLICTS — incompatible time scales / state assumptions on the same target.

It does NOT silently average incompatible mechanisms. Competing hypotheses are surfaced so the rollout can
branch particles across them and propagate mechanism DISAGREEMENT into the terminal distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: canonical "effect channel" each causal process writes — processes sharing a channel risk double-counting
EFFECT_CHANNEL = {
    "adoption_after_repeated_exposure": "adoption",
    "cascade_saturation": "adoption",
    "social_reinforcement": "adoption",
    "tipping": "adoption",
    "diffusion_timing": "adoption",
    "participation_after_mobilization": "participation",
    "turnout": "participation",
    "donation_after_ask": "participation",
    "belief_update_after_message": "belief",
    "opinion_convergence": "belief",
    "persuasion_success": "belief",
    "trust_change_after_interaction": "relationship",
    "reciprocity": "relationship",
    "reputation_update": "relationship",
    "offer_response": "decision",
    "bargaining_split": "decision",
    "actor_selects_typed_action": "decision",
    "content_response": "attention",
    "attention_after_exposure": "attention",
    "examination_by_rank": "attention",
}

STATUS_PRECEDENCE = {"production_eligible": 5, "transfer_validated": 4, "locally_validated": 3,
                     "domain_restricted": 2, "research_encoded": 1}


@dataclass
class CompositionPlan:
    ordered: list = field(default_factory=list)          # [{process, family_id, status, channel, tier}]
    competing: list = field(default_factory=list)        # [{process, families:[...], why}]
    double_counting: list = field(default_factory=list)  # [{channel, processes:[...], resolution}]
    conflicts: list = field(default_factory=list)        # [{reason, families}]

    def as_dict(self):
        return {"ordered": self.ordered, "competing": self.competing,
                "double_counting": self.double_counting, "conflicts": self.conflicts}


def compose(selections: list, *, time_scales: dict | None = None) -> CompositionPlan:
    """selections: list of select_for_process() results (one per required causal process). Returns a
    CompositionPlan. time_scales: optional {family_id: scale} to detect cross-timescale conflicts."""
    plan = CompositionPlan()
    channel_writers: dict = {}                            # channel -> [(process, family_id, status)]
    for sel in selections:
        proc = sel.get("process", "")
        chosen = sel.get("selected")
        if not chosen:
            continue
        fam = chosen["family_id"]
        status = chosen.get("status", "")
        channel = EFFECT_CHANNEL.get(proc, proc)
        tier = 6 - STATUS_PRECEDENCE.get(status, 0)      # 1 (production) .. 6 (none)
        plan.ordered.append({"process": proc, "family_id": fam, "status": status,
                             "channel": channel, "tier": max(1, tier)})
        channel_writers.setdefault(channel, []).append((proc, fam, status))
        # competing hypotheses for THIS process (kept, not averaged)
        comp = [c["family_id"] for c in sel.get("competing", [])]
        if comp:
            plan.competing.append({"process": proc, "selected": fam, "competing": comp,
                                   "why": "multiple families answer this process; evidence does not fully "
                                          "distinguish — kept as competing hypotheses (branch, don't average)"})

    # double-counting: >1 DISTINCT family writing the same effect channel via DIFFERENT processes
    for channel, writers in channel_writers.items():
        fams = {f for _, f, _ in writers}
        procs = sorted({p for p, _, _ in writers})
        if len(fams) > 1 and len(procs) > 1:
            # keep the highest-precedence writer; the others are flagged as double-count risks
            ranked = sorted(writers, key=lambda w: -STATUS_PRECEDENCE.get(w[2], 0))
            keep = ranked[0][1]
            plan.double_counting.append({
                "channel": channel, "processes": procs, "families": sorted(fams),
                "resolution": f"single-write channel: {keep} (highest precedence) writes {channel!r}; "
                              f"the others contribute as competing hypotheses, NOT additive effects — "
                              f"prevents double-counted {channel}"})

    # cross-timescale conflict: mechanisms on the same channel with incompatible declared time scales
    if time_scales:
        for channel, writers in channel_writers.items():
            scales = {time_scales.get(f) for _, f, _ in writers if time_scales.get(f)}
            if len(scales) > 1:
                plan.conflicts.append({"reason": f"channel {channel!r} written by mechanisms at "
                                       f"incompatible time scales {sorted(s for s in scales if s)}",
                                       "families": sorted({f for _, f, _ in writers})})
    return plan
