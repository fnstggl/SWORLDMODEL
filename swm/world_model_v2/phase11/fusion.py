"""Phase 11 — dependence-aware trigger fusion + false-positive control (spec §10).

Several detectors can see the SAME underlying event; several observations can share ONE source. Fusing them
as if independent would manufacture false confidence. So:

  * evidence is grouped by DEPENDENCE (shared source hashes / evidence ids / same trigger fingerprint); within
    a group we take the STRONGEST signal (a syndicated report is counted once, not N times — the same
    dependence-collapse discipline as Phase 3's ``_collapse_count_obs``);
  * across INDEPENDENT groups we combine with a noisy-OR (independent corroboration raises confidence,
    duplicates do not);
  * one ``impossible_event`` dominates many weak residuals (it floors the fused probability high);
  * contradictory detectors are retained, not averaged away;
  * cooldowns / hysteresis / persistence / a false-alarm budget prevent trigger storms and stop a single noisy
    observation (differing from the posterior mean) from causing a recompile.

Output classifies the situation as transient_anomaly / parameter_drift / local_structural / global_structural,
which the scope selector and decision policy consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_STRUCTURAL_FAMILIES = {"impossible_event", "new_actor", "new_institution", "rule_change", "authority_change",
                        "coalition_change", "network_restructuring", "outcome_space_change",
                        "mechanism_regime_change", "sustained_predictive_failure",
                        "mechanism_precondition_failure", "evidence_contradiction"}
_LOCAL_SCOPES = {"parameter_only", "observation_model", "latent_state", "actor", "relationship",
                 "local_network_region", "population_segment"}
_GLOBAL_SCOPES = {"outcome_contract", "full_plan", "action_space"}


def _dependence_key(ev) -> str:
    """Two pieces of evidence are DEPENDENT if they share any source hash or evidence id, else independent.
    We key a group by the sorted union of a representative source token so syndicated copies collapse."""
    prov = ev.provenance or {}
    srcs = tuple(sorted(prov.get("source_hashes", []) or prov.get("evidence_ids", []) or []))
    # fall back to the observation id set on supporting_observations (same obs → dependent)
    return srcs and "src:" + "|".join(srcs) or "obs:" + "|".join(sorted(ev.supporting_observations or []))


@dataclass
class FusedAssessment:
    fused_probability: float = 0.0
    by_family: dict = field(default_factory=dict)          # family -> fused prob (deduped within family)
    classification: str = "no_trigger"                     # transient_anomaly|parameter_drift|
    #                                                        local_structural|global_structural|no_trigger
    dominant_family: str = ""
    scope_candidates: list = field(default_factory=list)
    n_independent_groups: int = 0
    n_evidence: int = 0
    retained_contradictions: list = field(default_factory=list)
    suppressed_by_cooldown: list = field(default_factory=list)
    false_alarm_budget_remaining: float = 1.0
    proceed: bool = False                                   # does this warrant moving to a decision?
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TriggerFusion:
    """Stateful across observation steps within one simulation: holds cooldowns + the false-alarm budget."""
    cooldown_steps: int = 3                                 # a fired family is on cooldown this many steps
    persistence_required: dict = field(default_factory=lambda: {"unexplained_residual": 2, "parameter_drift": 2})
    false_alarm_budget: float = 3.0                        # allowed "soft" (non-structural) triggers per run
    min_fused_probability: float = 0.6                     # hysteresis floor to proceed on a soft trigger
    _cooldown: dict = field(default_factory=dict)          # family -> step_index last fired
    _spent_budget: float = 0.0
    _step: int = 0
    _family_run: dict = field(default_factory=dict)        # family -> consecutive-eligible count

    def cooldown_state(self) -> dict:
        return {fam: {"on_cooldown": (self._step - s) < self.cooldown_steps, "last_fired_at": s}
                for fam, s in self._cooldown.items()}

    def _noisy_or(self, probs):
        p = 1.0
        for x in probs:
            p *= (1.0 - max(0.0, min(1.0, x)))
        return 1.0 - p

    def fuse(self, evidence_list) -> FusedAssessment:
        """Fuse this step's trigger evidence into one assessment; update cooldown + budget state."""
        self._step += 1
        fa = FusedAssessment(n_evidence=len(evidence_list))
        if not evidence_list:
            fa.false_alarm_budget_remaining = round(self.false_alarm_budget - self._spent_budget, 3)
            return fa

        # ---- dedup within family (keep strongest); track persistence for soft families ----
        by_family = {}
        for ev in evidence_list:
            cur = by_family.get(ev.trigger_family)
            if cur is None or ev.trigger_probability > cur.trigger_probability:
                by_family[ev.trigger_family] = ev
        fired_families = set(by_family)
        for fam in list(self._family_run):
            if fam not in fired_families:
                self._family_run[fam] = 0
        for fam in fired_families:
            self._family_run[fam] = self._family_run.get(fam, 0) + 1

        # ---- cooldown/persistence suppression ----
        active = {}
        for fam, ev in by_family.items():
            on_cd = (self._step - self._cooldown.get(fam, -999)) < self.cooldown_steps
            need = self.persistence_required.get(fam, 1)
            if fam in ("impossible_event", "outcome_space_change", "rule_change", "new_institution",
                       "new_actor", "authority_change"):
                need = 1                                    # hard-structural: no persistence delay
            if on_cd and fam not in _STRUCTURAL_FAMILIES:
                fa.suppressed_by_cooldown.append(fam)
                continue
            if self._family_run.get(fam, 1) < need:
                fa.notes.append(f"{fam}: awaiting persistence ({self._family_run.get(fam,1)}/{need})")
                continue
            active[fam] = ev

        if not active:
            fa.false_alarm_budget_remaining = round(self.false_alarm_budget - self._spent_budget, 3)
            return fa

        # ---- dependence-aware fusion: collapse within a dependence group, noisy-OR across groups ----
        groups = {}
        for fam, ev in active.items():
            groups.setdefault(_dependence_key(ev), []).append(ev)
        group_probs = []
        for _k, evs in groups.items():
            group_probs.append(max(e.trigger_probability for e in evs))   # collapse: strongest, not sum
        fa.n_independent_groups = len(groups)
        fused = self._noisy_or(group_probs)

        # ---- one impossible_event dominates ----
        if "impossible_event" in active:
            fused = max(fused, active["impossible_event"].trigger_probability, 0.9)

        fa.by_family = {fam: round(ev.trigger_probability, 4) for fam, ev in active.items()}
        fa.fused_probability = round(fused, 4)
        fa.dominant_family = max(active, key=lambda f: active[f].trigger_probability)
        fa.scope_candidates = sorted({s for ev in active.values() for s in ev.affected_scope_candidates})
        fa.retained_contradictions = sorted({c for ev in active.values() for c in ev.contradictory_observations})

        # ---- classify the situation ----
        structural = fired_families & _STRUCTURAL_FAMILIES
        broad = any(s in _GLOBAL_SCOPES for s in fa.scope_candidates)
        if "impossible_event" in active or "outcome_space_change" in active or (structural and broad):
            fa.classification = "global_structural"
        elif structural:
            fa.classification = "local_structural"
        elif "parameter_drift" in active:
            fa.classification = "parameter_drift"
        else:
            fa.classification = "transient_anomaly"

        # ---- decide whether to proceed to a recompile decision (hysteresis + budget) ----
        hard = bool(active.keys() & _STRUCTURAL_FAMILIES)
        if hard:
            fa.proceed = fused >= 0.5
        else:
            # soft (drift/transient) needs a higher bar AND spends the false-alarm budget
            fa.proceed = fused >= self.min_fused_probability and self._spent_budget < self.false_alarm_budget
            if fa.proceed:
                self._spent_budget += 1.0
        if not fa.proceed and fa.classification in ("transient_anomaly",):
            fa.notes.append("noisy anomaly below hysteresis floor — executing current plan, no recompile")

        # ---- update cooldown for families that proceeded ----
        if fa.proceed:
            for fam in active:
                self._cooldown[fam] = self._step
        fa.false_alarm_budget_remaining = round(self.false_alarm_budget - self._spent_budget, 3)
        return fa
