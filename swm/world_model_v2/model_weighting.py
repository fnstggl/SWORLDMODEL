"""Grounded structural-world weighting (WSim): turn the surviving SIMULATED worlds into a genuine weighted
forecast — Bayesian model averaging with GROUNDED, CITED likelihood weights — instead of an equal-weight
blur, an arbitrary LLM confidence, or a silent discard in favour of the outside-view prior.

Pipeline:
  curate_worlds        drop MALFORMED (no simulated distribution / execution_failed), UNSUPPORTED (outcome
                       entirely unresolved in that world), and FALLBACK-DERIVED worlds (the world's outcome
                       came from ITS own prior, not a rollout — not a genuine simulated outcome).
  merge_duplicates     collapse worlds that are the SAME causal story (same structural signature) so a
                       repeated view cannot dominate the average by sheer count.
  grounded_world_weights   each surviving world gets weight ∝ objective_quality × cited LLM plausibility.
                       The objective anchor (support grade × evidence assimilation × resolved-outcome share)
                       forbids arbitrary confidence; the LLM MUST cite evidence / historical base rates /
                       actor state / institutional constraints and name each world's unsupported assumptions.
  weighted_distribution / p_yes    the rich forecast: Σ (world weight × outcome distribution in that world).

The grounded outside-view prior is NOT computed here — it stays a separately-reported baseline the runtime
falls back to ONLY when no valid simulated world exists."""
from __future__ import annotations

from dataclasses import dataclass, field

#: support grade → objective quality anchor. A world's grounding is never arbitrary LLM confidence: it is
#: ANCHORED on the model's own support grade, evidence assimilation, and how much of its outcome mass it
#: actually resolved, then MODULATED by the cited LLM plausibility.
SUPPORT_QUALITY = {"empirically_supported": 1.0, "transfer_supported": 0.72,
                   "exploratory": 0.45, "highly_speculative": 0.28,
                   "completed": 0.5, "": 0.4}
#: an LLM plausibility with NO citation is treated as weak — cited grounding is required, uncited confidence
#: is discounted hard (so the model cannot buy weight with an unsupported assertion).
UNCITED_PLAUSIBILITY_CAP = 0.35


@dataclass
class WorldInfo:
    model_id: str
    dist: dict                                       # simulated terminal outcome distribution (over the
                                                     # world's OWN multiple particle rollouts — "what
                                                     # happens inside this world"; NOT one draw)
    support_grade: str = "exploratory"
    unresolved_share: float = 0.0
    status: str = "completed"
    posterior_consumed: bool = False
    fallback_derived: bool = False                   # dist came from the model's fallback, not a rollout
    n_particles: int = 0                             # within-world stochastic rollouts behind `dist`
    thesis: str = ""
    actors: list = field(default_factory=list)
    institutions: list = field(default_factory=list)
    mechanisms: list = field(default_factory=list)
    critic_findings: list = field(default_factory=list)
    merged_from: list = field(default_factory=list)  # model_ids collapsed into this one as duplicates


@dataclass
class WorldWeight:
    model_id: str
    weight: float                                    # final NORMALIZED weight (Σ = 1 over surviving worlds)
    plausibility: float                              # LLM-proposed [0,1], cited
    objective_quality: float                         # anchor from support/posterior/unresolved
    dist: dict
    rationale: str = ""
    citations: list = field(default_factory=list)
    unsupported_assumptions: list = field(default_factory=list)
    merged_from: list = field(default_factory=list)


# ------------------------------------------------------------------ curation + dedup
def _world_signature(w: WorldInfo) -> str:
    """A causal-story signature: same decisive actors + institutions + mechanisms ⇒ the same world. Used to
    MERGE duplicates so a repeated story does not dominate the average by count."""
    def norm(xs):
        return tuple(sorted(str(x).strip().lower() for x in (xs or []) if str(x).strip()))
    return repr((norm(w.actors), norm(w.institutions), norm(w.mechanisms)))


def curate_worlds(worlds):
    """Remove worlds that must not enter the weighted forecast. Returns (valid, rejected) where rejected is
    [(model_id, reason)]. Malformed / execution-failed / entirely-unresolved / fallback-derived worlds are
    dropped — never silently, always with a reason."""
    valid, rejected = [], []
    for w in worlds:
        if not w.dist or not any(_num(v) for v in w.dist.values()):
            rejected.append((w.model_id, "no_simulated_distribution")); continue
        if w.status in ("execution_failed", "clarification_required"):
            rejected.append((w.model_id, f"status_{w.status}")); continue
        if float(w.unresolved_share or 0.0) >= 0.999:
            rejected.append((w.model_id, "outcome_entirely_unresolved_in_world")); continue
        if w.fallback_derived:
            rejected.append((w.model_id, "fallback_derived_not_a_simulated_outcome")); continue
        valid.append(w)
    return valid, rejected


def merge_duplicates(valid):
    """Collapse worlds with the SAME causal-story signature into one representative (best support grade
    kept), recording the merged ids. Returns (merged_worlds, merge_records)."""
    by_sig, order, merge_records = {}, [], []
    rank = {g: i for i, g in enumerate(
        ["empirically_supported", "transfer_supported", "exploratory", "highly_speculative"])}
    for w in valid:
        sig = _world_signature(w)
        if sig not in by_sig:
            by_sig[sig] = w
            order.append(sig)
        else:
            keep = by_sig[sig]
            loser, winner = sorted([keep, w], key=lambda x: rank.get(x.support_grade, 9))[::-1]
            winner.merged_from = list(winner.merged_from) + [loser.model_id] + list(loser.merged_from)
            by_sig[sig] = winner
            merge_records.append({"kept": winner.model_id, "merged": loser.model_id, "signature": sig[:80]})
    return [by_sig[s] for s in order], merge_records


# ------------------------------------------------------------------ grounded weights
def objective_quality(w: WorldInfo) -> float:
    """The anchor that forbids arbitrary confidence: support grade × evidence-assimilation bonus ×
    resolved-outcome-share penalty. Bounded in [0.05, 1.0]. Independent of any LLM opinion."""
    q = SUPPORT_QUALITY.get(w.support_grade, 0.4)
    q *= (1.0 - 0.6 * min(1.0, float(w.unresolved_share or 0.0)))     # unresolved mass lowers grounding
    if w.posterior_consumed:
        q *= 1.25                                                    # a world updated by as-of evidence
    if w.critic_findings:
        q *= max(0.5, 1.0 - 0.1 * len(w.critic_findings))            # open critic findings lower it
    return max(0.05, min(1.0, q))


_WEIGHT_PROMPT = """You are assigning LIKELIHOOD WEIGHTS to competing simulated worlds for a forecasting
question. Each world is a distinct causal model that was fully simulated (many stochastic rollouts inside
it); the worlds disagree. Judge how PLAUSIBLE each world is as a description of the ACTUAL situation —
grounded, not vibes. This is BAYESIAN model weighting: how likely is THIS world, given the evidence?

QUESTION: {question}
AS-OF EVIDENCE (the only admissible facts):
{evidence}

THE SIMULATED WORLDS (simulated_outcome = how often the outcome happened across that world's own rollouts):
{worlds}

For EACH world, plausibility in [0,1] judged on TWO things:
  (1) SETUP plausibility — is this world's premise consistent with the evidence, historical base rates,
      the actors' known state/incentives, and institutional constraints/procedures?
  (2) BEHAVIOUR consistency — did the simulated behaviour stay realistic? LOWER the plausibility for:
      behaviour inconsistent with the known evidence; impossible institutional actions; actors acting
      against what they demonstrably know; unrealistic timing; unsupported causal jumps.
You MUST justify the score by CITING concrete grounds (evidence facts, a named historical base rate, a known
actor state, an institutional rule). List each world's UNSUPPORTED assumptions (claims it needs but cannot
cite). A world resting on unsupported assumptions, or contradicted by the evidence, gets LOW plausibility.

HARD RULES: do NOT give every world the same score. Do NOT assign high confidence without a citation. Do NOT
reward a world for AGREEING with any outside-view / base-rate forecast — that is circular; judge the world on
its evidence and internal realism ONLY.

Return ONLY JSON: {{"worlds": [{{"model_id": "...", "plausibility": <0..1>,
"rationale": "<one line>", "citations": ["<grounded fact used>", ...],
"unsupported_assumptions": ["<claim it needs but cannot cite>", ...]}}, ...]}}"""


def _llm_plausibility(question, valid, evidence_text, llm) -> dict:
    """Ask the LLM for a CITED plausibility per world. Returns {model_id: {plausibility, rationale,
    citations, unsupported_assumptions}}. On any failure returns {} — the caller falls back to the
    objective anchor alone (recorded), never to equal weights silently."""
    if llm is None or not valid:
        return {}
    from swm.engine.grounding import parse_json
    worlds_txt = "\n".join(
        f"- {w.model_id}: thesis={w.thesis[:160]!r}; decisive_actors={w.actors[:6]}; "
        f"institutions={w.institutions[:5]}; mechanisms={w.mechanisms[:6]}; "
        f"simulated_outcome={ {k: round(_num(v), 3) for k, v in w.dist.items()} } "
        f"(across {w.n_particles or '?'} rollouts); "
        f"support={w.support_grade}; unresolved_share={round(float(w.unresolved_share or 0), 3)}"
        for w in valid)
    try:
        raw = parse_json(llm(_WEIGHT_PROMPT.format(
            question=str(question)[:400], evidence=str(evidence_text or "n/a")[:2400],
            worlds=worlds_txt[:3000]))) or {}
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for row in (raw.get("worlds") or []):
        mid = str(row.get("model_id", ""))
        if not mid:
            continue
        try:
            pl = float(row.get("plausibility"))
        except (TypeError, ValueError):
            continue
        out[mid] = {"plausibility": max(0.0, min(1.0, pl)),
                    "rationale": str(row.get("rationale", ""))[:200],
                    "citations": [str(c)[:160] for c in (row.get("citations") or [])][:5],
                    "unsupported_assumptions":
                        [str(a)[:160] for a in (row.get("unsupported_assumptions") or [])][:5]}
    return out


def grounded_world_weights(question, valid, evidence_text, llm) -> list:
    """weight ∝ objective_quality × cited LLM plausibility, normalized over the valid worlds. An uncited
    plausibility is capped (UNCITED_PLAUSIBILITY_CAP) so weight cannot be bought with an unsupported
    assertion. Falls back to the objective anchor alone if the LLM gives nothing (recorded, not equal)."""
    if not valid:
        return []
    obj = {w.model_id: objective_quality(w) for w in valid}
    scores = _llm_plausibility(question, valid, evidence_text, llm)
    raw = {}
    meta = {}
    for w in valid:
        s = scores.get(w.model_id, {})
        cited = bool(s.get("citations"))
        plaus = float(s.get("plausibility", 0.5))
        if not cited:
            plaus = min(plaus, UNCITED_PLAUSIBILITY_CAP)             # no citation ⇒ weak by construction
        raw[w.model_id] = max(1e-4, obj[w.model_id] * max(1e-3, plaus))
        meta[w.model_id] = (plaus, s)
    total = sum(raw.values()) or 1.0
    out = []
    for w in valid:
        plaus, s = meta[w.model_id]
        out.append(WorldWeight(
            model_id=w.model_id, weight=round(raw[w.model_id] / total, 6), plausibility=round(plaus, 4),
            objective_quality=round(obj[w.model_id], 4), dist=dict(w.dist),
            rationale=s.get("rationale", "") or ("objective-anchor only (no LLM plausibility returned)"
                                                 if not scores else ""),
            citations=s.get("citations", []), unsupported_assumptions=s.get("unsupported_assumptions", []),
            merged_from=list(w.merged_from)))
    return out


# ------------------------------------------------------------------ weighted aggregation
def weighted_distribution(weights) -> dict:
    """Σ (world weight × that world's outcome distribution). Weights are assumed normalized; renormalized
    defensively. Returns {} for no weights."""
    if not weights:
        return {}
    tot = sum(ww.weight for ww in weights) or 1.0
    options = sorted({k for ww in weights for k in ww.dist})
    return {o: round(sum((ww.weight / tot) * _num(ww.dist.get(o, 0.0)) for ww in weights), 6)
            for o in options}


def p_yes(dist: dict, *, yes_keys=None) -> float:
    """Project a distribution onto P(YES) using the same convention as the runtime's _binary_projection."""
    if not dist:
        return None
    keys = list(yes_keys or []) + ["True", "true", "yes", "Yes", "1"]
    for k in keys:
        if k in dist:
            return round(_num(dist[k]), 6)
    nonnull = {k: _num(v) for k, v in dist.items() if k not in ("None", "no_choice")}
    return round(max(nonnull.values()), 6) if nonnull else None


def weighted_spread(weights) -> float:
    """Weight-agnostic disagreement measure: max−min of the worlds' P(YES). Reported as uncertainty when the
    weighted headline is served despite disagreement."""
    ps = [p_yes(ww.dist) for ww in weights]
    ps = [p for p in ps if p is not None]
    return round(max(ps) - min(ps), 4) if len(ps) >= 2 else 0.0


def simulation_confidence(weights, worlds) -> dict:
    """How much to TRUST the weighted simulation vs. shrink toward the outside-view baseline — a transparent
    score in [0,1] from the SIMULATION's own support, NOT from whether it agrees with the baseline (that
    would be circular). Components: mean objective world quality; evidence assimilation (any world updated by
    as-of evidence); rollout depth (more within-world particles ⇒ steadier per-world outcome); disagreement
    penalty (very high spread across worlds ⇒ less trust). Returns {alpha, components}."""
    if not weights:
        return {"alpha": 0.0, "components": {"reason": "no valid simulated world"}}
    mean_quality = sum(ww.objective_quality for ww in weights) / len(weights)
    evidence_backed = any((w.posterior_consumed for w in (worlds or [])))
    total_particles = sum(int(getattr(w, "n_particles", 0) or 0) for w in (worlds or []))
    depth = min(1.0, total_particles / 120.0) if total_particles else 0.4  # ~24 particles/world × 5 ≈ 1.0
    spread = weighted_spread(weights)
    disagreement_factor = max(0.5, 1.0 - 0.5 * spread)                     # spread 1.0 ⇒ ×0.5
    alpha = mean_quality * (1.15 if evidence_backed else 0.9) * (0.6 + 0.4 * depth) * disagreement_factor
    alpha = round(max(0.0, min(1.0, alpha)), 4)
    return {"alpha": alpha, "components": {
        "mean_objective_quality": round(mean_quality, 4), "evidence_backed": bool(evidence_backed),
        "total_within_world_particles": total_particles, "rollout_depth": round(depth, 3),
        "cross_world_spread": spread, "disagreement_factor": round(disagreement_factor, 3),
        "note": "alpha does NOT reward agreement with the outside view (anti-circular); it measures the "
                "simulation's own grounding, evidence use, rollout depth, and internal agreement"}}


def final_forecast_selection(n_worlds_valid, weighted_p, outside_p, *, combined=None) -> dict:
    """The SELECTION CONTRACT (single source of truth, unit-testable): whenever ≥1 valid simulated world
    produced a weighted forecast, the source is 'weighted_simulation' and the outside-view fallback must NOT
    silently substitute. Only when NO valid simulated world exists is the source 'grounded_fallback'. This
    is what the no-silent-substitution test asserts against."""
    if int(n_worlds_valid or 0) >= 1 and weighted_p is not None:
        return {"source": "weighted_simulation", "forecast": weighted_p,
                "combined_forecast": combined if combined is not None else weighted_p}
    return {"source": "grounded_fallback", "forecast": outside_p, "combined_forecast": outside_p}


def combined_forecast(sim_p, outside_p, alpha) -> float:
    """The FINAL combined number: shrink the simulation forecast toward the outside-view baseline in
    proportion to how weak the simulation is (1−alpha). alpha≈1 ⇒ final≈simulation; alpha≈0 ⇒ final≈outside
    view. This is regularization by simulation SUPPORT, not by agreement — it never rewards a world for
    matching the baseline. Returns sim_p when the outside view is unavailable."""
    if sim_p is None:
        return outside_p
    if outside_p is None:
        return round(float(sim_p), 6)
    a = max(0.0, min(1.0, float(alpha)))
    return round(a * float(sim_p) + (1.0 - a) * float(outside_p), 6)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
