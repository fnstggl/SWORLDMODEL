"""The ensemble compiler — DEFAULT-ON structural-model uncertainty for every World Model V2 run.

The canonical runtime no longer begins with one `compile_world(...)` call. It begins here:

    Stage A  independent causal reconnaissance — SEPARATE actual LLM calls (normal target: four), each
             blind to every other candidate, each investigating through a different general causal
             perspective, each returning a complete causal model skeleton;
    critics  an adversarial structural-omission critic, per-candidate causal critics and a cross-model
             contrast critic (actual LLM calls, separate prompts/roles) that repair, reject, propose
             missing models and classify support — but never assign predictions, choose actions, or
             mint model probabilities;
    Stage B  evidence-conditioned executable compilation — each surviving candidate compiled SEPARATELY
             into its own `WorldExecutionPlan` through the canonical single-plan compiler with the SAME
             question/intervention/as-of/horizon and the SAME immutable shared evidence bundle, plus its
             own causal thesis as a structural directive (candidates never see each other's details);
    dedup    conservative deduplication — deterministic structural comparison first, a blind-label LLM
             equivalence judge only for unresolved cases; false merges are worse than duplicates.

Adaptive count (Section 6): at least three independent generation attempts, normally four, expansion when
coverage is weak (duplicates, shared decisive assumptions, critic-identified missing structures), soft
ceiling ~eight (higher in maximum-capacity mode). Reaching the ceiling with live critic findings marks the
run `structurally_underidentified` — the ceiling is never proof of completeness.

Failure behavior (Section 23): no LLM backend → loud CompilerExecutionError (never a silent deterministic
fallback plan); identical prompts across "independent" calls → EnsembleIntegrityError; a single surviving
model is legal ONLY with a recorded convergence certificate.
"""
from __future__ import annotations

import copy
import hashlib
import json

from swm.world_model_v2.result import CompilerExecutionError
from swm.world_model_v2.structural_contracts import (
    GENERATION_MAX_CAPACITY_CEILING, GENERATION_MIN_INDEPENDENT_CALLS, GENERATION_PERSPECTIVES,
    GENERATION_SOFT_CEILING, GENERATION_TARGET_CALLS, StructuralModelCandidate, StructuralModelEnsemble,
    EnsembleIntegrityError, schema_hash, structural_signature)
from swm.world_model_v2.llm_call_cache import CachedLLM, CallLedger

ENSEMBLE_COMPILER_VERSION = "structural-ensemble-1.0"


# ------------------------------------------------------------------ Stage A: independent reconnaissance
_RECON_PROMPT = """STRUCTURAL CAUSAL RECONNAISSANCE — you are ONE independent causal-model generator in an
ensemble. You see ONLY this brief; you never see any other generator's output, and no other generator sees
yours. Several materially different causal structures could plausibly determine this outcome — your job is
to construct the strongest COMPLETE causal model visible from your assigned reasoning perspective.

YOUR PERSPECTIVE ({role}): {perspective}
This perspective shapes what you actively investigate — NOT the scope of the model. Your output must still
be a complete causal model of the question (actors AND institutions AND constraints AND mechanisms), with
your perspective determining which of them you treat as DECISIVE.

QUESTION: {q}
INTERVENTION (optional): {intervention}
AS-OF: {as_of}   HORIZON: {horizon}
USER CONTEXT: {context}
GROUNDED EVIDENCE (may be empty at this stage):
{evidence}

Return ONLY JSON:
{{"causal_thesis": "<one or two sentences: the central causal claim of THIS model — what actually determines the outcome>",
 "decisive_actors": ["<actor>", ...],
 "decisive_institutions": ["<institution or formal procedure>", ...],
 "decisive_constraints": ["<resource/capacity/operational/legal/timing constraint>", ...],
 "decisive_mechanisms": ["<causal mechanism by which the outcome is produced>", ...],
 "external_systems": ["<relevant market/algorithmic/legal/environmental/third-party system>", ...],
 "world_boundary": "<what is inside this model's world and what is deliberately outside it>",
 "candidate_omissions": ["<component this model deliberately treats as negligible, and why>", ...],
 "required_evidence": [{{"claim": "<fact needed as of the cutoff>", "why": "<causal reason>"}}, ...],
 "falsifiers": ["<an observation that would make this causal model wrong>", ...],
 "intervention_propagation": "<how the intervention (if any) would propagate under THIS structure — and how that differs from an actor-only reading>"}}"""


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode()).hexdigest()[:12]


def _listify(v, cap=12):
    if isinstance(v, list):
        return [str(x)[:200] for x in v if str(x).strip()][:cap]
    if isinstance(v, str) and v.strip():
        return [v.strip()[:200]]
    return []


def _parse_recon(txt: str) -> dict:
    """Lenient recon parse: any JSON object is accepted and mapped onto the recon fields; a backend that
    answers with a full compiler decomposition still yields a usable skeleton (thesis from rationale,
    actors from entities). Parse quality is recorded, never hidden."""
    from swm.engine.grounding import parse_json
    raw = parse_json(txt) or {}
    if not isinstance(raw, dict) or not raw:
        return {}
    out = {"causal_thesis": str(raw.get("causal_thesis", "") or raw.get("rationale", "") or "")[:600],
           "decisive_actors": _listify(raw.get("decisive_actors")),
           "decisive_institutions": _listify(raw.get("decisive_institutions")),
           "decisive_constraints": _listify(raw.get("decisive_constraints")),
           "decisive_mechanisms": _listify(raw.get("decisive_mechanisms")),
           "external_systems": _listify(raw.get("external_systems")),
           "world_boundary": str(raw.get("world_boundary", ""))[:400],
           "candidate_omissions": _listify(raw.get("candidate_omissions") or raw.get("omitted")),
           "required_evidence": [r for r in (raw.get("required_evidence") or []) if isinstance(r, dict)][:10],
           "falsifiers": _listify(raw.get("falsifiers")),
           "intervention_propagation": str(raw.get("intervention_propagation", ""))[:400],
           "_parse_quality": "recon_schema" if raw.get("causal_thesis") else "salvaged"}
    if not out["decisive_actors"] and isinstance(raw.get("entities"), list):
        out["decisive_actors"] = _listify([e.get("id") for e in raw["entities"] if isinstance(e, dict)])
        out["_parse_quality"] = "salvaged_from_decomposition"
    return out


def reconnoiter_structures(question: str, *, llm, as_of: str, horizon: str, intervention: str = "",
                           user_context=None, evidence_text: str = "", seed: int = 0,
                           generation_policy: dict = None, ledger: CallLedger = None,
                           cache_store: dict = None) -> StructuralModelEnsemble:
    """Stage A: independent causal reconnaissance. Makes SEPARATE actual LLM calls (target four; policy-
    adaptive), each blind to the others, and returns an ensemble of candidate skeletons (no executable
    plans yet — Stage B compiles them against the shared evidence bundle)."""
    if llm is None:
        raise CompilerExecutionError(
            "structural-ensemble generation requires a functioning LLM backend; none supplied — refusing "
            "to fabricate a deterministic fallback model", taxonomy="unavailable_service")
    policy = dict(generation_policy or {})
    target = int(policy.get("target_calls", GENERATION_TARGET_CALLS))
    ceiling = int(policy.get("soft_ceiling",
                             GENERATION_MAX_CAPACITY_CEILING if policy.get("max_capacity")
                             else GENERATION_SOFT_CEILING))
    target = max(GENERATION_MIN_INDEPENDENT_CALLS, min(target, ceiling))
    ledger = ledger or CallLedger()
    gen_llm = CachedLLM(llm, ledger=ledger, stage="structural_generation", store=cache_store)

    ens = StructuralModelEnsemble(question=question, as_of=as_of, horizon=horizon,
                                  intervention=intervention,
                                  generation_policy={"target_calls": target, "soft_ceiling": ceiling,
                                                     "min_calls": GENERATION_MIN_INDEPENDENT_CALLS,
                                                     "max_capacity": bool(policy.get("max_capacity"))})
    ctx = json.dumps(user_context, default=str)[:800] if user_context else "(none)"
    perspectives = list(GENERATION_PERSPECTIVES)
    prompts_seen = []
    for i in range(target):
        role, perspective = perspectives[i % len(perspectives)]
        _generate_candidate(ens, question, role=role, perspective=perspective, llm=gen_llm,
                            as_of=as_of, horizon=horizon, intervention=intervention, ctx=ctx,
                            evidence_text=evidence_text, seed=seed, independent=True,
                            prompts_seen=prompts_seen)
    # loud failure: "independent" calls that shared one prompt mean the independence contract broke
    ok_prompts = [g["prompt_hash"] for g in ens.generation_manifest if g.get("independent")]
    if len(ok_prompts) > 1 and len(set(ok_prompts)) == 1:
        raise EnsembleIntegrityError("all independent generation calls used ONE prompt — generation bug")
    # degenerate-backend detection: byte-identical responses to DISTINCT prompts is a backend property,
    # recorded and later resolved by conservative merge (never treated as real structural diversity)
    ok_resps = [g["response_hash"] for g in ens.generation_manifest if g.get("ok")]
    if len(ok_resps) > 1 and len(set(ok_resps)) == 1:
        ens.generation_policy["degenerate_backend"] = True
    ens.candidates_generated = len(ens.candidates)
    return ens


def _generate_candidate(ens: StructuralModelEnsemble, question: str, *, role: str, perspective: str,
                        llm, as_of: str, horizon: str, intervention: str, ctx: str, evidence_text: str,
                        seed: int, independent: bool, prompts_seen: list, directive_extra: str = ""):
    """One actual generation call → one candidate (or a recorded failure). Never sees other candidates."""
    n = len(ens.generation_manifest)
    call_id = f"gen_{n}_{role}"
    prompt = _RECON_PROMPT.format(role=role, perspective=perspective, q=question,
                                  intervention=intervention or "(none)", as_of=as_of, horizon=horizon,
                                  context=ctx, evidence=(evidence_text or "(none)")[:2400])
    if directive_extra:
        prompt += f"\n\nEXPANSION DIRECTIVE (from the omission critic — a structure missing from every " \
                  f"existing candidate; you still see no other candidate's model):\n{directive_extra[:800]}"
    prompts_seen.append(prompt)
    try:
        txt = llm(prompt)
    except Exception as e:  # noqa: BLE001 — a failed generator is recorded, never silently skipped
        ens.record_generation(role=role, prompt_hash=_hash(prompt), response_hash="", ok=False,
                              independent=independent, error=f"{type(e).__name__}: {e}"[:160],
                              call_id=call_id)
        return None
    recon = _parse_recon(txt)
    ens.record_generation(role=role, prompt_hash=_hash(prompt), response_hash=_hash(txt),
                          ok=bool(recon), independent=independent,
                          error=("" if recon else "unparseable_recon_response"), call_id=call_id)
    cand = StructuralModelCandidate(
        model_id=f"m{len(ens.candidates)}_{role}", independent_generation_call_id=call_id,
        generation_role=role,
        causal_thesis=(recon or {}).get("causal_thesis", ""),
        decisive_actors=(recon or {}).get("decisive_actors", []),
        decisive_institutions=(recon or {}).get("decisive_institutions", []),
        decisive_constraints=(recon or {}).get("decisive_constraints", []),
        decisive_mechanisms=(recon or {}).get("decisive_mechanisms", []),
        world_boundary=(recon or {}).get("world_boundary", ""),
        omitted_components=(recon or {}).get("candidate_omissions", []),
        falsifiers=(recon or {}).get("falsifiers", []),
        intervention_response=(recon or {}).get("intervention_propagation", ""),
        provenance={"generation_prompt_hash": _hash(prompt), "seed": seed,
                    "perspective": perspective, "independent": independent,
                    "recon_parse_quality": (recon or {}).get("_parse_quality", "failed"),
                    "recon_required_evidence": (recon or {}).get("required_evidence", [])})
    ens.candidates.append(cand)
    return cand


# ------------------------------------------------------------------ shared evidence requirements (union)
def union_evidence_requirements(ens: StructuralModelEnsemble, *, as_of_iso: str, max_reqs: int = 18) -> list:
    """Union of nonduplicative typed evidence requirements across the reconnaissance candidates, plus the
    terminal-outcome requirement. Gathered ONCE through the canonical evidence system under ONE as-of
    boundary; the resulting immutable bundle is shared by every model (candidate-specific needs are
    recorded per candidate and become part of the same shared bundle when gathered)."""
    from swm.world_model_v2.evidence_requirements import EvidenceRequirement, _rid
    reqs = [EvidenceRequirement(
        requirement_id=_rid("outcome", ens.question), claim_or_quantity=ens.question,
        why_relevant="the terminal outcome standing as of the question date sets the base rate",
        affected_component="terminal_outcome", expected_sensitivity=1.0, expected_voi=1.0,
        as_of_constraint=as_of_iso, publication_time_scope="paired_after_before",
        absence_informative=True)]
    seen = {ens.question.strip().lower()[:80]}
    for cand in ens.candidates:
        for r in (cand.provenance.get("recon_required_evidence") or []):
            claim = str(r.get("claim", "")).strip()
            key = claim.lower()[:80]
            if not claim or key in seen:
                continue
            seen.add(key)
            reqs.append(EvidenceRequirement(
                requirement_id=_rid("recon", cand.model_id, claim),
                claim_or_quantity=claim[:240],
                why_relevant=str(r.get("why", "reconnaissance-identified causal requirement"))[:200],
                affected_component=f"structural_candidate:{cand.model_id}",
                expected_sensitivity=0.7, expected_voi=0.7, as_of_constraint=as_of_iso,
                publication_time_scope="paired_after_before"))
            cand.evidence_requirements.append(reqs[-1].as_dict())
    reqs.sort(key=lambda r: -r.expected_voi)
    return reqs[:max_reqs]


# ------------------------------------------------------------------ adversarial critics (actual LLM calls)
_OMISSION_PROMPT = """ADVERSARIAL STRUCTURAL-OMISSION CRITIC for a causal-model ensemble. Below are the
candidate causal models generated so far, under blind labels. Your job is to find what is MISSING from
every one of them — not to rank them, not to predict the outcome, not to assign probabilities.

QUESTION: {q}
AS-OF: {as_of}   HORIZON: {horizon}   INTERVENTION: {intervention}
EVIDENCE (shared, as-of):
{evidence}

CANDIDATE MODELS (blind labels):
{summaries}

Answer, strictly as JSON:
{{"missing_decisive_actor": "<an actor absent from EVERY model that could change the outcome, or null>",
 "missing_institution": "<an institution/procedure missing everywhere, or null>",
 "missing_constraint": "<a resource/operational/legal/market/algorithmic/physical/timing constraint missing everywhere, or null>",
 "missing_information_route": "<a communication/information route assumed but never represented, or null>",
 "external_event_reversal": "<an external event that could reverse the result and is unmodeled, or null>",
 "boundary_too_narrow": "<a way every model's world boundary is too narrow, or null>",
 "missing_causal_theory": "<a materially different causal theory that would respond differently to the intervention, or null>",
 "equivalent_sounding": [["<label>","<label>"], ...],
 "proposed_models": [{{"causal_thesis": "...", "decisive_actors": [...], "decisive_institutions": [...],
                      "decisive_constraints": [...], "decisive_mechanisms": [...],
                      "world_boundary": "...", "why_missing": "..."}}],
 "no_further_material_model": <true iff you cannot construct another plausible MATERIALLY different model>,
 "reasoning": "<short>"}}"""

_CANDIDATE_CRITIC_PROMPT = """CANDIDATE CAUSAL CRITIC — adversarial review of ONE candidate causal model.
Judge only THIS model against the question and the evidence. You may propose repairs, identify
contradictions and classify qualitative support. You may NOT predict the outcome, score other models,
or assign any numeric probability.

QUESTION: {q}
AS-OF: {as_of}
EVIDENCE (shared, as-of; items carry ids):
{evidence}

CANDIDATE MODEL:
{summary}

Answer, strictly as JSON:
{{"validity_conditions": ["<what must be true for this causal thesis to hold>", ...],
 "non_executable_mechanisms": ["<mechanism asserted but not executable as stated>", ...],
 "incorrectly_collapsed": ["<actor/institution wrongly merged into another>", ...],
 "skipped_intermediaries": ["<direct effect that skips a real intermediary>", ...],
 "ornamental_components": ["<part of the model not causally connected to the outcome>", ...],
 "evidence_contradictions": [{{"claim_id": "<id from the evidence, or null>", "claim": "<the contradicting fact>",
                              "why_contradicts": "..."}}],
 "missing_outcome_mechanisms": ["<outcome-relevant mechanism this model lacks>", ...],
 "intervention_differentiation": "<meaningful|weak|none — would this model respond differently to interventions than a generic model?>",
 "support_class": "<strongly_supported|plausible|weak_but_possible|contradicted|unresolved>",
 "support_basis": "<the evidence-fit reason for that class — cite claim ids where possible>",
 "repairs": [{{"field": "<candidate field>", "change": "<bounded repair>"}}],
 "reject": <true iff the model is INVALID (incoherent/boundary-violating), not merely weak>,
 "reject_reason": "<why, or null>"}}"""

_CONTRAST_PROMPT = """CROSS-MODEL CONTRAST CRITIC — blind comparison of candidate causal models. Identify
which candidates are GENUINELY structurally different (different decisive actors/institutions/constraints/
mechanisms/boundaries/intervention pathways) versus superficially different narrations of the same
executable structure. Do not rank, do not predict, do not assign probabilities.

QUESTION: {q}
CANDIDATES (blind labels):
{summaries}

Answer, strictly as JSON:
{{"genuinely_different": [["<label>","<label>"], ...],
 "superficial_only": [["<label>","<label>"], ...],
 "same_trajectory_pairs": [["<label>","<label>"], ...],
 "reversal_pairs": [["<label>","<label>","<why these two structures could reverse a forecast or recommendation>"], ...],
 "missing_axes": ["<axis of disagreement no candidate pair covers>", ...],
 "reasoning": "<short>"}}"""

#: numeric fields critics might try to smuggle in — always stripped (LLMs never mint model probabilities)
_FORBIDDEN_CRITIC_FIELDS = ("probability", "model_probability", "prior", "weight", "p_correct",
                            "confidence_score")


def _strip_minted_probabilities(d: dict) -> dict:
    """Drop any numeric model-probability field a critic returned despite instructions. Recorded upstream;
    the qualitative support_class is the ONLY judgment channel."""
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items()
            if not (k.lower() in _FORBIDDEN_CRITIC_FIELDS and isinstance(v, (int, float)))}


def _blind_summaries(cands: list) -> tuple:
    """Blind-label candidate summaries for critics/judges: label -> candidate, plus rendered text."""
    labels = {}
    lines = []
    for i, c in enumerate(cands):
        label = chr(ord("A") + i) if i < 26 else f"Z{i}"
        labels[label] = c
        s = c.summary()
        lines.append(f"Model {label}: thesis={s['causal_thesis'][:220]!r}; "
                     f"actors={s['decisive_actors'][:6]}; institutions={s['decisive_institutions'][:4]}; "
                     f"constraints={s['decisive_constraints'][:4]}; mechanisms={s['decisive_mechanisms'][:5]}; "
                     f"boundary={s['world_boundary'][:140]!r}; "
                     f"intervention_response={s['intervention_response'][:120]!r}")
    return labels, "\n".join(lines)


def run_omission_critic(ens: StructuralModelEnsemble, *, llm, evidence_text: str = "",
                        ledger: CallLedger = None, cache_store: dict = None) -> dict:
    """One actual omission-critic call over blind candidate summaries. Findings feed the adaptive
    expansion policy and the unresolved-alternatives record; the critic never edits models directly."""
    from swm.engine.grounding import parse_json
    crit_llm = CachedLLM(llm, ledger=ledger or CallLedger(), stage="structural_critic", store=cache_store)
    labels, summaries = _blind_summaries(ens.surviving())
    prompt = _OMISSION_PROMPT.format(q=ens.question, as_of=ens.as_of, horizon=ens.horizon,
                                     intervention=ens.intervention or "(none)",
                                     evidence=(evidence_text or "(none)")[:2000], summaries=summaries)
    try:
        raw = _strip_minted_probabilities(parse_json(crit_llm(prompt)) or {})
        ok, err = bool(raw), ""
    except Exception as e:  # noqa: BLE001
        raw, ok, err = {}, False, f"{type(e).__name__}: {e}"[:160]
    ens.critic_manifest.append({"critic": "structural_omission", "prompt_hash": _hash(prompt),
                                "ok": ok, "error": err, "n_candidates_seen": len(labels)})
    findings = {k: raw.get(k) for k in ("missing_decisive_actor", "missing_institution",
                                        "missing_constraint", "missing_information_route",
                                        "external_event_reversal", "boundary_too_narrow",
                                        "missing_causal_theory") if raw.get(k)}
    return {"ok": ok, "findings": findings,
            "equivalent_sounding": raw.get("equivalent_sounding") or [],
            "proposed_models": [p for p in (raw.get("proposed_models") or []) if isinstance(p, dict)][:4],
            "no_further_material_model": bool(raw.get("no_further_material_model")),
            "reasoning": str(raw.get("reasoning", ""))[:400], "blind_labels": labels}


def run_candidate_critics(ens: StructuralModelEnsemble, *, llm, evidence_text: str = "",
                          ledger: CallLedger = None, cache_store: dict = None):
    """One candidate-critic call per surviving candidate. Applies: qualitative support class (+basis),
    recorded findings, bounded repairs, rejection of INVALID models (with exact contradicting evidence
    when that is the ground). Never rejects for a low forecast, an inconvenient result or preference."""
    from swm.engine.grounding import parse_json
    crit_llm = CachedLLM(llm, ledger=ledger or CallLedger(), stage="structural_critic", store=cache_store)
    for cand in ens.surviving():
        prompt = _CANDIDATE_CRITIC_PROMPT.format(q=ens.question, as_of=ens.as_of,
                                                 evidence=(evidence_text or "(none)")[:2000],
                                                 summary=json.dumps(cand.summary())[:1800])
        try:
            raw = _strip_minted_probabilities(parse_json(crit_llm(prompt)) or {})
            ok, err = bool(raw), ""
        except Exception as e:  # noqa: BLE001
            raw, ok, err = {}, False, f"{type(e).__name__}: {e}"[:160]
        ens.critic_manifest.append({"critic": "candidate_causal", "model_id": cand.model_id,
                                    "prompt_hash": _hash(prompt), "ok": ok, "error": err})
        if not ok:
            continue
        cand.critic_findings.append({"critic": "candidate_causal",
                                     **{k: raw.get(k) for k in
                                        ("validity_conditions", "non_executable_mechanisms",
                                         "incorrectly_collapsed", "skipped_intermediaries",
                                         "ornamental_components", "missing_outcome_mechanisms",
                                         "intervention_differentiation") if raw.get(k)}})
        cand.unresolved_mechanisms = _listify(raw.get("non_executable_mechanisms"))
        sc = str(raw.get("support_class", "")).strip()
        contradictions = [c for c in (raw.get("evidence_contradictions") or []) if isinstance(c, dict)]
        if sc in ("strongly_supported", "plausible", "weak_but_possible", "contradicted", "unresolved"):
            cand.support_class = sc
            cand.support_basis = str(raw.get("support_basis", ""))[:300]
        if raw.get("reject") and str(raw.get("reject_reason", "")).strip():
            cand.promotion_status = "rejected"
            cand.promotion_reason = f"critic_invalid: {str(raw.get('reject_reason'))[:200]}"
        elif sc == "contradicted" and contradictions:
            # evidence-contradicted rejection carries the EXACT contradicting evidence
            cand.promotion_status = "rejected"
            cand.promotion_reason = "evidence_contradicted: " + "; ".join(
                f"[{c.get('claim_id') or 'unattributed'}] {str(c.get('claim', ''))[:120]}"
                for c in contradictions[:3])
            cand.critic_findings.append({"critic": "candidate_causal",
                                         "evidence_contradictions": contradictions[:5]})
        for rep in (raw.get("repairs") or [])[:4]:
            if isinstance(rep, dict) and rep.get("field") and rep.get("change"):
                cand.critic_findings.append({"critic": "candidate_causal", "repair": rep})


def run_contrast_critic(ens: StructuralModelEnsemble, *, llm, ledger: CallLedger = None,
                        cache_store: dict = None) -> dict:
    """One blind cross-model contrast call. Output feeds dedup hints + missing-axes expansion triggers."""
    from swm.engine.grounding import parse_json
    crit_llm = CachedLLM(llm, ledger=ledger or CallLedger(), stage="structural_critic", store=cache_store)
    labels, summaries = _blind_summaries(ens.surviving())
    prompt = _CONTRAST_PROMPT.format(q=ens.question, summaries=summaries)
    try:
        raw = _strip_minted_probabilities(parse_json(crit_llm(prompt)) or {})
        ok, err = bool(raw), ""
    except Exception as e:  # noqa: BLE001
        raw, ok, err = {}, False, f"{type(e).__name__}: {e}"[:160]
    ens.critic_manifest.append({"critic": "cross_model_contrast", "prompt_hash": _hash(prompt),
                                "ok": ok, "error": err})
    return {"ok": ok, "blind_labels": labels,
            "superficial_only": raw.get("superficial_only") or [],
            "genuinely_different": raw.get("genuinely_different") or [],
            "same_trajectory_pairs": raw.get("same_trajectory_pairs") or [],
            "reversal_pairs": raw.get("reversal_pairs") or [],
            "missing_axes": _listify(raw.get("missing_axes"))}


# ------------------------------------------------------------------ adaptive expansion (Section 6)
def expansion_triggers(ens: StructuralModelEnsemble, omission: dict, contrast: dict) -> list:
    """Deterministic expansion triggers from critic outputs + candidate structure. Returns reasons."""
    reasons = []
    surviving = ens.surviving()
    if omission.get("findings"):
        reasons.append("omission_critic_found_missing_structure")
    if omission.get("proposed_models"):
        reasons.append("omission_critic_proposed_models")
    theses = {c.causal_thesis[:120] for c in surviving if c.causal_thesis}
    if len(surviving) > 1 and len(theses) == 1:
        reasons.append("all_candidates_share_one_thesis")
    actor_sets = [frozenset(map(str.lower, map(str, c.decisive_actors))) for c in surviving]
    if len(surviving) > 2 and len(set(actor_sets)) == 1:
        reasons.append("all_candidates_share_decisive_assumptions")
    if contrast.get("missing_axes"):
        reasons.append("contrast_critic_missing_axes")
    n_dupes = sum(1 for c in ens.candidates if c.promotion_status == "merged")
    if n_dupes >= max(1, len(ens.candidates) // 2):
        reasons.append("candidates_mostly_duplicates")
    if not any(c.decisive_institutions for c in surviving) and \
            any(c.decisive_constraints for c in surviving):
        pass                                                  # constraints without institutions is fine
    if all(not c.decisive_constraints for c in surviving) and len(surviving) >= 2:
        reasons.append("no_candidate_represents_nonactor_constraints")
    return reasons


def expand_candidates(ens: StructuralModelEnsemble, omission: dict, *, llm, as_of: str, horizon: str,
                      intervention: str, user_context, evidence_text: str, seed: int,
                      ledger: CallLedger = None, cache_store: dict = None) -> int:
    """Generate expansion candidates for critic-identified missing structures, respecting the ceiling.
    Expansion calls are actual generation calls (marked independent=False — they know the GAP, never
    another candidate's content)."""
    ceiling = int(ens.generation_policy.get("soft_ceiling", GENERATION_SOFT_CEILING))
    gen_llm = CachedLLM(llm, ledger=ledger or CallLedger(), stage="structural_generation",
                        store=cache_store)
    ctx = json.dumps(user_context, default=str)[:800] if user_context else "(none)"
    added = 0
    prompts_seen = []
    proposals = list(omission.get("proposed_models") or [])
    if not proposals and omission.get("findings"):
        proposals = [{"why_missing": "; ".join(f"{k}: {v}" for k, v in omission["findings"].items())}]
    for prop in proposals:
        if len(ens.candidates) >= ceiling:
            ens.generation_policy["ceiling_reached"] = True
            break
        directive = json.dumps({k: prop.get(k) for k in
                                ("causal_thesis", "decisive_actors", "decisive_institutions",
                                 "decisive_constraints", "decisive_mechanisms", "world_boundary",
                                 "why_missing") if prop.get(k)}, default=str)
        cand = _generate_candidate(ens, ens.question, role="adversarial_alternative",
                                   perspective="the structure the omission critic found missing from "
                                               "every candidate",
                                   llm=gen_llm, as_of=as_of, horizon=horizon, intervention=intervention,
                                   ctx=ctx, evidence_text=evidence_text, seed=seed, independent=False,
                                   prompts_seen=prompts_seen, directive_extra=directive)
        if cand is not None:
            cand.provenance["expansion_source"] = "structural_omission_critic"
            added += 1
    ens.candidates_generated = len(ens.candidates)
    return added


# ------------------------------------------------------------------ conservative deduplication (Section 8)
_EQUIVALENCE_JUDGE_PROMPT = """STRUCTURAL EQUIVALENCE JUDGE — two candidate causal models, blind labels.
Decide whether they are EQUIVALENT ON EVERY CAUSAL ELEMENT THAT COULD CHANGE THE RESULT: world boundary,
decisive actors, decisive institutions, decisive constraints, decisive mechanisms, information routes,
intervention pathways, scheduled processes and resolution dependencies. Sharing some actors or mechanisms
is NOT equivalence; differently narrated but identically executing structures ARE equivalent.
A false merge is worse than a retained duplicate — when unsure, answer false.

MODEL X: {a}
MODEL Y: {b}
DETERMINISTIC STRUCTURAL COMPARISON (computed, not narrated): {comparison}

Return ONLY JSON:
{{"equivalent_on_result_relevant_elements": <true|false>,
 "confidence": "<high|medium|low>",
 "differences_that_could_change_result": ["..."],
 "reasoning": "<short>"}}"""


_SIG_SET_COMPONENTS = ("entities", "populations", "institutions", "mechanisms", "relations",
                       "latents", "quantities", "action_pathways", "scheduled_event_types")


def _sig_similarity(sig_a: dict, sig_b: dict) -> dict:
    """Deterministic per-component structural comparison. Jaccard per set-valued component + exact-match
    flags for scalar components. `structural_min` is the minimum over EVERY compared set component —
    the judge-eligibility gate (Section 8 compares boundary, actors, institutions, constraints,
    mechanisms, information routes, pathways and scheduled processes; a difference in ANY of them keeps
    the pair out of the judge and unmerged — false merges are worse than duplicates)."""
    out = {}
    for k in _SIG_SET_COMPONENTS:
        a, b = set(sig_a.get(k) or []), set(sig_b.get(k) or [])
        union = a | b
        out[k] = 1.0 if not union else round(len(a & b) / len(union), 3)
    out["outcome_family_equal"] = sig_a.get("outcome_family") == sig_b.get("outcome_family")
    out["outcome_options_equal"] = sig_a.get("outcome_options") == sig_b.get("outcome_options")
    out["readout_equal"] = sig_a.get("readout_var") == sig_b.get("readout_var")
    core = [out[k] for k in ("entities", "institutions", "mechanisms", "action_pathways")]
    out["core_min"] = min(core) if core else 0.0
    out["structural_min"] = min(out[k] for k in _SIG_SET_COMPONENTS)
    return out


def deduplicate_candidates(ens: StructuralModelEnsemble, *, llm=None, ledger: CallLedger = None,
                           cache_store: dict = None, contrast_hints: dict = None):
    """Conservative merge of structurally equivalent EXECUTABLE candidates.

    Order of authority: (1) exact schema-hash equality → merge (recorded; different causal theses over
    identical executable plans are additionally recorded as a compiler defect / superficial distinction);
    (2) high deterministic similarity (core components ≥ 0.9 AND identical outcome contract) → blind LLM
    equivalence judge; merge ONLY on equivalent=true with high confidence; (3) anything else stays. Every
    merge records both source IDs, the exact comparison, judge reasoning, confidence class, the surviving
    model and what was preserved from the merged one."""
    from swm.engine.grounding import parse_json
    survivors = [c for c in ens.surviving() if c.executable_plan is not None]
    judge_llm = None if llm is None else CachedLLM(llm, ledger=ledger or CallLedger(),
                                                   stage="structural_dedup", store=cache_store)
    for i in range(len(survivors)):
        a = survivors[i]
        if a.promotion_status == "merged":
            continue
        for j in range(i + 1, len(survivors)):
            b = survivors[j]
            if b.promotion_status == "merged":
                continue
            sig_a, sig_b = structural_signature(a.executable_plan), structural_signature(b.executable_plan)
            comparison = _sig_similarity(sig_a, sig_b)
            if a.schema_hash and a.schema_hash == b.schema_hash:
                _merge(ens, survivor=a, merged=b, comparison=comparison,
                       confidence="exact_structural_equality", judge_reasoning="deterministic",
                       method="schema_hash")
                if a.causal_thesis[:80] != b.causal_thesis[:80]:
                    ens.merge_manifest[-1]["compiler_defect_or_superficial_distinction"] = (
                        "different causal theses compiled into byte-identical executable structure")
                continue
            hi_sim = (comparison["structural_min"] >= 0.9 and comparison["outcome_family_equal"]
                      and comparison["outcome_options_equal"] and comparison["readout_equal"])
            if not hi_sim or judge_llm is None:
                continue
            prompt = _EQUIVALENCE_JUDGE_PROMPT.format(a=json.dumps(a.summary())[:1200],
                                                      b=json.dumps(b.summary())[:1200],
                                                      comparison=json.dumps(comparison))
            try:
                verdict = parse_json(judge_llm(prompt)) or {}
            except Exception:  # noqa: BLE001 — judge failure = NO merge (conservative)
                verdict = {}
            ens.critic_manifest.append({"critic": "equivalence_judge", "pair": [a.model_id, b.model_id],
                                        "prompt_hash": _hash(prompt), "ok": bool(verdict)})
            if (verdict.get("equivalent_on_result_relevant_elements") is True
                    and str(verdict.get("confidence", "")).lower() == "high"):
                _merge(ens, survivor=a, merged=b, comparison=comparison, confidence="judge_high",
                       judge_reasoning=str(verdict.get("reasoning", ""))[:300], method="llm_judge")
    ens.candidates_merged = sum(1 for c in ens.candidates if c.promotion_status == "merged")


def _merge(ens: StructuralModelEnsemble, *, survivor: StructuralModelCandidate,
           merged: StructuralModelCandidate, comparison: dict, confidence: str,
           judge_reasoning: str, method: str):
    merged.promotion_status = "merged"
    merged.promotion_reason = f"structurally_equivalent_to:{survivor.model_id} ({method})"
    merged.merge_record = {"merged_into": survivor.model_id, "method": method,
                           "confidence": confidence, "comparison": comparison}
    survivor.parent_ids = sorted(set(survivor.parent_ids) | {merged.model_id})
    preserved = {"causal_thesis": merged.causal_thesis, "falsifiers": merged.falsifiers[:4],
                 "generation_role": merged.generation_role}
    ens.merge_manifest.append({"survivor": survivor.model_id, "merged": merged.model_id,
                               "method": method, "confidence": confidence,
                               "judge_reasoning": judge_reasoning, "structural_comparison": comparison,
                               "information_preserved_from_merged": preserved})


# ------------------------------------------------------------------ Stage B: executable compilation
def compile_candidates(ens: StructuralModelEnsemble, *, llm, as_of: str, horizon: str,
                       intervention: str = "", evidence=None, seed: int = 0,
                       ledger: CallLedger = None, cache_store: dict = None, n_budget: int = 30):
    """Stage B: compile each surviving candidate SEPARATELY into its own executable WorldExecutionPlan
    through the canonical single-plan compiler — same question/intervention/as-of/horizon, the same
    immutable shared evidence bundle, its OWN structural directive, its OWN seed. Candidates never see
    each other's compilation. Executability is verified (world build + readout binding + operator
    instantiation); a failed candidate gets ONE bounded repair recompile, then is rejected loudly."""
    from swm.world_model_v2.compiler import compile_world
    comp_llm = CachedLLM(llm, ledger=ledger or CallLedger(), stage="structural_compile",
                         store=cache_store)
    for k, cand in enumerate(ens.surviving()):
        directive = _structural_directive(cand)
        try:
            plan = compile_world(ens.question, llm=comp_llm.with_stage("structural_compile",
                                                                       cand.model_id),
                                 evidence=evidence if evidence is not None else "",
                                 as_of=as_of, horizon=horizon, intervention=intervention,
                                 n_budget=n_budget, seed=seed + 101 * (k + 1), persist=False,
                                 structural_directive=directive)
        except Exception as e:  # noqa: BLE001 — recorded; candidate fails loudly, others continue
            cand.promotion_status = "failed"
            cand.promotion_reason = f"compile_failed: {type(e).__name__}: {e}"[:200]
            continue
        ok, why = _executability_check(plan, llm=comp_llm)
        if not ok:
            # bounded repair: ONE recompile with the executability failure appended to the directive
            try:
                plan = compile_world(ens.question, llm=comp_llm.with_stage("structural_repair",
                                                                           cand.model_id),
                                     evidence=evidence if evidence is not None else "",
                                     as_of=as_of, horizon=horizon, intervention=intervention,
                                     n_budget=n_budget, seed=seed + 101 * (k + 1) + 7, persist=False,
                                     structural_directive=directive +
                                     f"\nREPAIR REQUIRED — previous compilation was not executable: {why}")
                ok, why = _executability_check(plan, llm=comp_llm)
                if ok:
                    cand.promotion_status = "repaired"
                    ens.candidates_repaired += 1
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"repair_compile_failed: {type(e).__name__}: {e}"[:160]
        if not ok:
            cand.promotion_status = "rejected"
            cand.promotion_reason = f"nonexecutable_after_bounded_repair: {why}"[:220]
            cand.validation = {"executable": False, "why": why}
            continue
        plan.provenance["structural_model_id"] = cand.model_id
        plan.provenance["structural_generation_role"] = cand.generation_role
        plan.provenance["ensemble_id"] = ens.ensemble_id
        cand.executable_plan = plan
        cand.plan_hash = plan.plan_hash()
        cand.schema_hash = schema_hash(plan)
        cand.plan_lineage = [plan.plan_hash()]
        cand.validation = {"executable": True, "why": "world_build+readout_binding+operators verified"}
    ens.candidates_rejected = sum(1 for c in ens.candidates
                                  if c.promotion_status in ("rejected", "failed"))


def _structural_directive(cand: StructuralModelCandidate) -> str:
    """The candidate's own causal identity, injected into the canonical compiler prompt. Contains ONLY
    this candidate's content — never another candidate's details."""
    return ("STRUCTURAL DIRECTIVE — independent candidate model "
            f"{cand.model_id!r} (perspective: {cand.generation_role}).\n"
            "This compilation must realize THE FOLLOWING causal model — and only it — as the executable "
            "plan. Competing causal stories are compiled separately; do NOT import them.\n"
            f"- central causal thesis: {cand.causal_thesis or '(derive from the perspective)'}\n"
            f"- decisive actors: {cand.decisive_actors}\n"
            f"- decisive institutions: {cand.decisive_institutions}\n"
            f"- decisive constraints: {cand.decisive_constraints}\n"
            f"- decisive mechanisms: {cand.decisive_mechanisms}\n"
            f"- world boundary: {cand.world_boundary or '(as implied by the thesis)'}\n"
            "Represent every decisive component above as executable entities/institutions/quantities/"
            "latents/mechanisms with real names and real structure; treat the model's declared omissions "
            "as marginal.")


def _executability_check(plan, *, llm=None) -> tuple:
    """Deterministic executability critic: the retained model must materialize and bind through the
    canonical semantic runtime (world build + readout binding + operator instantiation). No LLM opinion
    substitutes for this check."""
    try:
        from swm.world_model_v2.materialize import (build_world, check_readout_binding,
                                                    operators_from_plan)
        base = build_world(plan)
        check_readout_binding(plan, base)
        ops, _rej = operators_from_plan(plan, llm=None)
        if not ops:
            return False, "no executable operator instantiated"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:180]


# ------------------------------------------------------------------ certificate + underidentification
def finalize_survivorship(ens: StructuralModelEnsemble, omission: dict):
    """Record stopping reason, the convergence certificate when exactly one model survives, and the
    structurally-underidentified marker when the ceiling was hit with live critic findings."""
    surviving = ens.surviving()
    unresolved = []
    if omission.get("findings") and not omission.get("proposed_models"):
        unresolved.extend(f"{k}: {v}" for k, v in omission["findings"].items())
    if ens.generation_policy.get("ceiling_reached") and omission.get("findings"):
        ens.structurally_underidentified = True
        unresolved.extend(f"unexpanded: {k}: {v}" for k, v in omission["findings"].items())
    ens.unresolved_alternatives = sorted(set(ens.unresolved_alternatives) | set(unresolved))
    if len(surviving) == 1:
        n_indep = ens.independent_generation_calls()
        dispositions = [{"model_id": c.model_id, "status": c.promotion_status,
                         "reason": c.promotion_reason[:160]}
                        for c in ens.candidates if c is not surviving[0]]
        exhausted = bool(omission.get("no_further_material_model")) or not omission.get("findings")
        ens.convergence_certificate = {
            "independent_generation_calls": n_indep,
            "alternatives_disposition": dispositions,
            "omission_critic_exhausted": exhausted,
            "omission_critic_open_findings": omission.get("findings") or {},
            "degenerate_backend": bool(ens.generation_policy.get("degenerate_backend")),
            "certified": n_indep >= GENERATION_MIN_INDEPENDENT_CALLS and
                         all(d["status"] in ("merged", "rejected", "failed") for d in dispositions)}
        if not ens.convergence_certificate["certified"]:
            ens.structurally_underidentified = True
    if not ens.stopping_reason:
        if ens.generation_policy.get("ceiling_reached"):
            ens.stopping_reason = "generation_ceiling_reached"
        elif omission.get("no_further_material_model"):
            ens.stopping_reason = "omission_critic_exhausted"
        elif len({c.schema_hash for c in surviving if c.schema_hash}) < len(surviving):
            ens.stopping_reason = "generators_converged"
        else:
            ens.stopping_reason = "coverage_targets_met"


# ------------------------------------------------------------------ the one-call orchestrator
def compile_world_ensemble(question: str, *, llm, as_of: str, horizon: str, intervention: str = "",
                           user_context=None, evidence=None, evidence_text: str = "", seed: int = 0,
                           generation_policy: dict = None, ledger: CallLedger = None,
                           cache_store: dict = None, n_budget: int = 30) -> StructuralModelEnsemble:
    """Generate → criticize → expand → compile → validate → deduplicate → certify. The default entry the
    canonical runtime calls in place of the old single `compile_world`. `evidence` (a frozen typed bundle
    or string) is the SHARED immutable evidence — one as-of boundary for every model."""
    ledger = ledger if ledger is not None else CallLedger()
    cache_store = cache_store if cache_store is not None else {}
    ens = reconnoiter_structures(question, llm=llm, as_of=as_of, horizon=horizon,
                                 intervention=intervention, user_context=user_context,
                                 evidence_text=evidence_text, seed=seed,
                                 generation_policy=generation_policy, ledger=ledger,
                                 cache_store=cache_store)
    # critics on the skeletons (pre-compile): omission search + expansion BEFORE spending compile calls
    omission = run_omission_critic(ens, llm=llm, evidence_text=evidence_text, ledger=ledger,
                                   cache_store=cache_store)
    contrast = run_contrast_critic(ens, llm=llm, ledger=ledger, cache_store=cache_store)
    if expansion_triggers(ens, omission, contrast):
        expand_candidates(ens, omission, llm=llm, as_of=as_of, horizon=horizon,
                          intervention=intervention, user_context=user_context,
                          evidence_text=evidence_text, seed=seed, ledger=ledger,
                          cache_store=cache_store)
    # candidate critics: support classes, contradictions, invalidity (evidence-aware)
    run_candidate_critics(ens, llm=llm, evidence_text=evidence_text, ledger=ledger,
                          cache_store=cache_store)
    # Stage B: separate evidence-conditioned executable compilation per surviving candidate
    compile_candidates(ens, llm=llm, as_of=as_of, horizon=horizon, intervention=intervention,
                       evidence=evidence, seed=seed, ledger=ledger, cache_store=cache_store,
                       n_budget=n_budget)
    if not any(c.executable_plan is not None for c in ens.surviving()):
        raise CompilerExecutionError(
            "no executable structural candidate remains after generation, critics and bounded repair — "
            "the ensemble cannot run", taxonomy="invalid_execution_plan")
    # conservative dedup over executable candidates (deterministic first, blind judge for unresolved)
    deduplicate_candidates(ens, llm=llm, ledger=ledger, cache_store=cache_store,
                           contrast_hints=contrast)
    finalize_survivorship(ens, omission)
    if evidence is not None and hasattr(evidence, "bundle_hash"):
        ens.shared_evidence_bundle_hash = evidence.bundle_hash()
        ens.shared_evidence_as_of = as_of
    ens.model_support = {c.model_id: c.support_class for c in ens.surviving()}
    ens.structural_coverage = {
        "axes_represented": sorted({c.generation_role for c in ens.surviving()}),
        "missing_axes": contrast.get("missing_axes", []),
        "equivalent_sounding_flagged": omission.get("equivalent_sounding", [])}
    ens.cost_manifest = ledger.as_dict()
    ens.validate_integrity()
    return ens
