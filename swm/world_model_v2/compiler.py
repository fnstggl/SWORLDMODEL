"""The universal world-slice compiler — Phase 1 (production, no-abstention).

Question → typed, executable WorldExecutionPlan for EVERY coherent social question. The compiler builds a
CAUSALLY SUFFICIENT model (every component that could materially change the outcome; prune only
demonstrably-negligible ones; represent uncertainty over inclusion when relevance is uncertain; never omit
high-sensitivity unknowns — use broad priors / competing hypotheses / sensitivity instead). It prefers the
least-unnecessarily-complex model among causally-sufficient ones — NOT the smallest absolute.

NO scenario-level routing (no `if election → …`; AST-pinned). The LLM PROPOSES a decomposition; the compiler
VALIDATES, REPAIRS and GUARANTEES executability:
 - competing outcome-contract interpretations for ambiguous questions (simulate each; clarify only when NO
   coherent interpretation exists — rare);
 - readout REPAIR (alias/derived/synthesized) so terminal readout always binds;
 - a mechanism FALLBACK HIERARCHY (fallback.py) so ≥1 executable mechanism always exists — `no validated
   mechanism` becomes a tier-6/7 generic mechanism with broad priors, NEVER `no forecast`;
 - a canonical resolve_outcome event so every plan reaches terminal readout.

Epistemic weakness NEVER prevents a forecast. It lowers the SUPPORT GRADE and widens uncertainty. Only
TECHNICAL failures raise CompilerExecutionError (→ execution_failed); only genuinely incoherent questions
raise ClarificationRequired (→ clarification_required, rare).
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field

from swm.world_model_v2.contracts import ContractError, FAMILIES, OutcomeContract
from swm.world_model_v2.events import register_event_type
from swm.world_model_v2.fallback import (MechanismChoice, overall_support_grade, select_tier,
                                        select_tier_for_process)
from swm.world_model_v2.mechanisms import known_mechanisms
from swm.world_model_v2.result import CompilerExecutionError, ClarificationRequired
from swm.world_model_v2.state import parse_time


class CompileAbstention(Exception):
    """DEPRECATED (pre-no-abstention). Retained so old imports resolve; the production compiler no longer
    raises this for epistemic reasons. New code raises CompilerExecutionError / ClarificationRequired."""


_DECOMPOSE_PROMPT = """You are the WORLD-SLICE COMPILER's proposal stage for a general social world model.
You propose a CAUSALLY SUFFICIENT decomposition; a validator will type-check, repair and execute it.

Build a model that includes EVERY actor, population, institution, relationship, information flow, latent
variable, mechanism, constraint and exogenous factor that could MATERIALLY change the outcome distribution.
Do NOT minimize — include a component whenever its relevance is uncertain (represent the uncertainty).
Omit a component only if it is clearly negligible. High-sensitivity unknowns that lack evidence must still
be listed as latents (they will get broad priors), never dropped.

You may reference mechanisms from the registry below by id; if a needed mechanism is missing, name it under
"missing_mechanisms" (it will be handled by the fallback hierarchy, never silently executed as validated).

QUESTION: {q}
INTERVENTION (optional): {intervention}
AS-OF: {as_of}   HORIZON: {horizon}
GROUNDED EVIDENCE:
{evidence}

MECHANISM REGISTRY: {mechanisms}
OUTCOME FAMILIES: {families}

Return ONLY JSON:
{{"coherent": true,
 "interpretations": [{{"id": "primary", "reading": "<one sentence>", "weight": <0..1>}}],
 "outcome": {{"family": "...", "options": [...], "resolution_rule": "...", "readout_var": "<entity.field or quantity whose terminal value answers the question>"}},
 "outcome_lean": "<strong_no|weak_no|neutral|weak_yes|strong_yes — a QUALITATIVE directional read from the evidence; NOT a probability>",
 "entities": [{{"id": "...", "type": "person|institution", "fields": {{"<universal field>": "<value or ?>"}}, "sensitivity": <0..1>}}],
 "populations": [{{"id": "...", "segments": [{{"id": "...", "weight": <0..1>, "differs_on": ["<dimension>", ...]}}], "sensitivity": <0..1>}}],
 "relations": [{{"src": "...", "rel": "<registered relation>", "dst": "..."}}],
 "institutions": [{{"id": "...", "rules": [{{"kind": "...", "params": {{}}}}], "sensitivity": <0..1>}}],
 "quantities": [{{"name": "...", "qtype": "...", "value": <num or null>, "sd": <num or null>}}],
 "latents": [{{"path": "<entity.field>", "why": "...", "lo": <num>, "hi": <num>, "sensitivity": <0..1>}}],
 "structural_hypotheses": [{{"id": "...", "describe": "<a competing causal structure>", "prior": <0..1>, "lean": "<how THIS structure leans the outcome: strong_no|weak_no|neutral|weak_yes|strong_yes>"}}],
 "actor_decisions": [{{"actor": "<entity id>", "role": "<role>", "at": "<RFC3339 or YYYY-MM-DD>",
   "candidate_actions": [{{"name": "<typed semantic action>", "family": "<action family>",
      "target": {{"target_type": "actor|institution|population|resource|none", "target_id": "..."}},
      "preconditions": [], "information_requirements": [], "authority_requirements": [],
      "resource_requirements": {{}}, "resource_costs": {{}}, "expected_duration_s": 0,
      "mechanisms_triggered": ["record_action|message_delivery|institution_processing|reaction_scheduling"],
      "possible_consequences": [], "possible_delayed_consequences": [],
      "inclusion_reason": "<causal relevance>"}}]}}],
 "scheduled_events": [{{"etype": "...", "at": "<RFC3339 or YYYY-MM-DD>", "participants": [...], "payload": {{}}}}],
 "hazards": [{{"etype": "...", "rate_per_day": <num>, "participants": [...]}}],
 "mechanisms": ["<registry id>", ...],
 "missing_mechanisms": [{{"name": "...", "why": "..."}}],
 "required_causal_processes": ["<short name of each process that determines the outcome>", ...],
 "causal_dependencies": {{"strategic_actor_decisions": <true if outcome depends on deliberate choices by identifiable strategic actors (negotiation, election strategy, organizational decisions, conflict)>,
   "aggregate_population_behavior": <true if outcome depends on aggregated heterogeneous behavior of many people (turnout, adoption, demand, participation, opinion)>,
   "networked_transmission": <true if outcome depends on transmission across relationships (communication, exposure, trust, influence, diffusion, contagion, coordination)>,
   "nonlinear_dynamics": <true if outcome involves thresholds, tipping points, saturation, cascades, self-excitation, feedback or regime shifts>,
   "institutional_decision_process": <true if the outcome itself is decided by a rule-governed institutional procedure (vote, confirmation, ruling, approval)>,
   "structural_change_monitoring": <true if the world structure itself may change before the horizon (new actors, rule changes, coalition shifts)>}},
 "omitted": [{{"component": "...", "reason": "negligible sensitivity because ...", "sensitivity": <0..1>}}],
 "domain": "<one short tag>", "population_kind": "<...>", "time_scale": "<hours|days|weeks|months>",
 "available_data": ["<evidence kinds present>", ...],
 "rationale": "<one sentence per major inclusion/omission>"}}

If the question is NOT coherent enough to define ANY outcome contract even after considering interpretations,
return {{"coherent": false, "why": "<precise reason>", "interpretations": [...tried...]}}."""


@dataclass
class WorldExecutionPlan:
    question: str
    outcome_contract: OutcomeContract
    as_of: float
    horizon_ts: float
    entities: list = field(default_factory=list)
    populations: list = field(default_factory=list)
    institutions: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    quantities: list = field(default_factory=list)
    latents: list = field(default_factory=list)
    scheduled_events: list = field(default_factory=list)
    stochastic_hazards: list = field(default_factory=list)
    accepted_mechanisms: list = field(default_factory=list)
    candidate_experimental_mechanisms: list = field(default_factory=list)
    rejected_mechanisms: list = field(default_factory=list)
    structural_hypotheses: list = field(default_factory=list)         # [{id, describe, prior}]
    actor_decisions: list = field(default_factory=list)               # compiler semantic proposals; no probabilities
    mechanism_choices: list = field(default_factory=list)             # [MechanismChoice.as_dict()]
    fallbacks_used: list = field(default_factory=list)
    support_grade: str = "exploratory"
    interpretations: list = field(default_factory=list)
    omissions: list = field(default_factory=list)
    degraded: bool = False
    fidelity_plan: dict = field(default_factory=dict)
    uncertainty_plan: dict = field(default_factory=dict)
    compute_plan: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    version: int = 1
    parent_version: int = 0

    def plan_hash(self) -> str:
        payload = f"{self.question}|{self.as_of}|{self.horizon_ts}|{len(self.entities)}|" \
                  f"{[m.get('mech_id') for m in self.accepted_mechanisms]}|{self.support_grade}"
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


COMPILER_VERSION = "phase1-no-abstention-1.0"


def _fidelity_plan(proposal: dict, n_budget: int = 30) -> dict:
    """Causal-sufficiency fidelity: HIGH-sensitivity components get explicit representation; LOW-sensitivity
    ones are MARGINALIZED (represented with uncertainty), NOT dropped. Uncertain-relevance components are
    kept. Particles scale with latents + structural hypotheses (competing structures need coverage)."""
    def sens(items, default=0.5):
        out = {}
        for it in (items or []):
            if isinstance(it, dict):
                out[it.get("id") or it.get("path") or it.get("name") or "?"] = float(it.get("sensitivity", default) or default)
        return out
    all_sens = {}
    for k in ("entities", "populations", "institutions", "latents"):
        all_sens.update(sens(proposal.get(k)))
    # mechanisms are causal components too: the top-level `sensitivity` map (mechanism/process → weight)
    # feeds the SAME marginalization decision, so a low-sensitivity mechanism is marginalized-with-
    # uncertainty rather than given full explicit fidelity (still executed — never dropped).
    mech_sens = proposal.get("sensitivity")
    if isinstance(mech_sens, dict):
        for name, s in mech_sens.items():
            try:
                all_sens[str(name)] = float(s)
            except (TypeError, ValueError):
                continue
    explicit = [c for c, s in all_sens.items() if s >= 0.35]
    marginalized = [c for c, s in all_sens.items() if s < 0.35]     # kept, represented with uncertainty
    n_latents = len(proposal.get("latents") or [])
    n_hyp = max(1, len(proposal.get("structural_hypotheses") or []))
    particles = max(12, min(80, n_budget + 5 * n_latents + 8 * (n_hyp - 1)))
    return {"explicit": explicit, "marginalized_with_uncertainty": marginalized,
            "n_particles": particles, "n_structural_hypotheses": n_hyp,
            "agent_samples": min(24, 4 + 2 * len(proposal.get("entities") or [])),
            "policy": "causal-sufficiency: low-sensitivity components marginalized (kept with uncertainty), "
                      "never dropped; high-sensitivity unknowns get broad priors"}


def _make_readout(var):
    def read(world):
        if var in world.quantities:
            q = world.quantities[var]
            return q.value if hasattr(q, "value") else q
        eid, _, fpath = var.partition(".")
        ent = world.entities.get(eid)
        if ent is None:
            return None
        fname, _, key = fpath.partition("[")
        return ent.value(fname, key=key.rstrip("]") or None)
    return read


def _salvage_json(txt: str):
    """Recover a usable dict from a TRUNCATED LLM JSON reply (max_tokens cut it off mid-object). parse_json's
    `{.*}` needs a closing brace a truncated reply lacks; here we walk the text tracking string state + bracket
    depth, close whatever is still open, and if that won't parse, trim the last comma-separated member and
    retry. This keeps the decomposition PREFIX (which lists `outcome`/entities first) rather than losing the
    whole plan to one truncated tail field — a coherent question still simulates. Returns {} if unsalvageable."""
    import re as _re
    if not isinstance(txt, str):
        return {}
    s = _re.sub(r"```(?:json)?|```", "", txt).strip()
    start = s.find("{")
    if start < 0:
        return {}
    s = s[start:]
    # walk, tracking string/escape state and bracket depth
    depth, in_str, esc, stack = 0, False, False, []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    closed = s + ('"' if in_str else "") + "".join(reversed(stack))
    try:
        obj = json.loads(closed)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        pass
    # iteratively drop the trailing (partial) member and re-close
    body = s
    for _ in range(40):
        cut = max(body.rfind(","), body.rfind("}"), body.rfind("]"))
        if cut <= 0:
            break
        body = body[:cut]
        d2, in_s2, e2, st2 = 0, False, False, []
        for ch in body:
            if in_s2:
                if e2:
                    e2 = False
                elif ch == "\\":
                    e2 = True
                elif ch == '"':
                    in_s2 = False
                continue
            if ch == '"':
                in_s2 = True
            elif ch in "{[":
                st2.append("}" if ch == "{" else "]")
            elif ch in "}]" and st2:
                st2.pop()
        try:
            obj = json.loads(body + ('"' if in_s2 else "") + "".join(reversed(st2)))
            if isinstance(obj, dict) and obj:
                return obj
        except ValueError:
            continue
    return {}


def _repair_readout(readout_var: str, proposal: dict) -> tuple:
    """READOUT REPAIR (B9): guarantee the terminal readout binds AND gets written. If the proposed
    readout_var references a declared QUANTITY (which the mechanism chain can populate), keep it. Otherwise
    route the contract through a canonical `outcome` quantity that the terminal resolver writes as a safety
    net — an entity.field readout is only safe when a domain mechanism is known to write it, which the
    general path does not yet guarantee (documented Phase-1 dependency). Returns
    (readout_var, canonical_quantity_the_resolver_writes, repaired: bool)."""
    var = (readout_var or "").strip()
    declared_quantities = {str(q.get("name")) for q in (proposal.get("quantities") or [])}
    if var and var in declared_quantities:
        return var, var, False                               # a declared quantity: resolver writes it (if unset)
    canonical = "outcome"
    return canonical, canonical, bool(var and var != canonical)


#: negation tokens used to detect which binary option is the NEGATIVE outcome. Curated whole-token set (no
#: risky prefixes like un-/in-/dis- that false-positive on "under"/"increase"/"individual").
_NEG_TOKENS = frozenset((
    "no", "not", "non", "none", "never", "fail", "fails", "failed", "failure", "reject", "rejected",
    "rejects", "decline", "declined", "declines", "deny", "denied", "denies", "against", "lose", "loses",
    "lost", "loss", "false", "absent", "unsuccessful", "withdraw", "withdrawn", "withdrew", "block",
    "blocked", "defeat", "defeated", "miss", "missed", "nay", "down", "below", "unresolved", "unmet",
    "nonoccurrence", "dont", "doesnt", "wont", "cannot", "nope", "unapproved", "unratified"))


def _negativity(opt: str) -> int:
    """Score how NEGATIVE a binary option label reads. options are split on _/-/space and matched against a
    curated negation lexicon; leading no_/not_/non- also count. Purely lexical (the LLM proposes only the
    qualitative labels — no probability is minted here)."""
    import re as _re
    s = opt.strip().lower().replace("-", "_")
    score = 0
    for tok in _re.split(r"[_\s]+", s):
        if tok in _NEG_TOKENS:
            score += 2
    if s.startswith(("no_", "not_", "non_", "n_")):
        score += 1
    return score


def _affirmative_first(options):
    """Return the 2 options with the AFFIRMATIVE (least-negative) one first. The generic resolver applies the
    outcome_lean toward options[0] and the binary projection reports P(options[0]); both rely on options[0]
    being the affirmative answer to the question. LLMs order options inconsistently (e.g. ['no_reply',
    'reply']), so we normalize by lexical negativity, keeping the LLM order on a tie."""
    if len(options) != 2:
        return options
    n0, n1 = _negativity(options[0]), _negativity(options[1])
    return [options[1], options[0]] if n1 < n0 else list(options)


def _coerce_outcome(o: dict) -> dict:
    """Repair a malformed outcome contract into a valid one rather than refusing. Fixes: bad family →
    infer from options; empty options for categorical → synthesize True/False; missing resolution rule;
    binary option ORDER normalized so the affirmative outcome is options[0] (lean + projection convention)."""
    fam = str(o.get("family", "")).strip()
    options = [str(x) for x in (o.get("options") or []) if str(x).strip()]
    if fam not in FAMILIES:
        fam = "binary" if len(options) <= 2 else "categorical"
    if fam in ("binary", "response_occurrence") and len(options) != 2:
        options = ["True", "False"]
    if fam == "categorical" and len(options) < 2:
        options = options + [f"option_{i}" for i in range(2 - len(options))]
    if fam in ("binary", "response_occurrence"):
        options = _affirmative_first(options)                # affirmative outcome first (polarity contract)
    return {"family": fam, "options": options,
            "resolution_rule": str(o.get("resolution_rule", "") or "resolved from terminal state")[:300]}


def compile_world(question: str, *, llm, evidence="", as_of: str, horizon: str,
                  intervention: str = "", n_budget: int = 30, seed: int = 0,
                  persist: bool = True) -> WorldExecutionPlan:
    """Compile a causally-sufficient, executable plan for ANY coherent question. Never abstains for
    epistemic reasons. Raises CompilerExecutionError (technical) or ClarificationRequired (incoherent, rare).
    `evidence` may be a typed EvidenceBundle or a legacy string."""
    from swm.engine.grounding import parse_json
    from swm.world_model_v2.init_state import LatentVariableRecord
    from swm.world_model_v2.transitions import _OPERATORS

    registry = known_mechanisms()
    bundle_hash, evidence_basis = "", "legacy_string_unaudited"
    if hasattr(evidence, "render") and hasattr(evidence, "bundle_hash"):
        bundle_hash, evidence_basis = evidence.bundle_hash(), "typed_bundle"
        evidence_text = evidence.render(max_chars=4000)
    else:
        evidence_text = str(evidence or "")[:4000]
    prompt = _DECOMPOSE_PROMPT.format(
        q=question, intervention=intervention or "(none)", as_of=as_of, horizon=horizon,
        evidence=evidence_text or "(none)",
        mechanisms=json.dumps({k: v.causal_role for k, v in registry.items()}),
        families=json.dumps(list(FAMILIES)))

    # ---- LLM decomposition with a bounded parse retry (parse failure is TECHNICAL, not epistemic) ----
    raw = None
    for attempt in range(2):
        try:
            txt = llm(prompt if attempt == 0 else prompt + "\n\nReturn STRICT JSON only, no prose.")
        except Exception as e:
            raise CompilerExecutionError(f"LLM call failed: {e}", taxonomy="unavailable_service")
        raw = parse_json(txt) or {}
        if not raw.get("outcome") and raw.get("coherent") is not False:
            salvaged = _salvage_json(txt)                     # recover a TRUNCATED decomposition's prefix
            if salvaged.get("outcome"):
                raw = salvaged
        if raw.get("outcome") or raw.get("coherent") is False:
            break
    raw = raw or {}

    # ---- coherence: only a genuinely incoherent question clarifies (rare) ----
    if raw.get("coherent") is False and not raw.get("outcome"):
        raise ClarificationRequired(
            str(raw.get("why", "no coherent outcome contract could be constructed"))[:300],
            interpretations_tried=list(raw.get("interpretations") or []))
    if not isinstance(raw.get("outcome"), dict):
        # parser gave nothing usable after retries → synthesize a minimal binary contract (degraded),
        # so a coherent question still simulates rather than refusing. Technical only if the LLM produced
        # literally nothing.
        if not raw:
            raise CompilerExecutionError("empty LLM decomposition after retries",
                                         taxonomy="parser_failure_after_retries")
        raw["outcome"] = {"family": "binary", "options": ["True", "False"],
                          "readout_var": "outcome", "resolution_rule": question[:200]}
        raw.setdefault("_repairs", []).append("synthesized_minimal_binary_outcome")

    interpretations = list(raw.get("interpretations") or [{"id": "primary", "reading": question[:120],
                                                           "weight": 1.0}])

    # ---- outcome contract (repaired, never refused) ----
    o = _coerce_outcome(raw["outcome"])
    proposed_readout = str(raw["outcome"].get("readout_var", "")).strip()
    readout_var, synth_q, readout_repaired = _repair_readout(proposed_readout, raw)
    try:
        contract = OutcomeContract(family=o["family"], options=o["options"],
                                   resolution_rule=o["resolution_rule"],
                                   readout=_make_readout(readout_var), readout_var=readout_var,
                                   horizon_ts=parse_time(horizon)).validate()
    except (ContractError, ValueError):
        # last-resort repair → binary on a synthesized outcome quantity
        readout_var, synth_q, readout_repaired = "outcome", "outcome", True
        contract = OutcomeContract(family="binary", options=["True", "False"],
                                   resolution_rule=o["resolution_rule"],
                                   readout=_make_readout("outcome"), readout_var="outcome",
                                   horizon_ts=parse_time(horizon)).validate()

    # ensure the synthesized readout quantity is declared so it materializes + binds
    quantities = list(raw.get("quantities") or [])
    if synth_q and synth_q not in {str(q.get("name")) for q in quantities}:
        quantities.append({"name": synth_q, "qtype": synth_q, "value": None, "sd": None})

    # ---- mechanisms: registry-vetted + executable; the rest handled by the FALLBACK HIERARCHY ----
    accepted, rejected = [], []
    for mid in raw.get("mechanisms") or []:
        m = registry.get(mid)
        if m is None:
            rejected.append({"id": mid, "rejection_reason": "not in registry"})
        elif not m.operator or m.operator not in _OPERATORS:
            rejected.append({"id": mid, "rejection_reason": f"no executable operator ({m.operator!r})"})
        elif _OPERATORS[m.operator]["experimental"] and mid != "generic_outcome_prior":
            rejected.append({"id": mid, "rejection_reason": f"operator {m.operator!r} experimental"})
        else:
            accepted.append({"mech_id": mid, "ontology_type": m.ontology_type, "causal_role": m.causal_role,
                             "parameter_source": m.parameter_source, "temporal_scale": m.temporal_scale,
                             "calibration_status": m.calibration_status, "operator": m.operator,
                             "sensitivity": (raw.get("sensitivity") or {}).get(mid, 0.5)})
    experimental = [{"name": str(m.get("name", ""))[:60], "why": str(m.get("why", ""))[:200],
                     "status": "experimental — handled by fallback hierarchy, not executed as validated"}
                    for m in (raw.get("missing_mechanisms") or []) if isinstance(m, dict)]

    # ---- Phase 4 production decision path -----------------------------------------------
    # A compiler may propose semantic actions, never numeric behavior probabilities.  Every
    # compiled decision is routed through the ActorView/feasibility/posterior operator.  If a
    # legacy `agent_decision` was proposed for the same event it is replaced, preventing two
    # decision operators from firing and preventing the old uniform/LLM path from bypassing Phase 4.
    actor_decisions = [d for d in (raw.get("actor_decisions") or []) if isinstance(d, dict)]
    has_decision_event = bool(actor_decisions) or any(
        isinstance(e, dict) and e.get("etype") == "decision_opportunity"
        for e in (raw.get("scheduled_events") or []))
    if has_decision_event:
        accepted = [m for m in accepted if m.get("operator") != "agent_decision"]
        if not any(m.get("operator") == "production_actor_policy" for m in accepted):
            pm = registry.get("production_actor_policy")
            if pm is not None:
                accepted.append({"mech_id": pm.mech_id, "ontology_type": pm.ontology_type,
                                 "causal_role": pm.causal_role,
                                 "parameter_source": pm.parameter_source,
                                 "temporal_scale": pm.temporal_scale,
                                 "calibration_status": pm.calibration_status,
                                 "operator": pm.operator, "sensitivity": 0.8})

    # ---- FALLBACK HIERARCHY: guarantee ≥1 executable mechanism + a canonical outcome resolver ----
    scenario = _scenario_descriptor(raw)
    mechanism_selection = _score_production_registry(scenario)
    processes = [str(p) for p in (raw.get("required_causal_processes") or [])] or ["outcome_resolution"]
    choices, fallbacks_used = [], []
    has_domain_mechanism = bool(accepted)
    # PHASE 6: request a mechanism BY CAUSAL PROCESS (not one scenario winner reused for every process).
    # Each required process is matched to the family that ANSWERS it; the tier reflects THAT family's
    # evidence; competing families are preserved; the composition engine flags double-counting.
    per_process, composition = _select_mechanisms_per_process(scenario, processes)
    # PHASE 10: for institutional causal processes, select the evidence-backed institution BY CAUSAL NEED
    # (as-of + jurisdiction), never by keyword. Recorded in provenance; the institution owns authority /
    # procedure / thresholds through which Phase-6 behavioral mechanisms operate.
    institution_selection = _select_institutions(raw, processes, as_of)
    for proc in processes:
        sel = per_process.get(proc, {})
        choice = select_tier_for_process(proc, sel.get("selected"),
                                         competing=[c["family_id"] for c in sel.get("competing", [])])
        choices.append(choice)
    # ALWAYS attach the generic outcome resolver as the terminal SAFETY NET (tier 6/7): it writes the
    # canonical readout quantity ONLY IF unset at the horizon, so a domain mechanism that genuinely
    # resolves the outcome takes precedence, and a plan whose domain mechanisms don't resolve it still
    # produces a broad-prior forecast rather than an unresolved no-op. This guarantees the readout binds
    # AND the option space is covered for EVERY coherent question.
    lean = str(raw.get("outcome_lean", "neutral"))
    resolve_var = synth_q or readout_var
    n_hyp = len(raw.get("structural_hypotheses") or [])
    accepted.append({"mech_id": "generic_outcome_prior", "ontology_type": "numerical",
                     "causal_role": "resolve terminal outcome from a broad prior (terminal safety net)",
                     "parameter_source": "broad prior; tier 6/7", "temporal_scale": "horizon",
                     "calibration_status": "experimental", "operator": "generic_outcome_prior",
                     "sensitivity": 1.0})
    fallbacks_used.append({"process": "outcome_resolution", "tier": 7 if n_hyp > 1 else 6,
                           "family": "generic_outcome_prior",
                           "why": ("outcome resolved from a broad prior — no held-out-validated mechanism "
                                   "writes this readout on the general path (documented Phase-1 dependency)")
                           if not has_domain_mechanism else
                           "broad-prior terminal safety net (writes only if domain mechanisms leave it unset)"})
    if not any(c.family == "generic_outcome_prior" for c in choices):
        choices.append(select_tier("outcome_resolution", None,
                                   competing=[h.get("id") for h in (raw.get("structural_hypotheses") or [])]
                                   if n_hyp > 1 else None))

    support = overall_support_grade(choices)
    degraded = support in ("transfer_supported", "exploratory", "highly_speculative") or bool(fallbacks_used) \
        or readout_repaired

    # ---- latents (always distributions; high-sensitivity unknowns kept with broad priors) ----
    latents = []
    for l in raw.get("latents") or []:
        try:
            lo, hi = float(l.get("lo", 0.0)), float(l.get("hi", 1.0))
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        latents.append(LatentVariableRecord(
            path=str(l.get("path", "")), method="prior",
            confidence=min(0.4, float(l.get("sensitivity", 0.3) or 0.3)),
            candidates={"mean": (lo + hi) / 2, "sd": (hi - lo) / 4, "lo": lo, "hi": hi},
            evidence=[str(l.get("why", ""))[:120]],
            sensitivity=float(l.get("sensitivity", 0.5) or 0.5)))

    # ---- events (+ the canonical resolve_outcome at the horizon) ----
    sched, hazards = _build_events(raw, contract, resolve_var, o, lean, horizon)

    plan = WorldExecutionPlan(
        question=question, outcome_contract=contract, as_of=parse_time(as_of),
        horizon_ts=parse_time(horizon),
        entities=list(raw.get("entities") or []), populations=list(raw.get("populations") or []),
        institutions=list(raw.get("institutions") or []), relations=list(raw.get("relations") or []),
        quantities=quantities, latents=latents, scheduled_events=sched, stochastic_hazards=hazards,
        accepted_mechanisms=accepted, candidate_experimental_mechanisms=experimental,
        rejected_mechanisms=rejected,
        structural_hypotheses=list(raw.get("structural_hypotheses") or []), actor_decisions=actor_decisions,
        mechanism_choices=[c.as_dict() for c in choices], fallbacks_used=fallbacks_used,
        support_grade=support, interpretations=interpretations,
        omissions=list(raw.get("omitted") or []), degraded=degraded,
        fidelity_plan=_fidelity_plan(raw, n_budget),
        uncertainty_plan={"latents": len(latents), "hazards": len(hazards),
                          "structural_hypotheses": len(raw.get("structural_hypotheses") or [])},
        compute_plan={"n_particles": _fidelity_plan(raw, n_budget)["n_particles"]},
        provenance={"prompt_hash": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                    "compiler_version": COMPILER_VERSION, "seed": seed,
                    "rationale": str(raw.get("rationale", ""))[:400], "scenario": scenario,
                    "evidence_basis": evidence_basis, "evidence_bundle_hash": bundle_hash,
                    "readout_repaired": readout_repaired, "outcome_lean": lean,
                    "causal_dependencies": (raw.get("causal_dependencies")
                                            if isinstance(raw.get("causal_dependencies"), dict) else {}),
                    "production_registry_selection": mechanism_selection,
                    "per_process_selection": {p: {"selected": (s.get("selected") or {}).get("family_id"),
                                                  "status": (s.get("selected") or {}).get("status"),
                                                  "competing": [c["family_id"] for c in s.get("competing", [])],
                                                  "n_candidates": s.get("n_candidates", 0)}
                                              for p, s in per_process.items()},
                    "mechanism_composition": composition,
                    "institution_selection": institution_selection,
                    "repairs": raw.get("_repairs", [])})
    if persist:
        try:
            _persist_compilation(plan, question, as_of)
        except Exception:
            pass                                             # persistence must never break compilation
    return plan


def _scenario_descriptor(raw: dict) -> dict:
    return {"domain": str(raw.get("domain", "") or ""),
            "population_kind": str(raw.get("population_kind", "") or ""),
            "time_scale": str(raw.get("time_scale", "") or ""),
            "available_state": ([k for k, v in (("network", raw.get("relations")),
                                                ("entities", raw.get("entities")),
                                                ("populations", raw.get("populations")),
                                                ("institutions", raw.get("institutions")),
                                                ("quantities", raw.get("quantities"))) if v]),
            "available_data": list(raw.get("available_data") or []),
            "institutional": bool(raw.get("institutions"))}


def _score_production_registry(scenario: dict) -> dict:
    try:
        from swm.world_model_v2.registry import load_registry
        from swm.world_model_v2.registry.applicability import rank_mechanisms
        return rank_mechanisms(load_registry(), scenario)
    except Exception as e:
        return {"selected": [], "rejected": [], "note": f"production registry unavailable: {e}"}


def _select_mechanisms_per_process(scenario: dict, processes: list) -> tuple:
    """Phase 6 per-process selection + composition. For each required causal process, select the family
    that ANSWERS it (registry.select_for_process), then compose the selections (double-counting detection,
    competing hypotheses, precedence). Returns ({process: selection}, composition_plan_dict). Degrades
    gracefully to an empty map (→ tier-6 generic) if the production registry is unavailable."""
    try:
        from swm.world_model_v2.registry import load_registry, select_for_process
        from swm.world_model_v2.registry.composition import compose
        store = load_registry()
        per_process, sels = {}, []
        for proc in processes:
            if proc == "outcome_resolution":
                continue
            r = select_for_process(store, proc, scenario)
            per_process[proc] = r
            sels.append(r)
        ts = {fid: store.records[fid].temporal_scale for fid in store.records}
        comp = compose(sels, time_scales=ts).as_dict()
        return per_process, comp
    except Exception as e:
        return {}, {"error": f"composition unavailable: {e}", "ordered": [], "competing": [],
                    "double_counting": [], "conflicts": []}


def _select_institutions(raw: dict, processes: list, as_of: str) -> dict:
    """Phase 10 — select evidence-backed institutions for institutional causal processes, BY CAUSAL NEED
    (as-of + jurisdiction), never keyword routing. Returns {process: selection} for institutional processes
    present in the plan. Degrades to {} if the institution registry is unavailable."""
    try:
        from swm.world_model_v2.institutions_v2 import load_store, select_institution
        from swm.world_model_v2.institutions_v2.compile import INSTITUTIONAL_PROCESSES
        inst_procs = [p for p in processes if p in INSTITUTIONAL_PROCESSES]
        if not inst_procs and not raw.get("institutions"):
            return {}
        store = load_store()
        jur = str(raw.get("jurisdiction", "") or raw.get("organization", "") or "")
        scenario = {"jurisdiction": jur, "organization": raw.get("organization", "")}
        out = {}
        for p in (inst_procs or ["issue_formal_decision"]):
            sel = select_institution(store, p, scenario, as_of=as_of, jurisdiction=jur)
            out[p] = {"family": sel.family_id, "template": sel.template_id, "tier": sel.tier,
                      "support_grade": sel.support_grade, "transported": sel.transported,
                      "missing_evidence": sel.missing_evidence}
        return out
    except Exception as e:
        return {"error": f"institution selection unavailable: {e}"}


def _build_events(raw, contract, resolve_var, o, lean, horizon):
    from swm.world_model_v2.events import event_type_registered
    sched = []
    for ev in raw.get("scheduled_events") or []:
        et = str(ev.get("etype", ""))
        if not et:
            continue
        if et == "resolve_outcome":
            continue                                          # the canonical terminal resolver is authoritative
        if not event_type_registered(et):
            register_event_type(et, scheduling="scheduled", validated=False,
                                parameter_source="compiler-proposed")
        try:
            ts = parse_time(ev.get("at"))
        except ValueError:
            continue
        sched.append({"etype": et, "ts": ts, "participants": list(ev.get("participants") or []),
                      "payload": dict(ev.get("payload") or {})})
    # Compiler-produced actor decisions use the same event queue as every other mechanism.
    # Numeric probabilities are intentionally not part of this semantic proposal contract.
    existing = {(e["etype"], tuple(e["participants"]), e["ts"]) for e in sched}
    for dec in raw.get("actor_decisions") or []:
        if not isinstance(dec, dict) or not dec.get("actor"):
            continue
        try:
            ts = parse_time(dec.get("at"))
        except (ValueError, TypeError):
            ts = parse_time(horizon) - 2.0
        key = ("decision_opportunity", (str(dec["actor"]),), ts)
        if key in existing:
            continue
        payload = {k: v for k, v in dec.items() if k not in ("actor", "at", "role")}
        payload["actor_role"] = str(dec.get("role", ""))
        sched.append({"etype": "decision_opportunity", "ts": ts,
                      "participants": [str(dec["actor"])], "payload": payload})
        existing.add(key)
    hazards = []
    for h in (raw.get("hazards") or []):
        if not isinstance(h, dict):
            continue
        het = str(h.get("etype", "distraction"))
        if not event_type_registered(het):
            register_event_type(het, scheduling="hazard", validated=False,
                                parameter_source="compiler-proposed")
        hazards.append({"etype": het, "rate_per_day": max(0.0, float(h.get("rate_per_day", 0.0) or 0.0)),
                        "participants": list(h.get("participants") or [])})
    # the canonical terminal resolution event — fires at the horizon so the readout always binds
    h_ts = parse_time(horizon)
    sched.append({"etype": "resolve_outcome", "ts": h_ts - 1.0, "participants": [],
                  "payload": {"outcome_var": resolve_var, "family": contract.family,
                              "options": contract.options, "lean": lean,
                              "lo": (raw.get("outcome") or {}).get("lo"),
                              "hi": (raw.get("outcome") or {}).get("hi")}})
    return sched, hazards


def _persist_compilation(plan: WorldExecutionPlan, question: str, as_of: str):
    """B10: persist every compilation attempt (append-only) for reproducibility + forensic review."""
    from pathlib import Path
    root = Path("experiments/results/compiler_attempts")
    root.mkdir(parents=True, exist_ok=True)
    rec = {"question": question, "as_of": as_of, "plan_hash": plan.plan_hash(),
           "compiler_version": COMPILER_VERSION, "support_grade": plan.support_grade,
           "degraded": plan.degraded, "n_entities": len(plan.entities),
           "n_institutions": len(plan.institutions), "n_populations": len(plan.populations),
           "n_latents": len(plan.latents), "n_structural_hypotheses": len(plan.structural_hypotheses),
           "accepted_mechanisms": [m["mech_id"] for m in plan.accepted_mechanisms],
           "fallbacks_used": plan.fallbacks_used, "mechanism_choices": plan.mechanism_choices,
           "readout_var": plan.outcome_contract.readout_var, "omissions": plan.omissions,
           "provenance": plan.provenance}
    fn = root / f"{plan.plan_hash()}.json"
    fn.write_text(json.dumps(rec, indent=1, default=str))


def recompile(plan: WorldExecutionPlan, *, llm, new_evidence: str, reason: str) -> WorldExecutionPlan:
    """Dynamic recompilation preserving history: a NEW versioned plan whose provenance chains to the old."""
    new = compile_world(plan.question, llm=llm, evidence=new_evidence,
                        as_of=_time.strftime("%Y-%m-%d", _time.gmtime(plan.as_of)),
                        horizon=_time.strftime("%Y-%m-%d", _time.gmtime(plan.horizon_ts)))
    new.version = plan.version + 1
    new.parent_version = plan.version
    new.provenance["recompiled_because"] = reason[:200]
    return new
