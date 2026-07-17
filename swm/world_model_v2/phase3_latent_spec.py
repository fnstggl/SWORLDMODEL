"""Typed latent-variable specification + claim tagging — Phase 3 (Part A + LLM inference contract).

A `LatentVariableSpec` is the typed contract for one inferred hidden variable: its measurable interpretation,
support, prior source, candidate observation models, the mechanism that CONSUMES it, its identifiability and
sensitivity. Unmeasurable / consumer-less proposals are downgraded to broad priors (never silently minted as
a precise value).

The LLM is the semantic mapping layer ONLY: for each Phase-2 claim it proposes QUALITATIVE tags — the
outcome direction the claim bears on, which structural hypotheses it supports, an evidential strength bucket,
whether the statement is strategic, and which latent it informs. It may NOT mint likelihoods, rates, weights
or posteriors — those come from registered observation models (phase3_observation.py). Every tag is validated
against the claim (the claim must exist, be span-verified, and be admissible).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

SUPPORT_TYPES = ("binary", "categorical", "ordinal", "bounded_continuous", "positive_continuous", "count",
                 "duration", "vector", "correlated_multivariate", "discrete_structural")
STRENGTHS = ("weak", "moderate", "strong")
DIRECTIONS = ("supports_yes", "supports_no", "neutral")
CLAIM_TAG_PROMPT_VERSION = "phase3-claim-tag-1.0"


@dataclass
class LatentVariableSpec:
    variable_id: str
    definition: str                                   # canonical semantic definition
    measurable_interpretation: str                    # how it would be measured operationally
    support_type: str = "bounded_continuous"          # SUPPORT_TYPES
    units: str = ""
    lo: float = 0.0
    hi: float = 1.0
    categories: list = field(default_factory=list)
    scope: str = "world"                              # actor/relationship/population/institution/world id
    causal_parents: list = field(default_factory=list)
    causal_children: list = field(default_factory=list)
    evidence_claim_ids: list = field(default_factory=list)
    observation_models: list = field(default_factory=list)   # candidate registered model names
    prior_source: str = "generic_weakly_informative"
    prior_version: str = "phase3-prior-1.0"
    inference_method: str = "particle_assimilation"
    posterior_representation: str = "beta"            # beta | particles | dirichlet | normal
    sensitivity: float = 0.5
    identifiability: str = "unknown"                  # identified | partially_identified | unidentified
    provenance_status: str = "inferred"
    consumed_by: list = field(default_factory=list)   # mechanisms/paths that read it (ornamental if empty)
    actor_visibility: str = "public"
    transport_risk: str = "none"
    unsupported_assumptions: list = field(default_factory=list)

    def measurable(self) -> bool:
        """A spec is measurable (and thus production-usable) iff it has a support type, an observation model,
        and a declared consumer. Unmeasurable proposals are kept for audit but marked ornamental."""
        return bool(self.support_type in SUPPORT_TYPES and self.observation_models and self.consumed_by)

    def as_dict(self):
        return asdict(self)


@dataclass
class ClaimTag:
    """The LLM's QUALITATIVE reading of one claim (no numbers). The numeric likelihood is produced downstream
    by a registered observation model keyed on these tags."""
    claim_id: str
    outcome_direction: str = "neutral"                # DIRECTIONS — which way the claim bears on the outcome
    supports_hypotheses: list = field(default_factory=list)   # hypothesis ids the claim supports
    opposes_hypotheses: list = field(default_factory=list)
    strength: str = "moderate"                        # STRENGTHS — evidential strength bucket
    is_strategic: bool = False                        # a strategic/public statement (discounted likelihood)
    informs_variable: str = "outcome_rate"
    dependence_group: str = ""
    reliability: float = 0.8                           # from source type; NOT LLM-minted (set from claim)

    def as_dict(self):
        return asdict(self)


_TAG_PROMPT = """You are reading verified as-of evidence claims to infer a hidden social outcome. For EACH
claim, give a QUALITATIVE reading only (no numbers, no probabilities). Reply ONLY JSON:
{{"tags": [{{"claim_id": "...",
  "outcome_direction": "supports_yes|supports_no|neutral",
  "supports_hypotheses": ["<hypothesis id>", ...], "opposes_hypotheses": ["<hypothesis id>", ...],
  "strength": "weak|moderate|strong",
  "is_strategic": true|false}}]}}
`supports_yes` = the claim makes the affirmative answer to the question MORE likely; `supports_no` = less
likely; `neutral` = background/irrelevant to direction. `is_strategic` = a public statement that may be
posturing rather than a costly signal. Judge only from the claim text. Do NOT invent claims.

QUESTION: {question}   (affirmative = a YES answer)
STRUCTURAL HYPOTHESES: {hypotheses}
CLAIMS:
{claims}"""

#: source-type → base reliability (fixed table; NOT LLM-minted). Costly/official signals > opinion/social.
_SOURCE_RELIABILITY = {"official_filing": 0.9, "wire": 0.82, "news": 0.75, "official_record": 0.88,
                       "dataset": 0.9, "wikipedia_revision": 0.8, "market": 0.85, "social": 0.55,
                       "user_provided": 0.7, "prior_world_state": 0.8, "unknown": 0.6}
#: claim-class → strategic-by-default (an actor's public statement is strategic unless a costly action)
_STRATEGIC_CLASSES = {"actor_statement", "promise", "opinion", "forecast"}


def tag_claims(question: str, bundle, plan, *, llm) -> list:
    """LLM-tag the bundle's INCLUDED claims (qualitative). Returns [ClaimTag]; reliability + strategic default
    are set from the claim's source type / class (fixed tables), then the LLM may only override is_strategic
    and the qualitative direction/strength. Claims not span-verified or not included are skipped."""
    from swm.engine.grounding import parse_json
    included = {c["claim_id"]: c for c in bundle.included_claims()}
    if not included or llm is None:
        return []
    src_type = {d["id"]: d.get("source_type", "news") for d in bundle.documents}
    hyps = [h.get("id") for h in (plan.structural_hypotheses or []) if isinstance(h, dict)]
    claims_text = "\n".join(
        f"- [{cid}] ({c['claim_class']}) {c['subject']} {c['predicate']} {c.get('object', '')} "
        f"{c.get('value', '')} :: \"{c.get('supporting_span', '')[:140]}\""
        for cid, c in list(included.items())[:24])
    prompt = _TAG_PROMPT.format(question=question, hypotheses=json.dumps(hyps), claims=claims_text)
    try:
        raw = parse_json(llm(prompt)) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    # model-shape tolerance: some models return the bare JSON array instead of {"tags": [...]}
    tags = raw if isinstance(raw, list) else (raw.get("tags") or []) if isinstance(raw, dict) else []
    by_id = {t.get("claim_id"): t for t in tags if isinstance(t, dict)}
    out = []
    for cid, c in included.items():
        t = by_id.get(cid, {})
        st = src_type.get(c.get("source_id", ""), "news")
        rel = _SOURCE_RELIABILITY.get(st, 0.6)
        strategic_default = c.get("claim_class") in _STRATEGIC_CLASSES
        direction = t.get("outcome_direction") if t.get("outcome_direction") in DIRECTIONS else "neutral"
        strength = t.get("strength") if t.get("strength") in STRENGTHS else "moderate"
        out.append(ClaimTag(
            claim_id=cid, outcome_direction=direction,
            supports_hypotheses=[h for h in (t.get("supports_hypotheses") or []) if h in hyps][:4],
            opposes_hypotheses=[h for h in (t.get("opposes_hypotheses") or []) if h in hyps][:4],
            strength=strength, is_strategic=bool(t.get("is_strategic", strategic_default)),
            informs_variable="outcome_rate", dependence_group=c.get("dependence_group", ""),
            reliability=rel))
    return out


def outcome_rate_spec(question: str, claim_ids: list) -> LatentVariableSpec:
    """The canonical scenario latent: the base rate of the AFFIRMATIVE outcome as of the question date.
    Consumed by the terminal resolver (its per-particle Bernoulli rate)."""
    return LatentVariableSpec(
        variable_id="outcome_rate",
        definition=f"P(affirmative answer) latent base rate for: {question[:120]}",
        measurable_interpretation="the long-run fraction of comparable situations resolving affirmative",
        support_type="bounded_continuous", lo=0.0, hi=1.0, units="probability",
        evidence_claim_ids=list(claim_ids)[:24], observation_models=["directional_rate"],
        prior_source="reference_class_or_generic", posterior_representation="beta",
        inference_method="beta_binomial_assimilation", sensitivity=1.0,
        consumed_by=["generic_outcome_prior:rate", "resolve_outcome:lean"], actor_visibility="public")


def structural_spec(plan) -> LatentVariableSpec | None:
    """The discrete structural latent over the plan's competing hypotheses. Consumed by particle stratification
    (each hypothesis is a particle stratum whose posterior weight sets its terminal mass)."""
    hyps = [h.get("id") for h in (plan.structural_hypotheses or []) if isinstance(h, dict)]
    if len(hyps) < 2:
        return None
    return LatentVariableSpec(
        variable_id="structural_hypothesis", definition="which competing causal structure holds",
        measurable_interpretation="the world structure most consistent with the admitted evidence",
        support_type="discrete_structural", categories=hyps, observation_models=["structural_detection"],
        prior_source="compiler_structural_prior", posterior_representation="dirichlet",
        inference_method="particle_assimilation", sensitivity=0.9,
        consumed_by=["materialize:_run_with_hypotheses:stratum_weight"], actor_visibility="public")
