"""LLM model-family diversity — reduce correlated failure without sacrificing strong coverage.

One underlying LLM family shares knowledge gaps, reasoning styles, cultural assumptions, salience
patterns and blind spots across every simulated actor. This module gives World Model V2 an
explicit, metadata-carrying model-family pool, deterministic traceable assignment of families to
(world particle × actor decision), the §17.3 accuracy rule (a weaker family may only ADD
adversarial coverage, never replace required strong-model particles), and honest monoculture
reporting when only one family is configured.

HARD RULES (enforced here + tests):
  * Two temperatures of one model are NOT two families (``FamilyIdentityError``). Family identity
    is (provider, model, version-lineage) — sampling temperature is a call parameter.
  * Families sharing a training LINEAGE (e.g. a chat model and a reasoner distilled from the same
    base) count as ONE lineage for monoculture purposes — a run served entirely by one lineage
    reports ``model_family_monoculture=true`` no matter how many model names it used.
  * Assignments are deterministic in (particle, actor) and PRESERVED through a branch; a
    mid-branch family switch is legal only as a RECORDED failure transition (§17.2/§19.1).
  * Actors are never described as independent minds. The honest phrase, used verbatim in
    reporting: "independently situated LLM actor instances with partially diversified model
    families."

The simulated person's state stays PORTABLE across families (§18): family routing wraps the
call, never the actor's qualitative schema; per-revision provenance records which family wrote
what so provider-style artifacts stay measurable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

FAMILY_SCHEMA = "model.family.v1"

#: capability tier — the §17.3 accuracy rule keys off this
STRENGTH_TIERS = ("strong", "weaker")

HONEST_ACTOR_LANGUAGE = ("independently situated LLM actor instances with partially "
                         "diversified model families")


class FamilyIdentityError(ValueError):
    """Raised when a registration tries to pass temperature/sampling variants off as families."""


def _hash(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class ModelFamily:
    """§17.1 metadata for one genuinely distinct configured family."""
    family_id: str
    provider: str
    model: str
    version: str = ""
    lineage: str = ""                     # training lineage key (monoculture is judged on this)
    capabilities: list = field(default_factory=list)
    context_limit: int = 0
    supported_schemas: list = field(default_factory=list)
    cost_per_mtok_usd: float = 0.0
    latency_class: str = ""               # "fast" | "medium" | "slow"
    availability: str = "unknown"         # "configured" | "unavailable" | "unknown"
    known_limitations: list = field(default_factory=list)
    strength_tier: str = "strong"         # STRENGTH_TIERS
    client: object = None                 # callable(prompt)->text; excluded from as_dict

    def __post_init__(self):
        if self.strength_tier not in STRENGTH_TIERS:
            raise ValueError(f"strength_tier must be one of {STRENGTH_TIERS}")
        self.lineage = self.lineage or f"{self.provider}:{self.model}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("client", None)
        return d


@dataclass
class FamilyPool:
    """The configured pool + deterministic assignment + monoculture reporting."""
    families: list = field(default_factory=list)          # [ModelFamily]
    assignment_log: list = field(default_factory=list)    # [{particle, actor, family, rule}]
    failure_transitions: list = field(default_factory=list)  # [{particle, actor, from, to, error, at}]

    # ------------------------------------------------------------------ registration
    def register(self, fam: ModelFamily) -> ModelFamily:
        for existing in self.families:
            if existing.provider == fam.provider and existing.model == fam.model:
                # same provider+model registered twice — only legal if it IS the same family;
                # differing only by sampling parameters is the exact forbidden move
                raise FamilyIdentityError(
                    f"{fam.provider}/{fam.model} is already registered as family "
                    f"{existing.family_id!r} — different temperatures/sampling settings of one "
                    "model are NOT different families (§17.1)")
            if existing.family_id == fam.family_id:
                raise FamilyIdentityError(f"duplicate family_id {fam.family_id!r}")
        self.families.append(fam)
        return fam

    # ------------------------------------------------------------------ views
    def configured(self) -> list:
        return [f for f in self.families if f.availability == "configured" and f.client is not None]

    def strong(self) -> list:
        return [f for f in self.configured() if f.strength_tier == "strong"]

    def weaker(self) -> list:
        return [f for f in self.configured() if f.strength_tier == "weaker"]

    def by_id(self, family_id: str) -> ModelFamily | None:
        for f in self.families:
            if f.family_id == family_id:
                return f
        return None

    def distinct_lineages(self) -> list:
        return sorted({f.lineage for f in self.configured()})

    def monoculture(self) -> bool:
        """True when the run is served by at most one training lineage (§17.4)."""
        return len(self.distinct_lineages()) <= 1

    # ------------------------------------------------------------------ §17.2 assignment
    def assign(self, *, particle_index: int, actor_id: str, high_sensitivity: bool = False,
               record: bool = True) -> str:
        """Deterministic, traceable family assignment for one (particle, actor).

        Primary coverage always comes from STRONG families; when several comparable strong
        families are configured, coherent particles are distributed across them by a stable
        hash (§17.2) — high-sensitivity actors therefore automatically span families. Weaker
        families never serve primary particles (§17.3); they are reached only through
        ``adversarial_extra_family``. With no configured family this raises — assignment can
        never invent a backend."""
        strong = self.strong()
        if not strong:
            raise RuntimeError("no configured strong model family — refusing to assign "
                               "(a weaker family cannot replace required strong coverage, §17.3)")
        if len(strong) == 1:
            fam, rule = strong[0], "single_strong_family"
        else:
            idx = int(_hash([actor_id, particle_index]), 16) % len(strong)
            fam, rule = sorted(strong, key=lambda f: f.family_id)[idx], \
                "stable_hash_across_comparable_strong_families"
        if record:
            self.assignment_log.append({"particle": particle_index, "actor": actor_id,
                                        "family": fam.family_id, "rule": rule,
                                        "high_sensitivity": bool(high_sensitivity)})
        return fam.family_id

    def adversarial_extra_family(self, *, exclude: str = "") -> str | None:
        """A weaker (or unused strong) family for ADDITIONAL adversarial hypothesis particles —
        added on top of, never instead of, required strong coverage (§17.3)."""
        for f in self.weaker():
            if f.family_id != exclude:
                return f.family_id
        for f in self.strong():
            if f.family_id != exclude:
                return f.family_id
        return None

    # ------------------------------------------------------------------ §19.1 failure routing
    def comparable_alternative(self, family_id: str) -> str | None:
        """The configured comparable (same strength tier) alternative for a failure transition,
        deterministic order. None when the pool has no alternative — the caller must then stop
        the branch rather than degrade psychology."""
        cur = self.by_id(family_id)
        if cur is None:
            return None
        peers = [f for f in self.configured()
                 if f.family_id != family_id and f.strength_tier == cur.strength_tier]
        peers.sort(key=lambda f: f.family_id)
        return peers[0].family_id if peers else None

    def record_failure_transition(self, *, particle_index: int, actor_id: str, from_family: str,
                                  to_family: str, error: str, at: float = 0.0) -> dict:
        rec = {"particle": particle_index, "actor": actor_id, "from": from_family,
               "to": to_family, "error": str(error)[:200], "at": at}
        self.failure_transitions.append(rec)
        return rec

    def call(self, family_id: str, prompt: str):
        fam = self.by_id(family_id)
        if fam is None or fam.client is None:
            raise RuntimeError(f"model family {family_id!r} is not configured/callable")
        return fam.client(prompt)

    # ------------------------------------------------------------------ reporting (§17.4 / §35.2)
    def report(self) -> dict:
        return {
            "schema": FAMILY_SCHEMA,
            "families": [f.as_dict() for f in self.families],
            "configured_families": [f.family_id for f in self.configured()],
            "distinct_lineages": self.distinct_lineages(),
            "model_family_monoculture": self.monoculture(),
            "monoculture_note": (
                "every simulated actor is served by one model lineage; actor instances share "
                "that lineage's knowledge, salience and blind spots — correlated failure risk "
                "is NOT diversified" if self.monoculture() else
                "coherent particles are distributed across distinct lineages; compare "
                "per-family decisions before trusting a recommendation"),
            "actor_independence_language": HONEST_ACTOR_LANGUAGE,
            "assignments": self.assignment_log[-200:],
            "n_assignments": len(self.assignment_log),
            "failure_transitions": self.failure_transitions[-50:],
        }

    def as_dict(self) -> dict:
        return self.report()


# ---------------------------------------------------------------------- construction
def default_family_pool(llm=None) -> FamilyPool:
    """Build the pool from what is ACTUALLY configured in this environment.

    The caller's ``llm`` (the canonical injected backend) becomes the primary strong family.
    Additional families register only when their credentials exist — nothing is fabricated.
    With exactly one lineage the pool is honest monoculture; the run continues and reports it."""
    pool = FamilyPool()
    if llm is not None:
        if os.environ.get("DEEPSEEK_API_KEY"):
            pool.register(ModelFamily(
                family_id="deepseek_v3", provider="deepseek", model="deepseek-chat",
                version=os.environ.get("SWM_DEEPSEEK_VERSION", ""), lineage="deepseek-v3",
                capabilities=["json", "long_context", "multilingual"], context_limit=64000,
                supported_schemas=["qualitative.actor.v1", "world.boundary.v1"],
                cost_per_mtok_usd=0.27, latency_class="medium",
                availability="configured",
                known_limitations=["single training lineage — correlated blind spots when "
                                   "serving every actor"],
                strength_tier="strong", client=llm))
        else:
            # an injected backend whose provider identity is unknown (tests, adapters): honest
            # generic labels — never claim a provider that was not verified
            pool.register(ModelFamily(
                family_id="primary_configured", provider="injected",
                model="configured-backend", lineage="injected-primary",
                availability="configured",
                known_limitations=["provider identity not verified from environment"],
                strength_tier="strong", client=llm))
    # optional additional families (register only when their keys are present AND a client can
    # be built without new dependencies). OpenRouter-served open-weight models qualify as
    # genuinely distinct lineages; absent credentials, the pool stays honest monoculture.
    if os.environ.get("OPENROUTER_API_KEY"):
        client = _openrouter_client(os.environ.get("SWM_ALT_FAMILY_MODEL",
                                                   "qwen/qwen-2.5-72b-instruct"))
        if client is not None:
            pool.register(ModelFamily(
                family_id="qwen_25", provider="openrouter", model="qwen/qwen-2.5-72b-instruct",
                lineage="qwen-2.5", capabilities=["json"], context_limit=32000,
                cost_per_mtok_usd=0.4, latency_class="medium", availability="configured",
                strength_tier="strong", client=client))
    return pool


def _openrouter_client(model: str):
    """Minimal OpenRouter chat client (stdlib only), or None when unusable."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return None
    import urllib.request

    def _call(prompt: str) -> str:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({"model": model, "temperature": 0.7,
                             "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]
    return _call
