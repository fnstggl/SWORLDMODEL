"""Phase 10 — universal institutional data model: FAMILY → TEMPLATE → INSTANCE, plus EVIDENCE and RULE.

Strict three-layer separation (Part 1):

  InstitutionFamily   a reusable STRUCTURAL pattern (hierarchical approval, collective vote body, legislative
                      process, court, agency, board, queue, …) — roles, authority relations, permitted
                      actions, stage/threshold/deadline/resource/appeal SEMANTICS. NOT one real org.
  InstitutionTemplate one REAL institution / procedural regime for a DEFINED PERIOD — evidence-backed,
                      temporally versioned (valid_from / valid_to, amendments, supersession).
  InstitutionInstance a SCENARIO-SPECIFIC executable instance — a template bound to actual actors, roles,
                      a matter, a current stage, resources, a clock, an as-of date, and posterior particles.

Everything is evidence-anchored: a production rule must carry an EvidenceRecord with source provenance and
temporal validity. The LLM may PROPOSE a candidate rule but cannot ESTABLISH one — a rule with no verifiable
source stays `unverified` and cannot promote (see evidence.py deterministic validation). Nothing here
fabricates legal text, thresholds, membership, or authority.

Lifecycle statuses (Part 26), enforced by promotion gates in store.py — never collapsed into one "supported":
  proposed → evidence_encoded → structurally_implemented → executable → locally_reconstructed →
  historically_replayed → cross_institution_tested → production_eligible, with domain_restricted /
  quarantined / rejected.
"""
from __future__ import annotations

import hashlib
import time as _time
from dataclasses import asdict, dataclass, field

STATUSES = ("proposed", "evidence_encoded", "structurally_implemented", "executable",
            "locally_reconstructed", "historically_replayed", "cross_institution_tested",
            "production_eligible", "domain_restricted", "quarantined", "rejected")

FAMILY_CATEGORIES = ("hierarchical_approval", "collective_vote_body", "legislative_process",
                     "bicameral_legislature", "executive_veto", "administrative_agency",
                     "adjudicative_court", "appellate_hierarchy", "corporate_board",
                     "committee_system", "electoral_administration", "regulatory_review",
                     "grant_or_procurement", "benefits_eligibility", "moderation_appeals",
                     "queue_capacity_service", "contract_approval", "budget_allocation",
                     "platform_governance", "coalition_formation", "direct_democracy")

#: authority types on the authority graph (Part 5)
AUTHORITY_TYPES = ("appoint", "remove", "recommend", "advise", "approve", "final_decision", "veto",
                   "amend", "agenda_control", "enforce", "appellate", "delegate", "temporary",
                   "emergency")

#: source types for institutional evidence (Part 2). Official > unofficial.
OFFICIAL_SOURCE_TYPES = ("constitution", "statute", "regulation", "administrative_code",
                         "rules_of_procedure", "court_rule", "standing_order", "bylaws", "charter",
                         "election_manual", "agency_guidance", "platform_policy", "minutes", "agenda",
                         "vote_record", "docket", "case_history", "committee_report", "executive_order",
                         "budget_document", "procurement_rule", "service_standard", "enforcement_record",
                         "appellate_outcome", "historical_process_data", "archival_version")
UNOFFICIAL_SOURCE_TYPES = ("secondary_summary", "news", "encyclopedia", "commentary")


class InstitutionError(ValueError):
    pass


def now_iso() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def _hash(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class EvidenceRecord:
    """One official institutional evidence record (Part 2). `verified` is set by deterministic checks +
    (for numeric rules) core-agent verification against the primary source — NOT by the LLM."""
    source_id: str
    source_type: str                          # OFFICIAL_/UNOFFICIAL_SOURCE_TYPES
    issuing_authority: str
    title: str
    jurisdiction: str = ""
    institution: str = ""
    publication_date: str = ""
    effective_date: str = ""                  # when the rule took effect
    supersession_date: str = ""               # when superseded/repealed ("" = still in force)
    retrieval_date: str = ""
    citation: str = ""                         # statute cite / rule number / docket / stable URL / archive id
    section: str = ""                          # exact relevant section
    extracted_text: str = ""                   # the verbatim source span
    interpreted_rule: str = ""                 # human-readable rule
    hierarchy_level: int = 0                    # 0=constitution highest; larger = lower authority
    amends: str = ""                            # source_id this amends
    conflicts_with: str = ""
    confidence: float = 1.0
    official: bool = True
    verified: bool = False
    content_hash: str = ""

    def __post_init__(self):
        if self.source_type not in OFFICIAL_SOURCE_TYPES + UNOFFICIAL_SOURCE_TYPES:
            raise InstitutionError(f"{self.source_id}: unknown source_type {self.source_type!r}")
        self.official = self.source_type in OFFICIAL_SOURCE_TYPES
        self.content_hash = self.content_hash or _hash(
            [self.source_id, self.section, self.extracted_text, self.effective_date])


@dataclass
class RuleRecord:
    """A typed institutional rule with its evidence span + temporal validity. `kind` reuses the executable
    kinds in institutions.EXECUTABLE_RULE_KINDS plus Phase-10 structural kinds (authority/stage/threshold/
    appeal/queue). `params` is the typed formalization; `evidence_id` links to an EvidenceRecord."""
    rule_id: str
    kind: str
    params: dict = field(default_factory=dict)
    evidence_id: str = ""                       # "" = no source → cannot be production (unverified)
    effective_date: str = ""
    supersession_date: str = ""
    ambiguity: str = ""                         # recorded interpretation risk
    alternatives: list = field(default_factory=list)   # competing formalizations (Part 13)
    verified: bool = False

    def active_at(self, as_of: str) -> bool:
        return _date_in_range(as_of, self.effective_date, self.supersession_date)


@dataclass
class Role:
    role_id: str
    title: str
    selection_rule: str = ""                    # how the role is filled (election/appointment/…)
    term: str = ""
    count: str = ""                             # e.g. "1", "9", "435", "variable"


@dataclass
class AuthorityEdge:
    """A typed authority relation on the authority graph (Part 5)."""
    holder_role: str                            # role that holds the authority
    authority: str                              # AUTHORITY_TYPES
    over: str = ""                              # role/matter/action the authority applies to
    subject_matter: list = field(default_factory=list)
    limits: str = ""
    delegable: bool = False


@dataclass
class Stage:
    """A node in the procedural stage graph (Part 8)."""
    stage_id: str
    entry_conditions: list = field(default_factory=list)
    authorized_roles: list = field(default_factory=list)
    permitted_actions: list = field(default_factory=list)
    decision_rule: str = ""                      # threshold spec id / "" (deterministic transition)
    outcomes: list = field(default_factory=list) # possible outcome labels
    next_stages: dict = field(default_factory=dict)  # {outcome: next_stage_id}
    deadline_days: float | None = None
    appealable: bool = False
    terminal: bool = False


@dataclass
class InstitutionFamily:
    """Reusable structural pattern (Part 1.1) — NOT one real org."""
    family_id: str
    version: str
    category: str                               # FAMILY_CATEGORIES
    title: str
    causal_question: str
    roles: list = field(default_factory=list)          # [Role]
    authority: list = field(default_factory=list)      # [AuthorityEdge]
    permitted_action_types: list = field(default_factory=list)
    information_rights: dict = field(default_factory=dict)   # {info_class: [roles that may observe]}
    stages: list = field(default_factory=list)         # [Stage]
    threshold_semantics: list = field(default_factory=list)  # supported threshold kinds
    resource_semantics: list = field(default_factory=list)
    deadline_semantics: list = field(default_factory=list)
    enforcement_semantics: list = field(default_factory=list)
    appeal_semantics: list = field(default_factory=list)
    invariants: list = field(default_factory=list)
    preconditions: list = field(default_factory=list)
    exclusion_conditions: list = field(default_factory=list)
    ambiguity_points: list = field(default_factory=list)
    composition_rules: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    answers_processes: list = field(default_factory=list)   # institutional causal processes (Part 17)
    code_ref: str = ""
    test_ref: str = ""
    status: str = "proposed"
    status_reason: str = ""
    created_at: str = ""

    def __post_init__(self):
        if self.category not in FAMILY_CATEGORIES:
            raise InstitutionError(f"{self.family_id}: unknown category {self.category!r}")
        if self.status not in STATUSES:
            raise InstitutionError(f"{self.family_id}: bad status {self.status!r}")

    def executable(self) -> bool:
        try:
            import importlib
            mod, _, name = self.code_ref.partition(":")
            return callable(getattr(importlib.import_module(mod), name))
        except Exception:
            return False


@dataclass
class InstitutionTemplate:
    """One REAL institution / procedural regime for a defined period (Part 1.2) — temporally versioned,
    evidence-backed. Reuses a family's structure; binds real roles/authority/thresholds from evidence."""
    template_id: str
    family_id: str
    family_version: str
    official_name: str
    jurisdiction: str
    organization: str = ""
    valid_from: str = ""
    valid_to: str = ""                          # "" = still in force
    roles: list = field(default_factory=list)          # [Role]
    authority: list = field(default_factory=list)      # [AuthorityEdge]
    stages: list = field(default_factory=list)         # [Stage]
    rules: list = field(default_factory=list)          # [RuleRecord]
    evidence: list = field(default_factory=list)       # [EvidenceRecord]
    thresholds: dict = field(default_factory=dict)     # {decision: threshold spec}
    quorums: dict = field(default_factory=dict)
    deadlines: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    appeals: dict = field(default_factory=dict)
    informal_practice: list = field(default_factory=list)   # evidence of practice ≠ formal rule (Part 12)
    discretion: list = field(default_factory=list)
    procedural_uncertainty: list = field(default_factory=list)
    interpretation_history: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    status: str = "proposed"
    status_reason: str = ""
    version: str = "1.0.0"
    content_hash: str = ""
    created_at: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise InstitutionError(f"{self.template_id}: bad status {self.status!r}")

    def active_at(self, as_of: str) -> bool:
        return _date_in_range(as_of, self.valid_from, self.valid_to)

    def rules_as_of(self, as_of: str) -> list:
        """The rules in force at `as_of` — the as-of versioning gate (Part 3). Post-as-of rules excluded."""
        return [r for r in self.rules if r.active_at(as_of)]

    def has_official_evidence(self) -> bool:
        return any(e.official and e.verified for e in self.evidence)

    def compute_hash(self):
        self.content_hash = _hash([self.template_id, self.version,
                                   [r.rule_id for r in self.rules], self.valid_from, self.valid_to])
        return self.content_hash


@dataclass
class InstitutionInstance:
    """Scenario-specific executable instance (Part 1.3). Built by the compiler at runtime; bound to actual
    actors, a matter, a current stage, resources, a clock, an as-of date, and posterior particles."""
    scenario_id: str
    template_id: str
    template_version: str
    as_of: str
    role_bindings: dict = field(default_factory=dict)   # {role_id: [actor_id]}
    actor_bindings: dict = field(default_factory=dict)  # {actor_id: role_id}
    matter: dict = field(default_factory=dict)          # the matter/case (Part 7)
    current_stage: str = ""
    agenda_status: str = ""
    deadlines: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    queue_state: dict = field(default_factory=dict)
    information_by_actor: dict = field(default_factory=dict)   # {actor: [observable info classes]}
    active_rules: list = field(default_factory=list)    # RuleRecord ids active as-of
    competing_models: list = field(default_factory=list)  # [{model_id, weight, rule_overrides}] (Part 13)
    posterior_weights: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    evidence_dependencies: list = field(default_factory=list)
    support_grade: str = ""
    transport_risk: str = "high"
    fallback_tier: int = 0
    seed: int = 0
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ date helpers (as-of versioning)
def _to_ymd(s: str):
    if not s:
        return None
    s = str(s)[:10]
    try:
        y, m, d = s.split("-")
        return (int(y), int(m), int(d))
    except Exception:
        return None


def _date_in_range(as_of: str, frm: str, to: str) -> bool:
    """True iff frm <= as_of < to (to empty = open-ended). Missing frm = treated as always-started."""
    a = _to_ymd(as_of)
    if a is None:
        return True                              # no as-of constraint → do not filter
    f = _to_ymd(frm)
    t = _to_ymd(to)
    if f is not None and a < f:
        return False
    if t is not None and a >= t:
        return False
    return True
