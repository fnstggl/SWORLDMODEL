"""Phase 10 — formal collective-decision mechanics (Part 9): thresholds, quorums, votes, vetoes, override.

`evaluate_decision(spec, votes, ...)` is deterministic and evidence-parameterized: the THRESHOLD/QUORUM come
from the institutional template's verified rules (never the LLM), the VOTES come from Phase 3 state / Phase 6
actor policy (never arbitrary constants), and this module only computes procedural validity + aggregate
effect. It distinguishes the adversarial cases Phase 10 must get right: quorum vs threshold, majority-of-
present vs majority-of-all-members, abstention vs absence, recusal, veto vs override.
"""
from __future__ import annotations

from dataclasses import dataclass, field

THRESHOLD_KINDS = ("simple_majority", "absolute_majority", "supermajority", "plurality",
                   "unanimous", "weighted_majority", "supermajority_of_quorum")


@dataclass
class ThresholdSpec:
    """A verified decision rule from evidence. base='present' → majority of those voting (quorum present);
    base='all_members' → majority of the full body (e.g. 'absolute majority'). fraction for supermajority."""
    kind: str                                   # THRESHOLD_KINDS
    fraction: float = 0.5                        # 0.5 majority, 2/3 supermajority, …
    base: str = "present"                        # 'present' (of those voting) | 'all_members'
    quorum_fraction: float = 0.5                 # quorum = this fraction of eligible (Art I §5: majority)
    quorum_base: str = "eligible"
    tie_breaker_role: str = ""                   # e.g. VP breaks Senate ties
    evidence_id: str = ""

    def __post_init__(self):
        if self.kind not in THRESHOLD_KINDS:
            raise ValueError(f"unknown threshold kind {self.kind!r}")


@dataclass
class DecisionResult:
    passed: bool
    quorum_met: bool
    yes: float
    no: float
    abstain: float
    present: float
    eligible: float
    needed: float
    base_count: float
    rule: str
    tie_broken_by: str = ""
    vetoed: bool = False
    overridden: bool = False
    reasons: list = field(default_factory=list)

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


def _majority_count(n: float) -> float:
    """Strict majority of n members = floor(n/2)+1 (51 of 100). Distinguishes 'majority of all members'
    from 'half'. For weighted/fractional n, falls back to n/2 + a hair."""
    import math
    if abs(n - round(n)) < 1e-9:
        return math.floor(round(n) / 2) + 1
    return n / 2.0 + 1e-6


def evaluate_decision(spec: ThresholdSpec, votes: dict, *, eligible: list, weights: dict | None = None,
                      recused: set | None = None, tie_break: str | None = None) -> DecisionResult:
    """votes: {voter: 'yes'|'no'|'abstain'}. eligible: full eligible membership. recused voters cannot count
    toward quorum or the vote. Present = those who cast any vote (yes/no/abstain) and are not recused; an
    ABSENT member is simply not in `votes` (distinct from an abstention, which IS present)."""
    w = weights or {}
    recused = recused or set()
    elig = [e for e in eligible if e not in recused]
    n_elig = sum(w.get(e, 1.0) for e in elig)
    cast = {v: c for v, c in votes.items() if v in elig}          # only eligible, non-recused votes count
    present = sum(w.get(v, 1.0) for v in cast)                    # abstain counts as present
    yes = sum(w.get(v, 1.0) for v, c in cast.items() if c == "yes")
    no = sum(w.get(v, 1.0) for v, c in cast.items() if c == "no")
    abstain = sum(w.get(v, 1.0) for v, c in cast.items() if c == "abstain")

    # quorum: a MAJORITY quorum (Art I §5) is floor(n/2)+1, not n/2 — 51 of 100, never 50.
    quorum_needed = (_majority_count(n_elig) if abs(spec.quorum_fraction - 0.5) < 1e-9
                     else spec.quorum_fraction * n_elig)
    quorum_met = present >= quorum_needed - 1e-9
    reasons = []
    if not quorum_met:
        reasons.append(f"quorum not met: {present} present < {round(quorum_needed, 2)} needed")

    base = n_elig if spec.base == "all_members" else (yes + no)   # 'present' base excludes abstentions
    if spec.kind == "plurality":
        needed = no                              # yes must simply exceed no
        passed = yes > no
    elif spec.kind == "unanimous":
        needed = base
        passed = no == 0 and yes > 0
    elif abs(spec.fraction - 0.5) < 1e-9:
        # a MAJORITY: 'majority of all members' (base=all_members) needs floor(n/2)+1; 'majority of those
        # present' needs strictly more yes than no. This is the adversarial distinction Phase 10 must keep.
        if spec.base == "all_members":
            needed = _majority_count(base)
            passed = yes >= needed
        else:
            needed = base / 2.0
            passed = yes > needed
    else:
        needed = spec.fraction * base            # supermajority: yes >= fraction of base
        passed = yes >= needed - 1e-9

    tie_by = ""
    if abs(yes - no) < 1e-9 and (tie_break or spec.tie_breaker_role):
        # a tie-breaker (e.g. VP) resolves; recorded explicitly
        tie_by = tie_break or spec.tie_breaker_role
        passed = True
        reasons.append(f"tie broken by {tie_by}")

    passed = passed and quorum_met
    return DecisionResult(passed=bool(passed), quorum_met=quorum_met, yes=yes, no=no, abstain=abstain,
                          present=present, eligible=n_elig, needed=round(float(needed), 3),
                          base_count=round(float(base), 3),
                          rule=f"{spec.kind}@{spec.fraction} base={spec.base}", tie_broken_by=tie_by,
                          reasons=reasons)


def apply_veto_and_override(decision: DecisionResult, *, vetoed: bool,
                            override_spec: ThresholdSpec, override_votes: dict, eligible: list,
                            **kw) -> DecisionResult:
    """Executive veto + legislative override (Art I §7). If the bill passed and is vetoed, it only becomes
    law on a successful override vote (2/3 of each chamber, of a quorum). Returns the final result."""
    if not decision.passed or not vetoed:
        decision.vetoed = vetoed
        return decision
    ov = evaluate_decision(override_spec, override_votes, eligible=eligible, **kw)
    decision.vetoed = True
    decision.overridden = ov.passed
    decision.passed = ov.passed
    decision.reasons.append(f"vetoed; override {'succeeded' if ov.passed else 'failed'} "
                            f"({ov.yes}/{round(ov.needed, 1)} needed, quorum={ov.quorum_met})")
    return decision
