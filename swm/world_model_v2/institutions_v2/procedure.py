"""Phase 10 — matters & agenda (Part 7), procedural stage engine (Part 8), resources & queues (Part 11).

The stage engine is a GENERAL executable procedural graph (no hardcoded legislative/court sequence): a
template supplies `Stage` nodes and outcome→next transitions, and `StageEngine.advance` moves a matter
through them, driven by deterministic rules or Phase-6/decision outcomes. Resources and queues are REAL
state constraints: a capacity-limited queue changes DECISION TIMING and completion probability (Part 11),
not a descriptive field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MATTER_TYPES = ("bill", "motion", "case", "application", "permit", "claim", "budget_request",
                "hiring_request", "compensation_request", "procurement", "appeal", "moderation_report",
                "policy_proposal", "contract", "petition", "election_certification", "enforcement_action")

AGENDA_OPS = ("file", "refer", "schedule", "prioritize", "table", "postpone", "withdraw", "amend",
              "consolidate", "sever", "dismiss", "expire", "resubmit")


@dataclass
class Matter:
    matter_id: str
    matter_type: str
    subject: str = ""
    filer: str = ""
    requested_action: str = ""
    filed_at: str = ""
    agenda_status: str = "unfiled"              # unfiled|filed|referred|scheduled|tabled|withdrawn|decided
    priority: float = 0.0
    stage: str = ""
    deadline: str = ""
    amendments: list = field(default_factory=list)
    objections: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    decision: str = ""
    appeal_status: str = ""
    history: list = field(default_factory=list)

    def log(self, event: str, **kw):
        self.history.append({"event": event, **kw})


@dataclass
class StageEngine:
    stages: dict = field(default_factory=dict)          # {stage_id: Stage}

    @classmethod
    def from_stages(cls, stage_list):
        return cls(stages={s.stage_id: s for s in stage_list})

    def get(self, stage_id):
        return self.stages.get(stage_id)

    def permitted_actions(self, stage_id) -> list:
        s = self.stages.get(stage_id)
        return list(s.permitted_actions) if s else []

    def authorized_roles(self, stage_id) -> list:
        s = self.stages.get(stage_id)
        return list(s.authorized_roles) if s else []

    def is_terminal(self, stage_id) -> bool:
        s = self.stages.get(stage_id)
        return bool(s and s.terminal)

    def next_stage(self, stage_id, outcome: str):
        """Deterministic transition: outcome → next stage id (or None if terminal / no transition)."""
        s = self.stages.get(stage_id)
        if not s:
            return None
        return s.next_stages.get(outcome) or s.next_stages.get("*")

    def advance(self, matter: Matter, outcome: str) -> tuple[str | None, bool]:
        """Move the matter to the next stage given an outcome. Returns (next_stage_id, terminal)."""
        cur = matter.stage
        nxt = self.next_stage(cur, outcome)
        if nxt is None:
            matter.log("stage_terminal", stage=cur, outcome=outcome)
            return None, True
        matter.log("stage_transition", frm=cur, to=nxt, outcome=outcome)
        matter.stage = nxt
        return nxt, self.is_terminal(nxt)

    def validate_acyclic(self) -> list:
        """Detect circular stage transitions that are not explicitly appeal/remand loops (Part 4 check)."""
        problems = []
        for sid, s in self.stages.items():
            for outcome, nxt in s.next_stages.items():
                if nxt == sid and outcome not in ("reconsider", "remand", "resubmit"):
                    problems.append(f"stage {sid} self-loops on non-remand outcome {outcome!r}")
                if nxt not in self.stages and nxt != "":
                    problems.append(f"stage {sid} → unknown stage {nxt!r}")
        return problems


# ------------------------------------------------------------------ resources & queues (Part 11)
QUEUE_DISCIPLINES = ("fifo", "priority", "statutory_priority", "emergency_priority", "risk_based")


@dataclass
class ResourceQueue:
    """A capacity-constrained service queue. `capacity` items complete per period; the rest wait. Queue
    time is REAL: it delays a matter's decision and grows the backlog. `discipline` orders service."""
    queue_id: str
    capacity_per_period: float = 1.0
    discipline: str = "fifo"
    service_days: float = 1.0
    items: list = field(default_factory=list)           # [{matter_id, priority, filed_period}]
    completed: list = field(default_factory=list)

    def enqueue(self, matter_id: str, *, priority: float = 0.0, period: int = 0):
        self.items.append({"matter_id": matter_id, "priority": priority, "filed_period": period})

    def _order(self):
        if self.discipline in ("priority", "statutory_priority", "emergency_priority", "risk_based"):
            return sorted(range(len(self.items)), key=lambda i: (-self.items[i]["priority"],
                                                                 self.items[i]["filed_period"]))
        return list(range(len(self.items)))              # fifo

    def service_period(self, period: int) -> list:
        """Serve up to capacity items this period; return the matter_ids completed. Backlog persists."""
        order = self._order()
        served, keep = [], set()
        cap = self.capacity_per_period
        for rank, i in enumerate(order):
            if rank < cap:
                served.append(self.items[i]["matter_id"])
            else:
                keep.add(i)
        self.items = [self.items[i] for i in sorted(keep)]
        self.completed.extend(served)
        return served

    def wait_periods(self, matter_id: str) -> int | None:
        """Expected periods a matter waits given its queue position and capacity (deadline-risk signal)."""
        order = self._order()
        for rank, i in enumerate(order):
            if self.items[i]["matter_id"] == matter_id:
                import math
                return int(math.floor(rank / max(1e-9, self.capacity_per_period)))
        return None

    def backlog(self) -> int:
        return len(self.items)
