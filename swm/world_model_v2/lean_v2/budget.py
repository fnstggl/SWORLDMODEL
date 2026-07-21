"""ConsumerComputeBudget — the run-level hard cap, enforced at the single external-call gateway.

DELIBERATELY LIBERAL: every cap sits ~4× above what a normal consumer run should spend, so a
healthy run never feels it — it exists to stop a runaway, not to shape a forecast. When it does
trip: optional escalations (critics, challengers, replicates) stop starting, completed simulation
state is preserved, and the run returns its best defensible LABELED probability with the skipped
work disclosed. Exhaustion NEVER replaces a result with 0.5 — there is no neutral default
anywhere in this runtime."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict


@dataclass
class ConsumerComputeBudget:
    """Liberal hard caps well above the expected consumer run (~18 calls / ~2 min observed on
    the EXP-112 five-question evaluation). The maximum is 25 min / 200 external calls — a
    runaway backstop, not a shaping constraint; 99% of runs never approach it."""
    max_wall_s: float = 1500.0                 # 25 min      (expected ~2-4 min)
    max_calls: int = 200                       #             (expected ~15-50)
    max_input_chars: int = 2_400_000           # provider-token proxy when usage is absent
    max_output_chars: int = 1_200_000
    max_structural_models: int = 4             #             (expected 1, +1 challenger)
    max_deliberations: int = 40                #             (expected 0-5)
    max_novel_consequence_compiles: int = 30   #             (expected 0-3)
    max_weighted_nodes: int = 4096             #             (expected tens-hundreds)
    max_provider_retries: int = 40             # one bounded retry per failed call, capped globally

    def caps(self) -> dict:
        return asdict(self)


class BudgetLedger:
    """Thread-safe live accounting against a ConsumerComputeBudget. The gateway consults
    `allow_call()` BEFORE every external call; optional escalations consult
    `can_afford(estimate)` BEFORE starting."""

    def __init__(self, budget: ConsumerComputeBudget = None):
        self.budget = budget or ConsumerComputeBudget()
        self.t0 = time.time()
        self._lock = threading.RLock()
        self.calls = 0
        self.input_chars = 0
        self.output_chars = 0
        self.structural_models = 0
        self.deliberations = 0
        self.novel_consequence_compiles = 0
        self.peak_weighted_nodes = 0
        self.provider_retries = 0
        self.by_stage: dict = {}               # stage -> {calls, input_chars, output_chars, latency_s}
        self.stops: list = []                  # every refusal, with what was skipped and why
        self.exhausted_dimension = ""

    # ---- clock ---------------------------------------------------------------------
    def wall_s(self) -> float:
        return time.time() - self.t0

    def wall_remaining_s(self) -> float:
        return max(0.0, self.budget.max_wall_s - self.wall_s())

    # ---- gate: hard external-call admission ---------------------------------------
    def allow_call(self, stage: str, prompt_chars: int) -> tuple:
        """(allowed, reason). Refusals are recorded — they are result-visible facts."""
        with self._lock:
            if self.wall_s() > self.budget.max_wall_s:
                return self._refuse(stage, "wall_clock_exhausted")
            if self.calls >= self.budget.max_calls:
                return self._refuse(stage, "call_budget_exhausted")
            if self.input_chars + prompt_chars > self.budget.max_input_chars:
                return self._refuse(stage, "input_char_budget_exhausted")
            if self.output_chars > self.budget.max_output_chars:
                return self._refuse(stage, "output_char_budget_exhausted")
            return True, ""

    def _refuse(self, stage: str, why: str) -> tuple:
        self.exhausted_dimension = self.exhausted_dimension or why
        self.stops.append({"stage": stage, "refused": why, "at_wall_s": round(self.wall_s(), 2)})
        return False, why

    # ---- gate: optional-escalation affordability ----------------------------------
    def can_afford(self, *, what: str, est_calls: int = 0, est_wall_s: float = 0.0,
                   structural_model: bool = False, deliberation: bool = False,
                   novel_consequence: bool = False) -> tuple:
        """Estimate-before-escalation (§budget). A refusal records what optional work was
        skipped so the result can disclose it."""
        with self._lock:
            if structural_model and self.structural_models >= self.budget.max_structural_models:
                return self._skip(what, "structural_model_cap")
            if deliberation and self.deliberations >= self.budget.max_deliberations:
                return self._skip(what, "deliberation_cap")
            if novel_consequence and self.novel_consequence_compiles \
                    >= self.budget.max_novel_consequence_compiles:
                return self._skip(what, "novel_consequence_cap")
            if self.calls + est_calls > self.budget.max_calls:
                return self._skip(what, "insufficient_call_budget_for_escalation")
            if self.wall_s() + est_wall_s > self.budget.max_wall_s:
                return self._skip(what, "insufficient_wall_clock_for_escalation")
            return True, ""

    def _skip(self, what: str, why: str) -> tuple:
        self.stops.append({"skipped_optional": what, "why": why,
                           "at_wall_s": round(self.wall_s(), 2)})
        return False, why

    # ---- recording -----------------------------------------------------------------
    def record_call(self, stage: str, *, prompt_chars: int, reply_chars: int, latency_s: float,
                    tier: str = "strong", retried: bool = False):
        with self._lock:
            self.calls += 1
            self.input_chars += prompt_chars
            self.output_chars += reply_chars
            if retried:
                self.provider_retries += 1
            row = self.by_stage.setdefault(stage, {"calls": 0, "input_chars": 0,
                                                   "output_chars": 0, "latency_s": 0.0,
                                                   "tiers": {}})
            row["calls"] += 1
            row["input_chars"] += prompt_chars
            row["output_chars"] += reply_chars
            row["latency_s"] = round(row["latency_s"] + latency_s, 3)
            row["tiers"][tier] = row["tiers"].get(tier, 0) + 1

    def record_structural_model(self):
        with self._lock:
            self.structural_models += 1

    def record_deliberation(self):
        with self._lock:
            self.deliberations += 1

    def record_novel_consequence(self):
        with self._lock:
            self.novel_consequence_compiles += 1

    def observe_nodes(self, n: int):
        with self._lock:
            self.peak_weighted_nodes = max(self.peak_weighted_nodes, int(n))

    def retries_allowed(self) -> bool:
        with self._lock:
            return self.provider_retries < self.budget.max_provider_retries

    def manifest(self) -> dict:
        with self._lock:
            return {"caps": self.budget.caps(),
                    "wall_s": round(self.wall_s(), 2),
                    "calls": self.calls,
                    "input_chars": self.input_chars, "output_chars": self.output_chars,
                    "structural_models": self.structural_models,
                    "deliberations": self.deliberations,
                    "novel_consequence_compiles": self.novel_consequence_compiles,
                    "peak_weighted_nodes": self.peak_weighted_nodes,
                    "provider_retries": self.provider_retries,
                    "by_stage": {k: dict(v) for k, v in sorted(self.by_stage.items())},
                    "exhausted_dimension": self.exhausted_dimension,
                    "stops": list(self.stops)[-40:]}
