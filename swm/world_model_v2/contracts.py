"""Outcome & intervention contracts — Phase 1.8 + Phase 6 readout projections.

Extends v1's typed families with delay/type/cascade/utility/best-action. The load-bearing property for the
whole architecture: every contract defines a PROJECTION from a terminal WorldState into the native answer
space — if the compiler cannot define that projection before rollout, the simulation must not proceed
(Phase 6). The final numbers are frequencies/statistics over terminal states, NEVER a post-hoc LLM forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FAMILIES = ("binary", "categorical", "continuous", "ranked_artifacts", "response_occurrence",
            "response_delay", "response_type", "reach_distribution", "cascade_structure",
            "utility_distribution", "best_action")


class ContractError(ValueError):
    pass


@dataclass
class OutcomeContract:
    family: str
    options: list = field(default_factory=list)   # named options / candidate artifacts (where applicable)
    resolution_rule: str = ""                     # human-readable statement of what resolves it
    readout: object = None                        # callable(world) -> native terminal value (REQUIRED pre-run)
    metric: str = ""                              # for ranked_artifacts/best_action: the explicit objective
    horizon_ts: float = None                      # when the outcome resolves (unix)
    readout_var: str = ""                         # declarative path the readout reads (binding-checked)

    def validate(self):
        if self.family not in FAMILIES:
            raise ContractError(f"unknown outcome family {self.family!r} (supported: {FAMILIES})")
        if self.family == "categorical" and len([o for o in self.options if str(o).strip()]) < 2:
            raise ContractError("categorical needs >=2 named options")
        if not callable(self.readout):
            raise ContractError(
                "no terminal-state readout projection defined — the simulation must not proceed (Phase 6): "
                "the answer must be READ from terminal world states, never asked of an LLM afterward")
        return self

    #: truthy / falsy literals a mechanism may write instead of the declared option labels
    _TRUE_LITERALS = frozenset(("true", "yes", "1", "1.0", "occurred", "success", "pass", "accept",
                                "approved", "ratified", "won", "happened"))
    _FALSE_LITERALS = frozenset(("false", "no", "0", "0.0", "not_occurred", "failure", "fail", "reject",
                                 "declined", "lost", "did_not_happen"))

    def _canonical_option(self, v, opts) -> str:
        """Reconcile a terminal value to a declared option label. A mechanism may write the readout in a
        different vocabulary than the contract options (e.g. boolean True / the string 'True' vs a
        yes/no contract). For a 2-option binary/response family, a truthy value maps to the AFFIRMATIVE
        option (options[0]) and a falsy value to the negative (options[1]); this only reconciles vocabulary,
        it never invents a value. Values already in the option set, and all categorical values, pass through."""
        key = str(v)
        if key in opts:
            return key
        if self.family in ("binary", "response_occurrence") and len(self.options) == 2:
            low = key.strip().lower()
            if v is True or low in self._TRUE_LITERALS:
                return str(self.options[0])
            if v is False or low in self._FALSE_LITERALS:
                return str(self.options[1])
        return key

    def project(self, terminal_branches) -> dict:
        """Aggregate the native answer over weighted terminal branches: frequencies for discrete families,
        weighted samples for continuous/delay/reach. `terminal_branches` = [WorldBranch].

        OPTION-SPACE COVERAGE (Tier A1): for discrete families with declared options, terminal values
        outside the option space (including None from worlds where nothing resolved) are reported as
        `unresolved_share`, NOT as answer mass — a silent no-op world must not read as a confident answer."""
        vals = [(b.weight, self.readout(b.world)) for b in terminal_branches]
        z = sum(w for w, _ in vals) or 1.0
        if self.family in ("binary", "categorical", "response_occurrence", "response_type", "best_action"):
            opts = {str(o) for o in self.options if str(o).strip()}
            freq, unresolved = {}, 0.0
            for w, v in vals:
                key = self._canonical_option(v, opts)
                if opts and key not in opts:
                    unresolved += w / z
                    continue
                freq[key] = freq.get(key, 0.0) + w / z
            out = {"distribution": {k: round(p, 4) for k, p in sorted(freq.items(), key=lambda kv: -kv[1])},
                   "n_worlds": len(vals)}
            if opts:
                out["unresolved_share"] = round(unresolved, 4)
                if unresolved > 0.5:
                    out["warning"] = (f"{unresolved:.0%} of terminal worlds resolved to values outside the "
                                      f"declared option space — the simulation likely did not execute the "
                                      f"causal chain; treat this answer as unsupported")
            return out
        # continuous-like: return weighted quantiles
        xs = sorted((float(v), w) for w, v in vals if isinstance(v, (int, float)))
        if not xs:
            return {"distribution": {}, "n_worlds": len(vals), "note": "no numeric terminal values"}
        def q(t):
            acc = 0.0
            for v, w in xs:
                acc += w / z
                if acc >= t:
                    return v
            return xs[-1][0]
        return {"quantiles": {"p10": q(.10), "p50": q(.50), "p90": q(.90)},
                "mean": round(sum(v * w for v, w in xs) / z, 4), "n_worlds": len(vals)}


@dataclass
class Intervention:
    """A typed intervention: apply(world) mutates a CLONED world at t0 (or schedules events into its queue).
    Arbitrary action spaces compose from these (discrete/continuous/message/timing/sequence/policy/none)."""
    intervention_id: str
    description: str = ""
    apply: object = None                  # callable(world, queue) -> None (mutates the clone)
    kind: str = "discrete"                # discrete | continuous | artifact | timing | sequence | policy | none

    def validate(self):
        if self.intervention_id != "none" and not callable(self.apply):
            raise ContractError(f"intervention {self.intervention_id!r} has no executable apply()")
        return self


@dataclass
class ActionSpace:
    interventions: list = field(default_factory=list)   # [Intervention] — always includes 'none' baseline

    def __post_init__(self):
        if not any(i.intervention_id == "none" for i in self.interventions):
            self.interventions.append(Intervention(intervention_id="none", description="take no action",
                                                   apply=lambda world, queue: None, kind="none"))


@dataclass
class UtilityFunction:
    """Explicit objective for best-action: utility(world) -> float on a terminal state."""
    name: str
    fn: object

    def score(self, terminal_branches) -> dict:
        vals = sorted((self.fn(b.world), b.weight) for b in terminal_branches)
        z = sum(w for _, w in vals) or 1.0
        mean = sum(v * w for v, w in vals) / z
        def q(t):
            acc = 0.0
            for v, w in vals:
                acc += w / z
                if acc >= t:
                    return v
            return vals[-1][0]
        return {"expected_utility": round(mean, 4), "downside_p10": round(q(.10), 4),
                "median": round(q(.50), 4)}
