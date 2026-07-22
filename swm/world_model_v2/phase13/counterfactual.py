"""Phase 13 matched counterfactual evaluation (Parts 8–10) — the paired engine every decision uses.

For every compared action/policy, hold constant: posterior particle identity, initial WorldState,
structural-hypothesis assignment (stratified BY PARTICLE INDEX, identical across arms), exogenous shock
streams (stream-partitioned CRN via `MatchedRolloutEngine`), horizon, and the utility evaluation.
Alternatives BRANCH from the same particle; the report is built from PAIRED differences
`U(action, particle_j) − U(reference, particle_j)`, with the matching's variance reduction measured,
not asserted. Independent Monte-Carlo runs with different seeds do NOT satisfy this contract.

The evaluator runs the SAME operator set the compiled plan names (all phase operators fire), plus the
registered `DecisionActionOperator` so decision events execute canonically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.world_model_v2.phase13.crn import MatchedRolloutEngine, exogenous_trace
from swm.world_model_v2.phase13.interventions import DecisionActionOperator, to_intervention


@dataclass
class ArmRollout:
    arm_id: str
    branches: list = field(default_factory=list)      # [WorldBranch per particle]
    outcomes: list = field(default_factory=list)      # [outcome dict per particle]
    n_deltas: int = 0


@dataclass
class MatchedBundle:
    """One matched evaluation: shared particles, per-arm rollouts, CRN manifest."""
    arms: dict = field(default_factory=dict)          # arm_id -> ArmRollout
    reference: str = "do_nothing"
    n_particles: int = 0
    seed: int = 0
    hypothesis_assignment: list = field(default_factory=list)   # per-particle hypothesis id
    crn_manifest: dict = field(default_factory=dict)

    def paired_diffs(self, arm_id: str, utility_of) -> list:
        a = self.arms[arm_id]
        r = self.arms[self.reference]
        return [utility_of(ai) - utility_of(ri) for ai, ri in zip(a.outcomes, r.outcomes)]


class MatchedEvaluator:
    """Owns ONE sampled particle set and evaluates arbitrary arms against it. Built either from a
    compiled WorldExecutionPlan (`from_plan` — the canonical path: posterior injection, hypothesis
    stratification, plan operators) or from raw runtime pieces (controlled benchmark tasks)."""

    def __init__(self, *, initial, queue_builder, operators, contract, n_particles=60, seed=0,
                 hypotheses=None, outcome_fn=None, max_events=500, decision_llm=None):
        self.initial = initial
        self.queue_builder = queue_builder
        self.operators = list(operators)
        if not any(getattr(op, "name", "") == "decision_action" for op in self.operators):
            self.operators.append(DecisionActionOperator(llm=decision_llm))
        self.contract = contract
        self.n_particles = int(n_particles)
        self.seed = int(seed)
        self.hypotheses = list(hypotheses or [])
        self.outcome_fn = outcome_fn                      # callable(world) -> outcome dict
        self.max_events = int(max_events)
        self._particles = None
        self._assignment = []

    # ---------------- construction from the canonical plan ----------------
    @classmethod
    def from_plan(cls, plan, *, llm=None, n_particles=None, seed=0, outcome_fn=None):
        from swm.world_model_v2.init_state import InitialStateModel
        from swm.world_model_v2.materialize import (_bind_scenario_schema, build_world,
                                                    check_readout_binding,
                                                    _inject_posterior_rate, operators_from_plan,
                                                    queue_builder_from_plan)
        base = build_world(plan, evidence_hash=(plan.provenance or {}).get("evidence_bundle_hash", ""))
        check_readout_binding(plan, base)
        # Phase 13 arms roll the SAME generated world semantics + causal boundary as ordinary
        # forecasts: the branch schema and its mechanisms ride on every cloned particle
        _bind_scenario_schema(plan, base, llm)
        _inject_posterior_rate(plan)
        ops, rejections = operators_from_plan(plan, llm=llm)
        init = InitialStateModel(base_world=base, latents=list(plan.latents))
        ev = cls(initial=init, queue_builder=queue_builder_from_plan(plan), operators=ops,
                 contract=plan.outcome_contract,
                 n_particles=n_particles or plan.compute_plan.get("n_particles", 60), seed=seed,
                 hypotheses=list(getattr(plan, "structural_hypotheses", []) or []),
                 outcome_fn=outcome_fn, decision_llm=llm)
        ev.operator_rejections = rejections
        return ev

    # ---------------- shared particles (sampled ONCE) ----------------
    def particles(self):
        if self._particles is None:
            worlds = self.initial.sample_particles(self.n_particles, seed=self.seed)
            self._particles = worlds
            self._assignment = self._assign_hypotheses(worlds)
        return self._particles

    def _assign_hypotheses(self, worlds) -> list:
        """Stratify particles across structural hypotheses BY INDEX — the assignment is a property of
        the particle, so every arm sees the identical structural world (Part 8's 'structural hypothesis
        held constant'). Weights: Phase-3 structural posterior when present, else compiler priors."""
        if not self.hypotheses:
            return ["H0"] * len(worlds)
        def w(h):
            return max(0.0, float(h.get("posterior", h.get("prior", 1.0)) or 1.0))
        z = sum(w(h) for h in self.hypotheses) or 1.0
        alloc, assigned = [], 0
        for i, h in enumerate(self.hypotheses):
            k = (len(worlds) - assigned if i == len(self.hypotheses) - 1
                 else max(1, round(len(worlds) * w(h) / z)))
            alloc.append(max(0, k))
            assigned += alloc[-1]
        out = []
        for h, k in zip(self.hypotheses, alloc):
            out.extend([str(h.get("id", "H"))] * k)
        out = out[:len(worlds)] + ["H0"] * max(0, len(worlds) - len(out))
        for wld, hid, h in zip(worlds, out, _expand(self.hypotheses, alloc)):
            wld.uncertainty_meta.setdefault("model", {})["hypothesis"] = hid
            if h and (h.get("lean") or h.get("outcome_lean")):
                wld.uncertainty_meta["hypothesis_lean"] = str(h.get("lean") or h.get("outcome_lean"))
        return out

    # ---------------- arm evaluation ----------------
    def evaluate_arm(self, arm_id: str, intervention=None) -> ArmRollout:
        """Clone every shared particle, apply the intervention to the clone's queue, roll with
        stream-partitioned CRN keyed on (seed, particle index) — identical across arms."""
        engine = MatchedRolloutEngine(operators=self.operators)
        arm = ArmRollout(arm_id=arm_id)
        for i, w0 in enumerate(self.particles()):
            w = w0.clone(branch_id=f"{w0.branch_id}:{arm_id}")
            q = self.queue_builder(w)
            if intervention is not None and intervention.apply is not None:
                intervention.apply(w, q)
            b = engine.run_branch(w, q, seed=self.seed * 7919 + i, max_events=self.max_events)
            arm.branches.append(b)
            arm.n_deltas += len(b.log)
            arm.outcomes.append(self._outcome(b.world))
        return arm

    def _outcome(self, world) -> dict:
        out = {"readout": None, "quantities": {}, "world": world}
        try:
            out["readout"] = self.contract.readout(world) if self.contract is not None else None
        except Exception as e:  # noqa: BLE001 — a broken readout is recorded, not silently zeroed
            out["readout_error"] = f"{type(e).__name__}"
        for name, q in (getattr(world, "quantities", {}) or {}).items():
            if isinstance(getattr(q, "value", None), (int, float, bool)):
                out["quantities"][name] = float(q.value)
        return out

    def evaluate(self, actions: list, *, problem=None, reference_id: str = None) -> MatchedBundle:
        """Evaluate ActionSchemas (converted to canonical Interventions). The reference arm is the
        explicit baseline (Part 9): `do_nothing` if present, else the first action."""
        bundle = MatchedBundle(n_particles=self.n_particles, seed=self.seed)
        self.particles()
        bundle.hypothesis_assignment = list(self._assignment)
        ids = [a.action_id for a in actions]
        bundle.reference = (reference_id if reference_id in ids
                            else ("do_nothing" if "do_nothing" in ids else (ids[0] if ids else "")))
        for a in actions:
            iv = to_intervention(a, problem)
            bundle.arms[a.action_id] = self.evaluate_arm(a.action_id, iv)
        bundle.crn_manifest = self.crn_manifest(bundle)
        return bundle

    # ---------------- CRN manifest + verification (Part 8 evidence, not assertion) ----------------
    def crn_manifest(self, bundle: MatchedBundle) -> dict:
        ref = bundle.arms.get(bundle.reference)
        man = {"root_seed": self.seed, "per_particle_seed": "seed*7919 + particle_index",
               "stream_partitioning": "sha256(seed|stream): hazard|<etype>, op|<operator>|<etype>, "
                                      "impl|<action_id>",
               "particles_sampled_once": True,
               "hypothesis_stratification": sorted(set(bundle.hypothesis_assignment)),
               "n_particles": self.n_particles}
        if ref is not None and ref.branches:
            # pairing check: every arm must share the reference's exogenous trace per particle
            match = {}
            ref_traces = [exogenous_trace(b) for b in ref.branches]
            for aid, arm in bundle.arms.items():
                same = sum(1 for rb, ab in zip(ref_traces, (exogenous_trace(b) for b in arm.branches))
                           if rb == ab)
                match[aid] = round(same / max(1, len(ref.branches)), 4)
            man["exogenous_trace_match_vs_reference"] = match
        return man


def _expand(hyps, alloc):
    out = []
    for h, k in zip(hyps, alloc):
        out.extend([h] * k)
    return out


# ---------------------------------------------------------------- paired statistics (Part 8 report)
def paired_report(diffs: list, *, conf: float = 0.8) -> dict:
    """Paired mean/median effect, P(improvement), quantiles, CI, and the variance-reduction evidence."""
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = sum(diffs) / n
    s = sorted(diffs)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else float("inf")
    z = 1.2816 if abs(conf - 0.8) < 1e-9 else 1.96
    p_imp = sum(1 for x in diffs if x > 0) / n
    p_tie = sum(1 for x in diffs if x == 0) / n
    return {"n": n, "paired_mean": round(mean, 6), "paired_median": round(med, 6),
            "p_improvement": round(p_imp, 4), "p_tie": round(p_tie, 4),
            "paired_q10": round(s[min(n - 1, int(0.10 * n))], 6),
            "paired_q90": round(s[min(n - 1, int(0.90 * n))], 6),
            "ci": [round(mean - z * se, 6), round(mean + z * se, 6)], "conf": conf,
            "paired_se": round(se, 6)}


def variance_reduction(arm_utils: list, ref_utils: list) -> dict:
    """How much the matching bought: Var(paired diff) vs Var(unpaired diff)=Var(a)+Var(r).
    effective_particles = n × unpaired/paired (the CRN sample-size multiplier)."""
    n = len(arm_utils)
    if n < 2:
        return {"n": n}
    diffs = [a - r for a, r in zip(arm_utils, ref_utils)]
    def _var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    vp = _var(diffs)
    vu = _var(arm_utils) + _var(ref_utils)
    ratio = (vu / vp) if vp > 1e-15 else float("inf")
    return {"var_paired": round(vp, 8), "var_unpaired_sum": round(vu, 8),
            "variance_reduction_factor": (round(ratio, 3) if ratio != float("inf") else "inf"),
            "effective_particles": (round(n * ratio, 1) if ratio != float("inf") else "inf"),
            "n": n}
