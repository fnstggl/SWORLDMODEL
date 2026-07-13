"""Phase 11 — the dynamic-recompilation controller (spec §7, §17, §19, §20).

One universal pipeline drives every domain (no per-domain recompiler):

  advance ensemble to the next EXTERNAL observation → assimilate (Phase-3 posterior update, always) →
  diagnostics → trigger detection (eligible observations only) → dependence-aware fusion → decision →
  scope → candidate generation → static validation → atomic migration (off-path, rollback on failure) →
  reproducible scoring → activate the retained plan MIXTURE → emit typed recompile events → continue →
  terminal distribution marginalised over the plan mixture + particles.

Discipline enforced here: a representable simulation-internal event is executed normally and NEVER recompiles
(``observation_eligible``); the current plan is always a scored candidate so "don't recompile" can win; a
normal one-shot forecast with no external evidence performs ZERO recompiles. Anti-thrashing: max recompiles
per horizon, min-new-evidence, cooldowns (via fusion), and A→B→A oscillation refusal (via lineage). The
recompile events are RECORDS of causal-model evolution; they do not themselves move terminal probabilities —
those come only from continued world execution.

Execution is via an injected ``ExecutionAdapter`` so the controller runs on the real V2 path (init_state →
queue_builder → operators → contract.project) in validation, and on lightweight numeric substrates in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.phase11 import diagnostics as D
from swm.world_model_v2.phase11._serial import plan_content_hash
from swm.world_model_v2.phase11.triggers import detect_all, TriggerContext, TriggerThresholds, observation_eligible
from swm.world_model_v2.phase11.fusion import TriggerFusion
from swm.world_model_v2.phase11.scope import select_scope
from swm.world_model_v2.phase11.candidates import generate_candidates, apply_transform, validate_candidate
from swm.world_model_v2.phase11.scoring import score_candidates
from swm.world_model_v2.phase11.migration import migrate
from swm.world_model_v2.phase11.lineage import (snapshot, RecompileTransaction, standard_invariants,
                                                LineageGraph)
from swm.world_model_v2.phase11.contracts import (RecompileDecision, PlanLineageNode, PlanLineageEdge,
                                                  RecompilationTrace, RecompileObservation)

RECOMPILE_EVENT_TYPES = ("recompile_triggered", "recompile_candidate_generated", "recompile_decision",
                         "plan_migrated", "event_canceled", "event_remapped", "plan_branch_added",
                         "plan_branch_pruned", "recompile_completed", "recompile_failed")


@dataclass
class ExecutionAdapter:
    """Bridges the controller to the world-execution substrate. The default reads a scalar observable path from
    each particle world and advances by elapsing the clock; validation supplies a real V2-path adapter."""
    observable_path: str = "outcome"

    def advance(self, worlds, weights, pending, until_ts):
        for w in worlds:
            clk = getattr(w, "clock", None)
            if clk is not None and until_ts > getattr(clk, "now", until_ts):
                try:
                    clk.advance_to(until_ts)
                except Exception:  # noqa: BLE001 — never move time backward
                    pass
        return worlds, weights, pending, []

    def predict(self, worlds, weights, obs):
        """Predictive distribution over the observable from the ensemble. Default: read the path per particle."""
        from swm.world_model_v2.posterior import _read_path
        vals = []
        for w in worlds:
            try:
                v = _read_path(w, getattr(obs, "observation_type", None) or self.observable_path)
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                vals.append(float(v) if isinstance(v, (int, float)) else v)
        return vals

    def assimilate(self, worlds, weights, obs):
        return weights                                     # default: no reweight (adapter-specific)

    def post_migration(self, worlds, weights, obs, sim_time):
        """Apply the ADOPTED structural revision's effect on the execution substrate. In the real V2 path this
        is init_state re-sampling the revised/new components; the spec (§15) requires introducing BROAD
        uncertainty for newly-created variables so a structural change is not falsely certain. Default: no-op."""
        return worlds, weights

    def terminal(self, worlds, weights):
        from swm.world_model_v2.posterior import _read_path
        num, z = 0.0, 0.0
        for w, wt in zip(worlds, weights):
            try:
                v = _read_path(w, self.observable_path)
            except Exception:  # noqa: BLE001
                v = None
            if isinstance(v, (int, float)):
                num += wt * v
                z += wt
        return {"mean": (num / z) if z else None, "n": len(worlds)}


@dataclass
class ControllerResult:
    terminal: dict = field(default_factory=dict)
    traces: list = field(default_factory=list)             # [RecompilationTrace.as_record()]
    lineage: dict = field(default_factory=dict)
    n_recompiles: int = 0
    n_observations: int = 0
    n_eligible: int = 0
    support_grade: str = "exploratory"
    status: str = "completed"                              # SIMULATION_STATUSES
    plan_mixture: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RecompilationController:
    thresholds: TriggerThresholds = None
    max_recompiles: int = 5
    min_new_evidence: int = 1
    llm: object = None
    seed: int = 0
    # ---- ablation / baseline-arm knobs (spec §24/§26). Inert at their defaults = full Phase 11 (B5). ----
    recompile_enabled: bool = True        # B0: no recompilation
    forced_scope: str = ""                # B1 parameter_only / B2 full_plan (override scope selection)
    require_score_gate: bool = True       # B3 LLM-only: skip evidence-based scoring gate
    branch_only: bool = False             # B6: only add a competing hypothesis; never migrate/replace
    oracle: dict = None                   # B4: {change_time, scope} supplied from labels
    fusion_enabled: bool = True           # ablation: no dependence-aware fusion
    scope_selection_enabled: bool = True  # ablation: always full recompile

    def run(self, *, plan, worlds, weights, pending_events, observations, horizon_ts, as_of,
            execution: ExecutionAdapter = None, terminal_sensitivity=0.6, plan_facts=None) -> ControllerResult:
        ex = execution or ExecutionAdapter()
        th = self.thresholds or TriggerThresholds()
        fusion = TriggerFusion()
        lineage = LineageGraph()
        res = ControllerResult(n_observations=len(observations))
        active_plan = plan
        plan_mixture = [{"plan_hash": plan_content_hash(plan), "weight": 1.0}]
        residual_hist, value_hist, pit_hist = [], [], []
        sim_time = as_of
        n_recompiles = 0
        llm_calls = 0
        node0 = PlanLineageNode(plan_id="p0", plan_hash=plan_mixture[0]["plan_hash"], plan_version=getattr(plan, "version", 1),
                                status="active", simulation_time=sim_time, revision_reason="genesis")
        lineage.add_node(node0)
        lineage.activate(plan_mixture[0]["plan_hash"])

        for obs in sorted(observations, key=lambda o: getattr(o, "event_time", 0.0)):
            reveal = getattr(obs, "event_time", sim_time)
            worlds, weights, pending_events, _ = ex.advance(worlds, weights, pending_events, reveal)
            sim_time = max(sim_time, reveal)

            eligible = observation_eligible(obs)
            # predictive surprise (diagnostics) — computed for every obs (feeds sustained-failure history)
            predictive = ex.predict(worlds, weights, obs)
            observed = (getattr(obs, "provenance", {}) or {}).get("observed_value")
            surprise = D.surprise(predictive, observed) if (predictive and observed is not None) else {}
            if surprise:
                residual_hist.append(surprise.get("residual", 0.0))
                if surprise.get("tail_prob") is not None:
                    pit_hist.append(surprise["tail_prob"])
            if isinstance(observed, (int, float)):
                value_hist.append(float(observed))

            if eligible and self.recompile_enabled:
                res.n_eligible += 1
                ess = D.ess_diagnostic(weights)
                ctx = TriggerContext(observation=obs, surprise=surprise, ess=ess,
                                     residual_history=list(residual_hist), value_history=list(value_hist),
                                     pre_residuals=residual_hist[:-1], post_residuals=residual_hist[-1:],
                                     plan_facts=plan_facts or {}, declared=_declared(obs),
                                     cooldown=fusion.cooldown_state())
                fired = detect_all(ctx, th)
                fused = fusion.fuse(fired) if self.fusion_enabled else self._nofuse(fired)
                if self.oracle is not None:               # B4 oracle: fire at the labelled change time
                    fused.proceed = abs(reveal - float(self.oracle.get("change_time", -1))) < 8 * 86400.0 or fused.proceed
                if fused.proceed and n_recompiles < self.max_recompiles:
                    trace = self._recompile(active_plan, worlds, weights, pending_events, obs, fused,
                                            sim_time, lineage, terminal_sensitivity, ex, plan_facts)
                    if trace is not None:
                        nw = trace.pop("_worlds", None)
                        if nw is not None:                 # activated (or no-op) → adopt returned ensemble
                            worlds = nw
                            weights = trace.pop("_weights", weights)
                            pending_events = trace.pop("_pending", pending_events)
                        active_plan = trace.pop("_active_plan", active_plan)
                        plan_mixture = trace.get("plan_mixture", plan_mixture)
                        llm_calls += trace.pop("_llm_calls", 0)
                        res.traces.append(trace)
                        if trace.get("decision", {}).get("action") not in (None, "no_change"):
                            n_recompiles += 1
                elif fused.proceed:
                    res.notes.append(f"recompile budget reached ({self.max_recompiles}) — continuing best plan mixture, degraded support")
            # Phase-3 posterior update (ALWAYS, recompile or not): assimilate the observation
            weights = ex.assimilate(worlds, weights, obs)

        # advance to horizon and read the terminal distribution (marginalized over the ensemble)
        worlds, weights, pending_events, _ = ex.advance(worlds, weights, pending_events, horizon_ts)
        res.terminal = ex.terminal(worlds, weights)
        res.terminal["plan_mixture"] = plan_mixture
        res.n_recompiles = n_recompiles
        res.plan_mixture = plan_mixture
        res.lineage = lineage.as_dict()
        res.cost = {"llm_calls": llm_calls, "n_recompiles": n_recompiles}
        res.support_grade = self._support_grade(n_recompiles, res, plan)
        cal = D.calibration_error(pit_hist)
        res.notes.append(f"calibration(ece={cal.get('ece')}, ks={cal.get('ks')}, n={cal.get('n')})")
        if n_recompiles >= self.max_recompiles:
            res.status = "completed_with_degradation"
        return res

    def _nofuse(self, fired):
        """Ablation: no dependence-aware fusion — treat each detector independently (over-counts)."""
        from swm.world_model_v2.phase11.fusion import FusedAssessment
        if not fired:
            return FusedAssessment()
        top = max(fired, key=lambda e: e.trigger_probability)
        fa = FusedAssessment(fused_probability=top.trigger_probability,
                             by_family={e.trigger_family: e.trigger_probability for e in fired},
                             dominant_family=top.trigger_family,
                             scope_candidates=sorted({s for e in fired for s in e.affected_scope_candidates}),
                             classification="local_structural", proceed=top.trigger_probability >= 0.5,
                             n_evidence=len(fired))
        return fa

    def _recompile(self, plan, worlds, weights, pending, obs, fused, sim_time, lineage, tsens, ex, plan_facts):
        """One recompilation cycle. Returns a RecompilationTrace dict (with the migrated ensemble under
        _worlds/_weights/_pending), or None if nothing was activated (rolled back)."""
        sel = select_scope(fused, plan=plan, terminal_sensitivity=tsens)
        # baseline-arm scope overrides (inert by default)
        if not self.scope_selection_enabled:
            sel.scope, sel.action = "full_plan", "full_recompile"
        if self.branch_only:
            from swm.world_model_v2.phase11.scope import SCOPE_ACTION
            sel.scope, sel.action = "structural_hypothesis", SCOPE_ACTION["structural_hypothesis"]
        if self.forced_scope:
            from swm.world_model_v2.phase11.scope import SCOPE_ACTION
            sel.scope, sel.action = self.forced_scope, SCOPE_ACTION.get(self.forced_scope, "parameter_update")
        if self.oracle is not None and self.oracle.get("scope"):
            from swm.world_model_v2.phase11.scope import SCOPE_ACTION
            sel.scope = self.oracle["scope"]
            sel.action = SCOPE_ACTION.get(sel.scope, sel.action)
        cands = generate_candidates(plan, sel, fused, obs, llm=self.llm)
        llm_calls = 1 if self.llm is not None else 0
        # validate candidates
        valid = []
        for cand, ops in cands:
            revised = apply_transform(plan, ops) if not cand.is_current_plan else plan
            v = validate_candidate(plan, revised, ops, obs, now=sim_time)
            cand.static_validation = v
            if v["ok"]:
                valid.append((cand, ops, revised))
        if not valid:
            return self._failed_trace(plan, obs, fused, sel, sim_time, "all candidates failed validation")

        # score (current plan included) BEFORE committing to migration
        score = score_candidates([(c, o) for c, o, _ in valid], plan, obs, fused)
        if self.require_score_gate and not score.recompile_warranted:
            dec = RecompileDecision(decision_id=f"dec::{obs.observation_id}", current_plan_hash=plan_content_hash(plan),
                                    decision_time=sim_time, action="no_change", selected_scope="no_model_change",
                                    rationale="current plan retained after scoring — recompilation not warranted")
            return self._noop_trace(plan, worlds, weights, pending, obs, fused, sel, score, dec, sim_time)

        # oscillation guard: refuse to re-activate a recently-active plan without new evidence
        if not self.require_score_gate:
            # B3 LLM-only: adopt the first non-current revision without evidence scoring (diagnostic baseline)
            winner = next(((c, o, r) for c, o, r in valid if not c.is_current_plan), valid[0])
        else:
            winner = next((c, o, r) for c, o, r in valid if c.candidate_id == score.top_candidate_id)
        cand, ops, revised = winner
        dest_hash = plan_content_hash(revised)
        if lineage.oscillation(dest_hash):
            return self._failed_trace(plan, obs, fused, sel, sim_time,
                                      "oscillation guard: destination plan recently active (A->B->A) — no new evidence")

        # atomic migration (off-path, rollback on failure)
        cp = snapshot(worlds, weights, pending, plan, sim_time)
        dest_valid = None if any(t.op == "full_recompile" for t in ops) else None
        def build():
            mo = migrate(plan, revised, ops, worlds=worlds, weights=weights, pending_events=pending,
                         sim_time=sim_time, dest_valid_etypes=dest_valid)
            return mo.worlds, mo.weights, mo.pending_events, mo.report
        tx = RecompileTransaction(source=cp)
        act = tx.run(build, standard_invariants)
        if not act["activated"]:
            return self._failed_trace(plan, obs, fused, sel, sim_time, act["reason"])
        # the adopted structure now governs execution: let the adapter reflect it (broad uncertainty over the
        # revised/new components — §15). This is what makes recompilation help BEYOND posterior updating.
        act["worlds"], act["weights"] = ex.post_migration(act["worlds"], act["weights"], obs, sim_time)

        # lineage + plan mixture from the scored mixture
        node = PlanLineageNode(plan_id=f"p{lineage.depth()}", plan_hash=dest_hash,
                               plan_version=getattr(revised, "version", 2), parent_plan_ids=[plan_content_hash(plan)],
                               revision_reason=sel.rationale, trigger_id=fused.dominant_family,
                               simulation_time=sim_time, status="active")
        lineage.add_node(node)
        lineage.add_edge(PlanLineageEdge(parent_plan_id=plan_content_hash(plan), child_plan_id=dest_hash,
                                         reason=sel.action))
        lineage.activate(dest_hash)
        plan_mixture = [{"plan_hash": (plan_content_hash(revised) if m["candidate_id"] != "cand::current"
                                       else plan_content_hash(plan)), "weight": m["weight"]} for m in score.mixture]

        dec = RecompileDecision(decision_id=f"dec::{obs.observation_id}", current_plan_hash=plan_content_hash(plan),
                                current_plan_version=getattr(plan, "version", 1), decision_time=sim_time,
                                trigger_evidence=[e for e in fused.by_family], action=sel.action,
                                rationale=sel.rationale, selected_scope=sel.scope,
                                expected_value_of_recompile=sel.expected_improvement,
                                deferred_scope=[a["scope"] for a in sel.alternatives],
                                support_grade="exploratory")
        events = self._emit_events(fused, cands, dec, act["report"], node)
        trace = RecompilationTrace(
            trace_id=f"tr::{obs.observation_id}", simulation_time=sim_time,
            observations=[obs.as_record()], diagnostics={"surprise": ex.predict and {} or {},
                                                         "fused": fused.as_dict()},
            trigger_posterior=fused.by_family, selected_scope=sel.scope, scope_alternatives=sel.alternatives,
            candidates=[c.as_record() for c, _, _ in valid], scores=[s.as_dict() for s in score.scores],
            rejected_candidates=[{"candidate_id": c.candidate_id, "problems": c.static_validation.get("problems", [])}
                                 for c, _ in cands if not c.static_validation.get("ok", True)],
            decision=dec.as_record(), migration_report=act["report"], plan_mixture=plan_mixture,
            lineage=lineage.as_dict(), events_emitted=events,
            terminal_effect={"note": "recompile records do not directly move terminal probs; continued "
                             "execution does"}, checksums={"migration_ok": act["report"].get("invariants_ok")})
        d = trace.as_record()
        d["_worlds"], d["_weights"], d["_pending"] = act["worlds"], act["weights"], act["pending"]
        d["_llm_calls"] = llm_calls
        d["_active_plan"] = revised                         # later steps compare against the revised structure
        d["plan_mixture"] = plan_mixture
        d["decision"] = dec.as_record()
        return d

    # ---- trace helpers (records for the no-op / failed / emitted-events cases) ----
    def _emit_events(self, fused, cands, dec, report, node):
        evs = [{"etype": "recompile_triggered", "family": fused.dominant_family, "prob": fused.fused_probability},
               {"etype": "recompile_candidate_generated", "n": len(cands)},
               {"etype": "recompile_decision", "action": dec.action, "scope": dec.selected_scope},
               {"etype": "plan_migrated", "retention": report.get("object_retention_rate"),
                "orphans": report.get("orphan_count")}]
        for r in report.get("canceled", []) or []:
            evs.append({"etype": "event_canceled", "reason": r})
        evs.append({"etype": "recompile_completed", "plan_hash": node.plan_hash})
        return evs

    def _noop_trace(self, plan, worlds, weights, pending, obs, fused, sel, score, dec, sim_time):
        d = RecompilationTrace(trace_id=f"tr::{obs.observation_id}", simulation_time=sim_time,
                               observations=[obs.as_record()], trigger_posterior=fused.by_family,
                               selected_scope="no_model_change", scores=[s.as_dict() for s in score.scores],
                               decision=dec.as_record(),
                               events_emitted=[{"etype": "recompile_triggered", "family": fused.dominant_family},
                                               {"etype": "recompile_decision", "action": "no_change"}],
                               terminal_effect={"note": "current plan retained"}).as_record()
        d["_worlds"], d["_weights"], d["_pending"] = worlds, weights, pending
        d["decision"] = dec.as_record()
        d["plan_mixture"] = [{"plan_hash": plan_content_hash(plan), "weight": 1.0}]
        return d

    def _failed_trace(self, plan, obs, fused, sel, sim_time, reason):
        dec = RecompileDecision(decision_id=f"dec::{obs.observation_id}", current_plan_hash=plan_content_hash(plan),
                                decision_time=sim_time, action="no_change", selected_scope="no_model_change",
                                rationale=f"recompile aborted: {reason}", limitations=[reason])
        d = RecompilationTrace(trace_id=f"tr::{obs.observation_id}", simulation_time=sim_time,
                               observations=[obs.as_record()], selected_scope="no_model_change",
                               decision=dec.as_record(), trigger_posterior=fused.by_family,
                               events_emitted=[{"etype": "recompile_failed", "reason": reason}],
                               terminal_effect={"note": "rolled back to source; current plan continues"}).as_record()
        # ensemble unchanged on failure — caller keeps its current worlds/weights/pending
        d["_worlds"] = None
        return d

    def _support_grade(self, n_recompiles, res, plan):
        base = getattr(plan, "support_grade", "exploratory")
        if n_recompiles == 0:
            return base
        # each recompile widens structural uncertainty → do not IMPROVE the grade; degrade if many
        order = ["empirically_supported", "transfer_supported", "exploratory", "highly_speculative"]
        i = order.index(base) if base in order else 2
        if n_recompiles >= 3:
            i = min(len(order) - 1, i + 1)
        return order[i]


def _declared(obs):
    prov = getattr(obs, "provenance", {}) or {}
    return dict(prov.get("declared", {}) or {})
