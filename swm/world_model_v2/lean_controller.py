"""The lean actor controller — the one object that turns the canonical actor runtime lean.

Attached to a `QualitativeActorPolicyRuntime` (see `attach_to_runtime`), it intercepts exactly two
seams of the canonical decision path and one seam of the operator path, leaving every validation,
feasibility, revision, consequence and persistence step of the full runtime running per branch:

  operator seam  — `should_invoke`: the causal-frontier gate (lean_invalidation);
  decision seam  — `decision_for`: execution classification → deterministic prechecks →
                   prior-decision validity → decision-relevant projection → equivalence cache
                   (single-flight) → ONE-call bounded cognition on a genuine miss → immutable
                   template storage; escalation to the staged full-fidelity pipeline only on a
                   recorded reason;
  post seam      — `after_execute`: persist the branch's PriorDecisionValidity + processed facts.

The controller is run-scoped: caches never cross runs. Attach order and the research-before-
psychology invariant are enforced by `arm_actor_calls` — a decision before the research ledger is
complete raises instead of silently front-running the evidence."""
from __future__ import annotations

import copy
import random as _random
import threading
from dataclasses import dataclass, field

from swm.world_model_v2 import bounded_cognition as BC
from swm.world_model_v2.lean_cognition import (COMPLEX_MENU_THRESHOLD, OneCallOutcome,
                                               assemble_from_response, run_one_call)
from swm.world_model_v2.lean_cohorts import ActorCohortManifest, LeanCohortHypothesizer
from swm.world_model_v2.lean_consequences import ConsequenceProgramCache
from swm.world_model_v2.lean_context import (DecisionRelevantContext,
                                             DecisionRelevantContextBuilder, context_rng_seed)
from swm.world_model_v2.lean_decision_cache import (ActorDecisionTemplate,
                                                    DecisionEquivalenceCache)
from swm.world_model_v2.lean_invalidation import (AvoidedCallLedger, DecisionInvalidationPolicy,
                                                  PrecheckVerdict, PriorDecisionValidity,
                                                  precheck, should_invoke)
from swm.world_model_v2.lean_prompts import (ActorContextSnapshot, ActorDecisionDelta,
                                             ActorPromptManifest)
from swm.world_model_v2.llm_call_cache import backend_fingerprint
from swm.world_model_v2.qualitative_actor import (QualitativeDecision, _date, _hash,
                                                  parse_qualitative_decision)

CONTROLLER_VERSION = "lean.controller.v1"


@dataclass
class LeanActorConfig:
    """The lean profile's compute-control settings — explicit, recorded, separate from any causal
    state (§17 tolerances live in lean_particles; these govern the actor layer only)."""
    behavioral_replicates_per_decision_context: int = 1
    cohort_ceiling: int = 6
    one_call_cognition: bool = True
    decision_cache: bool = True
    consequence_cache: bool = True
    prechecks: bool = True
    frontier_gate: bool = True
    #: escalation grounds that trigger the staged full pipeline (always recorded)
    escalate_on_complex_menu: bool = True

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


class LeanActorController:
    """Run-scoped. Thread-safe. One instance serves every particle of one structural model's
    rollout (the runtime object itself is already shared across branches)."""

    def __init__(self, *, config: LeanActorConfig = None, ledger=None, run_day: str = ""):
        self.config = config or LeanActorConfig()
        self.ledger = ledger
        self.run_day = str(run_day)
        self.cache = DecisionEquivalenceCache(
            behavioral_replicates_per_decision_context=
            self.config.behavioral_replicates_per_decision_context)
        self.invalidation = DecisionInvalidationPolicy()
        self.avoided = AvoidedCallLedger()
        self.cohort_manifest = ActorCohortManifest()
        self.prompt_manifest = ActorPromptManifest()
        self.consequence_cache: ConsequenceProgramCache | None = None
        self.escalations: list[dict] = []
        self.one_call_successes = 0
        self.frontier_skips = 0
        self._armed = False                          # research-before-psychology gate
        self._snapshots: dict = {}                   # (actor, cohort) -> (snapshot, baseline)
        self._prior_ctx: dict = {}                   # (branch, actor) -> projection of prior
        self._builder_cache: dict = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ arming invariant
    def arm_actor_calls(self, *, research_ledger: dict):
        """§research-first: the runtime refuses actor psychology until the ordered research
        stages are recorded complete. The ledger is stored for the manifest."""
        required = ("resolution_criterion_parsed", "question_type_fixed",
                    "evidence_requirements_built", "evidence_gathered",
                    "evidence_sufficiency_assessed", "calendar_recurrence_extracted",
                    "reference_prior_constructed", "institutions_normalized",
                    "actor_information_projected")
        missing = [k for k in required if not research_ledger.get(k)]
        if missing:
            raise RuntimeError(f"lean research-first invariant: actor calls armed before "
                               f"research completed — missing {missing}")
        self.research_ledger = dict(research_ledger)
        self._armed = True

    # ------------------------------------------------------------------ operator seam
    def gate_invocation(self, world, event, actor_id: str) -> tuple:
        if not self.config.frontier_gate:
            return True, "gate_disabled"
        invoke, reason = should_invoke(world, event, actor_id)
        if not invoke:
            self.frontier_skips += 1
            self.avoided.record(reason="no_observed_event" if reason == "no_observed_event"
                                else "outside_causal_frontier", actor_id=actor_id,
                                detail=reason, branch_id=str(getattr(world, "branch_id", "")))
        return invoke, reason

    # ------------------------------------------------------------------ decision seam
    def _builder(self, engine) -> DecisionRelevantContextBuilder:
        key = id(engine)
        b = self._builder_cache.get(key)
        if b is None:
            b = DecisionRelevantContextBuilder(
                prompt_version="lean.onecall.v1",
                backend_fingerprint=backend_fingerprint(engine.config.llm),
                structural_frame=engine.config.structural_frame,
                public_facts=engine.config.public_facts or [])
            self._builder_cache[key] = b
        return b

    @staticmethod
    def canonical_available(available: list) -> list:
        """Delivery order and delivery ids are implementation identity — two equivalent branches
        may receive the same facts in different order under different iids. Re-key the
        availability set canonically (sorted by (channel, canonical fact), deduplicated, ids
        a0..aN in canonical order) so the one-call prompt, the response's noticed ids, and every
        reuse refer to the SAME facts on every equivalent branch. The staged escalation path
        keeps the original bundle untouched (full-fidelity semantics)."""
        from swm.world_model_v2.lean_context import canonical_fact_id
        rows = []
        for it in available or []:
            fid = canonical_fact_id(it.get("summary", it.get("content", "")),
                                    it.get("channel", ""))
            rows.append((str(it.get("channel", "")), fid, it))
        rows.sort(key=lambda r: (r[0], r[1]))
        out, seen = [], set()
        for _ch, fid, it in rows:
            if fid in seen:
                continue
            seen.add(fid)
            out.append({**it, "obs_id": f"a{len(out)}", "fact_id": fid})
        return out

    @staticmethod
    def _availability(view, decision: dict, engine) -> tuple:
        """EXACTLY the staged pipeline's availability construction (mirrors
        QualitativeActorPolicyRuntime._run_bounded_cognition) — the information boundary is
        identical in both profiles."""
        bundle = decision.get("observation_bundle") or []
        if bundle:
            from swm.world_model_v2.bounded_cognition import EXACT_MESSAGE_CHARS, _is_exact_message
            available = []
            for i, it in enumerate(bundle[:16]):
                exact = _is_exact_message(it)
                available.append({"obs_id": str(it.get("iid") or f"ob{i}"),
                                  "channel": str(it.get("channel", ""))[:40],
                                  "source": str(it.get("source", ""))[:60],
                                  "summary": str(it.get("content", ""))[
                                      :EXACT_MESSAGE_CHARS if exact else 300],
                                  "urgency": str(it.get("urgency", ""))[:20],
                                  "interrupting": bool(it.get("interrupting")),
                                  "exact_realized_message": exact})
            return available, "delivered_observation_bundle"
        recent = list(reversed(view.observed_events))[:engine.config.prompt_events]
        return ([{"obs_id": str(e.get("event_id") or e.get("iid") or f"ev{i}"),
                  "channel": str(e.get("channel", e.get("etype", "")))[:40],
                  "source": str(e.get("source", ""))[:60],
                  "summary": str(e.get("content") or e.get("situation")
                                 or e.get("etype") or "")[:300]}
                 for i, e in enumerate(recent) if isinstance(e, dict)],
                "recent_view_items(no delivery bundle on this route; recorded)")

    def _snapshot_for(self, view, state, engine) -> tuple:
        key = (view.actor_id, str(getattr(state, "hypothesis_id", "") or ""))
        with self._lock:
            hit = self._snapshots.get(key)
        if hit is not None:
            self.prompt_manifest.record_snapshot(hit[0], reused=True)
            return hit
        try:
            from swm.world_model_v2.scheduled_facts import public_facts_lines
            fact_lines = public_facts_lines(engine.config.public_facts or [])
        except Exception:  # noqa: BLE001
            fact_lines = []
        snap = ActorContextSnapshot.build(view=view, state=state,
                                          public_facts_lines=fact_lines,
                                          structural_frame=engine.config.structural_frame)
        baseline = {s: copy.deepcopy(getattr(state, s, None)) for s in
                    ("current_private_beliefs", "personal_condition", "organizational_pressures",
                     "current_goals", "relationships", "beliefs_about_others",
                     "unresolved_uncertainties", "important_memories")} if state is not None \
            else {}
        with self._lock:
            self._snapshots.setdefault(key, (snap, baseline))
        self.prompt_manifest.record_snapshot(snap, reused=False)
        return self._snapshots[key]

    @staticmethod
    def _changed_rows(state, baseline: dict) -> list:
        """Branch drift since the cohort snapshot — the delta's private-state section."""
        rows = []
        if state is None or not baseline:
            return rows
        for section, base in baseline.items():
            cur = getattr(state, section, None)
            if cur == base:
                continue
            if section == "important_memories":
                new = [m for m in (cur or []) if m not in (base or [])]
                rows.extend(f"new memory ({_date(m.get('at'))}): {m.get('memory')}"
                            for m in new[-6:] if isinstance(m, dict))
            elif isinstance(cur, dict):
                for k, v in cur.items():
                    if (base or {}).get(k) != v:
                        rows.append(f"{section}.{k}: {v}")
            elif isinstance(cur, list):
                rows.extend(f"{section}: {v}" for v in cur if v not in (base or []))
            else:
                rows.append(f"{section}: {cur}")
        return rows[:12]

    def _replicate_for(self, world) -> int:
        """Deterministic particle→replicate assignment when replicates > 1 (explicit
        experiments only; the lean default is 1 — equivalent contexts share one draw)."""
        n = self.config.behavioral_replicates_per_decision_context
        if n <= 1:
            return 0
        try:
            return int(getattr(world, "particle_index", 0) or 0) % n
        except (TypeError, ValueError):
            return 0

    # ---- the main entry, called from QualitativeActorPolicyRuntime.decide ------------
    def decision_for(self, runtime, world, view, state, situation: str, menu: list,
                     decision: dict, actor_id: str, seed: int, obstacle: str = ""):
        """Returns (qd, cog, meta) — or raises exactly what the canonical path would raise.
        qd=None never happens: every path either produces a decision, defers honestly, or
        escalates into the canonical staged pipeline (which itself raises on hard failure)."""
        if not self._armed:
            raise RuntimeError("lean research-first invariant: decision requested before "
                               "arm_actor_calls (research stages incomplete)")
        engine = runtime.engine
        branch_id = str(getattr(world, "branch_id", ""))
        at = float(getattr(getattr(world, "clock", None), "now", 0.0) or 0.0)
        day = _date(view.observed_time)
        available, availability_rule = self._availability(view, decision, engine)
        prior_dict = None
        prior = self._load_prior(world, actor_id)
        if prior is not None:
            prior_dict = {"action": prior.chosen_action, "decided_day": prior.decided_day,
                          "act_or_wait": prior.act_or_wait,
                          "revisit": prior.revisit}
        ctx = self._builder(engine).build(
            view=view, state=state, situation=situation, menu=menu, decision=decision, day=day,
            replicate_index=self._replicate_for(world), prior_decision=prior_dict, world=world,
            obstacle=obstacle)
        meta = {"context_hash": ctx.signature()[:16], "availability_rule": availability_rule}
        # ---- deterministic prechecks (never on obstacle-revision calls) -------------
        if self.config.prechecks and not obstacle:
            verdict = self._precheck(ctx, state, view, menu, decision, world, actor_id, day)
            if verdict.skip:
                qd = self._noop_decision(actor_id, verdict, ctx)
                return qd, None, {**meta, "skip_reason": verdict.reason}
        # ---- equivalence cache ------------------------------------------------------
        if not self.config.decision_cache:
            return self._fresh(runtime, world, view, state, situation, menu, decision,
                               actor_id, seed, ctx, available, day, obstacle, meta,
                               store=False)
        key = self.cache.key_for(ctx)
        template = self.cache.get(key)
        if template is not None:
            got = self._reuse(template, world, view, actor_id, branch_id, at, available,
                              decision, ctx)
            if got is not None:
                qd, cog = got
                self.avoided.record(reason="equivalent_decision_context", actor_id=actor_id,
                                    detail=f"served by branch {template.source_branch}",
                                    branch_id=branch_id)
                return qd, cog, {**meta, "cache": "hit"}
            # revalidation/reassembly failed on the receiving branch → fresh call
        role, ev = self.cache.single_flight.begin(key)
        if role == "waiter":
            ev.wait(timeout=300)
            template = self.cache.get(key)
            if template is not None:
                got = self._reuse(template, world, view, actor_id, branch_id, at, available,
                                  decision, ctx)
                if got is not None:
                    qd, cog = got
                    self.avoided.record(reason="equivalent_decision_context",
                                        actor_id=actor_id,
                                        detail="single-flight wait served by leader",
                                        branch_id=branch_id)
                    return qd, cog, {**meta, "cache": "single_flight_wait"}
            role, ev = self.cache.single_flight.begin(key)   # leader failed → controlled retry
            if role == "waiter":                              # another retry leader exists
                ev.wait(timeout=300)
                template = self.cache.get(key)
                if template is not None:
                    got = self._reuse(template, world, view, actor_id, branch_id, at,
                                      available, decision, ctx)
                    if got is not None:
                        qd, cog = got
                        return qd, cog, {**meta, "cache": "single_flight_wait_retry"}
                role = "leader_after_double_wait"
        try:
            return self._fresh(runtime, world, view, state, situation, menu, decision,
                               actor_id, seed, ctx, available, day, obstacle, meta, store=True,
                               store_key=key)
        finally:
            self.cache.single_flight.finish(key)

    # ---- fresh decision (one-call, escalating to the staged pipeline on record) ------
    def _fresh(self, runtime, world, view, state, situation, menu, decision, actor_id, seed,
               ctx: DecisionRelevantContext, available, day, obstacle, meta, *, store: bool,
               store_key: str = None):
        engine = runtime.engine
        branch_id = str(getattr(world, "branch_id", ""))
        at = float(getattr(getattr(world, "clock", None), "now", 0.0) or 0.0)
        attention_context = self._attention_context(decision, state, world, actor_id)
        escalation_reason = ""
        if not self.config.one_call_cognition:
            escalation_reason = "one_call_disabled_by_config"
        elif self.config.escalate_on_complex_menu and len(menu or []) > COMPLEX_MENU_THRESHOLD:
            escalation_reason = f"complex_action_set:{len(menu)}_options"
        outcome = None
        if not escalation_reason:
            outcome = self._one_call(runtime, world, view, state, situation, menu, decision,
                                     actor_id, ctx, available, day, obstacle,
                                     attention_context)
            if outcome.escalated:
                escalation_reason = outcome.escalation_reason
            elif outcome.qd is not None and \
                    str(outcome.cog.interpretation.get("missing_decisive_fact", "")).strip():
                # the actor names a missing decisive fact — allowed escalation ground; the
                # staged pipeline gives interpretation/search their own full calls
                escalation_reason = "actor_identified_missing_decisive_fact"
        if escalation_reason and (outcome is None or outcome.qd is None
                                  or "missing_decisive_fact" in escalation_reason):
            self.escalations.append({"actor_id": actor_id, "branch_id": branch_id,
                                     "reason": escalation_reason})
            qd, cog = self._staged_full_pipeline(runtime, world, view, state, situation, menu,
                                                 decision, actor_id, seed, obstacle)
        else:
            qd, cog = outcome.qd, outcome.cog
            self.one_call_successes += 1
        if qd is None:
            self.cache.record_failure()               # never cached
            return qd, cog, {**meta, "escalation": escalation_reason}
        qd._lean_ctx = ctx
        if store and store_key is not None:
            self._store_template(store_key, qd, cog, ctx, outcome, branch_id,
                                 escalation_reason)
        return qd, cog, {**meta, "cache": "miss_fresh_call",
                         **({"escalation": escalation_reason} if escalation_reason else {})}

    def _one_call(self, runtime, world, view, state, situation, menu, decision, actor_id,
                  ctx, available, day, obstacle, attention_context) -> OneCallOutcome:
        engine = runtime.engine
        branch_id = str(getattr(world, "branch_id", ""))
        at = float(getattr(getattr(world, "clock", None), "now", 0.0) or 0.0)
        snap, baseline = self._snapshot_for(view, state, engine)
        ctx_rng = _random.Random(context_rng_seed(ctx.signature(),
                                                  replicate_index=ctx.replicate_index))
        retrieved = self._pre_call_retrieval(world, actor_id, branch_id, at, ctx_rng)
        canon = self.canonical_available(available)
        delta = ActorDecisionDelta.build(
            day=day, situation=self._situation_text(situation, obstacle),
            observations=canon,
            working_memory=ctx.working_memory, retrieved=retrieved,
            changed_state_rows=self._changed_rows(state, baseline),
            resources=ctx.resources, action_history=ctx.action_history,
            menu_lines=[m.get("line", "") for m in (menu or [])[:engine.config.max_menu]],
            prior_decision_note=self._prior_note(world, actor_id), obstacle=obstacle)
        self.prompt_manifest.record_call(
            prefix_chars=snap.n_chars, delta_chars=delta.n_chars,
            full_equivalent_chars=4 * (snap.n_chars + delta.n_chars))  # staged ≈ 4 full renders
        budgeted = engine._budgeted(actor_id=actor_id, branch_id=branch_id,
                                    stage="lean_one_call")
        return run_one_call(world=world, actor_id=actor_id, branch_id=branch_id, at=at, day=day,
                            available=canon, snapshot_rendered=snap.rendered,
                            delta_rendered=delta.rendered, attention_context=attention_context,
                            menu_lines=[m.get("line", "") for m in (menu or [])],
                            ctx_seed=context_rng_seed(ctx.signature(),
                                                      replicate_index=ctx.replicate_index),
                            budgeted_llm=budgeted)

    def _staged_full_pipeline(self, runtime, world, view, state, situation, menu, decision,
                              actor_id, seed, obstacle):
        """The recorded escalation path: EXACTLY the full-fidelity staged cognition + decision
        call. Raises ActorDecisionUnavailable exactly as full fidelity does."""
        cog = runtime._run_bounded_cognition(world, view, state, decision, actor_id, seed, menu)
        qd = runtime.engine.decide(view, state, situation, menu, obstacle=obstacle,
                                   cognition=cog)
        return qd, cog

    # ---- reuse ----------------------------------------------------------------------
    def _reuse(self, template: ActorDecisionTemplate, world, view, actor_id, branch_id, at,
               available, decision, ctx):
        """Receiving-branch reconstruction: deep-copied decision + cognition rebuilt from the
        immutable LLM payload with THIS branch's deterministic memory stages. Returns None when
        reassembly or revalidation fails (→ fresh call; the hit is counted invalidated)."""
        revalidation = {"ok": True,
                        "trigger_match": "structural (projection equality)",
                        "authority_match": "structural (projection equality)",
                        "action_set_match": "structural (projection equality)",
                        "feasibility": "re-run downstream on this branch (perceived + actual)",
                        "invalidation_conditions": "prior-decision layer re-checked pre-cache"}
        try:
            from swm.engine.grounding import parse_json
            qd_dict, cert = self.cache.reuse(template, receiving_branch=branch_id,
                                             revalidation=revalidation)
            qd = QualitativeDecision(**{k: v for k, v in qd_dict.items()
                                        if k in QualitativeDecision.__dataclass_fields__})
            payload = parse_json(template.response)
            if not isinstance(payload, dict):
                raise ValueError("stored template response no longer parses")
            ctx_rng = _random.Random(context_rng_seed(ctx.signature(),
                                                      replicate_index=ctx.replicate_index))
            self._pre_call_retrieval(world, actor_id, branch_id, at, ctx_rng)
            canon = self.canonical_available(available)
            cog, qd_check = assemble_from_response(
                payload, raw_text=template.response, world=world,
                actor_id=actor_id, branch_id=branch_id, at=at, available=canon,
                attention_context=self._attention_context(decision, None, world, actor_id),
                ctx_seed=context_rng_seed(ctx.signature(),
                                          replicate_index=ctx.replicate_index))
            if qd_check is not None and qd_check.chosen_action != qd.chosen_action:
                raise ValueError("reassembled decision diverged from the immutable snapshot")
            qd.llm_calls = 0                                   # reuse spends no provider calls
            qd.raw_source = f"lean_cache_reuse<{template.source_branch}"
            qd._lean_ctx = ctx
            qd._lean_certificate = cert
            return qd, cog
        except Exception as e:  # noqa: BLE001 — a failed reuse must fall back to a fresh call
            with self.cache._lock:
                self.cache.invalidated_hits += 1
            self.escalations.append({"actor_id": actor_id, "branch_id": branch_id,
                                     "reason": f"reuse_reassembly_failed:{type(e).__name__}"})
            return None

    def _store_template(self, key, qd, cog, ctx, outcome, branch_id, escalation_reason):
        import json as _json
        # never cache: failures (qd None — handled by caller), truncation-salvage, numeric
        # fallback sources, or unresolved outputs requiring unvalidated repair
        if qd is None or "salvaged" in (qd.raw_source or "") or "numeric" in (qd.raw_source or ""):
            self.cache.record_failure()
            return
        response = outcome.response if (outcome is not None and not escalation_reason) else \
            _json.dumps(self._payload_from(qd, cog), default=str)
        self.cache.store(key, ActorDecisionTemplate(
            context_hash=ctx.signature(), actor_id=qd.actor_id, cohort_id=ctx.cohort_id,
            prompt_hash=(outcome.prompt_hash if outcome is not None else qd.prompt_hash or ""),
            response_hash=(outcome.response_hash if outcome is not None else ""),
            response=response,
            qd_snapshot={f: copy.deepcopy(getattr(qd, f))
                         for f in QualitativeDecision.__dataclass_fields__},
            model_fingerprint=ctx.backend_fingerprint, prompt_version=ctx.prompt_version,
            replicate_index=ctx.replicate_index, source_branch=branch_id,
            context=ctx.as_dict(),
            validation_record={"escalation": escalation_reason,
                               "one_call": not bool(escalation_reason)}))

    @staticmethod
    def _payload_from(qd: QualitativeDecision, cog) -> dict:
        """A one-call-shaped payload reconstructed from a staged-escalation result, so reuse has
        one uniform template format."""
        return {"attention": {"noticed": (cog.attention.get("noticed") if cog else []) or [],
                              "ignored": (cog.attention.get("missed") if cog else []) or []},
                "interpretation": (cog.interpretation if cog else {}) or
                                  qd.situation_interpretation,
                "considered_actions": (cog.search.get("shortlist") if cog else []) or [],
                "screened_out": (cog.search.get("options_screened_out") if cog else []) or [],
                "decision": {"chosen_action": qd.chosen_action, "act_or_wait": qd.act_or_wait,
                             "target": qd.target, "timing": qd.timing,
                             "observability": qd.observability,
                             "intended_effect": qd.intended_effect,
                             "linked_actions": qd.linked_actions, "revisit": qd.revisit},
                "decision_summary": qd.decision_summary,
                "actor_state_update": qd.actor_state_update}

    # ---- prechecks / prior decisions -------------------------------------------------
    def _precheck(self, ctx, state, view, menu, decision, world, actor_id, day
                  ) -> PrecheckVerdict:
        state_with_prior = self._state_prior_proxy(world, actor_id, state)
        verdict = precheck(ctx=ctx, state=state_with_prior, view=view, menu=menu,
                           decision=decision, policy=self.invalidation,
                           prior_ctx=self._prior_ctx.get((str(getattr(world, "branch_id", "")),
                                                          actor_id)),
                           as_of=day)
        if verdict.classification is not None:
            self.avoided.record_classification(verdict.classification)
        if verdict.skip:
            self.avoided.record(reason=verdict.reason, actor_id=actor_id,
                                detail=verdict.detail,
                                branch_id=str(getattr(world, "branch_id", "")))
        return verdict

    class _PriorProxy:
        def __init__(self, prior):
            self._lean_prior = prior.as_dict() if prior is not None else None

    def _state_prior_proxy(self, world, actor_id, state):
        prior = self._load_prior(world, actor_id)
        proxy = self._PriorProxy(prior)
        return proxy if prior is not None else state

    def _load_prior(self, world, actor_id) -> PriorDecisionValidity | None:
        try:
            ent = (world.entities or {}).get(actor_id)
            if ent is None:
                return None
            raw = ent.value("latent_state", key="lean_prior_decision", default=None)
            if isinstance(raw, dict) and raw.get("context_signature"):
                return PriorDecisionValidity.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _prior_note(self, world, actor_id) -> str:
        p = self._load_prior(world, actor_id)
        if p is None:
            return ""
        return (f"on {p.decided_day} you decided: {p.chosen_action} ({p.act_or_wait}); it "
                f"stands unless something material changed")

    def after_execute(self, world, action, qd, state):
        """Post-execute seam: persist THIS branch's standing decision + processed facts."""
        ctx = getattr(qd, "_lean_ctx", None)
        if ctx is None:
            return
        try:
            from swm.world_model_v2.state import F
            payload = ctx.trigger.get("payload_facts") or []
            prior = PriorDecisionValidity(
                context_signature=ctx.signature(), chosen_action=qd.chosen_action,
                act_or_wait=qd.act_or_wait, decided_day=ctx.day,
                revisit=dict(qd.revisit or {}),
                processed_fact_ids=sorted({f.get("fact_id") for f in payload}
                                          | {o.get("fact_id") for o in ctx.observations}))
            ent = world.entity(action.actor_id)
            ent.set("latent_state", F(prior.as_dict(), status="derived",
                                      method="lean_prior_decision",
                                      updated_at=world.clock.now), key="lean_prior_decision")
            self._prior_ctx[(str(getattr(world, "branch_id", "")), action.actor_id)] = ctx
        except Exception:  # noqa: BLE001 — bookkeeping must never break execution
            pass

    # ---- misc -----------------------------------------------------------------------
    @staticmethod
    def _situation_text(situation: str, obstacle: str) -> str:
        return str(situation)[:400] or "a decision point"

    @staticmethod
    def _attention_context(decision, state, world, actor_id) -> dict:
        return {"focus": str((decision or {}).get("situation", ""))[:120],
                "workload": (state.organizational_pressures if state is not None else ""),
                "condition": (state.personal_condition if state is not None else ""),
                "obligations": [t.get("task", "") for t in
                                BC.load_memory(world, actor_id).unresolved_tasks[:4]]}

    def _pre_call_retrieval(self, world, actor_id, branch_id, at, ctx_rng) -> list:
        """Deterministic (context-seeded) episodic retrieval BEFORE the one call — the same
        imperfect-memory mechanism, drawn per decision context instead of per particle."""
        try:
            mem = BC.load_memory(world, actor_id)
            wm = BC.load_working_memory(world, actor_id)
            ret = BC.memory_retrieval_stage(mem=mem, wm=wm, actor_id=actor_id,
                                            branch_id=branch_id, at=at, rng=ctx_rng)
            BC.store_memory(world, mem)
            BC.store_working_memory(world, wm)
            return [{"content": str(m.get("content", ""))[:220]}
                    for m in ret.get("retrieved", [])][:6]
        except Exception:  # noqa: BLE001
            return []

    def _noop_decision(self, actor_id: str, verdict: PrecheckVerdict,
                       ctx: DecisionRelevantContext) -> QualitativeDecision:
        """The deterministic layer determined NO decision exists — the standing course
        continues. This is never a predicted human choice: it is the absence of a new one."""
        qd = QualitativeDecision(
            actor_id=actor_id, chosen_action="wait", act_or_wait="wait",
            decision_summary=f"[lean deterministic layer] {verdict.reason}: {verdict.detail}",
            raw_source=f"lean_precheck:{verdict.reason}")
        qd.llm_calls = 0
        qd._lean_ctx = ctx
        return qd

    # ---- attach + manifest ------------------------------------------------------------
    def attach_to_runtime(self, runtime, *, plan=None, decision_context_hint: str = ""):
        """Bind the controller into a constructed QualitativeActorPolicyRuntime: the decide/post
        seams, the lean cohort hypothesizer (same memo + assignment law), and the consequence
        cache around the consequence backend."""
        runtime.lean_controller = self
        engine = runtime.engine
        cfg = engine.config
        engine.hypothesizer = LeanCohortHypothesizer(
            (cfg.hypothesis_llm or cfg.llm) if cfg.llm_hypotheses else None,
            k=cfg.n_hypotheses, ceiling=self.config.cohort_ceiling,
            manifest=self.cohort_manifest, decision_context_hint=decision_context_hint,
            structural_frame=cfg.structural_frame, integrity=cfg.integrity)
        if self.config.consequence_cache and runtime.consequence_llm is not None:
            self.consequence_cache = ConsequenceProgramCache(
                runtime.consequence_llm, fingerprint=backend_fingerprint(cfg.llm),
                ledger=self.ledger)
            runtime.consequence_llm = self.consequence_cache
        return runtime

    def manifest(self) -> dict:
        by_reason = {}
        for e in self.escalations:
            r = e["reason"].split(":")[0]
            by_reason[r] = by_reason.get(r, 0) + 1
        return {"version": CONTROLLER_VERSION, "config": self.config.as_dict(),
                "armed": self._armed,
                "research_ledger": getattr(self, "research_ledger", {}),
                "one_call_successes": self.one_call_successes,
                "escalations_total": len(self.escalations),
                "escalations_by_reason": by_reason,
                "escalation_sample": self.escalations[:40],
                "frontier_skips": self.frontier_skips,
                "decision_cache": self.cache.manifest(),
                "avoided_calls": self.avoided.as_dict(),
                "cohorts": self.cohort_manifest.as_dict(),
                "prompts": self.prompt_manifest.as_dict(),
                "consequence_cache": (self.consequence_cache.manifest()
                                      if self.consequence_cache is not None else None)}
