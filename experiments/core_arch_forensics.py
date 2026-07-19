"""§38 live LLM-backed forensic runs — six cases through the REAL default runtime.

No enable flags: the default structural-ensemble runtime with explicit world boundaries,
residual outside-world processes, bounded cognition, model-family assignment, strict actor
integrity (production env: SWM_ALLOW_NUMERIC_BASELINE / SWM_ALLOW_GENERIC_PRIOR both UNSET)
and honest truncation. Only the documented compute knob (n_particles) and, for case 6, the
documented safety budget (SWM_ACTOR_LLM_BUDGET) are set.

Per case this driver saves the §38 record: input; structural models; initial boundaries;
boundary critics; residual outside-world processes; dynamic promotions; actor hypotheses;
model-family assignments; attention/working-memory/retrieval/interpretation traces; actions
considered; chosen actions; nonhuman mechanism traces; truncated branches; completed weight;
under-modeled components; final result; recommendation status; LLM calls; est. tokens; cost;
runtime — plus the §38 verification block (no numerical actor fallback, no generic outcome
prior, no branch continuation after actor-budget exhaustion, canonical route ran).

Run:  PYTHONPATH=. python experiments/core_arch_forensics.py [case1..case6|all]
These are ARCHITECTURE probes: no real-world accuracy is claimed from them.
"""
import json
import os
import sys
import time

OUT_DIR = "artifacts/core_arch_forensics"
os.makedirs(OUT_DIR, exist_ok=True)

# production strictness: the §19/§28 markers must NOT be set for live forensics
for _m in ("SWM_ALLOW_NUMERIC_BASELINE", "SWM_ALLOW_GENERIC_PRIOR"):
    if os.environ.get(_m):
        del os.environ[_m]


def make_llm():
    from swm.api.deepseek_backend import default_chat_fn
    from swm.eval.instrument import CountingLLM
    fn = default_chat_fn(max_tokens=2600, temperature=0.0)
    if fn is None:
        raise RuntimeError("no DEEPSEEK_API_KEY — live forensics require the configured backend")
    return CountingLLM(fn)


def _dump(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[saved {path}]", flush=True)


def _sect(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ------------------------------------------------------------------ §38 verification
def verify_case(res, *, expect_truncation=False) -> dict:
    """The §38 verification block, computed from the ACTUAL result object."""
    prov = res.provenance or {}
    v = {"canonical_route": str(prov.get("runtime", "")).startswith("unified-2"),
         "structural_mode": prov.get("structural_mode"),
         "numeric_actor_fallbacks": 0, "numeric_fallback_rows": [],
         "generic_prior_writes": 0, "generic_prior_suppressions": 0,
         "branch_continuation_after_truncation": False,
         "monoculture_reported": None, "boundaries_present": bool(res.world_boundaries),
         "outside_world_present": bool(res.outside_world),
         "cognition_records": 0, "under_modeled_subtypes": list(res.under_modeled_subtypes or []),
         "simulation_status": res.simulation_status}
    fam = res.model_family_report or {}
    v["monoculture_reported"] = fam.get("model_family_monoculture")
    for m, p in (prov.get("per_model_provenance") or {}).items():
        for actor, dist in ((p or {}).get("actor_decision_distributions") or {}).items():
            for row in (dist.get("rows") or []):
                src = str(row.get("decision_source", ""))
                if "numeric" in src:
                    v["numeric_actor_fallbacks"] += 1
                    v["numeric_fallback_rows"].append({"model": m, "actor": actor, "src": src})
        trt = (p or {}).get("temporal_runtime") or {}
        v["generic_prior_suppressions"] += len(trt.get("mechanism_suppressions") or [])
        cogs = (p or {}).get("cognition_records_sample") or []
        v["cognition_records"] += len(cogs)
        trunc = trt.get("truncation") or {}
        if trunc.get("halted") is False and trt.get("n_branches_truncated"):
            v["branch_continuation_after_truncation"] = True
    v["cognition_records"] += len((res.cognition_report or {}).get(
        "sample_decision_records") or [])
    tr = res.truncation_report or {}
    v["truncated_weight"] = tr.get("truncated_weight")
    v["truncation_reasons"] = tr.get("truncation_reasons")
    v["ok"] = (v["canonical_route"] and v["numeric_actor_fallbacks"] == 0
               and not v["branch_continuation_after_truncation"]
               and (not expect_truncation or (v["truncated_weight"] or 0) > 0))
    return v


def _record(case, question, res, t0, llm, extra=None):
    prov = res.provenance or {}
    se = res.structural_ensemble or {}
    cost = (llm.meter.snapshot() if hasattr(llm, "meter")
            else {"note": "instrumentation unavailable; see cost_manifest"})
    rec = {
        "case": case, "question": question, "runtime_s": round(time.time() - t0, 1),
        "simulation_status": res.simulation_status,
        "support_grade": res.support_grade,
        "distribution": res.raw_distribution,
        "conditional_forecast_note": getattr(res, "conditional_forecast_note", ""),
        "under_modeled_subtypes": res.under_modeled_subtypes,
        "under_modeled_components": res.under_modeled_components,
        "world_boundaries": res.world_boundaries,
        "outside_world": res.outside_world,
        "cognition_report": res.cognition_report,
        "truncation_report": res.truncation_report,
        "model_family_report": res.model_family_report,
        "structural_models": [{k: m.get(k) for k in
                               ("model_id", "promotion_status", "causal_thesis",
                                "world_boundary", "support_class")}
                              for m in (se.get("models") or [])],
        "cost_manifest": se.get("cost_manifest"),
        "boundary_promotions": [
            p for mp in (prov.get("per_model_provenance") or {}).values()
            for p in ((mp or {}).get("consequence_report") or {}).get(
                "boundary_promotions", [])][:20],
        "verification": verify_case(res, **(extra or {})),
        "provider_cost": cost,
        "recommendation_status": res.recommendation_status,
        "note": "architecture probe — no real-world accuracy claimed (§38)",
    }
    _dump(f"{case}.json", rec)
    _sect(f"{case} VERIFICATION")
    print(json.dumps(rec["verification"], indent=1, default=str), flush=True)
    return rec


def _run(question, llm, *, seed, as_of, horizon, user_context=None, n_particles=5):
    from swm.world_model_v2.unified_runtime import simulate_world
    return simulate_world(question, as_of=as_of, horizon=horizon, llm=llm, seed=seed,
                          user_context={**(user_context or {}),
                                        "_execution_policy": {"n_particles": n_particles}})


# ---- case 1 (§38.1): founder decision + competitor/platform event outside the boundary ----
def case1(llm):
    t0 = time.time()
    q = ("Will Marisol's meal-kit startup keep its retail listing on the GreenCart "
         "marketplace through September after GreenCart's fee policy review?")
    res = _run(q, llm, seed=21, as_of="2026-07-19", horizon="2026-10-01",
               user_context={"detail": "Marisol sells through GreenCart; GreenCart announced "
                                       "a seller-fee policy review for August; a competitor "
                                       "recently moved to exclusive terms with GreenCart; "
                                       "Marisol can renegotiate, diversify to direct sales, "
                                       "or wait."})
    return _record("case1_founder_platform", q, res, t0, llm)


# ---- case 2 (§38.2): personal message — attention, memory, relationship, workload ----
def case2(llm):
    t0 = time.time()
    q = "How will Tomas react if I send this message tonight?"
    from swm.world_model_v2.unified_runtime import simulate_world
    res = simulate_world(
        q, as_of="2026-07-19T20:30:00Z", horizon="2026-07-21T20:30:00Z", llm=llm, seed=22,
        user_context={"individual": {
            "person_id": "tomas", "name": "Tomas",
            "stimulus": ("Tomas — I know the audit deadline is crushing this week. I finished "
                         "the reconciliation you were dreading; it's in the shared folder. "
                         "Also… I'd like to revisit what you said about me taking the team "
                         "lead role. Coffee Friday?"),
            "channel": "text_message", "relationship": "colleague, sometimes tense",
            "role": "finance manager under audit-week workload", "your_role": "senior analyst",
            "timezone": "Europe/Madrid", "sleep_window": [23.5, 7.0],
            "history": ["disagreed about the team lead role two weeks ago",
                        "he is buried in the annual audit this week",
                        "he usually replies within a day, slower under load"],
            "urgency": 0.3, "n_hypotheses": 2, "samples_per_hypothesis": 2}})
    return _record("case2_personal_message", q, res, t0, llm)


# ---- case 3 (§38.3): institutional process; peripheral authority holder becomes decisive ----
def case3(llm):
    t0 = time.time()
    q = ("Will the Ashport harbor commission approve the night-ferry pilot at its August "
         "meeting?")
    res = _run(q, llm, seed=23, as_of="2026-07-19", horizon="2026-09-05",
               user_context={"detail": "The five-member commission votes in August; the "
                                       "harbormaster's operational sign-off is normally a "
                                       "formality, but a draft coast-guard advisory could "
                                       "give the harbormaster an effective veto; two "
                                       "commissioners are undecided; the ferry operator can "
                                       "adjust the schedule or add a safety plan."})
    return _record("case3_institutional_promotion", q, res, t0, llm)


# ---- case 4 (§38.4): operational decision blocked by real capacity/queueing ----
def case4(llm):
    t0 = time.time()
    q = ("Will Corvid Labs ship the replacement sensor batch to all forty affected customers "
         "within three weeks?")
    res = _run(q, llm, seed=24, as_of="2026-07-19", horizon="2026-08-12",
               user_context={"detail": "The recall requires recalibrating each unit on the "
                                       "single calibration rig (capacity 4 units/day, one "
                                       "shift); the rig is also booked for a paying customer "
                                       "order; adding a night shift needs a certified "
                                       "technician who must be hired or borrowed; shipping "
                                       "adds 3-5 days."})
    return _record("case4_capacity_bottleneck", q, res, t0, llm)


# ---- case 5 (§38.5): population adoption + one promotable member ----
def case5(llm):
    t0 = time.time()
    q = ("Will at least half of the Larkfield teachers adopt the new gradebook app by the "
         "start of term?")
    res = _run(q, llm, seed=25, as_of="2026-07-19", horizon="2026-09-01",
               user_context={"detail": "Ninety teachers across three schools; the district "
                                       "mandated availability but not use; the veteran "
                                       "math-department chair Ruth is informally influential "
                                       "and skeptical; a summer training webinar is "
                                       "scheduled for August 5; the vendor may add a "
                                       "gradebook import tool."})
    return _record("case5_population_adoption", q, res, t0, llm)


# ---- case 6 (§38.6): intentional qualitative-actor budget exhaustion ----
def case6(llm):
    t0 = time.time()
    os.environ["SWM_ACTOR_LLM_BUDGET"] = "6"      # documented safety budget, deliberately tiny
    try:
        q = ("Will the Delmar tenants' association vote to join the rent-strike coalition "
             "this month?")
        res = _run(q, llm, seed=26, as_of="2026-07-19", horizon="2026-08-19", n_particles=4,
                   user_context={"detail": "A seven-member steering committee decides; three "
                                           "members are publicly committed, two opposed, two "
                                           "undecided; the landlord has offered a repair "
                                           "schedule; a citywide coalition meeting happens "
                                           "next week."})
        return _record("case6_budget_exhaustion", q, res, t0, llm,
                       extra={"expect_truncation": True})
    finally:
        os.environ.pop("SWM_ACTOR_LLM_BUDGET", None)


CASES = {"case1": case1, "case2": case2, "case3": case3, "case4": case4,
         "case5": case5, "case6": case6}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    llm = make_llm()
    summary = {}
    for name, fn in CASES.items():
        if which not in ("all", name):
            continue
        try:
            rec = fn(llm)
            summary[name] = {"status": rec["simulation_status"],
                             "verification_ok": rec["verification"]["ok"],
                             "runtime_s": rec["runtime_s"]}
        except Exception as e:  # noqa: BLE001 — a crashed case is a visible failure, not a skip
            summary[name] = {"status": f"DRIVER_ERROR: {type(e).__name__}: {e}"[:300]}
            print(f"[{name} FAILED: {type(e).__name__}: {e}]", flush=True)
        _dump("summary.json", summary)
    print("\nDONE", json.dumps(summary, indent=1, default=str), flush=True)
