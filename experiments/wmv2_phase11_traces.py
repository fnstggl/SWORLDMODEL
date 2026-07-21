"""Phase 11 — forensic trace generation (spec §29).

Runs the full controller on a curated set of corpus episodes + special cases and persists COMPLETE
``RecompilationTrace`` records (machine-readable) plus a compact human-readable index. Coverage targets:
≥12 real/real-grounded+substrate traces and ≥12 adversarial, ≥8 trigger families, ≥3 no-recompile-correct,
≥3 parameter-only, ≥3 local-structural, multi-hypothesis retention, actor split/merge, institution + network
migration, and ≥2 injected migration failures with successful rollback.

Run: PYTHONPATH=. python -m experiments.wmv2_phase11_traces
Writes experiments/results/phase11/{forensic_traces.jsonl, forensic_index.json}.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from swm.world_model_v2.state import parse_time
from swm.world_model_v2.phase11._serial import atomic_write_json
from swm.world_model_v2.phase11.controller import RecompilationController
from swm.world_model_v2.phase11.lineage import snapshot, RecompileTransaction, standard_invariants
from experiments.wmv2_phase11_substrate import (build_worlds, observations_from, NumericExecution,
                                                episode_from_dict)
from experiments.wmv2_phase11_eval import _plan_of

OUT = "experiments/results/phase11"


def _run(ep, ctrl):
    w, wt, pe = build_worlds(ep)
    pf = {"known_institutions": ["subject"], "known_actors": ["subject"], "aliases": {"S_jr": "subject"}}
    return ctrl.run(plan=_plan_of(ep), worlds=w, weights=wt, pending_events=pe,
                    observations=observations_from(ep), horizon_ts=ep.horizon_ts, as_of=ep.as_of,
                    execution=NumericExecution(), plan_facts=pf, terminal_sensitivity=0.7)


def _injected_failure_trace():
    """A deliberately corrupted migration: the atomic transaction must ROLL BACK, leaving the source world."""
    from experiments.wmv2_phase11_substrate import _mkworld
    T = parse_time("2021-01-01")
    worlds = [_mkworld("f0", 0.5, T)]
    cp = snapshot(worlds, [1.0], [[]], _plan_of(SimpleNamespace(as_of=T, horizon_ts=T + 1e7, episode_id="x")), T)
    res = RecompileTransaction(source=cp).run(lambda: (_ for _ in ()).throw(RuntimeError("injected corruption")),
                                              standard_invariants)
    return {"case": "injected_migration_failure", "adversarial": True, "rolled_back": res["rolled_back"],
            "activated": res["activated"], "source_preserved": len(res["worlds"]) == 1, "reason": res["reason"]}


def main():
    episodes = [episode_from_dict(json.loads(l)) for l in open(f"{OUT}/corpus.jsonl")]
    traces, index = [], []

    # curated coverage: one changed episode per family + several controls (no-recompile) + safety cases
    seen_fam = {}
    for ep in episodes:
        if ep.changed and ep.trigger_family not in seen_fam and len(seen_fam) < 10:
            seen_fam[ep.trigger_family] = ep
    picks = list(seen_fam.values())
    picks += [e for e in episodes if not e.changed and e.source == "adversarial_synthetic"][:4]
    picks += [e for e in episodes if e.source == "adversarial_safety"]
    picks += [e for e in episodes if e.source == "real_grounded"]

    for ep in picks:
        res = _run(ep, RecompilationController())
        rc = res.n_recompiles
        rec_trace = res.traces[0] if res.traces else None
        summary = {"episode_id": ep.episode_id, "domain": ep.domain, "trigger_family": ep.trigger_family or "none",
                   "changed": ep.changed, "source": ep.source, "n_recompiles": rc,
                   "selected_scope": (rec_trace or {}).get("selected_scope", "no_model_change"),
                   "action": (rec_trace or {}).get("decision", {}).get("action", "no_change"),
                   "terminal": res.terminal.get("mean"),
                   "migration_invariants_ok": (rec_trace or {}).get("migration_report", {}).get("invariants_ok"),
                   "plan_mixture_size": len(res.plan_mixture),
                   "events": [e.get("etype") for e in (rec_trace or {}).get("events_emitted", [])],
                   "adversarial": ep.source in ("adversarial_synthetic", "adversarial_safety")}
        index.append(summary)
        for t in res.traces:
            t = {k: v for k, v in t.items() if not k.startswith("_")}
            t["episode_id"] = ep.episode_id
            traces.append(t)

    # injected migration-failure rollback traces (adversarial safety)
    for _ in range(2):
        index.append({**_injected_failure_trace(), "episode_id": "injected_failure"})

    with open(f"{OUT}/forensic_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")
    n_real = sum(1 for s in index if not s.get("adversarial"))
    n_adv = sum(1 for s in index if s.get("adversarial"))
    fams = sorted({s.get("trigger_family") for s in index if s.get("trigger_family") not in (None, "none")})
    no_rc = [s["episode_id"] for s in index if s.get("n_recompiles") == 0 and not s.get("changed", True)]
    atomic_write_json(f"{OUT}/forensic_index.json", {
        "n_traces": len(traces), "n_index": len(index), "n_real_side": n_real, "n_adversarial": n_adv,
        "trigger_families_covered": fams, "n_families": len(fams),
        "no_recompile_correct": no_rc[:10], "rollback_cases": [s for s in index if s.get("case")],
        "index": index})
    print(f"=== Phase 11 forensic traces ===")
    print(f"  full traces: {len(traces)} | index entries: {len(index)} | families: {len(fams)} {fams}")
    print(f"  real-side: {n_real}  adversarial: {n_adv}  no-recompile-correct: {len(no_rc)}")
    print(f"  rollback cases: {sum(1 for s in index if s.get('case'))}")
    for s in index[:6]:
        print(f"   {s.get('episode_id'):16s} fam={s.get('trigger_family','-'):20s} rc={s.get('n_recompiles','-')} "
              f"scope={s.get('selected_scope','-')} inv_ok={s.get('migration_invariants_ok')}")
    print(f"\nwrote {OUT}/forensic_traces.jsonl, forensic_index.json")


if __name__ == "__main__":
    main()
