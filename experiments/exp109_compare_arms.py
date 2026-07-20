"""EXP-109: the machine-readable five-question comparison — EXP-107 full fidelity vs EXP-108
lean adaptive, per §23 of the lean contract. Reads only the two arms' checkpoints; computes
nothing that was not measured.

Run: python -m experiments.exp109_compare_arms
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.exp101_btf3_pilot import fetch_btf3
from experiments.exp102_btf3_wmv2_full import QIDS

FF_DIR = Path("experiments/results/exp107_checkpoints")
LEAN_DIR = Path("experiments/results/exp108_checkpoints")
OUT = Path("experiments/results/exp109_comparison.json")

#: DeepSeek published prices per 1M tokens (USD) at run time — recorded as assumptions so cost
#: rows are reproducible from the token counts; tokens are the primary metric.
PRICE_ASSUMPTIONS = {"model": "deepseek-v4-flash",
                     "input_per_m_cache_miss": 0.28, "input_per_m_cache_hit": 0.028,
                     "output_per_m": 0.42,
                     "note": "cost = tokens × these unit prices; tokens are provider-reported"}


def _cost(m: dict) -> float:
    hit = m.get("provider_cache_hit_tokens") or 0
    miss = (m.get("provider_cache_miss_tokens")
            or max(0, (m.get("input_tokens") or 0) - hit))
    out = m.get("output_tokens") or 0
    return round(miss / 1e6 * PRICE_ASSUMPTIONS["input_per_m_cache_miss"]
                 + hit / 1e6 * PRICE_ASSUMPTIONS["input_per_m_cache_hit"]
                 + out / 1e6 * PRICE_ASSUMPTIONS["output_per_m"], 4)


def _load(d: Path, qid: str) -> dict:
    p = d / f"{qid}.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else {}


def _side(p) -> str:
    if p is None:
        return "no_forecast"
    return "yes" if p > 0.5 else ("no" if p < 0.5 else "exactly_half")


def compare() -> dict:
    rows = {r["question_id"]: r for r in fetch_btf3()}
    per_q, totals = [], {"ff": {}, "lean": {}}

    def _acc(side: dict, m: dict):
        for k in ("n_llm_calls", "input_tokens", "output_tokens",
                  "provider_cache_hit_tokens", "wall_clock_s"):
            side[k] = round(side.get(k, 0) + (m.get(k) or 0), 1)
        side["cost_usd"] = round(side.get("cost_usd", 0) + _cost(m), 4)

    for qid in QIDS:
        ff, lean = _load(FF_DIR, qid), _load(LEAN_DIR, qid)
        outcome = int(rows[qid]["resolution"])
        p_ff = ff.get("p_cal") if ff.get("p_cal") is not None else ff.get("p_raw")
        p_ln = lean.get("p_cal") if lean.get("p_cal") is not None else lean.get("p_raw")
        lm = lean.get("lean") or {}
        stage_ff = ff.get("llm_calls_by_stage") or {}
        stage_ln = lean.get("llm_calls_by_stage") or {}
        row = {
            "qid": qid, "question": (ff.get("question") or lean.get("question") or "")[:120],
            "outcome": outcome,
            "full_fidelity": {
                "prediction": p_ff, "status": ff.get("status"),
                "brier": None if p_ff is None else round((p_ff - outcome) ** 2, 4),
                "side_of_0.5": _side(p_ff),
                "correct_side": None if p_ff is None else bool((p_ff > 0.5) == outcome),
                "llm_calls": ff.get("n_llm_calls"), "calls_by_stage": stage_ff,
                "actor_calls": (stage_ff.get("actor_rollout") or 0) or None,
                "structural_models_generated": ff.get("structural_models_generated"),
                "structural_models_simulated": ff.get("structural_models_simulated"),
                "particles_by_model": ff.get("particles_by_model"),
                "input_tokens": ff.get("input_tokens"), "output_tokens": ff.get("output_tokens"),
                "provider_cache_hit_tokens": ff.get("provider_cache_hit_tokens"),
                "wall_clock_s": ff.get("wall_clock_s"), "cost_usd": _cost(ff) if ff else None,
                "truncation": len(ff.get("truncation") or []),
                "limitation_heads": [str(x)[:80] for x in (ff.get("limitations") or [])][:5]},
            "lean_adaptive": {
                "prediction": p_ln, "status": lean.get("status"),
                "brier": None if p_ln is None else round((p_ln - outcome) ** 2, 4),
                "side_of_0.5": _side(p_ln),
                "correct_side": None if p_ln is None else bool((p_ln > 0.5) == outcome),
                "llm_calls": lean.get("n_llm_calls"), "calls_by_stage": stage_ln,
                "actor_calls": (stage_ln.get("actor_rollout") or 0)
                + (stage_ln.get("lean_one_call") or 0) or None,
                "input_tokens": lean.get("input_tokens"),
                "output_tokens": lean.get("output_tokens"),
                "provider_cache_hit_tokens": lean.get("provider_cache_hit_tokens"),
                "wall_clock_s": lean.get("wall_clock_s"),
                "cost_usd": _cost(lean) if lean else None,
                "truncation": len(lean.get("truncation") or []),
                "limitation_heads": [str(x)[:80] for x in (lean.get("limitations") or [])][:5],
                **{k: lm.get(k) for k in
                   ("one_call_successes", "escalations", "unique_decision_contexts",
                    "decision_cache_hits", "invalidated_cache_hits", "largest_context_reuse",
                    "actor_calls_avoided_total", "actor_calls_avoided_by_reason",
                    "execution_classifications", "largest_cohorts", "under_modeled_actors",
                    "prompt_chars_sent", "prompt_chars_saved", "consequence_compile_calls",
                    "consequence_cache_reuses", "structural", "particle_stopping",
                    "stability_signals", "stability_replicate", "frontier_skips")}},
        }
        d_calls = (ff.get("n_llm_calls") or 0) - (lean.get("n_llm_calls") or 0)
        row["deltas"] = {
            "llm_calls_removed": d_calls if ff and lean else None,
            "prediction_moved": (None if p_ff is None or p_ln is None
                                 else round(p_ln - p_ff, 4)),
            "side_changed": (None if p_ff is None or p_ln is None
                             else _side(p_ff) != _side(p_ln)),
            "wall_clock_saved_s": (round((ff.get("wall_clock_s") or 0)
                                         - (lean.get("wall_clock_s") or 0), 1)
                                   if ff and lean else None),
            "input_tokens_saved": ((ff.get("input_tokens") or 0)
                                   - (lean.get("input_tokens") or 0)) if ff and lean else None,
            "output_tokens_saved": ((ff.get("output_tokens") or 0)
                                    - (lean.get("output_tokens") or 0)) if ff and lean else None}
        per_q.append(row)
        if ff:
            _acc(totals["ff"], ff)
        if lean:
            _acc(totals["lean"], lean)

    def _brier(side: str) -> dict:
        pairs = [(r[side]["prediction"], r["outcome"]) for r in per_q
                 if r[side].get("prediction") is not None]
        if not pairs:
            return {"n_scored": 0, "brier": None, "accuracy_at_0.5": None}
        return {"n_scored": len(pairs),
                "brier": round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4),
                "accuracy_at_0.5": round(sum((p > 0.5) == y for p, y in pairs) / len(pairs), 4)}

    out = {"experiment": "EXP-109 §23 comparison: full_fidelity (EXP-107) vs lean_adaptive "
                         "(EXP-108) on the five frozen BTF-3 questions",
           "price_assumptions": PRICE_ASSUMPTIONS,
           "totals": {"full_fidelity": {**totals["ff"], **_brier("full_fidelity")},
                      "lean_adaptive": {**totals["lean"], **_brier("lean_adaptive")}},
           "per_question": per_q}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["totals"], indent=1, default=str))
    return out


if __name__ == "__main__":
    compare()
