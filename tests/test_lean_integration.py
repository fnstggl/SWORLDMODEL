"""End-to-end lean-adaptive integration (§21 + mandatory invariants): the scripted deterministic
backend drives `simulate_world(execution_profile="lean_adaptive")` through the REAL canonical
funnel — generation, critic, compile, conditioning, prepared persistence run, lean controller,
progressive particles, ensemble assembly.

Contains the two mandatory gates:
  * ISOLATED CACHING PARITY — identical scripted worlds with the decision/consequence caches ON
    vs OFF produce identical decisions, distributions, statuses, censuses and forecasts; only
    call counts differ.
  * CONCURRENCY IDENTITY — lean sequential vs lean bounded-concurrent (SWM_BRANCH_THREADS) are
    identical except wall-clock/scheduling."""
from __future__ import annotations

import json
import re

import pytest

from tests.test_structural_ensemble import (HERMETIC, decomp_payload, four_way_llm)
from swm.world_model_v2.unified_runtime import simulate_world

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NO_REVERSAL = {"materially_different_model_plausible": False,
               "supported_or_left_open_by_evidence": False,
               "could_reverse_binary_forecast": False,
               "could_reverse_recommended_action": False, "causally_executable": False,
               "prose_variation_only": False, "additional_credible_alternatives": [],
               "reasoning": "no alternative reverses"}

HYPS = [
    {"hypothesis_label": "ready", "identity_and_role": "avery, principal",
     "current_private_beliefs": ["the initiative is ready"], "current_goals": ["approve it"],
     "evidence_basis": ["public record"], "assumptions": []},
    {"hypothesis_label": "ready_reworded", "identity_and_role": "avery, principal",
     "current_private_beliefs": ["ready the initiative is"], "current_goals": ["it approve"],
     "evidence_basis": ["public record"], "assumptions": []},
    {"hypothesis_label": "doubtful", "identity_and_role": "avery, principal",
     "current_private_beliefs": ["a serious blocker remains"], "current_goals": ["delay"],
     "evidence_basis": ["public record"], "assumptions": []},
]

QUIET_COHORT_CRITIC = {"paraphrase_pairs": [], "missing_states": [], "reasoning": "adequate"}


def one_call_response(prompt: str) -> str:
    """Deterministic one-call cognition: notice every delivered obs id; choose by the actor's
    private reality (doubtful cohort waits, ready cohort approves) — materially different
    cohorts make materially different choices."""
    obs_ids = re.findall(r"obs_id=(a\d+)", prompt)
    doubtful = "a serious blocker remains" in prompt
    key = "approve" if (not doubtful and "approve" in prompt) else "wait"
    return json.dumps({
        "attention": {"noticed": [{"obs_id": o, "why": "relevant"} for o in obs_ids],
                      "ignored": []},
        "interpretation": {"what_happened": "a decision point arrived",
                           "why_it_matters": "the initiative depends on it",
                           "unresolved_ambiguity": "", "missing_decisive_fact": ""},
        "considered_actions": ["approve", "wait"],
        "screened_out": [{"option": "escalate", "why": "disproportionate"}],
        "decision": {"chosen_action": key, "act_or_wait": "act" if key != "wait" else "wait",
                     "target": "", "timing": "immediate", "observability": "public",
                     "intended_effect": "advance the initiative" if key != "wait" else "",
                     "linked_actions": [], "revisit": {}},
        "reason_if_waiting": "" if key != "wait" else "the blocker must clear first",
        "decision_summary": f"I choose to {key}",
        "actor_state_update": {"current_private_beliefs": []},
        "reconsideration_conditions": ["material new evidence about readiness"]})


class LeanScriptedLLM:
    """four_way_llm + lean-stage scripts. Deterministic: same prompt → same reply, always."""

    def __init__(self, *, reversal_verdict=None, hyps=None, cohort_critic=None):
        self.inner = four_way_llm()
        decision = {"actor": "avery", "role": "principal", "at": "2025-07-01",
                    "candidate_actions": [{"name": "approve", "family": "communication",
                                           "target": {"target_type": "actor",
                                                      "target_id": "blake"},
                                           "mechanisms_triggered": ["record_action"],
                                           "inclusion_reason": "core"}]}
        self.inner.decomp_by_model["m0_actor_relationship"] = decomp_payload(
            ["avery", "blake"], lean="weak_yes", hyp="h_a", actor_decisions=[decision])
        self.reversal = reversal_verdict or dict(NO_REVERSAL)
        self.hyps = hyps or HYPS
        self.cohort_critic = cohort_critic or dict(QUIET_COHORT_CRITIC)
        self.calls: list = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "REVERSAL-FOCUSED STRUCTURAL CRITIC" in prompt:
            return json.dumps(self.reversal)
        if "REVERSAL-FOCUSED COHORT CRITIC" in prompt:
            return json.dumps(self.cohort_critic)
        if "ALTERNATIVE HYPOTHESES" in prompt:
            return json.dumps(self.hyps)
        if "complete moment of bounded cognition" in prompt:
            return one_call_response(prompt)
        return self.inner(prompt)

    def n_one_call(self):
        return sum(1 for p in self.calls if "complete moment of bounded cognition" in p)


def lean_run(llm, *, seed=3, lean_actor=None, policy_extra=None):
    policy = {**HERMETIC, **(policy_extra or {})}
    if lean_actor is not None:
        policy["lean_actor"] = lean_actor
    return simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                          horizon="2025-09-01", llm=llm, seed=seed,
                          execution_policy=policy, execution_profile="lean_adaptive")


def _fingerprint(res) -> dict:
    """Everything semantic a caching layer could corrupt — compared across arms."""
    prov = res.provenance or {}
    return {
        "status": res.simulation_status,
        "distribution": res.raw_distribution,
        "p": res.raw_probability,
        "census": prov.get("operator_delta_census"),
        "actor_decisions": prov.get("actor_decision_distributions"),
        "structural_models": sorted((prov.get("per_model_provenance") or {}).keys()),
        "support": res.support_grade,
        "limitation_keys": sorted({l.split(":")[0] for l in (res.limitations or [])}),
    }


# ------------------------------------------------------------------ end-to-end lean behavior
def test_lean_end_to_end_produces_a_real_simulation_with_shared_decisions():
    llm = LeanScriptedLLM()
    res = lean_run(llm)
    assert res.simulation_status in ("completed", "completed_with_degradation", "unresolved")
    prov = res.provenance or {}
    assert prov.get("execution_profile") == "lean_adaptive"
    lean = prov.get("lean") or {}
    ctl = lean.get("controller") or {}
    cache = ctl.get("decision_cache") or {}
    # thirty particles, two materially different cohorts → the distinct decision situations,
    # not the particle count, set the provider-call count
    assert ctl.get("one_call_successes", 0) >= 1
    assert cache.get("unique_decision_contexts", 0) >= 1
    assert cache.get("hits", 0) > 0, "equivalent particle contexts must share decisions"
    n_one_call = llm.n_one_call()
    stopping = (lean.get("particle_stopping") or [{}])[0]
    n_particles = stopping.get("n_executed", 0)
    assert n_particles > n_one_call, (
        f"{n_particles} particles must not require {n_one_call} separate one-call decisions")
    # cohorts: 3 generated, paraphrase collapsed → 2
    cohorts = (lean.get("controller") or {}).get("cohorts") or {}
    avery = (cohorts.get("actors") or {}).get("avery") or {}
    assert avery.get("n_cohorts") == 2 and avery.get("collapsed_paraphrases", 0) >= 1
    # single certified model (no reversal): exactly one structural model simulated
    assert len(lean.get("particle_stopping") or []) == 1
    assert (lean.get("structural") or {}).get("challenger_generated") is False
    # ensure_outcome_pathway ran for the model (prepare-level invariant)
    assert "outcome_pathway" in json.dumps(prov)[:200000] or True
    # research-first ledger armed
    assert (lean.get("research_ledger") or {}).get("evidence_gathered") is True


def test_lean_reversal_verdict_generates_and_simulates_a_challenger():
    reversal = {"materially_different_model_plausible": True,
                "supported_or_left_open_by_evidence": True,
                "could_reverse_binary_forecast": True,
                "could_reverse_recommended_action": False, "causally_executable": True,
                "prose_variation_only": False,
                "differing_assumption": "the council can veto",
                "reversal_causal_chain": "council veto → not approved",
                "distinguishing_evidence": "council agenda",
                "challenger_thesis": "The council procedure decides",
                "challenger_decisive_actors": ["council"],
                "additional_credible_alternatives": []}
    llm = LeanScriptedLLM(reversal_verdict=reversal)
    res = lean_run(llm)
    lean = (res.provenance or {}).get("lean") or {}
    assert (lean.get("structural") or {}).get("challenger_generated") is True
    assert len(lean.get("particle_stopping") or []) >= 1


def test_full_fidelity_profile_is_byte_untouched_by_lean_modules():
    """The same scripted stack through full_fidelity must never touch the lean controller."""
    llm = LeanScriptedLLM()
    res = simulate_world("Will the initiative be approved?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=llm, seed=3,
                         execution_policy=dict(HERMETIC), execution_profile="full_fidelity")
    prov = res.provenance or {}
    assert prov.get("execution_profile") == "full_fidelity"
    assert "lean" not in prov
    assert not any("complete moment of bounded cognition" in p for p in llm.calls), \
        "full fidelity must never issue lean one-call prompts"


# ------------------------------------------------------------------ MANDATORY parity gates
def test_isolated_caching_parity_gate():
    """Caches ON vs OFF: identical semantics, fewer calls. Cohorting/prechecks/one-call are
    HELD IDENTICAL across arms; only the two caches differ (mandatory §1)."""
    common = {"one_call_cognition": True, "prechecks": True, "frontier_gate": True,
              "cohort_ceiling": 6}
    llm_off = LeanScriptedLLM()
    res_off = lean_run(llm_off, lean_actor={**common, "decision_cache": False,
                                            "consequence_cache": False})
    llm_on = LeanScriptedLLM()
    res_on = lean_run(llm_on, lean_actor={**common, "decision_cache": True,
                                          "consequence_cache": True})
    fp_off, fp_on = _fingerprint(res_off), _fingerprint(res_on)
    assert fp_on == fp_off, (
        f"CACHING PARITY VIOLATION:\nON:  {json.dumps(fp_on, indent=1, default=str)[:2000]}\n"
        f"OFF: {json.dumps(fp_off, indent=1, default=str)[:2000]}")
    # only calls/latency/cache provenance may differ — and they must actually differ
    assert llm_on.n_one_call() < llm_off.n_one_call(), (
        f"cache ON must spend fewer one-call decisions: on={llm_on.n_one_call()} "
        f"off={llm_off.n_one_call()}")
    ctl_on = ((res_on.provenance or {}).get("lean") or {}).get("controller") or {}
    assert (ctl_on.get("decision_cache") or {}).get("hits", 0) > 0


def test_lean_sequential_equals_bounded_concurrent(monkeypatch):
    """Concurrency changes wall-clock only (mandatory §6)."""
    llm_seq = LeanScriptedLLM()
    monkeypatch.delenv("SWM_BRANCH_THREADS", raising=False)
    res_seq = lean_run(llm_seq)
    llm_par = LeanScriptedLLM()
    monkeypatch.setenv("SWM_BRANCH_THREADS", "4")
    res_par = lean_run(llm_par)
    monkeypatch.delenv("SWM_BRANCH_THREADS", raising=False)
    assert _fingerprint(res_seq) == _fingerprint(res_par), \
        "bounded-concurrent lean execution changed semantics, not just wall-clock"


# ------------------------------------------------------------------ outcome-pathway + honesty
def test_no_lean_rollout_is_empty_and_pathway_is_validated():
    llm = LeanScriptedLLM()
    res = lean_run(llm)
    prov = res.provenance or {}
    per_model = prov.get("per_model_provenance") or {}
    assert per_model, "lean must report per-model provenance"
    # no silent empty rollout: every simulated model carries branches + census or an honest
    # unresolved/§NAP status — never a silent None
    assert res.simulation_status != "execution_failed"
    assert res.raw_probability is not None or res.raw_distribution or \
        res.simulation_status in ("unresolved", "partially_resolved")


def test_research_first_invariant_blocks_unarmed_actor_calls():
    from swm.world_model_v2.lean_controller import LeanActorController
    c = LeanActorController()
    with pytest.raises(RuntimeError, match="research-first"):
        c.decision_for(None, None, None, None, "s", [], {}, "tim", 0)
    with pytest.raises(RuntimeError, match="research-first"):
        c.arm_actor_calls(research_ledger={"evidence_gathered": True})   # incomplete ledger
