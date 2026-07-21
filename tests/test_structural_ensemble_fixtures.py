"""Cross-domain structural-uncertainty fixtures (ensemble contract Section 25).

Ten domains whose DECISIVE causal structure differs (actor / institution / trust / network / resource /
legal / market / physical / hybrid / ambiguous). Entity and mechanism names are RANDOMIZED per run so
nothing can pass through memorized source literals. Test fixtures only — none of these names or domains
exist in production prompts or branches (enforced in test_no_fixture_leakage_into_production)."""
from __future__ import annotations

import json
import random

import pytest

from swm.world_model_v2.unified_runtime import simulate_world as _simulate_world_default
import functools
# These tests pin the FULL-FIDELITY pipeline (PR-#127 semantics). Since the §25 default
# switch, the bare entrypoint serves lean_adaptive, so the research-grade profile is
# selected EXPLICITLY here — same pin, same behavior, now by name.
simulate_world = functools.partial(_simulate_world_default, execution_profile="full_fidelity")
from tests.test_structural_ensemble import (EnsembleLLM, HERMETIC, OMISSION_QUIET, critic_ok,
                                            decomp_payload, recon_payload)

#: (domain, decisive-structure kind, alternative kind). The DECISIVE model is the ground truth the
#: candidate set must represent; the ALTERNATIVE must survive as a plausible competitor; a third
#: candidate is invalid (critic-rejected); the fourth duplicates the alternative (must merge).
DOMAINS = [
    ("supply_dispute", "actor", "institution"),
    ("licensing_gate", "institution", "actor"),
    ("partner_fallout", "relationship_trust", "actor"),
    ("feature_rollout", "distribution_network", "actor"),
    ("clinic_capacity", "resource_constraint", "institution"),
    ("zoning_appeal", "legal_procedural", "actor"),
    ("pricing_shock", "market_algorithmic", "actor"),
    ("harvest_window", "physical_external", "actor"),
    ("festival_launch", "hybrid_social_physical", "resource_constraint"),
    ("succession_call", "ambiguous", "institution"),
]


def _names(domain: str):
    rnd = random.Random(0xE17 ^ hash(domain) & 0xFFFF)

    def n(prefix):
        return f"{prefix}_{rnd.randrange(16 ** 6):06x}"
    return {"actor": n("actor"), "actor2": n("actor"), "inst": n("agency"), "mech": n("mech"),
            "constraint": n("limit"), "net": n("channel"), "phys": n("system")}


def _domain_llm(domain: str, decisive: str, alternative: str):
    nm = _names(domain)
    structures = {
        "actor": recon_payload(f"{nm['actor']}'s strategic choice decides {domain}",
                               [nm["actor"], nm["actor2"]], mechanisms=[nm["mech"]],
                               falsifiers=[f"{nm['actor']} recuses"]),
        "institution": recon_payload(f"the {nm['inst']} procedure decides {domain}",
                                     [nm["inst"]], institutions=[nm["inst"]],
                                     constraints=["quorum"], mechanisms=["formal_ruling"],
                                     falsifiers=[f"{nm['inst']} postpones"]),
        "relationship_trust": recon_payload(f"trust between {nm['actor']} and {nm['actor2']} decides",
                                            [nm["actor"], nm["actor2"]],
                                            constraints=["trust"], mechanisms=["reciprocity"],
                                            falsifiers=["a public falling-out"]),
        "distribution_network": recon_payload(f"the {nm['net']} distribution layer decides reach",
                                              [nm["net"]], constraints=["reach"],
                                              mechanisms=["diffusion"],
                                              falsifiers=[f"{nm['net']} outage"]),
        "resource_constraint": recon_payload(f"the {nm['constraint']} capacity binds",
                                             [nm["actor"]], constraints=[nm["constraint"]],
                                             mechanisms=["capacity_limit"],
                                             falsifiers=["capacity doubles"]),
        "legal_procedural": recon_payload(f"the statutory appeal path at {nm['inst']} controls timing",
                                          [nm["inst"]], institutions=[nm["inst"]],
                                          constraints=["statutory_deadline"],
                                          mechanisms=["appeal_process"],
                                          falsifiers=["the appeal is withdrawn"]),
        "market_algorithmic": recon_payload(f"the {nm['phys']} pricing algorithm reacts first",
                                            [nm["phys"]], constraints=["latency"],
                                            mechanisms=["feedback_repricing"],
                                            falsifiers=["the algorithm is frozen"]),
        "physical_external": recon_payload(f"the {nm['phys']} weather/physical window dominates",
                                           [nm["phys"]], constraints=["weather_window"],
                                           mechanisms=["physical_constraint"],
                                           falsifiers=["the window shifts by a month"]),
        "hybrid_social_physical": recon_payload(
            f"{nm['actor']}'s choices interact with the {nm['phys']} physical constraint",
            [nm["actor"], nm["phys"]], constraints=["weather_window", "staffing"],
            mechanisms=[nm["mech"], "physical_constraint"],
            falsifiers=["the venue closes"]),
        "ambiguous": recon_payload(f"no single structure dominates {domain}",
                                   [nm["actor"]], constraints=["unclear"],
                                   mechanisms=["mixed"], falsifiers=["clarifying disclosure"]),
    }
    invalid = recon_payload(f"the moon phase decides {domain}", ["moon"], mechanisms=["astrology"],
                            falsifiers=[])
    ent = {"actor": [nm["actor"], nm["actor2"]], "institution": [nm["inst"]],
           "relationship_trust": [nm["actor"], nm["actor2"]], "distribution_network": [nm["net"]],
           "resource_constraint": [nm["actor"]], "legal_procedural": [nm["inst"]],
           "market_algorithmic": [nm["phys"]], "physical_external": [nm["phys"]],
           "hybrid_social_physical": [nm["actor"], nm["phys"]], "ambiguous": [nm["actor"]]}
    inst_of = {"institution": [nm["inst"]], "legal_procedural": [nm["inst"]]}
    # each structure kind carries DISTINCT executable content (latents/relations), not just prose:
    # two kinds sharing actors (e.g. trust vs actor) must still compile materially different schemas
    lat_of = {"relationship_trust": [f"{nm['actor']}.trust_in_counterpart"],
              "distribution_network": [f"{nm['net']}.reach"],
              "resource_constraint": [f"{nm['actor']}.capacity"],
              "market_algorithmic": [f"{nm['phys']}.latency"],
              "physical_external": [f"{nm['phys']}.window"],
              "hybrid_social_physical": [f"{nm['actor']}.staffing", f"{nm['phys']}.window"]}
    rel_of = {"relationship_trust": [{"src": nm["actor"], "rel": "trusts", "dst": nm["actor2"]}],
              "distribution_network": [{"src": nm["net"], "rel": "reaches", "dst": nm["actor"]}]}

    def _decomp(kind, lean, hyp):
        return decomp_payload(ent[kind], lean=lean, hyp=hyp, institutions=inst_of.get(kind, ()),
                              relations=rel_of.get(kind, ()), latent_paths=lat_of.get(kind, ()))
    llm = EnsembleLLM(
        recon_by_role={
            "actor_relationship": structures[decisive],
            "institutional_procedural": structures[alternative],
            "resource_constraint": invalid,
            "information_distribution": structures[alternative],   # duplicate of the alternative
        },
        decomp_by_model={
            "m0_actor_relationship": _decomp(decisive, "weak_yes", "h_dec"),
            "m1_institutional_procedural": _decomp(alternative, "weak_no", "h_alt"),
            "m2_resource_constraint": decomp_payload(["moon"], lean="neutral", hyp="h_bad"),
            "m3_information_distribution": _decomp(alternative, "weak_no", "h_alt"),
        },
        critic_by_thesis={"the moon phase decides": critic_ok(
            reject=True, reject_reason="no causal mechanism connects the moon phase to this outcome")})
    return llm, nm, structures


@pytest.mark.parametrize("domain,decisive,alternative", DOMAINS,
                         ids=[d[0] for d in DOMAINS])
def test_domain_fixture_generates_represents_rejects_merges_pilots_promotes(domain, decisive,
                                                                            alternative):
    llm, nm, structures = _domain_llm(domain, decisive, alternative)
    res = simulate_world(f"Will the {domain.replace('_', ' ')} resolve favorably?",
                         as_of="2025-06-01", horizon="2025-09-01", llm=llm, seed=7,
                         execution_policy=dict(HERMETIC))
    # §NAP: fixtures whose models materially disagree serve per-model conditionals
    # (partially_resolved) instead of an averaged headline
    assert res.simulation_status in ("completed", "completed_with_degradation",
                                     "partially_resolved")
    se = res.structural_ensemble
    # several distinct models generated; the known decisive structure and one alternative represented
    promoted = [m for m in se["models"] if m["promotion_status"] == "promoted"]
    assert len(promoted) >= 2
    theses = " | ".join(m["causal_thesis"] for m in promoted)
    assert structures[decisive]["causal_thesis"] in theses
    assert structures[alternative]["causal_thesis"] in theses
    # the invalid alternative is rejected with the critic's reason, before any full simulation
    rejected = [m for m in se["rejected_and_merged"] if m["status"] == "rejected"]
    assert any("no causal mechanism" in m["reason"] for m in rejected)
    # duplicates merged conservatively
    assert se["n_merged"] >= 1
    # pilots ran for every surviving model; promoted models got full budgets
    for m in promoted:
        sim = se["simulation_manifest"][m["model_id"]]
        assert sim["pilot_particles"] >= 8
        assert sim["final_particles"] >= sim["full_budget_required"]
    # structural sensitivity exposed from actual per-model results
    assert se["structural_sensitivity"]["classification"] in (
        "structurally_stable", "mildly_structurally_sensitive",
        "materially_structurally_sensitive", "structurally_underidentified")
    assert se["model_distributions"] and len(se["model_distributions"]) == len(promoted)


def test_actions_evaluated_differently_when_causal_structure_differs():
    """Phase 13 across models: the decision-maker exists in the decisive-actor model but not in the
    institutional model, so the SAME action is feasible under one structure and not the other — the
    cross-model result must expose that instead of averaging it away."""
    from swm.world_model_v2.phase13.api import recommend_action
    from swm.world_model_v2.phase13.contracts import DecisionProblem, Stakeholder, UtilitySpec
    llm, nm, _ = _domain_llm("supply_dispute", "actor", "institution")
    res = simulate_world("Will the supply dispute resolve favorably?", as_of="2025-06-01",
                         horizon="2025-09-01", llm=llm, seed=7, execution_policy=dict(HERMETIC))
    problem = DecisionProblem(
        decision_id="dx", decision_maker=nm["actor"],
        authority=["communicate", "gather_information"], as_of="2025-06-01",
        utility=UtilitySpec(stakeholders=[Stakeholder(
            nm["actor"],
            utility_fn=lambda o: float(o.get("quantities", {}).get("outcome", 0.0) or 0.0))]))
    r = recommend_action(problem, res, budget="diagnostic", seed=2, n_particles=6)
    se = r.provenance["structural_ensemble"]
    assert se["n_models_evaluated"] >= 2
    winners = se["winner_by_model"]
    assert len(winners) >= 2
    per_model_rankings = se["ranking_by_model"]
    assert any(per_model_rankings[m] for m in per_model_rankings), "per-model rankings preserved"
    assert json.dumps(se["expected_utility_matrix"])   # per-action per-model utilities intact
    if len({w for w in winners.values()}) > 1:
        assert se["recommendation_stability"] != "structurally_stable"
        assert se["conditional_strategy"]


def test_no_fixture_leakage_into_production():
    """Fixture names/domains must not appear in production modules (anti-hardcoding, Section 25)."""
    from pathlib import Path
    prod = ""
    for p in Path("swm/world_model_v2").glob("*.py"):
        prod += p.read_text()
    for domain, _, _ in DOMAINS:
        assert domain not in prod, f"fixture domain {domain!r} leaked into production code"
    for domain, decisive, alternative in DOMAINS:
        nm = _names(domain)
        for v in nm.values():
            assert v not in prod, f"fixture name {v!r} leaked into production code"
