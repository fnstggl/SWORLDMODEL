"""Tests for deep per-person inference — multi-pass synthesis, depth-scaled confidence, as-of/no-leakage."""
from swm.variables.deep_inference import (DeepInferenceEngine, DeepPersonaStore, depth_factor,
                                          recency_weight)
from swm.variables.schema import PERSONA, BY_CATEGORY


def test_recency_weight_is_off_by_default_and_decays_when_enabled():
    assert recency_weight(0, 100, None) == 1.0                   # no half_life -> disabled
    assert recency_weight(None, 100, 10) == 1.0                  # no timestamp -> disabled
    assert recency_weight(100, 100, 10) == 1.0                   # age 0 -> full weight
    assert recency_weight(90, 100, 10) == 0.5                    # one half-life old -> half weight
    assert recency_weight(80, 100, 10) == 0.25                   # two half-lives -> quarter


def test_synthesize_is_backward_compatible_without_timestamps():
    """No timestamps/half_life -> identical to the salience-only behavior (strict extension)."""
    eng = DeepInferenceEngine()
    sig = [{"verbosity": {"value": 0.2, "salience": 0.8}},
           {"verbosity": {"value": 0.8, "salience": 0.8}}]
    p = eng.synthesize(sig)
    assert abs(p["verbosity"]["value"] - 0.5) < 1e-9             # equal salience, equal weight -> mean 0.5


def test_recency_shifts_value_toward_recent_evidence():
    """A person who drifted from low to high verbosity: with recency decay, the recent value dominates."""
    eng = DeepInferenceEngine(half_life=5.0)
    sig = [{"verbosity": {"value": 0.1, "salience": 0.8}},       # old
           {"verbosity": {"value": 0.9, "salience": 0.8}}]       # recent
    flat = eng.synthesize(sig)                                   # no timing -> mean 0.5
    decayed = eng.synthesize(sig, timestamps=[0, 40], now=40)    # old doc 8 half-lives back
    assert abs(flat["verbosity"]["value"] - 0.5) < 1e-9
    assert decayed["verbosity"]["value"] > 0.85                  # recent (0.9) dominates


def test_depth_factor_monotone_and_saturating():
    ds = [depth_factor(n) for n in (0, 1, 2, 4, 8, 16, 64)]
    assert ds[0] == 0.0
    assert all(ds[i] < ds[i + 1] for i in range(len(ds) - 1))    # strictly increasing in depth
    assert ds[-1] < 1.0 and ds[-1] > 0.9                          # saturates toward 1


def test_confidence_grows_with_corpus_depth():
    """The same consistent trait evidence should yield higher confidence from more documents."""
    eng = DeepInferenceEngine()
    sig = {"epistemic_rigor": {"value": 0.8, "salience": 0.7}}
    shallow = eng.synthesize([sig])
    deep = eng.synthesize([sig] * 10)
    assert deep["epistemic_rigor"]["confidence"] > shallow["epistemic_rigor"]["confidence"]
    assert abs(deep["epistemic_rigor"]["value"] - 0.8) < 1e-6     # value stable, confidence grows


def test_inconsistent_evidence_lowers_confidence():
    """Conflicting per-document signals (reflection pass) should reduce confidence vs consistent ones."""
    eng = DeepInferenceEngine()
    consistent = eng.synthesize([{"combativeness": {"value": 0.7, "salience": 0.6}}] * 6)
    conflicting = eng.synthesize([{"combativeness": {"value": v, "salience": 0.6}}
                                  for v in (0.1, 0.9, 0.2, 0.8, 0.15, 0.85)])
    assert conflicting["combativeness"]["confidence"] < consistent["combativeness"]["confidence"]


def test_persona_asof_excludes_future_documents():
    """No leakage: the as-of persona uses only documents strictly before `now`."""
    store = DeepPersonaStore()
    store.add_doc("u", 10, {"verbosity": {"value": 0.2, "salience": 0.8}})
    store.add_doc("u", 20, {"verbosity": {"value": 0.9, "salience": 0.8}})
    store.add_doc("u", 30, {"verbosity": {"value": 0.9, "salience": 0.8}})
    assert store.depth_asof("u", 25) == 2            # docs at ts 10 and 20 only
    assert store.depth_asof("u", 15) == 1
    p_early = store.persona_asof("u", 15)["verbosity"]["value"]
    assert abs(p_early - 0.2) < 1e-6                  # only the first doc is visible at ts=15


def test_persona_maps_into_schema_persona_category():
    eng = DeepInferenceEngine()
    persona = eng.infer_persona(["I think, and I could be wrong, that the evidence here is weak. "
                                 "For example, the study shows a small effect."])
    assert persona                                   # non-empty
    assert set(persona).issubset(set(BY_CATEGORY[PERSONA]))
    for v in persona.values():
        assert "value" in v and "confidence" in v and 0 <= v["confidence"] <= 1


def test_persona_to_vars_confidence_blend():
    from swm.variables.deep_inference import persona_to_vars
    persona = {"trait_openness": {"value": 1.0, "confidence": 0.8},   # deep, confident -> moves toward 1.0
               "combativeness": {"value": 1.0, "confidence": 0.1}}    # thin -> stays near the prior
    prior = {"trait_openness": 0.5, "combativeness": 0.5}
    v = persona_to_vars(persona, prior=prior)
    assert v["trait_openness"] > 0.85                                 # high-confidence trait moves
    assert abs(v["combativeness"] - 0.55) < 0.02                      # low-confidence trait ~ prior


def test_vars_asof_is_leakage_free_and_feeds_response_model():
    from swm.simulation.response_model import quantities
    store = DeepPersonaStore(engine=DeepInferenceEngine())
    # an open, humble persona vs a combative, certain one -> different receptivity in the Level-1 model
    for t in (1, 2, 3):
        store.add_doc("humble", t, {"trait_openness": {"value": 0.9, "salience": 0.8},
                                    "intellectual_humility": {"value": 0.9, "salience": 0.8}})
        store.add_doc("rigid", t, {"combativeness": {"value": 0.9, "salience": 0.8},
                                   "certainty_disposition": {"value": 0.9, "salience": 0.8}})
    prior = {"trait_openness": 0.5, "intellectual_humility": 0.5, "combativeness": 0.5,
             "certainty_disposition": 0.5}
    hv = store.vars_asof("humble", now=99, prior=prior)
    rv = store.vars_asof("rigid", now=99, prior=prior)
    msg = {"clarity": 0.7, "ask_directness": 0.6}
    assert quantities(hv, {}, msg)["receptivity"] > quantities(rv, {}, msg)["receptivity"]
    # leakage: as-of before any doc -> empty persona
    assert store.vars_asof("humble", now=1, prior=prior) == {} or \
        all(abs(x - prior[k]) < 1e-9 for k, x in store.vars_asof("humble", now=1, prior=prior).items())
