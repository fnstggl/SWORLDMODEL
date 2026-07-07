"""Tests for deep per-person inference — multi-pass synthesis, depth-scaled confidence, as-of/no-leakage."""
from swm.variables.deep_inference import DeepInferenceEngine, DeepPersonaStore, depth_factor
from swm.variables.schema import PERSONA, BY_CATEGORY


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
