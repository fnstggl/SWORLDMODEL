"""Phase 7 — adversarial tests (Part 25): the system must PREFER SIMPLE and REJECT the unsupported.

These encode the anti-scaffolding discipline: a flexible form must not win just because it is flexible; a
leaky feature must be refused; the quarantined Hawkes must not silently return; distinct phenomena must be
distinguishable by held-out fit."""
import random

from swm.world_model_v2.nonlinear import compare, fit
from swm.world_model_v2.nonlinear import context as ctx
from swm.world_model_v2.nonlinear import applicability as ap
from swm.world_model_v2.nonlinear.forms import get_form


def _synth(fn, n, seed, noise=0.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.uniform(0, 10)
        p = fn(x)
        if noise:
            p = min(1, max(0, p + rng.gauss(0, noise)))
        rows.append({"features": {"x": x}, "x": x, "y": 1 if rng.random() < p else 0})
    return rows


def test_prefers_linear_when_truth_is_linear():
    """Data with a purely linear log-odds signal → the parsimony rule must keep the simple form, not a GAM."""
    import math
    truth = lambda x: 1 / (1 + math.exp(-(0.4 * x - 2)))
    rows = _synth(truth, 3000, seed=1)
    tr, va, te = fit.random_split(rows, seed=1)
    logistic = get_form("logistic"); gam = get_form("gam")
    fl = fit.fit_logistic_form(tr, ["x"]); fg = fit.fit_gam(tr, [], {"x": 5})
    cands = {"constant": lambda r: sum(rr["y"] for rr in tr) / len(tr),
             "logistic": lambda r: logistic.eval(fl.params, {"features": r["features"]}),
             "gam": lambda r: gam.eval(fg.params, {"features": r["features"]})}
    comp = compare.compare_forms(cands, va, te)
    decision = compare.select_with_parsimony(comp, simpler=("constant", "logistic"))
    # even if the GAM wins validation by luck, the parsimony rule must not promote it over logistic
    # unless it beats it with CI<0 on test
    assert decision["promoted"] in ("logistic", "constant") or decision["beat_baseline"] is True
    # and specifically: a GAM that only ties must be rejected in favor of logistic
    if comp["selected"] == "gam":
        key = "gam_vs_logistic"
        if key in comp["paired_test_deltas"]:
            ci = comp["paired_test_deltas"][key]["ci95"]
            if ci[1] >= 0:                       # not clearly better
                assert decision["promoted"] == "logistic"


def test_detects_genuine_threshold_over_linear():
    """A hard-threshold truth → a threshold/GAM form should out-predict linear on held-out (distinguishable)."""
    truth = lambda x: 0.85 if x > 6 else 0.1
    rows = _synth(truth, 3000, seed=2)
    tr, va, te = fit.random_split(rows, seed=2)
    logistic = get_form("logistic")
    fl = fit.fit_logistic_form(tr, ["x"])
    ft = fit.fit_smooth_threshold(tr, "x")
    st = get_form("threshold_smooth")
    yt = [r["y"] for r in te]
    b_lin = compare.brier([logistic.eval(fl.params, {"features": r["features"]}) for r in te], yt)
    b_thr = compare.brier([st.eval(ft.params, {"x": r["features"]["x"]}) for r in te], yt)
    assert b_thr < b_lin + 1e-4, "threshold form should capture the step better than linear"


def test_context_schema_blocks_future_context():
    v = ctx.ContextVariable(name="poll", definition="a poll taken later", source="observed",
                            temporal_validity="event_time")
    schema = ctx.ContextSchema("m", [v])
    audit = schema.leakage_audit(as_of=100.0, available={"poll": {"value": 1, "valid_from": 200.0}})
    assert not audit["leakage_free"]
    assert audit["blocked"][0]["reason"] == "future_context"


def test_context_schema_blocks_outcome_derived():
    v = ctx.ContextVariable(name="post_outcome", definition="derived from the label", source="assumed",
                            derived_from_outcome=True)
    schema = ctx.ContextSchema("m", [v])
    audit = schema.leakage_audit(as_of=100.0)
    assert not audit["leakage_free"]
    assert any(b["reason"] == "derived_from_outcome" for b in audit["blocked"])


def test_hawkes_stays_quarantined_in_registry():
    """The preserved Hawkes failure must remain quarantined and out of the compiler's vocabulary."""
    from swm.world_model_v2.registry.store import RegistryStore
    reg = RegistryStore.load()
    rec = reg.records.get("hawkes_self_excitation")
    assert rec is not None and rec.status == "quarantined"
    # and its failure is preserved
    assert rec.failed_validations(), "the Hawkes held-out failure must be preserved"


def test_self_exciting_form_marked_candidate_not_validated():
    f = get_form("self_exciting")
    assert f.maturity == "structural_candidate"
    assert any("Hawkes" in fc or "quarantin" in fc.lower() for fc in f.failure_conditions)


def test_backfire_form_is_not_default_and_needs_evidence():
    """inverted_u (backfire) must be a structural_candidate, never assumed."""
    assert get_form("inverted_u").maturity == "structural_candidate"


def test_transport_refuses_on_regime_mismatch():
    res = ap.transport_check(input_support_overlap=0.9, threshold_shift=False, regime_mismatch=True,
                             population_mismatch=False, platform_mismatch=False, outcome_def_mismatch=False)
    assert not res["transportable"]


def test_saturation_distinguished_from_fatigue():
    """Saturation is in the STIMULUS level; fatigue is in the repetition COUNT — different inputs, not aliases."""
    sat = get_form("hill"); fat = get_form("fatigue")
    assert "n_exposures" in fat.required_inputs and "x" in sat.required_inputs
    # saturation increases with stimulus; fatigue decreases with count
    assert sat.eval({"theta": 1, "n": 2, "k": 2}, {"x": 8}) > sat.eval({"theta": 1, "n": 2, "k": 2}, {"x": 1})
    assert fat.eval({"A": 1, "gamma": 0.5}, {"n_exposures": 5}) < fat.eval({"A": 1, "gamma": 0.5},
                                                                           {"n_exposures": 0})


def test_extension_promotion_blocked_without_validation():
    from swm.world_model_v2.nonlinear.registry_ext import NonlinearExtension
    ext = NonlinearExtension(extension_id="x1", family_id="attrition_dropout_hazard",
                             causal_process="attrition", selected_form="gam", candidate_forms=["logistic", "gam"])
    # cannot jump to locally_validated without a passed held-out record
    blockers = ext.promotion_blockers("locally_validated")
    assert any("PASSED" in b for b in blockers)
