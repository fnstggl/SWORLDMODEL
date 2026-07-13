"""Typed nonlinear structural-form registry — Phase 7, Part 1.

A STRUCTURAL FORM is a named mathematical shape a mechanism's transition can take: linear, logistic,
Hill saturation, hard/soft threshold, fatigue, hysteresis, a change-point, a finite mixture, a regime
model, a survival hazard, a self-exciting intensity, … Phase 6 owns the CAUSAL FAMILY (what process this
is — complex contagion, trust repair, content response). Phase 7 owns the FORM the family's response
takes and whether the data support that form over a simpler one.

Design contract (why this is not "a bag of equations in Markdown", Part 1 + Part 27):
  * every form is EXECUTABLE here — `eval(params, inputs)` is real pure-Python math, deterministic and
    replayable, so a form can run inside WorldState with no third-party runtime dependency;
  * every form declares its full metadata (monotonicity, param schema, extrapolation + missing-data
    behavior, invariants, differentiability, cost, failure conditions) so applicability/transport/
    composition can reason about it BEFORE it executes;
  * a form being PRESENT here does not make it VALIDATED. Presence == "a real, evaluable candidate shape".
    Promotion to a mechanism requires held-out evidence (see registry_ext + the validation experiments).
    Forms carry a `maturity` label that is honest about this.

The runtime never fits here — it only EVALUATES serialized parameters (knots, breakpoints, coefficients)
produced by the offline fitting layer (`nonlinear/fit.py`). Evaluating a fitted spline/isotonic/Hill given
its parameters is trivial pure Python; the heavy estimation (numpy/scipy/sklearn, optional) stays offline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- output domains / maturity vocab
OUTPUT_DOMAINS = ("real", "unit_interval", "nonneg_rate", "prob_window", "count", "regime_index")
MONOTONICITY = ("increasing", "decreasing", "non_monotone", "unconstrained", "context")
MATURITY = ("primitive", "structural_candidate", "fitted_available", "validated_elsewhere")
#   primitive             a mathematically standard shape, evaluable, no scenario claim
#   structural_candidate  a plausible shape for a mechanism, not yet fit or validated on data
#   fitted_available      a fit routine exists (nonlinear/fit.py) and has produced real coefficients
#   validated_elsewhere   the shape earned a held-out win in a *specific* mechanism (see registry_ext)

# phenomena tags — the Part 4 vocabulary a form can express (used by applicability + audit)
PHENOMENA = ("threshold", "saturation", "diminishing_returns", "fatigue", "habituation", "refractory",
             "reinforcement", "backfire", "inverted_u", "u_shaped", "hysteresis", "path_dependence",
             "tipping", "regime", "self_excitation", "self_inhibition", "finite_population",
             "interaction", "heterogeneity", "recency", "none")


def _clip(z, lo, hi):
    return lo if z < lo else (hi if z > hi else z)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-_clip(z, -30.0, 30.0)))


class FormError(ValueError):
    pass


@dataclass
class StructuralForm:
    """One evaluable structural shape with its full typed metadata (Part 1 required fields)."""
    form_id: str
    version: str
    title: str
    output_domain: str
    monotonicity: str
    phenomena: tuple
    param_schema: dict                      # {name: {"desc", "lo"?, "hi"?, "required"?}}
    required_inputs: tuple                  # input keys eval() reads (context/history/exposure)
    time_scales: tuple = ("event", "interval", "window", "horizon")
    fitting_method: str = ""
    regularization: str = ""
    invariants: tuple = ()
    extrapolation_behavior: str = "clamp_to_observed_support"
    missing_data_behavior: str = "explicit_default_or_omit"
    computational_cost: str = "O(1)"
    differentiable: bool = True
    deterministic_replay: bool = True
    validation_requirements: tuple = ()
    failure_conditions: tuple = ()
    theory: str = ""                        # the scientific justification for the shape
    maturity: str = "primitive"
    code_ref: str = "swm.world_model_v2.nonlinear.forms"
    test_ref: str = "tests/test_wmv2_phase7_forms.py"
    _eval: object = field(default=None, repr=False)   # callable(params: dict, inputs: dict) -> float

    def __post_init__(self):
        if self.output_domain not in OUTPUT_DOMAINS:
            raise FormError(f"{self.form_id}: bad output_domain {self.output_domain!r}")
        if self.monotonicity not in MONOTONICITY:
            raise FormError(f"{self.form_id}: bad monotonicity {self.monotonicity!r}")
        if self.maturity not in MATURITY:
            raise FormError(f"{self.form_id}: bad maturity {self.maturity!r}")
        for ph in self.phenomena:
            if ph not in PHENOMENA:
                raise FormError(f"{self.form_id}: unknown phenomenon {ph!r}")

    def eval(self, params: dict, inputs: dict) -> float:
        """Evaluate the fitted form. Result is clamped to the declared output domain and guarded against
        NaN/inf (a form NEVER emits a non-finite value — see nonlinear.safety for the failure record)."""
        if self._eval is None:
            raise FormError(f"{self.form_id}: no evaluator bound")
        try:
            y = self._eval(params, inputs)
        except (OverflowError, ZeroDivisionError, ValueError) as e:
            # arithmetic blow-up (x**n overflow, log of ≤0, …) is a non-finite result — surface it as a
            # FormError so the operator's safety layer records it rather than crashing the rollout
            raise FormError(f"{self.form_id}: arithmetic failure ({type(e).__name__}: {e})")
        if y != y or y in (float("inf"), float("-inf")):    # NaN / inf guard
            raise FormError(f"{self.form_id}: non-finite output {y!r} for inputs {list(inputs)[:6]}")
        return self._domain_clip(y)

    def _domain_clip(self, y: float) -> float:
        if self.output_domain in ("unit_interval", "prob_window"):
            return _clip(y, 0.0, 1.0)
        if self.output_domain in ("nonneg_rate", "count"):
            return max(0.0, y)
        return y

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "_eval"}
        d["param_schema"] = self.param_schema
        return d


_FORMS: dict = {}


def register_form(form: StructuralForm, *, replace: bool = False) -> StructuralForm:
    if form.form_id in _FORMS and not replace:
        raise FormError(f"form {form.form_id!r} already registered (replace=True to override)")
    _FORMS[form.form_id] = form
    return form


def get_form(form_id: str) -> StructuralForm:
    if form_id not in _FORMS:
        raise KeyError(f"no structural form {form_id!r} (known: {sorted(_FORMS)})")
    return _FORMS[form_id]


def list_forms() -> list:
    return sorted(_FORMS)


def forms_for_phenomenon(ph: str) -> list:
    return sorted(fid for fid, f in _FORMS.items() if ph in f.phenomena)


def registry_snapshot() -> dict:
    """Machine-readable dump of the whole form registry (Part 31 artifact source)."""
    return {fid: _FORMS[fid].as_dict() for fid in sorted(_FORMS)}


# ================================================================ input helpers
def _x(inputs, key="x", default=0.0):
    v = inputs.get(key, default)
    return float(v) if v is not None else float(default)


def _linpred(params, inputs):
    """Additive linear predictor b + Σ w_j·feat_j over a fitted {weights, intercept, standardizer}."""
    s = float(params.get("intercept", 0.0))
    stdz = params.get("standardizer") or {}
    feats = inputs.get("features") or inputs
    for k, w in (params.get("weights") or {}).items():
        v = float(feats.get(k, 0.0) or 0.0)
        if k in stdz:
            mu, sd = stdz[k]
            v = (v - mu) / (sd or 1.0)
        s += float(w) * v
    s += _interaction_term(params, feats)
    return s


def _interaction_term(params, feats):
    """Explicit pairwise interactions (Part 4H). The raw product is standardized by a DEDICATED interaction
    standardizer (params['interactions_std'][key] = [mu, sd]) so fit and runtime agree exactly and the term is
    numerically safe regardless of the component feature scales (a raw tenure×contract product is huge)."""
    out = 0.0
    istd = params.get("interactions_std") or {}
    for pair, c in (params.get("interactions") or {}).items():
        key = pair if isinstance(pair, str) else f"{pair[0]}*{pair[1]}"
        a, b = key.split("*")
        prod = float(feats.get(a, 0.0) or 0.0) * float(feats.get(b, 0.0) or 0.0)
        if key in istd:
            mu, sd = istd[key]
            prod = (prod - mu) / (sd or 1.0)
        out += float(c) * prod
    return out


# ================================================================ the forms (real, evaluable)
def _register_core_forms():
    R = register_form

    # ---- 1. linear -------------------------------------------------------------------------------
    R(StructuralForm(
        form_id="linear", version="1.0.0", title="Linear predictor b + Σ w·x",
        output_domain="real", monotonicity="unconstrained", phenomena=("interaction",),
        param_schema={"weights": {"desc": "per-feature coefficients"}, "intercept": {"desc": "bias"},
                      "standardizer": {"desc": "{feat:[mu,sd]} train stats", "required": False},
                      "interactions": {"desc": "{a*b: coef} explicit pairwise", "required": False}},
        required_inputs=("features",), fitting_method="OLS / penalized MLE", regularization="L2",
        invariants=("additive in features",), extrapolation_behavior="linear_extend",
        theory="baseline additive response — the simpler form every nonlinear candidate must beat",
        maturity="primitive", _eval=lambda p, i: _linpred(p, i)))

    # ---- 2. logistic -----------------------------------------------------------------------------
    R(StructuralForm(
        form_id="logistic", version="1.0.0", title="Logistic σ(b + Σ w·x)",
        output_domain="unit_interval", monotonicity="context", phenomena=("saturation", "interaction"),
        param_schema={"weights": {"desc": "log-odds coefficients"}, "intercept": {"desc": "bias"},
                      "standardizer": {"desc": "{feat:[mu,sd]}", "required": False},
                      "interactions": {"desc": "{a*b: coef}", "required": False}},
        required_inputs=("features",), fitting_method="logistic MLE", regularization="L2",
        invariants=("output in [0,1]", "monotone in each linear term"),
        extrapolation_behavior="saturates_to_0_or_1",
        theory="canonical discrete-choice / adoption link; the nonlinearity is only the squashing link",
        maturity="fitted_available", _eval=lambda p, i: sigmoid(_linpred(p, i))))

    # ---- 3. complementary log-log window hazard --------------------------------------------------
    R(StructuralForm(
        form_id="cloglog_hazard", version="1.0.0",
        title="Window hazard P = 1 − exp(−exp(θ·x)·W)", output_domain="prob_window",
        monotonicity="context", phenomena=("saturation",),
        param_schema={"weights": {"desc": "log-rate coefficients"}, "intercept": {"desc": "log baseline"},
                      "standardizer": {"desc": "{feat:[mu,sd]}", "required": False}},
        required_inputs=("features", "window_days"), fitting_method="cloglog GLM (Newton-damped)",
        regularization="L2", invariants=("output in [0,1]", "P→1 as exposure→∞"),
        extrapolation_behavior="saturates_to_1", time_scales=("window", "horizon"),
        theory="survival mechanism: the SAME λ integrates step-by-step in rollout (Higgs exposure hazard)",
        maturity="validated_elsewhere",
        _eval=lambda p, i: 1.0 - math.exp(-min(50.0, math.exp(_clip(_linpred(p, i), -30, 30))
                                                * _x(i, "window_days", 1.0)))))

    # ---- 4. piecewise linear / segmented ---------------------------------------------------------
    def _piecewise(p, i):
        x = _x(i, "x")
        knots = p["knots"]            # ascending x breakpoints
        vals = p["values"]            # y at each knot (len == len(knots))
        if x <= knots[0]:
            if p.get("extrapolate") == "flat" or len(knots) < 2:
                return vals[0]
            sl = (vals[1] - vals[0]) / (knots[1] - knots[0])
            return vals[0] + sl * (x - knots[0])
        if x >= knots[-1]:
            if p.get("extrapolate") == "flat" or len(knots) < 2:
                return vals[-1]
            sl = (vals[-1] - vals[-2]) / (knots[-1] - knots[-2])
            return vals[-1] + sl * (x - knots[-1])
        for j in range(len(knots) - 1):
            if knots[j] <= x <= knots[j + 1]:
                t = (x - knots[j]) / (knots[j + 1] - knots[j])
                return vals[j] + t * (vals[j + 1] - vals[j])
        return vals[-1]
    R(StructuralForm(
        form_id="piecewise_linear", version="1.0.0", title="Piecewise-linear over fitted knots",
        output_domain="real", monotonicity="unconstrained", phenomena=("threshold", "saturation"),
        param_schema={"knots": {"desc": "ascending x breakpoints"}, "values": {"desc": "y at knots"},
                      "extrapolate": {"desc": "'flat' or 'linear'", "required": False}},
        required_inputs=("x",), fitting_method="least squares on fixed/selected knots",
        regularization="knot count (validation-selected)", invariants=("continuous", "interpolates knots"),
        extrapolation_behavior="flat_or_linear_beyond_knots",
        theory="flexible shape with interpretable breakpoints; guards against runaway extrapolation",
        maturity="fitted_available", _eval=_piecewise))

    # ---- 5. monotonic spline (isotonic step / PCHIP-serialized) ----------------------------------
    def _monotone(p, i):
        # serialized as ascending (x_k, y_k) with y monotone; linear interp, flat extrapolation
        x = _x(i, "x")
        xs, ys = p["x"], p["y"]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for j in range(len(xs) - 1):
            if xs[j] <= x <= xs[j + 1]:
                t = (x - xs[j]) / ((xs[j + 1] - xs[j]) or 1.0)
                return ys[j] + t * (ys[j + 1] - ys[j])
        return ys[-1]
    R(StructuralForm(
        form_id="monotonic_spline", version="1.0.0", title="Monotone interpolant (isotonic/PCHIP)",
        output_domain="real", monotonicity="increasing", phenomena=("saturation", "diminishing_returns"),
        param_schema={"x": {"desc": "ascending knots"}, "y": {"desc": "monotone values"},
                      "direction": {"desc": "increasing|decreasing", "required": False}},
        required_inputs=("x",), fitting_method="isotonic regression (sklearn) or PCHIP",
        regularization="monotonicity constraint", invariants=("monotone by construction",),
        extrapolation_behavior="clamp_to_endpoints",
        theory="enforces a known monotone dose-response WITHOUT assuming its parametric shape",
        maturity="fitted_available", _eval=_monotone))

    # ---- 6. unconstrained natural cubic spline ---------------------------------------------------
    def _cubic(p, i):
        # evaluate a natural cubic spline from serialized (x, a, b, c, d) segments: y=a+b·dx+c·dx²+d·dx³
        x = _x(i, "x")
        xs = p["x"]
        segs = p["segments"]          # [{a,b,c,d}] per interval [xs[j], xs[j+1]]
        if x <= xs[0]:
            s = segs[0]
            return s["a"] + s["b"] * (x - xs[0])           # linear tail (natural bc)
        if x >= xs[-1]:
            s = segs[-1]
            dx = xs[-1] - xs[-2]
            yv = s["a"] + s["b"] * dx + s["c"] * dx ** 2 + s["d"] * dx ** 3
            sl = s["b"] + 2 * s["c"] * dx + 3 * s["d"] * dx ** 2
            return yv + sl * (x - xs[-1])
        for j in range(len(xs) - 1):
            if xs[j] <= x <= xs[j + 1]:
                s = segs[j]
                dx = x - xs[j]
                return s["a"] + s["b"] * dx + s["c"] * dx ** 2 + s["d"] * dx ** 3
        return segs[-1]["a"]
    R(StructuralForm(
        form_id="cubic_spline", version="1.0.0", title="Natural cubic spline (GAM smooth term)",
        output_domain="real", monotonicity="non_monotone", phenomena=("inverted_u", "u_shaped"),
        param_schema={"x": {"desc": "knots"}, "segments": {"desc": "[{a,b,c,d}] cubic coeffs"}},
        required_inputs=("x",), fitting_method="penalized regression spline (offline)",
        regularization="smoothing penalty (validation-selected df)",
        invariants=("smooth C2 interior",), extrapolation_behavior="linear_tails_natural_bc",
        failure_conditions=("wild extrapolation beyond knots — transport must gate",),
        theory="flexible smooth term for GAMs; can capture inverted-U / non-monotone shapes",
        maturity="fitted_available", differentiable=True, _eval=_cubic))

    # ---- 7. hard threshold -----------------------------------------------------------------------
    R(StructuralForm(
        form_id="threshold_hard", version="1.0.0", title="Hard threshold (Granovetter step)",
        output_domain="unit_interval", monotonicity="increasing", phenomena=("threshold", "tipping"),
        param_schema={"tau": {"desc": "threshold on x"}, "low": {"desc": "y below"},
                      "high": {"desc": "y at/above"}},
        required_inputs=("x",), fitting_method="change-point search (grid on validation)",
        invariants=("piecewise constant",), extrapolation_behavior="flat",
        theory="Granovetter collective-behavior threshold; adoption once active-neighbor fraction ≥ τ",
        maturity="fitted_available", differentiable=False,
        _eval=lambda p, i: float(p.get("high", 1.0)) if _x(i, "x") >= float(p["tau"])
        else float(p.get("low", 0.0))))

    # ---- 8. smooth threshold ---------------------------------------------------------------------
    R(StructuralForm(
        form_id="threshold_smooth", version="1.0.0", title="Smooth threshold σ((x−τ)/s)",
        output_domain="unit_interval", monotonicity="increasing", phenomena=("threshold", "saturation"),
        param_schema={"tau": {"desc": "location"}, "s": {"desc": "softness (>0)"},
                      "low": {"desc": "floor", "required": False}, "high": {"desc": "ceil",
                                                                            "required": False}},
        required_inputs=("x",), fitting_method="nonlinear MLE",
        invariants=("monotone increasing", "→low as x→−∞, →high as x→∞"),
        theory="heterogeneous-threshold population → a smooth soft step (vs a single hard threshold)",
        maturity="fitted_available",
        _eval=lambda p, i: float(p.get("low", 0.0)) + (float(p.get("high", 1.0)) - float(p.get("low", 0.0)))
        * sigmoid((_x(i, "x") - float(p["tau"])) / (float(p.get("s", 1.0)) or 1.0))))

    # ---- 9. Hill saturation ----------------------------------------------------------------------
    R(StructuralForm(
        form_id="hill", version="1.0.0", title="Hill θ·xⁿ/(kⁿ+xⁿ)",
        output_domain="nonneg_rate", monotonicity="increasing",
        phenomena=("saturation", "threshold", "diminishing_returns", "tipping"),
        param_schema={"theta": {"desc": "max effect / scale"}, "n": {"desc": "Hill coefficient (n>1 → S-curve)"},
                      "k": {"desc": "half-saturation"}},
        required_inputs=("x",), fitting_method="grid on (n,k) + profiled scale",
        invariants=("0 at x=0", "→θ as x→∞", "sigmoidal for n>1"),
        extrapolation_behavior="saturates_to_theta",
        theory="dose-response / complex contagion: n>1 gives superlinear onset (social reinforcement) then "
               "saturation — the Higgs complex_contagion_hazard shape",
        maturity="validated_elsewhere",
        _eval=lambda p, i: (float(p["theta"]) * _x(i, "x") ** float(p["n"])
                            / (float(p["k"]) ** float(p["n"]) + _x(i, "x") ** float(p["n"]))
                            if _x(i, "x") > 0 else 0.0)))

    # ---- 10. Michaelis–Menten --------------------------------------------------------------------
    R(StructuralForm(
        form_id="michaelis_menten", version="1.0.0", title="Michaelis–Menten Vmax·x/(K+x)",
        output_domain="nonneg_rate", monotonicity="increasing",
        phenomena=("saturation", "diminishing_returns"),
        param_schema={"Vmax": {"desc": "ceiling"}, "K": {"desc": "half-saturation constant"}},
        required_inputs=("x",), fitting_method="Lineweaver–Burk / nonlinear LS",
        invariants=("0 at 0", "concave", "→Vmax"), extrapolation_behavior="saturates_to_Vmax",
        theory="hyperbolic saturation (Hill with n=1): pure diminishing returns, no threshold",
        maturity="fitted_available",
        _eval=lambda p, i: float(p["Vmax"]) * _x(i, "x") / (float(p["K"]) + _x(i, "x"))
        if (float(p["K"]) + _x(i, "x")) > 0 else 0.0))

    # ---- 11. logistic saturation (bounded growth) ------------------------------------------------
    R(StructuralForm(
        form_id="logistic_saturation", version="1.0.0", title="Logistic ceiling L/(1+e^{−k(x−x0)})",
        output_domain="nonneg_rate", monotonicity="increasing", phenomena=("saturation", "tipping"),
        param_schema={"L": {"desc": "carrying capacity"}, "k": {"desc": "growth rate"},
                      "x0": {"desc": "midpoint"}},
        required_inputs=("x",), fitting_method="nonlinear LS",
        invariants=("→L", "S-shaped"), theory="bounded growth toward a ceiling (adoption capacity)",
        maturity="fitted_available",
        _eval=lambda p, i: float(p["L"]) * sigmoid(float(p["k"]) * (_x(i, "x") - float(p["x0"])))))

    # ---- 12. exponential saturation --------------------------------------------------------------
    R(StructuralForm(
        form_id="exp_saturation", version="1.0.0", title="Exponential approach L(1−e^{−k·x})",
        output_domain="nonneg_rate", monotonicity="increasing",
        phenomena=("saturation", "diminishing_returns"),
        param_schema={"L": {"desc": "asymptote"}, "k": {"desc": "rate"}},
        required_inputs=("x",), fitting_method="nonlinear LS",
        invariants=("0 at 0", "concave", "→L"), theory="cumulative-exposure ceiling (attention saturation)",
        maturity="fitted_available",
        _eval=lambda p, i: float(p["L"]) * (1.0 - math.exp(-max(0.0, float(p["k"]) * _x(i, "x"))))))

    # ---- 13. inverted-U --------------------------------------------------------------------------
    R(StructuralForm(
        form_id="inverted_u", version="1.0.0", title="Inverted-U (Gaussian bump)",
        output_domain="real", monotonicity="non_monotone", phenomena=("inverted_u", "backfire"),
        param_schema={"peak": {"desc": "x of maximum"}, "spread": {"desc": "width (>0)"},
                      "height": {"desc": "peak value"}, "floor": {"desc": "baseline", "required": False}},
        required_inputs=("x",), fitting_method="nonlinear LS",
        invariants=("single interior maximum",), extrapolation_behavior="decays_to_floor",
        theory="weak-tie transmission / overexposure backfire: response rises then falls. NEVER assumed "
               "without evidence — must beat a monotone form on held-out",
        maturity="structural_candidate",
        _eval=lambda p, i: float(p.get("floor", 0.0)) + float(p["height"])
        * math.exp(-((_x(i, "x") - float(p["peak"])) ** 2) / (2.0 * float(p["spread"]) ** 2))))

    # ---- 14. U-shaped ----------------------------------------------------------------------------
    R(StructuralForm(
        form_id="u_shaped", version="1.0.0", title="U-shaped quadratic (a·(x−m)²+c, a>0)",
        output_domain="real", monotonicity="non_monotone", phenomena=("u_shaped",),
        param_schema={"a": {"desc": "curvature (>0)"}, "m": {"desc": "minimum location"},
                      "c": {"desc": "minimum value"}},
        required_inputs=("x",), fitting_method="quadratic LS", invariants=("single interior minimum",),
        theory="cost/benefit trade-offs with an interior optimum-avoidance", maturity="structural_candidate",
        _eval=lambda p, i: float(p["a"]) * (_x(i, "x") - float(p["m"])) ** 2 + float(p["c"])))

    # ---- 15. fatigue (declining response with cumulative exposure) -------------------------------
    R(StructuralForm(
        form_id="fatigue", version="1.0.0", title="Fatigue A·γ^n (response decays with exposure count)",
        output_domain="real", monotonicity="decreasing", phenomena=("fatigue", "habituation"),
        param_schema={"A": {"desc": "first-exposure response"}, "gamma": {"desc": "retention 0<γ<1"},
                      "floor": {"desc": "irreducible response", "required": False}},
        required_inputs=("n_exposures",), fitting_method="nonlinear MLE on repeated-exposure data",
        invariants=("A at n=0", "decreasing in n", "→floor"), extrapolation_behavior="approaches_floor",
        theory="message/donor/notification fatigue: the nth exposure moves the receiver less than the 1st. "
               "Distinct from saturation (which is about the STIMULUS level, not repetition count)",
        maturity="structural_candidate",
        _eval=lambda p, i: float(p.get("floor", 0.0)) + (float(p["A"]) - float(p.get("floor", 0.0)))
        * float(p["gamma"]) ** max(0.0, _x(i, "n_exposures"))))

    # ---- 16. habituation (hyperbolic decline) ----------------------------------------------------
    R(StructuralForm(
        form_id="habituation", version="1.0.0", title="Habituation A/(1+λ·n)",
        output_domain="real", monotonicity="decreasing", phenomena=("habituation", "fatigue"),
        param_schema={"A": {"desc": "initial response"}, "lam": {"desc": "habituation rate ≥0"}},
        required_inputs=("n_exposures",), fitting_method="nonlinear MLE",
        invariants=("A at n=0", "decreasing", "→0"),
        theory="hyperbolic novelty decay — slower tail than geometric fatigue", maturity="structural_candidate",
        _eval=lambda p, i: float(p["A"]) / (1.0 + max(0.0, float(p["lam"]) * _x(i, "n_exposures")))))

    # ---- 17. refractory-period response ----------------------------------------------------------
    R(StructuralForm(
        form_id="refractory", version="1.0.0", title="Refractory recovery A·(1−e^{−Δt/ρ})",
        output_domain="unit_interval", monotonicity="increasing", phenomena=("refractory", "recency"),
        param_schema={"A": {"desc": "fully-recovered response"}, "rho": {"desc": "recovery time-constant"}},
        required_inputs=("time_since_last",), fitting_method="nonlinear MLE",
        invariants=("0 immediately after event", "→A after long rest"),
        theory="a receiver just acted/was-exposed is temporarily unresponsive; responsiveness recovers with "
               "rest (neural/behavioral refractory period)", maturity="structural_candidate",
        _eval=lambda p, i: float(p["A"]) * (1.0 - math.exp(-max(0.0, _x(i, "time_since_last")
                                                                / (float(p["rho"]) or 1.0))))))

    # ---- 18. hysteresis (bistable, state-dependent thresholds) -----------------------------------
    def _hysteresis(p, i):
        # different activation/deactivation thresholds; requires current binary state on input
        x = _x(i, "x")
        state = int(inputs_get(i, "active", 0))
        up, down = float(p["tau_up"]), float(p["tau_down"])   # up > down
        if state == 0:
            return 1.0 if x >= up else 0.0
        return 0.0 if x <= down else 1.0
    R(StructuralForm(
        form_id="hysteresis", version="1.0.0", title="Hysteresis (τ_up > τ_down, state-dependent)",
        output_domain="unit_interval", monotonicity="context",
        phenomena=("hysteresis", "path_dependence", "tipping"),
        param_schema={"tau_up": {"desc": "activation threshold"}, "tau_down": {"desc": "deactivation"},
                      },
        required_inputs=("x", "active"), fitting_method="two-threshold change-point search",
        invariants=("τ_up ≥ τ_down", "bistable band", "output depends on prior state"),
        extrapolation_behavior="state_latched",
        failure_conditions=("τ_up < τ_down collapses to a single threshold (no hysteresis)",),
        theory="trust/adoption lock-in: harder to activate than to sustain (or vice-versa). Path dependence "
               "and irreversibility come from the state-dependent band", maturity="structural_candidate",
        differentiable=False, _eval=_hysteresis))

    # ---- 19. change-point (segmented regression, one break) --------------------------------------
    def _changepoint(p, i):
        x = _x(i, "x")
        cp = float(p["cp"])
        if x < cp:
            return float(p["a0"]) + float(p["b0"]) * (x - cp)
        return float(p["a1"]) + float(p["b1"]) * (x - cp)
    R(StructuralForm(
        form_id="change_point", version="1.0.0", title="Segmented regression (one change-point)",
        output_domain="real", monotonicity="unconstrained", phenomena=("regime", "threshold"),
        param_schema={"cp": {"desc": "break location"}, "a0": {"desc": "left value at cp"},
                      "b0": {"desc": "left slope"}, "a1": {"desc": "right value at cp"},
                      "b1": {"desc": "right slope"}},
        required_inputs=("x",), fitting_method="profile likelihood over cp (grid)",
        invariants=("two linear regimes",),
        theory="an endogenous or exogenous structural break splits the response into regimes",
        maturity="fitted_available", _eval=_changepoint))

    # ---- 20. finite mixture ----------------------------------------------------------------------
    def _mixture(p, i):
        # components: [{form_id, params, weight}]; expectation over latent class
        tot, z = 0.0, 0.0
        for comp in p["components"]:
            w = float(comp["weight"])
            tot += w * get_form(comp["form_id"]).eval(comp["params"], i)
            z += w
        return tot / (z or 1.0)
    R(StructuralForm(
        form_id="finite_mixture", version="1.0.0", title="Finite mixture Σ π_k f_k(x)",
        output_domain="real", monotonicity="unconstrained", phenomena=("heterogeneity", "regime"),
        param_schema={"components": {"desc": "[{form_id, params, weight}] latent classes"}},
        required_inputs=("x",), fitting_method="EM", regularization="component count (validation/BIC)",
        invariants=("weights sum to 1",),
        failure_conditions=("degenerate/empty component", "label switching"),
        theory="latent-class heterogeneity: distinct sub-populations respond via distinct shapes",
        maturity="fitted_available", _eval=_mixture))

    # ---- 21. latent regime model -----------------------------------------------------------------
    def _regime(p, i):
        # regime index supplied by context/Phase-3 inference; params indexed by regime
        r = str(inputs_get(i, "regime", p.get("default_regime", "0")))
        table = p["regimes"]
        spec = table.get(r) or table.get(p.get("default_regime", "0")) or next(iter(table.values()))
        return get_form(spec["form_id"]).eval(spec["params"], i)
    R(StructuralForm(
        form_id="regime_model", version="1.0.0", title="Regime-indexed response (regime from context)",
        output_domain="real", monotonicity="context", phenomena=("regime", "path_dependence"),
        param_schema={"regimes": {"desc": "{regime_id: {form_id, params}}"},
                      "default_regime": {"desc": "fallback id", "required": False}},
        required_inputs=("regime",), fitting_method="regime-conditional fits (regime from data/Phase-3)",
        invariants=("regime identity NEVER LLM-invented — from data/documented-event/Phase-3 latent",),
        failure_conditions=("unknown regime at runtime → widen / fallback",),
        theory="platform-policy / conflict-level / attention-state regimes change the mechanism's parameters",
        maturity="structural_candidate", _eval=_regime))

    # ---- 22. hidden-Markov regime (evaluate given regime posterior) ------------------------------
    def _hmm(p, i):
        # runtime consumes a regime posterior {regime_id: prob}; marginalize E_r[f_r(x)]
        post = inputs_get(i, "regime_posterior", None) or {p.get("default_regime", "0"): 1.0}
        table = p["regimes"]
        tot, z = 0.0, 0.0
        for r, pr in post.items():
            spec = table.get(str(r))
            if spec is None:
                continue
            tot += float(pr) * get_form(spec["form_id"]).eval(spec["params"], i)
            z += float(pr)
        return tot / (z or 1.0)
    R(StructuralForm(
        form_id="hmm_regime", version="1.0.0", title="HMM regime marginalization E_r[f_r(x)]",
        output_domain="real", monotonicity="context", phenomena=("regime", "path_dependence"),
        param_schema={"regimes": {"desc": "{regime_id: {form_id, params}}"},
                      "transition": {"desc": "regime transition matrix (offline-inferred)", "required": False}},
        required_inputs=("regime_posterior",), fitting_method="Baum–Welch (offline)",
        invariants=("marginalizes over regime uncertainty — no hard regime label at runtime",),
        theory="latent regime with temporal persistence; runtime honours the inferred regime posterior",
        maturity="structural_candidate", _eval=_hmm))

    # ---- 23. mixture-of-experts (gated) ----------------------------------------------------------
    def _moe(p, i):
        # gate weights from a softmax over gate features; experts are forms
        gates = p["gate"]             # {expert_id: {weights, intercept}}
        experts = p["experts"]        # {expert_id: {form_id, params}}
        logits = {e: _linpred(g, i) for e, g in gates.items()}
        m = max(logits.values())
        exps = {e: math.exp(_clip(v - m, -30, 30)) for e, v in logits.items()}
        z = sum(exps.values()) or 1.0
        return sum((exps[e] / z) * get_form(experts[e]["form_id"]).eval(experts[e]["params"], i)
                   for e in experts)
    R(StructuralForm(
        form_id="mixture_of_experts", version="1.0.0", title="Gated mixture of expert forms",
        output_domain="real", monotonicity="context", phenomena=("regime", "interaction", "heterogeneity"),
        param_schema={"gate": {"desc": "{expert: {weights,intercept}} softmax gate"},
                      "experts": {"desc": "{expert: {form_id, params}}"}},
        required_inputs=("features",), fitting_method="joint EM / gradient (offline)",
        invariants=("gate weights sum to 1",),
        theory="context selects which expert response applies; smoother than a hard regime split",
        maturity="structural_candidate", _eval=_moe))

    # ---- 24. survival hazard (general window) ----------------------------------------------------
    R(StructuralForm(
        form_id="survival_hazard", version="1.0.0", title="Proportional hazard window P=1−exp(−λ(x)·W)",
        output_domain="prob_window", monotonicity="context", phenomena=("saturation",),
        param_schema={"log_lambda": {"desc": "log baseline hazard"}, "weights": {"desc": "covariate log-HR"},
                      "standardizer": {"desc": "{feat:[mu,sd]}", "required": False}},
        required_inputs=("features", "window_days"), fitting_method="Cox/parametric survival (offline)",
        invariants=("hazard ≥ 0", "P in [0,1]"), time_scales=("window", "horizon"),
        theory="time-to-event mechanism (churn, response, activation); the window prob is what executes",
        maturity="fitted_available",
        _eval=lambda p, i: 1.0 - math.exp(-min(50.0, math.exp(_clip(float(p.get("log_lambda", 0.0))
                                                                     + _linpred({"weights": p.get("weights"),
                                                                                 "intercept": 0.0,
                                                                                 "standardizer": p.get(
                                                                                     "standardizer")}, i),
                                                                     -30, 30)) * _x(i, "window_days", 1.0)))))

    # ---- 25. recurrent-event hazard (history-conditioned) ----------------------------------------
    R(StructuralForm(
        form_id="recurrent_event_hazard", version="1.0.0",
        title="Recurrent-event intensity λ0·exp(β·history)", output_domain="nonneg_rate",
        monotonicity="context", phenomena=("recency", "reinforcement", "self_excitation"),
        param_schema={"lam0": {"desc": "baseline rate"}, "beta_count": {"desc": "prior-event count effect"},
                      "beta_recency": {"desc": "time-since-last effect"}},
        required_inputs=("cum_count", "time_since_last"), fitting_method="Andersen–Gill / point-process MLE",
        invariants=("rate ≥ 0",),
        theory="repeated actions where prior events raise/lower the next-event rate (participation streaks)",
        maturity="structural_candidate",
        _eval=lambda p, i: max(0.0, float(p["lam0"]) * math.exp(_clip(
            float(p.get("beta_count", 0.0)) * _x(i, "cum_count")
            + float(p.get("beta_recency", 0.0)) * math.exp(-_x(i, "time_since_last") / 24.0), -30, 30)))))

    # ---- 26. self-exciting (Hawkes intensity) — PRESERVED-QUARANTINED for promotion --------------
    def _hawkes(p, i):
        # λ(t) = μ + αω Σ exp(−ω(t−t_i)); runtime consumes precomputed kernel sum A(t)
        mu, al, om = float(p["mu"]), float(p["alpha"]), float(p["omega"])
        A = _x(i, "kernel_sum")
        return max(0.0, mu + al * om * A)
    R(StructuralForm(
        form_id="self_exciting", version="1.0.0", title="Self-exciting intensity μ+αωΣe^{−ω(t−tᵢ)}",
        output_domain="nonneg_rate", monotonicity="context", phenomena=("self_excitation", "tipping"),
        param_schema={"mu": {"desc": "background rate"}, "alpha": {"desc": "branching ratio (<1)"},
                      "omega": {"desc": "decay rate"}},
        required_inputs=("kernel_sum",), fitting_method="EM/MLE exponential kernel (offline)",
        invariants=("α<1 for stability",),
        failure_conditions=("α≥1 → explosive", "constant μ underfits bursty circadian streams (Higgs Hawkes "
                            "FAILED held-out vs Poisson — see failures ledger; do NOT auto-promote)"),
        theory="cascades where each event raises the near-term rate of further events; QUARANTINED on Higgs "
               "(held-out MAE 1098.9 > Poisson 973.0) — a context-specific pack may be re-tested separately",
        maturity="structural_candidate", _eval=_hawkes))

    # ---- 27. self-inhibiting intensity -----------------------------------------------------------
    R(StructuralForm(
        form_id="self_inhibiting", version="1.0.0", title="Self-inhibiting intensity μ·exp(−βΣe^{−ω·Δ})",
        output_domain="nonneg_rate", monotonicity="decreasing", phenomena=("self_inhibition", "refractory"),
        param_schema={"mu": {"desc": "background"}, "beta": {"desc": "inhibition strength ≥0"},
                      "omega": {"desc": "decay"}},
        required_inputs=("kernel_sum",), fitting_method="MLE (offline)", invariants=("rate ≥ 0",),
        theory="recent events SUPPRESS the near-term rate (attention exhaustion, saturation of an audience)",
        maturity="structural_candidate",
        _eval=lambda p, i: max(0.0, float(p["mu"]) * math.exp(-max(0.0, float(p["beta"])
                                                                   * float(p["omega"]) * _x(i, "kernel_sum"))))))

    # ---- 28. finite-population epidemic transition (SIR-like) -------------------------------------
    R(StructuralForm(
        form_id="finite_population", version="1.0.0", title="Finite-pop new-adopter rate β·I·S/N",
        output_domain="nonneg_rate", monotonicity="non_monotone",
        phenomena=("finite_population", "saturation", "tipping"),
        param_schema={"beta": {"desc": "transmission"}, "N": {"desc": "population size"}},
        required_inputs=("infected", "susceptible"), fitting_method="epidemic curve fit (offline)",
        invariants=("rate=0 when S=0 or I=0", "peaks then declines"),
        theory="susceptible depletion: adoption accelerates then dies as the susceptible pool runs out — the "
               "Bass/SIR saturation the cascade cannot exceed",
        maturity="fitted_available",
        _eval=lambda p, i: max(0.0, float(p["beta"]) * _x(i, "infected") * _x(i, "susceptible")
                              / (float(p["N"]) or 1.0))))

    # ---- 29. nonlinear state-space step (deterministic drift update) -----------------------------
    R(StructuralForm(
        form_id="nonlinear_state_space", version="1.0.0",
        title="Nonlinear state update xₜ₊₁ = x + dt·g(x, u)", output_domain="real",
        monotonicity="context", phenomena=("path_dependence", "hysteresis"),
        param_schema={"drift": {"desc": "{form_id, params} for g(x,u)"}, "dt": {"desc": "step"}},
        required_inputs=("x",), fitting_method="state-space MLE (offline)",
        invariants=("bounded step",),
        theory="latent state evolves nonlinearly through time (accumulated trust, momentum) — the engine "
               "steps it event-by-event so path dependence is real, not assumed",
        maturity="structural_candidate",
        _eval=lambda p, i: _x(i, "x") + float(p.get("dt", 1.0))
        * get_form(p["drift"]["form_id"]).eval(p["drift"]["params"], i)))


def _register_growth_forms():
    # logistic (Verhulst) growth increment ΔS = r·S·(1−S/L) — the saturating adoption ODE, executed as a
    # per-step increment by the state-step operator. Bends and peaks; the nonlinearity that stops overshoot.
    register_form(StructuralForm(
        form_id="logistic_growth", version="1.0.0", title="Logistic growth increment r·S·(1−S/L)",
        output_domain="real", monotonicity="non_monotone",
        phenomena=("saturation", "finite_population", "tipping"),
        param_schema={"r": {"desc": "intrinsic growth rate"}, "L": {"desc": "carrying capacity"}},
        required_inputs=("x",), fitting_method="nonlinear LS on the observed trajectory ≤ cutoff",
        invariants=("increment→0 as S→L", "self-limiting"), extrapolation_behavior="saturates_at_L",
        theory="Verhulst/Bass adoption saturation: growth slows as the susceptible pool depletes — the "
               "mechanism a linear/exponential extrapolation lacks (and why it overshoots)",
        maturity="fitted_available",
        _eval=lambda p, i: float(p["r"]) * _x(i, "x") * (1.0 - _x(i, "x") / (float(p["L"]) or 1e-9))))
    # linear/constant growth increment (the simpler Phase-6-style extrapolation baseline)
    register_form(StructuralForm(
        form_id="linear_growth", version="1.0.0", title="Linear/exponential growth increment g·S + c",
        output_domain="real", monotonicity="unconstrained", phenomena=("none",),
        param_schema={"g": {"desc": "proportional rate (exponential)"}, "c": {"desc": "constant slope"}},
        required_inputs=("x",), fitting_method="log-linear / linear trend on recent window",
        invariants=("no saturation — grows without bound",), extrapolation_behavior="unbounded_growth",
        theory="the non-saturating extrapolation a nonlinear saturation form must beat on a peaking trajectory",
        maturity="fitted_available",
        _eval=lambda p, i: float(p.get("g", 0.0)) * _x(i, "x") + float(p.get("c", 0.0))))


def _hinge_basis(x, knots):
    """Piecewise-linear spline basis: [x, (x−k1)+, (x−k2)+, …] — the standard GAM smooth-term basis,
    evaluable in pure Python from serialized knots."""
    return [x] + [max(0.0, x - k) for k in knots]


def _register_gam():
    def _gam(p, i):
        feats = i.get("features") or i
        s = float(p.get("intercept", 0.0))
        stdz = p.get("standardizer") or {}
        # linear terms
        for k, w in (p.get("linear") or {}).items():
            v = float(feats.get(k, 0.0) or 0.0)
            if k in stdz:
                mu, sd = stdz[k]
                v = (v - mu) / (sd or 1.0)
            s += float(w) * v
        # smooth (spline-basis) terms — nonlinear, additive
        for var, sm in (p.get("smooth") or {}).items():
            x = float(feats.get(var, 0.0) or 0.0)
            basis = _hinge_basis(x, sm["knots"])
            for b, coef in zip(basis, sm["coefs"]):
                s += float(coef) * b
        # explicit interactions (dedicated interaction standardizer — see _interaction_term)
        s += _interaction_term(p, feats)
        return sigmoid(s) if p.get("link", "logit") == "logit" else s
    register_form(StructuralForm(
        form_id="gam", version="1.0.0", title="Generalized additive model σ(Σ linear + Σ smooth(x))",
        output_domain="unit_interval", monotonicity="non_monotone",
        phenomena=("saturation", "diminishing_returns", "inverted_u", "interaction", "threshold"),
        param_schema={"linear": {"desc": "{feat: coef} linear terms"},
                      "smooth": {"desc": "{var: {knots, coefs}} piecewise-linear spline smooths"},
                      "interactions": {"desc": "{a*b: coef}", "required": False},
                      "intercept": {"desc": "bias"}, "standardizer": {"desc": "{feat:[mu,sd]}",
                                                                      "required": False},
                      "link": {"desc": "logit|identity", "required": False}},
        required_inputs=("features",), fitting_method="penalized IRLS / spline-basis logistic (offline)",
        regularization="knot count + ridge (validation-selected)",
        invariants=("additive in smooth terms",), extrapolation_behavior="linear_tails",
        theory="a nonlinear response that stays additive + interpretable: each feature gets its own fitted "
               "shape (the churn tenure hazard, the Upworthy headline-length curve) without a black box",
        maturity="fitted_available", _eval=_gam))


def inputs_get(inputs, key, default):
    v = inputs.get(key, default)
    return v if v is not None else default


_register_core_forms()
_register_gam()
_register_growth_forms()
