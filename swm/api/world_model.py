"""WorldModel — the front door. One `simulate(question)` call, the whole system underneath.

This realizes the vision in a single call: retrieve the relevant slice of the world, COMPILE it into the
right structural model (mechanism + variables + equations + calibrated timescales), roll it forward as a
Monte-Carlo ensemble, and return a calibrated outcome DISTRIBUTION with an honest reducible/irreducible
split and a forecastability verdict — never a false-confident point.

    wm = WorldModel(retriever, compiler)
    wm.simulate("Will the FOMC raise rates in July?")   # -> {mechanism, forecast, uncertainty, horizon, ...}

Both dependencies are pluggable: `retriever` from swm/api/retrieval.py (web search in prod, as-of in eval),
`compiler` a StructuralCompiler with an LLM or cached backend. The Levels 1-3, the bracket, and the
generic SCM are the compiler's mechanism library — this is the layer that turns them into "ask anything."

Self-correction is ON by default: the compiler is wrapped in a `ValidatingCompiler`, so every simulate()
call VALIDATES the compiled spec (simulate-and-inspect) and, if a `repair_fn` is supplied, REPAIRS
degeneracies (the EXP-067 loop) before running — the generated model is tested before it is trusted. The
`validation` report rides along in the output. Pass `validate=False` for the raw path.
"""
from __future__ import annotations

from dataclasses import dataclass

from swm.api.compiler import CompiledModel
from swm.api.spec_validator import ValidatingCompiler


@dataclass
class WorldModel:
    compiler: object                      # swm.api.compiler.StructuralCompiler
    retriever: object = None              # swm.api.retrieval.Retriever (optional; context can be passed in)
    n: int = 8000
    validate: bool = True                 # self-correct by default: validate (+ repair if repair_fn) each spec
    repair_fn: object = None              # LLM backend that fixes a flagged spec (None = validate-only)
    grounder: object = None               # swm.api.state_grounding.StateGrounder — auto-grounds the spec's
    #                                       high-leverage variable VALUES from live evidence before the run.
    #                                       `general_world_model()` wires the DeepSeek+web general router here.
    person_intake: object = None          # swm.api.person_intake.PersonIntake — for a question that turns on a
    #                                       SPECIFIC individual, assemble a dossier and ASK the user when the
    #                                       evidence is too thin, rather than fabricate a forecast (opt-in).

    def _compiler(self):
        if not self.validate:
            return self.compiler
        if isinstance(self.compiler, ValidatingCompiler):
            return self.compiler
        return ValidatingCompiler(self.compiler, repair_fn=self.repair_fn)

    def simulate(self, question: str, *, context: str = "", as_of: str = "", key: str = None,
                 n: int = None) -> dict:
        # PERSON INTAKE: if the question turns on a specific individual, assemble a dossier; ASK the user when
        # the evidence is too thin (never fabricate a person's disposition), else fold the dossier into context.
        person = None
        if self.person_intake is not None:
            try:
                pf = self.person_intake.preflight(question, user_context=context, as_of=(as_of or None))
            except Exception:
                pf = {"mode": "proceed", "person": None}
            if pf.get("mode") == "ask":
                return {"question": question, "mode": "needs_user_context", "person": pf["person"],
                        "questions": pf["questions"], "forecast": None, "grounding": None,
                        "headline": f"To simulate this I need your read on {pf['person']} — "
                                    f"{pf.get('reason', 'not enough public evidence to infer honestly')}."}
            if pf.get("person"):
                person = {k: pf.get(k) for k in ("person", "dossier_strength", "inferred_person_variables")}
                if pf.get("enriched_context"):
                    context = pf["enriched_context"]

        if self.retriever is not None and not context:
            ctx = self.retriever.retrieve(question, as_of=as_of)
            context = ctx.to_prompt()
            n_evidence = len(ctx)
        else:
            n_evidence = 0
        comp = self._compiler()
        compiled = comp.compile(question, context, key=key)
        validation = getattr(comp, "last_report", None)

        # AUTO-GROUND: measure the compiled spec's high-leverage variable VALUES from live evidence (triage ->
        # general router -> value + CI), so the simulation runs on THIS world, not the LLM's guessed state.
        grounding = None
        spec = compiled.spec
        if self.grounder is not None:
            try:
                spec, report = self.grounder.ground_spec(spec, question, as_of=(as_of or None))
                grounding = {"grounded": sum(1 for r in report if r.get("grounded")),
                             "n_high_leverage": sum(1 for r in report if r.get("high_leverage", True)),
                             "detail": report}
            except Exception as e:                        # grounding must never break a run — fall back to spec
                grounding = {"error": str(e)[:120]}
                spec = compiled.spec

        try:                                              # a spec validation could not repair may not run
            forecast = CompiledModel(spec).run(n=n or self.n)
        except Exception as e:
            forecast = {"mechanism": spec.mechanism, "error": str(e)[:120]}
        return {"question": question, "n_evidence": n_evidence,
                "mechanism": spec.mechanism, "forecast": forecast,
                "forecastable": _forecastable(forecast),
                "validation": validation, "grounding": grounding, "person": person,
                "headline": _headline(question, forecast),
                "spec": {"mechanism": spec.mechanism,
                         "variables": [(v.name, v.value, v.volatility) for v in spec.variables],
                         "equations": spec.equations, "outcome": spec.outcome,
                         "horizon": spec.horizon, "rationale": spec.rationale}}


def general_world_model(*, compile_fn=None, n=8000, ground=True, validate=True, person=True) -> WorldModel:
    """The recommended front door: compile ANY question → auto-GROUND its high-leverage variable values from
    live evidence (the DeepSeek+web general router, no feeds required) → run the calibrated simulation. This is
    the end-to-end default — a user asks anything, and the simulation runs on the real current world rather
    than the LLM's guessed state. Falls back to un-grounded compilation if no LLM key is configured.

    `person=True` also wires the PERSON-INTAKE preflight: a question that turns on a specific individual
    assembles a dossier and, when the evidence is too thin, ASKS the user for their read on that person instead
    of fabricating a disposition (EXP-089). Disabled automatically if no LLM key is configured."""
    from swm.api.compiler import StructuralCompiler
    if compile_fn is None:
        from swm.api.deepseek_backend import default_chat_fn
        compile_fn = default_chat_fn(system="You compile questions into runnable structural simulations. Emit "
                                            "ONLY the JSON spec.", max_tokens=1200)
    grounder = None
    if ground:
        from swm.api.live_grounding import live_router
        from swm.api.state_grounding import StateGrounder
        router = live_router()                            # general DeepSeek+web engine + structured overlays
        if router.retrieval is not None or router.sources:  # (Coinbase for observable market vars; the LLM
            grounder = StateGrounder(default=router)         # resolver routes to a feed only when one matches)
    intake = None
    if person:
        from swm.api.person_intake import build_person_intake
        intake = build_person_intake()                    # None if no LLM key -> front door behaves as before
    return WorldModel(compiler=StructuralCompiler(compile_fn), n=n, validate=validate, grounder=grounder,
                      person_intake=intake)


def _forecastable(forecast: dict):
    u = forecast.get("uncertainty")
    if isinstance(u, dict) and "forecastable" in u:
        return u["forecastable"]
    return None                           # not defined for categorical/agent mechanisms


def _headline(question: str, forecast: dict) -> str:
    m = forecast.get("mechanism")
    if m == "bracket":
        return (f"{forecast.get('favorite')} most likely ({forecast.get('p_target') or '-'} for the named "
                f"target); most of the spread is irreducible tournament variance")
    if forecast.get("p_event") is not None:
        return f"P(event) = {forecast['p_event']}  (80% interval {forecast.get('interval_80')})"
    if "mean" in forecast:
        return f"outcome ~ {forecast['mean']}  (80% interval {forecast.get('interval_80')})"
    return "see forecast"
