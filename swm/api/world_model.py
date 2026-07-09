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
                 n: int = None, answers=None, events=None, b0: float = None, horizon: float = None,
                 continuous_step=None, actions=None) -> dict:
        """One call. If a future-event `calendar` is available (passed as `events`, or an `EventCalendar`),
        this becomes a FORWARD question: roll the belief through sampled event trajectories and return the
        branching distribution + pivotal forks + (optional) best action — never a false-confident point,
        never abstention. Otherwise the compiler path runs; a question that turns on a SPECIFIC individual
        first assembles a dossier and ASKS the user when evidence is thin (answering re-invokes with
        `answers=` and auto-runs)."""
        # RESUME: answers to a previous needs_user_context ask are folded into the context, so the SAME call
        # auto-runs the full simulation (no separate "run" step — answering IS the trigger).
        if answers:
            context = (_format_answers(answers) + ("\n" + context if context else "")).strip()

        if events is not None:                            # FORWARD question -> branching-realities rollout
            return self.simulate_forward(question, events, context=context, as_of=as_of, b0=b0,
                                         horizon=horizon, continuous_step=continuous_step,
                                         actions=actions, n=n)

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
                        "resume_hint": "call simulate(question, answers=<your answers>) to auto-run",
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

    def simulate_forward(self, question: str, events, *, context: str = "", as_of: str = "",
                         b0: float = None, horizon: float = None, continuous_step=None, actions=None,
                         n: int = None) -> dict:
        """Forward question via the branching-realities rollout. `events` may be an `EventCalendar`, a list
        of structured event records, or a callable `build(question, context, horizon) -> EventCalendar`
        (e.g. `EventImpactJudge.build`). Returns the branching distribution + pivotal forks + reducible/
        irreducible split, and — if `actions` (a list of `(label, apply_fn(b0, calendar)->(b0, calendar))`)
        is given — the best action by P(desired outcome). Never abstains: past the horizon it returns the
        full branching distribution and labels which forks are reducible vs irreducible."""
        from swm.simulation.branching_rollout import forward_forecast
        from swm.transition.future_events import EventCalendar, events_from_records

        if b0 is None and self.retriever is not None:
            b0 = _retrieved_belief(self.retriever, question, as_of)
        b0 = 0.5 if b0 is None else float(b0)

        if callable(getattr(events, "build", None)):
            horizon = horizon or 6.0
            calendar = events.build(question, context, horizon)
        elif isinstance(events, EventCalendar):
            calendar = events
        else:
            calendar = events_from_records(list(events))
        if horizon is None:
            horizon = max([e.time for e in calendar.events], default=6.0)

        nn = n or self.n
        base = forward_forecast(b0, horizon, calendar, continuous_step=continuous_step, n=nn)

        best = None
        if actions:
            scored = []
            for label, apply_fn in actions:
                ab0, acal = apply_fn(b0, calendar)
                af = forward_forecast(ab0, horizon, acal, continuous_step=continuous_step, n=nn)
                scored.append({"action": label, "p_event": af["p_event"],
                               "interval_80": af["interval_80"], "pivotal_branches": af["pivotal_branches"]})
            scored.sort(key=lambda r: -r["p_event"])
            do_nothing = next((s for s in scored if s["action"] in ("do_nothing", "none", "baseline")), None)
            best = {"best": scored[0], "ranking": scored,
                    "lift_over_do_nothing": (round(scored[0]["p_event"] - do_nothing["p_event"], 4)
                                             if do_nothing else None)}

        return {"question": question, "mechanism": "branching", "b0": round(b0, 4), "horizon": horizon,
                "forecast": base, "forecastable": base["reducible_frac"] >= 0.1,
                "best_action": best, "headline": base["headline"],
                "events": [{"name": e.name, "time": e.time, "labels": e.labels()} for e in calendar.events]}


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


def _format_answers(answers) -> str:
    """Turn a user's answers to the intake questions into a context block the dossier can consume. Accepts a
    dict {question: answer}, a list of answers (or [question, answer] pairs), or a plain string."""
    if isinstance(answers, str):
        return answers.strip()
    if isinstance(answers, dict):
        return "\n".join(f"{q} — {a}" for q, a in answers.items() if str(a).strip())
    if isinstance(answers, (list, tuple)):
        parts = []
        for item in answers:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parts.append(f"{item[0]} — {item[1]}")
            elif str(item).strip():
                parts.append(str(item).strip())
        return "\n".join(parts)
    return str(answers or "").strip()


def _retrieved_belief(retriever, question: str, as_of: str):
    """Best-effort current belief (market/poll price) from retrieval; None if unavailable."""
    try:
        ctx = retriever.retrieve(question, as_of=as_of)
        for it in getattr(ctx, "items", []) or []:
            sc = getattr(it, "score", None)
            if isinstance(sc, (int, float)) and 0.0 <= sc <= 1.0:
                return float(sc)
    except Exception:
        pass
    return None


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
