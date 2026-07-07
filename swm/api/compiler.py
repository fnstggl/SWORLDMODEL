"""The world-model compiler — Stage ② : question → structural model → Monte-Carlo forecast.

This is the keystone. Everything else in the system becomes its library. For a question, the LLM emits a
`ModelSpec` (mechanism + variables + structural equations + outcome + horizon); this module INSTANTIATES
that spec into a runnable simulation and runs it through the calibrated-time Monte-Carlo engine
(`swm/simulation/structural.py`). The mechanism is chosen PER QUESTION from a library — a competition
compiles to a bracket, an institution to a committee, a population to an electorate, a coupled causal
system to a generic SCM — so the right generative process runs, not one generic mechanism forced onto
everything (the bug that lost the NBA forecast).

Backends are pluggable exactly like every other LLM-touching part: `anthropic_compile_fn` (production) or
`cached_compile_fn` (reproducible/dev). The compiler never evaluates arbitrary code — structural equations
run through the whitelisted `safe_eval`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from swm.api.model_spec import ModelSpec, parse_spec, safe_eval
from swm.simulation.structural import (StructuralModel, SVar, montecarlo, prob_of, variance_decomposition)


# ============================ mechanism library ============================
def _event_pred(outcome):
    """Build a predicate for the outcome event, e.g. {'op':'>','value':0.5}. None if no event."""
    ev = outcome.get("event")
    if not ev:
        return None
    op, val = ev.get("op", ">"), float(ev.get("value", 0.5))
    return {">": lambda x: x > val, ">=": lambda x: x >= val, "<": lambda x: x < val,
            "<=": lambda x: x <= val}.get(op, lambda x: x > val)


@dataclass
class Sampler:
    """One mechanism's per-trajectory sampler — the SINGLE SOURCE OF TRUTH for both the aggregate forecast
    (`_run_*` Monte-Carlo it) and the decision/navigable layers (which need per-sample outcomes + the
    exogenous factors that defined each world). `traced(rng) -> (outcome, factors)`; `once` is the scalar/
    label view. `kind` is 'numeric' or 'categorical'; `default_pred` is the P(event) predicate; `aux`
    carries mechanism extras (e.g. the strengths-known bracket sampler, the generic-SCM model for the
    reducible/irreducible split)."""
    traced: object
    kind: str = "numeric"
    labels: tuple = ()
    default_pred: object = None
    aux: dict = field(default_factory=dict)

    def once(self, rng):
        return self.traced(rng)[0]


def _sampler_generic_scm(spec: ModelSpec) -> Sampler:
    variables = {v.name: SVar(v.name, v.value, v.est_sd, v.volatility, v.lo, v.hi) for v in spec.variables}

    def drift_fn(state, dt):
        return {name: safe_eval(expr, state) for name, expr in spec.equations.items()}

    target = spec.outcome.get("variable") or (spec.variables[0].name if spec.variables else None)
    expr = spec.outcome.get("expr")                               # optional algebraic readout of final state
    if expr:                                                      # e.g. profit = price*demand — a decision KPI
        outcome_fn = lambda s: safe_eval(expr, s)
        target = expr
    else:
        outcome_fn = lambda s: s.get(target, 0.0)
    model = StructuralModel(variables, drift_fn=drift_fn, outcome_fn=outcome_fn)
    interventions = spec.extra.get("_interventions")             # temporal do-operators (inject_event / policy)
    return Sampler(traced=model.simulate_once_traced(spec.horizon, spec.dt, interventions=interventions),
                   kind="numeric", default_pred=_event_pred(spec.outcome),
                   aux={"model": model, "target": target})


def _run_generic_scm(spec: ModelSpec, n: int, seed: int) -> dict:
    """Coupled continuous variables with LLM-written structural equations, as a calibrated-time diffusion."""
    s = _sampler_generic_scm(spec)
    mc = montecarlo(s.once, n=n, seed=seed)
    p_event = prob_of(s.once, s.default_pred, n=n, seed=seed + 1) if s.default_pred else None
    decomp = variance_decomposition(s.aux["model"], spec.horizon, spec.dt, n=min(n, 4000))
    return {"mechanism": "generic_scm", "target": s.aux["target"], "mean": round(mc["mean"], 4),
            "interval_80": [round(mc["p05"], 4), round(mc["p95"], 4)],
            "p_event": round(p_event, 4) if p_event is not None else None, "uncertainty": decomp}


def _to_elo(strength):
    return float(strength) if strength > 100 else 1500.0 + (float(strength) - 0.5) * 600.0


def _series(a, b, elo, series_len, hca, rng):
    need = series_len // 2 + 1
    home_a = {1, 2, 5, 7} if series_len == 7 else set(range(1, series_len, 2))
    wa = wb = 0; g = 1
    while wa < need and wb < need:
        diff = (elo[a] + (hca if g in home_a else -hca)) - elo[b]
        if rng.random() < 1.0 / (1.0 + 10 ** (-diff / 400.0)):
            wa += 1
        else:
            wb += 1
        g += 1
    return a if wa >= need else b


def _bracket_once(names, elo, groups, series_len, hca, rng):
    def one_bracket(teams):
        seeds = sorted(teams, key=lambda t: -elo[t])
        while len(seeds) > 1:
            m = len(seeds); nxt = [_series(seeds[i], seeds[m - 1 - i], elo, series_len, hca, rng)
                                   for i in range(m // 2)]
            if m % 2:
                nxt.append(seeds[m // 2])
            seeds = sorted(nxt, key=lambda t: -elo[t])
        return seeds[0]
    if groups:
        winners = [one_bracket([t for t in names if groups.get(t) == g]) for g in sorted(set(groups.values()))]
        return one_bracket(winners) if len(winners) > 1 else winners[0]
    return one_bracket(names)


def _sampler_bracket(spec: ModelSpec) -> Sampler:
    comp = spec.extra.get("competitors", [])
    names = [c["name"] for c in comp]
    base = {c["name"]: _to_elo(c.get("strength", 0.5)) for c in comp}
    est = {c["name"]: float(c.get("est_sd", 30.0)) for c in comp}
    groups = spec.extra.get("groups") or {}
    slen = int(spec.extra.get("series_length", 7)); hca = float(spec.extra.get("home_advantage", 100.0))

    def traced(rng):
        elo, factors = {}, {}
        for t in names:                                  # one gauss per team, in order (rng-parity)
            z = rng.gauss(0, 1)
            elo[t] = base[t] + z * est[t]                # == base + gauss(0, est): same draw, same value
            factors[f"{t}~strength"] = z
        return _bracket_once(names, elo, groups, slen, hca, rng), factors

    def once_known(rng):
        return _bracket_once(names, base, groups, slen, hca, rng)

    return Sampler(traced=traced, kind="categorical", labels=tuple(names),
                   aux={"once_known": once_known, "names": names})


def _run_bracket(spec: ModelSpec, n: int, seed: int) -> dict:
    """A competition: seeded single-elimination bracket of best-of-k series between competitor strengths."""
    s = _sampler_bracket(spec)
    known = montecarlo(s.aux["once_known"], n=n, seed=seed)
    unc = montecarlo(s.once, n=n, seed=seed + 1)
    target = spec.outcome.get("target")
    return {"mechanism": "bracket", "champion_distribution": {k: round(v, 4) for k, v in
            list(unc["distribution"].items())[:8]}, "favorite": unc["mode"],
            "p_target": round(unc["distribution"].get(target, 0.0), 4) if target else None,
            "p_target_strengths_known": round(known["distribution"].get(target, 0.0), 4) if target else None,
            "irreducible_note": "gap between the two p_target values is what better strength estimates "
                                "could buy; the rest of the spread is irreducible tournament variance"}


def _sampler_committee(spec: ModelSpec) -> Sampler:
    from swm.simulation.agent_society import AgentSociety, PersonaAgent
    ag = spec.extra.get("agents", [])
    pos_sd = float(spec.extra.get("position_sd", 0.08))
    soc = AgentSociety(homophily=float(spec.extra.get("homophily", 0.5)),
                       consensus_pull=float(spec.extra.get("consensus_pull", 0.3)),
                       rounds=int(spec.extra.get("rounds", 5)),
                       public_field=spec.extra.get("public_field"))

    def traced(rng):
        factors, agents = {}, []
        for a in ag:                                     # one gauss per agent, in order (rng-parity)
            z = rng.gauss(0, 1)
            pos = min(1, max(0, a.get("position", 0.5) + z * pos_sd))
            factors[f"{a['id']}~pos"] = z
            agents.append(PersonaAgent(a["id"], {"pos": a.get("position", 0.5)}, position=pos,
                                       influence=a.get("influence", 1.0), openness=a.get("openness", 0.35),
                                       conviction=a.get("conviction", 0.45),
                                       public_sensitivity=a.get("public_sensitivity", 0.0)))
        for p in agents:
            p.variables["pos"] = p.position
        out = soc.simulate(None, agents, lambda a, prop: a.variables["pos"])
        return out["vote_share"], factors

    return Sampler(traced=traced, kind="numeric",
                   default_pred=_event_pred(spec.outcome) or (lambda x: x > 0.5), aux={"n_agents": len(ag)})


def _run_committee(spec: ModelSpec, n: int, seed: int) -> dict:
    """An institution: named stakeholders deliberate (AgentSociety); Monte-Carlo over position uncertainty."""
    s = _sampler_committee(spec)
    mc = montecarlo(s.once, n=n, seed=seed)
    p_event = prob_of(s.once, s.default_pred, n=n, seed=seed + 1)
    return {"mechanism": "committee", "mean_vote_share": round(mc["mean"], 4),
            "interval_80": [round(mc["p05"], 4), round(mc["p95"], 4)],
            "p_event": round(p_event, 4), "n_agents": s.aux["n_agents"]}


def _sampler_electorate(spec: ModelSpec) -> Sampler:
    from swm.simulation.mean_field import MeanFieldRollout
    from swm.simulation.population_simulator import (DemographicCell, PopulationSimulator, marginal_share,
                                                     share_aggregator)
    raw = spec.extra.get("cells", [])
    coupling = spec.extra.get("turnout_coupling", 0.0)
    steps = int(spec.horizon)
    agg = share_aggregator if spec.extra.get("participation_weighted") else marginal_share
    ksoc = float(spec.extra.get("k_social", 0.1)); kproof = float(spec.extra.get("k_proof", 0.0))

    def traced(rng):
        factors, cells = {}, []
        for i, c in enumerate(raw):                      # one gauss per cell, in order (rng-parity)
            z = rng.gauss(0, 1); sd = c.get("est_sd", 0.03)
            cid = c.get("id", str(i))
            factors[f"{cid}~stance"] = z
            cells.append(DemographicCell(cid, float(c.get("weight", 1.0)),
                                         min(1, max(0, c.get("stance", 0.5) + z * sd)),
                                         float(c.get("responsiveness", 0.3)), float(c.get("turnout", 1.0)),
                                         c.get("region", "")))
        sim = PopulationSimulator(rollout=MeanFieldRollout(k_social=ksoc, k_proof=kproof), aggregator=agg,
                                  turnout_coupling=coupling)
        return sim.simulate(cells, steps=max(1, steps))["coupled"], factors

    return Sampler(traced=traced, kind="numeric",
                   default_pred=_event_pred(spec.outcome) or (lambda x: x > 0.5),
                   aux={"n_cells": len(raw), "n_cap": 2000})


def _run_electorate(spec: ModelSpec, n: int, seed: int) -> dict:
    """A population: demographic cells with coupled opinion + turnout (PopulationSimulator)."""
    s = _sampler_electorate(spec)
    cap = min(n, s.aux["n_cap"])
    mc = montecarlo(s.once, n=cap, seed=seed)
    p_event = prob_of(s.once, s.default_pred, n=cap, seed=seed + 1)
    return {"mechanism": "electorate", "mean_share": round(mc["mean"], 4),
            "interval_80": [round(mc["p05"], 4), round(mc["p95"], 4)], "p_event": round(p_event, 4),
            "n_cells": s.aux["n_cells"]}


def _sampler_single_agent(spec: ModelSpec) -> Sampler:
    from swm.simulation.response_model import quantities
    person = spec.extra.get("person", {}); message = spec.extra.get("message", {})
    sds = spec.extra.get("est_sd", {})
    import math as _m

    def traced(rng):
        pv, factors = {}, {}
        for k, v in person.items():                      # one gauss per person-var, in order (rng-parity)
            z = rng.gauss(0, 1); sd = sds.get(k, 0.0)
            pv[k] = v + z * sd
            if sd > 0:
                factors[f"{k}~est"] = z
        q = quantities(pv, {}, message)
        zt = 2.2 * (q["receptivity"] - 0.5) + 2.2 * (q["quality"] - 0.5) + 1.5 * (q["receptivity"] * q["quality"] - 0.25)
        return 1.0 / (1.0 + _m.exp(-zt)), factors

    return Sampler(traced=traced, kind="numeric", aux={"message": message})


def _run_single_agent(spec: ModelSpec, n: int, seed: int) -> dict:
    """One person: response probability via the Level-1 quantities, Monte-Carlo over variable uncertainty."""
    s = _sampler_single_agent(spec)
    mc = montecarlo(s.once, n=n, seed=seed)
    return {"mechanism": "single_agent", "p_respond_mean": round(mc["mean"], 4),
            "interval_80": [round(mc["p05"], 4), round(mc["p95"], 4)]}


MECHANISMS = {"generic_scm": _run_generic_scm, "bracket": _run_bracket, "committee": _run_committee,
              "electorate": _run_electorate, "single_agent": _run_single_agent}
SAMPLERS = {"generic_scm": _sampler_generic_scm, "bracket": _sampler_bracket, "committee": _sampler_committee,
            "electorate": _sampler_electorate, "single_agent": _sampler_single_agent}


def build_sampler(spec: ModelSpec) -> Sampler:
    """The action/navigable layers' entry point: a per-trajectory sampler for ANY mechanism (single source
    of truth with `_run_*`). `build_sampler(action.apply(spec))` is how an intervention becomes a rollout."""
    return SAMPLERS.get(spec.mechanism, _sampler_generic_scm)(spec)


@dataclass
class CompiledModel:
    spec: ModelSpec

    def run(self, n: int = 8000, seed: int = 0) -> dict:
        handler = MECHANISMS.get(self.spec.mechanism, _run_generic_scm)
        out = handler(self.spec, n, seed)
        out.update({"horizon": self.spec.horizon, "dt": self.spec.dt, "rationale": self.spec.rationale})
        return out


# ============================ the compiler (LLM front) ============================
def build_compile_prompt(question: str, context: str = "") -> str:
    return (
        "You are the STRUCTURAL-MODEL COMPILER of a social world model. Given a question and evidence, do "
        "NOT answer it directly. Instead specify the SIMULATION that, run forward, produces the answer — "
        "mirror the real generative process.\n\n"
        f"QUESTION: {question}\n" + (("EVIDENCE:\n" + context + "\n\n") if context else "\n") +
        "Choose the MECHANISM that matches the real process:\n"
        "  bracket      — a competition/tournament (sports, playoffs, elimination contests)\n"
        "  committee    — a small set of named decision-makers who deliberate (court, board, FOMC)\n"
        "  electorate   — a large population by demographic segment (elections, referenda, mass opinion)\n"
        "  single_agent — one specific person's decision/response\n"
        "  generic_scm  — coupled quantitative variables evolving over time (economy, approval, prices...)\n\n"
        "For the chosen mechanism emit ONLY JSON. For generic_scm: {\"mechanism\":\"generic_scm\","
        "\"variables\":[{\"name\",\"value\"(now),\"est_sd\"(uncertainty in that estimate),\"volatility\""
        "(how much it really moves per unit time),\"lo\",\"hi\"}],\"equations\":{\"var\":\"drift expression "
        "in terms of the variables\"},\"outcome\":{\"variable\":\"..\",\"event\":{\"op\":\">\",\"value\":0.5}},"
        "\"horizon\":<units>,\"dt\":1,\"rationale\":\"..\"}. "
        "For bracket: extra.competitors[{name,strength(Elo or 0..1),est_sd}], optional extra.groups, "
        "outcome.target. For committee: extra.agents[{id,position,influence,openness,conviction}]. For "
        "electorate: extra.cells[{stance,weight,turnout,est_sd}]. CALIBRATE volatility to the horizon's "
        "time units so one step of dt moves each variable by a realistic amount.")


def cached_compile_fn(cache: dict):
    """Replay committed specs keyed by a stable question id; raises on miss so a run never silently guesses."""
    def fn(key):
        if key not in cache:
            raise KeyError(f"no cached spec for {key!r}")
        return cache[key]
    return fn


def anthropic_compile_fn(api_key: str, model: str = "claude-sonnet-5", max_tokens: int = 1200):
    """PRODUCTION backend: the LLM emits the structural-model spec. Pure-stdlib urllib."""
    import urllib.request
    def fn(prompt):
        body = json.dumps({"model": model, "max_tokens": max_tokens,
                           "system": "You compile questions into runnable structural simulations. Emit ONLY "
                                     "the JSON spec. Calibrate volatility to real timescales.",
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                     headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return fn


@dataclass
class StructuralCompiler:
    """question (+context) -> ModelSpec via a pluggable LLM backend `compile_fn(prompt) -> spec JSON|dict`."""
    compile_fn: object

    def compile(self, question: str, context: str = "", *, key: str = None) -> CompiledModel:
        raw = self.compile_fn(key if key is not None else build_compile_prompt(question, context))
        return CompiledModel(parse_spec(raw))
