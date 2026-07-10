"""AgentWorldModel — one call, ONE mechanism: grounded agents interacting.

The mechanism zoo is dead. There is no menu of bracket/committee/electorate/logistic for an LLM to pick
from (and pick wrong); there is one substrate — ground the scene, cast the real actors, let LLM-reasoned
personas interact over real dated rounds — and the only thing that varies per question is the CASTING
(who the agents are, what the answer space is, what the interaction structure is). Routing:

  collective_choice / population_share → SocietyRollout   (distribution over the NAMED options)
  individual_reaction                  → IndividualSimulator (this person × this exact message, N runs)
  artifact_optimization                → ArtifactOptimizer (real generated texts, ranked by the audience)

Sequence per call: ground (abstain loudly if the deciding facts can't be established — and short-circuit
with provenance if the evidence shows the world already answered) → cast → simulate → calibrate
(grade-or-abstain; a fitted shrink where the class earned one) → a native-typed Forecast.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from swm.engine.actions import ArtifactOptimizer
from swm.engine.calibrate import GradeRegistry, shrink_distribution
from swm.engine.casting import CastingDirector
from swm.engine.grounding import SceneGrounder
from swm.engine.individual import IndividualSimulator
from swm.engine.outcome import Forecast, build_headline
from swm.engine.router import ParadigmRouter
from swm.engine.society import SocietyRollout

_ARTIFACT_WORDS = ("headline", "subject line", "tagline", "copy", "slogan", "ad ", "landing page",
                   "best message", "best email")
_DIFFUSION_WORDS = ("go viral", "viral", "spread", "how far will", "reach how many", "shares will",
                    "retweet", "repost", "word of mouth", "buzz", "trend on", "amplif")


@dataclass
class AgentWorldModel:
    llm: object                            # cold backend (grounding, casting, variants)
    llm_hot: object = None                 # hot backend (persona decisions; temperature > 0)
    branches: int = 3
    max_rounds: int = 2
    k_artifacts: int = 5
    registry: GradeRegistry = field(default_factory=GradeRegistry)
    search_fn: object = None               # override retrieval (fixtures in tests)
    today: str = ""
    parametric: object = None              # main's parametric engine (swm.api.world_model.general_world_model)
    #                                        result object with .simulate — the NON-HUMAN stochastic fallback
    router: object = None                  # ParadigmRouter (people → agents, process → parametric)
    event_engine: str = "panel"            # binary "will X happen?" → "panel" (base-rate-anchored observer
    #                                        forecasters, accurate on markets) or "society" (voter segments)
    panel_reps: int = 2                    # observer-panel resamples per lens
    model_llms: dict = None                # {family: callable} multi-family panel backends (Lever 3); None ⇒
    #                                        single-model. Genuinely different pretraining decorrelates errors.
    route_contests: bool = True            # Lever 2: send contests/announcements to the parametric kernel
    flywheel: object = None                # FlywheelLog — the outcome-data moat. When set, every emitted
    #                                        (non-abstained) forecast is logged for later resolution + refit.

    def __post_init__(self):
        self.llm_hot = self.llm_hot or self.llm
        if self.router is None:
            self.router = ParadigmRouter(self.llm)

    # ---------------- the one entry point ----------------
    def simulate(self, question: str, **kw) -> dict:
        """The public call. Runs the simulation, then — the FLYWHEEL — logs every non-abstained forecast
        to the outcome stream so its eventual real resolution re-calibrates the engine (the moat)."""
        res = self._simulate(question, **kw)
        if self.flywheel is not None and not res.get("abstain") and res.get("distribution"):
            try:
                dist = res["distribution"]
                p = dist.get("yes", dist.get("responds"))
                p = float(p) if p is not None else float(max(dist.values()))
                cast = (res.get("detail") or {}).get("cast") or {}
                self.flywheel.log(
                    question=question, question_class=(res.get("calibration") or {}).get("class", "society:event"),
                    domain=(res.get("detail") or {}).get("domain", "deliberation"),
                    mechanism=res.get("mechanism", "?"), p=p, distribution=dist,
                    as_of=kw.get("as_of") or self.today, resolve_by=cast.get("resolve_by", ""),
                    engine_config={"branches": self.branches, "panel_reps": self.panel_reps,
                                   "event_engine": self.event_engine},
                    grounding={k: (res.get("grounding") or {}).get(k)
                               for k in ("coverage", "n_passages", "abstain")})
            except Exception:                              # logging must never break a forecast
                pass
        return res

    def _simulate(self, question: str, *, message: str = None, recipient: str = None,
                  channel: str = "email", evidence=None, as_of: str = None,
                  binary: bool = False, search_fn=None) -> dict:
        """`evidence` + `as_of` drive the LEAK-FREE backtest path: supply frozen as-of context (no live
        retrieval) and date the rollout from `as_of`, so a resolved-in-the-past question is scored on the
        information available BEFORE it resolved — the no-cheat contract. `binary=True` forces a yes/no
        answer space (for scoring against event-market benchmarks like ForecastBench)."""
        today = as_of or self.today or _time.strftime("%Y-%m-%d")

        # explicit individual ask (recipient + exact artifact) skips scene-casting: the scene IS the person
        if recipient and message:
            return self._individual(question, recipient, message, channel, today)

        # PARADIGM ROUTE — people → agents (always); a genuinely non-human stochastic process (price/rate/
        # record/launch) → main's parametric kernels. Backtest (binary) mode is pre-filtered to people, so it
        # never diverts. The router is biased hard toward agents; ties and ambiguity stay here.
        if not binary and self.parametric is not None and self.router.route(question) == "parametric":
            out = self.parametric.simulate(question, as_of=(as_of or ""))
            out["engine"] = "parametric_mechanism"        # main's grounded stochastic kernel, not the readout
            out.setdefault("routed", "process→parametric (no human-behavior generative core)")
            return out

        dossier = SceneGrounder(self.llm, search_fn=(search_fn or self.search_fn), today=today).ground(
            question, evidence=evidence)
        if dossier.abstain:
            f = Forecast(question=question, mechanism="abstained", abstain=True,
                         abstain_reason=dossier.abstain_reason, grounding=dossier.as_report(),
                         calibration={"grade": "n/a", "note": "no forecast was emitted"})
            f.headline = build_headline(f)
            return f.as_dict()

        if dossier.resolved:                              # the world already answered — report it, cited
            f = Forecast(question=question, mechanism="resolved_by_evidence",
                         distribution={str(dossier.resolved["answer"]): 1.0},
                         grounding=dossier.as_report(),
                         calibration={"grade": "resolved", "abstain_confident": False,
                                      "note": "outcome already decided per the cited evidence"},
                         headline=(f"ALREADY RESOLVED: {dossier.resolved['answer']} "
                                   f"[{dossier.resolved.get('source', '?')}] — "
                                   f"\"{str(dossier.resolved.get('evidence', ''))[:140]}\""))
            return f.as_dict()

        if binary:                                        # a binary "will X happen?" event (backtest/market)
            # Lever 2: a CONTEST (physical) or ANNOUNCEMENT (release/launch) is not social deliberation — route
            # it to the parametric kernel, where the panel was confidently wrong (n=127: sports/tech).
            kind = self.router.binary_kind(question)
            if kind in ("contest", "announcement") and self.route_contests:
                return self._parametric_binary(question, dossier, kind, as_of)
            if self.event_engine == "panel":
                return self._panel(question, dossier, today, domain=kind)
            cast = CastingDirector(self.llm).cast(question, dossier.brief(), today=today)
            cast.answer_space = {"type": "binary", "options": ["yes", "no"]}
            return self._society(question, cast, dossier, today, class_key="society:event")

        # DIFFUSION class: "how far does this spread / will it go viral" — the class where interaction IS
        # the outcome. Needs the actual artifact (message=) or the question's own content description.
        if any(w in question.lower() for w in _DIFFUSION_WORDS):
            return self._diffusion(question, message or question, dossier, today)

        cast = CastingDirector(self.llm).cast(question, dossier.brief(), today=today)
        if cast.process == "individual_reaction":
            person = recipient or (cast.actors[0].name if cast.actors else "")
            return self._individual(question, person, message or question, channel, today,
                                    dossier=dossier)
        if cast.process == "artifact_optimization" or (message is None and any(
                w in question.lower() for w in _ARTIFACT_WORDS)):
            return self._artifacts(question, cast, dossier, today)
        return self._society(question, cast, dossier, today)

    # ---------------- modes ----------------
    def _society(self, question, cast, dossier, today, class_key=None):
        res = SocietyRollout(self.llm_hot, llm=self.llm, branches=self.branches,
                             max_rounds=self.max_rounds).run(question, cast, dossier, today=today)
        cal = self.registry.calibration_for(class_key or f"society:{cast.process}")
        dist = shrink_distribution(res.distribution, cal.get("shrink", 1.0))
        f = Forecast(question=question, mechanism=f"grounded_agents:{cast.process}",
                     answer_space=cast.answer_space,
                     distribution={o: round(p, 4) for o, p in dist.items()},
                     interval=res.interval, calibration=cal, grounding=dossier.as_report(),
                     audit=res.audit, rounds=res.rounds, n_personas=res.n_personas,
                     n_llm_calls=res.n_calls, detail={"cast": cast.as_dict(),
                                                      "branch_distributions": res.branch_distributions})
        if not f.distribution:
            f.abstain, f.abstain_reason = True, "no persona decisions parsed — simulation produced nothing"
        f.headline = build_headline(f)
        return f.as_dict()

    def _panel(self, question, dossier, today, domain="deliberation"):
        """Binary event → a base-rate-anchored, multi-family observer-forecaster panel."""
        from swm.engine.calibrate import apply_temperature
        from swm.engine.observer_panel import ObserverPanel
        pf = ObserverPanel(self.llm_hot, reps_per_lens=self.panel_reps,
                           model_llms=self.model_llms).forecast(question, dossier, today=today)
        cal = self.registry.calibration_for("society:event")
        f = Forecast(question=question, mechanism="grounded_agents:observer_panel",
                     answer_space={"type": "binary", "options": ["yes", "no"]},
                     calibration=cal, grounding=dossier.as_report(),
                     audit=pf.audit, n_personas=pf.n_forecasters, n_llm_calls=pf.n_calls,
                     detail={"standing": dossier.standing, "spread": pf.spread, "trust": pf.trust,
                             "families": pf.families, "domain": domain})
        if pf.p_event is None:
            f.abstain, f.abstain_reason = True, "observer panel produced no parseable forecast"
        else:
            from swm.engine.calibrate import clamp_p
            T = self.registry.temperature_for("society:event", domain=domain, default=cal.get("temperature", 1.0))
            p = clamp_p(apply_temperature(pf.p_event, T)) if T and T != 1.0 else pf.p_event
            f.distribution = {"yes": round(p, 4), "no": round(1 - p, 4)}
        f.headline = build_headline(f)
        return f.as_dict()

    def _parametric_binary(self, question, dossier, kind, as_of):
        """Lever 2: a contest/announcement → main's parametric kernel (LEAK-FREE: base-rate + mechanism
        structure, state grounding OFF). Returns a native binary Forecast — an honest, non-overconfident
        P(event) rather than the deliberation panel's confident-wrong guess (n=127: sports/tech)."""
        p = parametric_binary_p(question, as_of, self.llm)
        if p is None:                                     # honest fallback: the reference-class base rate
            s = dossier.standing_struct or {}
            fav = str(s.get("favored", "")).lower()
            try:
                conf = float(s.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            p = 0.5 + 0.4 * conf if fav in ("yes", "true") else (0.5 - 0.4 * conf if fav in ("no", "false")
                                                                 else 0.5)
        p = min(0.97, max(0.03, float(p)))
        f = Forecast(question=question, mechanism=f"parametric:{kind}",
                     answer_space={"type": "binary", "options": ["yes", "no"]},
                     distribution={"yes": round(p, 4), "no": round(1 - p, 4)},
                     grounding=dossier.as_report(),
                     calibration=self.registry.calibration_for("society:event"),
                     detail={"routed": f"{kind}→parametric kernel (not social deliberation)"})
        f.headline = build_headline(f)
        return f.as_dict()

    def _diffusion(self, question, artifact, dossier, today):
        """Spread/virality → grounded-archetype cascade simulation (native: reach distribution, narrative
        leaders, inflection). Ships flagged until the class earns a grade on real cascade data."""
        from swm.engine.diffusion import DiffusionSimulator
        df = DiffusionSimulator(self.llm_hot, llm=self.llm).simulate(artifact, dossier)
        cal = self.registry.calibration_for("diffusion:reach")
        f = Forecast(question=question, mechanism="grounded_agents:diffusion",
                     answer_space={"type": "reach_distribution",
                                   "options": list(df.p_over.keys())},
                     distribution={f"reach>{t}": p for t, p in df.p_over.items()},
                     calibration=cal, grounding=dossier.as_report(), audit=df.archetypes,
                     n_personas=len(df.archetypes), n_llm_calls=df.n_calls,
                     detail={"reach": df.reach, "narrative_leaders": df.narrative_leaders,
                             "inflection_round": df.inflection_round, "sentiment": df.sentiment,
                             "n_worlds": df.n_worlds,
                             "assumption": "heavy-tailed follower graph; propensities are SAMPLED reasoned "
                                           "decisions on the actual content, early vs late exposure"})
        if not df.reach:
            f.abstain, f.abstain_reason = True, "audience identification failed — no cascade simulated"
            f.headline = build_headline(f)
        else:
            lead = df.narrative_leaders[0]["archetype"] if df.narrative_leaders else "?"
            f.headline = (f"median reach {df.reach['p50']:.0%} of the plausible audience "
                          f"(p10 {df.reach['p10']:.0%} – p90 {df.reach['p90']:.0%}); "
                          f"P(>20% reach) = {df.p_over.get('0.2', 0):.0%}; narrative leader: {lead}; "
                          f"inflection ~round {df.inflection_round}")
            if cal.get("abstain_confident"):
                f.headline += f"  [{cal.get('grade', 'ungraded')} — hypothesis, not a calibrated forecast]"
        return f.as_dict()

    def _individual(self, question, person, message, channel, today, dossier=None):
        if not person:
            f = Forecast(question=question, mechanism="abstained", abstain=True,
                         abstain_reason="individual_reaction with no identifiable person — "
                                        "pass recipient=<name>")
            f.headline = build_headline(f)
            return f.as_dict()
        res = IndividualSimulator(self.llm_hot, llm=self.llm).simulate(person, message, channel=channel)
        cal = self.registry.calibration_for("individual:response")
        f = Forecast(question=question, mechanism="grounded_agents:individual_reaction",
                     answer_space={"type": "binary", "options": ["responds", "does_not_respond"]},
                     calibration=cal, abstain=res.abstain, abstain_reason=res.abstain_reason,
                     grounding={**res.grounding, "detail": [], "abstain": res.abstain},
                     n_personas=len(res.per_state), n_llm_calls=res.n_runs,
                     detail={"per_state": res.per_state, "reasons": res.reasons,
                             "person": person})
        if not res.abstain:
            p = shrink_distribution({"responds": res.p_response,
                                     "does_not_respond": 1 - res.p_response},
                                    cal.get("shrink", 1.0))["responds"]
            f.distribution = {"responds": round(p, 4), "does_not_respond": round(1 - p, 4)}
            f.interval = {"responds": res.interval_80}
            f.headline = (f"{person} responds to THIS message: {p:.0%} "
                          f"(80% sampling interval {res.interval_80[0]:.0%}–{res.interval_80[1]:.0%}, "
                          f"{res.n_runs} grounded runs)")
            if cal.get("abstain_confident"):
                f.headline += f"  [{cal.get('grade')} — hypothesis, not a calibrated forecast]"
        else:
            f.headline = build_headline(f)
        return f.as_dict()

    def _artifacts(self, question, cast, dossier, today):
        res = ArtifactOptimizer(self.llm_hot, llm=self.llm).run(
            question, cast, dossier, k=self.k_artifacts, today=today)
        cal = self.registry.calibration_for("artifact:engagement")
        f = Forecast(question=question, mechanism="grounded_agents:artifact_optimization",
                     answer_space={"type": "artifacts",
                                   "options": [a["text"] for a in res.get("ranked", [])]},
                     ranked_artifacts=res.get("ranked", []), calibration=cal,
                     grounding=dossier.as_report(),
                     n_personas=res.get("n_personas", 0), n_llm_calls=res.get("n_calls", 0),
                     detail={"cast": cast.as_dict(), "note": res.get("note", "")})
        if not f.ranked_artifacts:
            f.abstain, f.abstain_reason = True, res.get("error", "no artifacts survived evaluation")
        f.headline = build_headline(f)
        return f.as_dict()


def hybrid_world_model(*, branches=3, max_rounds=2, k_artifacts=5, today="", parametric=True):
    """The recommended front door: the reasoning-agent engine for anything driven by PEOPLE (the default and
    the vast majority of social questions), with main's parametric mechanism engine wired in ONLY as the
    fallback for genuinely non-human stochastic processes (a price crossing a level, a launch, a record).
    The router is biased hard toward agents — we never regress to a parametric readout for a people question.
    `parametric=False` returns the pure agent engine (no fallback)."""
    wm = agent_world_model(branches=branches, max_rounds=max_rounds, k_artifacts=k_artifacts, today=today)
    if parametric:
        try:
            from swm.api.world_model import general_world_model
            wm.parametric = general_world_model()          # main's grounded parametric kernels
        except Exception:
            wm.parametric = None                           # no fallback available → pure agents
    return wm


def parametric_binary_p(question, as_of, llm):
    """Leak-free P(YES) from main's parametric kernel: compile the question as-of, force state grounding OFF
    (base-rate + mechanism structure only, no live data to leak), run the Monte-Carlo. None on failure."""
    try:
        from swm.api.backtest_harness import _apply_toggles, _p_from_forecast
        from swm.api.compiler import CompiledModel, build_compile_prompt
        from swm.api.model_spec import parse_spec
        ctx = (f"TODAY'S DATE IS {as_of or ''}. Use ONLY information available on or before this date; do not "
               f"use knowledge of anything after it. Define outcome.event so P(event) = P(the question "
               f"resolves YES).")
        spec = parse_spec(llm(build_compile_prompt(question, ctx)))
        spec = _apply_toggles(spec, ground=False)          # neutralize state estimates → base-rate + mechanism
        return _p_from_forecast(CompiledModel(spec).run(n=3000))
    except Exception:
        return None


def multi_family_backends(*, temperature=0.5, max_tokens=500):
    """Lever 3: {family: callable} across genuinely different pretraining (DeepSeek/Qwen/Llama/Mixtral/Gemma)
    via main's inner_crowd — the panel backends. Unreachable families (no HF credit) drop out rather than
    collapse onto DeepSeek, so errors stay decorrelated; degrades to whatever is actually available."""
    try:
        from swm.api.inner_crowd import model_panel_llms
        backends = model_panel_llms(system="You inhabit one specific forecaster. Reason as them.",
                                    temperature=temperature, max_tokens=max_tokens)
        return backends or None
    except Exception:
        return None


def agent_world_model(*, branches=3, max_rounds=2, k_artifacts=5, today="",
                      multi_family=False, log_forecasts=False) -> AgentWorldModel:
    """The recommended front door. DeepSeek backends: cold (t=0.2) for grounding/casting, hot (t=0.9)
    for persona decisions. `multi_family=True` wires the cross-family observer panel (Lever 3). Raises if no
    LLM key is configured: this engine does not degrade to heuristics, it refuses (constitution rule 2)."""
    from swm.api.deepseek_backend import default_chat_fn
    cold = default_chat_fn(system="You are a precise assistant inside a forecasting engine. "
                                  "Reply with ONLY compact JSON.", max_tokens=1600, temperature=0.2)
    hot = default_chat_fn(system="You inhabit one specific person inside a grounded social simulation. "
                                 "Reason as them, not as an analyst. Reply with ONLY compact JSON.",
                          max_tokens=500, temperature=0.9)
    if cold is None:
        raise RuntimeError("no LLM backend configured (DEEPSEEK_API_KEY / HF_TOKEN) — the agent engine "
                           "refuses to run ungrounded heuristics in place of reasoning")
    model_llms = multi_family_backends() if multi_family else None
    fw = None
    if log_forecasts:                                      # the outcome flywheel: log → resolve → refit
        from swm.engine.flywheel import FlywheelLog
        fw = FlywheelLog()
    return AgentWorldModel(llm=cold, llm_hot=hot, branches=branches, max_rounds=max_rounds,
                           k_artifacts=k_artifacts, today=today, model_llms=model_llms, flywheel=fw)
